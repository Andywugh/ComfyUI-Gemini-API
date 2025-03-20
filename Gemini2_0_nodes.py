import os
import base64
import io
import json
import torch
import numpy as np
from PIL import Image
import requests
import tempfile
from io import BytesIO
from google import genai
from google.genai import types
import time
import traceback
import pathlib

class GeminiImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["models/gemini-2.0-flash-exp"], {"default": "models/gemini-2.0-flash-exp"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "temperature": ("FLOAT", {"default": 1, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999}),
                "image": ("IMAGE",),
                "keep_temp_files": ("BOOLEAN", {"default": False}),
                "max_retries": ("INT", {"default": 5, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "API Respond")
    FUNCTION = "generate_image"
    CATEGORY = "Google-Gemini"
    
    def __init__(self):
        """初始化日志系统和API密钥存储"""
        self.log_messages = []  # 全局日志消息存储
        # 获取节点所在目录
        self.node_dir = os.path.dirname(os.path.abspath(__file__))
        self.key_file = os.path.join(self.node_dir, "gemini_api_key.txt")
        
        # 创建图像保存目录
        self.images_dir = os.path.join(self.node_dir, "generated_images")
        if not os.path.exists(self.images_dir):
            try:
                os.makedirs(self.images_dir)
                self.log(f"创建图像保存目录: {self.images_dir}")
            except Exception as e:
                self.log(f"创建图像目录失败: {e}")
                self.images_dir = self.node_dir  # 如果创建失败，使用节点目录
        
        # 创建诊断日志目录
        self.log_dir = os.path.join(self.node_dir, "logs")
        if not os.path.exists(self.log_dir):
            try:
                os.makedirs(self.log_dir)
                self.log(f"创建诊断日志目录: {self.log_dir}")
            except Exception as e:
                self.log(f"创建日志目录失败: {e}")
                self.log_dir = self.node_dir  # 如果创建失败，使用节点目录
        
        # 检查google-genai版本
        try:
            import importlib.metadata
            genai_version = importlib.metadata.version('google-genai')
            self.log(f"当前google-genai版本: {genai_version}")
            
            # 检查版本是否满足最低要求
            from packaging import version
            if version.parse(genai_version) < version.parse('1.5.0'):  
                self.log("警告: google-genai版本过低，建议升级到最新版本")
                self.log("建议执行: pip install -q -U google-genai")
        except Exception as e:
            self.log(f"无法检查google-genai版本: {e}")
        
        self.temp_files = []  # 添加临时文件跟踪列表
        
        # 重试设置
        self.max_retries = 5  # 最大重试次数
        self.initial_retry_delay = 2  # 初始重试延迟（秒）
    
    def log(self, message):
        """全局日志函数：记录到日志列表"""
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        return message
    
    def get_api_key(self, user_input_key):
        """获取API密钥，优先使用用户输入的密钥"""
        # 如果用户输入了有效的密钥，使用并保存
        if user_input_key and len(user_input_key) > 10:
            self.log("使用用户输入的API密钥")
            # 保存到文件中
            try:
                with open(self.key_file, "w") as f:
                    f.write(user_input_key)
                self.log("已保存API密钥到节点目录")
            except Exception as e:
                self.log(f"保存API密钥失败: {e}")
            return user_input_key
            
        # 如果用户没有输入，尝试从文件读取
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, "r") as f:
                    saved_key = f.read().strip()
                if saved_key and len(saved_key) > 10:
                    self.log("使用已保存的API密钥")
                    return saved_key
            except Exception as e:
                self.log(f"读取保存的API密钥失败: {e}")
                
        # 如果都没有，返回空字符串
        self.log("警告: 未提供有效的API密钥")
        return ""
    
    def generate_empty_image(self, width, height):
        """生成标准格式的空白RGB图像张量 - 确保ComfyUI兼容格式 [B,H,W,C]"""
        # 创建一个符合ComfyUI标准的图像张量
        # ComfyUI期望 [batch, height, width, channels] 格式!
        empty_image = np.ones((height, width, 3), dtype=np.float32) * 0.2
        tensor = torch.from_numpy(empty_image).unsqueeze(0) # [1, H, W, 3]
        
        self.log(f"创建ComfyUI兼容的空白图像: 形状={tensor.shape}, 类型={tensor.dtype}")
        return tensor
    
    def validate_and_fix_tensor(self, tensor, name="图像"):
        """验证并修复张量格式，确保完全兼容ComfyUI"""
        try:
            # 基本形状检查
            if tensor is None:
                self.log(f"警告: {name} 是None")
                return None
                
            self.log(f"验证 {name}: 形状={tensor.shape}, 类型={tensor.dtype}, 设备={tensor.device}")
            
            # 确保形状正确: [B, C, H, W]
            if len(tensor.shape) != 4:
                self.log(f"错误: {name} 形状不正确: {tensor.shape}")
                return None
                
            if tensor.shape[1] != 3:
                self.log(f"错误: {name} 通道数不是3: {tensor.shape[1]}")
                return None
                
            # 确保类型为float32
            if tensor.dtype != torch.float32:
                self.log(f"修正 {name} 类型: {tensor.dtype} -> torch.float32")
                tensor = tensor.to(dtype=torch.float32)
                
            # 确保内存连续
            if not tensor.is_contiguous():
                self.log(f"修正 {name} 内存布局: 使其连续")
                tensor = tensor.contiguous()
                
            # 确保值范围在0-1之间
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            
            if min_val < 0 or max_val > 1:
                self.log(f"修正 {name} 值范围: [{min_val}, {max_val}] -> [0, 1]")
                tensor = torch.clamp(tensor, 0.0, 1.0)
                
            return tensor
        except Exception as e:
            self.log(f"验证张量时出错: {e}")
            traceback.print_exc()
            return None
    
    def save_tensor_as_image(self, image_tensor, file_path):
        """将图像张量保存为文件"""
        try:
            # 转换为numpy数组
            if torch.is_tensor(image_tensor):
                if len(image_tensor.shape) == 4:
                    image_tensor = image_tensor[0]  # 获取批次中的第一张图像
                
                # [C, H, W] -> [H, W, C]
                image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image_tensor
            
            # 缩放到0-255
            image_np = (image_np * 255).astype(np.uint8)
            
            # 创建PIL图像
            pil_image = Image.fromarray(image_np)
            
            # 保存到文件
            pil_image.save(file_path, format="PNG")
            self.log(f"已保存图像到: {file_path}")
            return True
        except Exception as e:
            self.log(f"图像保存错误: {str(e)}")
            return False
    
    def process_image_data(self, image_data, width, height):
        """处理API返回的图像数据，返回ComfyUI格式的图像张量 [B,H,W,C]"""
        try:
            # 打印图像数据类型和大小以便调试
            self.log(f"图像数据类型: {type(image_data)}")
            self.log(f"图像数据长度: {len(image_data) if hasattr(image_data, '__len__') else '未知'}")
            
            # 尝试直接转换为PIL图像
            try:
                pil_image = Image.open(BytesIO(image_data))
                self.log(f"成功打开图像, 尺寸: {pil_image.width}x{pil_image.height}, 模式: {pil_image.mode}")
            except Exception as e:
                self.log(f"无法直接打开图像数据: {e}")
                
                # 尝试其他方式解析，例如base64解码
                try:
                    # 检查是否是base64编码的字符串
                    if isinstance(image_data, str):
                        # 尝试移除base64前缀
                        if "base64," in image_data:
                            image_data = image_data.split("base64,")[1]
                        decoded_data = base64.b64decode(image_data)
                        pil_image = Image.open(BytesIO(decoded_data))
                    else:
                        # 如果是向量或其他格式，生成一个占位图像
                        self.log("无法解析图像数据，创建一个空白图像")
                        return self.generate_empty_image(width, height)
                except Exception as e2:
                    self.log(f"备用解析方法也失败: {e2}")
                    return self.generate_empty_image(width, height)
            
            # 确保图像是RGB模式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                self.log(f"图像已转换为RGB模式")
            
            # 调整图像大小
            if pil_image.width != width or pil_image.height != height:
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
                self.log(f"图像已调整为目标尺寸: {width}x{height}")
            
            # 关键修复: 使用ComfyUI兼容的格式 [batch, height, width, channels]
            # 而不是PyTorch标准的 [batch, channels, height, width]
            img_array = np.array(pil_image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            self.log(f"生成的图像张量格式: 形状={img_tensor.shape}, 类型={img_tensor.dtype}")
            return (img_tensor,)
            
        except Exception as e:
            self.log(f"处理图像数据时出错: {e}")
            traceback.print_exc()
            return self.generate_empty_image(width, height)
    
    def cleanup_temp_files(self, keep_files=False):
        """清理临时文件"""
        if keep_files:
            self.log(f"保留临时文件，共 {len(self.temp_files)} 个")
            return
            
        cleaned = 0
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    cleaned += 1
            except Exception as e:
                self.log(f"清理文件失败: {file_path}, 错误: {e}")
        
        self.log(f"清理了 {cleaned}/{len(self.temp_files)} 个临时文件")
        self.temp_files = []  # 清空列表
    
    def create_diagnostic_report(self, error_msg, api_response=None, prompt=None, model=None):
        """创建诊断报告，帮助用户解决问题"""
        try:
            # 创建诊断时间戳
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(self.log_dir, f"gemini_diagnostic_{timestamp}.txt")
            
            # 收集系统信息
            import platform
            import sys
            
            # 写入报告
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("=== Gemini API 诊断报告 ===\n\n")
                f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Python版本: {sys.version}\n")
                f.write(f"操作系统: {platform.system()} {platform.version()}\n")
                
                # 添加google-genai库版本
                try:
                    import importlib.metadata
                    genai_version = importlib.metadata.version('google-genai')
                    f.write(f"google-genai版本: {genai_version}\n")
                except:
                    f.write("无法获取google-genai版本\n")
                
                # 添加请求信息
                f.write("\n=== 请求信息 ===\n")
                if prompt:
                    f.write(f"提示词: {prompt}\n")
                if model:
                    f.write(f"模型: {model}\n")
                
                # 添加错误信息
                f.write("\n=== 错误信息 ===\n")
                f.write(f"{error_msg}\n")
                
                # 添加API响应
                if api_response:
                    f.write("\n=== API响应 ===\n")
                    # 尝试获取响应的各种属性
                    if hasattr(api_response, '__dict__'):
                        f.write(f"响应对象类型: {type(api_response)}\n")
                        for attr in dir(api_response):
                            if not attr.startswith('_'):
                                try:
                                    value = getattr(api_response, attr)
                                    if not callable(value):
                                        f.write(f"{attr}: {value}\n")
                                except:
                                    pass
                    else:
                        f.write(f"{api_response}\n")
                
                # 添加日志消息
                f.write("\n=== 处理日志 ===\n")
                for msg in self.log_messages:
                    f.write(f"{msg}\n")
            
            self.log(f"已创建诊断报告: {report_path}")
            return report_path
        except Exception as e:
            self.log(f"创建诊断报告失败: {e}")
            return None
    
    def retry_api_call(self, client, model, contents, config, prompt, max_retries=5):
        """
        使用指数退避重试API调用
        返回: (response, error_message)，如果成功则error_message为None
        """
        # 使用提供的max_retries或者实例默认值
        retry_count = max_retries if max_retries > 0 else self.max_retries
        
        for attempt in range(1, retry_count + 1):
            try:
                self.log(f"API调用尝试 #{attempt}/{retry_count}")
                
                # 调用API
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )
                
                # 记录API响应基本信息
                if hasattr(response, 'candidates') and response.candidates:
                    self.log(f"API响应包含 {len(response.candidates)} 个候选项")
                    
                    # 验证响应内容是否有效
                    if (hasattr(response.candidates[0], 'content') and 
                        response.candidates[0].content is not None and
                        hasattr(response.candidates[0].content, 'parts') and
                        response.candidates[0].content.parts is not None):
                        self.log(f"API调用成功（尝试 #{attempt}）")
                        return response, None
                    else:
                        error = "API响应中content结构无效或parts为空"
                        self.log(f"无效响应（尝试 #{attempt}）: {error}")
                else:
                    error = "API响应中没有候选项"
                    self.log(f"无效响应（尝试 #{attempt}）: {error}")
                
                # 如果响应无效但没有抛出异常，记录详细信息
                self.log(f"响应对象类型: {type(response)}")
                response_attrs = [attr for attr in dir(response) if not attr.startswith('_')]
                self.log(f"响应对象属性: {', '.join(response_attrs)}")
                
                # 检查finish_reason
                if (hasattr(response, 'candidates') and response.candidates and
                    hasattr(response.candidates[0], 'finish_reason')):
                    self.log(f"完成原因: {response.candidates[0].finish_reason}")
                
                # 如果已是最后一次尝试，返回最后一次的响应
                if attempt == retry_count:
                    self.log("达到最大重试次数，返回最后一次响应")
                    return response, "多次尝试后API未返回有效内容"
                
            except Exception as api_error:
                error_message = str(api_error)
                self.log(f"API调用错误（尝试 #{attempt}）: {error_message}")
                
                # 对于一些特定错误，不再重试
                if "400 Bad Request" in error_message and "Invalid value" in error_message:
                    self.log("参数错误，不再重试")
                    return None, error_message
                
                if "401 Unauthorized" in error_message:
                    self.log("认证失败，不再重试")
                    return None, error_message
                
                # 如果已是最后一次尝试，返回错误
                if attempt == retry_count:
                    self.log("达到最大重试次数，返回错误")
                    return None, error_message
            
            # 计算下一次重试的延迟（指数退避）
            retry_delay = self.initial_retry_delay * (2 ** (attempt - 1))
            self.log(f"等待 {retry_delay} 秒后重试...")
            time.sleep(retry_delay)
        
        # 不应该到达这里，但以防万一
        return None, "重试失败，未获取有效响应"
    
    def generate_image(self, prompt, api_key, model, width, height, temperature, seed=0, image=None, keep_temp_files=False, max_retries=5):
        """生成图像 - 使用简化的API密钥管理"""
        temp_img_path = None
        response_text = ""
        
        # 重置日志消息和临时文件列表
        self.log_messages = []
        self.temp_files = []
        
        try:
            # 获取API密钥
            actual_api_key = self.get_api_key(api_key)
            
            if not actual_api_key:
                error_message = "错误: 未提供有效的API密钥。请在节点中输入API密钥或确保已保存密钥。"
                self.log(error_message)
                full_text = "## 错误\n" + error_message + "\n\n## 使用说明\n1. 在节点中输入您的Google API密钥\n2. 密钥将自动保存到节点目录，下次可以不必输入"
                self.cleanup_temp_files(keep_temp_files)
                return (self.generate_empty_image(width, height), full_text)
            
            # 创建客户端实例
            client = genai.Client(api_key=actual_api_key)
            
            # 处理种子值
            if seed == 0:
                import random
                seed = random.randint(1, 2**31 - 1)
                self.log(f"生成随机种子值: {seed}")
            else:
                # 确保种子值在INT32范围内
                max_int32 = 2**31 - 1  # 2,147,483,647
                if seed > max_int32:
                    original_seed = seed
                    seed = seed % max_int32
                    self.log(f"种子值 {original_seed} 超出INT32范围，已调整为: {seed}")
                self.log(f"使用指定的种子值: {seed}")
            
            # 构建包含尺寸信息的提示
            aspect_ratio = width / height
            if aspect_ratio > 1:
                orientation = "landscape (horizontal)"
            elif aspect_ratio < 1:
                orientation = "portrait (vertical)"
            else:
                orientation = "square"

            simple_prompt = f"Create a detailed image of: {prompt}. Generate the image in {orientation} orientation with exact dimensions of {width}x{height} pixels. Ensure the composition fits properly within these dimensions without stretching or distortion."
            
            # 配置生成参数，使用用户指定的温度值
            gen_config = types.GenerateContentConfig(
                temperature=temperature,
                seed=seed,
                response_modalities=['Text', 'Image']
            )
            
            # 记录温度设置
            self.log(f"使用温度值: {temperature}，种子值: {seed}")
            
            # 处理参考图像
            contents = {}
            has_reference = False
            
            if image is not None:
                try:
                    # 确保图像格式正确
                    if len(image.shape) == 4 and image.shape[0] == 1:  # [1, H, W, 3] 格式
                        # 获取第一帧图像
                        input_image = image[0].cpu().numpy()
                        
                        # 转换为PIL图像
                        input_image = (input_image * 255).astype(np.uint8)
                        pil_image = Image.fromarray(input_image)
                        
                        # 保存为临时文件
                        temp_img_path = os.path.join(self.images_dir, f"reference_{int(time.time())}.png")
                        pil_image.save(temp_img_path)
                        self.temp_files.append(temp_img_path)  # 添加到临时文件列表
                        
                        self.log(f"参考图像处理成功，尺寸: {pil_image.width}x{pil_image.height}")
                        
                        # 读取图像
                        with open(temp_img_path, "rb") as f:
                            image_bytes = f.read()
                        
                        # 使用最新的Gemini API格式 - 简化多模态内容创建
                        self.log("使用最新的Gemini API格式添加参考图像")
                        # 更新提示，明确指示使用参考图像
                        enhanced_prompt = simple_prompt + " Use this reference image as style guidance."
                        
                        try:
                            # 使用正确的Gemini API多模态格式
                            # from google.generativeai.types import content_types
                            # from google.generativeai.types import Part
                            # from io import BytesIO
                            
                            # 创建Part对象
                            img_part = {"inline_data": {"mime_type": "image/png", "data": image_bytes}}
                            txt_part = {"text": simple_prompt + " Use this reference image as style guidance."}
                            # image_part = Part.from_data(mime_type="image/png", data=image_bytes)
                            # text_part = Part.from_text(enhanced_prompt)
                            
                            # 使用content_types.to_contents函数来创建正确的内容格式
                            contents = [img_part, txt_part]
                            
                            has_reference = True
                            self.log("已成功添加参考图像到请求中")
                        except Exception as e:
                            self.log(f"创建多模态内容失败: {e}")
                            contents = simple_prompt
                            has_reference = False
                    else:
                        self.log(f"参考图像格式不正确: {image.shape}")
                        contents = simple_prompt
                except Exception as img_error:
                    self.log(f"参考图像处理错误: {str(img_error)}")
                    contents = simple_prompt
            else:
                # 没有参考图像，只使用文本
                contents = simple_prompt
            
            # 打印请求信息
            self.log(f"请求Gemini API生成图像，种子值: {seed}, 包含参考图像: {has_reference}")
            
            # 详细记录请求内容
            if isinstance(contents, str):
                self.log(f"请求内容类型: 纯文本, 长度: {len(contents)}")
            elif isinstance(contents, list):
                self.log(f"请求内容类型: 多模态列表, 元素数量: {len(contents)}")
                for i, content_part in enumerate(contents):
                    if isinstance(content_part, dict) and 'inline_data' in content_part:
                        mime = content_part.get('inline_data', {}).get('mime_type', 'unknown')
                        self.log(f"  内容项[{i}]: 图像数据, MIME类型: {mime}")
                    elif isinstance(content_part, dict) and 'text' in content_part:
                        text = content_part.get('text', '')[:50]
                        self.log(f"  内容项[{i}]: 文本数据, 前50字符: {text}...")
                    else:
                        self.log(f"  内容项[{i}]: 未知类型: {type(content_part)}")
            
            try:
                # 调用API（使用重试机制）
                response, error = self.retry_api_call(
                    client=client, 
                    model=model, 
                    contents=contents, 
                    config=gen_config, 
                    prompt=prompt,
                    max_retries=max_retries
                )
                
                # 如果返回了错误而不是响应
                if error and not response:
                    # 处理错误
                    error_detail = error
                    self.log(f"API调用最终失败: {error_detail}")
                    
                    # 检查是否包含特定错误信息并提供更友好的提示
                    if "400 Bad Request" in error_detail:
                        self.log("可能是请求参数不正确，请检查API版本和参数")
                        error_message = "请求参数错误 (400)：可能是模型名称不正确或API参数格式有误。请确保您使用了正确的模型ID和参数格式。"
                    elif "401 Unauthorized" in error_detail:
                        self.log("API密钥认证失败，请检查密钥是否正确")
                        error_message = "API密钥认证失败 (401)：请检查您的API密钥是否正确，或者是否有使用Gemini 2.0的权限。"
                    elif "429 Too Many Requests" in error_detail:
                        self.log("API请求频率超限，请稍后再试")
                        error_message = "请求频率超限 (429)：您的API请求过于频繁，已超出配额限制。请稍后再试或升级您的API配额。"
                    elif "500 Internal Server Error" in error_detail:
                        self.log("Google API服务器内部错误，请稍后再试")
                        error_message = "服务器内部错误 (500)：Google Gemini API服务器出现问题。请稍后再试。"
                    elif "Model not found" in error_detail or "model not found" in error_detail.lower():
                        self.log("指定的模型不存在或无权访问")
                        error_message = f"模型不存在：'{model}' 模型不存在或您没有访问权限。请确认模型名称是否正确，以及您的账户是否有权访问此模型。"
                    elif "NoneType" in error_detail and "has no attribute" in error_detail:
                        self.log("API返回了空响应或结构不符合预期")
                        error_message = "API响应结构错误：服务器返回了空或不完整的数据结构。这可能是由于：\n1. 提示内容触发了内容政策\n2. 模型暂时不可用\n3. 多模态内容格式不正确"
                    elif "Invalid value at 'generation_config.seed'" in error_detail:
                        self.log("种子值超出有效范围")
                        error_message = "种子值错误：种子值超出了有效范围（必须在 -2,147,483,648 到 2,147,483,647 之间）。\n已自动修复此问题，请重新运行节点。"
                        # 添加自动修复说明
                        max_int32 = 2**31 - 1
                        if seed > max_int32:
                            adjusted_seed = seed % max_int32
                            error_message += f"\n\n原始种子值: {seed}\n调整后的种子值: {adjusted_seed}\n\n请使用较小的种子值或让系统自动生成。"
                    else:
                        error_message = f"API错误：{error_detail}\n\n可能的解决方案：\n1. 检查API密钥是否正确\n2. 检查网络连接\n3. 确认提示内容不违反内容政策\n4. 稍后再试"
                    
                    # 检查是否包含重试信息，添加到错误消息
                    if "多次尝试后" in error_detail:
                        error_message += f"\n\n系统已尝试 {max_retries} 次请求，但仍未获得有效响应。您可以尝试增加重试次数或稍后再试。"
                    
                    # 创建诊断报告
                    report_path = self.create_diagnostic_report(
                        error_msg=error_detail,
                        api_response=response,  # 可能为None
                        prompt=prompt,
                        model=model
                    )
                    
                    if report_path:
                        error_message += f"\n\n诊断报告已保存到：{report_path}"
                    
                    # 合并日志和错误信息
                    full_text = "## 处理日志\n" + "\n".join(self.log_messages) + f"\n\n## API错误\n{error_message}"
                    self.cleanup_temp_files(keep_temp_files)
                    return (self.generate_empty_image(width, height), full_text)
                
                # 多次尝试后有响应，但有警告
                if error and response:
                    self.log(f"API响应成功但有警告: {error}")
                
                self.log("API响应接收成功，正在处理...")
                
                # 检查响应中是否有图像
                image_found = False
                
                # 遍历响应部分
                for part in response.candidates[0].content.parts:
                    # 检查是否为文本部分
                    if hasattr(part, 'text') and part.text is not None:
                        text_content = part.text
                        response_text += text_content
                        self.log(f"API返回文本: {text_content[:100]}..." if len(text_content) > 100 else text_content)
                    
                    # 检查是否为图像部分
                    elif hasattr(part, 'inline_data') and part.inline_data is not None:
                        self.log("API返回数据解析处理")
                        try:
                            # 获取图像数据
                            image_data = part.inline_data.data
                            mime_type = part.inline_data.mime_type if hasattr(part.inline_data, 'mime_type') else "未知"
                            self.log(f"图像数据类型: {type(image_data)}, MIME类型: {mime_type}, 数据长度: {len(image_data) if image_data else 0}")
                            
                            # 跳过空数据
                            if not image_data or len(image_data) < 100:
                                self.log("警告: 图像数据为空或太小")
                                continue
                            
                            # 直接保存为文件 - 跳过BytesIO
                            timestamp = int(time.time())
                            filename = f"gemini_image_{timestamp}.raw"
                            img_file = os.path.join(self.images_dir, filename)
                            
                            with open(img_file, "wb") as f:
                                f.write(image_data)
                            self.log(f"已保存原始图像数据到: {img_file}")
                            self.temp_files.append(img_file)  # 添加到临时文件列表
                            
                            # 创建默认空白图像
                            pil_image = Image.new('RGB', (width, height), color=(128, 128, 128))
                            
                            # 尝试解析文件
                            success = False
                            
                            # 尝试直接打开原始文件
                            try:
                                saved_image = Image.open(img_file)
                                self.log(f"成功打开原始图像，格式: {saved_image.format}, 尺寸: {saved_image.width}x{saved_image.height}")
                                success = True
                                
                                # 确保是RGB模式
                                if saved_image.mode != 'RGB':
                                    saved_image = saved_image.convert('RGB')
                                
                                # 调整尺寸
                                if saved_image.width != width or saved_image.height != height:
                                    saved_image = saved_image.resize((width, height), Image.Resampling.LANCZOS)
                                
                                pil_image = saved_image
                                
                            except Exception as e1:
                                self.log(f"无法直接打开原始文件: {str(e1)}")
                                
                                # 尝试转换为PNG后打开
                                png_file = os.path.join(self.images_dir, f"gemini_image_{timestamp}.png")
                                try:
                                    with open(png_file, "wb") as f:
                                        f.write(image_data)
                                    self.log(f"已保存数据为PNG: {png_file}")
                                    self.temp_files.append(png_file)  # 添加到临时文件列表
                                    
                                    saved_image = Image.open(png_file)
                                    self.log(f"成功通过PNG打开图像，尺寸: {saved_image.width}x{saved_image.height}")
                                    success = True
                                    
                                    # 确保是RGB模式并调整尺寸
                                    if saved_image.mode != 'RGB':
                                        saved_image = saved_image.convert('RGB')
                                    
                                    if saved_image.width != width or saved_image.height != height:
                                        saved_image = saved_image.resize((width, height), Image.Resampling.LANCZOS)
                                    
                                    pil_image = saved_image
                                    
                                except Exception as e2:
                                    self.log(f"PNG格式打开也失败: {str(e2)}")
                                    self.log("使用默认空白图像")
                            
                            # 转换为ComfyUI格式
                            img_array = np.array(pil_image).astype(np.float32) / 255.0
                            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                            
                            self.log(f"图像转换为张量成功, 形状: {img_tensor.shape}")
                            image_found = True
                            
                            # 清理临时文件
                            self.cleanup_temp_files(keep_temp_files)
                            
                            # 合并日志和API返回文本
                            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## API返回\n" + response_text
                            return (img_tensor, full_text)
                        except Exception as e:
                            self.log(f"图像处理错误: {e}")
                            traceback.print_exc()  # 添加详细的错误追踪信息
                
                # 检查是否找到图像数据
                if not image_found:
                    self.log("API响应中未找到图像数据")
                    
                    # 检查是否有安全过滤信息
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                        prompt_feedback = response.prompt_feedback
                        
                        # 检查是否有安全评级
                        if hasattr(prompt_feedback, 'safety_ratings') and prompt_feedback.safety_ratings:
                            self.log("发现安全评级信息:")
                            for rating in prompt_feedback.safety_ratings:
                                if hasattr(rating, 'category') and hasattr(rating, 'probability'):
                                    self.log(f"- 类别: {rating.category}, 概率: {rating.probability}")
                    
                        # 检查是否被屏蔽
                        if hasattr(prompt_feedback, 'block_reason') and prompt_feedback.block_reason:
                            self.log(f"请求被屏蔽，原因: {prompt_feedback.block_reason}")
                            response_text = f"请求被Google安全过滤器屏蔽，原因: {prompt_feedback.block_reason}\n\n您可以尝试修改提示内容，避免使用可能触发安全过滤的词语。"
                    
                    # 检查finish_reason
                    if hasattr(response.candidates[0], 'finish_reason') and response.candidates[0].finish_reason:
                        reason = response.candidates[0].finish_reason
                        self.log(f"API响应完成原因: {reason}")
                        
                        # 添加用户友好的解释
                        if reason == "SAFETY":
                            response_text = "由于安全原因，Google Gemini无法生成此图像。请修改您的提示，避免可能违反使用条款的内容。"
                        elif reason == "RECITATION":
                            response_text = "Google Gemini无法生成此图像，因为您的提示可能要求复制受保护的内容。"
                        elif reason == "STOP":
                            response_text = "Google Gemini已完成但没有生成图像。请尝试使用更具描述性的提示。"
                    
                    # 如果没有找到具体原因，提供一般性建议
                    if not response_text:
                        response_text = "Google Gemini没有生成图像。您可以尝试:\n1. 使用更具描述性的提示\n2. 避免要求特定知名人物或品牌\n3. 调整提示以避免可能违反使用条款的内容"
                
                    # 清理临时文件
                    self.cleanup_temp_files(keep_temp_files)
                    
                    # 合并日志和API返回文本
                    full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## API返回\n" + response_text
                    return (self.generate_empty_image(width, height), full_text)
            
            except Exception as e:
                error_message = f"处理过程中出错: {str(e)}"
                self.log(f"Gemini图像生成错误: {str(e)}")
                
                # 创建诊断报告
                report_path = self.create_diagnostic_report(
                    error_msg=str(e),
                    api_response=None,
                    prompt=prompt,
                    model=model
                )
                
                if report_path:
                    error_message += f"\n\n诊断报告已保存到：{report_path}"
                    
                # 创建用户友好的错误消息
                user_error = f"生成图像失败: {str(e)}\n\n"
                user_error += "可能的解决方案:\n"
                user_error += "1. 检查您的API密钥是否正确\n"
                user_error += "2. 确认您有访问Gemini 2.0模型的权限\n"
                user_error += "3. 检查您的提示是否符合Google内容政策\n"
                user_error += "4. 尝试使用更简单的提示或不同的参考图像\n"
                user_error += "5. 更新google-genai库到最新版本: pip install -U google-generativeai\n"
                
                if "NoneType" in str(e) and "has no attribute" in str(e):
                    user_error += "\n此错误通常表明API返回了空响应。这可能是由于:\n"
                    user_error += "- 提示内容触发了内容过滤\n"
                    user_error += "- API密钥无效或配额已用尽\n"
                    user_error += "- Google服务器暂时问题\n"
                
                # 清理临时文件
                self.cleanup_temp_files(keep_temp_files)
                
                # 合并日志和错误信息
                full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + user_error
                return (self.generate_empty_image(width, height), full_text)
        
        except Exception as e:
            error_message = f"处理过程中出错: {str(e)}"
            self.log(f"Gemini图像生成错误: {str(e)}")
            
            # 创建诊断报告
            report_path = self.create_diagnostic_report(
                error_msg=str(e),
                api_response=None,
                prompt=prompt,
                model=model
            )
            
            if report_path:
                error_message += f"\n\n诊断报告已保存到：{report_path}"
                
            # 创建用户友好的错误消息
            user_error = f"生成图像失败: {str(e)}\n\n"
            user_error += "可能的解决方案:\n"
            user_error += "1. 检查您的API密钥是否正确\n"
            user_error += "2. 确认您有访问Gemini 2.0模型的权限\n"
            user_error += "3. 检查您的提示是否符合Google内容政策\n"
            user_error += "4. 尝试使用更简单的提示或不同的参考图像\n"
            user_error += "5. 更新google-genai库到最新版本: pip install -U google-generativeai\n"
            
            if "NoneType" in str(e) and "has no attribute" in str(e):
                user_error += "\n此错误通常表明API返回了空响应。这可能是由于:\n"
                user_error += "- 提示内容触发了内容过滤\n"
                user_error += "- API密钥无效或配额已用尽\n"
                user_error += "- Google服务器暂时问题\n"
            
            # 清理临时文件
            self.cleanup_temp_files(keep_temp_files)
            
            # 合并日志和错误信息
            full_text = "## 处理日志\n" + "\n".join(self.log_messages) + "\n\n## 错误\n" + user_error
            return (self.generate_empty_image(width, height), full_text)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "Google-Gemini": GeminiImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Google-Gemini": "Gemini 2.0 image"
} 