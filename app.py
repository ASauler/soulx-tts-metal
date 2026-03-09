import os
import json
import time
import threading
import torch
import gradio as gr
import soundfile as sf
import io
import numpy as np

# Metal GPU 全局锁：MLX 和 MPS 并发访问 Metal 会导致 GPU 崩溃
_gpu_lock = threading.Lock()
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# 导入 SoulX-Podcast 模块（已集成到项目中）
from soulxpodcast.config import SamplingParams
from soulxpodcast.utils.parser import podcast_format_parser
from soulxpodcast.utils.infer_utils import initiate_model, process_single_input
from soulxpodcast.utils.podcast_utils import (
    auto_parse_script, 
    validate_script, 
    PodcastScript,
    create_example_script,
    create_example_json_script
)

# ============== FastAPI 应用 ==============
app = FastAPI(title="SoulX Podcast TTS", description="基于 SoulX-Podcast-1.7B 的语音合成服务")

# ============== 配置 ==============
MODEL_PATH = "pretrained_models/SoulX-Podcast-1.7B-dialect"  # 方言模型路径
PROMPT_AUDIO_DIR = "prompt_audios"
SAMPLE_RATE = 24000  # 官方采样率
SEED = 1988

# ============== 自动扫描参考音频 ==============
def scan_prompt_audios():
    """
    自动扫描 prompt_audios 目录中的音频文件
    
    文件命名规范:
    - female_1.wav -> 女声1
    - male_1.wav -> 男声1
    - female_sweet.wav -> 女声sweet
    - 自定义名称.wav -> 自定义名称
    
    Returns:
        dict: 说话人配置字典
    """
    speakers = {}
    
    if not os.path.exists(PROMPT_AUDIO_DIR):
        print(f"[WARNING] 参考音频目录不存在: {PROMPT_AUDIO_DIR}")
        return speakers
    
    # 扫描目录中的 .wav 文件
    audio_files = [f for f in os.listdir(PROMPT_AUDIO_DIR) 
                   if f.endswith('.wav') and not f.startswith('.')]
    
    if not audio_files:
        print(f"[WARNING] 未找到参考音频文件在: {PROMPT_AUDIO_DIR}")
        return speakers
    
    for audio_file in sorted(audio_files):
        # 移除扩展名获取 ID
        speaker_id = audio_file[:-4]  # 移除 .wav
        
        # 生成友好的显示名称
        display_name = generate_display_name(speaker_id)
        
        # 读取音频文件的文本描述（如果有同名 .txt 文件）
        txt_file = os.path.join(PROMPT_AUDIO_DIR, f"{speaker_id}.txt")
        if os.path.exists(txt_file):
            with open(txt_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read().strip()
        else:
            # 默认文本
            prompt_text = f"这是{display_name}的参考音频。"
        
        speakers[display_name] = {
            "id": speaker_id,
            "audio": os.path.join(PROMPT_AUDIO_DIR, audio_file),
            "text": prompt_text
        }
    
    print(f"[INFO] 已扫描到 {len(speakers)} 个说话人: {', '.join(speakers.keys())}")
    return speakers


def generate_display_name(speaker_id):
    """
    根据文件名生成友好的显示名称
    
    规则:
    - female_1 -> 女声1
    - male_1 -> 男声1
    - female_sweet -> 女声sweet
    - custom_name -> custom_name (保持原样)
    """
    # 映射表
    prefix_map = {
        'female': '女声',
        'male': '男声',
        'neutral': '中性',
        'child': '童声',
    }
    
    # 尝试匹配前缀
    for eng_prefix, cn_prefix in prefix_map.items():
        if speaker_id.startswith(eng_prefix + '_'):
            suffix = speaker_id[len(eng_prefix) + 1:]
            return f"{cn_prefix}{suffix}"
    
    # 如果没有匹配，返回原始 ID
    return speaker_id


# 自动扫描并加载说话人配置
SPEAKERS = scan_prompt_audios()

# 如果没有扫描到任何音频，提供默认配置
if not SPEAKERS:
    print("[WARNING] 未找到参考音频，使用空配置")
    SPEAKERS = {}

# 方言配置
DIALECTS = {
    "普通话": {"code": "mandarin", "prefix": ""},
    "四川话": {"code": "sichuan", "prefix": "<|Sichuan|>"},
    "河南话": {"code": "henan", "prefix": "<|Henan|>"},
    "粤语": {"code": "yue", "prefix": "<|Yue|>"}
}

# ============== 全局模型 ==============
model = None
dataset = None
audio_cache = {}  # 缓存预处理的音频数据
is_warmed_up = False  # 模型预热标志


def preload_reference_audios():
    """
    预加载和缓存所有参考音频
    
    避免每次生成时重复加载和处理相同的参考音频文件。
    """
    import time
    import torchaudio
    
    if not SPEAKERS:
        print("[WARNING] 没有可用的说话人，跳过音频预加载")
        return
    
    print("[INFO] 🎵 预加载参考音频...")
    start_time = time.time()
    
    for speaker_name, speaker_config in SPEAKERS.items():
        audio_path = speaker_config["audio"]
        
        if not os.path.exists(audio_path):
            print(f"[WARNING] 参考音频不存在: {audio_path}")
            continue
        
        try:
            # 加载音频
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 重采样到 24kHz
            if sample_rate != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
                waveform = resampler(waveform)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # 缓存波形数据
            audio_cache[speaker_name] = {
                "waveform": waveform,
                "sample_rate": SAMPLE_RATE,
                "text": speaker_config["text"],
                "path": audio_path
            }
            
            duration = waveform.shape[1] / SAMPLE_RATE
            print(f"  ✓ {speaker_name}: {duration:.2f}s")
            
        except Exception as e:
            print(f"[ERROR] 加载 {speaker_name} 失败: {e}")
    
    elapsed = time.time() - start_time
    print(f"[INFO] ✅ 预加载完成！共 {len(audio_cache)} 个音频，耗时 {elapsed:.2f}s\n")


def warmup_model():
    """
    模型预热
    
    首次加载后运行一次推理，避免首次请求时的额外延迟。
    """
    global is_warmed_up
    
    if is_warmed_up or not SPEAKERS:
        return
    
    print("[INFO] 🔥 模型预热中...")
    import time
    start_time = time.time()
    
    try:
        # 使用第一个说话人进行预热
        first_speaker = list(SPEAKERS.keys())[0]
        warmup_text = "系统初始化。"
        
        # 执行两次完整推理，确保 MLX JIT 编译完成
        _ = generate_speech(warmup_text, first_speaker, "普通话")
        _ = generate_speech("你好世界。", first_speaker, "普通话")

        elapsed = time.time() - start_time
        print(f"[INFO] ✅ 预热完成！耗时 {elapsed:.2f}s\n")
        is_warmed_up = True
        
    except Exception as e:
        print(f"[WARNING] 模型预热失败: {e}\n")


def load_model():
    """加载 SoulX-Podcast 模型（单例模式 + 预加载优化）"""
    global model, dataset
    
    if model is None:
        print("[INFO] 正在加载 SoulX-Podcast 模型...")
        
        # 检查模型路径
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"模型路径不存在: {MODEL_PATH}\n"
                f"请先下载模型:\n"
                f"huggingface-cli download --resume-download Soul-AILab/SoulX-Podcast-1.7B-dialect "
                f"--local-dir {MODEL_PATH}"
            )
        
        # 初始化模型
        # 优先 MLX (Apple Silicon 加速)，不可用则回退 HF
        try:
            from soulxpodcast.engine.llm_engine import SUPPORT_MLX
            mlx_model_path = MODEL_PATH + "-mlx"
            if SUPPORT_MLX and os.path.exists(mlx_model_path):
                engine = "mlx"
            else:
                engine = "hf"
                if not SUPPORT_MLX:
                    print("[INFO] MLX 不可用，使用 HF 引擎")
                else:
                    print(f"[INFO] MLX 模型不存在 ({mlx_model_path})，使用 HF 引擎")
        except Exception:
            engine = "hf"

        model, dataset = initiate_model(
            seed=SEED,
            model_path=MODEL_PATH,
            llm_engine=engine,
            fp16_flow=True
        )
        
        print("[INFO] 模型加载完成！\n")
        
        # 预加载参考音频（如果还没加载）
        if not audio_cache:
            preload_reference_audios()
        
        # 模型预热
        warmup_model()
    
    return model, dataset


def get_cached_audio(speaker_name: str):
    """
    获取缓存的音频数据
    
    Args:
        speaker_name: 说话人名称
    
    Returns:
        缓存的音频数据，如果不存在则返回 None
    """
    return audio_cache.get(speaker_name)


def preprocess_text(text: str) -> str:
    """
    预处理输入文本
    
    处理内容:
    1. 将多行文本合并为单行
    2. 移除多余的空白字符
    3. 保留标点符号和副语言标签
    
    Args:
        text: 原始文本
    
    Returns:
        处理后的文本
    """
    # 将换行符替换为逗号（保持语义分隔）
    text = text.replace('\r\n', '，')  # Windows 换行
    text = text.replace('\n', '，')    # Unix/Mac 换行
    text = text.replace('\r', '，')    # 旧 Mac 换行
    
    # 移除连续的逗号
    while '，，' in text:
        text = text.replace('，，', '，')
    
    # 移除多余的空白字符（但保留副语言标签中的内容）
    import re
    # 保护副语言标签
    tags = re.findall(r'<\|[^|]+\|>', text)
    for i, tag in enumerate(tags):
        text = text.replace(tag, f'__TAG_{i}__')
    
    # 移除多余空白
    text = ' '.join(text.split())
    
    # 恢复副语言标签
    for i, tag in enumerate(tags):
        text = text.replace(f'__TAG_{i}__', tag)
    
    # 移除首尾逗号和空格
    text = text.strip('，').strip()
    
    return text


def generate_speech(text: str, speaker: str, dialect: str):
    """
    生成语音
    
    Args:
        text: 输入文本（支持多行）
        speaker: 说话人名称（如 "女声1"）
        dialect: 方言名称（如 "普通话"）
    
    Returns:
        audio_array: 音频数据 (numpy array)
    """
    # 预处理文本
    text = preprocess_text(text)
    
    if not text:
        raise ValueError("文本内容为空")
    
    print(f"[INFO] 处理后的文本: {text[:100]}{'...' if len(text) > 100 else ''}")
    
    # 加载模型
    model, dataset = load_model()
    
    # 获取说话人配置
    speaker_config = SPEAKERS.get(speaker)
    if not speaker_config:
        raise ValueError(f"未知说话人: {speaker}")
    
    # 获取方言配置
    dialect_config = DIALECTS.get(dialect)
    if not dialect_config:
        raise ValueError(f"未知方言: {dialect}")
    
    dialect_prefix = dialect_config["prefix"]
    use_dialect_prompt = len(dialect_prefix) > 0
    
    # 构建输入数据（按照官方格式）
    if use_dialect_prompt:
        # 方言模式：需要 dialect_prompt
        inputs_dict = {
            "speakers": {
                "S1": {
                    "prompt_audio": speaker_config["audio"],
                    "prompt_text": speaker_config["text"],
                    "dialect_prompt": f"{dialect_prefix}{speaker_config['text']}"
                }
            },
            "text": [
                ["S1", f"{dialect_prefix}{text}"]
            ]
        }
    else:
        # 普通话模式：不需要 dialect_prompt
        inputs_dict = {
            "speakers": {
                "S1": {
                    "prompt_audio": speaker_config["audio"],
                    "prompt_text": speaker_config["text"]
                }
            },
            "text": [
                ["S1", text]
            ]
        }
    
    import time
    
    # ========== 阶段 1: 解析输入 ==========
    stage_start = time.time()
    inputs = podcast_format_parser(inputs_dict)
    parse_time = time.time() - stage_start
    print(f"[PERF] 输入解析: {parse_time:.3f}s")
    
    # ========== 阶段 2: 预处理数据（CPU密集）==========
    stage_start = time.time()
    data = process_single_input(
        dataset,
        inputs['text'],
        inputs['prompt_wav'],
        inputs['prompt_text'],
        inputs['use_dialect_prompt'],
        inputs['dialect_prompt_text'],
    )
    preprocess_time = time.time() - stage_start
    print(f"[PERF] 数据预处理（音频tokenization）: {preprocess_time:.3f}s [CPU]")
    
    # ========== 阶段 3: 模型推理（GPU加速）==========
    stage_start = time.time()
    print(f"[INFO] 开始生成语音...")
    results_dict = model.forward_longform(**data)
    inference_time = time.time() - stage_start
    print(f"[PERF] 模型推理（LLM+Flow+Vocoder）: {inference_time:.3f}s [GPU]")
    
    # ========== 阶段 4: 后处理 ==========
    stage_start = time.time()
    target_audio = None
    for wav in results_dict["generated_wavs"]:
        if target_audio is None:
            target_audio = wav
        else:
            target_audio = torch.cat([target_audio, wav], dim=1)
    
    # 转换为 numpy 数组
    audio_array = target_audio.cpu().squeeze(0).numpy()
    postprocess_time = time.time() - stage_start
    print(f"[PERF] 后处理: {postprocess_time:.3f}s")
    
    # 总结
    total_time = parse_time + preprocess_time + inference_time + postprocess_time
    audio_duration = len(audio_array) / SAMPLE_RATE
    rtf = total_time / audio_duration  # Real-Time Factor
    print(f"[INFO] ✅ 生成完成！音频: {audio_duration:.2f}s | 耗时: {total_time:.2f}s | RTF: {rtf:.2f}x")
    
    return audio_array


def generate_multiperson_podcast(script: PodcastScript, silence_duration: float = 0.5):
    """
    生成多人播客
    
    Args:
        script: 播客脚本对象
        silence_duration: 对话之间的静音时长（秒）
    
    Returns:
        audio_array: 完整的播客音频 (numpy array)
    """
    import time
    
    print("=" * 60)
    print("🎙️  开始生成多人播客")
    print("=" * 60)
    
    total_start = time.time()
    
    # 验证脚本
    is_valid, error_msg = validate_script(script, list(SPEAKERS.keys()))
    if not is_valid:
        raise ValueError(f"脚本验证失败: {error_msg}")
    
    print(f"[INFO] 角色数量: {len(script.speakers)}")
    print(f"[INFO] 对话数量: {len(script.dialogues)}")
    for name, config in script.speakers.items():
        print(f"  - {name}: {config['voice']} ({config['dialect']})")
    print()
    
    # 生成静音片段
    silence_samples = int(SAMPLE_RATE * silence_duration)
    silence = np.zeros(silence_samples, dtype=np.float32)
    
    # 存储所有音频片段
    audio_segments = []
    
    # 逐个生成对话
    for i, dialogue in enumerate(script.dialogues, 1):
        speaker = dialogue["speaker"]
        text = dialogue["text"]
        
        speaker_config = script.speakers[speaker]
        voice = speaker_config["voice"]
        dialect = speaker_config["dialect"]
        
        print(f"[{i}/{len(script.dialogues)}] 生成中: [{speaker}] {text[:50]}{'...' if len(text) > 50 else ''}")
        
        try:
            # 生成单句语音
            audio = generate_speech(text, voice, dialect)
            audio_segments.append(audio)
            
            # 添加静音（除了最后一句）
            if i < len(script.dialogues):
                audio_segments.append(silence)
            
            print(f"  ✓ 完成！音频时长: {len(audio)/SAMPLE_RATE:.2f}s\n")
            
        except Exception as e:
            print(f"  ✗ 生成失败: {e}\n")
            raise
    
    # 合并所有音频片段
    print("[INFO] 合并音频片段...")
    final_audio = np.concatenate(audio_segments)
    
    # 统计信息
    total_time = time.time() - total_start
    total_duration = len(final_audio) / SAMPLE_RATE
    
    print("=" * 60)
    print("✅ 多人播客生成完成！")
    print(f"  - 总时长: {total_duration:.2f}s")
    print(f"  - 耗时: {total_time:.2f}s")
    print(f"  - 平均速度: {total_duration/total_time:.2f}x 实时")
    print("=" * 60)
    
    return final_audio


# ============== REST API ==============
class TTSRequest(BaseModel):
    text: str
    speaker: str = "女声1"
    dialect: str = "普通话"


class PodcastRequest(BaseModel):
    script: str  # 脚本文本（简单格式或 JSON 格式）
    silence_duration: float = 0.5  # 对话间隔（秒）


@app.post("/api/tts")
def api_tts(req: TTSRequest):
    """REST API 接口：文本转语音（单人）"""
    try:
        with _gpu_lock:
            audio_array = generate_speech(req.text, req.speaker, req.dialect)

        # 写入 WAV 格式
        buf = io.BytesIO()
        sf.write(buf, audio_array, SAMPLE_RATE, format="wav")
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="audio/wav")

    except Exception as e:
        return {"error": str(e)}


@app.post("/api/podcast")
def api_podcast(req: PodcastRequest):
    """REST API 接口：多人播客生成"""
    try:
        # 解析脚本
        script = auto_parse_script(req.script)
        
        # 生成播客
        with _gpu_lock:
            audio_array = generate_multiperson_podcast(script, req.silence_duration)

        # 写入 WAV 格式
        buf = io.BytesIO()
        sf.write(buf, audio_array, SAMPLE_RATE, format="wav")
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="audio/wav")
    
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/podcast/example")
def api_podcast_example(format: str = "simple"):
    """
    获取示例脚本
    
    Args:
        format: 脚本格式 ("simple" 或 "json")
    """
    if format == "json":
        return {
            "format": "json",
            "script": create_example_json_script()
        }
    else:
        return {
            "format": "simple",
            "script": create_example_script()
        }


# ============== Gradio Web 界面 ==============
def tts_web(text, speaker, dialect):
    """Gradio 界面的 TTS 函数（单人）"""
    try:
        if not text or len(text.strip()) == 0:
            return None, "请输入文本！"

        start_time = time.time()

        with _gpu_lock:
            audio_array = generate_speech(text, speaker, dialect)
        
        # 计算耗时
        elapsed_time = time.time() - start_time
        
        # 计算音频时长
        audio_duration = len(audio_array) / SAMPLE_RATE
        
        # Gradio 返回格式: (sample_rate, audio_array)
        status_msg = f"✅ 生成成功！\n⏱️ 耗时: {elapsed_time:.2f} 秒\n🎵 音频时长: {audio_duration:.2f} 秒"
        return (SAMPLE_RATE, audio_array), status_msg
    
    except Exception as e:
        return None, f"❌ 生成失败: {str(e)}"


def asr_recognize(ref_audio):
    """用 mlx-whisper 识别参考音频文本"""
    try:
        if ref_audio is None:
            return "请先上传参考音频！"
        import mlx_whisper
        import tempfile
        if isinstance(ref_audio, tuple):
            sr, audio_data = ref_audio
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp_file.name, audio_data, sr)
            audio_path = tmp_file.name
        else:
            audio_path = ref_audio
        with _gpu_lock:
            result = mlx_whisper.transcribe(audio_path, language='zh')
        return result['text'].strip()
    except Exception as e:
        return f"识别失败: {str(e)}"


def clone_web(text, ref_audio, ref_text, dialect):
    """Gradio 界面的零样本克隆函数"""
    try:
        if not text or len(text.strip()) == 0:
            return None, "请输入要合成的文本！"
        if ref_audio is None:
            return None, "请上传参考音频！"
        if not ref_text or len(ref_text.strip()) == 0:
            return None, "请输入参考音频对应的文本！"

        # ref_audio 是 gradio 返回的 (sample_rate, numpy_array)
        if isinstance(ref_audio, tuple):
            sr, audio_data = ref_audio
            import tempfile
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp_file.name, audio_data, sr)
            ref_audio_path = tmp_file.name
        else:
            ref_audio_path = ref_audio

        # 临时注册为说话人，用唯一 key 防止并发冲突
        import uuid
        clone_key = f"__clone_{uuid.uuid4().hex[:8]}__"
        SPEAKERS[clone_key] = {
            "id": "clone",
            "audio": ref_audio_path,
            "text": ref_text.strip()
        }

        start_time = time.time()
        try:
            with _gpu_lock:
                audio_array = generate_speech(text, clone_key, dialect)
        finally:
            SPEAKERS.pop(clone_key, None)
        elapsed_time = time.time() - start_time

        audio_duration = len(audio_array) / SAMPLE_RATE
        status_msg = f"✅ 克隆生成成功！\n⏱️ 耗时: {elapsed_time:.2f} 秒\n🎵 音频时长: {audio_duration:.2f} 秒"
        return (SAMPLE_RATE, audio_array), status_msg

    except Exception as e:
        import traceback
        return None, f"❌ 生成失败: {str(e)}\n\n{traceback.format_exc()}"


def podcast_web(script_text, silence_duration):
    """Gradio 界面的多人播客生成函数"""
    try:
        if not script_text or len(script_text.strip()) == 0:
            return None, "请输入播客脚本！"
        
        # 记录开始时间
        start_time = time.time()
        
        # 解析脚本
        script = auto_parse_script(script_text)
        
        # 生成播客
        with _gpu_lock:
            audio_array = generate_multiperson_podcast(script, silence_duration)
        
        # 计算耗时
        elapsed_time = time.time() - start_time
        
        # 计算音频时长
        audio_duration = len(audio_array) / SAMPLE_RATE
        
        # 统计对话数量
        dialogue_count = len(script.dialogues)
        speaker_count = len(script.speakers)
        
        # Gradio 返回格式: (sample_rate, audio_array)
        status_msg = (
            f"✅ 播客生成成功！\n"
            f"⏱️ 总耗时: {elapsed_time:.2f} 秒\n"
            f"🎵 音频时长: {audio_duration:.2f} 秒\n"
            f"👥 参与者: {speaker_count} 人\n"
            f"💬 对话数: {dialogue_count} 段"
        )
        return (SAMPLE_RATE, audio_array), status_msg
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return None, f"❌ 生成失败: {str(e)}\n\n详细错误:\n{error_detail}"


# 创建 Gradio 界面
with gr.Blocks(title="SoulX Podcast TTS", theme=gr.themes.Soft()) as gr_app:
    gr.Markdown("""
    # 🎙️ SoulX Podcast TTS
    
    基于 **SoulX-Podcast-1.7B-dialect** 模型的高质量语音合成服务
    
    ✨ 支持单人语音、多人播客、多方言、零样本声音克隆
    """)
    
    with gr.Tabs():
        # ========== 单人语音生成 ==========
        with gr.Tab("🎤 单人语音"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="输入文本",
                        placeholder="请输入要转换的文本...\n\n支持副语言标签：<|laughter|> (笑声)、<|sigh|> (叹气)、<|breathing|> (呼吸)、<|coughing|> (咳嗽)",
                        lines=5
                    )
                    
                    with gr.Row():
                        speaker_dropdown = gr.Dropdown(
                            choices=list(SPEAKERS.keys()),
                            value="女声1" if SPEAKERS else None,
                            label="说话人"
                        )
                        
                        dialect_dropdown = gr.Dropdown(
                            choices=list(DIALECTS.keys()),
                            value="普通话",
                            label="方言"
                        )
                    
                    generate_btn = gr.Button("🎵 生成语音", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    status_text = gr.Textbox(label="状态", value="就绪", lines=4)
                    audio_output = gr.Audio(label="生成的语音", type="numpy")
            
            # 事件绑定
            generate_btn.click(
                fn=tts_web,
                inputs=[text_input, speaker_dropdown, dialect_dropdown],
                outputs=[audio_output, status_text]
            )
        
        # ========== 零样本声音克隆 ==========
        with gr.Tab("🎭 声音克隆"):
            with gr.Row():
                with gr.Column(scale=2):
                    clone_text_input = gr.Textbox(
                        label="要合成的文本",
                        placeholder="输入想用克隆声音说的话...",
                        lines=4
                    )
                    clone_ref_audio = gr.Audio(
                        label="上传参考音频（3-15秒，清晰人声）",
                        type="numpy"
                    )
                    clone_ref_text = gr.Textbox(
                        label="参考音频对应的文本",
                        placeholder="请准确输入参考音频中说的话（或点击下方按钮自动识别）...",
                        lines=2
                    )
                    asr_btn = gr.Button("🎤 自动识别文本 (Whisper)", size="sm")
                    clone_dialect = gr.Dropdown(
                        choices=list(DIALECTS.keys()),
                        value="普通话",
                        label="方言"
                    )
                    clone_btn = gr.Button("🎭 克隆生成", variant="primary", size="lg")

                with gr.Column(scale=1):
                    clone_status = gr.Textbox(label="状态", value="就绪", lines=4)
                    clone_audio_output = gr.Audio(label="克隆语音", type="numpy")

            asr_btn.click(
                fn=asr_recognize,
                inputs=[clone_ref_audio],
                outputs=[clone_ref_text]
            )
            clone_btn.click(
                fn=clone_web,
                inputs=[clone_text_input, clone_ref_audio, clone_ref_text, clone_dialect],
                outputs=[clone_audio_output, clone_status]
            )

        # ========== 多人播客生成 ==========
        with gr.Tab("🎙️ 多人播客"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("""
                    ### 📝 播客脚本格式说明
                    
                    **简单格式示例：**
                    ```
                    # 角色定义
                    @角色: 主持人, 女声1, 普通话
                    @角色: 嘉宾, 男声1, 普通话
                    
                    # 对话内容
                    [主持人]: 大家好，欢迎收听今天的节目！
                    [嘉宾]: 你好，很高兴来到这里。
                    ```
                    
                    **支持副语言标签:** `<|laughter|>` (笑声)、`<|sigh|>` (叹气) 等
                    """)
                    
                    script_input = gr.Textbox(
                        label="播客脚本",
                        placeholder="在此输入播客脚本...\n或点击下方按钮加载示例脚本",
                        lines=15,
                        value=""
                    )
                    
                    with gr.Row():
                        load_example_btn = gr.Button("📄 加载示例脚本", size="sm")
                        clear_btn = gr.Button("🗑️ 清空", size="sm")
                    
                    silence_slider = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.5,
                        step=0.1,
                        label="对话间隔（秒）",
                        info="对话之间的静音时长"
                    )
                    
                    generate_podcast_btn = gr.Button("🎙️ 生成播客", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    podcast_status = gr.Textbox(label="状态", value="就绪", lines=6)
                    podcast_audio = gr.Audio(label="生成的播客", type="numpy")
            
            # 事件绑定
            generate_podcast_btn.click(
                fn=podcast_web,
                inputs=[script_input, silence_slider],
                outputs=[podcast_audio, podcast_status]
            )
            
            load_example_btn.click(
                fn=lambda: create_example_script(),
                inputs=[],
                outputs=[script_input]
            )
            
            clear_btn.click(
                fn=lambda: "",
                inputs=[],
                outputs=[script_input]
            )
            
            gr.Markdown("""
            ### 💡 使用提示
            
            1. **定义角色**: 使用 `@角色: 名称, 声音, 方言` 格式定义参与者
            2. **编写对话**: 使用 `[角色名]: 对话内容` 格式编写对话
            3. **添加情感**: 在对话中插入副语言标签，如 `<|laughter|>` 增强表现力
            4. **调整间隔**: 根据需要调整对话之间的静音时长
            
            **可用声音：** """ + ", ".join(SPEAKERS.keys() if SPEAKERS else ["请先配置参考音频"]) + """
            
            **可用方言：** """ + ", ".join(DIALECTS.keys()) + """
            """)
    
    gr.Markdown("""
    ---
    ### 📚 API 调用示例
    
    **单人语音生成:**
    ```bash
    curl -X POST http://localhost:8000/api/tts \\
      -H "Content-Type: application/json" \\
      -d '{"text": "你好，欢迎来到 SoulX 播客！", "speaker": "女声1", "dialect": "普通话"}' \\
      --output output.wav
    ```
    
    **多人播客生成:**
    ```bash
    curl -X POST http://localhost:8000/api/podcast \\
      -H "Content-Type: application/json" \\
      -d '{
        "script": "# 角色定义\\n@角色: 主持人, 女声1, 普通话\\n@角色: 嘉宾, 男声1, 普通话\\n\\n[主持人]: 大家好！\\n[嘉宾]: 你好！",
        "silence_duration": 0.5
      }' \\
      --output podcast.wav
    ```
    
    **获取示例脚本:**
    ```bash
    curl http://localhost:8000/api/podcast/example?format=simple
    ```
    
    ---
    **模型**: [SoulX-Podcast-1.7B-dialect](https://huggingface.co/Soul-AILab/SoulX-Podcast-1.7B-dialect) | 
    **项目**: [GitHub](https://github.com/Soul-AILab/SoulX-Podcast)
    """)

# 挂载 Gradio 到 FastAPI
app = gr.mount_gradio_app(app, gr_app, path="/")

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🎙️  SoulX Podcast TTS 服务启动中...")
    print("=" * 60)
    print(f"📂 模型路径: {MODEL_PATH}")
    print(f"🎵 采样率: {SAMPLE_RATE} Hz")
    print(f"🌐 访问地址: http://localhost:8000")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
