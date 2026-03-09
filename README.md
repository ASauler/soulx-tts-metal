# SoulX Podcast TTS — Apple Silicon MLX 加速版

基于 [SoulX-Podcast-1.7B-dialect](https://huggingface.co/Soul-AILab/SoulX-Podcast-1.7B-dialect) 的本地高质量语音合成服务，专门针对 Apple Silicon 做了 MLX 推理加速优化。

## 核心优化

本项目 fork 自官方 SoulX-Podcast，主要做了以下优化：

### 1. MLX 推理引擎（核心提速）

将 LLM 推理从 PyTorch MPS 迁移到 Apple MLX 框架：

| 指标 | HF 引擎 (PyTorch MPS) | MLX 引擎 |
|------|----------------------|----------|
| RTF (实时比) | ~11x | **1.0x - 2.3x** |
| 稳定性 | 100% | **100%** |
| 速度提升 | 基准 | **快 5-10 倍** |

**架构**: LLM 在 MLX GPU 上运行，Flow/HiFi-GAN 保持在 PyTorch MPS 上，两者共享 Metal GPU 但互不干扰。

**关键实现细节**:
- `mlx_lm.generate.generate_step` + `make_prompt_cache` 做 KV cache 管理
- `make_logits_processors(repetition_penalty=1.25)` — MLX 原生重复惩罚，防止 speech token 跑飞（这是稳定性的关键）
- `make_sampler(temp, top_p, top_k)` — 使用 MLX 内置采样器，自定义采样器会导致 Metal GPU page fault
- `mx.eval(mx.zeros(1))` 做 MLX/MPS 同步（不能用 `torch.mps.synchronize()`，会死锁）
- 每次 generate 调用新建 fresh KV cache，避免跨调用 cache 污染

### 2. Web UI + REST API

- Gradio Web 界面，支持单人 TTS / 声音克隆 / 多人播客
- RESTful API，可供程序调用
- `threading.Lock()` 序列化 GPU 访问，防止并发请求导致 Metal 崩溃

### 3. 声音克隆

- 上传 3-10 秒参考音频即可零样本克隆
- 集成 mlx-whisper 自动识别参考音频文本（ASR）

### 4. 多人播客生成

- 支持多角色对话脚本，自动生成播客音频
- 支持普通话、四川话、河南话、粤语

### 5. 稳定性优化

- 双轮预热：首次启动执行两次推理，完成 MLX JIT 编译，消除首次请求延迟
- speech token 上限（500 token = 20秒），防止极端情况下的跑飞
- RAS (Repetition Aware Sampling) + repetition penalty 双重保障

## 系统要求

- **硬件**: Apple Silicon Mac (M1/M2/M3/M4)，16GB+ 内存
- **系统**: macOS
- **Python**: 3.11+
- **存储**: ~10GB（模型文件）

> NVIDIA GPU 用户请直接使用[官方版本](https://github.com/Soul-AILab/SoulX-Podcast)，支持 CUDA + vLLM 加速。

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/jianyun8023/soulx-tts-metal.git
cd soulx-tts-metal
```

### 2. 安装依赖

```bash
bash setup.sh
```

或手动安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install mlx mlx-lm    # MLX 加速（Apple Silicon 必装）
pip install mlx-whisper    # ASR 识别（可选）
```

### 3. 下载模型

```bash
# HuggingFace 原始模型
huggingface-cli download --resume-download Soul-AILab/SoulX-Podcast-1.7B-dialect \
  --local-dir pretrained_models/SoulX-Podcast-1.7B-dialect

# 转换为 MLX 格式
python -m mlx_lm.convert \
  --hf-path pretrained_models/SoulX-Podcast-1.7B-dialect \
  --mlx-path pretrained_models/SoulX-Podcast-1.7B-dialect-mlx
```

### 4. 准备参考音频

将 WAV 格式的参考音频放入 `prompt_audios/` 目录，每个音频配一个同名 `.txt` 文件写上对应文本。

项目已包含示例参考音频（女声、男声、方言）。

### 5. 启动服务

```bash
source .venv/bin/activate
python app.py
```

打开浏览器访问: http://localhost:8000

## 使用方式

### Web 界面

- **单人语音**: 输入文本，选择说话人和方言，生成语音
- **声音克隆**: 上传参考音频，自动识别文本，克隆声音
- **多人播客**: 编写对话脚本，生成多角色播客

### REST API

```bash
# 单人 TTS
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "你好世界", "speaker": "女声1", "dialect": "普通话"}' \
  --output output.wav

# 多人播客
curl -X POST http://localhost:8000/api/podcast \
  -H "Content-Type: application/json" \
  -d '{"script": "@角色: 主持人, 女声1, 普通话\n@角色: 嘉宾, 男声1, 普通话\n\n[主持人]: 大家好！\n[嘉宾]: 你好！"}' \
  --output podcast.wav
```

### 播客脚本格式

```
# 角色定义
@角色: 主持人, 女声1, 普通话
@角色: 嘉宾, 男声1, 四川话

# 对话内容
[主持人]: 欢迎收听今天的节目！
[嘉宾]: 你好，很高兴来到这里。<|laughter|>
```

副语言标签: `<|laughter|>` 笑声 / `<|sigh|>` 叹气 / `<|breathing|>` 呼吸 / `<|coughing|>` 咳嗽

## 引擎切换

在 `app.py` 中修改 `llm_engine` 参数：

```python
model, dataset = initiate_model(
    seed=SEED,
    model_path=MODEL_PATH,
    llm_engine="mlx",   # "mlx" (推荐) | "hf" (兜底)
    fp16_flow=True
)
```

- `mlx`: MLX 加速，RTF ~1.5x，需要 MLX 模型文件
- `hf`: PyTorch HuggingFace，RTF ~11x，兼容性最好

## 项目结构

```
soulx-tts-metal/
├── app.py                           # 主应用 (Web UI + API)
├── soulxpodcast/
│   ├── engine/
│   │   ├── llm_engine.py            # HF/vLLM 推理引擎
│   │   └── mlx_engine.py            # MLX 推理引擎 (新增)
│   ├── models/
│   │   ├── soulxpodcast.py          # 主模型 (MLX/MPS 协调)
│   │   └── modules/                 # Flow, HiFi-GAN 等子模块
│   └── utils/                       # 工具函数
├── prompt_audios/                   # 参考音频
├── pretrained_models/               # 模型文件 (gitignore)
│   ├── SoulX-Podcast-1.7B-dialect/  # HF 原始模型
│   └── SoulX-Podcast-1.7B-dialect-mlx/  # MLX 转换模型
└── requirements.txt
```

## 踩坑记录

在 Apple Silicon 上跑 TTS 模型踩了不少坑，记录在这里供参考：

1. **MLX 自定义采样器导致 Metal GPU page fault**: `mx.sort/mx.argsort` 等操作在自定义采样器中会导致 GPU 崩溃。解决: 用 `mlx_lm.sample_utils.make_sampler`。

2. **`torch.mps.synchronize()` 与 MLX 死锁**: 两个框架同时使用 Metal GPU 时，MPS sync 会等待 MLX 的操作完成，但 MLX 也在等 Metal 资源，导致死锁。解决: 只用 `mx.eval(mx.zeros(1))` 做同步。

3. **MLX 缺少 repetition penalty 导致 30% 跑飞**: MLX `generate_step` 默认没有重复惩罚，speech token 一旦进入重复循环就停不下来。解决: 用 `mlx_lm.sample_utils.make_logits_processors(repetition_penalty=1.25)` 传给 `generate_step`。

4. **MLX JIT 首次编译慢**: 第一次推理需要编译 Metal shader，耗时 20-30 秒。解决: 启动时做两轮预热。

5. **MLX 量化模型 (4-bit/8-bit) EOS 不稳定**: 量化会损失 EOS token 的预测精度，导致更高的跑飞率。解决: 使用 fp32 MLX 模型。

6. **并发请求 Metal 崩溃**: Gradio 多个请求同时访问 GPU 会导致 `MTLCommandBuffer` 断言失败。解决: `threading.Lock()` 序列化所有 GPU 操作。

## 致谢

- [Soul AI Lab](https://github.com/Soul-AILab/SoulX-Podcast) — SoulX-Podcast 模型
- [Apple MLX](https://github.com/ml-explore/mlx) — Apple Silicon 机器学习框架
- [mlx-lm](https://github.com/ml-explore/mlx-examples) — MLX 语言模型推理库

## 许可证

Apache 2.0
