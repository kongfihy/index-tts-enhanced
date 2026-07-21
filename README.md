<div align="center">
  <img src="assets/index_icon.png" width="180" alt="IndexTTS Enhanced" />

# IndexTTS Enhanced

**面向本地视频补录与中文配音工作流的 IndexTTS2 非官方增强版本**

[上游项目](https://github.com/index-tts/index-tts) · [开发计划](docs/DEVELOPMENT_PLAN.md) · [许可证](LICENSE)
</div>

> [!IMPORTANT]
> 本项目是基于 [IndexTTS2](https://github.com/index-tts/index-tts) 的非官方衍生版本，主要增加本地任务管理、多候选生成、音频交付处理和 macOS 使用体验。
> 本项目不代表 IndexTTS、IndexTeam 或 bilibili 的官方版本、认可或担保。原始模型、代码和相关权利归原项目权利方所有。

## 项目定位

IndexTTS Enhanced 不是新的基础语音模型，也不重新分发模型权重。它保留 IndexTTS2 的核心推理能力，在其上补充更适合视频制作和本地补录的完整工作流：

```text
参考音频检查
→ 文本和生成参数确认
→ 一个或多个可复现候选
→ 原始干声与可选交付匹配版
→ 项目化任务历史
→ 试听、选择最佳版本和下载
```

项目当前以本地单机和可信局域网使用为主，重点是：

- 不覆盖原始输出；
- 不丢失历史版本；
- 参数和随机种子可追溯；
- 音频后处理可开关并可进行 A/B 对比；
- 在 Apple Silicon / MPS 环境中保持可控的内存占用；
- 所有真实推理、测试素材和生产部署由使用者本地控制。

## 相比上游增加了什么

### 中文生成界面

- 中文优先的 WebUI；
- 深浅色自适应和响应式布局；
- 参考音频时长、响度、削波和静音比例检查；
- 更稳妥的参考音频裁剪与预处理；
- 文本分句预览和更清晰的高级参数说明。

### 多候选与可复现生成

- 固定随机种子；
- 单次任务可生成 1～3 个候选；
- 每个候选使用独立 seed 和独立输出文件；
- 相同项目自动保存为多个历史版本；
- 结果不会因为下一次生成而被覆盖；
- 可以在任务中心标记并持久保存“最佳版本”。

当前模型推理仍然是**单 Worker 串行执行**。多进程推理尚未作为默认功能开放，必须先完成真实三候选基线和稳定性测试，详见 [开发计划](docs/DEVELOPMENT_PLAN.md)。

### 原始干声与交付版本

每个候选始终保留模型原始干声，并可额外生成：

- 安全响度匹配版；
- 与参考音频采样率、声道数和有效人声响度匹配的交付版；
- 约 `-1 dBFS` 峰值保护；
- 单独下载或打包下载全部结果。

这些交付副本不会触发第二次模型推理，也不会替换原始输出。

### 本地任务中心

- SQLite 本地任务历史；
- 排队、准备、生成、后处理、完成、失败和取消状态；
- 项目分组和版本展开；
- 当前任务进度和安全取消；
- 每个生成结果单独下载；
- 全部候选打包下载；
- 局域网可查看和下载，本机拥有管理操作权限；
- 服务重启后恢复无法继续的任务状态，避免幽灵排队任务。

### Apple Silicon 与 macOS

- MPS 推理内存清理和缓存释放；
- 分段推理临时张量回收；
- Hugging Face 和运行时缓存使用稳定的 macOS 可写目录；
- macOS 菜单栏应用源码；
- 本机/局域网访问切换、服务状态、任务状态和取消入口。

## 快速开始

### 1. 准备环境

需要：

- macOS 或支持 PyTorch 的 Linux 环境；
- Python 与 [uv](https://docs.astral.sh/uv/)；
- Git 和 [Git LFS](https://git-lfs.com/)；
- 足够的磁盘与内存空间存放 IndexTTS2 模型。

```bash
git lfs install
git clone https://github.com/kongfihy/index-tts-enhanced.git
cd index-tts-enhanced
git lfs pull
uv sync --all-extras
```

### 2. 下载官方模型

模型权重不包含在本仓库中。可以通过 Hugging Face 下载：

```bash
uv tool install "huggingface_hub[cli]"
hf download IndexTeam/IndexTTS-2 --local-dir checkpoints
```

也可以参考上游项目提供的 [ModelScope 下载方式](https://github.com/index-tts/index-tts#model-download)。

### 3. 启动 WebUI

```bash
uv run webui.py \
  --host 127.0.0.1 \
  --port 7860 \
  --model_dir ./checkpoints
```

启动后：

- 生成页面：`http://127.0.0.1:7860`
- 任务中心：`http://127.0.0.1:7861`

如果需要在可信局域网中访问：

```bash
uv run webui.py \
  --host 0.0.0.0 \
  --port 7860 \
  --model_dir ./checkpoints
```

> [!WARNING]
> 当前 WebUI 没有面向公网的登录、权限、配额和安全隔离。不要直接暴露到互联网，只应在本机或可信局域网中使用。

## macOS 菜单栏应用

菜单栏应用源码位于：

```text
apps/macos/IndexTTSMenuBar
```

运行测试：

```bash
swift test --package-path apps/macos/IndexTTSMenuBar
```

构建与安装说明见：

```text
apps/macos/IndexTTSMenuBar/README.md
```

构建开发应用不会自动替换已经安装的生产版本。

## 测试

Python 测试：

```bash
PYTHONPATH="$PWD" uv run python -m unittest discover -s tests -p 'test_*.py'
```

Swift 测试：

```bash
swift test --package-path apps/macos/IndexTTSMenuBar
```

真实模型推理不属于默认单元测试。真实参考音频、模型权重、生成结果、数据库和日志不应提交到仓库。

## 输出与隐私

仓库默认忽略：

- `checkpoints/` 模型权重；
- `outputs/` 生成结果；
- `prompts/` 用户参考音频；
- 常见音频和视频文件；
- SQLite 数据库、日志和进程状态；
- `.env`、密钥和签名材料；
- macOS 构建产物和本地交接记录。

提交或发布前仍应检查 Git 状态和差异，不要使用未经审查的 `git add .`。

## 开发计划

近期顺序：

1. 将候选数量移到主生成操作区；
2. 提供“快速生成 1 个”和“生成 3 个候选”；
3. 在主页面使用候选卡片直接试听、下载和标记最佳版本；
4. 使用用户确认的真实音频建立默认平衡模式的三候选串行基线；
5. 基线完成后比较单进程串行与两个独立模型进程；
6. 只有吞吐、稳定性、可复现性和内存测试全部通过，才开放可配置并发。

完整计划见 [docs/DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md)。

## 与上游同步

本项目保留 Fork 关系：

```text
origin   → kongfihy/index-tts-enhanced
upstream → index-tts/index-tts
```

同步上游前应先检查改动范围并在独立分支验证，不应直接覆盖增强功能或已经完成的 MPS 修复。

```bash
git fetch upstream
```

## Codex 辅助开发说明

本项目的部分需求整理、架构分析、代码修改、测试设计、浏览器回归和文档工作，由 **OpenAI Codex** 在项目维护者的明确指令、审核和本地权限边界下辅助完成。

Codex 是开发辅助工具，不是本项目的维护者或发布者。功能取舍、试听判断、测试素材、许可证遵守、部署与最终发布决定均由项目维护者负责。

该说明仅用于透明披露开发方式，不表示 OpenAI、Codex、IndexTTS、IndexTeam 或 bilibili 对本项目提供认可、担保或联合发布。

## 上游项目与论文

- 上游代码：[index-tts/index-tts](https://github.com/index-tts/index-tts)
- IndexTTS2 模型：[IndexTeam/IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2)
- IndexTTS2 论文：[arXiv:2506.21619](https://arxiv.org/abs/2506.21619)
- 上游演示：[IndexTTS2 Demo](https://index-tts.github.io/index-tts2.github.io/)

如果问题只与原始模型、权重或官方实现有关，请优先查阅上游文档；如果问题与任务中心、多候选、音频交付、macOS 菜单栏或本增强工作流有关，请在本仓库反馈。

## 许可证

本仓库保留原项目的 `LICENSE` 和版权信息。模型、原始代码以及本项目的衍生修改均受仓库内许可证约束。

使用、修改、分发或提供下游版本前，请完整阅读 [LICENSE](LICENSE)，并自行确认具体使用场景是否满足其中关于衍生作品、下游接收者、商业规模、内容合规和权利保护的要求。
