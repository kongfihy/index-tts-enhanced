# IndexTTS MenuBar

独立的 macOS 菜单栏控制器，用于在不显示 Terminal 的情况下启动和管理现有 IndexTTS WebUI。

## 默认路径

- IndexTTS：`~/index-tts`
- uv：`~/.local/bin/uv`
- WebUI：`http://127.0.0.1:7860`
- 后端日志：`~/Library/Logs/IndexTTSMenuBar/backend.log`
- 控制器日志：`~/Library/Logs/IndexTTSMenuBar/launcher.log`

## 构建

```bash
cd apps/macos/IndexTTSMenuBar
swift test
./scripts/build_app.sh
```

构建结果：

```text
dist/IndexTTS Menu.app
```

## 安装

```bash
./scripts/install_app.sh
```

第一版默认不自动启动模型。打开菜单栏 APP 后，点击“启动服务”。

## 环境变量覆盖

调试或移动目录后可以覆盖：

```bash
INDEXTTS_REPO_PATH=/path/to/index-tts
INDEXTTS_UV_PATH=/path/to/uv
INDEXTTS_PORT=7860
INDEXTTS_MODEL_DIR=./checkpoints
```

## 安全策略

默认使用 `127.0.0.1`，仅本机访问。只有用户主动开启“允许局域网访问”并重启后，才使用 `0.0.0.0`。
