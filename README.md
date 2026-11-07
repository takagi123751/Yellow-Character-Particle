# Yellow Character Particle Animation

一个使用 NumPy、ModernGL 和 GLFW 编写的实时三维粒子动画。

人物由约 201,000 个粒子组成，包含：

- 黄色身体与浅米色腹部
- 绿色眼睛与黑色瞳孔
- 大笑动作、牙齿、眯眼和身体摇摆
- 星光背景与粒子聚合入场

## 环境要求

- Python 3.10 或更高版本
- 支持 OpenGL 3.3 的显卡和驱动

## 运行

### Windows 一键运行

双击 `start_character.bat`。首次运行时脚本会自动创建 `.venv` 并安装依赖，因此需要联网；安装完成后可以离线运行。

### 命令行

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe yellow_character.py
```

### VS Code

打开 `yellow_character.code-workspace`，按 `F5`，必要时选择“运行黄色粒子人物”。

按 `Esc` 或关闭动画窗口即可退出。
