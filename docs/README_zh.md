# 项目 11 中文使用指南

## 先运行演示

解压 WindowsDemo 后，双击 pendulum-lab.exe。默认模型已经内置，不需要安装 Rust、Python，
也不需要联网。project11.exe 是命令行版本，直接启动也会打开 GUI；平时展示优先使用没有控制台窗口的 pendulum-lab.exe。

需要 Windows x64 和能够运行 OpenGL 的显示环境。程序为课程用途的未签名构建，遵守学校电脑的软件安全策略。
不要关闭系统防护。如果场地限制运行，提前联系管理员或在获准的电脑从源码编译。

## 材料阅读顺序

1. 阅读 Project11_Report.docx，先理解状态、增量预测、MSE 梯度和数据划分。
2. 打开 Project11_Presentation.pptx，主体 10 页，另外 2 页是提问附录。
3. 对照 SPEAKER_NOTES_bilingual.md 排练英文讲解，目标总计约 7 分钟。
4. 按 DEMO_AND_QA_zh.md 排练 3 分钟演示和常见问题。
5. 填写 TEAM_RECORD.md 和报告封面的实名信息、学号、截止日期与真实分工。

## 从源码运行

安装 Rust stable 和 Windows C++ Build Tools。在仓库根目录的 PowerShell 中执行：

    cargo run --release --bin pendulum-lab

复现完整实验时建议使用新目录，不要覆盖仓库中供报告引用的冻结结果：

    cargo run --release --bin project11 -- reproduce --out experiment
    cargo run --release --bin project11 -- benchmark --out experiment --epochs 50
    python -m pip install -r scripts/requirements.txt
    python scripts/benchmark_numpy.py --root experiment

第一条命令完成数据生成、两种模型的训练和测试。后两条涉及 Rust/Python 对比，
因此需要 Python 和 NumPy；这不是桌面演示的运行前提。

## 关键实验设置

80 条不同初态轨迹，每条 10 秒、500 次转移，总计 40,000 个样本。
按整条轨迹分为 56 条训练、12 条验证和 12 条测试。标准化统计量只由训练集计算。
网络预测的是 0.02 秒的状态增量，而不是直接调用已知单摆方程。

主网络为 2-32-32-2 Tanh，共 1,218 个参数；小网络为 2-3-2 Sigmoid，共 17 个参数。
Sigmoid、ReLU、Tanh 和线性函数均为手写，前向、反向和 SGD 也都是手写，ndarray 只提供矩阵运算。

## 数字怎样解读

- 一步角度 RMSE 约 0.000274 弧度。标准化一步 RMSE 比保持原状态改善约 99.10%。
- 12 条留出轨迹的 10 秒自主滚动角度 RMSE 合并后约为 0.034884 弧度。
- 范围外初态 [2.5, 0] 的角度 RMSE 约为 0.498919 弧度，泛化有明显边界。
- 同样 50 轮训练的五次中位数为 Rust 1.055673 秒、NumPy 2.469241 秒。
- 性能数字只代表该机器与实现，不是普遍语言速度结论，也未测量相对 RK4 的推理速度优势。
- RK4 在本实验中比神经网络更准确，报告如实保留这一结果。

## 常用文件

results/evaluation.json 是逐模型、逐初态的结果，rollouts.csv 是完整滚动轨迹。
history_main.csv 与 history_tiny.csv 是逐轮损失，benchmark_*.csv 保存每次计时，parity.json 保存跨语言数值差异。
summary.json 仅为这些原始结果的派生汇总。

data 和 benchmark_fixture 被 Git 忽略，可以由命令重建；完整本地包保留了它们。
不应把重跑产生的计时数值直接套进旧讲稿，除非同步更新报告和 PPT。

## Git 与验收

公开仓库为 xiaobo121388/project11-neural-ode。开发按功能分步提交，保留远程初始化历史，没有伪造成员作者或工作记录。
Windows CI 执行格式检查、单元/集成测试、无 GUI 核心测试、Clippy、Rustdoc、Release 构建和独立 NumPy 校验。
当前 GUI 套件为 13 项单元测试加 3 项集成测试。

最终互评分数按课程安排在实际评价时填写，本项目不提供虚构的评分或个人贡献。
