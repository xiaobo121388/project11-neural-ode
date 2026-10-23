# Team 11 English Presentation with Chinese Reference

Main presentation: 7 minutes. Live demo: 3 minutes. Slides 11 and 12 are Q&A appendices.

## Slide 1 Neural Network from Scratch for ODE Solving

Target 35 seconds.

### English

Hello, we are Team 11. Our project asks whether a small neural network, implemented from first principles in Rust, can learn the motion of a pendulum. We wrote the dense layers, activations, loss, gradients and training loop ourselves. The main result is a large reduction in one-step prediction error compared with predicting no movement. However, we also show accumulated error and an unfamiliar initial condition, because a good local fit alone does not establish a reliable dynamical model.

### 中文对照

大家好，我们是第 11 组。项目研究用 Rust 从零实现的小型神经网络能否学习单摆运动。全连接层、激活函数、损失、梯度和训练循环均为手写。一步预测比保持原状态明显更好，但我们也会展示长期累积误差和训练范围外的失败。

## Slide 2 Physical System

Target 40 seconds.

### English

We use a damped nonlinear pendulum with angle and angular velocity as its two state coordinates. Gravity is 9.81, length is one metre, and damping is 0.1 per second. The network predicts the state after 0.02 seconds. To generate labels, our fourth-order Runge-Kutta implementation takes ten smaller steps of 0.002 seconds. We call this the numerical reference rather than an exact truth. Equilibrium, convergence, the small-angle analytical limit and decreasing damped energy are checked independently.

### 中文对照

状态包括角度和角速度。重力加速度为 9.81，摆长为 1 米，阻尼为每秒 0.1。网络预测 0.02 秒后的状态；标签使用手写 RK4，每次走 10 个 0.002 秒小步。它是高精度数值参考解，而非精确解析真值。另有平衡、收敛、小角度和能量测试。

## Slide 3 Dataset Design

Target 45 seconds.

### English

A fixed seed generates eighty initial conditions. Each produces five hundred transitions over ten seconds, giving forty thousand samples. We split by complete trajectory: fifty-six for training, twelve for validation and twelve for final testing. Randomly splitting individual rows would leak almost neighbouring states between partitions. Input and target means and standard deviations are therefore fitted on training trajectories only. The output target is the next-state increment. The stored checkpoint includes every normalization statistic needed to convert its predictions back to physical units.

### 中文对照

固定种子生成 80 个初态，每条轨迹 10 秒、500 次状态转移，总共 4 万个样本。按整条轨迹划分 56 条训练、12 条验证、12 条测试，避免相邻状态泄漏。标准化统计量仅来自训练集。网络输出状态增量，模型文件保留所有还原物理单位所需的统计量。

## Slide 4 Handwritten Neural Network

Target 40 seconds.

### English

The main architecture has two inputs, two hidden layers with thirty-two Tanh neurons each, and two linear outputs. It contains 1,218 trainable parameters. We also retain a deliberately small two-three-two Sigmoid baseline with seventeen parameters, following the assignment's suggested starting point. ReLU is implemented and tested as well. ndarray performs matrix arithmetic only. There is no automatic differentiation or high-level training framework. At inference, the learned increment is decoded and added to the network's current state, without calling the pendulum equation.

### 中文对照

主网络结构为 2-32-32-2，隐藏层使用 Tanh，输出层为线性，共 1,218 个参数。另保留作业建议的小型 2-3-2 Sigmoid 基线，只有 17 个参数。ReLU 也已实现并测试。ndarray 只做矩阵运算；推理阶段不调用单摆方程进行修正。

## Slide 5 Backpropagation

Target 45 seconds.

### English

Our loss averages squared error over both samples and output coordinates. That makes its output derivative two times the residual divided by batch size times output width. We propagate this derivative through activation functions and transposed weight matrices. Weight gradients are the previous activations transposed times the local derivative; bias gradients sum over samples. We compute all gradients before changing any weights. Central differences check every parameter of a small network, and an independent NumPy implementation checks forward values, gradients and one update.

### 中文对照

MSE 同时对样本数和输出维度取平均，因此输出导数含 2 除以批量大小和输出宽度。反向传播使用激活函数导数与转置权重；权重梯度为前层激活转置乘局部导数，偏置梯度按样本求和。所有梯度算完后再更新参数，并用有限差分和独立 NumPy 实现交叉验证。

## Slide 6 Training and Validation

Target 40 seconds.

### English

Both models use mini-batch gradient descent with batch size 128, learning rate 0.01 and two hundred epochs. The checkpoint with the lowest validation loss is retained. The test partition never enters this selection. This plot shows the base-ten logarithm of validation MSE, so changes across several orders of magnitude remain visible. The larger model achieves a much lower validation error. We did not use the proposed fallback hyperparameter sweep, because the default configuration already passed the predefined acceptance condition.

### 中文对照

两种模型都使用批量 128、学习率 0.01、训练 200 轮的小批量梯度下降，只按验证损失选检查点，不让测试集参与选择。图中纵轴是验证 MSE 的十进制对数。默认配置已达到预设验收标准，因此没有启动备用调参搜索。

## Slide 7 One Step Results

Target 40 seconds.

### English

The frozen main model is evaluated on six thousand held-out transitions. Its angle RMSE is about 0.000274 radians and velocity RMSE about 0.000765 radians per second. Standardized error improves by 99.10 percent over persistence, exceeding our fifty-percent target. The main network also improves on Euler for this experiment. Nevertheless, RK4 is much more accurate, because it directly uses the known physical equation. These are local predictions from the real current state, so we must still test an autonomous rollout.

### 中文对照

冻结后的主模型在 6,000 个留出转移上测试，角度 RMSE 约 0.000274 弧度，角速度约 0.000765 弧度每秒。标准化误差相对保持原状态改善 99.10%，也优于本次实验中的 Euler；但 RK4 明显更准确。一步测试每次输入真实当前状态，所以仍须检查自主连续预测。

## Slide 8 Autonomous Rollouts

Target 45 seconds.

### English

In these plots the network receives the same initial state as the reference, then uses only its own previous output for five hundred steps. There is no teacher forcing or hidden reset. The left plot shows an unseen test trajectory within the initial-condition range. The right starts at 2.5 radians, outside that range. Its larger phase and amplitude discrepancy is a genuine failure of generalization. The GUI preserves this error and labels the unfamiliar initial state rather than correcting the model using reference physics.

### 中文对照

网络和参考解从同一初态出发，此后 500 步都只递推自己的输出，没有教师强制或隐藏重置。左图是训练初态范围内的留出轨迹，右图从范围外的 2.5 弧度开始，明显出现更大的相位和振幅偏差。界面如实保留误差并显示范围警告。

## Slide 9 Rust and Python Benchmark

Target 50 seconds.

### English

For the timing comparison, Rust and NumPy share the same float64 dataset, initial parameters and every batch permutation for fifty epochs. Each process warms up once, then runs five timed trials. Startup, data loading and validation are excluded. Median times are 1.056 seconds for Rust and 2.469 seconds for NumPy on this machine. Numerical parity agrees near floating-point roundoff. We report backend differences explicitly, including NumPy's OpenBLAS implementation, so this result is specific to the workload and does not prove a universal language-speed advantage.

### 中文对照

计时比较共享双精度数据、初始参数和全部批次顺序，统一训练 50 轮，各预热一次、实测五次，排除启动、读文件和验证。该机器上的中位数为 Rust 1.056 秒、NumPy 2.469 秒，数值差异接近浮点舍入误差。由于数值库后端不同，不能推广为普遍语言性能结论。

## Slide 10 Live Demonstration

Target 40 seconds.

### English

We will now demonstrate the offline desktop lab. We will begin with a held-out initial state, pause and step through the motion, and inspect the time and phase plots. Next we will switch to the smaller network and compare its error. Finally, the high-angle preset shows where generalization deteriorates. Our conclusion is that a handwritten Rust network can learn a useful local transition map, while classical numerical solvers and explicit long-horizon tests remain essential. The next three minutes show these comparisons directly.

### 中文对照

接下来演示离线桌面程序：先展示留出初态，暂停、单步并查看时间曲线和相图，再切换小网络比较误差，最后使用高角度预设展示泛化局限。结论是手写 Rust 网络能学到有用的局部转移映射，但长期检验和经典数值方法仍不可替代。之后安排三分钟实机操作。

## Slide 11 Appendix A Gradient Details

Q&A only.

### English

Use this appendix for questions about shapes and loss scaling. Samples are rows. The factor 2/(B D) normalizes all scalar outputs. ReLU uses derivative zero at zero and is checked away from the kink. Both weights and biases pass central differences with epsilon 1e-6 and absolute tolerance 1e-7.

### 中文对照

附录用于回答矩阵维度与损失缩放问题。样本按行排列，导数包含 2/(B D)。ReLU 在零点采用导数 0，有限差分避开不可导点；epsilon 为 1e-6，绝对误差容差为 1e-7。

## Slide 12 Appendix B Reproducibility and Limits

Q&A only.

### English

Use this appendix for reproduction and scope questions. The model is a fixed-step discrete state-transition surrogate, not a PINN or a continuous learned vector field. Physical parameters are fixed. One seed and one OOD example do not establish broad generalization. No inference-speed advantage over RK4 was measured.

### 中文对照

附录用于回答复现和适用范围问题。模型是固定步长的离散状态转移代理，不是 PINN，也不是连续学习向量场。物理参数固定；单个种子与单个范围外案例不能证明广泛泛化；没有测量相对 RK4 的推理速度优势。
