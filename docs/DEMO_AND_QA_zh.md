# 项目 11 现场演示与答辩准备

## 三分钟演示

演示不需要现场训练或联网。提前解压 WindowsDemo，双击 pendulum-lab.exe。
源代码、模型和原始数据以仓库中的实际内容为准；不要把训练时间对比当成推理速度对比。

| 时间 | 操作 | 英文讲解重点 |
| --- | --- | --- |
| 0:00–0:40 | 选择 Main 模型及 Held-out 68，播放后暂停 | Both simulations start from the same unseen initial state. Blue is the RK4 reference; orange is the neural prediction. |
| 0:40–1:10 | 使用单步按钮、拖动时间轴，指出时间曲线及相图 | The network recursively uses its own prediction. The displayed error is not hidden by resetting to the reference. |
| 1:10–1:45 | 切换 Tiny 模型，保持相同初态并观察 10 秒 RMSE | The seventeen-parameter baseline has visibly larger accumulated error than the main model. |
| 1:45–2:35 | 切回 Main，再选 Outside training range，播放或拖到后半段 | The unfamiliar initial angle causes larger phase and amplitude errors. This is a generalization limit, not a visualization bug. |
| 2:35–3:00 | 恢复 Held-out 68，结束演示 | A low one-step loss does not guarantee long-horizon accuracy. Our source, data and tests make the distinction reproducible. |

播放按钮是三角形，暂停是双竖线，重置是回转箭头，单步是带竖线的前进箭头。悬停可查看名称。
空格键也可播放或暂停。到达 10 秒会停止，再按播放会从头开始。

## 常见问答

### 1. 这算从零实现吗

English: Yes. Dense layers, activation functions, MSE, analytical backpropagation and SGD are handwritten in Rust. ndarray supplies matrix operations only. No high-level ML framework or automatic differentiation is used.

中文：是。网络、梯度和训练循环都是手写；ndarray 仅做线性代数，符合原文允许使用 ndarray 的提示。

### 2. 为什么使用 Tanh，而不只用 Sigmoid 和 ReLU

English: We implement and test Sigmoid and ReLU as required. Tanh is an additional smooth activation for the main model. The small baseline uses Sigmoid. Because the models differ in both capacity and activation, this experiment does not isolate the effect of the activation function.

中文：要求中的 Sigmoid、ReLU 均已实现并测试，Tanh 是为平滑动力学增加的选项。主模型和小模型同时改变了容量与激活函数，所以不能据此断言 Tanh 普遍优于 Sigmoid。

### 3. 反向传播最容易错在哪里

English: Samples are rows, so dW is the transpose of previous activations times the local derivative. MSE averages over B times D scalar outputs, so its derivative contains 2/(B D). Bias gradients sum over batch rows. All gradients are computed before any parameter update.

中文：注意样本按行排列、权重矩阵方向和 MSE 中输出维度 D 的因子，偏置梯度应按批量求和，不能一边传播一边使用更新后的权重。

### 4. 怎么证明梯度正确

English: Central differences check every weight and bias of a small network for all supported activations. Epsilon is 1e-6 and absolute tolerance is 1e-7. Independent NumPy code also matches forward values, gradients and a full update within 1e-10.

中文：既有逐参数中央差分检查，也有独立 NumPy 实现的交叉校验。ReLU 的检查避开零点处的不可导位置。

### 5. 为什么不随机划分所有样本行

English: Adjacent rows from the same trajectory are highly related. A row-level split could leak almost neighbouring states across training and evaluation. We assign complete trajectories to disjoint partitions and fit normalization only on training trajectories.

中文：随机拆行会让同一轨迹上的相邻状态出现在训练与测试两侧，导致结果虚高，因此按整条轨迹划分。

### 6. 数值参考解是否等于精确真值

English: No. It is a fine-step RK4 approximation. We verify convergence, equilibrium, damped energy behaviour and the small-angle analytical limit. We explicitly distinguish this reference from an exact analytical solution.

中文：不等于。它是经过数值验证的高精度近似解，报告中明确称为 numerical reference。

### 7. 为什么一步误差很小，连续预测仍会偏离

English: In a rollout, the input distribution changes because each predicted state becomes the next input. Small local errors can accumulate into phase and amplitude errors. One-step testing uses the real current state and therefore measures a different property.

中文：连续预测会把前一步误差带入后续输入，相位和振幅逐渐偏移，不能把一步精度直接等同于长期精度。

### 8. 是否比 RK4 更快或更准确

English: Single-step RK4 is substantially more accurate in this experiment. We did not measure an inference-speed advantage over RK4. The timing study compares Rust and NumPy implementations of the same training workload, not neural inference against RK4 integration.

中文：本实验中 RK4 更准确，也没有测量神经网络相对 RK4 的推理速度优势。不要混淆两种不同的性能比较。

### 9. Rust/Python 比较公平吗

English: The data, initial parameters, precision, batch order, batch size and learning rate are identical. Both processes warm up and perform five timed runs under a single-thread configuration. We exclude startup, loading and validation. However, the numerical backends differ and background load is not isolated, so the conclusion is workload-specific.

中文：控制了主要训练输入和流程，并公开后端差异及原始耗时；不能把一次机器上的结果推广成普遍语言速度结论。

### 10. 为什么保留那么小的网络

English: The 2-3-2 baseline follows the suggested starting point and has only 17 parameters. It establishes an understandable low-capacity comparison. The main 2-32-32-2 model has 1,218 parameters and learns this dataset more accurately.

中文：小模型对应原文起步建议，可解释性强，便于展示网络容量不足时的误差，不是伪装成主模型。

### 11. 模型是否适用于不同摆长或时间步长

English: Not without retraining or extending the input representation and dataset. The checkpoint stores fixed physical parameters and a fixed prediction interval. Initial-state controls do not retrain the model or change its governing parameters.

中文：当前只针对固定参数和固定步长。界面调整的是初态，不是改变摆长、重力或时间步长后仍保证泛化。

### 12. 如果文件缺失怎么办

English: The default desktop executable contains its trained checkpoints, so it needs no external model files. An explicitly requested custom checkpoint is validated and produces a visible error if missing or incompatible. The program never substitutes RK4 for a failed neural model.

中文：默认模型内置，不依赖旁边的 JSON 文件；只有显式加载自定义模型时才需要外部文件。出错不会悄悄换成参考解。

## 上台前检查

1. 在演示电脑离线打开 WindowsDemo 中的程序，确认两只摆、图表和播放按钮正常。
2. 排练 7 分钟讲解与 3 分钟演示，附录只在提问时展示。
3. 核对成员名单和截止日期。
4. 熟悉一步预测、滚动预测、范围外测试和训练时间对比之间的区别。
