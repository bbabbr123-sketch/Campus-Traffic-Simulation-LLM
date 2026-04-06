Campus Traffic Simulation & Dynamic Scheduling Engine
**基于 LLM 与物理势场的微观交通仿真与动态调度引擎**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)]()
[![Simulation](https://img.shields.io/badge/Simulation-Multi--Agent-success.svg)]()
[![LLM](https://img.shields.io/badge/LLM-DashScope-orange.svg)]()

> 本项目是一个从零构建的多智能体微观仿真系统，旨在解决校园共享单车时空供需不平衡问题。系统集成了大语言模型（LLM）进行宏观决策，并引入人工势场法与防死锁机制模拟微观人群交互。

🔗 **[点击此处在线体验仿真沙盘可视化大屏]**(此处替换为你的GitHub Pages网页链接)

---

## 🎯 核心特性 (Core Features)

### 1. LLM 调度中枢与沙盒评测闭环 (LLM & Sandbox Verification)
* **大模型接入**：将大语言模型作为宏观调度大脑，实现自然语言到系统底层调度指令的转化。
* **沙盒推演拦截**：独创指令校验闭环。在真实执行 LLM 指令前，系统会进行 1 小时的物理超前虚拟推演。成功计算时空缺口并拦截非法/冗余指令，彻底阻断 LLM 幻觉导致的“羊群效应”。

### 2. 微观状态预测与防死锁机制 (Microscopic Prediction & Anti-Deadlock)
* **全局人工势场 (APF)**：基于离散元胞空间建立引力场，模拟人群与目标建筑的复杂空间排斥交互。采用 BFS 算法实现势场降维，将寻路复杂度从 $O(V \log V)$ 优化至 $O(1)$。
* **防死锁概率决策**：针对多智能体高密度环境（人群/单车拥堵），设计基于 Softmax 的防死锁概率决策模型，替代传统贪心算法。
* **时空缺口预测**：集成多层感知机（MLP）与 One-hot 编码，对系统未来的时空供需缺口进行超前预测。综合优化后，系统全局“情绪均值”指标显著提升。

### 3. 全链路数据采集与可观测性 (Data Pipeline & Observability)
* **高频状态快照**：开发底层物理状态采集脚本，实现多维特征的动态本地数据集（JSON/CSV）持续写入。
* **可视化大屏开发**：基于动静分离的双图层 Canvas 架构，独立开发终端监控面板。支持高帧率动态热力图渲染、LLM 日志流输出及核心指标实时折线图监控。

---

## 📁 目录结构 (Structure)

```text
├── with shuttle.py         # 核心物理引擎、LLM API 接入与预测控制主程序
├── without shuttle_csv.py  # 无干预状态下的基础时空数据采集脚本（基线）
├── sim_data.json           # 仿真系统导出的高压缩比状态切片数据
├── index.html              # 纯前端 (HTML/Canvas/ECharts) 可视化监控大屏
└── historical_demand.csv   # 用于训练 MLP 神经网络的历史需求数据集
