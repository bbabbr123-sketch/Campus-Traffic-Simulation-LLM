# 🚴‍♂️ Campus Traffic Simulation & Dynamic Scheduling Engine
**基于 LLM 与物理势场的微观交通仿真与动态调度引擎**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)]()
[![Simulation](https://img.shields.io/badge/Simulation-Multi--Agent-success.svg)]()
[![LLM](https://img.shields.io/badge/LLM-DashScope-orange.svg)]()

> 本项目是一个从零构建的多智能体微观仿真系统，解决校园共享单车时空供需不平衡问题。系统集成大语言模型（LLM）进行宏观决策，并引入人工势场法与防死锁机制模拟微观人群交互。



---

## 📺 系统运行演示 (Demo)

*(![Image](https://github.com/user-attachments/assets/85d8eeab-dd5e-4b70-92b0-8e6f9f802b3f))*

---

## 🎯 核心特性 (Core Features)

### 1. LLM 调度中枢与沙盒评测闭环
* **大模型接入**：将 LLM 作为宏观调度大脑，实现自然语言到系统底层调度指令的转化。
* **沙盒推演拦截**：在执行 LLM 指令前进行 1 小时物理超前虚拟推演，计算时空缺口并拦截非法指令，阻断 LLM 幻觉导致的“羊群效应”。

### 2. 微观状态预测与防死锁机制
* **全局人工势场 (APF)**：基于离散元胞空间建立引力场，模拟多智能体空间排斥。采用 BFS 算法实现势场降维，寻路复杂度降至 $O(1)$。
* **防死锁概率决策**：针对高密度拥堵环境，设计基于 Softmax 的防死锁概率决策模型。
* **时空缺口预测**：集成多层感知机（MLP）对时空供需缺口进行超前预测，系统全局情绪指标显著提升。

---

## 📁 目录结构 (Structure)

```text
├── with shuttle.py         # 核心物理引擎、LLM API 接入与预测控制主程序
├── without shuttle_csv.py  # 无干预状态下的基础时空数据采集脚本
├── without_contrast.py     # 无调度的拥堵情况基线对比程序
└── historical_demand.csv   # 用于训练 MLP 神经网络的历史需求数据集
