---
title: "闲话TPU #4 Coral Edge TPU赋能移动端"
categories: "TPU Blog"
author: "Cy.Feng"
date: 2019-03-04
---

版权声明：本博文欢迎分享与转载，转载请注明出处和作者。<cy.z.feng@gmail.com>

> [闲话TPU #1 背景/价格/TFRC计划及羊毛](http://cyfeng.science/tpu/blog/2019/03/07/chat-about-tpu-1.html)
>
> [闲话TPU #2 配置GCP环境/创建TPU实例](http://cyfeng.science/tpu/blog/2019/03/06/chat-about-tpu-2.html)
>
> [闲话TPU #3 模型编写](http://cyfeng.science/tpu/blog/2019/03/05/chat-about-tpu-3.html)
>

## 伍/边缘计算场景中TPU硬件

## Five/Edge TPU and Other AI Accelerators

### 1. 为什么Edge TPU值得期待

AIY Edge TPU开发板在树莓派的基础上完成了AI进化，并且性能也要高于树莓派，可谓是青出于蓝而胜于蓝,而AIY Edge TPU加速器可以说是直接正面争锋Movidius NCS.

Edge TPU/Edge TPU Dev Board/Edge TPU Accelerator将共同促进边缘AI的蓬勃发展.



### 2. Edge TPU

![AI at the edge](../../../../../assets/2019/chat-about-tpu-4-1.png)

树莓派(Raspberry Pi)出现的时候我还在修习本科学位,最后一个学期尝试的是IoT方向.当时国内用来做IoT模拟实验的还是类似仿真电路模拟的那一个臃肿箱子,当时好友@宋恺睿 *songkairui*@hackret.com第一次拿出树莓派的时候,我:哇塞!!!没有想到那么小一个东西的如此的富有功能性.

> [Raspberry Pi](https://www.raspberrypi.org/)
>
> 我们的使命是将计算和数字化的力量交付给全世界人民。 我们这样做是为了让更多人能够利用计算和数字技术的力量开展工作，解决对他们而言至关重要的问题，并激发他们的创造性。



就像我没有想到在同样的大小(量级)上能赋予了边缘计算设备AI推理的能力的设备来的如此快.不同的是这次并没有那么惊讶罢了.虽然周边的朋友没有一些在用,在见识过Intel家的神经元计算棒Movidius Neural Compute Stick后一直期待着有那么一款能真正普及的IoT-AI辅助设备.

> [Movidius Neural Compute Stick](https://software.intel.com/en-us/movidius-ncs)
>
> 英特尔®Movidius™神经计算棒（NCS）是一款微型无风扇的深度学习硬件USB驱动器，用于学习AI编程。 NCS由相同的低功耗高性能Movidius™视觉处理单元（VPU）供电。 VPU可以在数百万智能安防摄像机，手势控制无人机，工业机器视觉设备等中找到。



![IoT Edge Module artwork](../../../../../assets/2019/chat-about-tpu-4-2.png)

> [Edge TPU](https://cloud.google.com/edge-tpu/)
>
> Google’s purpose-built ASIC designed to run inference at the edge.

Edge TPU可以在边缘部署高质量的ML推理。它增强了Google的云TPU和云物联网，以提供端到端（云端到边缘，硬件+软件）基础架构，以促进基于AI的解决方案的部署。除了开源TensorFlow Lite编程环境之外，Edge TPU最初将部署多个Google AI模型，结合了Google在AI和硬件方面的专业知识。

Edge TPU补充了CPU，GPU，FPGA和其他ASIC解决方案，以便在边缘运行AI，这将由Cloud IoT Edge提供支持。

|            | 边缘**<br/>**(设备/节点/网关/服务器) |                            谷歌云                            |
| :--------: | :----------------------------------: | :----------------------------------------------------------: |
| 可进行任务 |             机器学习推理             |                      机器学习训练和推理                      |
| 软件, 服务 |       Cloud IoT Edge, Linux OS       | Cloud ML Engine, Kubernetes Engine,**<br/>**Compute Engine, Cloud IoT Core |
|   ML框架   |       TensorFlow Lite, NN API        |           TensorFlow, scikit-learn,XGBoost, Keras            |
| 硬件加速器 |          Edge TPU, GPU, CPU          |                   Cloud TPU, GPU, and CPU                    |

Edge TPU的尺寸约为1美分硬币的1/8大小，它可以在较小的物理尺寸以及功耗范围内提供不错的性能（目前具体性能指标不清楚，官方称可以在高清分辨率的视频上以每秒30帧的速度，在每帧上同时执行多个最先进的AI模型），支持PCIe以及USB接口。

Edge TPU优势在于可以加速设备上的机器学习推理，或者也可以与Google Cloud配对以创建完整的云端到边缘机器学习堆栈。在任一配置中，Edge TPU通过直接在设备本地处理数据，这样不仅保护隐私，而且消除对持久网络连接的需要，减少延迟，允许使用更少的功率和性能。



### 3. AIY Edge TPU开发板

![](../../../../../assets/2019/chat-about-tpu-4-3.jpg)

[AIY Edge TPU开发板](https://aiyprojects.withgoogle.com/edge-tpu)是一款搭载了Edge TPU的单板计算机，功能非常丰富。开发板分为底板跟核心板，底板包括一些常用的外设接口，而核心板是基于Google Edge TPU的模块化系统子板（核心板与底板可以分离），也就是下图中带屏蔽罩的那个SOM（system-on-module ）。

__边缘TPU模块（SOM）规格__

| 中央处理器 | 恩智浦i.MX 8M SOC（quad Cortex-A53，Cortex-M4F）  |
| ---------- | ------------------------------------------------- |
| GPU        | 集成的GC7000 Lite图形处理器                       |
| ML加速器   | Google Edge TPU协处理器                           |
| 内存       | 1 GB LPDDR4                                       |
| 闪存       | 8 GB eMMC                                         |
| 无线       | Wi-Fi 2x2 MIMO（802.11b / g / n / ac 2.4 / 5GHz） |
|            | 蓝牙4.1                                           |
| 外形尺寸   | 40毫米x 48毫米                                    |

__底板规格__

| 闪存     | MicroSD插槽                                 |
| -------- | ------------------------------------------- |
| USB      | Type-C OTG                                  |
|          | Type-C电源                                  |
|          | Type-A 3.0主机                              |
|          | Micro-B串口控制台                           |
| LAN      | 千兆以太网端口                              |
| 音频     | 3.5mm音频插孔（符合CTIA标准）               |
|          | 数字PDM麦克风（x2）                         |
|          | 2.54mm 4针端子，用于立体声扬声器            |
| 视频     | HDMI 2.0a（全尺寸）                         |
|          | 用于MIPI-DSI显示器的39针FFC连接器（4通道）  |
|          | 用于MIPI-CSI2摄像机的24针FFC连接器（4通道） |
| GPIO     | 40针扩展头                                  |
| 功率     | 5V DC（USB Type-C）                         |
| 外形尺寸 | 85毫米x 56毫米                              |

__支持的操作系统__

Debian Linux

__支持的框架__

TensorFlow Lite



### 4. Edge TPU协处理器

![](../../../../../assets/2019/chat-about-tpu-4-4.jpg)

[AIY Edge TPU加速器](https://aiyprojects.withgoogle.com/edge-tpu)是一个基于Google Edge TPU的USB设备型的神经网络加速设备，通过USB Type-C可以连接到任何基于Linux系统的PC机、单板计算机如树莓派等的设备上去执行机器学习推理。

__产品规格__

| ML加速器 | Google Edge TPU协处理器   |
| -------- | ------------------------- |
| 连接器   | USB Type-C *（数据/电源） |
| 外形尺寸 | 65毫米x 30毫米            |

*仅与USB 2.0速度兼容Raspberry Pi主板。

__支持的操作系统__

Debian Linux

__支持的框架__

TensorFlow Lite



### 5. NVIDIA Jetson Nano($99!)

![](../../../../../assets/2019/chat-about-tpu-4-5.jpg)

__AI 新维度__

Jetson Nano 模组仅有 70 x 45 毫米，是体积非常小巧的 Jetson 设备。 为多个行业（从智慧城市到机器人）的边缘设备部署 AI 时，此生产就绪型模组系统 (SOM) 可以提供强大支持。

__较高计算性能__

Jetson Nano 提供 472 GFLOP，用于快速运行现代 AI 算法。 它可以并行运行多个神经网络，同时处理多个高分辨率传感器，非常适合入门级网络硬盘录像机 (NVR)、家用机器人以及具备全面分析功能的智能网关等应用。

__低功率需求__

Jetson Nano 为您节约时间和精力，助力您实现边缘创新。 体验功能强大且高效的 AI、计算机视觉和高性能计算，功耗仅为 5 至 10 瓦。

__技术规格__

| GPU      | NVIDIA Maxwell™ 架构，配备 128 个 NVIDIA CUDA® 核心 |
| -------- | --------------------------------------------------- |
| CPU      | 四核 ARM® Cortex®-A57 MPCore 处理器                 |
| 内存     | 4 GB 64 位 LPDDR4                                   |
| 存储空间 | 16 GB eMMC 5.1 闪存                                 |
| 视频编码 | 4K @ 30 (H.264/H.265)                               |
| 视频解码 | 4K @ 60 (H.264/H.265)                               |
| 摄像头   | 12 通道（3x4 或 4x2）MIPI CSI-2 DPHY 1.1 (1.5 Gbps) |
| 连接     | 千兆以太网                                          |
| 显示器   | HDMI 2.0 或 DP1.2 \| eDP 1.4 \| DSI (1 x2) 2 同步   |
| UPHY     | 1 x1/2/4 PCIE、1x USB 3.0、3x USB 2.0               |
| I/O      | 1x SDIO/2x SPI/6x I2C/2x I2S/GPIO                   |
| 尺寸     | 69.6 mm x 45 mm                                     |
| 规格尺寸 | 260 针边缘连接器                                    |



### 6.References

\[1\] [https://www.raspberrypi.org/](https://www.raspberrypi.org/)

\[2\] [https://software.intel.com/en-us/movidius-ncs](https://software.intel.com/en-us/movidius-ncs)

\[3\] [https://cloud.google.com/edge-tpu/](https://cloud.google.com/edge-tpu/)

\[4\] [https://aiyprojects.withgoogle.com/edge-tpu](https://aiyprojects.withgoogle.com/edge-tpu)

\[5\] [https://www.nvidia.cn/autonomous-machines/embedded-systems/jetson-nano/](https://www.nvidia.cn/autonomous-machines/embedded-systems/jetson-nano/)



## ToDo

- [ ] 更新信息
- [ ] 购买Edge TPU/Jetson Nano测试&&报告