---
title: "闲话TPU #1 背景/价格/TFRC计划及羊毛"
categories: "TPU Blog"
author: "Cy.Feng"
date: 2019-03-07
---

版权声明：本博文欢迎分享与转载，转载请注明出处和作者。<cy.z.feng@gmail.com>

> [闲话TPU #2 配置GCP环境/创建TPU实例](http://cyfeng.science/tpu/blog/2019/03/06/chat-about-tpu-2.html)
>
> [闲话TPU #3 模型编写](http://cyfeng.science/tpu/blog/2019/03/05/chat-about-tpu-3.html)
>
> [闲话TPU #4 Coral Edge TPU赋能移动端](http://cyfeng.science/tpu/blog/2019/03/04/chat-about-tpu-4.html)

## 零/为什么会出现这篇博客?

## Zero/Why is there this blog?

**[补充说明1: 这个并不是篇纯粹的Tech Blog, 更多的是一些经验而谈和碎碎念, 如何省钱搞大事儿, 如何更加便捷的使用工具, 有哪些可以不掉进去的坑, 期待pure tech的敬请移步版[官方英文Docs](https://cloud.google.com/tpu/docs/?hl=zh-cn)]**

**[补充说明2: Blog将持续更新相关信息/新鲜的坑/碎碎念, 也欢迎客官老爷随时提出意见建议和批评指正, Email: cy.z.feng@gmail.com, CyFeng16@GitHub]**

up是一名大学研究狗, 彼时所在实验室的GPU资源总是不够用(相信分家产的事情大家都或多或少的遇见过啦), 包括但不限于互相kill the task或遇见嫌疑人X独占整台机器的情况.

另一个很大的原因是, idea迭代的速度确确实实的限制了我们无意间天马行空的想象力的落地, 像是看到一篇新鲜出炉的paper的motivation or idea和你我曾经迸发的点子似曾相识, 这个时候的挫败感相比碌碌无为来的更为严重. 

本着~~[打不过就加入的原则]~~ 以更快速地工具验证自己idea的原则, 在中国好师兄*李卓换*助力下, 时至今日update这篇Blog时, 使用TPU时长已经高达6 months ^ ^.

嗯, Blog就到这里了, 完结撒花**:tada:** **:tada:** **:tada:** ~~[就会被满怀期待看Blog的人寄刀片的啊吧]~~

---

---

---

---

---

---

## 壹/简单介绍下TPU

## One/Brief introduction of TPU

TPU, 全称Tensor Processing Unit, 从名字看就是专门用来加速Tensor计算的, 何谓Tensor计算? 矩阵计算是也. 最近有幸受邀去PnP中一家某清的创业孵化器公司交流TPU的使用, 了解到NVIDIA家最新的2080Ti在HPC和Deep Learning中Performance up了大约10X的样子, 我想应该是受益于20系列GPU计算卡中特有的Tensor core计算单元, 然而…然而要知道N家的卡中Tensor core只有2个96×96(16-bit)的阵列呀~~[以后升级了不负责任的说]~~, TPU中的矩阵计算单元却有128X128, 这也说明了为什么TPU在合理优化的前提下, 模型运行速度会飞起来~~(内存加载的优化也至关重要, XLA编译下Fusion大大优化了内存访存.)

> ```markdown
> **PEP20**
> Beautiful is better than ugly.
> Explicit is better than implicit.
> Simple is better than complex.
> Complex is better than complicated.
> Flat is better than nested.
> Sparse is better than dense.
> Readability counts.
> Special cases aren't special enough to break the rules.
> Although practicality beats purity.
> Errors should never pass silently.
> Unless explicitly silenced.
> In the face of ambiguity, refuse the temptation to guess.
> There should be one-- and preferably only one --obvious way to do it.
> Although that way may not be obvious at first unless you're Dutch.
> Now is better than never.
> Although never is often better than *right* now.
> If the implementation is hard to explain, it's a bad idea.
> If the implementation is easy to explain, it may be a good idea.
> Namespaces are one honking great idea -- let's do more of those!
> ```

**节省时间即是珍惜生命, 不是么!!!** ~~[顺便防脱发(误)]~~

希望更多的国人researcher加入使用TPU的大家庭中, 也希望大G家在硬件方面更加open, V4什么的快点呀!!! (定价更平民一些就更好了, 毕竟都是要恰饭的嘛! :-D)

| Cloud TPU version | Support started   | Support ends           |
| ----------------- | ----------------- | ---------------------- |
| v3-8              | October 10, 2018  | (End date not yet set) |
| v2-8              | February 12, 2018 | (End date not yet set) |
| v2-32 *(alpha)*   | November 7, 2018  | (End date not yet set) |
| v2-128 *(alpha)*  | November 7, 2018  | (End date not yet set) |
| v2-256 *(alpha)*  | November 7, 2018  | (End date not yet set) |
| v2-512 *(alpha)*  | November 7, 2018  | (End date not yet set) |

当前(2019年03月31日11:41:06)面向公众开放的TPU分为两代, Cloud TPU v2 and Cloud TPU v3, 从实际运行的经验上比较, V3的速度大约是V2的1.8X左右(数据来源于each step的比较, 上下行和checkpoint存储为CPU的工作, TPU不背这个锅), 目前2代TPU拥有Pod的使用模式支持(可以想象成GPU的单机多卡并行), 最高支持256 个 TPU 芯片（16x16 切片）共512 cores, 性能美如画, 比肩**$399,000**的**DGX-2**, 每小时的price也是美如画 - -. 当前Google Cloud Platform上的TPU定价如下表(北美区域): 

| Cloud TPU v2  | 每小时每个 TPU $4.50。 |
| ------------- | ---------------------- |
| 抢占式 TPU v2 | 每小时每个 TPU $1.35。 |

| Cloud TPU v3  | 每小时每个 TPU $8.00。 |
| ------------- | ---------------------- |
| 抢占式 TPU v3 | 每小时每个 TPU $2.40。 |

| v2-32 Cloud TPU v2 Pod（Alpha 版）  | 每小时每个 Pod 切片 $24.00。  |
| ----------------------------------- | ----------------------------- |
| v2-128 Cloud TPU v2 Pod（Alpha 版） | 每小时每个 Pod 切片 $96.00。  |
| v2-256 Cloud TPU v2 Pod（Alpha 版） | 每小时每个 Pod 切片 $192.00。 |
| v2-512 Cloud TPU v2 Pod（Alpha 版） | 每小时每个 Pod 切片 $384.00。 |

- [抢占式 TPU](https://cloud.google.com/tpu/docs/preemptible?hl=zh-cn) 是指 Cloud TPU 在需要将资源分配给另一项任务时，可以随时终止（抢占）的 TPU。__抢占式 TPU 的费用要比普通 TPU 低廉得多__。

各个Area的价格大概是 北美<欧洲<亚太(也就酱, 没有太多的槽点).

**[划重点/划重点/划重点]可以薅的羊毛有大G家用来支持研究和教育事业的[TFRC计划](https://www.tensorflow.org/tfrc), 如果加入了 [TFRC 计划](https://www.tensorflow.org/tfrc/?hl=zh-cn)，则可在限定时间内免费使用 Cloud TPU v2 和 v3。只要 TPU 在 us-central1-f 地区内运行，就无需为 Cloud TPU 支付费用。**(调用TPU的GCE/VM的money还要自己出啦…呵)

根据身边被安利的同志们的reply而言, 通过给TFRC小组去Email, 平均回复(核可)时间是2 weeks, 一般情况下granted TPU数量为: 5XTPUv2/100X TPUv2(preemptible)/2XTPUv3酱紫, 限免使用时长为1 month. [相信聪颖智慧的各位一定会善加利用free的资源~~或者想到搞到更多free资源的method~~]

曾经算过笔账, 不是密集型生产实践任务, 很单纯的使用TPU进行一些模型的验证, 每天消耗的小钱钱up的钱包还是能承受起的啦(难免心痛).

**[划重点/划重点]可以薅的羊毛还有[Google Colab](https://colab.research.google.com/). Colaboratory是一个研究项目，可免费使用的 Jupyter 笔记本环境，不需要进行任何设置就可以使用，并且完全在云端运行, 目前提供三种免费的硬件支持: `CPU`/`GPU|K80`/`TPU|v2`.** 缺点是系统可能会停止长时间运行的后台计算, 你的计算也可能会因为浏览器自动节能等等原因断开连接.

好, 在下一篇文章中, 我们会进入下一个话题, ~~[母猪的产后护理]~~ 如何开始使用TPU.