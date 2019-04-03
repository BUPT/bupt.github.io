---
layout: post
title: "闲话TPU #2 配置GCP环境/创建TPU实例"
categories: "TPU Blog"
author: "Cy.Feng"
date: 2019-03-06
---

版权声明：本博文欢迎分享与转载，转载请注明出处和作者。<cy.z.feng@gmail.com>

> [闲话TPU #1 背景/价格/TFRC计划及羊毛](http://cyfeng.science/tpu/blog/2019/03/07/Chat-about-TPU-1.html)
>
> [闲话TPU #3 模型编写](http://cyfeng.science/tpu/blog/2019/03/05/Chat-about-TPU-3.html)
>
> [闲话TPU #4 Coral Edge TPU赋能移动端](http://cyfeng.science/tpu/blog/2019/03/04/Chat-about-TPU-4.html)

## 贰/配置GCP和创建TPU实例

## Two/Configure Google Cloud Platform and Create a TPU Instance

首先推荐一下几个官方网址, 在这之上我们可以找到(近乎)bug free的教程(有些坑我们之后再聊): 

`Cloud TPU 主页`: https://cloud.google.com/tpu/?hl=zh-cn

`Quickstart`: https://cloud.google.com/tpu/docs/quickstart?hl=zh-cn

`TensorFlow`: https://www.tensorflow.org/

> PyTorch: 大G一些官方和非官方人员共同合作为PyTorch-1.0后的版本开发了可以将PyTorch语言编译为TPU硬件可识别的XLA编译器, 说实话, 应用效果一般且坑还是很多的, 对于支持的标准模型/用法, 计算效率到时没差(某B的非会员降速会员原速呵), 且现在TPU-XLA的支持还没有merge到PyTorch的官方repo中, 持观望态度.

`TPU Available TensorFlow Ops`: https://cloud.google.com/tpu/docs/tensorflow-ops

> 枚举了所有TPU所支持的Ops, 当然实现一个操作的不同Ops之间还是存在实际速度差异的, 太详细的可以自己去tpu-profile(TensorBoard inside)中查看“操作时间”的分页.

`TPU Tools`: https://cloud.google.com/tpu/docs/cloud-tpu-tools

> 主要是[cloud_tpu_profiler](https://cloud.google.com/tpu/docs/cloud-tpu-tools#install_cloud_tpu_profiler)这个pip包, cloud_tpu_profiler将TPU profiler可视化的集成到TensorBoard中, 对于我们进行bottleneck分析十分有帮助.

### 1. 创建一个GCP账号

创建一个GCP账号(推荐用gmail账号直接创建), 创建一个GCP项目.

### 2. 开启GCP账号的结算功能

__修改项目的结算设置__

如果您只是一个结算帐号的结算管理员，则您创建的新项目会自动关联到您现有的结算帐号。如果您创建多个结算帐号且有权访问这些帐号，则可以更改项目的结算帐号。本文介绍了如何更改项目的结算帐号，以及如何启用和停用项目的结算功能。

如果您希望通过电子邮件接收帐单或对帐单，或者您想要更改接收帐单或对帐单的人员，请参阅[更改结算联系人和通知](https://cloud.google.com/billing/docs/how-to/modify-contacts)。

__更改项目的结算帐号__

要更改现有项目的结算帐号，**您必须是该项目的所有者，并且是目标结算帐号的结算管理员**。如需了解有关结算管理员和结算权限的信息，请参阅[访问权限控制概览](https://cloud.google.com/billing/docs/how-to/billing-access)。

要更改结算帐号，请执行以下操作：

1. 转到 [Google Cloud Platform Console](https://console.cloud.google.com/)。
2. 打开 Console 左侧菜单，然后选择**结算**。
3. 如果您有多个结算帐号，系统会提示您选择**转至关联的结算帐号**以管理当前项目的结算。
4. 在**与此结算帐号相关联的项目**下，找到要更改结算帐号的项目的名称，然后点击该名称旁边的菜单。
5. 选择**更改结算帐号**，然后选择所需的目标结算帐号。
6. 点击**设置帐号**。

尚未在交易记录中记录的已经产生的费用将被计入原来的结算帐号。这可包括项目移动之前最多 2 天内的费用。

__为项目启用结算功能__

启用结算功能的方式取决于您是创建新项目还是为现有项目重新启用结算功能。

__为新项目启用结算功能__

当您创建新项目时，系统会提示您选择要将哪个结算帐号关联到项目。如果您只有一个结算帐号，该帐号会自动关联到您的项目。

如果您没有结算帐号，则必须先创建一个并为项目启用结算功能，然后才能使用各项 Google Cloud Platform 功能。要创建新的结算帐号并为项目启用结算功能，请按照[创建新结算帐号](https://cloud.google.com/billing/docs/how-to/manage-billing-account)中的说明操作。

__为现有项目启用结算功能__

如果您有暂时停用了结算功能的项目，则可以按照以下步骤重新启用结算功能：

1. 转到 [Google Cloud Platform Console](https://console.cloud.google.com/)。
2. 从项目列表中，选择要为其重新启用结算功能的项目。
3. 打开 Console 左侧菜单，然后选择**结算**。
4. 点击**关联结算帐号**。
5. 选择结算帐号，然后点击**设置帐号**。

__停用项目的结算功能__

要停止为某个项目自动付款，您需要停用该项目的结算功能。请注意，即使您停用了结算功能，也仍需负责结算帐号中的所有未结费用，我们会通过您列出的付款方式扣除相应费用。

要停用项目的结算功能，请执行以下操作：

1. 转到 [Google Cloud Platform Console](https://console.cloud.google.com/)。
2. 打开左侧菜单，然后选择**结算**。
3. 如果您有多个结算帐号，请选择**转至关联的结算帐号**以管理当前项目的结算。要查找不同的结算帐号，请选择**管理结算帐号**。
4. 在**与此结算帐号相关联的项目**下，找到要停用其结算功能的项目的名称，然后从旁边的菜单中选择**停用结算功能**。系统会提示您确认是否要停用此帐号的结算功能。
5. 点击**停用结算功能**。

### 3. 开启TPU API

如果是新创建的GCP账号, 或者是第一次使用TPU, 需要开启TPU service的API, 在Google Cloud Platform界面下左侧边栏中选择Compute Engine菜单, 进而选择其中的TPU子菜单, 现在我们可以在页面中央区域开间“Enable TPU API”的button, click it and wait for minutes.

>  __(很多第一次尝试TPU的朋友反应这个API的开启速度实在是慢的没边er了, 没错就是这么慢, 您且等会er)__

### 4. 设置资源(Google Cloud Storage)

GCS(Google Cloud Storage)是默认也是唯一TPU使用过程中I/O的容器, 我们必须创建一个`Cloud Storage Bucket`用来盛放我们的data以及results(checkpoints).

1. 转到GCP控制台上的“云存储/存储/Storage”页面。

    [转到云存储页面](https://console.cloud.google.com/storage/browser?hl=zh-cn)

2. 创建一个新存储桶，指定以下选项：

    - 唯一名称。

    - 默认存储类： `Regional`

    - > 存储类别中第一类是在多个分区都可以使用的, 但是如非必要不应选择, 原因是我们进行科研/业务本就是单一区域的任务, 选择第一类存储是`加价不加量`的行为….三类和四类适合静态存储, 尤其是四类, 如果仅仅是想贮藏/掩埋的数据, 丢这里就对了(而且很廉价的说).

    - 位置：如果要使用Cloud TPU设备，请接受默认设置。如果要使用Cloud TPU Pod片，则必须指定Cloud TPU Pod 可用的[区域](https://cloud.google.com/tpu/docs/regions?hl=zh-cn)。

> __`其中几个土亢`__:
>
> 1. 我们使用的GCE(VM虚拟机)/Cloud TPU/GCS三者的位置必须一致, 其中GCE和Cloud TPU的位置是精确到zone级别的, GCS使用的时候只需要选择相应的area就可以. e.g. TPU@us-central1-f, GCE@us-central1-f, GCS@us-central1.
>
> 2. 新用户创建了当前地区的VM之后经常会发现仍然不能读取/写入到相应的bucket之中, 这是因为配置VM的时候没有授予VM GCS I/O的权限. 正确配置方式如下: 
>
>     在`新建虚拟机实例`中拖到最下方`身份和 API 访问权限`选择`允许所有Cloud API的全面访问权限`.

### 5. 创建VM

TPU unit实际上是一个整体的instance, 其中包括了`XLA CPU` `XLA GPU` 8*`Tensor Core` , 所以我们创建的VM在某种意义上单纯的是一个switch(控制开关), 用来给TPU unit传递“何时干活, 干什么活, 怎么干这个活”的控制类消息. 

~~[省钱小窍门]~~我们创建的VM大小一般来说1~2CPU`n1-standard-1` `n1-standard-2`即可, ~~[极限省钱的情况下]~~甚至是抢占式的共享VM也可以, 但是有一个前提条件是, 我们的模型必然在VM instance上init, 所以VM必须要满足model所需的内存占用量. 

科研过程中自己写的那种比较特殊的模型, 尤其是没有或者优化失败的模型, 请务必使用`n1-highmen-1`等高内存系列, 不过话说回来, 没有完整优化的模型, 即是在VM上成功init了也并不意味着这个模型会成功的在TPU上运行成功, 毕竟TPU也是有自身的HBM(on-chip memory)的限制的.

> 8 GB of on-chip memory (HBM) is associated with each Tensor Core for Cloud TPU v2; 16 GB for Cloud TPU v3.
>
> 即: 2代TPU有64GB可用显存(姑且这么说吧, 顺嘴), 3代TPU拥有128GB可用显存.

如果对TPU的软件架构和硬件架构有深入了解的兴趣, 请移步参考 https://cloud.google.com/tpu/docs/system-architecture . 其中会介绍XLA编译, 还有为什么TPU会仅接受GCS作为I/O对象. ~~[还不是为了把用户群体绑定在GCP平台上…]~~

### 6. 创建TPU

在`Compute Engine`中选择`TPU`-`创建TPU节点`, 进入`创建Cloud TPU`界面.

> 名称: 随你喜欢,都听你的
>
> 地区: 要注意选择你所需要的地区, 保持和GCE/GCS地区相同. 参加TFRC计划的记得选中`us-central1-f`才是免费的池子. zone-f的quote会在TFRC计划时间内生效, 其余时间zone-f不会出现配额, in another word, 不用担心会因为TFRC计划超时而产生额外的费用, 安心用~
>
> TPU类型: 选择V2 or V3, 单个instance or Pod(看您钱包鼓不鼓咯, 爷请上座~)
>
> **TensorFlow版本**: 这个是比较重要的选项, 每个被创建的TPU都是被严格规定了版本的, 其实是XLA编译的选择, 一些本地版本不匹配的情况可能会提示warning, TPU可能会直接撂挑子不干的哦~ 所以务必使用和本地编译bug-free的相同版本笨笨.
>
> 网络: default
>
> IP地址范围: 按照*CIDR*书写IP地址保证区域内可用IP地址数量≥8即可(8 cores for each TPU unit)
>
> 隐藏部分可用选择添加Tag(???), 还未曾见过instance多到需要tag区分, `抢占式TPU节点`在这里选择.

现在我们有了resource(GCS), 有了switch(GCE), 有了hardware(TPU)下一步就是如何通过software-level控制/告诉TPU如何运行了.

## ToDo

- [ ] 介绍一下CTPU工具的使用
- [ ] 介绍一下gcloud命令行工具
- [ ] 介绍一下Colab上TPU的使用