---
title: "闲话TPU #5 那些使用TPU训练的巨型模型(时间和算力需求)"
categories: "TPU Blog"
author: "Cy.Feng"
date: 2019-03-04
---

> [闲话TPU #1 背景/价格/TFRC计划及羊毛](http://cyfeng.science/tpu/blog/2019/03/07/chat-about-tpu-1.html)
>
> [闲话TPU #2 配置GCP环境/创建TPU实例](http://cyfeng.science/tpu/blog/2019/03/06/chat-about-tpu-2.html)
>
> [闲话TPU #3 模型编写](http://cyfeng.science/tpu/blog/2019/03/05/chat-about-tpu-3.html)
>
> [闲话TPU #4 Coral Edge TPU赋能移动端](http://cyfeng.science/tpu/blog/2019/03/04/chat-about-tpu-4.html)

## 那些使用TPU训练的巨型语言模型

## Six/Those Huge Neural Network Models, Training Time and Computing Resources

---

### BERT

原论文中描述，大型 BERT 模型在 16 个 Cloud TPU 上需要训练 4 天：

> Training of BERT_BASE was performed on 4 Cloud TPUs in Pod configuration (16 TPU chips total).13 Training of BERT_LARGE was performed on 16 Cloud TPUs (64 TPU chips total). Each pretraining took 4 days to complete.

现在我们来算一下成本，16 个 Cloud TPU v3 总训练成本为 16×8×24×4=12288 美元。有研究者在 Reddit 中回复作者，他们可以使用更便宜的抢占式（Preemptible）TPU 训练模型，那样成本约为 16×2.4×24×4=3686.4 美元。不过一般的 TPU 优先于抢占式 TPU，如果它们需要计算资源，可以暂停抢占式对资源的调用。

BERT 的作者在 Reddit 上也表示预训练的计算量非常大，Jacob 说：「OpenAI 的 Transformer 有 12 层、768 个隐藏单元，他们使用 8 块 P100 在 8 亿词量的数据集上训练 40 个 Epoch 需要一个月，而 BERT-Large 模型有 24 层、2014 个隐藏单元，它们在有 33 亿词量的数据集上需要训练 40 个 Epoch，因此在 8 块 P100 上可能需要 1 年？16 个 Cloud TPU 已经是非常大的计算力了。」

为了做对比，这里统一用一般的 TPU 价格计算成本，因此 BERT 训练一次大概需要 1.23 万美元。

### GPT-2

今年另一个非常受关注的语言模型就是 GPT-2 了，它充分展示了什么才算大模型。我们可以理解为，GPT-2就是在 GPT 的基础上放大十多倍，它需要的算力应该比 BERT 还大。堆了这么多算力与数据，GPT-2 的效果确实惊人，它根据一个前提就能从容地把故事编下去。

但是在 GPT-2 原论文中，我们没找到关于算力的描述，只找到了疑似论文作者的描述。他表明 GPT-2 用了 64 个 Cloud TPU v3，训练了一周多一点。

如果按这个数据，那么训练成本为 32×8×24×7=43008 美元，这个成本已经是训练 BERT 的 3 到 4 倍了。

### XLNet

2018 年，谷歌发布大规模预训练语言模型 BERT ，为 NLP 领域带来了极大的惊喜。但最近，Quoc V. Le 等研究者提出的 XLNet 在 20 个任务上超过了 BERT 的表现，并在 18 个任务上取得了当前最佳效果。既然效果这么好，那么它的成本会不会也超过 BERT？

在原论文中，作者表示 XLNet 大模型在 128 个 Cloud TPU v3 下需要训练 2 天半：

> We train XLNet-Large on 512 TPU v3 chips for 500K steps with an Adam optimizer, linear learning rate decay and a batch size of 2048, which takes about 2.5 days. 

这样算起来，128×8×24×2.5=61440 美元，没想到 XLNet 训练一次的费用比 GPT-2 还高，达到了 BERT 的 5 倍。既然成本这么高，以后可以考虑用预训练的 XLNet 代替 BERT 了。

在看了 XLNet 的算力成本之后，有开发者感叹：「谢天谢地我不在 NLP 领域工作，要是让我去说服老板训练一个模型花 6 万多美元，而且还不能保证这个模型一定好用，我觉得我会哭……」

### BigGAN

视觉模型中，常见高成本任务就是训练高分辨率的 GAN 了。在去年，研究者表示他们训练 512×512 像素的图像需要 64 个 Cloud TPU v3，训练 24 到 48 个小时：

> We train on a Google TPU v3 Pod, with the number of cores proportional to the resolution: 128 for 128×128, 256 for 256×256, and 512 for 512×512. Training takes between 24 and 48 hours for most models.

如果我们用最大训练时间 48 小时为基准，那么训练成本为 64×8×48=24576 美元。是的，BigGAN 的训练成本也比 BERT 高，大约是它的两倍左右。

### StyleGAN

最后，我们统计一下 StyleGAN 的训练成本，因为这篇论文是英伟达提出来的，所以用的是 Tesla V100。该论文使用的 FFHQ 数据集由 1024×1024 的人脸图像组成，模型使用 8 张 Tesla V100 需要训练一星期：

> Our training time is approximately one week on an NVIDIA DGX-1 with 8 Tesla V100 GPUs.

这里我们按照谷歌云的价格计算总成本，从而更好地做对比。总体而言，训练成本为 8×2.48×24×7=3333.12 美元。可能因为数据集仅限于人脸，StyleGAN 的成本要比 BigGAN 低很多。

---

参考链接：

https://www.reddit.com/r/MachineLearning/comments/c2pfgb/d_how_can_you_do_great_ai_research_when_you_dont/

https://www.reddit.com/r/MachineLearning/comments/c59ikz/r_it_costs_245000_to_train_the_xlnet_model512_tpu/

https://www.reddit.com/r/MachineLearning/comments/9nfqxz/r_bert_pretraining_of_deep_bidirectional/