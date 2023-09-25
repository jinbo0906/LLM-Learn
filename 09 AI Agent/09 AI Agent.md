# 09 AI Agent

## 目录

-   [深入AI Agent架构](#深入AI-Agent架构)
    -   [1.Planning：任务规划与决策](#1Planning任务规划与决策)
    -   [2.Memory：记忆能力](#2Memory记忆能力)
    -   [3.Tools：工具使用](#3Tools工具使用)
-   [架构实现](#架构实现)

**在大模型时代，AI Agent的通用定义是：通过大模型驱动，能够自主的理解、规划、执行，并最终完成任务的智能程序。**

AI Agent与大模型的区别与关系可以概括为：

-   大模型只是一个大脑，而AI Agent是一个完整体。 &#x20;
-   大模型通常不会使用工具，而AI Agent则有使用工具的能力。
-   大模型只会告诉你怎么做，而AI Agent会直接帮你做。
-   AI Agent会与外部环境交互，并借助大模型来决策行动。
-   AI Agent通过大模型来驱动，大大提升了自身的理解、规划与决策能力。

## 深入AI Agent架构

OpenAI应用研究主管LilianWeng把AI Agent总结为：

> 📌**Agent = LLM + 记忆 + 规划技能 + 工具使用**

**简单的说，AI Agent就是在LLM作为智慧的“大脑”的基础上通过增加Memory，Planning，Tools三大能力，从而构建一个具有自主认知与行动能力的完全“智能体”。**

![](https://mmbiz.qpic.cn/sz_mmbiz_png/wT3y8Vg9pFGAicsmBeGlqGgyNdiaLEY6MGRW8CBmwRGNgzrb37tJjFrpeYZf9U9t4EEWYfvUk0R6X2libOVTJZAicQ/640?wx_fmt=png\&wxfrom=5\&wx_lazy=1\&wx_co=1)

### 1.Planning：任务规划与决策

**借助于LLM将大型任务分解出多个子目标与小任务，并设定与调整优先级。这一部分主要涉及到任务分解与自我反思。**

**【任务分解】借助LLM自身算法优化以及提示工程来将复杂任务分解成多个小型的、简单任务。这里面经常使用的一种提示词技术是**思维链（Chain-of-thought)和思维树（Tree-of-thought），也就是通过提示LLM“**一步一步的思考**”，来把复杂任务拆分成多个步骤的、小任务树来完成。

**【自我反思】** 能够在任务执行过程中，不断完善自身的任务决策，甚至纠正以前的错误来不断迭代改进。

这种自我纠正其实我们在OpenAI的chatGPT的code interpreter的使用中可以看到类似能力：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/wT3y8Vg9pFHQiamSbNre9Odo89Haz82xfEricjsuR9bwvbWG5pDI0na5PwQVWQBraHas6fDPlLVkibBGf0SLsJotQ/640?wx_fmt=png\&wxfrom=5\&wx_lazy=1\&wx_co=1)

在AI Agent执行任务的过程中，“自我反思”更多的体现在任务过程中，**依据外部环境的交互结果（比如搜索引擎的反馈）来进行思考与下一步任务决策**。最常提到的一种范式是ReAct：

**Resoning and Acting**，推理与行动。这种方式的主要思想是：**通过把行动（Act，通常使用工具）获取的外部知识，反馈给LLM帮助推理（Reason）并做出下一步的行动决策**。

以下是一个典型的ReAct范式下LLM推理并完成一个任务的过程：

![](https://mmbiz.qpic.cn/sz_mmbiz_png/wT3y8Vg9pFHQiamSbNre9Odo89Haz82xfdDSsibQNnbtr7mqgwqpUwiaVSuPJ25JnjbIgNiafibSSyQ51QPfnfgiaEdA/640?wx_fmt=png\&wxfrom=5\&wx_lazy=1\&wx_co=1)

可以看到，一个符合ReAct的行动流程里，包含多次Thought-Act-Obs的迭代，也就是思考-行动-观察/分析的反复过程。

-   **Thought（思考）：** 反映了LLM大模型的思考过程，这体现出了LLM的“大脑“的作用。LLM根据输入或者外部环境反馈生成本次迭代需要完成的行动。比如：
    > “苹果遥控器最初用来设计控制Front Row媒体中心应用，那么我下面需要去搜索下Front Row这个应用...“
-   **Act（行动）：** 根据思考的结果采取的具体行动，这个具体的行动最终体现到某个外部工具的使用，实现与外部环境的交互并获得结果。比如：
    > “使用Google搜索引擎API接口搜索Front Row“
-   **Obs（观察）：** 从外部行动获取到的反馈信息，用于LLM做出下一步的行动决策。比如：
    > “无法搜索到Front Row，近似的信息包括Front Row (Software)...”

通过这样的循环过程，最终完成任务。

### 2.Memory：记忆能力

当前的LLM自身是没有记忆能力的（我们目前看到的大模型上下文能力也是通过会话历史的重新发送而实现）。AI Agent作为一个具备自主完成任务能力的智能体，需要对LLM补充的记忆能力主要包括：

-   短期记忆：一次任务过程中的上下文记忆。比如在任务过程中与大模型LLM的对话历史，会受到大模型窗口大小的限制，比如16K。
-   长期记忆：存储在向量数据库中可随时检索访问的外部数据，用来补充增强大模型自身的训练知识。这种知识通常需要在任务执行过程中通过向量相似算法来进行检索，并交给LLM作为参考。这为AI Agent提供了长期保留和调用无限信息的能力。

### 3.Tools：工具使用

工具的使用是人类一个最显著的特征，也是AI Agent在LLM基础上实现的最重要能力。借助于工具的使用，相当于给LLM安装上了四肢，可以显著的扩展LLM模型的功能。比如：

-   调用其他的AI模型，比如其他的专有任务模型 &#x20;
-   网络搜索引擎，比如Google搜索、Bing搜索
-   常见的开放API，比如天气查询、航班查询 &#x20;
-   企业信息获取，比如产品信息、CRM客户信息 &#x20;

AI Agent的工具使用能力的核心问题不在于工具本身的构建，我们认为需要关注另外两个问题：

-   **如何让LLM正确的做出使用工具的决策，即应该在什么时候使用什么工具？** 我们知道LLM的唯一输入是提示词，因此需要给予LLM足够的工具使用提示似乎是唯一的办法，而正确性则有赖于LLM自身的推理能力。
-   **如何构建正确的工具使用输入信息？** 如果你需要使用搜索引擎这个工具，那么你的搜索关键词是什么？显然，这需要LLM结合自身知识、上下文、外部环境进行推理。再比如，你需要LLM访问你的企业系统，而你的系统输入要求是严谨的JSON格式，那么如何让LLM能够从自然语言推理出符合规范的JSON结构呢？

## 架构实现

[https://mp.weixin.qq.com/s/WU1947wpuHONWNSaC3t02w](https://mp.weixin.qq.com/s/WU1947wpuHONWNSaC3t02w "https://mp.weixin.qq.com/s/WU1947wpuHONWNSaC3t02w")
