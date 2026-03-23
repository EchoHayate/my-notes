# 深入浅出 Ring Attention：从 Ring All-Reduce 到极致的长上下文序列并行

经常看到在技术讨论群里，大家聊各种 Sequence Parallelism 和支持百万上下文的 Ring Attention 机制，其中包含许多底层的通信原语我偶尔也有盲区，让人产生了一些技术焦虑。前几天在面试的时候，发现自己对于 Ring 拓扑如何跟 Attention 计算真正结合也知之甚少。沿着这个契机，我决定结合手里的笔记和 DeepSpeed 源码，把 Ring All-Reduce 和 Ring Attention 的底层脉络重新梳理一遍。

本文的核心学习路线图如下：
1. 重温传统的 PS 架构之痛与 Ring All-Reduce 的物理拓扑。
2. 从 DeepSpeed 源码看 Ring 集合通信在工程上的落地优化。
3. 从梯度通信推导到注意力计算，引入 Ring Attention 的核心机制。
4. 剖析 Ring Attention 的计算与通信重叠设计。

---

## 1. 重构物理拓扑：从 Parameter Server 到 Ring All-Reduce

> 本节整理了我之前关于多卡通信底层拓扑的思考与笔记。

💥 **传统的宕机灾难：Parameter Server (参数服务器) 瓶颈**

在 Ring All-Reduce 诞生之前，大家搞多卡训练（比如 8 张 GPU 一起算梯度），用的是 Parameter Server（PS 架构）。
*   **运行逻辑**：设定一台中心服务器（PS），所有 GPU 把自己算好的梯度全部发给它。PS 把所有梯度加起来求平均，然后再把最新的模型参数分发回给所有的 GPU。
*   **致命缺陷**：这是一个典型的“星型拓扑”。当 GPU 数量急剧增加（比如扩展到 1000 张卡），中心节点的带宽瞬间被直接挤爆（DDoS 级拥堵）。增加的算力全浪费在排队等网络传输上了，扩展性极其拉垮。

🔄 **架构重构：Ring All-Reduce 的物理拓扑**

为了消灭这个单点瓶颈，百度（硅谷 AI 实验室）在 2017 年把 **Ring All-Reduce** 引入了深度学习领域。它的核心思想极其暴力且优雅：**消灭中心节点，让所有 GPU 连成一个逻辑上的“闭环（Ring）”，每个节点只和自己左右两边的邻居单线联系！**

它将整个数据同步过程拆分成了两个完美的阶段（假设有 $N$ 张 GPU，模型数据被切分成 $N$ 个数据块）：

### 1.1 Phase 1：Scatter-Reduce（打散并规约）

在这个阶段，我们的目标是让每张 GPU 最终只包含“某一个数据块”的完整累加结果。
所有卡同时将自己手里的某一块数据，传给右边的下一个节点。接收到邻居传来的数据后，和自己本地对应的数据块进行相加（Reduce）。这个传递和相加的动作在环里循环进行。
经过 $N-1$ 步的轮转后，奇迹出现了：**每张卡上都持有了其中一个数据块的最终完整和**（比如 GPU 0 持有块 A 的总和，GPU 1 持有块 B 的总和）。

### 1.2 Phase 2：All-Gather（全收集广播）

现在每张卡都有了一块终极碎片，接下来的任务是把这些终极碎片同步给所有人。
同样是在环里，每张卡把自己拥有的“完整数据块”传给右边的节点。接收到邻居的完整块后，直接覆盖自己本地对应的旧数据（因为不需要再相加了，直接复制）。
再次经过 $N-1$ 步的轮转，所有的终极碎片传遍全网。最终，每一张 GPU 都获取到了完整的、经过全量累加的模型梯度！

🚀 **为什么它是 Infra 架构的神作？**

这套协议最伟大的地方，在于它的**网络吞吐量复杂度是 $O(1)$ 的**。无论您是在用 8 张卡，还是 8000 张卡组成这个环，每个节点在同步过程中传输的总数据量恒定约为：
$$2 \times \frac{N-1}{N} \times S$$
（其中 $S$ 是模型总参数大小）。当节点数 $N$ 极大时，这个值无限趋近于 $2S$。网络带宽被完美均摊，彻底消灭了单点阻塞。所有的 GPU 都在同时发送和接收数据，把每一根网线（NVLink 或 PCIe）的带宽全部榨干，没有任何节点在闲置摸鱼。

---

## 2. 工程落地：DeepSpeed 是如何做 Reduce-Scatter 的？

有了这些理论基础，我们要看在真实生产系统中这是怎么代码落地的。以 ZeRO 为核心的 DeepSpeed 框架，在分割优化器状态和梯度时（ZeRO Stage 1 & 2），大量依赖了上面提到的 Reduce-Scatter 操作。

> 源码跟踪：`deepspeedai/DeepSpeed`
> 具体文件：`https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/comm/coalesced_collectives.py`

在 DeepSpeed 的实际代码中，如果对每个被切段的梯度 Parameter 都单独发起以此集合通信算子，极高的 kernel 启动延迟会导致效率坍塌。因此，DeepSpeed 引入了 **`reduce_scatter_coalesced` (合并规约分发)** 技术来摊销这些开销并吃满带宽。

```python
@instrument_w_nvtx
@torch.no_grad()
def reduce_scatter_coalesced(
    tensors: List[Tensor],
    group: ProcessGroup = None,
) -> List[Tensor]:
    """同时对一组 tensor 列表执行 reduce-scatter。这比对每个 tensor 单独调用更高效。"""
    this_rank = dist.get_rank(group)
    world_sz = dist.get_world_size(group)

    partition_lst_for_each_tensor = [None] * len(tensors)
    for tensor_idx, tensor in enumerate(tensors):
        flattened_tensor = tensor.view(-1)
        # 1. 显存物理平铺分块：根据参与通信的 GPU 数量把张量切块
        chunk_sz = math.ceil(tensor.numel() / world_sz)
        partition_lst_for_each_tensor[tensor_idx] = [
            flattened_tensor[rank * chunk_sz:rank * chunk_sz + chunk_sz] for rank in range(0, world_sz)
        ]

    # ... 省略部分 padding 对齐逻辑 ...
    
    # 2. Coalesce (平铺合并) 的艺术
    tensor_partition_flat_buffer = instrument_w_nvtx(torch.cat)(tensor_partitions_lst_with_padding)
    tensor_partition_flat_buffer.div_(world_sz)  # 在 reduce 前预做 avg_pool 级别的标量除法

    tensor_partition_buffer_for_each_rank: List[Tensor] = torch.chunk(tensor_partition_flat_buffer, world_sz)

    # 3. 发起由于合并而变得极其巨大的单次底层 reduce-scatter通信
    _torch_reduce_scatter_fn(tensor_partition_flat_buffer,
                             tensor_partition_buffer_for_each_rank[this_rank],
                             group=group)

    # 最后按此卡负责的分块大小进行逆向解析返回
    # ...
```

**代码逐步分析：**
- **逻辑分块**：`chunk_sz = math.ceil(tensor.numel() / world_sz)` 正是在物理层面上将数据块以 World size 作为粒度做了平均切分，精准对应了前文我们所说的“切分成 $N$ 个数据块”。
- **强力拼装**：它没有写个 for 循环遍历调用底层的 `reduce_scatter`，而是用 `torch.cat` 把需要通信的内容强行拼成了一个巨大的 `tensor_partition_flat_buffer`。因为对于 Ring 拓扑而言，单次传输一块 1GB 的连通内存块，其吞吐表现远远超过你分 1000 次传输 1MB。
- **底层交付**：`_torch_reduce_scatter_fn` 调用最终将其打入 NCCL 等底层的通信后端，后端会在硬件上严格执行刚才说的 Phase 1 环节。当它返回时，当前 `this_rank` 手里就已经紧紧握住了全部卡在这部分参数上累加完毕的结果。

---

## 3. 从模型并行走向序列并行：Ring Attention 的推演

上一章讲解了如何利用环形网络传递模型的梯度碎片，沿着这个思路推导下去，我们来到了如今这篇笔记最核心的文章**驱动问题**：

> 既然梯度的数组分片可以在 GPU 之间通过 Ring 高效传递，**那为什么不能把 Transformer 注意力机制中的 KV 缓存 (KV Cache) 也切片通过 Ring 环形传递呢？**

众所周知，《Attention is All You Need》虽然极其强大，但它的核心 Self-Attention 具有关于输入长度的具有 $O(L^2)$ 空间和时间复杂度。假设我们当前有一个上下文大小 $L=1000K$ 的文本（比如一本厚重的小说或者代码库），单张 GPU 的显存直接被暴涨的全局 KV Cache 瞬间打穿。

这就是 **Ring Attention** 诞生的绝佳舞台。

### 3.1 核心定义与运行机制

**Ring Attention 的核心思想是：将超长输入序列（Sequence Length）进行等长分块切片，每张 GPU 一开始只处理原文本序列的 $1/N$（这种方式被称为 Sequence Parallelism 序列并行），以此消灭单卡的显存壁垒。随后，让算好的 KV 分块在各个卡构成的逻辑闭环中顺时针“流动”。**

让我们加上**数值化的具体例子**来感知一下。假设某超长输入的 Sequence Length 是 `1024K`，系统配置了 4 张 GPU 连成了一个环：
- GPU_0 被分配前 `256K` 并计算其对应的局部 $Q_0, K_0, V_0$。
- GPU_1 拿到第二个 `256K` 块产生 $Q_1, K_1, V_1$ ... 依此类推。

如果按照传统算力，GPU_0 为了算出正确的全局 $Attention_0$，它不仅仅必须要 $K_0, V_0$，还需要全网的 $K_{1..3}$ 与 $V_{1..3}$。这显然不现实。于是：

1. **阶段一（自激）**：GPU_0 拿着自己本地的 $Q_0$，去乘目前拥有的 $K_0, V_0$，计算出一个局部的 Block Attention 值并缓存统计量（借助类似 FlashAttention 在线 rescaled Softmax 技术）。
2. **阶段二（起步环流）**：GPU_0 算完后，直接将自身的 $(K_0, V_0)$ 一包通过 NVLink 发送给右手边的 GPU_1；于此同时，它的左手边 GPU_3 正把自身的 $(K_3, V_3)$ 传给它。
3. **阶段三（交融更新）**：GPU_0 成功接收了新的 $(K_3, V_3)$。它立马使用最初的那个 $Q_0$ ，继续乘以上这批新的键值对。得到的输出在本地按照数学等价的方式与之前阶段一的局部结果累加重缩放。
4. **阶段四（流动继续）**：计算完毕后，GPU_0 像击鼓传花一样，把自己手头刚用完的 $(K_3, V_3)$ 又传递给 GPU_1，并从左边接手传来的 $(K_2, V_2)$。以此类推周而复始。

经过完整的 $N$ 次传递打转后，每一个 GPU 都仿佛“不出门便知天下事”，在没有占用 $N$ 倍显存的前提下，间接地让自己的局部 $Q$ 和全网所有的 $KV$ 发生了交互！

### 3.2 掩盖开销的戏法 (Overlap)

这里存在一个工程优缺点上非常核心的设计平衡：一直在等网络把别人的 KV 数据包传回来，GPU 核里的 Tensor Core 岂不是在干等闲置？

这就是这套架构真正优美之所在：它的计算和通信是**完全正交且可以重叠 (Overlap) 的**。由于我们切分的块大小完全固定，底层通过 Pipeline 我们可以同时做两件事：
*   **计算 (Compute)**：执行当前局部块的 Attention 内核乘加。
*   **通信 (Comm)**：挂在后台起异步任务执行 P2P (`isend/irecv`) 预抓取下一个需要的 KV Block。

只要能通过 Block Size 调优，使得：**硬件计算单个 Block Attention 的耗时 $\geq$ KV Block 通过网络传输给邻居的耗时**，那么这极巨量的网络传输时间实际上就被“隐藏（Hidden）”在了计算时间之中，最终达到了几乎无额外开销的理想上限。

---

## 4. 总结展望

总的来说，传统 **Parameter Server 的坍塌**教育了我们多节点星型拓扑在超大规模面前的脆弱，倒逼出了网络带宽复杂度恒定为 $O(1)$ 的 **Ring All-Reduce**。由于它优雅的切分、平摊通信逻辑极具拓展性的大成之美，使得我们在多年后面临长上下文（Long-Context）的 Attention 内存洪流大考时，能再次回想起了这门环形接力手艺。

将序列并行（Sequence Parallelism）与 Ring 拓扑结合，便造就了解决现代大语言模型长文本训练推理的终极利器——**Ring Attention**。知易行难，看似只是换了个维度画圈传数，但在 DeepSpeed 和底层 Kernel 层面要做到极致的内存分配与 P2P 异步流的通信重叠，仍处处是工程底座较量的心血。这也正是我们不断阅读源码、打磨技术深度的乐趣所在。
