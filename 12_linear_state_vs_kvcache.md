# Linear attention 的 state vs. full attention 的 KV cache:prefix caching 到底要花多少存储?

**问题。** 大家选 linear attention 是冲着省内存去的:recurrent state 相对序列长度是 O(1),
而 full attention 的 KV cache 是 O(L)。那么在 **prefix caching** 这一层——也就是 LMCache
所在的这一层——这个优势还在吗?为了干净地回答,我们固定同一个参考模型的维度,对比两个
**同构**变体:整个模型全用 linear attention,vs. 整个模型全用 full attention。

**结论速览。** 在 vLLM 为 hybrid 模型实际选定的 block size 下,一个 linear 层的缓存成本
**和一个 full attention 层完全相等**——因为 vLLM 把 block 撑到"**一份 state 快照 ≈ 一个
attention block 的 KV**"。于是全 linear 模型和全 full 模型的 per-token 缓存足迹**一模一样**。
Linear attention 的省内存优势在缓存层**彻底消失**。更糟的是,它**无法**做细粒度缓存——存储
量会随粒度以 `1/C` 爆炸,而 full attention 的成本与粒度无关。**压缩 state 是唯一能把这个优势
拿回来的杠杆。**

---

## 1. 参考模型与假设

维度取自 **Qwen3-Next-80B-A3B**(`config.json`),仅作为一组具体的真实数字;文末公式可推广到
任意模型。

| | 值 |
|---|---|
| 层数 | 48 |
| dtype | bf16(2 B/元素) |
| **linear 层(Gated DeltaNet)** | `num_value_heads=32`、`d_k=d_v=128`、`conv_kernel_dim=4` |
| **full attn 层(Gated Attention)** | `num_kv_heads=2`(GQA)、`head_dim=256` |

真实部署机制(已对照 LMCache v0.5.1 + vLLM v0.25.1 源码核实):

- **State 在 block 边界打快照,不是逐 token。** 在 `mamba_cache_mode="align"` 模式下,vLLM
  每个 scheduler block 才给 recurrent state 打一次快照
  (`lmcache_mp_connector.py:validate_mamba_step_alignment`)。checkpoint 周期 `C` = block size。
- **vLLM 会放大 block size,让 attention page 和 mamba page 对齐**
  (`kv_cache_group_edits.py`,`_SubpagedAttentionViewEdit`):"one page = one recurrent
  state snapshot"。代码里的实例是 **Qwen3.5-0.8B 用 544 token**——**不是 16**。
- 要复用某段前缀,就必须在那个边界上有快照,所以一段长度为 `L` 的缓存序列会持有 `L/C` 份
  **独立**快照——而每一份快照都是**完整的累积 state**(大小固定,与所在位置无关)。

---

## 2. 单层基本量

**每层 linear state**(一份快照,与序列长度无关,固定):

```
state/层 = num_value_heads × d_k × d_v × dtype
         = 32 × 128 × 128 × 2 B  =  1,048,576 B  =  1.00 MiB
```

(conv state = `(conv_kernel_dim−1) × 通道数 ≈` 几十 KiB,相对 1 MiB 可忽略。)

**每层 full attention 每 token 的 KV:**

```
KV/token/层 = 2(K,V) × num_kv_heads × head_dim × dtype
            = 2 × 2 × 256 × 2 B  =  2,048 B  =  2.00 KiB
```

整模型:

- **全 linear:** 一份完整快照 = `48 × 1 MiB` = **48 MiB**(固定,与长度无关)。
- **全 full:** 每 token KV = `48 × 2 KiB` = **96 KiB/token**(随长度增长)。

---

## 3. Block size 恰好被设在 break-even 点上

vLLM 选 block size 的原则是:让一个 attention block 的 KV 字节数等于一份 state 的字节数
(这样 page 才能统一)。这个值恰好是:

```
C*  =  state/层  ÷  (KV/token/层)  =  1 MiB / 2 KiB  =  512 token
```

(和代码里 0.8B 模型的 544 是同一量级——原理相同,数字随模型维度缩放。)

**`C*` 就是 break-even 周期。** 当 `C = C*` 时,存 linear state 的 per-token 成本 =
存 full attention KV 的 per-token 成本——单层如此,所以整个同构模型也如此。

---

## 4. 核心对比 —— 总缓存足迹

一段长度为 `L` 的缓存序列,每 `C` 个 token 打一次 state 快照:

```
全 full   总量  =  L × 96 KiB                    (与 C 无关)
全 linear 总量  =  (L / C) × 48 MiB  =  L × (48 MiB / C)
```

两者都对 `L` 线性,所以**比值与序列长度无关**:

```
全 linear / 全 full  =  (48 MiB / C) / 96 KiB  =  C* / C  =  512 / C
```

| checkpoint 周期 `C` | 换来什么 | linear ÷ full |
|---|---|---|
| 2048 token | 只有很粗的命中 | **0.25×** |
| **512 token(vLLM 默认,= C\*)** | block 对齐的命中 | **1.0×(持平)** |
| 128 token | 更细的部分前缀命中 | **4×** |
| 64 token | | **8×** |
| 16 token(full-attn 的缓存粒度) | 任意前缀命中 | **32×** |
| —— 只存最终态(每文档 1 份快照) | 只能整文档复用 | **≈0.03×(见 §6)** |

**怎么读这张表:** 在真实部署用的设置下(`C = C* = 512`),全 linear 模型和全 full 模型缓存
成本**相同**——linear attention 没省下任何东西。一旦往 full attention 免费享有的细粒度推进,
linear 的存储就以 `512/C` 爆炸。

---

## 5. 真正的不对称:粒度成本

最关键的不是某一行数字,而是两条曲线对 `C` 的响应方式:

- **Full attention:** 总量 = `L × 96 KiB`,**与 `C` 无关。** 每 token 的 KV 只存一份,且可在
  任意边界寻址。16-token 粒度和 512-token 粒度成本一样——细粒度前缀命中**免费**。
- **Linear attention:** 总量 = `L × (48 MiB / C)`,**`∝ 1/C`。** 每个更细的 checkpoint 都会把
  **整段**前缀历史重存一遍(state 是累积的、不可切分),所以粒度越细,存储成倍膨胀。

这就是 state 必须压缩的根本原因:**linear attention 买不起 attention 天生免费的细粒度前缀
复用。** vLLM 今天是靠把 block 撑到 `C*=512` 来掩盖这一点,而这同时把 **attention** 层也拖到
512-token 的缓存粒度上(正常 attention 本可以在 16 就缓存),属于连带损伤。

---

## 6. 具体存储场景

语料:**512 篇文档 × 16 K token** = 840 万个缓存 token(典型的长上下文 / RAG)。

| 配置 | 总缓存足迹 |
|---|---|
| 全 full attention | **768 GiB** |
| 全 linear,`C = 512`(vLLM 默认) | **768 GiB**(持平) |
| 全 linear,`C = 128`(更细命中) | **3.0 TiB** |
| 全 linear,`C = 16`(与 attention 同粒度) | **24 TiB** |
| 全 linear,只存最终态(不支持部分复用) | **24 GiB** |

`24 GiB` 那一行是"linear attention 的梦想":每篇文档只存一份 48 MiB 的 state,比 full
attention **便宜 32×**——**但**它只允许把文档当作**整段前缀**复用。一旦你想从任意位置续上
(拼接的多篇文档、部分共享的前缀),就需要很多份快照,优势立刻塌回持平(`C=512`)甚至更差。

**这个 idea 什么时候才有意义:** 恰恰在复用是**部分 / 任意前缀**、而非整文档的时候。而这正是
prefix caching 的一般情形,也正是压缩能把 768 GiB–24 TiB 这一列拉回 24 GiB 梦想的地方。

---

## 7. 压缩:那个杠杆

把 state 压缩 `r` 倍,`state/层` 降到 `1 MiB / r`,于是:

1. **break-even block 降到** `C* = 512 / r`。当 `r = 8` 时 `C* = 64`:在**相同存储**下可以
   多打 8× 的 checkpoint —— 前缀缓存粒度细 8×,部分共享前缀的命中率更高。
2. **给 attention block "松绑"。** state 变小后,vLLM 不必再把 attention page 撑到 512,
   attention 层也重新获得细粒度缓存。
3. **终于带来存储收益:** 在固定粒度下,全 linear 模型变得比 full attention 便宜 `r×`——这正是
   用户当初期待 linear attention 带来的省内存优势,只不过兑现在缓存层,而不再只是 decode 时的
   HBM 里。

可行性:delta-rule / GLA 的 state 通常低秩或对量化容忍度高,所以 4–10× 无损压缩是有希望的——
这正是待研究的开放问题。

---

## 附录 —— 通用公式

单层,linear 头数 `H_v`、key/value 维 `d_k,d_v`;attention KV 头数 `H_kv`、头维 `d`;
每元素 `b` 字节;checkpoint 周期 `C`(token):

```
state/层         = H_v · d_k · d_v · b
KV/token/层      = 2 · H_kv · d · b
break-even C*    = state/层 ÷ KV/token/层 = (H_v · d_k · d_v) / (2 · H_kv · d)
足迹比值         = (全 linear)/(全 full) = C* / C          (与长度无关)
```

方头特例(`d_k=d_v=d`、`H_v=H_kv`):`C* = d/2`。当 `d=128` 时 `C*=64`——在去掉 GQA 和头数差异后,
每个 head 上出现的正是同一个 break-even。
