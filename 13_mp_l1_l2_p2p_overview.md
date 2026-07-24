# MP 模式代码总览（lmcache server 视角）：L1 → L2 → P2P 三层递进

**范围与版本。** 只讲 **MP 模式**（`LMCacheMPConnector`），不涉及老的 in-process 路径
（`cache_engine.py` / `storage_backend/` / `cache_controller/`，我们的部署永远不会执行）。
所有 `file:line` 均指 **LMCache v0.5.1**（`onbording/src/lmcache` 的 editable clone，可直接
打断点）。配套调试环境见 `src/vllm-dbg.sh` / `src/req.sh`。

**视角。** 本文以 **lmcache server 进程为中心** —— 那是 PyCharm 单步的对象，也是所有数据
搬运真正发生的地方。vLLM 被当成一个**黑盒客户端**：它只会在固定时机发 5 种消息（第 0 章的
契约表），除此之外我们不关心它内部怎么运转（需要时看附录 A 的一分钟版）。

**读法。** 三层递进：先看纯 L1 时 server 对每种消息做什么（全部"前台"逻辑）；再加 L2
（前台不变，多两个后台搬运线程）；最后加 P2P（连后台机制都不变，只是多了几个"长得像 L2
的 peer"和一个控制面）。

---

## 第 0 章：客户端契约 —— vLLM 会发来什么

vLLM 通过 `--kv-transfer-config` 加载 LMCache 的 connector 插件，插件经 ZMQ
（`mq.py`，msgspec 编码）连到 server。对 server 而言，客户端的全部行为就是下面 5 种消息：

| 消息 | 时机 | 请求带什么 | 响应回什么 |
|---|---|---|---|
| `REGISTER_KV_CACHE` | 启动时一次 | 每层 KV 张量的 **CUDA-IPC handle** + layout（shape/dtype、group 信息）+ instance_id/model/world_size | 空 ack（同步等） |
| `LOOKUP` | 每个新请求到达时，恰好一次 | **完整 token ids**（截断到 chunk 对齐）+ request_id + cache_salt + tp_size | 空 ack，仅表示"prefetch 任务已登记"（瞬间即回）；命中结果不在这里 |
| `QUERY_PREFETCH_STATUS` | 每个调度周期轮询一次，直到命中的 KV **全部就位于本机 L1** | request_id | `None`（KV 还没搬齐到 L1，下周期再问）或命中 chunk 数（× chunk_size = 命中 token 数）。**返回非 None 的不变式：所有命中 KV 已在本机 L1 且被读锁锁住**——不管它原来在 L1、L2 还是 peer 那里；此后 RETRIEVE 只做 L1→GPU 最后一跳，必然成功 |
| `RETRIEVE` | 命中确认、GPU blocks 分配好之后，至多一次 | request_id + token ids 及范围 [start,end) + **gpu_block_ids（目的地址）** + skip_first_n_tokens + 客户端 event handle | `(CUDA event 的 IPC handle, bool)`——响应到达仅代表拷贝已 enqueue，**event 完成才代表拷贝做完**；发完不等，每周期 query |
| `STORE` | 每攒满新的完整 chunk 一次（一个请求多次） | request_id + token ids 及范围 [start,end) + **gpu_block_ids（源地址）** + 客户端 event handle | 同上；bool=false 只被 log（缓存少存一段，客户端无感） |

（RETRIEVE/STORE 的消息里也带 token ids，是因为 key 是自包含的：server 端用 session 缓存
lookup 时算过的链式哈希，按 [start,end) 直接复用，不必重算。）

三条契约性质的事实，后文反复用到：

1. **数据从不走消息。** 启动注册时 server 拿到 vLLM GPU KV pages 的 CUDA-IPC 映射，之后
   所有 GPU↔CPU 拷贝都由 **server 在自己的 CUDA stream 上直接读写对方显存**。消息里只有
   request_id、block id 列表、event handle 这类元数据。**这就是 profiling 必须抓 server
   进程的根本原因** —— 客户端侧只有 submit 开销。
2. **客户端从不阻塞。** LOOKUP/RETRIEVE/STORE 都是发完即走；"没好"的结果客户端用
   自己的调度循环消化（该请求先不跑就是了）。所以 server 端任何一步慢，表现都是客户端
   的轮询多转几圈，而不是谁卡住。
3. **完成信号是双条件的。** RETRIEVE/STORE 的 ZMQ 响应只代表"拷贝已全部 *enqueue* 到
   server 的 stream"；响应里带一个 CUDA event 的 IPC handle，event 完成才代表拷贝真正
   做完。客户端把两者绑成一个 future（`futures.py:79` `CUDAMessagingFuture`：:115
   `from_ipc_handle`，:137 `synchronize`），每周期 `query()` 一次。
   反方向也有一个 event：STORE 消息里带着客户端的 event，server **先等它**再动手 D2H
   （保证 KV 已经写进 GPU pages，`lmcache_driven_transfer.py:1008-1011`）；RETRIEVE 的
   协议里同样带，但 v0.5.1 的 retrieve 没有用它（目标 blocks 是新分配的，无在途写）。

---

## 第一层：纯 L1 —— server 对每种消息做什么

server 进程的骨架：`MPCacheServer`（`v1/multiprocess/server.py`，`run_cache_server`
是主入口）跑一个 MQ 主循环，把每种消息分发给注册了 handler 的 **EngineModule**。
L1 场景涉及三个 module：Management（杂务）、Lookup、LMCacheDrivenTransfer。存储侧是
`StorageManager`（L1/L2 统一入口）+ `L1Manager`（pinned DRAM 本体）。

### REGISTER_KV_CACHE → 拿到 GPU 显存的钥匙

`LMCacheDrivenTransferModule.register_kv_cache`（`lmcache_driven_transfer.py:816`）：
打开消息里的 CUDA-IPC handle，建 `GPUCacheContext` —— vLLM GPU KV pages 在 server
地址空间里的直接映射，连同 layout（每层 shape/dtype、chunk 对应多少 block）注册进
`layout_desc_registry`。每个 GPU 实例（instance_id = 客户端 pid）一份。

同期客户端还会同步问一次 chunk size（`modules/management.py:163`）；chunk size 必须
整除 vLLM block size，且 server 重启即换代（L1 随进程死）。

### LOOKUP → 查索引 + 加锁（不搬任何数据）

`LookupModule.lookup`（`lookup.py:206`）：

1. **算哈希**（:262，`token_hasher.py:192`）：chunk 粒度的**链式前缀哈希**
   `hash_i = H(hash_{i-1}, chunk_i 的 tokens)`，默认 blake3。第 i 个 chunk 的哈希编码了
   从头到 i 的整个前缀 —— 一个 token 不同，其后所有哈希全变。不满一 chunk 的尾巴直接
   丢弃（永不缓存）。哈希只在 server 端算；session 会缓存结果，后续 STORE/RETRIEVE 复用
   （`engine_context.py:247` `resolve_obj_keys`）。
2. **拼 key**：`ObjectKey = (chunk_hash, model_name, kv_rank, object_group_id,
   cache_salt)`（`v1/distributed/api.py:57`）—— 匹配的完整语义是"同模型、同 TP rank、
   同 KV 组、同用户盐、且前缀逐 token 一致"。
3. **提交 prefetch**：`storage_manager.submit_prefetch_task`（`storage_manager.py:399`）。
   纯 L1 时 :457 的注释就是全部真相：
   > "now we only have L1, so the prefetch is essentially checking how many objects are
   > already in L1, and adding read locks to them."

   **匹配就是 dict 精确查找**（L1 索引是 `dict[ObjectKey, L1ObjectState]`，
   `l1_manager.py:192`），按 chunk 顺序扫、**第一个 miss 就 break**（:509-519）——
   链式哈希保证"chunk i 命中 ⟹ 前 i+1 个 chunk 逐 token 一致"，所以数 leading hits
   就够了。命中的 chunk 全部加**读锁**（`reserve_read`，防止 lookup 与 retrieve 之间
   被驱逐）。

### QUERY_PREFETCH_STATUS → 报命中数（exactly-once）

`lookup.py:369`：prefetch 没完成回 None（客户端下周期再问）；完成了返回命中 chunk 数
（× chunk_size = 命中 token 数），并把 job 从表里删掉。纯 L1 时"prefetch"瞬间完成，
第一次 QUERY 就能拿到数。

客户端拿到命中数后会把自己已有的部分（它自己的 prefix cache 命中段）通过
`FREE_LOOKUP_LOCKS`（`lookup.py:461`）提前还锁 —— 那段它不来取了。

### RETRIEVE → L1 → GPU，批量 H2D

`LMCacheDrivenTransferModule.retrieve`（`lmcache_driven_transfer.py:1117`）：

1. 按 instance_id 找回 `GPUCacheContext`（:1149）；
2. `read_prefetched_results`（`storage_manager.py:254`）取出 lookup 时**已锁好**的
   L1 对象 —— 注意它用的是 `unsafe_read`（:290）：不查、不等，假定对象必然在场；
   此处任何 miss 都是锁/驱逐竞态**异常**（:315），不是正常 cache miss；
3. `transfer_kv_per_object_group(..., batch_size=max_batch_size, direction=H2D)`
   （:1240-1248）—— **批量** H2D，从 pinned DRAM 散射进消息给定的 GPU blocks；
4. `event.record()`（:1257）→ event handle 随响应回去；`finish_read_prefetched` 作为
   stream callback 释放读锁。

### STORE → GPU → L1，逐 chunk D2H，all-or-nothing

`LMCacheDrivenTransferModule.store`（:907）：

1. `reserve_write(obj_keys, layout_desc, "new")`（:1049 → `l1_manager.py:438`）在 L1
   分配 pinned 内存；**已存在的 key 被跳过**（:1058）—— 天然去重，重复 store 不花钱；
2. `vllm_event.wait(stream=...)`（:1008-1011）—— 先等客户端 event（KV 已写进 GPU
   pages）再开始读；
3. `transfer_kv_per_object_group(..., batch_size=1, direction=D2H)`（:1065-1073），
   注释明说 "**batch_size must stay 1 for store**"，逐 chunk 拷；
4. 全部成功才 `finish_write` 提交进 L1 索引（:1084）—— **all-or-nothing**（:935-941）：
   任何 chunk 失败整个 store 作废（后续查询当 miss、客户端重算），绝不留半截缓存；
5. `event.record()` 回传。

### 一图流

```
客户端(vLLM)                          lmcache server
────────────                         ──────────────────────────────────────
LOOKUP ──────────────────────────▶ lookup:   哈希 → dict 查 L1 → 数前缀 → 加读锁
QUERY ◀──命中 token 数─────────────┘
RETRIEVE(block_ids, event) ──────▶ retrieve: 取锁定的 L1 对象 ═批量 H2D═▶ GPU pages
   （发完即走，靠 event 闭环）        └ event.record() → 还读锁
STORE(block_ids, event) ─────────▶ store:    reserve_write(去重) → 等客户端 event
   （发完即走，靠 event 闭环）        └ ═逐 chunk D2H═▶ L1 → finish_write 提交索引
```

### 涉及文件（L1 主线）

| 文件（`lmcache/` 下） | 角色 |
|---|---|
| `v1/multiprocess/server.py` | `MPCacheServer`：组装 EngineModule、MQ 主循环 |
| `v1/multiprocess/mq.py` + `protocol.py`（+`protocols/`） | ZMQ+msgspec 消息队列；5 种消息的 request/response struct |
| `v1/multiprocess/modules/lookup.py` | LOOKUP / QUERY / FREE_LOOKUP_LOCKS 的 handler |
| `v1/multiprocess/modules/lmcache_driven_transfer.py` | REGISTER / STORE / RETRIEVE 的 handler + 真正下拷贝的 `transfer_kv_per_object_group` |
| `v1/multiprocess/modules/management.py` | GET_CHUNK_SIZE、心跳应答 |
| `v1/multiprocess/token_hasher.py` | 链式前缀哈希 |
| `v1/multiprocess/engine_context.py` | 模块共享 ctx（storage_manager、session、event_bus 都挂这） |
| `v1/multiprocess/futures.py` | `CUDAMessagingFuture`（双条件完成语义的实现） |
| `v1/distributed/storage_manager.py` | 存储统一入口：submit_prefetch_task / read_prefetched_results / reserve_write / finish_write |
| `v1/distributed/l1_manager.py` | L1 本体：pinned DRAM 分配、`dict[ObjectKey,…]` 索引、读写锁、驱逐 |

客户端侧（跑在 vLLM 进程里的 LMCache 插件，一般不需要读）：
`integration/vllm/lmcache_mp_connector.py`（钩进 vLLM 的 connector）、
`vllm_multi_process_adapter.py`（ZMQ 客户端）、`kv_cache_group_edits.py`
（注册前的 KV 视图对齐，hybrid 模型相关）。

### 三个必记的设计点

1. **控制流走 ZMQ，数据流走 CUDA IPC + server 自己的 stream。**
2. **完成语义 = ZMQ 响应（已 enqueue）+ CUDA event（已做完），双条件。**
3. **retrieve 批量（`max_batch_size`）、store 逐 chunk（`batch_size=1`）** —— 这个
   不对称是 store 侧 submit-bound（16% duty cycle，见 doc 11）的代码级出处。

---

## 第二层：加入 L2 —— 前台不变，多两个后台搬运线程

5 种消息、两条 GPU 拷贝路径**全部不变**。变化都在 server 内部：

### 行为变化

**① LOOKUP 从"查索引"变成"真正的预取"。**
`submit_prefetch_task` 先照旧算 L1 前缀命中并加锁，剩下的 miss 交给
**`PrefetchController`**（`storage_controllers/prefetch_controller.py`）后台线程
（`storage_manager.py:553-568`）。它做四件事：向所有 L2 adapter 发 `lookup_and_lock` →
按 TrimPolicy 裁剪保留集 → 在 L1 `reserve_write` 分配落地缓冲 → 发 load 任务，完成后把
这些 L1 条目从写锁翻成读锁。最终命中 = L1 前缀 + L2 装载成功部分（`_combine_found`，:593）。

**② QUERY 的语义随之变重。**
`query_prefetch_status`（`lookup.py:369`）只在**整个 L2→L1 搬运完成后**才返回命中数，
进行中一律 None。所以客户端看到的"lookup 窗口"被拉长到覆盖整个 L2 读盘/读网 ——
但契约没变：客户端每周期问一次，没好就先跑别的请求，**没有任何人阻塞**。

**③ RETRIEVE 一字不改。**
还是 `unsafe_read` + 批量 H2D —— prefetch 已保证对象在 L1 且锁好。
**L2 数据永远不直接进 GPU，路径固定 L2→L1→GPU。**
（"Lookup does not read KV — it starts a prefetch" 在 L2 场景下才字面成立。）

**④ STORE 的 D2H 不变，多一条异步下沉流水线。**
`finish_write` 提交进 L1 索引的那一刻，STORE 对客户端就算完成了。它触发的 L1 写完成
事件由 **`StoreController`**（`storage_controllers/store_controller.py`）后台线程监听：
按 `StorePolicy` 决定下沉去向 → 按 shape 分组批量 `submit_store_task` → eventfd 等 L2
完成 → 释放 L1 读锁、可选删 L1 条目。整个 L1→L2 对客户端不可见。

**⑤ 两个新语义。**
- **L1 变成 L2 之上的缓存层**：条目持久化到 L2 后可被 `eviction_controller` 从 L1 驱逐，
  容量上限从 L1 的 DRAM 变成整个 L2；驱逐掉的下次 LOOKUP 由 prefetch 拉回。
- **eventfd 驱动**：两个 controller 的循环都是 `select` eventfd —— fs/nixl 这类 adapter
  有真 completion fd；没有的（见 P2P）靠周期脉冲模拟。

### 新增文件

| 文件（`lmcache/v1/distributed/` 下） | 角色 |
|---|---|
| `storage_controllers/prefetch_controller.py` | L2→L1 异步预取线程 |
| `storage_controllers/store_controller.py` | L1→L2 异步下沉线程 |
| `storage_controllers/prefetch_policy.py` / `store_policy.py` | 保留集裁剪策略（PREFIX/SPARSE）/ 下沉去向策略 |
| `storage_controllers/eviction_controller.py` | L1 驱逐 |
| `l2_adapters/base.py` + 具体 adapter（如 `fs_native_l2_adapter.py`） | L2 的 submit/query 任务式接口 + 真正的 IO |
| `l2_adapters/serde_wrapper.py` | 需要序列化的 adapter 的包装层 |

```
        LOOKUP                        RETRIEVE           STORE
客户端 ── ZMQ ───────────────────────── ZMQ ────────────── ZMQ ──────────
server  L1 查索引+锁 ──miss──▶ ┌────────┐   L1 ═H2D═▶ GPU   GPU ═D2H═▶ L1
                              │Prefetch│                       │finish_write 事件
                              │Ctrl 线程│ L2 ─load─▶ L1   ┌────▼───┐
                              └────────┘  (写锁→读锁)     │Store   │
                                                          │Ctrl 线程│ L1 ─store─▶ L2
                                                          └────────┘
```

一句话：**L1-only 是"前台"，L2 只是挂了两个 eventfd 驱动的后台搬运工；代价是 QUERY
要多转几圈才能拿到命中数。**

---

## 第三层：加入 P2P —— 连后台机制都不变，只是多了"长得像 L2 的 peer"

对本机链路而言，P2P 没有新机制：**`P2PL2Adapter` 就是插进 PrefetchController 扇出列表里
的又一个 L2 adapter**（每个活着的 peer 一个实例）。LOOKUP / RETRIEVE / STORE 全部复用
第二层的路径。所有新东西藏在 adapter 接口背后 + 一个新控制面。

### Adapter 背后的三个不同

**① lookup_and_lock 变成跨机 RPC。**
`P2PL2Adapter.submit_lookup_and_lock_task`（`l2_adapters/p2p_l2_adapter.py`）把 keys 经
ZMQ 发给**对端 server 的 `P2PController.p2p_lookup_and_lock`**
（`modules/p2p_controller.py:222`）。对端拿这批 keys 在自己的 StorageManager 上跑
`submit_prefetch_task(skip_l2=True)`（:244）：

> **peer 只贡献自己 L1 里现成的热数据，绝不为你翻自己的磁盘**（:243 注释原文：
> "skip_l2=True — only objects already resident in L1 are locked"）。

P2P 的语义是"借邻居的 DRAM"，不是"借邻居的整个存储栈"。命中部分在**对端 L1 被读锁
pin 住**（防止网络读期间被驱逐 —— 锁第一次跨机），`p2p_query_lookup_results`（:262）
返回每个 key 的 `TransferChannelAddress`（offset+size，miss 为负 offset）。

**② load 变成内存到内存的网络读。**
本机照旧 `reserve_write` 分配 L1 缓冲，transfer channel（NIXL/UCX）按地址直接从对端 L1
pinned 内存读进本机 L1，完事 `p2p_unlock_objects`（:304）解对端的 pin。三种命中路径：

```
L1 命中:   本机 L1 ═H2D═▶ GPU
L2 命中:   磁盘 ─load─▶ 本机 L1 ═H2D═▶ GPU
P2P 命中:  对端 L1 ══NIXL/UCX══▶ 本机 L1 ═H2D═▶ GPU
```

（doc 10 量到的 2× UCX 传输放大发生在第三行的网络段。）

**③ store 侧对 P2P 是零。**
`P2PL2Adapter.submit_store_task` 是 no-op（peer 只读），StorePolicy 不会把数据下沉给
邻居。数据在集群里的扩散方式只有一种：**每个节点缓存自己算过的东西，别人来读**，
没有主动复制。

### 新增的控制面（P2P 独有）

`P2PController` 除了应答 RPC，还跑一个后台 poll 线程：向 coordinator 注册自己、周期
`GET /instances` 发现 peer、为新 peer 建 adapter、**连续 3 次不见就拆**（`_MAX_MISSES=3`）。
peer 挂了/网络抖了 → 那个 adapter 的 lookup 超时算 miss（RPC 3s，lookup/load 各 10s
deadline）→ 退化成纯 L2/重算，**请求永不被 P2P 阻塞**。

工程细节：RPC 与 transfer 读都没有 completion fd，`PeriodicEventNotifier` 每 5ms 脉冲
eventfd 让 PrefetchController 的 select 循环醒来轮询 `query_*` —— 这是 P2P adapter 与
fs adapter 在事件模型上唯一的差别。

### 新增文件

| 文件 | 角色 |
|---|---|
| `v1/multiprocess/modules/p2p_controller.py` | 控制面（coordinator 注册 / peer 发现 / adapter 生命周期）+ 应答 peer 的 lookup RPC |
| `v1/distributed/l2_adapters/p2p_l2_adapter.py` | 消费端：把"某个 peer 的缓存"伪装成本机一块只读 L2 |
| `v1/distributed/transfer_channel/` | NIXL/UCX 数据面（内存到内存网络读） |
| `v1/distributed/storage_controllers/adapter_lifecycle.py` | 运行时增删 adapter 的 Add/RemoveAdapterOp（纯 L2 时 adapter 集合是静态的，P2P 才让它动起来） |

**对称性**：每个节点同时是客户端和服务端 —— 它的 PrefetchController 通过 P2P adapter
读别人，它的 P2PController 把自己的 L1 借给别人读。两个方向共用同一套 L1 锁和 prefetch
机制。

---

## 附录 A：vLLM 侧一分钟版（只在需要对面视角时看）

- vLLM 内部分两种角色：**scheduler**（EngineCore 进程，决定每步跑哪些请求、管 GPU block
  账本，从不碰张量）和 **worker**（每 GPU 一个，持有权重和 KV 张量、跑 forward）。
- LMCache 的 connector 作为插件被实例化**两次**（`lmcache_mp_connector.py:608/:621`）：
  scheduler 进程里那份发 LOOKUP/QUERY、决定取多少；worker 进程里那份发
  REGISTER/RETRIEVE/STORE、录 event。两份各自持有到 server 的 ZMQ 连接。
- **"发完就走"能成立**，是因为等 KV 的请求根本不在当前 forward 的 batch 里：scheduler
  把它挂起（`WAITING_FOR_REMOTE_KVS`），直到 worker 侧的 future（双条件）完成、
  `get_finished` 上报，才重新调度它 —— 那时 KV 已在它的 blocks 里。
- 客户端异常时的兜底：server 心跳丢失 → connector 进入 degraded mode（lookup 全 miss、
  store 跳过、在途 retrieve 的 blocks 标记为 error 让 vLLM 重算），恢复后自动重新
  REGISTER。

---

## 附录 B：单步调试速查（配合 src/ 调试环境）

| 想看什么 | 断点 | 触发方式 |
|---|---|---|
| lookup（哈希 + 匹配 + 加锁） | `lookup.py:206` | `./req.sh send`（首次，miss） |
| store D2H | `lmcache_driven_transfer.py:907` | 同上，forward 结束后到达 |
| retrieve H2D | `lmcache_driven_transfer.py:1117` | `./req.sh clear_cache` 后再 `./req.sh send` |
| L2 预取 | `prefetch_controller.py` 的请求处理循环 | 需配 L2 后驱逐 L1（或重启 server 保留 L2） |
| L1→L2 下沉 | `store_controller.py` 的 L1 写监听 | store 后自动触发 |
| P2P 应答端 | `p2p_controller.py:222` | 需双实例 + coordinator（见 `l2_support/p2p-demo.sh`） |

注意：断点一律 **Suspend: Thread**（挂住整个进程会饿死心跳，客户端进入 degraded mode）；
server 重启即清空 L1。
