# MP 模式代码总览：vLLM + LMCache 如何协作 —— L1 → L2 → P2P 三层递进

**范围与版本。** 只讲 **MP 模式**（`LMCacheMPConnector`），不涉及老的 in-process 路径
（`cache_engine.py` / `storage_backend/` / `cache_controller/`，那条路我们的部署永远不会执行）。
所有 `file:line` 均指 **LMCache v0.5.1**（`onbording/src/lmcache` 的 editable clone，可直接打断点），
vLLM 为 v0.25.1。配套调试环境见 `src/vllm-dbg.sh` / `src/req.sh`。

**读法。** 三层递进：先看纯 L1 时两个进程怎么配合（这是全部"前台"逻辑）；再加 L2，看行为哪里变了、
多了哪些文件（答案：前台不变，多了两个后台搬运线程）；最后加 P2P（答案：连后台机制都不变，只是多了
几个"长得像 L2 的 peer"和一个控制面）。

---

## 第一层：纯 L1 —— vLLM 与 lmcache server 的完整协作

进程模型：vLLM（scheduler + worker）和 lmcache server 是**两个独立进程**。控制消息走 ZMQ
（`mq.py`，msgspec 编码），数据不走消息——server 通过启动时的 CUDA IPC 注册**直接读写 vLLM 的
GPU KV pages**。vLLM 全程不搬一个字节。

### 第 0 步（启动，一次性）：把 GPU 显存交给 server

- vLLM worker：`LMCacheMPConnector.register_kv_caches`（`lmcache_mp_connector.py:702`）
  把每层 KV 张量包成 CUDA-IPC wrapper（`vllm_multi_process_adapter.py:150`），连同 layout
  信息发 `REGISTER_KV_CACHE`。注册前先过 `kv_cache_group_edits.py` 做视图编辑（hybrid 模型的
  page 对齐发生在这里）。
- server：`LMCacheDrivenTransferModule.register_kv_cache`（`lmcache_driven_transfer.py:816`）
  打开 IPC handle，建 `GPUCacheContext` —— 从此 server 手握 vLLM GPU pages 的直接映射，
  之后所有 H2D/D2H 都在 **server 自己的 CUDA stream** 上做。**这就是 profiling 必须抓 server
  进程的根本原因。**
- 同时 adapter 在 init 时同步取 chunk size（`vllm_multi_process_adapter.py:1151`，
  `GET_CHUNK_SIZE`，超时 `lmcache.mp.mq_timeout`）；chunk size 必须整除 vLLM block size。

### 第 1 步：Lookup（scheduler 侧，纯异步，从不阻塞）

1. scheduler 对新请求调 `get_num_new_matched_tokens`（`lmcache_mp_connector.py:898`）：
   `maybe_submit_lookup_request`（adapter:699）把全部 token ids 发出去，**立即返回**；
   结果没到就回 `(None, True)`（:942）——"下个 step 再问"。
2. server 端 `LookupModule.lookup`（`lookup.py:206`）：token → chunk 哈希（:262），
   然后 `storage_manager.submit_prefetch_task`（`storage_manager.py:399`）。纯 L1 时
   :457 的注释就是全部真相：
   > "now we only have L1, so the prefetch is essentially checking how many objects are
   > already in L1, and adding read locks to them."

   即：**查 L1 索引 + 给命中 chunk 加读锁**（`l1_manager.reserve_read`，:460）。读锁保证
   lookup 与 retrieve 之间条目不被驱逐。
3. 下个 step，`check_lookup_result`（adapter:815）拿到命中前缀长度 `ret`（chunk 对齐），
   connector 算 `need_to_load = ret - num_computed_tokens`（减掉 vLLM 自己 APC 已有的），
   返回 `(need_to_load, True)` —— 异步装载。

### 第 2 步：分配与下发（scheduler → worker 只传元数据）

- `update_state_after_alloc`（:970）：把分配的 GPU block ids 记进 request tracker，状态
  `PREFETCHING → WAITING_FOR_LOAD`；**提前释放 vLLM APC 已覆盖段的读锁**（:1023-1043，
  `free_lookup_locks`）。
- `build_connector_meta`（:1050）：把每个请求的 RETRIEVE/STORE 操作（含 block ids）打包成
  connector metadata，随 scheduler output 发给 worker。

### 第 3 步：Retrieve —— L1 → GPU（H2D，server 执行）

- worker 在 forward 前调 `start_load_kv`（:725）：录一个 `interprocess=True` 的 CUDA event
  （:757），`batched_submit_retrieve_requests`（adapter:1509）把
  `request_id + gpu_block_ids + event IPC handle` 发过去，**发完就跑 forward，不等**。
- server `retrieve`（`lmcache_driven_transfer.py:1117`）：
  1. 按 instance_id 找回 `GPUCacheContext`（:1149）；
  2. `read_prefetched_results`（`storage_manager.py:254`）取出 lookup 时已锁好的 L1 对象；
  3. `transfer_kv_per_object_group(..., batch_size=max_batch_size, direction=H2D)`
     （:1240-1248）—— **批量** H2D，从 pinned DRAM 直接散射进 vLLM 的 GPU pages；
  4. `event.record()`（:1257）回传；`finish_read_prefetched` 作为 stream callback 释放读锁。
- vLLM 侧请求挂在 `WAITING_FOR_REMOTE_KVS`，直到 `get_finished`（:836 → adapter:1554）
  确认 event 完成才恢复调度。

### 第 4 步：Store —— GPU → L1（D2H，server 执行）

- forward 结束时 `wait_for_save`（:799）：在当前 stream 录 interprocess event（语义：
  "forward 对 KV pages 的写到此为止"），`batched_submit_store_requests`（adapter:1483），
  同样发完即走。
- server `store`（:907）：
  1. `reserve_write(obj_keys, layout_desc, "new")`（:1049 → `l1_manager.py:438`）在 L1 分配
     pinned 内存；**已存在的 key 被跳过**（:1058）——天然去重；
  2. `vllm_event.wait(stream=...)`（:1008-1011）—— 先等 vLLM forward 写完才读 GPU；
  3. `transfer_kv_per_object_group(..., batch_size=1, direction=D2H)`（:1065-1073），
     注释明说 "**batch_size must stay 1 for store**"，逐 chunk 拷；
  4. 全部成功才 `finish_write` 提交进 L1 索引（:1084）——**all-or-nothing**（:935-941）：
     任何 chunk 失败整个 store 作废，绝不留半截缓存；
  5. event 回传，vLLM 的 `get_finished` 收到后才释放那些 GPU blocks。

### 时序总览

```
vLLM scheduler 进程                 vLLM worker 进程                lmcache server 进程
──────────────────                 ────────────────               ─────────────────────
get_num_new_matched_tokens ──LOOKUP(ZMQ)──────────────────────▶ LookupModule.lookup
  (返回 None,True 先放行)                                          └ L1 查索引 + 加读锁
check_lookup_result ◀──命中前缀长度──────────────────────────────┘
update_state_after_alloc（分块、放掉 APC 段的锁）
build_connector_meta ──metadata──▶ start_load_kv ──RETRIEVE(ZMQ, block_ids+event)──▶ retrieve
                                     │ forward 同时进行              └ 读锁定的 L1 对象
                                     │                              └ 批量 H2D 进 vLLM GPU pages
                                   get_finished ◀───CUDA event──────┘ (释放读锁)
                                   wait_for_save ──STORE(ZMQ, block_ids+event)──▶ store
                                     (发完即走)                      └ reserve_write 分配 L1
                                                                    └ 等 vLLM event(forward 写完)
                                                                    └ 逐 chunk D2H → finish_write
                                   get_finished ◀───CUDA event──────┘
```

### 涉及文件（纯 L1 主线 = 8 个）

| 层 | 文件（`lmcache/` 下） | 角色 |
|---|---|---|
| vLLM 进程 | `integration/vllm/lmcache_mp_connector.py` | vLLM `KVConnectorBase_V1` 实现：scheduler 钩子 + worker 钩子 + request tracker |
| vLLM 进程 | `integration/vllm/vllm_multi_process_adapter.py` | ZMQ 客户端（scheduler/worker adapter 各一个类）：lookup 提交与轮询、KV 张量 IPC 注册、batched store/retrieve、心跳 |
| vLLM 进程 | `integration/vllm/kv_cache_group_edits.py` | 注册前的 KV 视图编辑/对齐（hybrid 模型 page 对齐） |
| 通道 | `v1/multiprocess/mq.py` | ZMQ + msgspec 消息队列 |
| 通道 | `v1/multiprocess/protocol.py`（+`protocols/`） | 消息类型定义（LOOKUP / STORE / RETRIEVE / REGISTER_KV_CACHE / QUERY_PREFETCH_* …） |
| server | `v1/multiprocess/server.py` | `MPCacheServer`：组装 EngineModule、MQ 主循环（`run_cache_server`） |
| server | `v1/multiprocess/modules/lookup.py` | `LookupModule`：lookup / query_prefetch_status / free_lookup_locks |
| server | `v1/multiprocess/modules/lmcache_driven_transfer.py` | `LMCacheDrivenTransferModule`：register_kv_cache、store（D2H）、retrieve（H2D）、`transfer_kv_per_object_group` |
| 存储 | `v1/distributed/storage_manager.py` | `StorageManager`：submit_prefetch_task / read_prefetched_results / reserve_write / finish_write |
| 存储 | `v1/distributed/l1_manager.py` | L1 本体：pinned DRAM 分配、索引、读写锁、驱逐 |

外围路过但不必细读：`modules/management.py`（GET_CHUNK_SIZE、心跳应答）、
`multiprocess/token_hasher.py`（token→chunk 哈希）、`multiprocess/engine_context.py`
（模块共享 ctx）。

### 三个必记的设计点

1. **控制流走 ZMQ，数据流走 CUDA IPC + server 自己的 stream** —— vLLM 零拷贝零阻塞。
2. **跨进程同步靠两个方向的 interprocess CUDA event**：vLLM→server（"forward 写完了，可以
   D2H"）；server→vLLM（"copy 完成了，可以继续调度/释放 block"）。
3. **retrieve 批量（`max_batch_size`）、store 逐 chunk（`batch_size=1`）** —— 这个不对称是
   store 侧 submit-bound（16% duty cycle，见 doc 11）的代码级出处。

---

## 第二层：加入 L2 —— 前台不变，多两个后台搬运线程

**vLLM 侧一行不变，两条 GPU 拷贝也不变。**变化全部在 server 进程内部：

### 行为变化

**① Lookup 从"查索引"变成"真正的预取"。**
`submit_prefetch_task` 先照旧算 L1 前缀命中并加锁，剩下的 miss 交给
**`PrefetchController`**（`storage_controllers/prefetch_controller.py`）后台线程
（`storage_manager.py:553-568`）。它做四件事：向所有 L2 adapter 发 `lookup_and_lock` →
按 TrimPolicy 裁剪保留集 → 在 L1 `reserve_write` 分配落地缓冲 → 发 load 任务，完成后把这些
L1 条目从写锁翻成读锁。最终命中 = L1 前缀 + L2 装载成功部分（`_combine_found`，:593）。

关键时序语义：vLLM scheduler adapter 用的是 **`QUERY_PREFETCH_STATUS`**（adapter:819），
它**只在整个 prefetch 完成后**才返回命中数（`lookup.py:369`，进行中回 None）。所以：
> **加了 L2 后，"lookup 窗口"被拉长到覆盖整个 L2→L1 的数据搬运。** vLLM 每 step 轮询一次，
> 期间请求挂着、其它请求正常调度 —— L2 的磁盘/网络延迟藏进异步窗口，scheduler 永不阻塞。

**② Retrieve 完全不变。**
`read_prefetched_results`（`storage_manager.py:254`）用 `unsafe_read`（:290）——不等待、
不查 L2，假定对象已在 L1 且被 prefetch 锁好；此处任何 miss 都算锁/驱逐竞态异常（:315），
不是正常 cache miss。**L2 数据永远不直接进 GPU，路径固定 L2→L1→GPU。**
（"Lookup does not read KV — it starts a prefetch" 在 L2 场景下才字面成立。）

**③ Store 的 D2H 不变，多一条异步下沉流水线。**
`store()` 照旧 D2H 进 L1 + `finish_write` 提交 —— **vLLM 眼里数据进 L1 那刻 store 就完成了**。
`finish_write` 触发的 L1 写完成事件由 **`StoreController`**
（`storage_controllers/store_controller.py`）后台线程监听：按 `StorePolicy` 决定下沉去向 →
按 shape 分组批量 `submit_store_task` → eventfd 等 L2 完成 → 释放 L1 读锁、可选删 L1 条目。
整个 L1→L2 对 vLLM 不可见。

**④ 两个新语义。**
- **L1 变成 L2 之上的缓存层**：条目持久化到 L2 后可被 `eviction_controller` 从 L1 驱逐，
  容量上限从 L1 的 DRAM 变成整个 L2；驱逐掉的下次 lookup 由 prefetch 拉回。
- **eventfd 驱动**：两个 controller 的循环都是 `select` eventfd —— fs/nixl 这类 adapter 有真
  completion fd；没有的（见 P2P）靠周期脉冲模拟。

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
                 lookup                     retrieve            store
vLLM ────────── ZMQ ──────────────────────── ZMQ ─────────────── ZMQ ────────
server   L1 查索引+锁 ──miss──▶ ┌────────┐    L1 ═H2D═▶ GPU      GPU ═D2H═▶ L1
                               │Prefetch│                            │finish_write 事件
                               │Ctrl 线程│ L2 ─load─▶ L1        ┌────▼───┐
                               └────────┘  (写锁→读锁)          │Store   │
                                                               │Ctrl 线程│ L1 ─store─▶ L2
                                                               └────────┘
```

一句话：**L1-only 是"前台"，L2 只是挂了两个 eventfd 驱动的后台搬运工；代价是 lookup 的
异步等待窗口变长。**

---

## 第三层：加入 P2P —— 连后台机制都不变，只是多了"长得像 L2 的 peer"

对本地链路而言，P2P 没有新机制：**`P2PL2Adapter` 就是插进 PrefetchController 扇出列表里的
又一个 L2 adapter**（每个活着的 peer 一个实例）。lookup / retrieve / store 全部复用第二层的
路径。所有新东西藏在 adapter 接口背后 + 一个新控制面。

### Adapter 背后的三个不同

**① lookup_and_lock 变成跨机 RPC。**
`P2PL2Adapter.submit_lookup_and_lock_task`（`l2_adapters/p2p_l2_adapter.py`）把 keys 经
ZMQ 发给**对端的 `P2PController.p2p_lookup_and_lock`**（`modules/p2p_controller.py:222`）。
对端拿这批 keys 在自己的 StorageManager 上跑 `submit_prefetch_task(skip_l2=True)`（:244）：

> **peer 只贡献自己 L1 里现成的热数据，绝不为你翻自己的磁盘**（:243 注释原文：
> "skip_l2=True — only objects already resident in L1 are locked"）。

P2P 的语义是"借邻居的 DRAM"，不是"借邻居的整个存储栈"。命中部分在**对端 L1 被读锁 pin 住**
（防止网络读期间被驱逐 —— 锁第一次跨机），`p2p_query_lookup_results`（:262）返回每个 key 的
`TransferChannelAddress`（offset+size，miss 为负 offset）。

**② load 变成内存到内存的网络读。**
本地照旧 `reserve_write` 分配 L1 缓冲，transfer channel（NIXL/UCX）按地址直接从对端 L1
pinned 内存读进本地 L1，完事 `p2p_unlock_objects`（:304）解对端的 pin。三种命中路径对比：

```
L1 命中:   本地 L1 ═H2D═▶ GPU
L2 命中:   磁盘 ─load─▶ 本地 L1 ═H2D═▶ GPU
P2P 命中:  对端 L1 ══NIXL/UCX══▶ 本地 L1 ═H2D═▶ GPU
```

（doc 10 量到的 2× UCX 传输放大发生在第三行的网络段。）

**③ store 侧对 P2P 是零。**
`P2PL2Adapter.submit_store_task` 是 no-op（peer 只读），StorePolicy 不会把数据下沉给邻居。
数据在集群里的扩散方式只有一种：**每个节点缓存自己算过的东西，别人来读**，没有主动复制。

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
| `v1/distributed/l2_adapters/p2p_l2_adapter.py` | 消费端：把"某个 peer 的缓存"伪装成本地一块只读 L2 |
| `v1/distributed/transfer_channel/` | NIXL/UCX 数据面（内存到内存网络读） |
| `v1/distributed/storage_controllers/adapter_lifecycle.py` | 运行时增删 adapter 的 Add/RemoveAdapterOp（纯 L2 时 adapter 集合是静态的，P2P 才让它动起来） |

**对称性**：每个节点同时是客户端和服务端 —— 它的 PrefetchController 通过 P2P adapter 读别人，
它的 P2PController 把自己的 L1 借给别人读。两个方向共用同一套 L1 锁和 prefetch 机制。

---

## 附：单步调试建议（配合 src/ 调试环境）

| 想看什么 | 断点 | 触发方式 |
|---|---|---|
| lookup（第一命中判定） | `lookup.py:206` | `./req.sh send`（首次，miss） |
| store D2H | `lmcache_driven_transfer.py:907` | 同上，forward 结束后到达 |
| retrieve H2D | `lmcache_driven_transfer.py:1117` | `./req.sh clear_cache` 后再 `./req.sh send` |
| L2 预取 | `prefetch_controller.py` 的请求处理循环 | 需配 L2 后驱逐 L1（或重启 server 保留 L2） |
| L1→L2 下沉 | `store_controller.py` 的 L1 写监听 | store 后自动触发 |
| P2P 应答端 | `p2p_controller.py:222` | 需双实例 + coordinator（见 `l2_support/p2p-demo.sh`） |

注意：断点一律 **Suspend: Thread**（挂住整个进程会饿死心跳，vLLM 侧进入 degraded mode）；
server 重启即清空 L1。
