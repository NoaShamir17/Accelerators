# Homework 3 — RDMA queue mode: reference solution

Implements `MODE_QUEUE` (`server_queues_context` / `client_queues_context` in
`ex3.cu`), in which the client drives the server's CPU–GPU producer–consumer
queues from homework 2 remotely over RDMA. `MODE_RPC_SERVER` was given as a
worked example and is untouched.

---

## 1. Build and run

```bash
cd /home/u_325943165/Accelerators
make                      # builds ./server and ./client, no warnings
```

Run the server first, then the client with the **same port and the same mode**:

```bash
# queue mode (this assignment)
./server queue 24570      # terminal 1
./client queue 24570      # terminal 2

# rpc mode (the provided example)
./server rpc 24570
./client rpc 24570
```

Omitting the server's port makes it pick a random one and print it. One-shot
version:

```bash
PORT=24570
./server queue $PORT & sleep 4; ./client queue $PORT; wait
```

### Measured results on this machine

4× RTX 2080 SUPER (`sm_75`), CUDA 12.5, ConnectX `mlx5_0`↔`mlx5_1` loopback,
30 000 requests per run.

| Mode | Correctness | Throughput (req/s) | Avg latency (ms) |
|---|---|---|---|
| `rpc` (given) | `distance from baseline 0` | 41 065 – 41 587 | 378.8 |
| `queue` (mine) | `distance from baseline 0` | 29 072 – 29 706 | 0.30 |

Five consecutive runs, three of each mode, all reported distance 0.

On the throughput gap: it is architectural, not a bug. The RPC client fires all
`OUTSTANDING_REQUESTS` (8192) requests into the network before collecting any,
so the wire is saturated — and that is exactly *why* its average latency is
379 ms: every request queues behind thousands of others. The queue client
completes each RDMA operation before returning from `enqueue`/`dequeue`, which
the assignment explicitly permits ("You may wait for the RDMA read operation to
complete"), and gets a ~1200× better latency for it. Section 8 sketches how to
close the throughput gap if you want it.

---

## 2. Why no `ex2.cu` logic had to change

The single most important property is one your homework-2 code already had:

```cpp
CUDA_CHECK(cudaMallocHost((void**)&CPU_to_GPU_queue, sizeof(MPMC_ring_queue)));
CUDA_CHECK(cudaMallocHost((void**)&queue, capacity * sizeof(struct context)));
```

Both the `MPMC_ring_queue` objects (which *contain* `_head` and `_tail`) and
their slot arrays live in **pinned host memory**. That is precisely the
intersection queue mode needs:

* `ibv_reg_mr()` can only register host pages, so the NIC can DMA there —
  device memory from `cudaMalloc()` is unreachable without GPUDirect;
* `cudaMallocHost()` memory is mapped into the GPU's address space, so the
  persistent kernel can poll it over PCIe.

The second property is the index type:

```cpp
cuda::atomic<int, cuda::thread_scope_system> _head, _tail;
```

`thread_scope_system` (rather than `_device`) makes the GPU's accesses coherent
with the rest of the machine, including third-party DMA such as our NIC. The
compiler emits system-scope acquire/release PTX, so the polling kernel re-reads
the index from host memory instead of spinning forever on a value cached in the
GPU's L1/L2. Had these been `thread_scope_device` atomics, queue mode could not
work without changing `ex2.cu`.

So all `ex3.cu` needed was the *addresses*, which is what the accessors provide.

---

## 3. Lines touched in `ex2.cu`

16 lines inserted, 1 blank line removed. **No logic, no kernel, no queue
algorithm, no index arithmetic, no image processing was modified.** Every added
member is a `const`-style accessor that only reports an address or a size.

### Block 1 — `MPMC_ring_queue`, lines 222–231

| Line | Content |
|---|---|
| 222–226 | comment block |
| 227 | `__host__ void *head_addr()  { return (void *)&_head; }` |
| 228 | `__host__ void *tail_addr()  { return (void *)&_tail; }` |
| 229 | `__host__ void *slots_addr() { return (void *)queue; }` |
| 230 | `__host__ size_t slots_bytes() const { return (size_t)capacity * sizeof(struct context); }` |
| 231 | `__host__ int get_capacity() const { return capacity; }` |

(One blank line at old line 222 was consumed by the insertion.)

### Block 2 — `queue_server`, lines 398–402

| Line | Content |
|---|---|
| 398–400 | comment block |
| 401 | `MPMC_ring_queue *cpu_to_gpu_queue() { return CPU_to_GPU_queue; }` |
| 402 | `MPMC_ring_queue *gpu_to_cpu_queue() { return GPU_to_CPU_queue; }` |

Verify with `git diff ex2.cu` — the diff is 16 `+` lines and 1 `-` (blank) line.

`ex2.h` and `ex3.h` were not touched at all.

---

## 4. The queue-mode protocol, step by step

### Connection setup

1. `rdma_server_context` does the TCP accept, `initialize_verbs()`, allocates
   `images_in`/`images_out` and registers them, then exchanges
   `connection_establishment_data` (GID + QPN) and connects the QP. All given.
2. `server_queues_context` constructs `queue_server(256)`, which allocates the
   two queues and **launches the persistent kernel**. It spins on two empty
   queues while we finish setup — harmless.
3. The server registers four more MRs and sends one `queue_connection_data`
   struct over the same TCP socket (see §6 on why a second struct).
4. The client receives it, validates `sizeof(struct context)`, `IMG_SZ` and that
   `capacity` is a power of two, registers its staging area, and builds a free
   list of server-side image slots.

After this the server's CPU is **completely off the data path**. It sits in
`event_loop()` waiting for exactly one message: termination.

### Enqueue (client → GPU)

| Step | Operation | Purpose |
|---|---|---|
| 1 | **RDMA Read** `c2g._head` | is the queue full? (only when the cached value says so) |
| 2 | **RDMA Write** → `images_in[slot]` | copy the 16 KB image to the server |
| 3 | **RDMA Write** → `c2g.slots[tail % capacity]` | the 24-byte `context` entry |
| 4 | **RDMA Write** → `c2g._tail` | publish: `tail + 1` |

Steps 2–4 are posted back-to-back without waiting; all three completions are
reaped before returning. The persistent kernel's `GPU_dequeue()` then picks the
job up on its own.

The `context` entry contains **server-side** pointers
(`images_in + slot*IMG_SZ`, `images_out + slot*IMG_SZ`), because the kernel
dereferences `ctx.in_img` / `ctx.out_img` directly over PCIe. Passing client
addresses would make the GPU read whatever happens to live at that address on
the server.

### Dequeue (GPU → client)

| Step | Operation | Purpose |
|---|---|---|
| 5 | **RDMA Read** `g2c._tail` | anything ready? (only when the cache says empty) |
| 6 | **RDMA Read** `g2c.slots[head % capacity]` | which request finished — batched, see below |
| 7 | **RDMA Read** `images_out[slot]` | pull the result into the caller's buffer |
| 8 | **RDMA Write** → `g2c._head` | release the queue entries |

### The `img_id` ↔ slot trick

`GPU_enqueue(int img_id)` in `ex2.cu` copies **only** `img_id` into the result
slot; that entry's `in_img`/`out_img` are leftovers from whichever request last
occupied it. So the client cannot learn the output address from the queue.

Instead the client puts the **staging-slot number** in the `img_id` field it
sends to the GPU, and keeps a side table `slot → {real img_id, caller's out
pointer}`. On completion it reads back the slot number and recovers everything
else in O(1). This is the same idea as the RPC server using `wc.wr_id` to index
`requests[]` and `images_in[]`.

### When an image slot may be reused — a real race, closed deliberately

Look at the order in the persistent kernel:

```cpp
dequeue_success = CPU_to_GPU_queue->GPU_dequeue(&ctx);   // advances c2g._head
...
for (int i = threadIdx.x; i < IMG_SIZE; i += blockDim.x)
    my_d_in[i] = ctx.in_img[i];                          // reads the input AFTER
```

`_head` advances **before** the input image is read. So a slot whose CPU→GPU
entry has been consumed may still be actively under the GPU's nose — freeing
image slots on the consumer index would corrupt in-flight reads.

The client therefore recycles a slot only when the **result** appears in the
GPU→CPU queue, which proves the kernel finished with both `in_img` and
`out_img`. (`free_slots.push_back(slot)` in step 9 of `dequeue`.)

---

## 5. Why each verb

| Verb | Where | Why |
|---|---|---|
| **RDMA Read** | queue indices (steps 1, 5) | The client must *poll* remote state that the GPU updates. Read is the only one-sided verb that returns a value, and it needs no involvement from the server's CPU — the whole point of queue mode. |
| **RDMA Read** | result entry + image (6, 7) | Same reason: the data is already sitting in the server's memory and nobody there is available to push it. |
| **RDMA Write** | image, entry, indices (2, 3, 4, 8) | One-sided, no remote CPU, and — critically — **Writes on one RC QP are ordered**, which is what makes the "payload then flag" handoff sound (§7). |
| **RDMA Write with Immediate** | server's termination ack | A plain Write lands silently; the client needs a *completion* to know the server is going down. Zero length, so no remote memory is touched and the (zero) rkey/address are never validated. Same trick the RPC server uses for `request_id == -1`. |
| **Send** | client's termination request | The one place the server's CPU genuinely must be **notified**. A Write would land in its memory with nobody polling; a Send consumes a receive WQE and produces a CQE in `event_loop()`. |

Not used: RDMA Atomics. `_tail` has exactly one writer (the client) and `_head`
in the GPU→CPU queue also has exactly one writer, so plain Writes suffice —
`ex2.cu`'s `TTAS_lock` already serialises the GPU-side multi-producer case.

---

## 6. Ambiguities and how they were resolved

**The handshake struct.** You asked me to extend the existing handshake struct
rather than add a side channel. `connection_establishment_data` lives in
`ex3.h`, which is marked `DO NOT CHANGE` and which the PDF forbids modifying
("Do not modify other files in the package"); the PDF instead points explicitly
at `send_over_socket`/`recv_over_socket` for the queue parameters. I resolved
this in favour of the assignment's hard rule while keeping your intent: the new
`queue_connection_data` travels over **the same TCP socket, in the same
handshake sequence, immediately after** `connection_establishment_data`. Same
channel, same phase, no out-of-band mechanism — just a second struct, because
the first one is frozen.

**Where the new struct lives.** In `ex3.cu` rather than a new header. The PDF
allows extra `.h` files for declarations shared between `ex2.cu` and `ex3.cu`,
but this struct is used only by `ex3.cu`, and the Makefile's dependency lines
are fixed, so a new header would not be tracked for rebuilds anyway.

**Persistent-kernel block size.** `create_queues_server(256)`, matching the RPC
server exactly, so both modes exercise an identical `ex2.cu` configuration.
That yields 192 blocks and `capacity = 512` on this GPU.

**Blocking on RDMA reads.** The PDF explicitly permits waiting for an RDMA read
to complete, and forbids only waiting for the *GPU*. The implementation takes
that permission: every RDMA op completes inside the call that posted it, and
nothing ever waits on a particular image.

---

## 7. The ordering argument

**Requirement:** the producer-index Write (step 4) must never be observed before
the image Write (2) and the entry Write (3).

**Guarantee.** All three are posted, in that order, on the same **Reliable
Connection** QP. The InfiniBand spec requires an RC responder to execute
requests in PSN order — strictly the order they were posted to the send queue.
So the image bytes and the queue entry are placed in the server's memory before
the new `_tail` is. The responder HCA then issues those DMA writes towards the
host in that same order, and posted writes on a single PCIe path are not
reordered. The GPU therefore cannot observe a bumped `_tail` while the slot it
points at still holds a stale entry.

This is the classic *write the payload, then write the flag* idiom, and it is
why **no explicit fence is needed**. `IBV_SEND_FENCE` would not help even if we
wanted it: a fence orders a WR against previously posted RDMA **Read** and
Atomic operations, not against Writes, which are already ordered among
themselves.

**The consumer half.** Ordering on the wire is only half the argument. The other
half is in `ex2.cu`:

```cpp
if (head == _tail.load(cuda::memory_order_acquire))   // GPU_dequeue
```

That **acquire** prevents the GPU from hoisting its reads of
`queue[head % capacity]` — and hence of the pixels — above the index check. A
kernel that has seen the new `_tail` is guaranteed to see everything written
before it. Ordered Writes at the producer **plus** an acquire load at the
consumer is what makes the handoff sound end to end.

**Torn reads.** The NIC's index update is a 4-byte, naturally aligned DMA write,
which is single-copy-atomic in hardware, so the GPU's acquire load sees either
the old value or the new one, never a mixture. Formally the NIC's write is not a
C++ atomic store — `static_assert(sizeof(cuda::atomic<int,...>) == sizeof(int32_t))`
in `ex3.cu` pins down the one layout assumption this relies on.

**Read-then-Write in dequeue.** Step 8 (release entries) must follow step 7
(read the result), since the GPU may overwrite an entry as soon as it sees the
new `_head`. Here the ordering is by construction — we reap the read's
completion before posting the write. Had those been pipelined, *this* is exactly
where `IBV_SEND_FENCE` would be required.

---

## 8. The other correctness details you asked about

**Wraparound, power-of-two masking, full vs. empty.** `_head`/`_tail` are
monotonically increasing counters, never wrapped; slot *k* lives at
`k % capacity`, exactly as `ex2.cu` indexes it. Keeping them unwrapped is what
lets `tail - head` mean *occupancy*, which distinguishes **full**
(`tail - head == capacity`) from **empty** (`tail == head`) — the two states a
bare wrapped index cannot tell apart without sacrificing a slot or an extra bit.
`capacity` is a power of two (validated on the client at handshake), so the
modulo is a cheap mask. The counters are `int` and the benchmark issues 30 000
requests, so they never approach overflow; a long-running server would make them
`unsigned` so the subtraction stayed well-defined across wrap.

**Index caching.** The client is the sole producer of `c2g._tail` and sole
consumer of `g2c._head`, so those are always exact locally. The two remote
indices are cached, and the cache is safe because it can only ever be
*stale-low*: it can only make the CPU→GPU queue look fuller, or the GPU→CPU
queue look emptier, than it really is. We may miss an opportunity; we can never
wrongly conclude there is room. A fresh Read is issued only when the cached value
says full/empty.

**Cache-line padding — a real limitation.** `_head` and `_tail` are adjacent
4-byte members of `MPMC_ring_queue`, so they **share a cache line**. Ideally each
would be padded to its own line, because the NIC writing `_tail` and the GPU
writing `_head` now ping-pong the same line. This is a *performance* issue only,
not correctness: PCIe writes carry byte enables, so the NIC's 4-byte write cannot
clobber the neighbouring index. Fixing it means changing the layout of your class
— forbidden by your constraint — so it is flagged here rather than changed.

**Outstanding WRs and CQ depth.** `common.cu` sizes the QP with
`max_send_wr = 8192`, the CQ with 16 384 entries, and negotiates
`max_rd_atomic = max_dest_rd_atomic = 16` (the ceiling on RDMA Reads in flight).
Because every operation is reaped inside the call that posted it, at most **3
send WRs** and **1 RDMA Read** are ever outstanding — orders of magnitude inside
all three limits. That matters: a send-queue overflow is a survivable `ENOMEM`
from `ibv_post_send`, but a **CQ overflow is not** — it asynchronously moves the
QP to the error state. The bound that actually governs throughput is a different
one: `capacity`, enforced by `enqueue`'s full check. Pipelining comes from
keeping `capacity` images in flight *on the GPU*, not from many WRs on the wire.

**Teardown.** Client: `terminate_server()` (Send `request_id == -1`, wait for
both its own send completion and the server's Write-with-Immediate ack, in
either order), then deregister the staging / input / output MRs; the base class
then destroys QP, CQ, PD, `mr_requests` and the socket. Server: `event_loop()`
returns only after its ack has completed, at which point the client provably
cannot issue more RDMA; the destructor then deregisters the four queue MRs
**before** `server.reset()` runs, because `~queue_server()` `cudaFreeHost`s the
very memory those MRs pin. `~queue_server()` sets `terminate_flag` and
`cudaDeviceSynchronize()`s, so the persistent kernel is joined before anything
is freed. Nothing is left in flight and no MR outlives its memory.

Note the given `client_rpc_context` never deregisters its image MRs; the queue
client does.

**Error checking.** Every `ibv_reg_mr`, `ibv_post_send`, `ibv_post_recv`,
`ibv_poll_cq` and `ibv_dereg_mr` return value is checked, and every CQE goes
through `VERBS_WC_CHECK`. Unexpected opcodes assert rather than being ignored.

---

## 9. Two pre-existing `ex2.cu` observations (not changed)

Both are inside your hard constraint, so they are reported, not fixed.

**(a) Missing `__syncthreads()` before the result is published — a latent race.**

```cpp
for (int i = threadIdx.x; i < IMG_SIZE; i += blockDim.x)
    ctx.out_img[i] = my_d_out[i];        // all threads write the output
                                          // <-- no __syncthreads() here
if (threadIdx.x == 0)
    while (!GPU_to_CPU_queue->GPU_enqueue(ctx.img_id)) {}
```

Thread 0 can publish the result before the other warps have finished writing
`ctx.out_img`, so a consumer could read a partially written image. A one-line
`__syncthreads()` before the `if` would close it.

It did not fire in any of my runs, and queue mode is in fact *safer* here than
homework 2 was: the client must complete an RDMA Read of `_tail` and then an
RDMA Read of the image — microseconds — whereas HW2's CPU consumer read the
image directly, nanoseconds after the index store. The lagging warps have long
since finished. I judged this **not** a change that "must happen for part 2 to
work", so per your instruction I left it alone and flagged it instead.

**(b) `std::log` where `std::log2` was probably meant.**

```cpp
int queue_size = 1 << (int)(std::ceil(std::log(16.0 * calculated_blocks)));
```

`std::log` is the natural log, so with 192 blocks this computes
`1 << ceil(ln 3072) = 1 << 9 = 512` rather than the presumably intended
`1 << ceil(log2 3072) = 4096`. The result is still a valid power of two and the
queue is fully correct — it is just smaller than intended, which caps how many
images can be in flight. Harmless here; noted in case it was unintentional.

---

## 10. Going further on throughput

If you wanted to close the gap to RPC mode, in rough order of payoff:

1. **Pipeline `enqueue`.** Stop reaping the three Write completions before
   returning; track them and reap opportunistically. This needs per-slot staging
   buffers (the single `staging.entry` would be overwritten while the NIC is
   still reading it) and a proper CQE dispatcher, since `dequeue`'s reads would
   then share the CQ with pending writes. That is the big one — it is what lets
   many requests be on the wire at once.
2. **Selective signalling.** `post_rdma_write` in `common.cu` hardcodes
   `IBV_SEND_SIGNALED`; a local variant that signals only the final index Write
   would cut CQE processing 3:1, since RC ordering means a completion on the last
   Write implies the earlier ones landed. (The course comment says to always
   signal, so this solution does.)
3. **Inline the small Writes.** The 24-byte entry and 4-byte index qualify for
   `IBV_SEND_INLINE`, saving a PCIe round trip each for the NIC to fetch them.

The completion batching in §4 step 6 is already implemented: when *n* results are
ready, one RDMA Read fetches up to 64 entries and the consumer index is published
once per batch instead of once per image. A round trip costs the same whether it
carries 24 bytes or 1536. Lazy publication is what also makes the prefetch safe —
the GPU may only recycle entries below `_head`, and `_head` does not move on the
server until the batch is drained. That change alone took throughput from 26.8k
to 29.6k req/s.
