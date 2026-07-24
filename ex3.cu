/* CUDA 10.2 has a bug that prevents including <cuda/atomic> from two separate
 * object files. As a workaround, we include ex2.cu directly here. */
#include "ex2.cu"

#include <cassert>
#include <vector>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <infiniband/verbs.h>

class server_rpc_context : public rdma_server_context {
private:
    std::unique_ptr<queue_server> gpu_context;

public:
    explicit server_rpc_context(uint16_t tcp_port) : rdma_server_context(tcp_port),
        gpu_context(create_queues_server(256))
    {
    }

    virtual void event_loop() override
    {
        /* so the protocol goes like this:
         * 1. we'll wait for a CQE indicating that we got an Send request from the client.
         *    this tells us we have new work to do. The wr_id we used in post_recv tells us
         *    where the request is.
         * 2. now we send an RDMA Read to the client to retrieve the request.
         *    we will get a completion indicating the read has completed.
         * 3. we process the request on the GPU.
         * 4. upon completion, we send an RDMA Write with immediate to the client with
         *    the results.
         */
        rpc_request* req;
        uchar *img_in;
        uchar *img_out;

        bool terminate = false, got_last_cqe = false;

        while (!terminate || !got_last_cqe) {
            // Step 1: Poll for CQE
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) {
		VERBS_WC_CHECK(wc);

                switch (wc.opcode) {
                case IBV_WC_RECV:
                    /* Received a new request from the client */
                    req = &requests[wc.wr_id];
                    img_in = &images_in[wc.wr_id * IMG_SZ];

                    /* Terminate signal */
                    if (req->request_id == -1) {
                        printf("Terminating...\n");
                        terminate = true;
                        goto send_rdma_write;
                    }

                    /* Step 2: send RDMA Read to client to read the input */
                    post_rdma_read(
                        img_in,             // local_src
                        req->input_length,  // len
                        mr_images_in->lkey, // lkey
                        req->input_addr,    // remote_dst
                        req->input_rkey,    // rkey
                        wc.wr_id);          // wr_id
                    break;

                case IBV_WC_RDMA_READ:
                    /* Completed RDMA read for a request */
                    req = &requests[wc.wr_id];
                    img_in = &images_in[wc.wr_id * IMG_SZ];
                    img_out = &images_out[wc.wr_id * IMG_SZ];

                    // Step 3: Process on GPU
                    while(!gpu_context->enqueue(wc.wr_id, img_in, img_out)){};
		    break;
                    
                case IBV_WC_RDMA_WRITE:
                    /* Completed RDMA Write - reuse buffers for receiving the next requests */
                    post_recv(wc.wr_id % OUTSTANDING_REQUESTS);

		    if (terminate)
			got_last_cqe = true;

                    break;
                default:
                    printf("Unexpected completion\n");
                    assert(false);
                }
            }

            // Dequeue completed GPU tasks
            int dequeued_img_id;
            if (gpu_context->dequeue(&dequeued_img_id)) {
                req = &requests[dequeued_img_id];
                img_out = &images_out[dequeued_img_id * IMG_SZ];

send_rdma_write:
                // Step 4: Send RDMA Write with immediate to client with the response
		post_rdma_write(
                    req->output_addr,                       // remote_dst
                    terminate ? 0 : req->output_length,     // len
                    req->output_rkey,                       // rkey
                    terminate ? 0 : img_out,                // local_src
                    mr_images_out->lkey,                    // lkey
                    dequeued_img_id + OUTSTANDING_REQUESTS, // wr_id
                    (uint32_t*)&req->request_id);           // immediate
            }
        }
    }
};

class client_rpc_context : public rdma_client_context {
private:
    uint32_t requests_sent = 0;
    uint32_t send_cqes_received = 0;

    struct ibv_mr *mr_images_in; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
public:
    explicit client_rpc_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
    }

    ~client_rpc_context()
    {
        kill();
    }

    virtual void set_input_images(uchar *images_in, size_t bytes) override
    {
        /* register a memory region for the input images. */
        mr_images_in = ibv_reg_mr(pd, images_in, bytes, IBV_ACCESS_REMOTE_READ);
        if (!mr_images_in) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        /* register a memory region for the output images. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
    }

    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        if (requests_sent - send_cqes_received == OUTSTANDING_REQUESTS)
            return false;

        struct ibv_sge sg; /* scatter/gather element */
        struct ibv_send_wr wr; /* WQE */
        struct ibv_send_wr *bad_wr; /* ibv_post_send() reports bad WQEs here */

        /* step 1: send request to server using Send operation */
        
        struct rpc_request *req = &requests[requests_sent % OUTSTANDING_REQUESTS];
        req->request_id = img_id;
        req->input_rkey = img_in ? mr_images_in->rkey : 0;
        req->input_addr = (uintptr_t)img_in;
        req->input_length = IMG_SZ;
        req->output_rkey = img_out ? mr_images_out->rkey : 0;
        req->output_addr = (uintptr_t)img_out;
        req->output_length = IMG_SZ;

        /* RDMA send needs a gather element (local buffer)*/
        memset(&sg, 0, sizeof(struct ibv_sge));
        sg.addr = (uintptr_t)req;
        sg.length = sizeof(*req);
        sg.lkey = mr_requests->lkey;

        /* WQE */
        memset(&wr, 0, sizeof(struct ibv_send_wr));
        wr.wr_id = img_id; /* helps identify the WQE */
        wr.sg_list = &sg;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED; /* always set this in this excersize. generates CQE */

        /* post the WQE to the HCA to execute it */
        if (ibv_post_send(qp, &wr, &bad_wr)) {
            perror("ibv_post_send() failed");
            exit(1);
        }

        ++requests_sent;

        return true;
    }

    virtual bool dequeue(int *img_id) override
    {
        /* When WQE is completed we expect a CQE */
        /* We also expect a completion of the RDMA Write with immediate operation from the server to us */
        /* The order between the two is not guarenteed */

        struct ibv_wc wc; /* CQE */
        int ncqes = ibv_poll_cq(cq, 1, &wc);
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        if (ncqes == 0)
            return false;

	VERBS_WC_CHECK(wc);

        switch (wc.opcode) {
        case IBV_WC_SEND:
            ++send_cqes_received;
            return false;
        case IBV_WC_RECV_RDMA_WITH_IMM:
            *img_id = wc.imm_data;
            break;
        default:
            printf("Unexpected completion type\n");
            assert(0);
        }

        /* step 2: post receive buffer for the next RPC call (next RDMA write with imm) */
        post_recv();

        return true;
    }

    void kill()
    {
        while (!enqueue(-1, // Indicate termination
                       NULL, NULL)) ;
        int img_id = 0;
        bool dequeued;
        do {
            dequeued = dequeue(&img_id);
        } while (!dequeued || img_id != -1);
    }
};

/* ============================================================================
 *                                 QUEUE MODE
 * ============================================================================
 *
 * In RPC mode (above) the server's CPU sits squarely on the data path: it
 * receives a request, RDMA-Reads the image, feeds the GPU, and RDMA-Writes the
 * answer back. In queue mode the server's CPU steps out of the way completely.
 * After the handshake it does nothing but wait for a "terminate" message. The
 * *client* drives the CPU->GPU and GPU->CPU producer-consumer queues from
 * ex2.cu over RDMA, and the persistent kernel picks the work up on its own.
 *
 * Server-side memory map (every byte of it is pinned host memory, registered
 * with the NIC, and simultaneously visible to the GPU over PCIe):
 *
 *   images_in [OUTSTANDING_REQUESTS][IMG_SZ]  <- client RDMA-Writes input here
 *   images_out[OUTSTANDING_REQUESTS][IMG_SZ]  -> GPU writes results, client Reads
 *
 *   CPU_to_GPU_queue  object : _head  (consumer index, advanced by the GPU)
 *                              _tail  (producer index, advanced by the client)
 *                     slots[] : context entries, written by the client
 *
 *   GPU_to_CPU_queue  object : _head  (consumer index, advanced by the client)
 *                              _tail  (producer index, advanced by the GPU)
 *                     slots[] : context entries, img_id written by the GPU
 *
 * WHY THE INDICES MUST LIVE IN HOST MEMORY THE NIC CAN REACH
 * ----------------------------------------------------------
 * The NIC can only DMA into memory that has been registered with it via
 * ibv_reg_mr(), and ibv_reg_mr() pins ordinary *host* pages -- plain device
 * memory from cudaMalloc() is not reachable this way without GPUDirect. At the
 * same time the persistent kernel has to poll those very same indices from the
 * GPU. ex2.cu allocates both the MPMC_ring_queue objects and their slot arrays
 * with cudaMallocHost(), i.e. pinned host memory, which is exactly the
 * intersection of the two requirements: registrable by the NIC *and* mapped
 * into the GPU's address space so the kernel can read it over PCIe. That is
 * why queue mode needs no change to ex2.cu -- the memory was already in the
 * right place.
 *
 * HOW THE GPU SEES THE HOST WRITES
 * --------------------------------
 * ex2.cu declares the indices as cuda::atomic<int, cuda::thread_scope_system>.
 * The "system" scope is the important part: it makes the GPU's loads and
 * stores coherent with the rest of the machine (CPU *and* third-party DMA such
 * as our NIC) rather than merely with other threads on the device. Concretely
 * the compiler emits system-scope acquire/release PTX for these accesses, so
 * the polling kernel re-reads the index from host memory instead of spinning
 * forever on a value cached in the GPU's L1/L2.
 *
 * The NIC's update of an index is a 4-byte, naturally aligned DMA write, which
 * is single-copy-atomic at the hardware level; the GPU's system-scope acquire
 * load therefore observes either the old value or the new one, never a torn
 * mixture. (Formally the NIC's write is not a C++ atomic store, but a plain
 * aligned 32-bit write is indivisible on this hardware, and this "write the
 * data, then write the index" idiom is precisely what every RDMA queue
 * protocol is built on.)
 */

/* The persistent kernel is launched with the same block size the RPC server
 * uses, so both modes exercise the identical ex2.cu configuration. */
#define QUEUE_SERVER_THREADS 256

/* ex2.cu's indices are cuda::atomic<int>; the client updates them with plain
 * 4-byte RDMA Writes, so the wrapper must not add padding or a lock word. */
static_assert(sizeof(cuda::atomic<int, cuda::thread_scope_system>) == sizeof(int32_t),
              "queue mode RDMA-writes the queue indices as plain 4-byte integers");

/* Everything the client needs in order to operate the server's GPU queues on
 * its own. This travels over the very same TCP socket that carries
 * connection_establishment_data, immediately after it, as part of one
 * handshake. (The assignment marks ex3.h "DO NOT CHANGE", so the existing
 * connection_establishment_data struct cannot be extended in place; the PDF
 * explicitly points at send_over_socket/recv_over_socket for these parameters.
 * See SOLUTION.md.) */
struct queue_connection_data {
    /* --- CPU -> GPU queue: the client is the producer --- */
    uint64_t c2g_head_addr;     /* &_head: consumer index, advanced by the GPU    */
    uint64_t c2g_tail_addr;     /* &_tail: producer index, advanced by the client */
    uint32_t c2g_indices_rkey;  /* rkey of the MR covering the queue object       */
    uint64_t c2g_slots_addr;    /* base of context slots[capacity]                */
    uint32_t c2g_slots_rkey;

    /* --- GPU -> CPU queue: the client is the consumer --- */
    uint64_t g2c_head_addr;     /* &_head: consumer index, advanced by the client */
    uint64_t g2c_tail_addr;     /* &_tail: producer index, advanced by the GPU    */
    uint32_t g2c_indices_rkey;
    uint64_t g2c_slots_addr;
    uint32_t g2c_slots_rkey;

    /* --- image staging buffers on the server --- */
    uint64_t images_in_addr;
    uint32_t images_in_rkey;
    uint64_t images_out_addr;
    uint32_t images_out_rkey;

    /* --- geometry --- */
    uint32_t capacity;      /* queue depth, a power of two (see ex2.cu)      */
    uint32_t num_slots;     /* number of image staging slots on the server   */
    uint32_t img_size;      /* IMG_SZ                                        */
    uint32_t context_size;  /* sizeof(struct context) -- layout sanity check */
};

class server_queues_context : public rdma_server_context {
private:
    std::unique_ptr<queue_server> server;

    /* Memory regions exposing the two GPU queues to the client. The queue
     * object and its slot array are two separate cudaMallocHost() allocations
     * in ex2.cu, so each queue needs two MRs. */
    ibv_mr *mr_c2g_indices = nullptr;
    ibv_mr *mr_c2g_slots   = nullptr;
    ibv_mr *mr_g2c_indices = nullptr;
    ibv_mr *mr_g2c_slots   = nullptr;

    /* ibv_reg_mr() + error check, so every registration below stays one line. */
    ibv_mr *register_region(void *addr, size_t length, int access, const char *what)
    {
        ibv_mr *mr = ibv_reg_mr(pd, addr, length, access);
        if (!mr) {
            fprintf(stderr, "ibv_reg_mr() failed for %s: %s\n", what, strerror(errno));
            exit(1);
        }
        return mr;
    }

public:
    /* Note the initialisation order: rdma_server_context first (TCP accept,
     * verbs resources, images_in/images_out and their MRs), then the queue
     * server -- which allocates the queues and launches the persistent kernel.
     * The kernel starts spinning on two empty queues while we register the MRs
     * below, which is harmless: it simply finds nothing to dequeue until the
     * client publishes its first producer index. */
    explicit server_queues_context(uint16_t tcp_port) :
        rdma_server_context(tcp_port),
        server(create_queues_server(QUEUE_SERVER_THREADS))
    {
        MPMC_ring_queue *c2g = server->cpu_to_gpu_queue();
        MPMC_ring_queue *g2c = server->gpu_to_cpu_queue();

        /* --- Register the queue memory for remote access -------------------
         * Access flags follow exactly who touches what:
         *   c2g indices : client Reads _head and Writes _tail  -> READ|WRITE
         *   c2g slots   : client Writes context entries        -> WRITE
         *   g2c indices : client Reads _tail and Writes _head  -> READ|WRITE
         *   g2c slots   : client Reads the completed img_id    -> READ
         * IBV_ACCESS_REMOTE_WRITE requires IBV_ACCESS_LOCAL_WRITE, hence the
         * pairing. We register the whole MPMC_ring_queue object rather than the
         * two index words individually: they are adjacent members, one MR is
         * cheaper, and the client only ever addresses the exact offsets we hand
         * it below. */
        mr_c2g_indices = register_region(
            c2g, sizeof(MPMC_ring_queue),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE,
            "CPU->GPU queue indices");
        mr_c2g_slots = register_region(
            c2g->slots_addr(), c2g->slots_bytes(),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
            "CPU->GPU queue slots");
        mr_g2c_indices = register_region(
            g2c, sizeof(MPMC_ring_queue),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE,
            "GPU->CPU queue indices");
        mr_g2c_slots = register_region(
            g2c->slots_addr(), g2c->slots_bytes(),
            IBV_ACCESS_REMOTE_READ,
            "GPU->CPU queue slots");

        /* --- Hand the client everything it needs ------------------------- */
        queue_connection_data data = {};

        data.c2g_head_addr    = (uintptr_t)c2g->head_addr();
        data.c2g_tail_addr    = (uintptr_t)c2g->tail_addr();
        data.c2g_indices_rkey = mr_c2g_indices->rkey;
        data.c2g_slots_addr   = (uintptr_t)c2g->slots_addr();
        data.c2g_slots_rkey   = mr_c2g_slots->rkey;

        data.g2c_head_addr    = (uintptr_t)g2c->head_addr();
        data.g2c_tail_addr    = (uintptr_t)g2c->tail_addr();
        data.g2c_indices_rkey = mr_g2c_indices->rkey;
        data.g2c_slots_addr   = (uintptr_t)g2c->slots_addr();
        data.g2c_slots_rkey   = mr_g2c_slots->rkey;

        /* images_in/images_out and their MRs already exist in the base class:
         * the RPC server uses them as GPU staging buffers, and queue mode uses
         * them for exactly the same purpose -- only now the client fills and
         * drains them itself instead of asking the server's CPU to do it. */
        data.images_in_addr   = (uintptr_t)images_in;
        data.images_in_rkey   = mr_images_in->rkey;
        data.images_out_addr  = (uintptr_t)images_out;
        data.images_out_rkey  = mr_images_out->rkey;

        data.capacity     = c2g->get_capacity();
        data.num_slots    = OUTSTANDING_REQUESTS;
        data.img_size     = IMG_SZ;
        data.context_size = sizeof(struct context);

        /* Both queues are constructed with the same capacity in ex2.cu; the
         * client assumes one value for both, so make that explicit. */
        assert(c2g->get_capacity() == g2c->get_capacity());

        send_over_socket(&data, sizeof(data));

        printf("Queue server ready: capacity=%u, %u image slots, %u B/image\n",
               data.capacity, data.num_slots, data.img_size);
    }

    ~server_queues_context()
    {
        /* Order matters. By the time event_loop() returned, the client had
         * already received our termination acknowledgement, so it can no longer
         * issue RDMA against us: it is safe to deregister first. We must
         * deregister *before* ~queue_server() runs, because that destructor
         * cudaFreeHost()s the very memory these MRs pin. */
        if (mr_c2g_indices && ibv_dereg_mr(mr_c2g_indices))
            perror("ibv_dereg_mr() failed for CPU->GPU queue indices");
        if (mr_c2g_slots && ibv_dereg_mr(mr_c2g_slots))
            perror("ibv_dereg_mr() failed for CPU->GPU queue slots");
        if (mr_g2c_indices && ibv_dereg_mr(mr_g2c_indices))
            perror("ibv_dereg_mr() failed for GPU->CPU queue indices");
        if (mr_g2c_slots && ibv_dereg_mr(mr_g2c_slots))
            perror("ibv_dereg_mr() failed for GPU->CPU queue slots");

        /* Stops the persistent kernel (sets terminate_flag, then
         * cudaDeviceSynchronize) and frees the queues. */
        server.reset();
    }

    virtual void event_loop() override
    {
        /* A much simpler loop than the RPC one, because the server CPU is not
         * on the data path at all: the client operates the queues by itself and
         * the persistent kernel consumes them. The only message we ever expect
         * is the termination request, which we answer exactly the way the RPC
         * server answers request_id == -1. */
        bool terminate = false, got_last_cqe = false;

        while (!terminate || !got_last_cqe) {
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes == 0)
                continue;

            VERBS_WC_CHECK(wc);

            switch (wc.opcode) {
            case IBV_WC_RECV: {
                rpc_request *req = &requests[wc.wr_id];
                if (req->request_id != -1) {
                    printf("Unexpected request %d in queue mode\n", req->request_id);
                    assert(false);
                    break;
                }
                printf("Terminating...\n");
                terminate = true;

                /* Zero-length RDMA Write with immediate: no remote memory is
                 * touched, so the (zero) rkey and address are never validated.
                 * All the client cares about is the completion it generates --
                 * the same acknowledgement trick server_rpc_context uses. */
                post_rdma_write(req->output_addr,       // remote_dst
                                0,                      // len
                                req->output_rkey,       // rkey
                                nullptr,                // local_src
                                mr_images_out->lkey,    // lkey
                                wc.wr_id,               // wr_id
                                (uint32_t *)&req->request_id); // immediate
                break;
            }

            case IBV_WC_RDMA_WRITE:
                /* Our acknowledgement left the NIC; nothing else is in flight. */
                got_last_cqe = true;
                break;

            default:
                printf("Unexpected completion (opcode %d) in queue mode\n", wc.opcode);
                assert(false);
            }
        }
    }
};

class client_queues_context : public rdma_client_context {
private:
    /* ---- Outstanding work and QP limits -----------------------------------
     * common.cu sizes the QP with max_send_wr = OUTSTANDING_REQUESTS (8192)
     * and the CQ with 2*OUTSTANDING_REQUESTS entries, and connect_qp()
     * negotiates max_rd_atomic = max_dest_rd_atomic = 16 -- the ceiling on
     * RDMA Reads in flight on the wire.
     *
     * This client stays far inside all three limits. Every RDMA operation it
     * posts is reaped before the enqueue()/dequeue() that posted it returns,
     * so at most three send WRs and at most one RDMA Read are ever
     * outstanding. The send queue therefore cannot overflow (ibv_post_send
     * would report ENOMEM), and -- more importantly -- the CQ cannot be
     * overrun, which is not a graceful failure at all: it asynchronously moves
     * the QP to the error state.
     *
     * The bound that actually governs throughput is a different one: the depth
     * of the GPU queue itself, `capacity`, enforced by enqueue()'s full check.
     * The pipelining that matters comes from keeping `capacity` images in
     * flight *on the GPU*, not from many RDMA operations in flight on the
     * wire. */

    /* Parameters received from the server during the handshake. */
    queue_connection_data srv = {};
    int capacity = 0;               /* queue depth; a power of two */

    /* ---- Our view of the two queues ---------------------------------------
     * These are monotonically increasing counters, not wrapped indices: slot k
     * of a queue lives at k % capacity, exactly as ex2.cu indexes it. Keeping
     * them unwrapped is what lets `tail - head` mean "occupancy" and so
     * distinguishes a full queue (tail - head == capacity) from an empty one
     * (tail == head) -- the two states a bare wrapped index cannot tell apart
     * without burning an extra slot or an extra bit. Because capacity is a
     * power of two, the k % capacity mapping is a cheap mask and stays correct
     * across wraparound of the physical slot ring. (The counters themselves are
     * ints and the benchmark issues 30000 requests, so they never come close to
     * overflowing; a long-running server would make them unsigned so that the
     * subtraction stayed well-defined on wrap.) */
    int c2g_tail = 0;        /* CPU->GPU producer index: ours, always exact  */
    int c2g_head_cached = 0; /* CPU->GPU consumer index: the GPU's, cached   */
    int g2c_head = 0;        /* GPU->CPU consumer index: ours, always exact  */
    int g2c_tail_cached = 0; /* GPU->CPU producer index: the GPU's, cached   */

    /* How many GPU->CPU entries we prefetch in one RDMA Read. See dequeue(). */
    static const int MAX_COMPLETION_BATCH = 64;

    /* Local staging area: the source of every index/entry we Write and the
     * destination of every index/entry we Read. One MR covers all of it. */
    struct {
        int32_t index_write;         /* value pushed into a remote index    */
        int32_t index_read;          /* value pulled from a remote index    */
        struct context entry;        /* queue entry pushed to the server    */
        /* Prefetched GPU->CPU queue entries. Only their img_id fields are
         * meaningful (GPU_enqueue writes nothing else). */
        struct context batch[MAX_COMPLETION_BATCH];
    } staging = {};
    struct ibv_mr *mr_staging = nullptr;

    /* Cursor into staging.batch: [batch_pos, batch_count) are completions we
     * have fetched but not yet handed back to the caller. */
    int batch_count = 0;
    int batch_pos = 0;

    struct ibv_mr *mr_images_in = nullptr;  /* Memory region for input images */
    struct ibv_mr *mr_images_out = nullptr; /* Memory region for output images */

    /* What we must remember about a request while the GPU works on it. The
     * GPU echoes back a single int, so we send it the *slot number* and keep
     * the caller's real img_id and destination pointer here, indexed by slot.
     * This mirrors how the RPC server uses wc.wr_id to index requests[]. */
    struct inflight_request {
        int    img_id;   /* the id the caller gave us            */
        uchar *out;      /* where the caller wants the result    */
    };
    std::vector<inflight_request> inflight;
    std::vector<uint32_t> free_slots;   /* stack of free server staging slots */

    /* wr_id values. Completions are matched positionally (we reap exactly what
     * we post), so these exist purely to make a CQE readable in a debugger. */
    enum {
        WR_READ_INDEX = 1, WR_WRITE_INDEX, WR_WRITE_IMAGE,
        WR_WRITE_ENTRY, WR_READ_ENTRY, WR_READ_IMAGE, WR_TERMINATE,
    };

    /* Block until exactly one completion is reaped, and check it succeeded. */
    struct ibv_wc wait_for_cqe()
    {
        struct ibv_wc wc;
        int ncqes;
        do {
            ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
        } while (ncqes == 0);
        VERBS_WC_CHECK(wc);
        return wc;
    }

    /* Reap `count` completions, all of which must carry `opcode`. */
    void wait_for_completions(int count, enum ibv_wc_opcode opcode)
    {
        for (int i = 0; i < count; ++i) {
            struct ibv_wc wc = wait_for_cqe();
            if (wc.opcode != opcode) {
                printf("Unexpected completion opcode %d (expected %d)\n",
                       wc.opcode, opcode);
                assert(false);
            }
        }
    }

    /* RDMA-Read a single 32-bit queue index off the server. */
    int32_t read_remote_index(uint64_t remote_addr, uint32_t rkey)
    {
        post_rdma_read(&staging.index_read, sizeof(staging.index_read),
                       mr_staging->lkey, remote_addr, rkey, WR_READ_INDEX);
        wait_for_completions(1, IBV_WC_RDMA_READ);
        return staging.index_read;
    }

    /* RDMA-Write a single 32-bit queue index to the server. */
    void write_remote_index(uint64_t remote_addr, uint32_t rkey, int32_t value)
    {
        staging.index_write = value;
        post_rdma_write(remote_addr, sizeof(staging.index_write), rkey,
                        &staging.index_write, mr_staging->lkey, WR_WRITE_INDEX);
        wait_for_completions(1, IBV_WC_RDMA_WRITE);
    }

    /* Ask the server to shut down, using the RPC protocol's "poison pill". */
    void terminate_server()
    {
        struct rpc_request *req = &requests[0];
        memset(req, 0, sizeof(*req));
        req->request_id = -1;   /* the agreed termination marker */

        struct ibv_sge sg = {};
        sg.addr   = (uintptr_t)req;
        sg.length = sizeof(*req);
        sg.lkey   = mr_requests->lkey;

        struct ibv_send_wr wr = {}, *bad_wr;
        wr.wr_id      = WR_TERMINATE;
        wr.sg_list    = &sg;
        wr.num_sge    = 1;
        wr.opcode     = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED;

        /* A Send (not a Write) because this is the one place the server's CPU
         * genuinely has to be *notified*: an RDMA Write would land silently in
         * its memory with nobody polling for it, whereas a Send consumes a
         * receive WQE and produces a CQE in the server's event_loop. */
        if (ibv_post_send(qp, &wr, &bad_wr)) {
            perror("ibv_post_send() failed for termination request");
            exit(1);
        }

        /* Wait both for our Send to complete and for the server's zero-length
         * RDMA Write with immediate acknowledging it. The order between the
         * two is not guaranteed, so accept them in either order. */
        bool got_send = false, got_ack = false;
        while (!got_send || !got_ack) {
            struct ibv_wc wc = wait_for_cqe();
            switch (wc.opcode) {
            case IBV_WC_SEND:               got_send = true; break;
            case IBV_WC_RECV_RDMA_WITH_IMM: got_ack  = true; break;
            default:
                printf("Unexpected completion opcode %d while terminating\n", wc.opcode);
                assert(false);
            }
        }
    }

public:
    client_queues_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
        /* Second half of the handshake, on the same TCP socket that just
         * carried connection_establishment_data. */
        recv_over_socket(&srv, sizeof(srv));

        /* We build context entries locally and RDMA-Write them into the
         * server's queue, so both sides must agree byte-for-byte on the
         * layout. Both are compiled from the same ex2.cu, so this only ever
         * catches an ABI surprise -- but it is cheap insurance against silently
         * corrupting the GPU's queue. */
        if (srv.context_size != sizeof(struct context) || srv.img_size != IMG_SZ) {
            printf("Server/client layout mismatch: context %u vs %zu, image %u vs %u\n",
                   srv.context_size, sizeof(struct context), srv.img_size, (unsigned)IMG_SZ);
            exit(1);
        }
        /* ex2.cu documents capacity as a power of two, and both its index
         * arithmetic and ours rely on it. */
        if (srv.capacity == 0 || (srv.capacity & (srv.capacity - 1)) != 0) {
            printf("Queue capacity %u is not a power of two\n", srv.capacity);
            exit(1);
        }
        capacity = (int)srv.capacity;

        /* IBV_ACCESS_LOCAL_WRITE because RDMA Reads land in this buffer. */
        mr_staging = ibv_reg_mr(pd, &staging, sizeof(staging), IBV_ACCESS_LOCAL_WRITE);
        if (!mr_staging) {
            perror("ibv_reg_mr() failed for staging area");
            exit(1);
        }

        /* Free list of server-side image staging slots. There are far more of
         * them (OUTSTANDING_REQUESTS) than can ever be in flight: the pipeline
         * is bounded by `capacity` entries queued to the GPU, plus the blocks
         * currently processing, plus `capacity` results waiting for us. The
         * pool therefore never runs dry in practice, so enqueue() returns false
         * only for the reason the protocol intends -- a full CPU->GPU queue --
         * but we still check, because silently reusing a live slot would
         * corrupt an image the GPU is mid-way through reading. */
        inflight.resize(srv.num_slots);
        free_slots.reserve(srv.num_slots);
        for (uint32_t i = srv.num_slots; i-- > 0; )
            free_slots.push_back(i);

        printf("Queue client ready: capacity=%d, %u image slots\n", capacity, srv.num_slots);
    }

    ~client_queues_context()
    {
        /* By the time we get here client.cu has dequeued every request it
         * enqueued, so nothing is in flight: each RDMA operation was completed
         * inside the call that posted it, so the CQ is drained and both queues
         * are empty. Tell the server to stop, and wait for its acknowledgement
         * before tearing anything down. */
        terminate_server();

        if (mr_staging && ibv_dereg_mr(mr_staging))
            perror("ibv_dereg_mr() failed for staging area");
        if (mr_images_in && ibv_dereg_mr(mr_images_in))
            perror("ibv_dereg_mr() failed for input images");
        if (mr_images_out && ibv_dereg_mr(mr_images_out))
            perror("ibv_dereg_mr() failed for output images");
        /* The QP, CQ, PD, mr_requests and the socket are released by
         * ~rdma_context(), which runs next. */
    }

    virtual void set_input_images(uchar *images_in, size_t bytes) override
    {
        /* In queue mode the client *pushes* images, so this buffer is only ever
         * the local source of an RDMA Write. A local source needs no remote
         * access rights at all -- contrast RPC mode, which registers the same
         * buffer IBV_ACCESS_REMOTE_READ because there it is the server that
         * reaches in and reads it. Granting no remote rights here is not just
         * tidiness: it means a buggy or hostile peer cannot touch our input. */
        mr_images_in = ibv_reg_mr(pd, images_in, bytes, IBV_ACCESS_LOCAL_WRITE);
        if (!mr_images_in) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        /* Destination of our own RDMA Reads, hence IBV_ACCESS_LOCAL_WRITE and,
         * again, no remote rights -- unlike RPC mode, which needs
         * IBV_ACCESS_REMOTE_WRITE here because the server writes results into
         * it. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
    }

    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        /* Need somewhere on the server to stage this image. */
        if (free_slots.empty())
            return false;

        /* ---- Step 1: is there room in the CPU->GPU queue? -----------------
         * We are the queue's only producer, so c2g_tail is exact. The consumer
         * index belongs to the GPU, so our cached copy can only be stale-*low*
         * -- it can only make the queue look fuller than it really is. That
         * asymmetry is what makes caching safe: we may miss an opportunity, but
         * we can never conclude "there is room" when there is none. So we spend
         * an RDMA Read only when the cached value says the queue is full, and
         * then re-check with the fresh value. */
        if (c2g_tail - c2g_head_cached == capacity) {
            c2g_head_cached = read_remote_index(srv.c2g_head_addr, srv.c2g_indices_rkey);
            if (c2g_tail - c2g_head_cached == capacity)
                return false;   /* genuinely full: the caller will retry */
        }

        uint32_t slot = free_slots.back();
        free_slots.pop_back();
        inflight[slot].img_id = img_id;
        inflight[slot].out    = img_out;

        /* ---- Step 2: copy the image into the server's staging buffer ----- */
        post_rdma_write(srv.images_in_addr + (uint64_t)slot * IMG_SZ, // remote_dst
                        IMG_SZ,                                      // len
                        srv.images_in_rkey,                          // rkey
                        img_in,                                      // local_src
                        mr_images_in->lkey,                          // lkey
                        WR_WRITE_IMAGE);

        /* ---- Step 3: write the queue entry the GPU will dequeue ----------
         * The pointers inside it are *server* virtual addresses: the persistent
         * kernel dereferences ctx.in_img / ctx.out_img directly over PCIe, so
         * they must name the server's staging buffers. Handing it a client
         * address would make the GPU read whatever happens to live at that
         * address on the server.
         *
         * img_id carries the slot number rather than the caller's id, because a
         * slot is all we need to find everything else again on completion. */
        staging.entry.in_img  = (uchar *)(uintptr_t)(srv.images_in_addr  + (uint64_t)slot * IMG_SZ);
        staging.entry.out_img = (uchar *)(uintptr_t)(srv.images_out_addr + (uint64_t)slot * IMG_SZ);
        staging.entry.img_id  = (int)slot;

        post_rdma_write(srv.c2g_slots_addr
                            + (uint64_t)(c2g_tail % capacity) * sizeof(struct context),
                        sizeof(struct context), srv.c2g_slots_rkey,
                        &staging.entry, mr_staging->lkey, WR_WRITE_ENTRY);

        /* ---- Step 4: publish the work by advancing the producer index ---- */
        staging.index_write = c2g_tail + 1;
        post_rdma_write(srv.c2g_tail_addr, sizeof(staging.index_write),
                        srv.c2g_indices_rkey, &staging.index_write,
                        mr_staging->lkey, WR_WRITE_INDEX);

        /*  ORDERING: why step 4 can never be observed before steps 2 and 3
         *  ---------------------------------------------------------------
         *  All three Writes are posted, in this order, on the same Reliable
         *  Connection QP. For an RC QP the InfiniBand spec requires the
         *  responder to execute requests in PSN order -- that is, strictly in
         *  the order they were posted to the send queue. So the image bytes and
         *  the queue entry are placed in the server's memory before the new
         *  producer index is. The responder HCA then issues those DMA writes
         *  towards the host in that same order, and posted writes on a single
         *  PCIe path are not reordered, so the GPU cannot observe a bumped
         *  _tail while the slot it points at still holds a stale entry.
         *
         *  This is the classic "write the payload, then write the flag" idiom,
         *  and it is why no explicit fence is needed. IBV_SEND_FENCE would not
         *  help even if we wanted it: a fence orders a WR against previously
         *  posted RDMA *Read* and Atomic operations, not against Writes, which
         *  are already ordered among themselves.
         *
         *  The other half of the argument lives on the consumer side, in
         *  ex2.cu: GPU_dequeue() loads _tail with cuda::memory_order_acquire.
         *  That acquire prevents the GPU from hoisting its reads of
         *  queue[head % capacity] -- and hence of the pixels -- above the index
         *  check, so a kernel that has seen the new _tail is guaranteed to see
         *  everything written before it. Ordering on the wire plus an acquire
         *  at the consumer is what makes the handoff sound end to end.
         *
         *  We deliberately do not wait *between* the three posts: ordering is a
         *  property of the queue, not of our polling. We do reap all three
         *  completions before returning, which bounds outstanding work at three
         *  WRs and guarantees the NIC has finished reading the staging buffers
         *  before the next enqueue() overwrites them. */
        wait_for_completions(3, IBV_WC_RDMA_WRITE);

        ++c2g_tail;
        return true;
    }

    virtual bool dequeue(int *img_id) override
    {
        /* ---- Steps 5 and 6: has the GPU published a result, and which? ----
         * Symmetric to enqueue: we are the only consumer, so g2c_head is
         * exact, and a cached producer index can only be stale-low -- it can
         * only make the queue look emptier than it is, never fuller. So we
         * re-read it only when the cache says "empty".
         *
         * This is also the only thing we ever block on: we wait for our own
         * RDMA Read to complete, which the assignment explicitly permits, but
         * never for the GPU to finish a particular image. If nothing is ready
         * we return false at once and the caller goes off to enqueue more work
         * -- which is exactly what keeps the GPU pipeline full.
         *
         * When results *are* ready we fetch up to MAX_COMPLETION_BATCH queue
         * entries in a single RDMA Read and serve the following dequeue() calls
         * straight out of that local copy. A round trip costs the same whether
         * it carries 24 bytes or 1536, so this amortises both the producer-index
         * read and the entry read across a whole batch instead of paying three
         * serialised round trips per image.
         *
         * The prefetch is safe precisely because we publish the consumer index
         * lazily (step 8): the GPU may only overwrite entries below _head, and
         * _head does not move on the server until we have drained the batch. So
         * the entries we copied cannot be recycled underneath us. */
        if (batch_pos == batch_count) {
            if (g2c_head == g2c_tail_cached) {
                g2c_tail_cached = read_remote_index(srv.g2c_tail_addr, srv.g2c_indices_rkey);
                if (g2c_head == g2c_tail_cached)
                    return false;   /* nothing has completed yet */
            }

            /* Fetch as much as is ready, but stop at the end of the ring (a
             * single RDMA Read is one contiguous range, and slot k lives at
             * k % capacity, so a batch that wraps would need two reads). */
            int ring_pos  = g2c_head % capacity;
            int available = g2c_tail_cached - g2c_head;
            int n = available;
            if (n > capacity - ring_pos)     n = capacity - ring_pos;
            if (n > MAX_COMPLETION_BATCH)    n = MAX_COMPLETION_BATCH;

            post_rdma_read(staging.batch, n * sizeof(struct context),
                           mr_staging->lkey,
                           srv.g2c_slots_addr + (uint64_t)ring_pos * sizeof(struct context),
                           srv.g2c_slots_rkey, WR_READ_ENTRY);
            wait_for_completions(1, IBV_WC_RDMA_READ);

            batch_count = n;
            batch_pos   = 0;
        }

        /* ex2.cu's GPU_enqueue() stores only img_id into the result slot; the
         * in_img/out_img fields of the entry are leftovers from whichever
         * request last occupied it, so img_id is the only field we may trust. */
        uint32_t slot = (uint32_t)staging.batch[batch_pos].img_id;
        ++batch_pos;
        if (slot >= srv.num_slots) {
            printf("Corrupt completion: slot %u out of range\n", slot);
            exit(1);
        }

        /* ---- Step 7: pull the result straight into the caller's buffer --- */
        post_rdma_read(inflight[slot].out, IMG_SZ, mr_images_out->lkey,
                       srv.images_out_addr + (uint64_t)slot * IMG_SZ,
                       srv.images_out_rkey, WR_READ_IMAGE);
        wait_for_completions(1, IBV_WC_RDMA_READ);

        /* ---- Step 8: release the queue entries ---------------------------
         * Advancing _head tells the GPU it may reuse those *queue slots*. It
         * has to come after the reads above, because the moment the GPU sees
         * the new index it is free to overwrite the entries. We waited for each
         * read's completion before posting this write, so the two are strictly
         * ordered by construction and need no fence. (Had we pipelined them,
         * this is precisely where IBV_SEND_FENCE would be required -- a fence
         * is what orders a Write against a preceding RDMA Read.)
         *
         * We publish once per batch rather than once per image. That is safe in
         * both directions: holding the index back only ever makes the queue look
         * *fuller* to the GPU, so it can never cause the GPU to overwrite a live
         * entry -- it can at worst make the GPU wait. And since a batch is at
         * most MAX_COMPLETION_BATCH entries out of `capacity` (64 of 512 here),
         * the effective queue depth the GPU sees stays comfortably large. */
        ++g2c_head;
        if (batch_pos == batch_count)
            write_remote_index(srv.g2c_head_addr, srv.g2c_indices_rkey, g2c_head);

        /* ---- Step 9: recycle the image staging slot ----------------------
         * Only now is `slot` genuinely free, and tying its reuse to the arrival
         * of the *result* rather than to the CPU->GPU consumer index is what
         * makes that true. Look at the persistent kernel in ex2.cu: it calls
         * GPU_dequeue() -- which already advances the CPU->GPU _head -- and
         * only afterwards copies ctx.in_img into device memory. So a slot whose
         * c2g entry has been consumed may still be actively under the GPU's
         * nose. Waiting for the result to surface in the GPU->CPU queue proves
         * the kernel is finished with both in_img and out_img for this request,
         * which is the only safe moment to hand the slot out again. */
        *img_id = inflight[slot].img_id;
        free_slots.push_back(slot);
        return true;
    }
};

std::unique_ptr<rdma_server_context> create_server(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<server_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<server_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
}

std::unique_ptr<rdma_client_context> create_client(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<client_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<client_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
    
}
