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

/* QUEUE MODE: the client drives ex2.cu's CPU<->GPU queues remotely; server CPU stays off the data path. */

#define QUEUE_SERVER_THREADS 256

/* Indices are RDMA-written as plain 4-byte ints. */
static_assert(sizeof(cuda::atomic<int, cuda::thread_scope_system>) == sizeof(int32_t),
              "queue mode RDMA-writes the queue indices as plain 4-byte integers");

/* Sent server->client after connection_establishment_data, same socket (ex3.h is DO NOT CHANGE). */
struct queue_connection_data {
    /* CPU -> GPU queue (client is the producer) */
    uint64_t c2g_head_addr;     /* consumer index, advanced by the GPU */
    uint64_t c2g_tail_addr;     /* producer index, advanced by us      */
    uint32_t c2g_indices_rkey;
    uint64_t c2g_slots_addr;
    uint32_t c2g_slots_rkey;

    /* GPU -> CPU queue (client is the consumer) */
    uint64_t g2c_head_addr;     /* consumer index, advanced by us  */
    uint64_t g2c_tail_addr;     /* producer index, advanced by GPU */
    uint32_t g2c_indices_rkey;
    uint64_t g2c_slots_addr;
    uint32_t g2c_slots_rkey;

    /* image staging buffers on the server */
    uint64_t images_in_addr;
    uint32_t images_in_rkey;
    uint64_t images_out_addr;
    uint32_t images_out_rkey;

    uint32_t capacity;      /* queue depth, power of two */
    uint32_t num_slots;     /* image staging slots       */
    uint32_t img_size;      /* IMG_SZ                    */
    uint32_t context_size;  /* sizeof(struct context)    */
};

class server_queues_context : public rdma_server_context {
private:
    std::unique_ptr<queue_server> server;

    /* Each queue needs two MRs: the object (_head/_tail) and the slots array. */
    ibv_mr *mr_c2g_indices = nullptr;
    ibv_mr *mr_c2g_slots   = nullptr;
    ibv_mr *mr_g2c_indices = nullptr;
    ibv_mr *mr_g2c_slots   = nullptr;

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
    explicit server_queues_context(uint16_t tcp_port) :
        rdma_server_context(tcp_port),
        server(create_queues_server(QUEUE_SERVER_THREADS))
    {
        MPMC_ring_queue *c2g = server->cpu_to_gpu_queue();
        MPMC_ring_queue *g2c = server->gpu_to_cpu_queue();

        /* Access flags follow who reads/writes what on each queue. */
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

        /* images_in/images_out and their MRs already exist in the base class. */
        data.images_in_addr   = (uintptr_t)images_in;
        data.images_in_rkey   = mr_images_in->rkey;
        data.images_out_addr  = (uintptr_t)images_out;
        data.images_out_rkey  = mr_images_out->rkey;

        data.capacity     = c2g->get_capacity();
        data.num_slots    = OUTSTANDING_REQUESTS;
        data.img_size     = IMG_SZ;
        data.context_size = sizeof(struct context);

        assert(c2g->get_capacity() == g2c->get_capacity());

        send_over_socket(&data, sizeof(data));

        printf("Queue server ready: capacity=%u, %u image slots, %u B/image\n",
               data.capacity, data.num_slots, data.img_size);
    }

    ~server_queues_context()
    {
        /* Deregister before ~queue_server() cudaFreeHost()s this memory. */
        if (mr_c2g_indices && ibv_dereg_mr(mr_c2g_indices))
            perror("ibv_dereg_mr() failed for CPU->GPU queue indices");
        if (mr_c2g_slots && ibv_dereg_mr(mr_c2g_slots))
            perror("ibv_dereg_mr() failed for CPU->GPU queue slots");
        if (mr_g2c_indices && ibv_dereg_mr(mr_g2c_indices))
            perror("ibv_dereg_mr() failed for GPU->CPU queue indices");
        if (mr_g2c_slots && ibv_dereg_mr(mr_g2c_slots))
            perror("ibv_dereg_mr() failed for GPU->CPU queue slots");

        server.reset(); /* stops the persistent kernel, frees the queues */
    }

    virtual void event_loop() override
    {
        /* Server CPU is off the data path; just wait for termination. */
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

                /* Zero-length write with immediate, just to ack the client. */
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
    /* Every RDMA op is reaped before the call that posted it returns, so we stay well inside the QP/CQ limits. */

    queue_connection_data srv = {};
    int capacity = 0;               /* queue depth; a power of two */

    /* Monotonic counters (slot k lives at k % capacity); tail - head is occupancy. */
    int c2g_tail = 0;        /* CPU->GPU producer: ours, exact  */
    int c2g_head_cached = 0; /* CPU->GPU consumer: the GPU's, cached */
    int g2c_head = 0;        /* GPU->CPU consumer: ours, exact  */
    int g2c_tail_cached = 0; /* GPU->CPU producer: the GPU's, cached */

    /* How many GPU->CPU entries we prefetch per RDMA Read. */
    static const int MAX_COMPLETION_BATCH = 64;

    /* Local staging area backing every Write/Read we issue. */
    struct {
        int32_t index_write;
        int32_t index_read;
        struct context entry;         /* queue entry pushed to the server */
        struct context batch[MAX_COMPLETION_BATCH]; /* prefetched completions */
    } staging = {};
    struct ibv_mr *mr_staging = nullptr;

    /* [batch_pos, batch_count) are completions fetched but not yet returned. */
    int batch_count = 0;
    int batch_pos = 0;

    struct ibv_mr *mr_images_in = nullptr;  /* Memory region for input images */
    struct ibv_mr *mr_images_out = nullptr; /* Memory region for output images */

    /* The GPU echoes back only a slot number, so we track the real img_id/out here. */
    struct inflight_request {
        int    img_id;
        uchar *out;
    };
    std::vector<inflight_request> inflight;
    std::vector<uint32_t> free_slots;   /* stack of free server staging slots */

    enum {
        WR_READ_INDEX = 1, WR_WRITE_INDEX, WR_WRITE_IMAGE,
        WR_WRITE_ENTRY, WR_READ_ENTRY, WR_READ_IMAGE, WR_TERMINATE,
    };

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

    int32_t read_remote_index(uint64_t remote_addr, uint32_t rkey)
    {
        post_rdma_read(&staging.index_read, sizeof(staging.index_read),
                       mr_staging->lkey, remote_addr, rkey, WR_READ_INDEX);
        wait_for_completions(1, IBV_WC_RDMA_READ);
        return staging.index_read;
    }

    void write_remote_index(uint64_t remote_addr, uint32_t rkey, int32_t value)
    {
        staging.index_write = value;
        post_rdma_write(remote_addr, sizeof(staging.index_write), rkey,
                        &staging.index_write, mr_staging->lkey, WR_WRITE_INDEX);
        wait_for_completions(1, IBV_WC_RDMA_WRITE);
    }

    /* Reuse the RPC protocol's poison pill to shut the server down. */
    void terminate_server()
    {
        struct rpc_request *req = &requests[0];
        memset(req, 0, sizeof(*req));
        req->request_id = -1;

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

        /* Send, not Write: the server's CPU must be notified here. */
        if (ibv_post_send(qp, &wr, &bad_wr)) {
            perror("ibv_post_send() failed for termination request");
            exit(1);
        }

        /* Order of our Send completion vs. the server's ack is not guaranteed. */
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
        /* Second half of the handshake. */
        recv_over_socket(&srv, sizeof(srv));

        if (srv.context_size != sizeof(struct context) || srv.img_size != IMG_SZ) {
            printf("Server/client layout mismatch: context %u vs %zu, image %u vs %u\n",
                   srv.context_size, sizeof(struct context), srv.img_size, (unsigned)IMG_SZ);
            exit(1);
        }
        if (srv.capacity == 0 || (srv.capacity & (srv.capacity - 1)) != 0) {
            printf("Queue capacity %u is not a power of two\n", srv.capacity);
            exit(1);
        }
        capacity = (int)srv.capacity;

        mr_staging = ibv_reg_mr(pd, &staging, sizeof(staging), IBV_ACCESS_LOCAL_WRITE);
        if (!mr_staging) {
            perror("ibv_reg_mr() failed for staging area");
            exit(1);
        }

        /* Free list of server-side image staging slots. */
        inflight.resize(srv.num_slots);
        free_slots.reserve(srv.num_slots);
        for (uint32_t i = srv.num_slots; i-- > 0; )
            free_slots.push_back(i);

        printf("Queue client ready: capacity=%d, %u image slots\n", capacity, srv.num_slots);
    }

    ~client_queues_context()
    {
        /* Nothing is in flight; tell the server to stop before tearing down. */
        terminate_server();

        if (mr_staging && ibv_dereg_mr(mr_staging))
            perror("ibv_dereg_mr() failed for staging area");
        if (mr_images_in && ibv_dereg_mr(mr_images_in))
            perror("ibv_dereg_mr() failed for input images");
        if (mr_images_out && ibv_dereg_mr(mr_images_out))
            perror("ibv_dereg_mr() failed for output images");
    }

    virtual void set_input_images(uchar *images_in, size_t bytes) override
    {
        /* Client only pushes these, so no remote access needed. */
        mr_images_in = ibv_reg_mr(pd, images_in, bytes, IBV_ACCESS_LOCAL_WRITE);
        if (!mr_images_in) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        /* Destination of our own Reads; no remote access needed. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
    }

    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        if (free_slots.empty())
            return false;

        /* Step 1: room in the CPU->GPU queue? Re-Read only if cache says full. */
        if (c2g_tail - c2g_head_cached == capacity) {
            c2g_head_cached = read_remote_index(srv.c2g_head_addr, srv.c2g_indices_rkey);
            if (c2g_tail - c2g_head_cached == capacity)
                return false;
        }

        uint32_t slot = free_slots.back();
        free_slots.pop_back();
        inflight[slot].img_id = img_id;
        inflight[slot].out    = img_out;

        /* Step 2: copy the image into the server's staging buffer. */
        post_rdma_write(srv.images_in_addr + (uint64_t)slot * IMG_SZ, // remote_dst
                        IMG_SZ,                                      // len
                        srv.images_in_rkey,                          // rkey
                        img_in,                                      // local_src
                        mr_images_in->lkey,                          // lkey
                        WR_WRITE_IMAGE);

        /* Step 3: write the queue entry (server pointers; img_id is the slot number). */
        staging.entry.in_img  = (uchar *)(uintptr_t)(srv.images_in_addr  + (uint64_t)slot * IMG_SZ);
        staging.entry.out_img = (uchar *)(uintptr_t)(srv.images_out_addr + (uint64_t)slot * IMG_SZ);
        staging.entry.img_id  = (int)slot;

        post_rdma_write(srv.c2g_slots_addr
                            + (uint64_t)(c2g_tail % capacity) * sizeof(struct context),
                        sizeof(struct context), srv.c2g_slots_rkey,
                        &staging.entry, mr_staging->lkey, WR_WRITE_ENTRY);

        /* Step 4: publish by advancing the producer index. */
        staging.index_write = c2g_tail + 1;
        post_rdma_write(srv.c2g_tail_addr, sizeof(staging.index_write),
                        srv.c2g_indices_rkey, &staging.index_write,
                        mr_staging->lkey, WR_WRITE_INDEX);

        /* ORDERING: RC QPs execute writes in post order, and GPU_dequeue() loads
         * _tail with memory_order_acquire, so seeing the new _tail implies seeing
         * the image and entry too -- no fence needed. See SOLUTION.md. */
        wait_for_completions(3, IBV_WC_RDMA_WRITE);

        ++c2g_tail;
        return true;
    }

    virtual bool dequeue(int *img_id) override
    {
        /* Steps 5-6: refill the local batch of completions if we've drained it. */
        if (batch_pos == batch_count) {
            if (g2c_head == g2c_tail_cached) {
                g2c_tail_cached = read_remote_index(srv.g2c_tail_addr, srv.g2c_indices_rkey);
                if (g2c_head == g2c_tail_cached)
                    return false;
            }

            /* Stop at the end of the ring; one Read is one contiguous range. */
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

        /* GPU_enqueue() only writes img_id; other fields are stale. */
        uint32_t slot = (uint32_t)staging.batch[batch_pos].img_id;
        ++batch_pos;
        if (slot >= srv.num_slots) {
            printf("Corrupt completion: slot %u out of range\n", slot);
            exit(1);
        }

        /* Step 7: pull the result into the caller's buffer. */
        post_rdma_read(inflight[slot].out, IMG_SZ, mr_images_out->lkey,
                       srv.images_out_addr + (uint64_t)slot * IMG_SZ,
                       srv.images_out_rkey, WR_READ_IMAGE);
        wait_for_completions(1, IBV_WC_RDMA_READ);

        /* Step 8: release entries; must follow the reads above. Published once per batch. */
        ++g2c_head;
        if (batch_pos == batch_count)
            write_remote_index(srv.g2c_head_addr, srv.g2c_indices_rkey, g2c_head);

        /* Step 9: recycle the slot only now that the result has arrived. */
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
