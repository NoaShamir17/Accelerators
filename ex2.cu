/* This file should be almost identical to ex2.cu from homework 2. */
/* once the TODOs in this file are complete, the RPC version of the server/client should work correctly. */

#include "ex3.h"
#include "ex2.h"
#include <cuda/atomic>


#define NUM_THREADS 1024
#define NUM_THREADS_PER_TILE 256
#define IMG_SIZE (IMG_HEIGHT * IMG_WIDTH)

__device__ void prefix_sum(int arr[], int arr_size) {
    // TODO complete according to hw1
    int increment = 0;
    int in_tile_tid = threadIdx.x % NUM_THREADS_PER_TILE;
    for(int stride = 1; stride <= arr_size/2; stride *= 2){
        if(in_tile_tid >= stride){
            increment = arr[in_tile_tid] + arr[in_tile_tid - stride];
        }
        __syncthreads();
        if(in_tile_tid >= stride){
            arr[in_tile_tid] = increment;
        }
        __syncthreads();

    }
    return; 
}

__device__ void build_histogram(int hist[], uchar all_in[IMG_HEIGHT][IMG_WIDTH], 
                                int tile_start_pixel_row, int tile_start_pixel_col){
    int in_tile_tid = threadIdx.x % NUM_THREADS_PER_TILE;

    hist[in_tile_tid] = 0;
    __syncthreads();

    int row;
    int col;
    for(int stride = 0; stride < TILE_WIDTH * TILE_WIDTH; stride += NUM_THREADS_PER_TILE){
        row = tile_start_pixel_row + (in_tile_tid + stride) / TILE_WIDTH;
        col = tile_start_pixel_col + in_tile_tid % TILE_WIDTH; //stride is a multiply of TILE_WIDTH bc NUM_THREADS_PER_TILE = k * TILE_WIDTH
        atomicAdd(&hist[all_in[row][col]], 1);
    }

    __syncthreads();

}

__device__ void calc_m_v(uchar maps_3d_array[TILE_COUNT][TILE_COUNT][256], int *CDF, int tile_row, int tile_col){
    int in_tile_tid = threadIdx.x % NUM_THREADS_PER_TILE;
    maps_3d_array[tile_row][tile_col][in_tile_tid] = CDF[in_tile_tid] * 255  /  (TILE_WIDTH * TILE_WIDTH);
    __syncthreads();
}

/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__
 void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

__device__
void process_image(uchar *in, uchar *out, uchar* maps) {
    // TODO complete according to hw1
    __shared__ int hist[TILE_COUNT*TILE_COUNT][256]; //shared for atomic add
    int *CDF;

    int stride = blockDim.x / NUM_THREADS_PER_TILE;
    // IMG_WIDTH=128 / TILE_WIDTH=64 - > 2 tiles per row, 2 tiles per column, 4 tiles in total
    // 256 thread per tile, 4 tiles in total -> 1024 threads per block
    for (int tile_idx = threadIdx.x / NUM_THREADS_PER_TILE; tile_idx < TILE_COUNT * TILE_COUNT; tile_idx += stride) {

        int tile_row = tile_idx / TILE_COUNT;
        int tile_col = tile_idx % TILE_COUNT;
        int tile_start_pixel_row = tile_row * TILE_WIDTH;
        int tile_start_pixel_col = tile_col * TILE_WIDTH;
        
        build_histogram(hist[tile_idx], (uchar (*)[IMG_WIDTH])in, tile_start_pixel_row, tile_start_pixel_col);
        CDF = hist[tile_idx];
        prefix_sum(CDF, 256);

        calc_m_v((uchar (*)[TILE_COUNT][256])maps, CDF, tile_row, tile_col);
    }
    
    
    interpolate_device(maps, in, out);
    return; 
}


__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar* maps)
{
    process_image(all_in, all_out, maps);
}


// TODO implement a lock

class TTAS_lock
{
private:
    cuda::atomic<bool, cuda::thread_scope_device>* _lock;

public:
    TTAS_lock(){

        CUDA_CHECK(cudaMalloc((void**)&_lock, sizeof(cuda::atomic<bool, cuda::thread_scope_device>)));
        CUDA_CHECK(cudaMemset(_lock, 0, sizeof(cuda::atomic<bool, cuda::thread_scope_device>)));
    }

    __device__ void lock(){
        while(true){

            while(_lock->load(cuda::memory_order_relaxed) == true);

            if(_lock->exchange(true,cuda::memory_order_acquire)==false){
                return;
            }
        }
    }

    __device__ void unlock(){
        _lock->store(false, cuda::memory_order_release);
    }
};

// TODO implement a MPMC queue
struct context{
    uchar *in_img;
    uchar *out_img;
    int img_id;
};

class MPMC_ring_queue
{
private:
    TTAS_lock gpu_lock;
    cuda::atomic<int, cuda::thread_scope_system> _head;
    cuda::atomic<int, cuda::thread_scope_system> _tail;
    int capacity; //we assume is a power of 2
    struct context *queue;

public:
    MPMC_ring_queue(int capacity) : capacity(capacity) {
        CUDA_CHECK(cudaMallocHost((void**)&queue, capacity * sizeof(struct context))); //pinned memory allocation
        _head.store(0, cuda::memory_order_relaxed);
        _tail.store(0, cuda::memory_order_relaxed);

    }

    ~MPMC_ring_queue() {
        CUDA_CHECK(cudaFreeHost(queue));
    }

    __host__ bool CPU_enqueue(uchar *in_img, uchar *out_img, int img_id){
        

        int tail = _tail.load(cuda::memory_order_relaxed);
        if (tail - _head.load(cuda::memory_order_acquire) == capacity) //queue is full
        {
            return false;
        }
        queue[_tail % capacity].in_img = in_img;
        queue[_tail % capacity].out_img = out_img;
        queue[_tail % capacity].img_id = img_id;
        _tail.store(tail + 1, cuda::memory_order_release);
        
        
        return true;
    }

    __device__ bool GPU_enqueue(int img_id){
        
        gpu_lock.lock();

        int tail = _tail.load(cuda::memory_order_relaxed);
        if (tail - _head.load(cuda::memory_order_acquire) == capacity) //queue is full
        {
            gpu_lock.unlock();
            return false;
        }
        queue[_tail % capacity].img_id = img_id;
        _tail.store(tail + 1, cuda::memory_order_release);
        
        
        gpu_lock.unlock();
        return true;
    }


    __host__  bool CPU_dequeue(int *img_id) {
        int head = _head.load(cuda::memory_order_relaxed);
        if (head == _tail.load(cuda::memory_order_acquire)) //queue is empty
        {
            return false;
        }
        *img_id = queue[head % capacity].img_id;
        _head.store(head + 1, cuda::memory_order_release);
        return true;
    }

    __device__  bool GPU_dequeue(struct context *ctx) {
        gpu_lock.lock();
        int head = _head.load(cuda::memory_order_relaxed);
        if (head == _tail.load(cuda::memory_order_acquire)) //queue is empty
        {
            gpu_lock.unlock();
            return false;
        }
        *ctx = queue[head % capacity];
        _head.store(head + 1, cuda::memory_order_release);
        gpu_lock.unlock();
        return true;
    }


};

// TODO implement the persistent kernel
__global__ void persistent_kernel(volatile bool *terminate_flag, MPMC_ring_queue *CPU_to_GPU_queue,
                                     MPMC_ring_queue *GPU_to_CPU_queue, uchar* maps_array){
    __shared__ struct context ctx;
    __shared__ bool dequeue_success;
    while(!*terminate_flag){
        if(threadIdx.x == 0){
            
            dequeue_success = CPU_to_GPU_queue->GPU_dequeue(&ctx);
            if(dequeue_success){
                //printf("GPU dequeued image with ID: %d\n", ctx.img_id);
            }
            

        }

        __syncthreads();

        if(!dequeue_success){
            continue;
        }

        process_image(ctx.in_img, ctx.out_img, maps_array + blockIdx.x * TILE_COUNT * TILE_COUNT * 256);

        __syncthreads();

        if(threadIdx.x == 0){
            //printf("Attempting to enqueue result for image with ID: %d\n", ctx.img_id);
            while(!GPU_to_CPU_queue->GPU_enqueue(ctx.img_id)){} //wait until we can enqueue the result
            //printf("GPU enqueued result for image with ID: %d\n", ctx.img_id);


        }
        
        __syncthreads();


    }
}   
// TODO implement a function for calculating the threadblocks count
int calculate_max_threadblocks(int threads_per_block, size_t shared_mem_per_block, int regs_per_thread) {
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    //printf("Device: %s, SMs: %d, Max threads per SM: %d, Shared mem per SM: %zu, Regs per SM: %d\n", prop.name, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor, prop.sharedMemPerMultiprocessor, prop.regsPerMultiprocessor);

    // 1. Thread Limit
    int limit_threads = prop.maxThreadsPerMultiProcessor / threads_per_block;

    // 2. Shared Memory Limit 
    int limit_shmem = prop.sharedMemPerMultiprocessor / shared_mem_per_block;

    // 3. Register Limit
    int regs_per_block = threads_per_block * regs_per_thread;
    int limit_regs = prop.regsPerMultiprocessor / regs_per_block;
    //printf("Limit threads: %d, Limit shmem: %d, Limit regs: %d\n", limit_threads, limit_shmem, limit_regs);

    // Find the tightest hardware bottleneck per SM
    int max_blocks_per_SM = std::min(limit_threads, std::min(limit_shmem, limit_regs));

    // Multiply by total SMs to get total active blocks for the whole GPU grid
    return max_blocks_per_SM * prop.multiProcessorCount;
}

class queue_server : public image_processing_server
{
private:
    
    // TODO define queue server context (memory buffers, etc...)
    
    //context
    //uchar **in_img_array;
    //uchar **out_img_array;
    uchar *maps_array;
    
    MPMC_ring_queue *CPU_to_GPU_queue;
    MPMC_ring_queue *GPU_to_CPU_queue;
    volatile bool *terminate_flag;

public:
    queue_server(int threads)
    {
        // TODO initialize host state
        // 5120 bytes is our combined shared memory size, 32 is our register cap
        int calculated_blocks = calculate_max_threadblocks(threads, 5120, 32);
        printf("Calculated max threadblocks: %d\n", calculated_blocks);
        int queue_size = 1 << (int)(std::ceil(std::log(16.0 * calculated_blocks)));
        printf("Queue size (next power of 2): %d\n", queue_size);


        //allocate the context arrays in pinned memory
        // CUDA_CHECK(cudaMalloc((void**)&in_img_array, calculated_blocks * sizeof(uchar*)));
        // CUDA_CHECK(cudaMalloc((void**)&out_img_array, calculated_blocks * sizeof(uchar*)));
        CUDA_CHECK(cudaMalloc((void**)&maps_array, calculated_blocks * TILE_COUNT * TILE_COUNT * 256 * sizeof(uchar)));

        
        printf("Allocated maps arrays in pinned memory.\n");

        //allocate the queues and the terminate flag in pinned memory
        CUDA_CHECK(cudaMallocHost((void**)&CPU_to_GPU_queue, sizeof(MPMC_ring_queue)));
        CUDA_CHECK(cudaMallocHost((void**)&GPU_to_CPU_queue, sizeof(MPMC_ring_queue)));
        printf("Allocated queues in pinned memory.\n");

        new (CPU_to_GPU_queue) MPMC_ring_queue(queue_size);
        new (GPU_to_CPU_queue) MPMC_ring_queue(queue_size);

        printf("Initialized queues with size: %d\n", queue_size);

        CUDA_CHECK(cudaMallocHost((void**)&terminate_flag, sizeof(volatile bool)));
        *terminate_flag = false;
        printf("Queue server initialized with %d threads and %d threadblocks.\n", threads, calculated_blocks);
        persistent_kernel<<<calculated_blocks, threads>>>(terminate_flag, CPU_to_GPU_queue, GPU_to_CPU_queue, maps_array);
        printf("Persistent kernel launched with %d threads and %d threadblocks.\n", threads, calculated_blocks);
        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
    }

    ~queue_server() override
    {
        //terminate the persistent kernel
        *terminate_flag = true;
        // TODO wait for the persistent kernel to finish
        CUDA_CHECK(cudaDeviceSynchronize());
        // TODO free resources allocated in constructor
        CPU_to_GPU_queue->~MPMC_ring_queue();
        GPU_to_CPU_queue->~MPMC_ring_queue();
        CUDA_CHECK(cudaFreeHost(CPU_to_GPU_queue));
        CUDA_CHECK(cudaFreeHost(GPU_to_CPU_queue));
        CUDA_CHECK(cudaFreeHost((void*)terminate_flag));
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO push new task into queue if possible
        bool res = CPU_to_GPU_queue->CPU_enqueue(img_in, img_out, img_id);
        //printf("Enqueued image with ID: %d res: %s\n", img_id, res ? "true" : "false");
        return res;
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        bool res = GPU_to_CPU_queue->CPU_dequeue(img_id);
        //printf("Dequeued image with ID: %d res: %s\n", *img_id, res ? "true" : "false");
        return res;
    }
};


std::unique_ptr<queue_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
