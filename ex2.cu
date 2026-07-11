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

__global__
void process_image_kernel(uchar *in, uchar *out, uchar* maps){
    process_image(in, out, maps);
}

class streams_server : public image_processing_server
{
private:
    // TODO define stream server context (memory buffers, streams, etc...)

    cudaStream_t streams[STREAM_COUNT];
    bool is_occupied[STREAM_COUNT];
    int img_id_array[STREAM_COUNT];
    int next_checked_stream;

    //context
    uchar *in_img_array[STREAM_COUNT];
    uchar *out_img_array[STREAM_COUNT];
    uchar *maps_array[STREAM_COUNT];






public:
    streams_server()
    {
        // TODO initialize context (memory buffers, streams, etc...)
        //initialize streams, allocate memory buffers, etc...
        next_checked_stream = 0;
        for(int i = 0; i < STREAM_COUNT; i++){
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            is_occupied[i] = false;

            CUDA_CHECK(cudaMalloc((void**) &in_img_array[i], IMG_SIZE));
            CUDA_CHECK(cudaMalloc((void**) &out_img_array[i], IMG_SIZE));
            CUDA_CHECK(cudaMalloc((void**) &maps_array[i], TILE_COUNT * TILE_COUNT * 256));
        }


        
    }

    ~streams_server() override
    {
        // TODO free resources allocated in constructor
        for(int i = 0; i < STREAM_COUNT; i++){
            CUDA_CHECK(cudaStreamDestroy(streams[i]));

            CUDA_CHECK(cudaFree(in_img_array[i]));
            CUDA_CHECK(cudaFree(out_img_array[i]));
            CUDA_CHECK(cudaFree(maps_array[i]));
        }
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO place memory transfers and kernel invocation in streams if possible.
        for(int stream = 0; stream < STREAM_COUNT; stream++){
            if(is_occupied[stream] == false){

                CUDA_CHECK(cudaMemcpyAsync(in_img_array[stream], img_in, IMG_SIZE, cudaMemcpyHostToDevice,streams[stream]));
                process_image_kernel<<<1, NUM_THREADS,0,streams[stream]>>>(in_img_array[stream], out_img_array[stream], maps_array[stream]);
                CUDA_CHECK(cudaMemcpyAsync(img_out, out_img_array[stream], IMG_SIZE, cudaMemcpyDeviceToHost,streams[stream]));
                is_occupied[stream] = true;
                img_id_array[stream] = img_id;
                
                return true;

            }
            
        }
        return false;
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) streams for any completed requests.
        for (int i = 0; i < STREAM_COUNT; i++)
        {
            if(is_occupied[next_checked_stream] == false){
                next_checked_stream = (next_checked_stream + 1) % STREAM_COUNT;
                continue;
            }
            cudaError_t status = cudaStreamQuery(streams[next_checked_stream]); // TODO query diffrent stream each iteration
            switch (status) {
            case cudaSuccess:
                *img_id = img_id_array[next_checked_stream];
                is_occupied[next_checked_stream] = false;
                next_checked_stream = (next_checked_stream + 1) % STREAM_COUNT;
                return true;
            case cudaErrorNotReady:
                next_checked_stream = (next_checked_stream + 1) % STREAM_COUNT;
                continue;
            default:
                CUDA_CHECK(status);
                next_checked_stream = (next_checked_stream + 1) % STREAM_COUNT;
                return false;
            }
        }
        return false;
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}


// TODO implement a lock

class TTAS_lock
{
private:
    cuda::atomic<bool> lock;

public:
    __device__ void lock(){
        while(true){

            while(lock.load(cuda::memory_order_relaxed) == true);

            if(lock.exchange(false,cuda::memory_order_acquire)==false){
                return;
            }
        }
    }

    __device__ void unlock(){
        lock.store(false, cuda::memory_order_release);
    }
};

// TODO implement a MPMC queue
struct context{
    uchar *in_img;
    uchar *out_img;
    uchar *maps;
};

class MPMC_ring_queue
{
private:
    TTAS_lock producer_lock;
    TTAS_lock consumer_lock;
    cuda::atomic<int> head;
    cuda::atomic<int> tail;
    int capacity;
    int *queue;

public:
    MPMC_ring_queue(int capacity) : capacity(capacity) {
        queue = new int[capacity];
        head.store(0, cuda::memory_order_relaxed);
        tail.store(0, cuda::memory_order_relaxed);
    }

    ~MPMC_ring_queue() {
        delete[] queue;
    }

    bool enqueue(){}

    bool enqueue(int value) {
        producer_lock.lock();
        int current_tail = tail.load(cuda::memory_order_relaxed);
        int next_tail = (current_tail + 1) % capacity;

        if (next_tail == head.load(cuda::memory_order_acquire)) {
            producer_lock.unlock();
            return false; // Queue is full
        }

        queue[current_tail] = value;
        tail.store(next_tail, cuda::memory_order_release);
        producer_lock.unlock();
        return true;
    }

    bool dequeue(int *value) {
        consumer_lock.lock();
        int current_head = head.load(cuda::memory_order_relaxed);

        if (current_head == tail.load(cuda::memory_order_acquire)) {
            consumer_lock.unlock();
            return false; // Queue is empty
        }

        *value = queue[current_head];
        head.store((current_head + 1) % capacity, cuda::memory_order_release);
        consumer_lock.unlock();
        return true;
    }


};

// TODO implement the persistent kernel
// TODO implement a function for calculating the threadblocks count
int calculate_max_threadblocks(int threads_per_block, size_t shared_mem_per_block, int regs_per_thread) {
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    printf("Device: %s, SMs: %d, Max threads per SM: %d, Shared mem per SM: %zu, Regs per SM: %d\n", prop.name, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor, prop.sharedMemPerMultiprocessor, prop.regsPerMultiprocessor);

    // 1. Thread Limit
    int limit_threads = prop.maxThreadsPerMultiProcessor / threads_per_block;

    // 2. Shared Memory Limit 
    int limit_shmem = prop.sharedMemPerMultiprocessor / shared_mem_per_block;

    // 3. Register Limit
    int regs_per_block = threads_per_block * regs_per_thread;
    int limit_regs = prop.regsPerMultiprocessor / regs_per_block;
    printf("Limit threads: %d, Limit shmem: %d, Limit regs: %d\n", limit_threads, limit_shmem, limit_regs);

    // Find the tightest hardware bottleneck per SM
    int max_blocks_per_SM = std::min(limit_threads, std::min(limit_shmem, limit_regs));

    // Multiply by total SMs to get total active blocks for the whole GPU grid
    return max_blocks_per_SM * prop.multiProcessorCount;
}

class queue_server : public image_processing_server
{
private:
    
    // TODO define queue server context (memory buffers, etc...)
public:
    queue_server(int threads)
    {
        // Calculate how many blocks can concurrently run based on the user-requested thread count
        // 5120 bytes is our combined shared memory size, 32 is our register cap
        int calculated_blocks = calculate_max_threadblocks(threads, 5120, 32);
        printf("Calculated max threadblocks: %d\n", calculated_blocks);

        // TODO initialize host state
        // TODO launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
    }

    ~queue_server() override
    {
        // TODO free resources allocated in constructor
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO push new task into queue if possible
        return false;
    }

    bool dequeue(int *img_id) override
    {
        // TODO query (don't block) the producer-consumer queue for any responses.
        return false;

        // TODO return the img_id of the request that was completed.
        //*img_id = ... 
        return true;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
