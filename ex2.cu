#include "ex2.h"
#include <cuda/atomic>

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

    // IMG_WIDTH=128 / TILE_WIDTH=64 - > 2 tiles per row, 2 tiles per column, 4 tiles in total
    // 256 thread per tile, 4 tiles in total -> 1024 threads per block
    int tile_idx = threadIdx.x / 256;

    int tile_row = tile_idx / TILE_COUNT;
    int tile_col = tile_idx % TILE_COUNT;
    int tile_start_pixel_row = tile_row * TILE_WIDTH;
    int tile_start_pixel_col = tile_col * TILE_WIDTH;
    
    build_histogram(hist[tile_idx], (uchar (*)[IMG_WIDTH])in, tile_start_pixel_row, tile_start_pixel_col);
    CDF = hist[tile_idx];
    prefix_sum(CDF, 256);

    calc_m_v((uchar (*)[TILE_COUNT][256])maps, CDF, tile_row, tile_col);

    
    
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

public:
    streams_server()
    {
        // TODO initialize context (memory buffers, streams, etc...)
        //initialize streams, allocate memory buffers, etc...
        
    }

    ~streams_server() override
    {
        // TODO free resources allocated in constructor
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        // TODO place memory transfers and kernel invocation in streams if possible.
        return false;
    }

    bool dequeue(int *img_id) override
    {
        return false;

        // TODO query (don't block) streams for any completed requests.
        //for ()
        //{
            cudaError_t status = cudaStreamQuery(0); // TODO query diffrent stream each iteration
            switch (status) {
            case cudaSuccess:
                // TODO return the img_id of the request that was completed.
                //*img_id = ...
                return true;
            case cudaErrorNotReady:
                return false;
            default:
                CUDA_CHECK(status);
                return false;
            }
        //}
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

// TODO implement a lock
// TODO implement a MPMC queue
// TODO implement the persistent kernel
// TODO implement a function for calculating the threadblocks count

class queue_server : public image_processing_server
{
private:
    // TODO define queue server context (memory buffers, etc...)
public:
    queue_server(int threads)
    {
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
