#include "ex1.h"

#define NUM_THREADS 256
#define IMG_SIZE (IMG_HEIGHT * IMG_WIDTH)

//We know the arr is of size 2^k
//We chose #threads = arr size = 256
__device__ void prefix_sum(int arr[], int arr_size) {
    
    int increment = 0;
    int tid = threadIdx.x;
    for(int stride = 1; stride <= arr_size/2; stride *= 2){
        if(tid >= stride){
            increment = arr[tid] + arr[tid - stride];
        }
        __syncthreads();
        if(tid >= stride){
            arr[tid] = increment;
        }
        __syncthreads();

    }
    return; 
}



__device__ void build_histogram(int hist[], uchar all_in[IMG_HEIGHT][IMG_WIDTH], 
                                int tile_start_pixel_row, int tile_start_pixel_col){
    int tid = threadIdx.x;
    int row;
    int col;
    for(int stride = 0; stride < TILE_WIDTH * TILE_WIDTH; stride += NUM_THREADS){
        row = tile_start_pixel_row + (tid + stride) / TILE_WIDTH;
        col = tile_start_pixel_col + tid % TILE_WIDTH; //stride is a multiply of TILE_WIDTH bc NUM_THREADS = k * TILE_WIDTH
        atomicAdd(&hist[all_in[row][col]], 1);
    }

    __syncthreads();

}

__device__ void calc_m_v(uchar maps_3d_array[TILE_COUNT][TILE_COUNT][256], int *CDF, int tile_row, int tile_col){
    int tid = threadIdx.x;
    maps_3d_array[tile_row][tile_col][tid] = CDF[tid] * 255  /  (TILE_WIDTH * TILE_WIDTH);
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


__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps) {
    int hist[256];
    int *CDF;

    int tile_row;
    int tile_col;
    int tile_start_pixel_row;
    int tile_start_pixel_col;
    
    for (int tile_idx = 0; tile_idx < TILE_COUNT*TILE_COUNT; tile_idx++){
        tile_row = tile_idx / TILE_COUNT;
        tile_col = tile_idx % TILE_COUNT;
        tile_start_pixel_row = tile_row * TILE_WIDTH;
        tile_start_pixel_col = tile_col * TILE_WIDTH;
        
        build_histogram(hist, (uchar (*)[IMG_WIDTH])all_in, tile_start_pixel_row, tile_start_pixel_col);
        CDF = hist;
        prefix_sum(CDF, 256);

        calc_m_v((uchar (*)[TILE_COUNT][256])maps, CDF, tile_row, tile_col);

    }
    
    interpolate_device(maps, all_in, all_out);
    return; 
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    uchar *d_in_img;
    uchar *d_out_img;
    uchar *d_maps; 
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;

    cudaMalloc((void**) &context->d_in_img, IMG_SIZE);
    cudaMalloc((void**) &context->d_out_img, IMG_SIZE);
    cudaMalloc((void**) &context->d_maps, TILE_COUNT * TILE_COUNT * 256);

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{

    for(int i = 0; i < N_IMAGES; i++){

        // 1. copy the relevant image from images_in to the GPU memory you allocated
        cudaMemcpy(context->d_in_img, images_in + i * IMG_SIZE, IMG_SIZE, cudaMemcpyHostToDevice);

        // 2. invoke GPU kernel on this image
        process_image_kernel<<<1, NUM_THREADS>>>(context->d_in_img, context->d_out_img, context->d_maps);

        cudaDeviceSynchronize();

        // 2.5. get error
        cudaError_t error = cudaGetLastError();
        if (error!=cudaSuccess){
            fprintf(stderr, "Kernel execution failed:%s\n", cudaGetErrorString(error));
            return;
        }

        // 3. copy output from GPU memory to relevant location in images_out_gpu_serial
        cudaMemcpy(images_out + i * IMG_SIZE, context->d_out_img, IMG_SIZE, cudaMemcpyDeviceToHost);

    }

}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    //free resources allocated in task_serial_init
    cudaFree(context->d_in_img);
    cudaFree(context->d_out_img);
    cudaFree(context->d_maps);

    free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // TODO define bulk-GPU memory buffers
};

/* Allocate GPU memory for all the input images, output images, and maps.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    //TODO: allocate GPU memory for all the input images, output images, and maps

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    //TODO: free resources allocated in gpu_bulk_init

    free(context);
}
