///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <time.h>


#define IMG_HEIGHT 512
#define IMG_WIDTH 512
#define TILE_WIDTH_LOG2 6
#define TILE_WIDTH 64
#define TILE_HALF_WIDTH (TILE_WIDTH / 2)
#define TILE_COUNT (IMG_WIDTH / TILE_WIDTH)
#define N_IMAGES 1000


typedef unsigned char uchar;

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

double static inline get_time_msec(void) {
    struct timespec t;
    int res = clock_gettime(CLOCK_MONOTONIC, &t);
    if (res) {
        perror("clock_gettime failed");
        exit(1);
    }
    return t.tv_sec * 1e+3 + t.tv_nsec * 1e-6;
}

void cpu_process(uchar *img_in, uchar *img_out, int width, int height);

struct task_serial_context;

struct task_serial_context *task_serial_init();
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out);
void task_serial_free(struct task_serial_context *context);

struct gpu_bulk_context;

struct gpu_bulk_context *gpu_bulk_init();
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out);
void gpu_bulk_free(struct gpu_bulk_context *context);

///////////////////////////////////////////////////////////////////////////////////////////////////////////

