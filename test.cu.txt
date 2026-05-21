/* test.cu – Correctness tests for ex1 GPU implementations.
 *
 * Tests are organised according to the assignment requirements:
 *   Section 0 – Unit test : prefix_sum device function          (requirement 2a)
 *   Section 1 – Correctness: GPU serial & bulk must match CPU   (requirements 3/4)
 *   Section 2 – Known-output: verify m[v]=floor(CDF[v]/T²×255) analytically
 *   Section 3 – Determinism: same input → same output
 *   Section 4 – Performance: bulk must be ≥10× faster than serial (requirement 4f)
 *
 * Compile & run:  make ex1 && make -f Makefile.test test && ./test
 */

#include "ex1.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>

// ── prefix_sum is a __device__ function defined in ex1.cu.
// With relocatable device code (-dc / RDC) both TUs are linked together so
// we can call it from a test kernel after declaring it extern.
extern __device__ void prefix_sum(int arr[], int arr_size);

// Wrapper kernel: loads d_in into shared memory, calls prefix_sum, writes
// the result back to d_out.
__global__ void prefix_sum_test_kernel(int *d_in, int *d_out, int len)
{
    __shared__ int s[256];
    for (int i = threadIdx.x; i < len; i += blockDim.x)
        s[i] = d_in[i];
    __syncthreads();
    prefix_sum(s, len);
    for (int i = threadIdx.x; i < len; i += blockDim.x)
        d_out[i] = s[i];
}

// ── image generators ────────────────────────────────────────────────────────

static void fill_constant(uchar *imgs, uchar val)
{
    memset(imgs, val, (size_t)N_IMAGES * IMG_WIDTH * IMG_HEIGHT);
}

static void fill_gradient_h(uchar *imgs)
{
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img = imgs + (size_t)i * IMG_WIDTH * IMG_HEIGHT;
        for (int r = 0; r < IMG_HEIGHT; r++)
            for (int c = 0; c < IMG_WIDTH; c++)
                img[r * IMG_WIDTH + c] = (uchar)((unsigned)c * 255 / (IMG_WIDTH - 1));
    }
}

static void fill_gradient_v(uchar *imgs)
{
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img = imgs + (size_t)i * IMG_WIDTH * IMG_HEIGHT;
        for (int r = 0; r < IMG_HEIGHT; r++)
            for (int c = 0; c < IMG_WIDTH; c++)
                img[r * IMG_WIDTH + c] = (uchar)((unsigned)r * 255 / (IMG_HEIGHT - 1));
    }
}

static void fill_checkerboard(uchar *imgs)
{
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img = imgs + (size_t)i * IMG_WIDTH * IMG_HEIGHT;
        for (int r = 0; r < IMG_HEIGHT; r++)
            for (int c = 0; c < IMG_WIDTH; c++) {
                int tr = r / TILE_WIDTH, tc = c / TILE_WIDTH;
                img[r * IMG_WIDTH + c] = ((tr + tc) & 1) ? 255 : 0;
            }
    }
}

static void fill_random(uchar *imgs, unsigned seed)
{
    srand(seed);
    size_t n = (size_t)N_IMAGES * IMG_WIDTH * IMG_HEIGHT;
    for (size_t i = 0; i < n; i++)
        imgs[i] = (uchar)(rand() & 0xFF);
}

static void fill_two_values(uchar *imgs, uchar a, uchar b)
{
    size_t n = (size_t)N_IMAGES * IMG_WIDTH * IMG_HEIGHT;
    for (size_t i = 0; i < n; i++)
        imgs[i] = (i & 1) ? b : a;
}

/* Each image gets a unique constant value (image i → value i % 256).
 * This is the most important extra test: verifies blocks don't interfere. */
static void fill_mixed_constants(uchar *imgs)
{
    for (int i = 0; i < N_IMAGES; i++) {
        uchar val = (uchar)(i % 256);
        memset(imgs + (size_t)i * IMG_WIDTH * IMG_HEIGHT, val,
               IMG_WIDTH * IMG_HEIGHT);
    }
}

/* Each image gets a unique random seed – different data in every block. */
static void fill_mixed_random(uchar *imgs)
{
    for (int i = 0; i < N_IMAGES; i++) {
        srand((unsigned)i * 2654435761u); /* Knuth multiplicative hash */
        uchar *img = imgs + (size_t)i * IMG_WIDTH * IMG_HEIGHT;
        for (int p = 0; p < IMG_WIDTH * IMG_HEIGHT; p++)
            img[p] = (uchar)(rand() & 0xFF);
    }
}

/* 1-pixel-wide alternating stripes (thin = very different histogram shape). */
static void fill_stripes_1px(uchar *imgs)
{
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img = imgs + (size_t)i * IMG_WIDTH * IMG_HEIGHT;
        for (int r = 0; r < IMG_HEIGHT; r++)
            for (int c = 0; c < IMG_WIDTH; c++)
                img[r * IMG_WIDTH + c] = (uchar)(((r + c) & 1) ? 255 : 0);
    }
}

/* Value flips exactly at every tile boundary (stresses tile-edge interpolation). */
static void fill_tile_boundary_flip(uchar *imgs)
{
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img = imgs + (size_t)i * IMG_WIDTH * IMG_HEIGHT;
        for (int r = 0; r < IMG_HEIGHT; r++)
            for (int c = 0; c < IMG_WIDTH; c++) {
                int tr = r / TILE_WIDTH, tc = c / TILE_WIDTH;
                img[r * IMG_WIDTH + c] = (uchar)(((tr ^ tc) & 1) ? 200 : 50);
            }
    }
}

/* Only one pixel per tile is non-zero (sparse histogram). */
static void fill_sparse(uchar *imgs)
{
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img = imgs + (size_t)i * IMG_WIDTH * IMG_HEIGHT;
        memset(img, 0, IMG_WIDTH * IMG_HEIGHT);
        /* set the centre pixel of each tile to 200 */
        int half = TILE_WIDTH / 2;
        for (int tr = 0; tr < TILE_COUNT; tr++)
            for (int tc = 0; tc < TILE_COUNT; tc++) {
                int r = tr * TILE_WIDTH + half;
                int c = tc * TILE_WIDTH + half;
                img[r * IMG_WIDTH + c] = 200;
            }
    }
}

/* Diagonal gradient (combination of row + col index). */
static void fill_diagonal(uchar *imgs)
{
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img = imgs + (size_t)i * IMG_WIDTH * IMG_HEIGHT;
        for (int r = 0; r < IMG_HEIGHT; r++)
            for (int c = 0; c < IMG_WIDTH; c++)
                img[r * IMG_WIDTH + c] =
                    (uchar)(((unsigned)(r + c)) * 255 / ((IMG_HEIGHT - 1) + (IMG_WIDTH - 1)));
    }
}

// ── small wrappers so every fill has the same void(*)(uchar*) signature ─────

static void fill_zeros(uchar *i)      { fill_constant(i, 0);          }
static void fill_white(uchar *i)      { fill_constant(i, 255);        }
static void fill_gray(uchar *i)       { fill_constant(i, 128);        }
static void fill_grad_h(uchar *i)     { fill_gradient_h(i);           }
static void fill_grad_v(uchar *i)     { fill_gradient_v(i);           }
static void fill_chess(uchar *i)      { fill_checkerboard(i);         }
static void fill_alt_0_255(uchar *i)  { fill_two_values(i, 0, 255);   }
static void fill_alt_64_192(uchar *i) { fill_two_values(i, 64, 192);  }
static void fill_rand42(uchar *i)     { fill_random(i, 42);           }
static void fill_rand123(uchar *i)    { fill_random(i, 123);          }
static void fill_rand999(uchar *i)    { fill_random(i, 999);          }
static void fill_mixed_c(uchar *i)    { fill_mixed_constants(i);      }
static void fill_mixed_r(uchar *i)    { fill_mixed_random(i);         }
static void fill_stripes(uchar *i)    { fill_stripes_1px(i);          }
static void fill_tbflip(uchar *i)     { fill_tile_boundary_flip(i);   }
static void fill_spar(uchar *i)       { fill_sparse(i);               }
static void fill_diag(uchar *i)       { fill_diagonal(i);             }

// ── test table ───────────────────────────────────────────────────────────────

struct TestCase { const char *name; void (*fill)(uchar *); };

static const TestCase TESTS[] = {
    { "All zeros (black image)",           fill_zeros      },
    { "All 255  (white image)",            fill_white      },
    { "All 128  (mid-gray image)",         fill_gray       },
    { "Horizontal gradient  0→255",        fill_grad_h     },
    { "Vertical gradient    0→255",        fill_grad_v     },
    { "Diagonal gradient",                 fill_diag       },
    { "Checkerboard (tile-sized squares)", fill_chess      },
    { "Alternating pixels   0/255",        fill_alt_0_255  },
    { "Alternating pixels  64/192",        fill_alt_64_192 },
    { "1-pixel stripes (thin pattern)",    fill_stripes    },
    { "Tile-boundary value flip",          fill_tbflip     },
    { "Sparse (1 hot pixel per tile)",     fill_spar       },
    { "Mixed batch: unique constant/img",  fill_mixed_c    },
    { "Mixed batch: unique random/img",    fill_mixed_r    },
    { "Random (seed  42)",                 fill_rand42     },
    { "Random (seed 123)",                 fill_rand123    },
    { "Random (seed 999)",                 fill_rand999    },
};
static const int N_TESTS = (int)(sizeof(TESTS) / sizeof(TESTS[0]));

// ── helpers ──────────────────────────────────────────────────────────────────

static long long sq_distance(const uchar *a, const uchar *b, size_t n)
{
    long long d = 0;
    for (size_t i = 0; i < n; i++) {
        long long diff = (long long)a[i] - b[i];
        d += diff * diff;
    }
    return d;
}

// ── Section 0: prefix_sum unit tests ────────────────────────────────────────
// Requirement 2a: prefix_sum must compute an in-place inclusive prefix sum.
static void run_prefix_sum_tests(int &passed, int &failed)
{
    struct Case { const char *name; int in[8]; int ex[8]; int len; };
    static const Case cases[] = {
        { "all ones   [1,1,1,1,1,1,1,1]",
          {1,1,1,1,1,1,1,1}, {1,2,3,4,5,6,7,8}, 8 },
        { "powers of 2 [1,2,4,8,16,32,64,128]",
          {1,2,4,8,16,32,64,128}, {1,3,7,15,31,63,127,255}, 8 },
        { "all zeros  [0,0,0,0,0,0,0,0]",
          {0,0,0,0,0,0,0,0}, {0,0,0,0,0,0,0,0}, 8 },
        { "single element [42]",
          {42,0,0,0,0,0,0,0}, {42,0,0,0,0,0,0,0}, 1 },
        { "ascending  [1,2,3,4,5,6,7,8]",
          {1,2,3,4,5,6,7,8}, {1,3,6,10,15,21,28,36}, 8 },
    };
    const int NC = (int)(sizeof(cases) / sizeof(cases[0]));

    int *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  256 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, 256 * sizeof(int)));

    printf("=== Section 0: prefix_sum unit tests ===\n");
    for (int c = 0; c < NC; c++) {
        CUDA_CHECK(cudaMemcpy(d_in, cases[c].in,
                              cases[c].len * sizeof(int), cudaMemcpyHostToDevice));
        prefix_sum_test_kernel<<<1, 256>>>(d_in, d_out, cases[c].len);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        int res[8] = {};
        CUDA_CHECK(cudaMemcpy(res, d_out,
                              cases[c].len * sizeof(int), cudaMemcpyDeviceToHost));
        bool ok = (memcmp(res, cases[c].ex, cases[c].len * sizeof(int)) == 0);
        printf("  %-52s [%s]\n", cases[c].name, ok ? "PASS" : "FAIL");
        if (!ok) {
            printf("    expected: ");
            for (int i = 0; i < cases[c].len; i++) printf("%d ", cases[c].ex[i]);
            printf("\n    got:      ");
            for (int i = 0; i < cases[c].len; i++) printf("%d ", res[i]);
            printf("\n");
        }
        if (ok) passed++; else failed++;
    }

    /* Full 256-bin histogram: uniform distribution (16 counts per bin).
     * T=64, T²=4096; expected CDF: [16, 32, 48, ..., 4096]. */
    {
        int h_in[256], h_ex[256];
        const int per_bin = TILE_WIDTH * TILE_WIDTH / 256; /* 16 */
        for (int i = 0; i < 256; i++) h_in[i] = per_bin;
        int run = 0;
        for (int i = 0; i < 256; i++) { run += per_bin; h_ex[i] = run; }
        CUDA_CHECK(cudaMemcpy(d_in, h_in, 256 * sizeof(int), cudaMemcpyHostToDevice));
        prefix_sum_test_kernel<<<1, 256>>>(d_in, d_out, 256);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        int res[256];
        CUDA_CHECK(cudaMemcpy(res, d_out, 256 * sizeof(int), cudaMemcpyDeviceToHost));
        bool ok = (memcmp(res, h_ex, 256 * sizeof(int)) == 0);
        printf("  %-52s [%s]\n",
               "Uniform histogram 256 bins (16 each → CDF 16,32,..4096)",
               ok ? "PASS" : "FAIL");
        if (ok) passed++; else failed++;
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    printf("\n");
}

// ── Section 2: known-output tests ────────────────────────────────────────────
// For a spatially uniform image (every tile has the same pixel distribution)
// bilinear interpolation collapses to a simple map lookup because all four
// tile-centre maps are identical:
//   v' = (1-α)(1-β)m[v] + α(1-β)m[v] + (1-α)β m[v] + αβ m[v]  =  m[v]
//
// This lets us verify  m[v] = floor(CDF[v] / T² × 255)  pixel-by-pixel.
static void run_known_output_tests(
    struct task_serial_context *ts, struct gpu_bulk_context *gb,
    uchar *imgs_in, uchar *imgs_serial, uchar *imgs_bulk,
    int &passed, int &failed)
{
    const size_t total = (size_t)N_IMAGES * IMG_WIDTH * IMG_HEIGHT;
    printf("=== Section 2: known-output tests (formula m[v]=floor(CDF[v]/T^2*255)) ===\n");

    // ── Case A: constant image → every output pixel must be 255 ─────────────
    // hist[v]=T²; CDF[v']=T² for all v'≥v; m[v]=floor(T²/T²×255)=255.
    {
        const uchar VALS[] = {0, 1, 128, 254, 255};
        for (uchar val : VALS) {
            memset(imgs_in, val, total);
            task_serial_process(ts, imgs_in, imgs_serial);
            gpu_bulk_process(gb, imgs_in, imgs_bulk);

            bool ser_ok = true, bulk_ok = true;
            for (size_t i = 0; i < total && ser_ok;  i++) ser_ok  = (imgs_serial[i] == 255);
            for (size_t i = 0; i < total && bulk_ok; i++) bulk_ok = (imgs_bulk[i]   == 255);

            char name[72];
            snprintf(name, sizeof(name),
                     "Constant val=%3d → all 255", (int)val);
            bool ok = ser_ok && bulk_ok;
            printf("  %-52s [%s]\n", name, ok ? "PASS" : "FAIL");
            if (!ok) {
                if (!ser_ok)  printf("    serial: found output pixel != 255\n");
                if (!bulk_ok) printf("    bulk:   found output pixel != 255\n");
            }
            if (ok) passed++; else failed++;
        }
    }

    // ── Case B: alternating 0/255 → pixel 0→127, pixel 255→255 ─────────────
    // T=64, T²=4096. Each tile has exactly 2048 zeros + 2048 255s (T is even,
    // IMG_WIDTH is even → every tile starts at an even flat index).
    // CDF[0]=2048 → m[0]=floor(2048/4096×255)=floor(127.5)=127
    // CDF[255]=4096 → m[255]=255
    {
        for (size_t i = 0; i < total; i++)
            imgs_in[i] = (uchar)((i & 1) ? 255 : 0);

        task_serial_process(ts, imgs_in, imgs_serial);
        gpu_bulk_process(gb, imgs_in, imgs_bulk);

        bool ser_ok = true, bulk_ok = true;
        for (size_t i = 0; i < total && ser_ok;  i++) {
            uchar exp = (uchar)((i & 1) ? 255 : 127);
            ser_ok = (imgs_serial[i] == exp);
        }
        for (size_t i = 0; i < total && bulk_ok; i++) {
            uchar exp = (uchar)((i & 1) ? 255 : 127);
            bulk_ok = (imgs_bulk[i] == exp);
        }
        bool ok = ser_ok && bulk_ok;
        printf("  %-52s [%s]\n",
               "Alternating 0/255 → 127/255  (map formula)", ok ? "PASS" : "FAIL");
        if (!ok && !ser_ok) {
            for (size_t i = 0; i < total; i++) {
                uchar exp = (uchar)((i & 1) ? 255 : 127);
                if (imgs_serial[i] != exp) {
                    printf("    serial mismatch pixel %zu: in=%d exp=%d got=%d\n",
                           i, (int)imgs_in[i], (int)exp, (int)imgs_serial[i]);
                    break;
                }
            }
        }
        if (ok) passed++; else failed++;
    }

    // ── Case C: alternating 64/192 → pixel 64→127, pixel 192→255 ────────────
    // Same CDF analysis: two equally-represented values per tile.
    // CDF[64]=2048 → m[64]=127;  CDF[192]=4096 → m[192]=255.
    {
        for (size_t i = 0; i < total; i++)
            imgs_in[i] = (uchar)((i & 1) ? 192 : 64);

        task_serial_process(ts, imgs_in, imgs_serial);
        gpu_bulk_process(gb, imgs_in, imgs_bulk);

        bool ser_ok = true, bulk_ok = true;
        for (size_t i = 0; i < total && ser_ok;  i++) {
            uchar exp = (uchar)((i & 1) ? 255 : 127);
            ser_ok = (imgs_serial[i] == exp);
        }
        for (size_t i = 0; i < total && bulk_ok; i++) {
            uchar exp = (uchar)((i & 1) ? 255 : 127);
            bulk_ok = (imgs_bulk[i] == exp);
        }
        bool ok = ser_ok && bulk_ok;
        printf("  %-52s [%s]\n",
               "Alternating 64/192 → 127/255  (CDF accumulation)", ok ? "PASS" : "FAIL");
        if (ok) passed++; else failed++;
    }

    printf("\n");
}

// ── main ─────────────────────────────────────────────────────────────────────

int main()
{
    /* ---- device selection (match main.cu) ---- */
    int device_id = 3;
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (device_id >= ndev) device_id = 0;
    CUDA_CHECK(cudaSetDevice(device_id));
    printf("Using device %d\n\n", device_id);

    const size_t total = (size_t)N_IMAGES * IMG_WIDTH * IMG_HEIGHT;

    /* ---- allocate pinned host buffers ---- */
    uchar *imgs_in, *imgs_cpu, *imgs_serial, *imgs_bulk, *imgs_bulk2;
    CUDA_CHECK(cudaHostAlloc(&imgs_in,     total, 0));
    CUDA_CHECK(cudaHostAlloc(&imgs_cpu,    total, 0));
    CUDA_CHECK(cudaHostAlloc(&imgs_serial, total, 0));
    CUDA_CHECK(cudaHostAlloc(&imgs_bulk,   total, 0));
    CUDA_CHECK(cudaHostAlloc(&imgs_bulk2,  total, 0));

    /* ---- init GPU contexts once ---- */
    struct task_serial_context *ts = task_serial_init();
    struct gpu_bulk_context    *gb = gpu_bulk_init();

    int passed = 0, failed = 0;

    // ── Section 0: prefix_sum unit tests ─────────────────────────────────
    run_prefix_sum_tests(passed, failed);

    // ── Section 1: correctness ────────────────────────────────────────────
    printf("=== Correctness tests ===\n");
    printf("%-42s  %8s  %8s  %8s  %s\n",
           "Test", "ser↔cpu", "bulk↔cpu", "ser↔bulk", "Result");
    printf("%s\n", "------------------------------------------------------------------------------");

    for (int t = 0; t < N_TESTS; t++) {
        TESTS[t].fill(imgs_in);

        for (int i = 0; i < N_IMAGES; i++)
            cpu_process(imgs_in  + (size_t)i * IMG_WIDTH * IMG_HEIGHT,
                        imgs_cpu + (size_t)i * IMG_WIDTH * IMG_HEIGHT,
                        IMG_WIDTH, IMG_HEIGHT);

        task_serial_process(ts, imgs_in, imgs_serial);
        long long d_serial = sq_distance(imgs_cpu, imgs_serial, total);

        gpu_bulk_process(gb, imgs_in, imgs_bulk);
        long long d_bulk = sq_distance(imgs_cpu, imgs_bulk, total);

        long long d_sb = sq_distance(imgs_serial, imgs_bulk, total);

        bool ok = (d_serial == 0 && d_bulk == 0 && d_sb == 0);
        printf("%-42s  %8lld  %8lld  %8lld  [%s]\n",
               TESTS[t].name, d_serial, d_bulk, d_sb, ok ? "PASS" : "FAIL");
        if (ok) passed++; else failed++;
    }
    printf("%s\n\n", "------------------------------------------------------------------------------");

    // ── Section 2: known-output tests ────────────────────────────────────
    run_known_output_tests(ts, gb, imgs_in, imgs_serial, imgs_bulk, passed, failed);

    // ── Section 3: determinism – same input must give same output ─────────
    printf("=== Section 3: Determinism test (random seed 42, run twice) ===\n");
    fill_random(imgs_in, 42);
    gpu_bulk_process(gb, imgs_in, imgs_bulk);   /* run 1 */
    gpu_bulk_process(gb, imgs_in, imgs_bulk2);  /* run 2 */
    {
        long long d = sq_distance(imgs_bulk, imgs_bulk2, total);
        bool ok = (d == 0);
        printf("  bulk run1 vs run2 distance: %lld  [%s]\n\n", d, ok ? "PASS" : "FAIL");
        if (ok) passed++; else failed++;
    }

    // ── Section 3: performance – bulk must be >> faster than serial ───────
    printf("=== Section 4: Performance test (bulk must be >=10x faster than serial) ===\n");
    fill_random(imgs_in, 77);

    double t0 = get_time_msec();
    task_serial_process(ts, imgs_in, imgs_serial);
    double t_serial = get_time_msec() - t0;

    t0 = get_time_msec();
    gpu_bulk_process(gb, imgs_in, imgs_bulk);
    double t_bulk = get_time_msec() - t0;

    double speedup = t_serial / t_bulk;
    /* Bulk parallelises all N_IMAGES blocks; expect at least 10× speedup. */
    bool perf_ok = (speedup >= 10.0);
    printf("  task serial: %.1f ms  |  bulk: %.1f ms  |  speedup: %.1fx  [%s]\n\n",
           t_serial, t_bulk, speedup, perf_ok ? "PASS" : "FAIL");
    if (perf_ok) passed++; else failed++;

    // ── Summary ───────────────────────────────────────────────────────────
    printf("%d / %d tests passed.\n", passed, passed + failed);

    /* ---- cleanup ---- */
    task_serial_free(ts);
    gpu_bulk_free(gb);
    CUDA_CHECK(cudaFreeHost(imgs_in));
    CUDA_CHECK(cudaFreeHost(imgs_cpu));
    CUDA_CHECK(cudaFreeHost(imgs_serial));
    CUDA_CHECK(cudaFreeHost(imgs_bulk));
    CUDA_CHECK(cudaFreeHost(imgs_bulk2));

    return (failed > 0) ? 1 : 0;
}
