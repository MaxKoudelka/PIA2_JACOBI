// nvcc -O3 -use_fast_math Jacobi_cuda.cu -o solver_cuda

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;
using real = float;

// CUDA kernel pro inicializaci polí //

__global__ void init(real* old_u, real* new_u, real* f, int N, real h) {        // kernel pro inicializaci polí
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        int idx = i * N + j;                                                    // Převod 2D indexu na 1D index

        old_u[idx] = 0.0f;                                                      // Inicializace počáteční aproximace řešení
        new_u[idx] = 0.0f;

        real x = i * h;
        real y = j * h;

        f[idx] = 2.0f * M_PI * M_PI * sinf(M_PI * x) * sinf(M_PI * y);          // Inicializace pravé strany Poissonovy rovnice

    }
}

// CUDA kernel pro jednu Jacobiho iteraci //

__global__ void jacobi(real* old_u, real* new_u, real* f, int N, real h2_4) {   // kernel pro Jacobiho iteraci
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;                          // Posun o +1 kvůli vynechání okrajových bodů
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i < N - 1 && j < N - 1) {
        int idx = i * N + j;

        new_u[idx] = 0.25f * (                                                  // Jacobi
            old_u[idx - N] +
            old_u[idx + N] +
            old_u[idx - 1] +
            old_u[idx + 1] +
            h2_4 * f[idx]
        );
    }
}

int main() {
    int N = 8192;
    int max_iterations = 100;

    real L = 1.0f;
    real h = L / (N - 1);
    real h2_4 = (h * h) / 4.0f;

    size_t size = N * N * sizeof(real);

    real *old_u, *new_u, *f;

    cudaMalloc(&old_u, size);                                               // Alokace paměti na GPU
    cudaMalloc(&new_u, size);
    cudaMalloc(&f, size);

    dim3 block(16, 16);                                                     // Nastavení velikosti bloku vláken
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);                                 // Nastavení velikosti gridu

    init<<<grid, block>>>(old_u, new_u, f, N, h);
    cudaDeviceSynchronize();                                                // Synchronizace doběhu kernelu

    auto start = chrono::high_resolution_clock::now();

    for (int iter = 0; iter < max_iterations; iter++) {
        jacobi<<<grid, block>>>(old_u, new_u, f, N, h2_4);
        cudaDeviceSynchronize();

        real* temp = old_u;                                                 // Prohození ukazaelů
        old_u = new_u;
        new_u = temp;
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    // Výpočet výkonu //

    double total_points = (double)(N - 2) * (N - 2) * max_iterations;
    double gflops = (total_points * 7.0) / (elapsed.count() * 1e9);
    double bandwidth = (total_points * 6.0 * sizeof(real)) /
                       (elapsed.count() * 1024 * 1024 * 1024);

    // Výpis //
    
    cout << "Iterations: " << max_iterations << endl;
    cout << "Elapsed time: " << elapsed.count() << " s" << endl;
    cout << "Performance: " << gflops << " GFLOPS" << endl;
    cout << "Memory Bandwidth: " << bandwidth << " GB/s" << endl;

    cudaFree(old_u);                                                       // Uvolnění paměti
    cudaFree(new_u);
    cudaFree(f);

    return 0;
}
