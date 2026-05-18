**Jednoduchý CUDA program pro řešení 2D Poissonovy rovnice pomocí Jacobiho iterační metody na GPU**

**Inicializace pole**

__global__ void init(real* old_u, real* new_u, real* f, int N, real h) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        int idx = i * N + j;

        old_u[idx] = 0.0f;
        new_u[idx] = 0.0f;

        real x = i * h;
        real y = j * h;

        f[idx] = 2.0f * M_PI * M_PI * sinf(M_PI * x) * sinf(M_PI * y);
    }
}
