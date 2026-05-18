**CUDA program pro řešení 2D Poissonovy rovnice pomocí Jacobiho iterační metody na GPU**

**Inicializace pole**
```cpp
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
```
Inicializační CUDA jádro init slouží k přípravě dat před samotným výpočtem Jacobiho iterací. Každé GPU vlákno zpracovává jeden bod dvourozměrné mřížky, přičemž z indexů vláken a bloků vypočítá odpovídající souřadnice (i, j). Pro každý bod jsou pole old_u a new_u nastavena na nulovou počáteční aproximaci řešení. Následně se z diskrétních indexů vypočítají fyzikální souřadnice x a y v oblasti [0,1] × [0,1] a do pole f se uloží hodnota pravé strany Poissonovy rovnice definovaná funkcí $f(x,y)=2\pi^2\sin(\pi x)\sin(\pi y)$. Jádro běží paralelně na GPU, takže všechny body mřížky jsou inicializovány současně, což výrazně urychluje přípravu dat pro následný numerický výpočet.

**Jacobiho metoda**
```cpp
__global__ void jacobi(real* old_u, real* new_u, real* f, int N, real h2_4) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i < N - 1 && j < N - 1) {
        int idx = i * N + j;

        new_u[idx] = 0.25f * (
            old_u[idx - N] +
            old_u[idx + N] +
            old_u[idx - 1] +
            old_u[idx + 1] +
            h2_4 * f[idx]
        );
    }
}
```
Jádro jacobi provádí jednu iteraci Jacobiho metody pro řešení Poissonovy rovnice na 2D mřížce. Každé vlákno na GPU počítá jeden vnitřní bod mřížky podle hodnot jeho čtyř sousedů z předchozí iterace. Indexy i a j jsou posunuté o +1, aby se nepočítaly okrajové body, protože na hranicích jsou nastavené nulové okrajové podmínky. Nejprve se vypočítá lineární index idx pro přístup do jednorozměrného pole v paměti GPU a potom se spočítá nová hodnota podle diskrétního tvaru Poissonovy rovnice:
<p align="center">
<img width="638" height="85" alt="image" src="https://github.com/user-attachments/assets/fe32cae6-ec50-4479-866f-afd8bfafc5d1" />
</p>

**main funkce**
1.  Nastavení parametrů simulace
```cpp
    int N = 8192;
    int max_iterations = 100;

    real L = 1.0f;
    real h = L / (N - 1);
    real h2_4 = (h * h) / 4.0f;

    size_t size = N * N * sizeof(real);

    real *old_u, *new_u, *f;
```
Nejprve se nastaví parametry simulace, jako velikost mřížky N a počet Jacobiho iterací max_iterations. Dále se vypočítá krok mřížky h.

2.  Nastavení CUDA
```cpp
    cudaMalloc(&old_u, size);
    cudaMalloc(&new_u, size);
    cudaMalloc(&f, size);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    init<<<grid, block>>>(old_u, new_u, f, N, h);
    cudaDeviceSynchronize();
```
Tato část alokuje paměť na GPU pomocí funkcí cudaMalloc pro pole old_u, new_u a f. Poté se nastaví konfigurace CUDA kernelů pomocí objektů dim3, kde block(16,16) určuje počet vláken v jednom bloku a grid(...) počet bloků potřebných pro pokrytí celé mřížky.

3.  Výpočet
```cpp
    auto start = chrono::high_resolution_clock::now();

    for (int iter = 0; iter < max_iterations; iter++) {
        jacobi<<<grid, block>>>(old_u, new_u, f, N, h2_4);
        cudaDeviceSynchronize();

        real* temp = old_u;
        old_u = new_u;
        new_u = temp;
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
```
Ve for cyklu se opakovaně spouští kernel jacobi, který provádí jednotlivé Jacobiho iterace. Po každé iteraci dojde k prohození ukazatelů old_u a new_u, takže nově vypočítané hodnoty se použijí jako vstup pro další iteraci. Synchronizace výpočtu se provádí pomocí cudaDeviceSynchronize().

4.  Výkon a Paměťová propustnost
```cpp
    double total_points = (double)(N - 2) * (N - 2) * max_iterations;
    double gflops = (total_points * 7.0) / (elapsed.count() * 1e9);
    double bandwidth = (total_points * 6.0 * sizeof(real)) /
                       (elapsed.count() * 1024 * 1024 * 1024);
```
Po skončení iterací program vypočítá výkon GPU v jednotkách GFLOPS a odhad paměťové propustnosti v GB/s.

5.  Výpis
```cpp
    cout << "Iterations: " << max_iterations << endl;
    cout << "Elapsed time: " << elapsed.count() << " s" << endl;
    cout << "Performance: " << gflops << " GFLOPS" << endl;
    cout << "Memory Bandwidth: " << bandwidth << " GB/s" << endl;
```
Výsledky se následně vypíšou do konzole.

6.  Uvolnění paměti
```cpp
    cudaFree(old_u);
    cudaFree(new_u);
    cudaFree(f);
```
Nakonec se uvolní alokovaná GPU paměť pomocí cudaFree a program se ukončí.
