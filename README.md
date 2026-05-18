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

Jádro jacobi provádí jednu iteraci Jacobiho metody pro řešení Poissonovy rovnice na 2D mřížce. Každé vlákno na GPU počítá jeden vnitřní bod mřížky podle hodnot jeho čtyř sousedů z předchozí iterace. Indexy i a j jsou posunuté o +1, aby se nepočítaly okrajové body, protože na hranicích jsou nastavené nulové okrajové podmínky. Nejprve se vypočítá lineární index idx pro přístup do jednorozměrného pole v paměti GPU a potom se spočítá nová hodnota podle diskrétního tvaru Poissonovy rovnice:
<p align="center">
<img width="638" height="85" alt="image" src="https://github.com/user-attachments/assets/fe32cae6-ec50-4479-866f-afd8bfafc5d1" />
</p>
V programu je výraz $h^2/4$ uložený v proměnné h2_4, aby se během výpočtu nemuselo znovu počítat dělení. Hodnoty z předchozí iterace jsou uložené v poli old_u a nové vypočítané hodnoty se zapisují do new_u. Výpočet probíhá paralelně na GPU, takže se všechny body mřížky počítají současně.
