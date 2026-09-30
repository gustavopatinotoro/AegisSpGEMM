# Aegis-SpGEMM Engine

![Apple Silicon](https://img.shields.io/badge/Hardware-Apple_Silicon_M--Series-black?logo=apple)
![Swift](https://img.shields.io/badge/Swift-6.0-FA7343?logo=swift)
![Metal](https://img.shields.io/badge/Metal-C++14-00599C?logo=c%2B%2B)
![Architecture](https://img.shields.io/badge/Architecture-UMA_Zero--Copy-success)
![Status](https://img.shields.io/badge/Status-v0.7.1_Stream_Compaction-blueviolet)

**Aegis-SpGEMM** es un motor experimental de búsqueda vectorial e inferencia topológica de alta precisión, diseñado con arquitectura específicamente para el Chip (SoC) de Apple Serie-M1.

El objetivo principal del motor es **destruir el cuello de botella del *Pointer-Chasing* (persecución de punteros) en la CPU** al navegar estructuras de grafos de alta dimensionalidad. Para lograrlo, el sistema comprime la topología del índice en estructuras **CSR (Compressed Sparse Row)** alojadas en **Memoria Unificada (UMA)** y ejecuta una canalización híbrida de **Enrutamiento Grueso SIMD (AMX/vDSP) + Compactación de Flujo (*Stream Compaction*) en la GPU**.

---

## 1. Evolución Arquitectónica: De HNSW a IVF-CSR con Stream Compaction

### 1.1 El Límite Físico de HNSW en GPUs
HNSW (*Hierarchical Navigable Small World*) es el estándar de oro para búsqueda aproximada en CPU mediante navegación voraz secuencial salto a salto ($h_0 \rightarrow h_1 \rightarrow \dots \rightarrow h_t$). Sin embargo, su ejecución directa sobre una GPU presenta tres barreras arquitectónicas severas:
1. **Cobertura Nula en Pase Único (*Single-Hop*):** Evaluar únicamente la vecindad inmediata de un nodo semilla en un grafo HNSW Nivel 0 de $N = 120,000$ nodos explora apenas $\le 64$ candidatos ($0.05\%$ del espacio), produciendo un **Recall@10 del 0.00%**.
2. **Penalización por *Ping-Pong* CPU-GPU:** Ejecutar el bucle iterativo `while` de navegación HNSW requiere decenas de despachos secuenciales de `MTLCommandBuffer` por consulta, donde la latencia de sincronización del *driver* supera el tiempo real de cómputo algebraico.
3. **Divergencia de Hilos (*SIMD-group Divergence*):** Lanzar un hilo por cada uno de los $N = 120,000$ vectores utilizando una máscara de estado dispersa provoca que el $98.4\%$ de los hilos aborten temprano, dejando inactivos 31 de cada 32 carriles físicos dentro de cada *SIMD-group* de la GPU de Apple.

### 1.2 La Solución: Topología Bipartita IVF-CSR + Stream Compaction
Para alcanzar latencias de pocos milisegundos con alta exactitud en un **único despacho de GPU**, **Aegis-SpGEMM** implementa un diseño híbrido en tres etapas:
* **Etapa 1 — Enrutamiento Grueso en CPU (`Accelerate / vDSP`):** La CPU evalúa el vector de consulta contra los $C = 1,024$ centroides de Voronoi utilizando instrucciones vectoriales SIMD/AMX y selecciona las $\text{nprobe} = 16$ celdas óptimas en $\sim 0.9\text{ ms}$.
* **Etapa 2 — Unión de Listas Invertidas (*Posting List Compaction*):** Utilizando los arreglos `rowPtr` y `colIdx` en memoria, la CPU ensambla de forma exacta los identificadores únicos de los candidatos ($\sim 1,875$ vectores) pertenecientes a las 16 celdas ganadoras.
* **Etapa 3 — Evaluación Densa Libre de Divergencia en GPU (Metal):** La GPU lanza exactamente $\vert{}\mathcal{K}\vert{} \approx 1,875$ hilos mediante el kernel `sparse_similarity_compact`, logrando un **100% de ocupación SIMD**, cero ramas condicionales muertas y lectura directa *Zero-Copy* desde la UMA.

---

## 2. Fundamentos Matemáticos

### 2.1 Equivalencia Métrica en la Hiperesfera Unitaria
Las unidades aritméticas (ALU) de la GPU de Apple Silicon alcanzan su máximo rendimiento de cómputo ejecutando instrucciones **FMA (*Fused Multiply-Add*)** propias del Producto Interno. Para evaluar un espacio definido bajo Distancia Euclidiana ($L_2$) mediante Producto Interno sin pérdida de exactitud, proyectamos todos los vectores base $x \in \mathbb{R}^D$ y vectores de consulta $q \in \mathbb{R}^D$ sobre la hiperesfera unitaria $\mathbb{S}^{D-1}$:

$$\hat{x} = \frac{x}{\Vert{}x\Vert{}_2}, \quad \hat{q} = \frac{q}{\Vert{}q\Vert{}_2} \implies \Vert{}\hat{x}\Vert{}_2 = \Vert{}\hat{q}\Vert{}_2 = 1$$

Desarrollando el cuadrado de la distancia euclidiana entre dos vectores normalizados:

$$\Vert{}\hat{q} - \hat{x}\Vert{}_2^2 = \Vert{}\hat{q}\Vert{}_2^2 + \Vert{}\hat{x}\Vert{}_2^2 - 2(\hat{q} \cdot \hat{x}) = 2 - 2\sum_{d=0}^{D-1} \hat{q}_d \hat{x}_d$$

Dado que la función $f(s) = 2 - 2s$ es estrictamente decreciente respecto al producto interno $s = \hat{q} \cdot \hat{x}$, minimizar la distancia $L_2$ es matemáticamente idéntico a maximizar el producto interno:

$$\arg\min_{x \in \mathcal{X}} \Vert{}\hat{q} - \hat{x}\Vert{}_2^2 = \arg\max_{x \in \mathcal{X}} (\hat{q} \cdot \hat{x})$$

### 2.2 Enrutamiento Grueso sobre Celdas de Voronoi (CPU SIMD)
Sea $M \in \mathbb{R}^{C \times D}$ el tensor de centroides entrenado mediante K-Means esférico ($C = 1,024$). El puntaje de afinidad $s_c$ para cada celda $c \in \{0, 1, \dots, C-1\}$ se calcula mediante `vDSP_dotpr`:

$$s_c = \sum_{d=0}^{D-1} \hat{q}_d \cdot M_{c, d}$$

Seleccionando los índices de las $P = \text{nprobe}$ celdas con mayor afinidad, obtenemos el conjunto de entrada activo $\mathcal{C}^* \subset \{0, 1, \dots, C-1\}$ tal que $\vert{}\mathcal{C}^*\vert{} = P$.

### 2.3 Representación CSR y Compactación de Candidatos
La partición del espacio se codifica como un grafo bipartito dirigido $(C \times N)$ sin *padding* mediante dos vectores enteros de 32 bits (`Int32`):
* **Punteros de Fila ($R$):** Arreglo `rowPtr` de longitud $C + 1$ que indica el desplazamiento inicial de cada lista invertida.
* **Índices de Columna ($I$):** Arreglo `colIdx` de longitud $N$ que almacena los identificadores de los vectores base agrupados por celda.

El flujo compactado de candidatos $\mathcal{K} = \{k_0, k_1, \dots, k_{M-1}\}$ se construye mediante la unión de los segmentos contiguos en $I$:

$$\mathcal{K} = \bigcup_{c \in \mathcal{C}^*} \{ I[j] \mid R[c] \le j < R[c+1] \}$$

### 2.4 Kernel de Similitud Compactada en GPU
Para cada hilo con identificador global $m \in \{0, 1, \dots, \vert{}\mathcal{K}\vert{} - 1\}$, el kernel de Metal recupera el índice físico $k_m = \mathcal{K}[m]$ y computa el producto interno denso sobre la matriz aplanada de documentos $V \in \mathbb{R}^{N \times D}$ alojada en la UMA:

$$y_m = \sum_{d=0}^{D-1} \hat{q}_d \cdot V_{k_m, d}$$

---

## 3. Innovaciones de Ingeniería en Silicio

1. **Zero-Copy Real en Memoria Unificada (UMA):** A través de un puente nativo en `Objective-C++` (`UMAMemoryManager`), los tensores topológicos y densos se asignan con `MTLResourceStorageModeShared`. Tanto la CPU como la GPU operan sobre las mismas páginas físicas de RAM con **0 bytes copiados** a través de buses externos.
2. **Scratchpads Transaccionales Pre-asignados:** Se eliminaron por completo las llamadas a `device.makeBuffer()` dentro del bucle de búsqueda. Los búferes de consulta, candidatos e inferencia se reservan una única vez durante la ingesta *Ahead-of-Time* (AoT), evitando llamadas al sistema operativo en caliente.
3. **Inspección Directa con `UnsafeBufferPointer`:** Al concluir la ejecución del `MTLCommandBuffer`, el motor expone directamente el puntero de memoria compartida a la CPU sin instanciar ni clonar arreglos intermedios en el *Heap* de Swift.
4. **Inyección Binaria vía `mmap`:** El artefacto pre-compilado (`ag_news-384-ivf-csr.bin`) utiliza un diseño binario contiguo *Little-Endian* de 188 MB que se mapea directamente desde el SSD hacia el espacio de direcciones virtuales.

---

## 4. Estructura Táctica del Sistema

El motor mantiene un aislamiento modular estricto por responsabilidades:

- 📂 **`Core/` (Phase 1):** Puente de bajo nivel en `Objective-C++` (`UMAMemoryManager`) para asignación alineada en Memoria Unificada.
- 📂 **`IO/` (Phase 6):** Cargador binario `BinaryLoader.swift` con mapeo de memoria (`alwaysMapped`) para lectura instantánea de centroides, vectores, topología CSR y *Ground Truth*.
- 📂 **`Index/` (Phase 2 & 7.1):** `CSRHardwareContext.swift` y `CSRTopologyCompiler.swift`. Compilador AoT y gestor de *Scratchpads* persistentes en UMA.
- 📂 **`Metal/` (Phase 3 & 7.1):** `SpGEMMKernel.metal` y `MetalSpGEMMEngine.swift`. Estado de tubería de cómputo (`MTLComputePipelineState`) y despacho de hilos compactados.
- 📂 **`Pipeline/` (Phase 4 & 7.1):** `SearchOrchestrator.swift`. Fachada principal que coordina el enrutamiento grueso con `Accelerate/vDSP`, la compactación CSR y la extracción Top-K.
- 📂 **`Telemetry/` (Phase 5 & 7.1):** `BenchmarkSuite.swift`. Perfilador académico de percentiles de latencia ($P_{95}$, $P_{99}$), ancho de banda efectivo (GB/s) y *Recall@K*.

---

## 5. Resultados Empíricos (Benchmark AG News — 120,000 Vectores, 384D)

### 5.1 Evolución del Rendimiento a través de las Fases
| Fase del Proyecto | Arquitectura del Motor | Artefacto (`.bin`) | Candidatos Evaluados | Latencia Promedio | Latencia Warm (GPU) | Recall@10 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fase 6A** | HNSW-L0 (`entryPoint = 0`) | 219 MB | $\le 64$ ($0.05\%$) | 20.91 ms | ~18.50 ms | 0.00% |
| **Fase 6B** | IVF-CSR Base (Loops Swift + Mask) | 188 MB | ~1,875 ($1.56\%$) | 62.37 ms | ~59.40 ms | **92.00%** |
| **Fase 7.0** | IVF-CSR + `vDSP` + Scratchpad UMA | 188 MB | ~1,875 ($1.56\%$) | 18.19 ms | ~14.65 ms | **92.00%** |
| **Fase 7.1** | **IVF-CSR + `vDSP` + Stream Compaction** | **188 MB** | **~1,875 ($1.56\%$)** | **7.10 ms** | **2.97 ms** | **92.00%** |

### 5.2 Telemetría Real en Consola (Fase 7.1)
```text
=========================================================
  AegisSpGEMM Engine - PHASE 7.1 (COMPACTED SILICON)     
=========================================================

[System] Localizando artefacto en: Dev/AegisSpGEMM/AegisSpGEMM/data/ag_news-384-ivf-csr.bin
[BinaryLoader] IVF-CSR: 120000 vectores, 1024 centroides, 384 dims.
[BinaryLoader] Memoria cargada: ~188 MB.
[Aegis-Info] Inicializando puente UMA con CSR y Scratchpad compacto...
[Aegis-Success] Traspaso Zero-Copy completado en 167.5470 ms.
[Aegis-Telemetry] Vectores: 120000 | Centroides: 1024 | Dim: 384

[Hybrid-Dispatch] Procesando 10 consultas con Stream Compaction...
 ↳ Query 01 | Total: 22.862 ms (CPU Union: 2.211 ms | GPU Compact: 20.651 ms) | Recall: 8/10 (80%)
 ↳ Query 02 | Total:  7.637 ms (CPU Union: 1.122 ms | GPU Compact:  6.515 ms) | Recall: 10/10 (100%)
 ↳ Query 03 | Total:  6.170 ms (CPU Union: 1.065 ms | GPU Compact:  5.105 ms) | Recall: 7/10 (70%)
 ↳ Query 04 | Total:  4.806 ms (CPU Union: 1.014 ms | GPU Compact:  3.792 ms) | Recall: 10/10 (100%)
 ↳ Query 05 | Total:  5.073 ms (CPU Union: 1.032 ms | GPU Compact:  4.041 ms) | Recall: 10/10 (100%)
 ↳ Query 06 | Total:  5.936 ms (CPU Union: 1.736 ms | GPU Compact:  4.200 ms) | Recall: 9/10 (90%)
 ↳ Query 07 | Total:  3.848 ms (CPU Union: 0.876 ms | GPU Compact:  2.972 ms) | Recall: 10/10 (100%)
 ↳ Query 08 | Total:  3.996 ms (CPU Union: 0.871 ms | GPU Compact:  3.125 ms) | Recall: 8/10 (80%)
 ↳ Query 09 | Total:  4.220 ms (CPU Union: 0.877 ms | GPU Compact:  3.343 ms) | Recall: 10/10 (100%)
 ↳ Query 10 | Total:  6.415 ms (CPU Union: 0.929 ms | GPU Compact:  5.486 ms) | Recall: 10/10 (100%)

=========================================================
 Latencia CPU (Union Posting) : 1.1733 ms
 Latencia GPU (Compact SIMD)  : 5.9230 ms
 Latencia Total Promedio      : 7.0963 ms
 Recall Promedio (@10)        : 92.00%
=========================================================
Program ended with exit code: 0
```

---

## 6. A Hombros de Gigantes (Fundamentos Teóricos)

Esta arquitectura se apoya en décadas de investigación fundamental en recuperación de información, cuantización vectorial y álgebra lineal numérica:

* **Estructuras de Recuperación (1960):** El trabajo pionero de **Edward Fredkin** con la invención del *Trie* ([Trie Memory - ACM](https://dl.acm.org/doi/epdf/10.1145/367390.367400)), que sentó las bases algorítmicas para el almacenamiento de memoria estructurada y la recuperación eficiente de información.
* **Índices de Archivos Invertidos en Espacios Vectoriales — IVF (2003 & 2011):** El trabajo seminal de **J. Sivic y A. Zisserman** ([Video Google - ICCV](https://ieeexplore.ieee.org/document/1238663)), que adaptó por primera vez los archivos invertidos léxicos a espacios métricos continuos mediante cuantización K-Means, y la formulación moderna de **H. Jégou, M. Douze y C. Schmid** ([Product Quantization for Nearest Neighbor Search - IEEE TPAMI](https://ieeexplore.ieee.org/document/5432202)), que formalizó la arquitectura **IVF (*Inverted File System*)** sobre particiones de Voronoi y búsqueda no exhaustiva controlada por $\text{nprobe}$.
* **Búsqueda IVF Acelerada por GPU a Escala Masiva (2019):** La investigación de **J. Johnson, M. Douze y H. Jégou** en **FAISS** ([Billion-scale similarity search with GPUs - IEEE Transactions on Big Data](https://arxiv.org/abs/1702.08734)), que demostró cómo acoplar listas invertidas IVF con ejecución masivamente paralela en GPU, inspirando directamente nuestra adaptación *Zero-Copy* sobre la Memoria Unificada de Apple Silicon.
* **Redes de Pequeños Mundos y HNSW (2018):** La investigación de **Yu. A. Malkov y D. A. Yashunin** ([HNSW - IEEE TPAMI](https://ieeexplore.ieee.org/document/8594636)), referente en navegación sobre grafos de proximidad multicapa cuyo análisis de cuellos de botella en GPU motivó la transición de *Aegis* desde el *pointer-chasing* hacia la compactación topológica IVF-CSR.
* **Aceleración por Matrices Dispersas y Producto Interno (2020–2026):** Los avances de **Google Research** con **ScaNN** ([Announcing ScaNN](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/)) y **STATIC** ([Vectorizing the Trie](https://arxiv.org/pdf/2602.22647)), que validaron empíricamente cómo transformar estructuras de búsqueda discretas en operaciones matriciales dispersas (SpMV/SpGEMM) libres de divergencia de ramas en hardware moderno.

---

## 7. Requisitos y Despliegue

### Requisitos del Entorno
* **Hardware:** Mac con procesador Apple Silicon (Serie M1, M2, M3, M4) con Memoria Unificada (UMA).
* **IDE:** Xcode (configurado como *macOS Command Line Tool*).
* **Lenguajes y Frameworks:** Swift 6, Objective-C++, Metal Shading Language (C++14), Apple `Accelerate` (`vDSP`).

### Instalación y Ejecución
1. Clonar el repositorio:
```bash
git clone https://github.com/gustavopatinotoro/AegisSpGEMM.git
cd AegisSpGEMM
```
2. Colocar el artefacto binario pre-compilado `ag_news-384-ivf-csr.bin` dentro del directorio `AegisSpGEMM/data/`.
3. Abrir el proyecto en Xcode y verificar en **Build Phases** del target `AegisSpGEMM` que `SpGEMMKernel.metal` esté incluido en **Compile Metal Sources** y que el archivo `AegisSpGEMM-Bridging-Header.h` esté enlazado en **Build Settings**.
4. Ejecutar con `Cmd + R` (para pruebas de desarrollo) o compilar el esquema en modo **Release (`-O`)** para perfilado de latencia máxima en silicio.

---

## Seguridad y Licencia
Desarrollado bajo principios estrictos de seguridad de memoria y Diseño por Contratos (DbC) como proyecto de investigación aplicada en motores de búsqueda vectorial sobre arquitecturas heterogéneas. Consulta el archivo `LICENSE` para más detalles.
