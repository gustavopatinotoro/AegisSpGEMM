# Aegis-SpGEMM Engine

![Apple Silicon](https://img.shields.io/badge/Hardware-Apple_Silicon_M--Series-black?logo=apple)
![ANE](https://img.shields.io/badge/Neural_Engine-CoreML_FP16-FF2D55?logo=apple)
![Swift](https://img.shields.io/badge/Swift-6.0-FA7343?logo=swift)
![Metal](https://img.shields.io/badge/Metal-C++14-00599C?logo=c%2B%2B)
![Architecture](https://img.shields.io/badge/Architecture-Tri--Processor_UMA_Zero--Copy-success)
![Status](https://img.shields.io/badge/Status-v2.5.0_Thesis_Release-blueviolet)
![Universidad](https://img.shields.io/badge/Universidad-SantoTom%C3%A1s-0969da?style=flat-square)




**Aegis-SpGEMM** es un motor de búsqueda vectorial aproximada (*Approximate Nearest Neighbor Search — ANNS*) e inferencia topológica de alta precisión, co-diseñado a nivel de silicio para la arquitectura **Apple Silicon (Serie M)** bajo una restricción estricta de **8 GB de Memoria Unificada (UMA)**.

El objetivo científico del motor es **erradicar simultáneamente dos cuellos de botella históricos** en la recuperación de información de alta dimensionalidad:
1. **El cuello de botella físico del *Pointer-Chasing* y la divergencia SIMD en GPU** al navegar grafos dispersos, resuelto mediante una topología bipartita **IVF-CSR (*Inverted File — Compressed Sparse Row*)** acoplada a **Compactación de Flujo (*Stream Compaction*)** sin asignaciones dinámicas en memoria.
2. **La rigidez geométrica y de exploración ($\text{nprobe}$ estático) de la cuantización de Voronoi clásica (K-Means)**, superada mediante un **Enrutador Neuronal Residual ejecutado en el Apple Neural Engine (ANE)** en precisión `FP16`, gobernado por una ley de control adaptativa basada en la **Entropía de Shannon ($H_{32}$)** y muestreo de núcleo (*Adaptive Nucleus Probing*).

---

## 1. Evolución Arquitectónica: De HNSW a la Orquestación Cognitiva Tri-Procesador

### 1.1 El Límite Físico de HNSW en GPUs
HNSW (*Hierarchical Navigable Small World*) es el estándar de referencia para búsqueda vectorial en CPU mediante navegación voraz secuencial salto a salto ($h_0 \rightarrow h_1 \rightarrow \dots \rightarrow h_t$). Sin embargo, su traducción directa a una GPU presenta tres barreras arquitectónicas severas:
1. **Cobertura Nula en Pase Único (*Single-Hop*):** Evaluar únicamente la vecindad inmediata de un nodo semilla en un grafo HNSW Nivel 0 de $N = 120,000$ nodos explora apenas $\le 64$ candidatos ($0.05\%$ del espacio), produciendo un **Recall@10 del 0.00%**.
2. **Penalización por *Ping-Pong* CPU-GPU:** Ejecutar el bucle iterativo `while` de navegación HNSW requiere decenas de despachos secuenciales de `MTLCommandBuffer` por consulta, donde la latencia de sincronización del *driver* y mapeo IOMMU/UAT supera el tiempo real de cómputo algebraico.
3. **Divergencia de Hilos (*SIMD-group Divergence*):** Lanzar un hilo por cada uno de los $N = 120,000$ vectores utilizando una máscara de estado dispersa provoca que el $98.4\%$ de los hilos aborten temprano, dejando inactivos 31 de cada 32 carriles físicos dentro de cada *SIMD-group* de la GPU de Apple.

### 1.2 La Solución: Co-Diseño Heterogéneo ANE + CPU AMX + GPU Metal en UMA
Para alcanzar latencias de un solo dígito de milisegundo y superar el **97% de Recall@10** en un **único despacho de GPU**, **Aegis-SpGEMM** divide el problema entre los tres procesadores especializados del SoC Apple Silicon compartiendo las mismas páginas físicas de RAM:

```text
                       ┌──────────────────────────────────────────────────────────┐
                       │           MEMORIA UNIFICADA APPLE SILICON (UMA)          │
                       │  • Vectores Planos (184 MB)   • Topología CSR (rowPtr)   │
                       │  • Scratchpad Candidatos      • Punteros Zero-Copy       │
                       └───────────┬────────────────────────┬─────────────────────┘
                                   │                        │
        ┌──────────────────────────▼──────┐      ┌──────────▼─────────────────────────┐
        │   ETAPA 1: APPLE NEURAL ENGINE  │      │   ETAPA 2: CPU + COPROCESADOR AMX  │
        │  • Grafo CoreML FP16 Aislado    │─────▶│  • Entropía de Cola H_32 (Shannon) │
        │  • Residual Voronoi MLP Router  │      │  • Ley Adaptativa: nprobe ∈ [10,32]│
        │  • Predicción no-lineal: 0.98ms │      │  • Unión Disjunta memcpy: ~0.03ms  │
        └─────────────────────────────────┘      └──────────────────┬─────────────────┘
                                                                    │
                                                 ┌──────────────────▼─────────────────┐
                                                 │      ETAPA 3: GPU METAL SIMD       │
                                                 │  • Kernel sparse_similarity_compact│
                                                 │  • 100% Ocupación SIMD (0% Diverg.)│
                                                 │  • Silicio Puro: 0.46ms - 0.97ms   │
                                                 └────────────────────────────────────┘
```

* **Etapa 1 — Enrutamiento Cognitivo en el Apple Neural Engine (ANE) o CPU AMX:**
  * *Modo Cognitivo (ANE CoreML FP16):* El vector de consulta $\hat{\mathbf{q}} \in \mathbb{R}^{384}$ se inyecta en un tensor `MLMultiArray` pre-asignado hacia el modelo `AegisNeuralRouter.mlpackage`, restringido estrictamente a `.cpuAndNeuralEngine` para no competir por la GPU. El ANE predice la distribución de probabilidad $\mathbf{p} \in \Delta^{1023}$ sobre las $C = 1,024$ celdas de Voronoi en **0.982 ms**, corrigiendo las distorsiones no lineales en las fronteras de K-Means.
  * *Modo Estático Baseline (CPU AMX `vDSP_mmul`):* Evalúa el producto matriz-vector exacto $\mathbf{s} = M\hat{\mathbf{q}}$ en un único pase sobre el coprocesador matricial AMX en **0.914 ms**.
* **Etapa 2 — Control Adaptativo por Entropía ($H_{32}$) y Compactación Disjunta (`memcpy`):**
  La CPU mide la incertidumbre topológica de la consulta mediante la Entropía de Shannon de las 32 celdas principales ($H_{32}$). Si la consulta cae en el centro de un clúster ($H_{32}$ baja), reduce $\text{nprobe}$ al mínimo para ahorrar GPU; si cae en una frontera difusa ($H_{32}$ alta), expande $\text{nprobe}$ hasta $32$ celdas. Acto seguido, copia los segmentos contiguos de `colIdx` directamente al `candidateIndicesBuffer` en la UMA mediante llamadas `memcpy` sin usar estructuras `Set<Int32>`.
* **Etapa 3 — Similitud Densa Libre de Divergencia en GPU (Metal):**
  La GPU lanza exactamente $\vert{}\mathcal{K}\vert{}$ hilos mediante el kernel `sparse_similarity_compact`, logrando un **100% de ocupación SIMD**, cero ramas condicionales muertas y un ancho de banda sostenido en silicio de **6.05 GB/s**.

---

## 2. Fundamentos Matemáticos

### 2.1 Equivalencia Métrica en la Hiperesfera Unitaria
Las unidades aritméticas (ALU) de la GPU de Apple Silicon alcanzan su máximo rendimiento ejecutando instrucciones **FMA (*Fused Multiply-Add*)** propias del Producto Interno. Para evaluar un espacio definido bajo Distancia Euclidiana ($L_2$) mediante Producto Interno sin pérdida de exactitud, proyectamos todos los vectores base $\mathbf{x} \in \mathbb{R}^D$ y vectores de consulta $\mathbf{q} \in \mathbb{R}^D$ sobre la hiperesfera unitaria $\mathbb{S}^{D-1}$:

$$\hat{\mathbf{x}} = \frac{\mathbf{x}}{\Vert{}\mathbf{x}\Vert{}_2}, \quad \hat{\mathbf{q}} = \frac{\mathbf{q}}{\Vert{}\mathbf{q}\Vert{}_2} \implies \Vert{}\hat{\mathbf{x}}\Vert{}_2 = \Vert{}\hat{\mathbf{q}}\Vert{}_2 = 1$$

Desarrollando el cuadrado de la distancia euclidiana entre dos vectores normalizados:

$$\Vert{}\hat{\mathbf{q}} - \hat{\mathbf{x}}\Vert{}_2^2 = \Vert{}\hat{\mathbf{q}}\Vert{}_2^2 + \Vert{}\hat{\mathbf{x}}\Vert{}_2^2 - 2(\hat{\mathbf{q}} \cdot \hat{\mathbf{x}}) = 2 - 2\sum_{d=0}^{D-1} \hat{q}_d \hat{x}_d$$

Dado que $f(s) = 2 - 2s$ es estrictamente decreciente respecto al producto interno $s = \hat{\mathbf{q}} \cdot \hat{\mathbf{x}}$, minimizar la distancia $L_2$ es matemáticamente idéntico a maximizar el producto interno:

$$\arg\min_{\mathbf{x} \in \mathcal{X}} \Vert{}\hat{\mathbf{q}} - \hat{\mathbf{x}}\Vert{}_2^2 = \arg\max_{\mathbf{x} \in \mathcal{X}} (\hat{\mathbf{q}} \cdot \hat{\mathbf{x}})$$

---

### 2.2 Enrutamiento Grueso Estático como Operación BLAS-2 GEMV (AMX `vDSP_mmul`)
Sea $M \in \mathbb{R}^{C \times D}$ la matriz continua de centroides alojada en la UMA (`centroidsBuffer`). Para una consulta $\hat{\mathbf{q}} \in \mathbb{R}^{D \times 1}$, el vector completo de afinidades gruesas $\mathbf{s} \in \mathbb{R}^{C \times 1}$ se calcula en un único despacho matricial sobre el coprocesador AMX:

$$\mathbf{s} = M \hat{\mathbf{q}} \quad \Longleftrightarrow \quad s_c = \sum_{d=0}^{D-1} M_{[c \cdot D + d]} \cdot \hat{q}_d, \quad \forall c \in \{0, 1, \dots, C-1\}$$

---

### 2.3 Enrutador Neuronal Residual en el ANE y Minería de Fronteras Duras
Las celdas de Voronoi reales en $D = 384$ dimensiones no son esferas isotrópicas; cuando una consulta cae en la región tubular de frontera $\partial \mathcal{V}_\delta = \{\hat{\mathbf{q}} \in \mathbb{S}^{D-1} : \vert{}\hat{\mathbf{q}} \cdot \mathbf{m}_{c_1} - \hat{\mathbf{q}} \cdot \mathbf{m}_{c_2}\vert{} < \delta\}$, la distancia lineal al centroide falla en identificar celdas adyacentes que contienen vecinos verdaderos.

Para corregir esta distorsión sin perder la estructura global de Voronoi, definimos el **Enrutador Neuronal Residual** ejecutado en precisión `FP16` en el ANE:

$$\mathbf{z}_\theta(\hat{\mathbf{q}}) = \underbrace{\tau \cdot (M \hat{\mathbf{q}})}_{\text{Prior Geométrico de Voronoi}} + \underbrace{\lambda_{\text{res}} \cdot \tau \cdot \tanh\left( W_2 \cdot \text{GELU}\left(\text{LayerNorm}(W_1 \hat{\mathbf{q}} + \mathbf{b}_1)\right) + \mathbf{b}_2 \right)}_{\text{Corrección Neuronal Acotada de Frontera } \Delta_\theta(\hat{\mathbf{q}})}$$

Donde $W_1 \in \mathbb{R}^{512 \times 384}$, $W_2 \in \mathbb{R}^{1024 \times 512}$, $\lambda_{\text{res}} = 0.35$ y $\tau \in [6.0, 14.0]$. El modelo se entrena de forma auto-supervisada (sin tocar el conjunto de consultas de prueba) generando consultas sintéticas $\tilde{\mathbf{q}}$ en las fronteras entre celdas y minimizando la Divergencia de Kullback-Leibler frente a la distribución real $\mathbf{y}(\tilde{\mathbf{q}})$ de los $K = 10$ vecinos exactos:

$$
\mathcal{L}_{\text{KL}}(\theta) = \mathbb{E}_{\tilde{\mathbf{q}}} \left[ \sum_{c=0}^{C-1} y_c(\tilde{\mathbf{q}}) \ln \left( \frac{y_c(\tilde{\mathbf{q}})}{p_c(\tilde{\mathbf{q}}) + \varepsilon} \right) \right], \quad \mathbf{p}(\tilde{\mathbf{q}}) = \text{Softmax}(\mathbf{z}_\theta(\tilde{\mathbf{q}}))
$$


---

### 2.4 Ley de Control Adaptativa Híbrida: Masa Acumulada + Entropía de Cola ($H_{32}$)
Ordenando las probabilidades predichas por el ANE de forma decreciente $p_{\pi(0)} \ge p_{\pi(1)} \ge \dots \ge p_{\pi(C-1)}$, el sistema cuantifica la incertidumbre en la cola mediante la **Entropía de Shannon sobre las primeras $W = 32$ celdas**:

$$H_{32}(\mathbf{p}) = -\sum_{j=0}^{31} p_{\pi(j)} \ln\left( \max(p_{\pi(j)}, 10^{-9}) \right)$$

A partir de $H_{32}(\mathbf{p})$, se establece un piso dinámico continuo $P_{\text{floor}}(H_{32})$ con umbral $H_0 = 0.25$ y ganancia $\kappa = 12.0$:

$$P_{\text{floor}}(H_{32}) = \begin{cases} P_{\min} & \text{si } H_{32}(\mathbf{p}) \le H_0 \\ \min\left( P_{\max}, \; P_{\min} + \left\lceil \kappa \cdot \left(H_{32}(\mathbf{p}) - H_0\right) \right\rceil \right) & \text{si } H_{32}(\mathbf{p}) > H_0 \end{cases}$$

El número adaptativo de celdas a explorar $P_{\text{adapt}}(\hat{\mathbf{q}}) \in [P_{\min}, P_{\max}]$ combina el criterio de masa acumulada ($\tau_{\text{conf}} = 0.985$) con el piso de entropía:


$$
P_{\text{adapt}}(\hat{\mathbf{q}}) = \min \left( P_{\max}, \; \max \left( P_{\text{floor}}(H_{32}), \; \inf \left\lbrace m \in \{1, \dots, C\} \mid \sum_{j=0}^{m-1} p_{\pi(j)} \ge \tau_{\text{conf}} \right\rbrace \right) \right)
$$


---

### 2.5 Teorema de Disjunción Topológica CSR y Compactación Directa `memcpy`
Dado que la partición de Voronoi $\mathcal{V} = \{V_0, V_1, \dots, V_{C-1}\}$ asigna cada vector base a una única celda ($V_{c_1} \cap V_{c_2} = \emptyset$ para $c_1 \neq c_2$), los segmentos del arreglo CSR de columnas $I[R[c] \dots R[c+1]-1]$ son **mutuamente disjuntos**. Por consiguiente, la cardinalidad de la unión de candidatos es exactamente la suma de las longitudes de cada segmento:

$$
\left| \bigcup_{c \in \mathcal{C}_-} V_c \right| = \sum_{c \in \mathcal{C}_-} (R[c+1] - R[c])
$$


Esto permite reemplazar la inserción elemento por elemento en tablas *hash* (`Set<Int32>`) por $P_{\text{adapt}}$ copias de memoria contigua (`memcpy`) directamente hacia el búfer `candidateIndicesBuffer` en la UMA, con **cero asignaciones en el Heap**. Además, durante la compilación del índice, el tamaño máximo del *Scratchpad* $K_{\max}$ se calcula dinámicamente sumando las 64 celdas más pobladas del grafo ($K_{\max} = 24,730$ en AG News), garantizando inmunidad frente a desbordamientos de búfer.

---

## 3. Innovaciones de Ingeniería en Silicio

1. **Aislamiento Físico del ANE y *Hardware Warm-up*:** El enrutador neuronal se instancia con `MLModelConfiguration.computeUnits = .cpuAndNeuralEngine`, garantizando cero contención con los *Shader Cores* de la GPU. En el arranque, `performHardwareWarmup()` ejecuta 3 inferencias sintéticas para despertar al demonio del sistema `aned` y fijar los pesos en la SRAM del ANE, reduciendo la latencia de la primera consulta de $7.27\text{ ms}$ a $1.30\text{ ms}$.
2. **Ingesta Binaria Plana sin Fragmentación (`42.02 ms`):** `BinaryLoader` lee los tensores `flattenedCentroids` y `flattenedVectors` directamente como bloques contiguos desde el archivo mapeado con `mmap` (`alwaysMapped`), evitando la creación de $120,000$ sub-arreglos `[[Float]]` intermedios y reduciendo el tiempo de traspaso Zero-Copy de $167.55\text{ ms}$ a **42.02 ms**.
3. **Scratchpads Pre-asignados en Metal y CoreML:** Ni el camino de GPU (`MTLBuffer`) ni el camino de ANE (`MLMultiArray` de entrada y `probabilitiesScratchpad` de salida) realizan reservas de memoria dinámica durante las consultas.
4. **Telemetría de Silicio Puro (`gpuEndTime - gpuStartTime`):** `MetalSpGEMMEngine` instrumenta los registros físicos de tiempo del procesador gráfico en cada `MTLCommandBuffer`, aislando el tiempo real de ejecución en las unidades ALU SIMD (**0.46 ms – 0.97 ms**) respecto al tiempo de reloj de pared del sistema operativo.

---

## 4. Estructura Táctica del Sistema

El motor mantiene un aislamiento modular estricto por responsabilidades:

- 📂 **`Core/` (Phase 1):** Puente nativo en `Objective-C++` (`UMAMemoryManager.h` / `.mm`) para asignación alineada en Memoria Unificada (`MTLResourceStorageModeShared`).
- 📂 **`IO/` (Phase 6 & 7.2):** `BinaryLoader.swift`. Cargador binario universal con mapeo `mmap`, validación de fronteras de bytes y extracción directa de tensores planos.
- 📂 **`Index/` (Phase 2 & 7.2):** `CSRHardwareContext.swift` y `CSRTopologyCompiler.swift`. Compilador AoT, alojamiento de matriz continua de centroides y dimensionamiento dinámico de *Scratchpads* ($K_{\max}$).
- 📂 **`CoreML/` (Phase 8.0 & 8.1):** `ANENeuralRouter.swift`. Gestor del modelo neuronal en el Apple Neural Engine con *Hardware Warm-up*, *Scratchpads* pre-asignados y Ley de Control Híbrida Masa-Entropía ($H_{32}$).
- 📂 **`Metal/` (Phase 3, 7.1 & 7.2):** `SpGEMMKernel.metal` y `MetalSpGEMMEngine.swift`. Kernel `sparse_similarity_compact` libre de divergencia y telemetría de contadores físicos de silicio GPU.
- 📂 **`Pipeline/` (Phase 4, 7.2 & 8.1):** `SearchOrchestrator.swift`. Orquestador heterogéneo que coordina el enrutamiento ANE/AMX, la compactación directa `memcpy` en UMA y el despacho Metal.
- 📂 **`Telemetry/` (Phase 5, 7.2 & 8.1):** `BenchmarkSuite.swift`. Suite estadística de grado académico para evaluación sobre $N = 1,000$ consultas con percentiles extremo a extremo ($P_{50}, P_{95}, P_{99}$) y ancho de banda efectivo en silicio (GB/s).

---

## 5. Resultados Empíricos (Benchmark AG News — 120,000 Vectores, 384D)

### 5.1 Validación Estadística de Tesis ($N_q = 1,000$ Consultas, $10,000$ Vecinos Verificados)
| Métrica Estadística / Hardware | Baseline Fase 7.2 (`AMX vDSP_mmul`, $\text{nprobe} = 16$) | Fase 8.1 Cognitiva (`ANE CoreML FP16` + Ley $H_{32}$) | Impacto Científico Verificado |
| :--- | :---: | :---: | :--- |
| **Recall@10 Global ($N_q = 1,000$)** | 95.07% | **97.26%** | **+2.19 pt absolutos** (Supera el umbral del 97%) |
| **Tasa de Fallo (*Miss Rate* / Falsos Negativos)** | 4.93% *(493 perdidos)* | **2.74%** *(274 perdidos)* | **-44.42% de reducción relativa de errores (RMRR)** |
| **Latencia del Enrutador (Etapa 1)** | 0.914 ms (CPU AMX) | **0.982 ms (ANE FP16)** | **Solo +68 µs** por inferir una red neuronal en el ANE |
| **Tiempo Puro de Silicio GPU (Etapa 2)** | 0.634 ms | **0.974 ms** | Escalabilidad lineal exacta respecto al volumen de candidatos |
| **Ancho de Banda en Silicio GPU** | 6.10 GB/s | **6.05 GB/s** | **99.2% de invarianza** de ancho de banda en la UMA |
| **Latencia Total Promedio** | 4.937 ms | **6.970 ms** | Cumple holgadamente el presupuesto de tiempo real (<15 ms) |
| **Percentiles End-to-End ($P_{50} / P_{95} / P_{99}$)** | 4.65 / 7.23 / 8.06 ms | **7.10 / 9.68 / 11.40 ms** | Percentil $P_{99}$ contenido en **11.396 ms** |
| **Celdas Exploradas ($\text{nprobe}$ Promedio)** | 16.00 (Fijo) | **23.80 (Adaptativo)** | Modulación automática entre $10$ y $32$ celdas según $H_{32}$ |

---

### 5.2 Evolución Histórica del Motor a través de todas las Fases
| Fase | Arquitectura e Hito Técnico | Ingesta UMA | Latencia Media (10 Q) | Silicio GPU | Recall@10 (10 Q) | Recall@10 (1,000 Q) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Fase 6A** | HNSW-L0 en GPU (`entryPoint = 0`, 98.4% divergencia) | 207.99 ms | 20.91 ms | N/A | 0.00% | 0.00% |
| **Fase 6B** | Pivote a IVF-CSR Bipartito (Loops Swift + Máscara GPU) | 185.90 ms | 62.37 ms | N/A | 92.00% | ~95.07% |
| **Fase 7.0** | Enrutamiento Vectorizado CPU (`vDSP_dotpr`) + Máscara | 195.77 ms | 18.19 ms | N/A | 92.00% | ~95.07% |
| **Fase 7.1** | GPU *Stream Compaction* (`sparse_similarity_compact` + `Set`) | 167.55 ms | 7.10 ms | N/A | 92.00% | ~95.07% |
| **Fase 7.2** | Ingesta Plana + AMX `vDSP_mmul` + `memcpy` Directo UMA | **42.02 ms** | 5.26 ms | 0.511 ms | 92.00% | 95.07% |
| **Fase 8.0** | ANE CoreML FP16 + *Nucleus Probing* ($\tau_{\text{conf}} = 0.965$) | 54.43 ms | **4.41 ms** | **0.466 ms** | **93.00%** | — |
| **Fase 8.1** | **ANE Warm-up + Ley de Entropía $H_{32}$ + Benchmark $N=1,000$** | **51.86 ms** | **6.24 ms** | **0.910 ms** | **92.00%** | **97.26%** |

---

### 5.3 Telemetría Real en Consola (Fase 8.1 — Trazabilidad 10 Q + Estadística 1,000 Q)
```text
=========================================================
  AegisSpGEMM Engine - PHASE 8.1 (CALIBRATED ANE + N=1000)
=========================================================

[System] Localizando artefacto en: Dev/AegisSpGEMM/AegisSpGEMM/Data/ag_news-384-ivf-csr.bin
[BinaryLoader] IVF-CSR: 120000 vectores, 1024 centroides, 384 dims.
[BinaryLoader] Memoria cargada de forma contigua: ~191 MB.
[Aegis-Info] Inicializando puente UMA con tensores planos y Scratchpad dinámico...
[Aegis-Success] Traspaso Zero-Copy completado en 51.8570 ms.
[Aegis-Telemetry] Vectores: 120000 | Centroides: 1024 | Dim: 384 | MaxCandidates UMA: 24730
[ANE-Info] Compilando y ejecutando Warm-up de 'AegisNeuralRouter.mlpackage' en el Apple Neural Engine...
[ANE-Success] Enrutador Neuronal ANE online y caliente en 244.15 ms (ComputeUnits: .cpuAndNeuralEngine).

---------------------------------------------------------
 [A] BASELINE FASE 7.2: AMX Estático (nprobe fijo = 16)
---------------------------------------------------------
 ↳ Baseline Promedio | Total: 5.2434 ms (AMX Route: 1.0783 ms | GPU Wall: 4.1651 ms [Silicio: 0.6960 ms]) | Recall@10: 92.00%

---------------------------------------------------------
 [B] FASE 8.1 COGNITIVA: ANE CoreML FP16 (Warm + Entropy Law)
---------------------------------------------------------
 ↳ Query 01 | nprobe: 20 (H=1.07) | Total:  6.070 ms (Route: 1.307 ms | GPU: 4.763 ms [Silicon: 0.869 ms]) | Recall: 8/10 (80%)
 ↳ Query 02 | nprobe: 13 (H=0.45) | Total:  4.210 ms (Route: 1.033 ms | GPU: 3.177 ms [Silicon: 0.607 ms]) | Recall: 10/10 (100%)
 ↳ Query 03 | nprobe: 32 (H=1.97) | Total:  7.736 ms (Route: 1.102 ms | GPU: 6.634 ms [Silicon: 1.337 ms]) | Recall: 7/10 (70%)
 ↳ Query 04 | nprobe: 31 (H=1.95) | Total:  6.993 ms (Route: 1.205 ms | GPU: 5.788 ms [Silicon: 1.092 ms]) | Recall: 9/10 (90%)
 ↳ Query 05 | nprobe: 23 (H=1.26) | Total:  4.021 ms (Route: 1.089 ms | GPU: 2.932 ms [Silicon: 0.546 ms]) | Recall: 10/10 (100%)
 ↳ Query 06 | nprobe: 31 (H=1.70) | Total:  7.758 ms (Route: 1.016 ms | GPU: 6.742 ms [Silicon: 1.007 ms]) | Recall: 10/10 (100%)
 ↳ Query 07 | nprobe: 24 (H=1.34) | Total:  6.634 ms (Route: 0.993 ms | GPU: 5.641 ms [Silicon: 0.910 ms]) | Recall: 10/10 (100%)
 ↳ Query 08 | nprobe: 32 (H=2.13) | Total:  6.383 ms (Route: 1.192 ms | GPU: 5.191 ms [Silicon: 1.084 ms]) | Recall: 8/10 (80%)
 ↳ Query 09 | nprobe: 18 (H=0.86) | Total:  5.032 ms (Route: 1.138 ms | GPU: 3.894 ms [Silicon: 0.652 ms]) | Recall: 10/10 (100%)
 ↳ Query 10 | nprobe: 26 (H=1.52) | Total:  7.575 ms (Route: 1.164 ms | GPU: 6.411 ms [Silicon: 0.994 ms]) | Recall: 10/10 (100%)

=========================================================
  RESUMEN MICRO-BENCHMARK (10 CONSULTAS CANÓNICAS)
=========================================================
 Modo de Enrutamiento Activo   : ANE CoreML FP16 (Warm + Entropy Law)
 nprobe Adaptativo Promedio    : 25.00 celdas (Rango: 10..32)
 Latencia Router (ANE Warm)    : 1.1239 ms
 Latencia Compact + GPU Wall   : 5.1173 ms
 Tiempo Puro de Silicio GPU    : 0.9099 ms
 Latencia Total Promedio       : 6.2412 ms
 Recall@10 (Fase 7.2 -> 8.1)   : 92.00% -> 92.00%

=========================================================
  VALIDACIÓN ESTADÍSTICA DE TESIS (N = 1000 CONSULTAS)
=========================================================
 [Baseline AMX nprobe=16 ] Recall@10:  95.07% | Avg:  4.937 ms (Route: 0.914 | GPU: 4.023 | Silicon: 0.634) | P50: 4.655 | P95: 7.233 | P99: 8.060 ms | Silicon BW:   6.10 GB/s
 [Fase 8.1 ANE Adaptativo] Recall@10:  97.26% | Avg:  6.970 ms (Route: 0.982 | GPU: 5.988 | Silicon: 0.974) | P50: 7.104 | P95: 9.679 | P99: 11.396 ms | Silicon BW:   6.05 GB/s | nprobe: 23.80
=========================================================
Program ended with exit code: 0
```

---

## 6. A Hombros de Gigantes (Fundamentos Teóricos)

Esta arquitectura se apoya en décadas de investigación fundamental en recuperación de información, cuantización vectorial, aprendizaje de particiones espaciales y álgebra lineal numérica:

* **Estructuras de Recuperación (1960):** El trabajo pionero de **Edward Fredkin** con la invención del *Trie* ([Trie Memory - ACM](https://dl.acm.org/doi/epdf/10.1145/367390.367400)), que sentó las bases algorítmicas para el almacenamiento de memoria estructurada y la recuperación eficiente de información.
* **Índices de Archivos Invertidos en Espacios Vectoriales — IVF (2003 & 2011):** El trabajo seminal de **J. Sivic y A. Zisserman** ([Video Google - ICCV](https://ieeexplore.ieee.org/document/1238663)), que adaptó por primera vez los archivos invertidos léxicos a espacios métricos continuos mediante cuantización K-Means, y la formulación moderna de **H. Jégou, M. Douze y C. Schmid** ([Product Quantization for Nearest Neighbor Search - IEEE TPAMI](https://ieeexplore.ieee.org/document/5432202)), que formalizó la arquitectura **IVF (*Inverted File System*)** sobre particiones de Voronoi y búsqueda no exhaustiva controlada por $\text{nprobe}$.
* **Redes de Pequeños Mundos y HNSW (2018):** La investigación de **Yu. A. Malkov y D. A. Yashunin** ([HNSW - IEEE TPAMI](https://ieeexplore.ieee.org/document/8594636)), referente en navegación sobre grafos de proximidad multicapa cuyo análisis de cuellos de botella en GPU motivó la transición de *Aegis* desde el *pointer-chasing* hacia la compactación topológica IVF-CSR.
* **Búsqueda IVF Acelerada por GPU a Escala Masiva (2019):** La investigación de **J. Johnson, M. Douze y H. Jégou** en **FAISS** ([Billion-scale similarity search with GPUs - IEEE Transactions on Big Data](https://arxiv.org/abs/1702.08734)), que demostró cómo acoplar listas invertidas IVF con ejecución masivamente paralela en GPU, inspirando directamente la adaptación *Zero-Copy* sobre la Memoria Unificada de Apple Silicon Serie-M1.
* **Particionamiento Espacial Aprendido y Exploración Adaptativa (2020–2021):** Los trabajos de **Y. Dong et al.** en **Neural LSH** ([Learning Space Partitions for Nearest Neighbor Search - ICLR 2020](https://arxiv.org/abs/1901.08544)) y **C. Li et al.** ([Improving Approximate Nearest Neighbor Search through Learned Adaptive Early Termination - ACM SIGMOD 2020](https://dl.acm.org/doi/10.1145/3318464.3380600)), que demostraron teóricamente la superioridad de los enrutadores neuronales y la terminación adaptativa frente a K-Means estático, el cual se implementó en **Aegis-SpGEMM** logrando una ejecución heterogénea sobre el **Apple Neural Engine (ANE)**.
* **Aceleración por Matrices Dispersas y Producto Interno (2020–2026):** Los avances de **Google Research** con **ScaNN** ([Announcing ScaNN](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/)) y **STATIC** ([Vectorizing the Trie](https://arxiv.org/pdf/2602.22647)), que validaron empíricamente cómo transformar estructuras de búsqueda discretas en operaciones matriciales dispersas (SpMV/SpGEMM) libres de divergencia de ramas en hardware moderno.

---

## 7. Requisitos y Despliegue

### Requisitos del Entorno
* **Hardware:** Mac con procesador Apple Silicon (Serie M1, M2, M3, M4) con Memoria Unificada (UMA) y Apple Neural Engine de 16 núcleos.
* **IDE:** Xcode (configurado como *macOS Command Line Tool*).
* **Lenguajes y Frameworks:** Swift 6, Objective-C++, Metal Shading Language (C++14), Apple `CoreML`, Apple `Accelerate` (`vDSP` / AMX).

### Instalación y Ejecución
1. Clonar el repositorio:
```bash
git clone https://github.com/gustavopatinotoro/AegisSpGEMM.git
cd AegisSpGEMM
```
2. Colocar los artefactos pre-compilados dentro del directorio `AegisSpGEMM/Data/`:
   * `ag_news-384-ivf-csr.bin` (Índice binario IVF-CSR de 191 MB con vectores normalizados, centroides, listas invertidas y *Ground Truth*).
   * `AegisNeuralRouter.mlpackage` (Enrutador Neuronal Residual exportado en precisión `FP16` para el Apple Neural Engine). *(Si se ejecuta sin el `.mlpackage`, el motor activa automáticamente su respaldo matemático en el coprocesador AMX).*
3. Abrir el proyecto en Xcode y verificar en **Build Phases** del target `AegisSpGEMM` que `SpGEMMKernel.metal` esté incluido en **Compile Metal Sources** y que el archivo `AegisSpGEMM-Bridging-Header.h` esté enlazado en **Build Settings**.
4. Ejecutar con `Cmd + R` (para pruebas de trazabilidad y benchmark $N = 1,000$) o compilar el esquema en modo **Release (`-O`)** para perfilado de latencia mínima en silicio.

---

## Seguridad y Licencia
Desarrollado bajo principios estrictos de seguridad de memoria y Diseño por Contratos (DbC) como proyecto de tesis de maestría en Analítica de Datos y Sistemas Inteligentes en motores de búsqueda vectorial sobre arquitecturas heterogéneas. Consulta el archivo `LICENSE` para más detalles.
