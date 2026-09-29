# Aegis-SpGEMM Engine

![Apple Silicon](https://img.shields.io/badge/Hardware-Apple_Silicon_M--Series-black?logo=apple)
![Swift](https://img.shields.io/badge/Swift-6.0-FA7343?logo=swift)
![Metal](https://img.shields.io/badge/Metal-C++14-00599C?logo=c%2B%2B)
![Architecture](https://img.shields.io/badge/Architecture-UMA_Zero--Copy-success)
![Status](https://img.shields.io/badge/Status-V1.0.0_Core-blueviolet)

**Aegis-SpGEMM** es un motor experimental de búsqueda vectorial e inferencia topológica, diseñado con arquitectura de grado militar (DARPA-style) específicamente para los Sistemas en Chip (SoC) de Apple. 

El objetivo principal del motor es **destruir el cuello de botella del *Pointer-Chasing* en la CPU** al navegar grafos tipo HNSW (Hierarchical Navigable Small World). Para lograrlo, el sistema "aplasta" la topología del grafo en tensores CSR y delega la búsqueda a los núcleos de la GPU mediante **Multiplicación de Matriz-Vector Dispersa (SpMV)**.

## Innovaciones Arquitectónicas

1. **Zero-Copy en Memoria Unificada (UMA):** A diferencia de las arquitecturas CUDA/PCIe tradicionales, *Aegis* utiliza un puente Objective-C++ para invocar `MTLResourceStorageModeShared`. La CPU (Swift) y la GPU (Metal) acceden a los mismos bloques de RAM física, logrando transferencias instantáneas (0 bytes copiados) sin sobrepasar los límites de los SoCs de entrada (8GB de RAM).
2. **SpGEMM vs Heurística:** Las búsquedas vectoriales tradicionales usan bucles `while/for` y condicionales `if/else` (Branch Divergence) para saltar de nodo en nodo. *Aegis* propaga la señal de búsqueda evaluando toda la frontera de vecinos en un único ciclo de reloj matricial en la GPU.
3. **Diseño por Contratos (DbC):** El núcleo de memoria previene el agotamiento físico (Out-Of-Memory) pre-calculando las dimensiones topológicas mediante asignaciones estrictas (`reserveCapacity`) integradas al ciclo ARC de Swift.

## Estructura Táctica del Sistema (Fases)

El motor fue desarrollado bajo un protocolo estricto de aislamiento por fases:

- 📂 **`Core/` (Phase 1):** Gestión nativa de memoria UMA en `Objective-C++`. Garantiza el modo compartido y captura aserciones de hardware temprano.
- 📂 **`Index/` (Phase 2):** Compilador *Ahead-of-Time* (AoT) escrito en `Swift`. Traduce grafos de listas de adyacencia direccionales a matrices comprimidas **CSR** (Compressed Sparse Row) en 32-bits para ahorro masivo de RAM.
- 📂 **`Shaders/` (Phase 3):** Aceleradores matemáticos en `Metal (C++14)`. Contiene los Kernels paralelos que difunden la activación topológica y calculan la Similitud Coseno Condicional (Dot Product).
- 📂 **`Pipeline/` (Phase 4):** Orquestador híbrido que actúa como fachada (Facade) integrando ingesta, compilación, despacho en GPU y post-procesamiento en CPU (Extracción de Top-K).

## 🏛️ A Hombros de Gigantes (Fundamentos Teóricos)

Esta arquitectura no surge del vacío; **Aegis-SpGEMM** se construye sobre décadas de investigación fundamental en ciencias de la computación y álgebra lineal aplicada:

* **Estructuras de Recuperación (1960):** El trabajo pionero de **Edward Fredkin** con la invención del *Trie* (árbol de prefijos) [Trie Memory - ACM](https://dl.acm.org/doi/epdf/10.1145/367390.367400), que sentó las bases algorítmicas para el almacenamiento de memoria estructurada y la recuperación eficiente de información (Information Retrieval).
* **Redes de Pequeños Mundos y HNSW:** La revolucionaria investigación de **Yu. A. Malkov y D. A. Yashunin** [HNSW - IEEE](https://ieeexplore.ieee.org/document/8594636), quienes llevaron el concepto de los "Pequeños Mundos" al espacio vectorial creando *Hierarchical Navigable Small World (HNSW)*. Su demostración matemática de cómo los grafos probabilísticos multicapa pueden buscar en miles de millones de dimensiones inspiró directamente la topología del sistema.
* **Aceleración por Matrices Dispersas (Google Research):** El trabajo de Google con el algoritmo **ScaNN** [ScaNN](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/) y proyectos como **STATIC** [Vectorizing the Trie](https://arxiv.org/pdf/2602.22647), que demostraron empíricamente cómo las topologías complejas pueden aplanarse y resolverse masivamente en paralelo utilizando *Sparse Matrices*, validando el uso de SpGEMM como un reemplazo superior al *Pointer-Chasing* en hardware moderno.

## Requisitos del Entorno

- **Hardware:** Mac con procesador Apple Silicon (M1, M2, M3, M4...). Arquitectura de memoria unificada requerida.
- **IDE:** Xcode 27+ (Target configurado como macOS Command Line Tool).
- **Lenguajes:** Swift 6, Objective-C++, Metal Shading Language.

## Configuración y Despliegue

### 1. Clonar el repositorio

```bash
git clone https://github.com/gustavopatinotoro/AegisSpGEMM.git
cd AegisSpGEMM
```

### 2. Configuración en Xcode (Crítico)
Debido a la naturaleza CLI del proyecto, se debe asegurar que Xcode reconozca y ensamble los *Shaders* de Metal:
1. Abre el proyecto en Xcode.
2. Selecciona el *Target* principal (`AegisSpGEMM`).
3. Ve a la pestaña **Build Phases**.
4. Asegúrate de que `SpGEMMKernel.metal` esté incluido dentro de la fase de compilación (**Compile Sources** o **Compile Metal Rules**).
5. Confirma que la ruta hacia el *Bridging Header* (`AegisSpGEMM-Bridging-Header.h`) esté registrada en **Build Settings**.

### 3. Ejecutar Simulación
Se debe asegurar que el *Scheme* activo sea el Target principal (no el entorno de pruebas) y presiona `Cmd + R` para disparar el CLI.

## Telemetría de Ejemplo (Salida Estándar)

Al ejecutar la simulación de prueba en hardware M-Series, la propagación matricial arroja latencias puras (sin cachés de arranque) de inferencia paralela:

```text
=========================================================
      AegisSpGEMM Engine - Apple Silicon UMA Core        
=====================================================gapt

[System] Inicializando Subsistemas Metal y Compiladores...
[System] Subsistemas Online.

[Simulation] Generando Grafo Topológico Dummy...
[Aegis-Info] Iniciando compilación topológica a formato CSR...
[Aegis-Success] Compilación finalizada en 1.0570 ms.
[Aegis-Telemetry] Nodos: 5 | Aristas: 8 | Dimensión: 3

[GPU-Dispatch] Despachando Query al Motor de Metal...
[Aegis-Execution] SpGEMM Search Kernel ejecutado en 13.9170 ms.

=========================================================
      RESULTADOS TOP-K OBTENIDOS DESDE EL SILICIO        
=====================================================gapt
 Rank 1: Nodo [2] | Similitud Métrica: 1.3400
 Rank 2: Nodo [1] | Similitud Métrica: 0.8450

[System] Secuencia finalizada. Memoria UMA liberada vía ARC.
Program ended with exit code: 0
```

## Seguridad y Licencia
* Desarrollado bajo principios de seguridad de memoria estrictos. Este es un proyecto de investigación enfocado a optimización de motores de bases de datos vectoriales.

* Consulta el archivo `LICENSE` para más información.
