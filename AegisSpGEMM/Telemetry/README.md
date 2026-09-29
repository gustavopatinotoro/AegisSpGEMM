#  Carpeta Métricas


Para lograr esto en el ecosistema de Apple Silicon, se debe dividir estas métricas en dos  categorías: **Las calculables en tiempo de ejecución (Software)** y **Las perfilables por hardware (OS/Instruments)**. El *sandbox* de macOS restringe el acceso directo a sensores de energía y contadores de caché desde código de usuario (Swift regular), pero existen herramientas del sistema para triangular esa información.


### 1. Métricas Algorítmicas y Analíticas (Implementables en Swift/Metal)

Se puede programar un motor de *Benchmarking* directamente en la base de código para obtener estas métricas en cada ejecución:

* **Recall (Exhaustividad):** Calculable al 100%. Implica ejecutar un "Ground Truth" (Fuerza bruta calculando el Dot Product contra el millón de vectores en CPU o en un kernel ingenuo de GPU) y comparar la intersección topológica de esos resultados con los resultados del sistema SpGEMM.
* **Percentiles P95 / P99:** Calculable. Generar un lote (batch) de 10,000 *queries* aleatorias, las despachamos, registramos sus latencias individuales, se ordenan y se  extraen los índices 9,500 (P95) y 9,900 (P99).
* **NNZ Procesados / Query:** Reemplaza los "hops". Se calcula sumando la cantidad de aristas (conexiones) que pertenecen a los nodos que el primer *Shader* (SpMV) activó. Representa qué porcentaje de la matriz CSR realmente tocó la GPU.
* **Dispatch Overhead:** Calculable usando el reloj de la CPU. Es la resta entre el tiempo justo antes de `commandBuffer.commit()` y el instante exacto en que la GPU comienza a procesar.
* **BW Efectivo Analítico:** Calculable. Fórmula: `((NNZ * 4 bytes) + (Nodos_Activos * 384 * 4 bytes)) / Tiempo_GPU_Segundos`. Esto dará los Gigabytes por segundo (GB/s) reales que el SpGEMM extrajo de la UMA.

### 2. Métricas de Silicio (Perfiladas vía Apple Instruments / OS)

Estas métricas requieren herramientas de bajo nivel fuera del compilador, ya que Apple bloquea la lectura de los PMU (Performance Monitoring Units) a nivel de usuario:

* **Hit Rate del SLC (System Level Cache):** Se obtiene ejecutando el binario a través de **Apple Instruments > Metal System Trace**. Allí se puede ver la utilización del SLC de 32MB y cómo los punteros CSR impactan la caché L2 de la GPU.
* **QPS/W y mJ/query (Energía):** Se mide utilizando la herramienta de terminal de macOS `sudo powermetrics --samplers gpu_power`. Al sincronizar las marcas de tiempo de `powermetrics` con el batch de P99, se puede extraer los Watts consumidos por la GPU de Apple durante la ventana de inferencia y dividirlo por el número de *queries*.


`BenchmarkSuite.swift`: Este módulo contiene la matemática forense para perfilar el orquestador 


