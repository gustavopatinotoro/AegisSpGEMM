#  Carpeta Shaders

Alojará el código fuente de Metal (.metal). Aquí se programará el núcleo de `SpGEMM` y las
operaciones de similitud matricial.
**Los dos componentes que destruyen el paradigma de pointer-chasing de HNSW:**
`Shaders/SpGEMMKernel.metal:` El código en C++14 para Metal. Este kernel se ejecutará simultáneamente en los miles de núcleos de tu GPU Apple Silicon. Propagará la activación topológica usando multiplicaciones de matrices dispersas.
`Shaders/MetalSpGEMMEngine.swift:`El orquestador táctico. Este archivo configurará el Compute Pipeline State, calculará el tamaño de los hilos (Threadgroups) y despachará el trabajo masivo a la GPU para que el cálculo ocurra en microsegundos.

## La aceleración matemática pura mediante la GPU.
En esta fase 3, se reemplaza por completo la lógica secuencial de la CPU `(if/else y pointer-chasing)`. Se convierte la navegación por el grafo en dos operaciones masivamente paralelas que se ejecutarán en los núcleos tensoriales del chip Serie-M de Apple:
**SpMV (Multiplicación Matriz-Vector Dispersa): ** La matriz CSR multiplicará un "vector de activación", propagando la señal a todos los vecinos en 1 ciclo matemático.
**MIPS / Similitud Coseno Condicional:** La GPU evaluará la distancia únicamente sobre los nodos que la operación SpMV acaba de iluminar.
