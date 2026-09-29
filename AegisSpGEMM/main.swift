//
//  main.swift
//  AegisSpGEMM
//
//  Created for Phase 4 - Hybrid Pipeline.
//  Strict Memory Standard: DARPA Simulation and Telemetry Runner.
//

import Foundation

print("=========================================================")
print("      AegisSpGEMM Engine - Apple Silicon UMA Core        ")
print("===================================================gapt\n")

do {
    // 1. Inicialización del Sistema Maestro
    print("[System] Inicializando Subsistemas Metal y Compiladores...")
    let orchestrator = try SearchOrchestrator()
    print("[System] Subsistemas Online.\n")
    
    // 2. Simulación de Datos (Mocking)
    print("[Simulation] Generando Grafo Topológico Dummy...")
    let vectorDimension = 3
    
    // Grafo simulado de 5 nodos, ahora BIDIRECCIONAL (Simétrica)
    // Esto permite que el álgebra de Metal (Gather SpMV) propague la activación correctamente.
    let simulatedAdjacency: [[Int32]] = [
        [1, 2],     // Nodo 0 (Semilla) conecta con 1 y 2
        [0, 3, 4],  // Nodo 1 conecta de vuelta con 0, y empuja hacia 3 y 4
        [0],        // Nodo 2 conecta de vuelta con 0
        [1],        // Nodo 3 conecta de vuelta con 1
        [1]         // Nodo 4 conecta de vuelta con 1
    ]
    
    // Embeddings simulados (Vectores densos)
    let simulatedVectors: [[Float]] = [
        [0.1, 0.2, 0.3], // Nodo 0
        [0.4, 0.5, 0.6], // Nodo 1 (Alta similitud con Query A)
        [0.7, 0.8, 0.9], // Nodo 2 (Alta similitud con Query B)
        [1.0, 1.1, 1.2], // Nodo 3
        [1.3, 1.4, 1.5]  // Nodo 4
    ]
    
    // 3. Ingestión y Compilación a la Memoria Unificada (UMA Zero-Copy)
    try orchestrator.ingestIndex(adjacencyList: simulatedAdjacency, denseVectors: simulatedVectors, vectorDim: vectorDimension)
    print("")
    
    // 4. Ejecución del Kernel en GPU (Inferencia)
    // Supongamos que un usuario hizo una pregunta. El LLM/ANE la convirtió en este vector:
    let queryVector: [Float] = [0.45, 0.55, 0.65]
    
    // Queremos empezar a buscar desde el Nodo 0 (Entry Point de nuestro grafo)
    let entryPointNode = 0
    let topK_Results = 2
    
    print("[GPU-Dispatch] Despachando Query al Motor de Metal...")
    let results = try orchestrator.executeSearch(queryVector: queryVector, entryPoint: entryPointNode, topK: topK_Results)
    
    print("\n=========================================================")
    print("      RESULTADOS TOP-K OBTENIDOS DESDE EL SILICIO          ")
    print("=====================================================gapt\n")
    for (rank, result) in results.enumerated() {
        // Se observa que solo devuelve Nodos 1 y 2, porque el SpGEMM iluminó esas aristas
        // descartando el resto matemáticamente sin usar if/else.
        let formattedScore = String(format: "%.4f", result.score)
        print(" Rank \(rank + 1): Nodo [\(result.nodeId)] | Similitud Métrica: \(formattedScore)")
    }
    
} catch {
    print("\n[CRITICAL FAILURE] El motor abortó con el siguiente estado:")
    print(error)
}

print("\n[System] Secuencia finalizada. Memoria UMA liberada vía ARC.")

