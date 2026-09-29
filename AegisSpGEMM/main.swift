//
//  main.swift
//  AegisSpGEMM
//
//  Created for Phase 4 - Stress Testing (1 Million Nodes).
//  Strict Memory Standard: DARPA Simulation and Telemetry Runner.
//

import Foundation

print("=========================================================")
print("      AegisSpGEMM Engine - EXTREME STRESS TEST           ")
print("===================================================gapt\n")

do {
    // 1. Configuración de la Carga Útil (Payload)
    let totalNodes = 1_000_000   // 1 Millón de Nodos
    let vectorDimension = 384    // MiniLM standard
    let edgesPerNode = 32        // Topología HNSW M=32
    
    print("[System] Configurando simulación: \(totalNodes) nodos, \(vectorDimension) dims, \(edgesPerNode) aristas/nodo.")
    print("[System] Memoria UMA proyectada: ~1.66 GB. Iniciando generador sintético...\n")
    
    let genStartTime = CFAbsoluteTimeGetCurrent()
    
    // 2. Pre-asignación Estricta (Prevención de fragmentación de RAM)
    var simulatedAdjacency = [[Int32]](repeating: [], count: totalNodes)
    var simulatedVectors = [[Float]](repeating: [], count: totalNodes)
    
    // Generación de la "Aguja en el Pajar" (El vector de consulta)
    var queryVector = [Float]()
    queryVector.reserveCapacity(vectorDimension)
    for _ in 0..<vectorDimension {
        queryVector.append(Float.random(in: -1.0...1.0))
    }
    
    // Llenado masivo (Simulación de un grafo aleatorio disperso)
    for i in 0..<totalNodes {
        // Generar vector aleatorio
        var vec = [Float]()
        vec.reserveCapacity(vectorDimension)
        for _ in 0..<vectorDimension {
            vec.append(Float.random(in: -1.0...1.0))
        }
        simulatedVectors[i] = vec
        
        // Generar conexiones aleatorias (simulando un "Small World" probabilístico)
        // Nota: En un entorno aleatorio masivo bidireccional puro es costoso de generar en CPU,
        // por lo que generamos conexiones salientes, la ley de grandes números garantiza
        // que estadísticamente habrá recolección (Gather) suficiente en el SpMV.
        var edges = [Int32]()
        edges.reserveCapacity(edgesPerNode)
        for _ in 0..<edgesPerNode {
            edges.append(Int32.random(in: 0..<Int32(totalNodes)))
        }
        simulatedAdjacency[i] = edges
    }
    
    // INYECCIÓN DE CONTROL: Se Planta una coincidencia perfecta en el Nodo 999,999
    // y lo conectamos a la semilla (Nodo 0) para asegurar que la ruta topológica exista.
    simulatedVectors[999_999] = queryVector // Similitud Coseno máxima garantizada
    simulatedAdjacency[999_999].append(0)   // 999,999 mira al 0
    simulatedAdjacency[0].append(999_999)   // 0 mira al 999,999
    
    let genElapsed = CFAbsoluteTimeGetCurrent() - genStartTime
    print(String(format: "[Simulation] Grafo sintético masivo generado en %.2f segundos.\n", genElapsed))
    
    // 3. Inicialización del Orquestador
    print("[System] Inicializando Subsistemas Metal...")
    let orchestrator = try SearchOrchestrator()
    
    // 4. Ingestión y Compilación a la Memoria Unificada
    try orchestrator.ingestIndex(
        adjacencyList: simulatedAdjacency,
        denseVectors: simulatedVectors,
        vectorDim: vectorDimension
    )
    print("")
    
    // 5. Ejecución del Kernel en GPU (La verdadera prueba de estrés)
    let entryPointNode = 0
    let topK_Results = 3
    
    print("[GPU-Dispatch] Despachando \(totalNodes) vectores a la GPU...")
    let results = try orchestrator.executeSearch(queryVector: queryVector, entryPoint: entryPointNode, topK: topK_Results)
    
    print("\n=========================================================")
    print("      RESULTADOS TOP-K OBTENIDOS DESDE EL SILICIO          ")
    print("=====================================================gapt\n")
    for (rank, result) in results.enumerated() {
        let formattedScore = String(format: "%.4f", result.score)
        print(" Rank \(rank + 1): Nodo [\(result.nodeId)] | Similitud Métrica: \(formattedScore)")
    }
    
} catch {
    print("\n[CRITICAL FAILURE] El motor abortó:")
    print(error)
}

print("\n[System] Secuencia finalizada. Liberando 1.6GB de Memoria UMA...")
