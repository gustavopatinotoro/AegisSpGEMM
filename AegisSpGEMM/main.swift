//
//  main.swift
//  AegisSpGEMM
//
//  Created for Phase 5 - Telemetry & Academic Profiling.
//  Strict Memory Standard: P99, Recall and UMA Bandwidth validation.
//

import Foundation

print("=========================================================")
print("  AegisSpGEMM Engine - ACADEMIC TELEMETRY SUITE (PHASE 5)")
print("===================================================gapt\n")

do {
    let totalNodes = 10_000
    let vectorDimension = 128
    let edgesPerNode = 32
    
    print("[System] Generando entorno de validación (10k Nodos, 128 Dims)...")
    
    var simulatedAdjacency = [[Int32]](repeating: [], count: totalNodes)
    var simulatedVectors = [[Float]](repeating: [], count: totalNodes)
    
    // Generador Sintético Bidireccional Básico
    for i in 0..<totalNodes {
        var vec = [Float]()
        vec.reserveCapacity(vectorDimension)
        for _ in 0..<vectorDimension { vec.append(Float.random(in: -1.0...1.0)) }
        simulatedVectors[i] = vec
        
        var edges = [Int32]()
        edges.reserveCapacity(edgesPerNode)
        for _ in 0..<edgesPerNode { edges.append(Int32.random(in: 0..<Int32(totalNodes))) }
        simulatedAdjacency[i] = edges
    }
    
    // Asegurar conectividad en el nodo semilla (0)
    simulatedAdjacency[0] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in 1...10 { simulatedAdjacency[i].append(0) }
    
    let orchestrator = try SearchOrchestrator()
    try orchestrator.ingestIndex(adjacencyList: simulatedAdjacency, denseVectors: simulatedVectors, vectorDim: vectorDimension)
    
    // Generar Lote de Consultas (Batch)
    let numQueries = 1_000
    var batchQueries = [[Float]]()
    batchQueries.reserveCapacity(numQueries)
    for _ in 0..<numQueries {
        var q = [Float]()
        q.reserveCapacity(vectorDimension)
        for _ in 0..<vectorDimension { q.append(Float.random(in: -1.0...1.0)) }
        batchQueries.append(q)
    }
    
    // Desactivar logs de ejecución individual en el Orchestrator
    print("\n[Telemetry] Despachando ráfaga de \(numQueries) queries a la GPU...")
    
    let suite = BenchmarkSuite(orchestrator: orchestrator, vectors: simulatedVectors, adjacency: simulatedAdjacency)
    let report = try suite.runAcademicBenchmark(queries: batchQueries, entryPoint: 0, topK: 5)
    
    print("\n=========================================================")
    print("               REPORTE ACADÉMICO DE RENDIMIENTO            ")
    print("=======================================================gapt")
    print(String(format: " Recall@5 (Exactitud) : %.2f%%", report.recallTopK))
    print(String(format: " Latencia Promedio    : %.4f ms", report.avgLatencyMs))
    print(String(format: " Percentil P95        : %.4f ms", report.p95LatencyMs))
    print(String(format: " Percentil P99        : %.4f ms", report.p99LatencyMs))
    print(String(format: " Dispatch Overhead    : %.6f ms / query", report.dispatchOverheadMs))
    print(String(format: " Ancho de Banda (UMA) : %.2f GB/s", report.effectiveBandwidthGBs))
    print("========================================================gapt")
    
} catch {
    print("[CRITICAL FAILURE] \(error)")
}
