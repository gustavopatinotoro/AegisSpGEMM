//
//  main.swift
//  AegisSpGEMM
//
//  Created for Phase 7.2 - Universal Hardened Pipeline Validation.
//  Strict Memory Standard: Zero-Fragmentation Ingestion, AMX vDSP_mmul & Direct UMA Compaction.
//

import Foundation

print("=========================================================")
print("  AegisSpGEMM Engine - PHASE 7.2 (UNIVERSAL HARDENED)    ")
print("=========================================================\n")

do {
    let sourceFileURL = URL(fileURLWithPath: #filePath)
    let sourceDirURL = sourceFileURL.deletingLastPathComponent()
    
    // Permite pasar cualquier archivo .bin por CLI o usa AG News por defecto
    let binURL: URL
    if CommandLine.arguments.count > 1 {
        binURL = URL(fileURLWithPath: CommandLine.arguments[1])
    } else {
        var defaultURL = sourceDirURL.appendingPathComponent("data/ag_news-384-ivf-csr.bin")
        if !FileManager.default.fileExists(atPath: defaultURL.path) {
            let projectRootURL = sourceDirURL.deletingLastPathComponent()
            defaultURL = projectRootURL.appendingPathComponent("data/ag_news-384-ivf-csr.bin")
        }
        binURL = defaultURL
    }
    
    print("[System] Localizando artefacto en: \(binURL.path)")
    
    let dataset = try BinaryLoader.loadDataset(from: binURL)
    
    let orchestrator = try SearchOrchestrator()
    try orchestrator.ingestIndex(
        rowPtr: dataset.rowPtr,
        colIdx: dataset.colIdx,
        flattenedVectors: dataset.flattenedVectors,
        flattenedCentroids: dataset.flattenedCentroids,
        numVectors: dataset.numVectors,
        vectorDim: dataset.dimension,
        numCentroids: dataset.numCentroids
    )
    
    let batchSize = min(10, dataset.queries.count)
    let topKSearch = 10
    let nprobe = min(16, dataset.numCentroids)
    
    print("\n[Hybrid-Dispatch] Procesando \(batchSize) consultas (AMX vDSP_mmul + Direct UMA memcpy)...")
    
    var totalRecall = 0.0
    var totalLatencyMs = 0.0
    var totalCpuMs = 0.0
    var totalGpuWallMs = 0.0
    var totalGpuSiliconMs = 0.0
    
    for i in 0..<batchSize {
        let query = dataset.queries[i]
        let expectedGT = dataset.groundTruth[i]
        
        // Fase 1: Enrutamiento Grueso Matricial en un solo pase AMX (vDSP_mmul)
        let cpuStart = CFAbsoluteTimeGetCurrent()
        let winningCentroids = try orchestrator.selectTopCentroids(
            queryVector: query,
            nprobe: nprobe
        )
        let cpuEnd = CFAbsoluteTimeGetCurrent()
        
        // Fase 2: Compactación Directa memcpy + Similitud Compacta en GPU
        let gpuStart = CFAbsoluteTimeGetCurrent()
        let results = try orchestrator.executeSearch(
            queryVector: query,
            entryPoints: winningCentroids,
            topK: topKSearch
        )
        let gpuEnd = CFAbsoluteTimeGetCurrent()
        
        let cpuMs = (cpuEnd - cpuStart) * 1000.0
        let gpuWallMs = (gpuEnd - gpuStart) * 1000.0
        let gpuSiliconMs = orchestrator.lastGpuSiliconMs
        let totalQueryMs = cpuMs + gpuWallMs
        
        totalCpuMs += cpuMs
        totalGpuWallMs += gpuWallMs
        totalGpuSiliconMs += gpuSiliconMs
        totalLatencyMs += totalQueryMs
        
        let resultIds = results.map { $0.nodeId }
        let expectedTopK = Array(expectedGT.prefix(topKSearch))
        let intersection = Set(resultIds).intersection(Set(expectedTopK))
        let queryRecall = Double(intersection.count) / Double(topKSearch)
        
        totalRecall += queryRecall
        
        print(String(format: " ↳ Query %02d | Total: %6.3f ms (AMX Route: %5.3f ms | Compact+GPU: %5.3f ms [Silicon: %5.3f ms]) | Recall: %d/%d (%.0f%%)",
                     i + 1, totalQueryMs, cpuMs, gpuWallMs, gpuSiliconMs, intersection.count, topKSearch, queryRecall * 100.0))
    }
    
    print("\n=========================================================")
    print(String(format: " Latencia CPU (AMX vDSP_mmul)  : %.4f ms", totalCpuMs / Double(batchSize)))
    print(String(format: " Latencia Compact + GPU Wall   : %.4f ms", totalGpuWallMs / Double(batchSize)))
    print(String(format: " Tiempo Puro de Silicio GPU    : %.4f ms", totalGpuSiliconMs / Double(batchSize)))
    print(String(format: " Latencia Total Promedio       : %.4f ms", totalLatencyMs / Double(batchSize)))
    print(String(format: " Recall Promedio (@%d)         : %.2f%%", topKSearch, (totalRecall / Double(batchSize)) * 100.0))
    print("=========================================================")
    
} catch {
    print("[CRITICAL FAILURE] \(error)")
}
