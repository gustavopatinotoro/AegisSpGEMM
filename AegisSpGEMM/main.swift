//
//  main.swift
//  AegisSpGEMM
//
//  Created for Phase 7.1 - Compacted Execution Validation.
//  Strict Memory Standard: Stream Compaction Verification (Sub-3ms target).
//

import Foundation

print("=========================================================")
print("  AegisSpGEMM Engine - PHASE 7.1 (COMPACTED SILICON)     ")
print("===================================================gapt\n")

do {
    let sourceFileURL = URL(fileURLWithPath: #filePath)
    let sourceDirURL = sourceFileURL.deletingLastPathComponent()
    
    var binURL = sourceDirURL.appendingPathComponent("data/ag_news-384-ivf-csr.bin")
    if !FileManager.default.fileExists(atPath: binURL.path) {
        let projectRootURL = sourceDirURL.deletingLastPathComponent()
        binURL = projectRootURL.appendingPathComponent("data/ag_news-384-ivf-csr.bin")
    }
    
    print("[System] Localizando artefacto en: \(binURL.path)")
    
    let dataset = try BinaryLoader.loadAGNews(from: binURL)
    
    let orchestrator = try SearchOrchestrator()
    try orchestrator.ingestIndex(
        rowPtr: dataset.rowPtr,
        colIdx: dataset.colIdx,
        denseVectors: dataset.vectors,
        vectorDim: dataset.dimension,
        numCentroids: dataset.numCentroids
    )
    
    let batchSize = 10
    let topKSearch = 10
    let nprobe = 16
    
    print("\n[Hybrid-Dispatch] Procesando \(batchSize) consultas con Stream Compaction...")
    
    var totalRecall = 0.0
    var totalLatencyMs = 0.0
    var totalCpuMs = 0.0
    var totalGpuMs = 0.0
    
    for i in 0..<batchSize {
        let query = dataset.queries[i]
        let expectedGT = dataset.groundTruth[i]
        
        let cpuStart = CFAbsoluteTimeGetCurrent()
        let winningCentroids = orchestrator.selectTopCentroids(
            queryVector: query,
            centroids: dataset.centroids,
            nprobe: nprobe
        )
        let cpuEnd = CFAbsoluteTimeGetCurrent()
        
        let gpuStart = CFAbsoluteTimeGetCurrent()
        let results = try orchestrator.executeSearch(
            queryVector: query,
            entryPoints: winningCentroids,
            topK: topKSearch
        )
        let gpuEnd = CFAbsoluteTimeGetCurrent()
        
        let cpuMs = (cpuEnd - cpuStart) * 1000.0
        let gpuMs = (gpuEnd - gpuStart) * 1000.0
        let totalQueryMs = cpuMs + gpuMs
        
        totalCpuMs += cpuMs
        totalGpuMs += gpuMs
        totalLatencyMs += totalQueryMs
        
        let resultIds = results.map { $0.nodeId }
        let expectedTopK = Array(expectedGT.prefix(topKSearch))
        let intersection = Set(resultIds).intersection(Set(expectedTopK))
        let queryRecall = Double(intersection.count) / Double(topKSearch)
        
        totalRecall += queryRecall
        
        print(String(format: " ↳ Query %02d | Total: %6.3f ms (CPU Union: %5.3f ms | GPU Compact: %5.3f ms) | Recall: %d/%d (%.0f%%)",
                     i + 1, totalQueryMs, cpuMs, gpuMs, intersection.count, topKSearch, queryRecall * 100.0))
    }
    
    print("\n=========================================================")
    print(String(format: " Latencia CPU (Union Posting) : %.4f ms", totalCpuMs / Double(batchSize)))
    print(String(format: " Latencia GPU (Compact SIMD)  : %.4f ms", totalGpuMs / Double(batchSize)))
    print(String(format: " Latencia Total Promedio      : %.4f ms", totalLatencyMs / Double(batchSize)))
    print(String(format: " Recall Promedio (@%d)         : %.2f%%", topKSearch, (totalRecall / Double(batchSize)) * 100.0))
    print("======================================================gapt")
    
} catch {
    print("[CRITICAL FAILURE] \(error)")
}
