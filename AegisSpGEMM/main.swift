//
//  main.swift
//  AegisSpGEMM
//
//  Created for Phase 6 - IVF-CSR Validation.
//  Strict Memory Standard: CPU-GPU Hybrid Execution (Coarse + Fine Search).
//

import Foundation

print("=========================================================")
print("  AegisSpGEMM Engine - PHASE 6 (IVF-CSR VALIDATION)      ")
print("=========================================================\n")

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
    let nprobe = 16 // Exploraremos ~1,800 candidatos topológicamente óptimos
    
    print("\n[Hybrid-Dispatch] Procesando \(batchSize) consultas (nprobe = \(nprobe))...")
    
    var totalRecall = 0.0
    var totalLatency = 0.0
    
    for i in 0..<batchSize {
        let query = dataset.queries[i]
        let expectedGT = dataset.groundTruth[i]
        
        let start = CFAbsoluteTimeGetCurrent()
        
        // Fase 1 (CPU): Enrutamiento Grueso
        var centroidScores = [(index: Int, score: Float)]()
        centroidScores.reserveCapacity(dataset.numCentroids)
        
        for (cIdx, centroid) in dataset.centroids.enumerated() {
            var score: Float = 0.0
            for d in 0..<dataset.dimension {
                score += query[d] * centroid[d]
            }
            centroidScores.append((cIdx, score))
        }
        
        centroidScores.sort { $0.score > $1.score }
        let winningCentroids = centroidScores.prefix(nprobe).map { $0.index }
        
        // Fase 2 (GPU): Búsqueda Fina Masiva
        let results = try orchestrator.executeSearch(queryVector: query, entryPoints: winningCentroids, topK: topKSearch)
        
        let end = CFAbsoluteTimeGetCurrent()
        let latencyMs = (end - start) * 1000.0
        totalLatency += latencyMs
        
        // Calcular Recall
        let resultIds = results.map { $0.nodeId }
        let expectedTopK = Array(expectedGT.prefix(topKSearch))
        let intersection = Set(resultIds).intersection(Set(expectedTopK))
        let queryRecall = Double(intersection.count) / Double(topKSearch)
        
        totalRecall += queryRecall
        
        print(String(format: " ↳ Query %02d | Latencia: %6.3f ms | Recall: %d/%d (%.0f%%)",
                     i+1, latencyMs, intersection.count, topKSearch, queryRecall * 100.0))
    }
    
    print("\n=========================================================")
    print(String(format: " ⏱️ Latencia Promedio (CPU+GPU) : %.4f ms", totalLatency / Double(batchSize)))
    print(String(format: " 📈 Recall Promedio (@%d)       : %.2f%%", topKSearch, (totalRecall / Double(batchSize)) * 100.0))
    print("=========================================================")
    
} catch {
    print("[CRITICAL FAILURE] \(error)")
}
