//
//  main.swift
//  AegisSpGEMM
//
//  Created for Phase 8 - Heterogeneous ANE Neural Router + GPU Stream Compaction.
//  Strict Memory Standard: Tri-Processor Execution (ANE -> CPU AMX/CSR -> GPU Metal UMA).
//

import Foundation

print("=========================================================")
print("  AegisSpGEMM Engine - PHASE 8 (ANE COGNITIVE ROUTER)    ")
print("=========================================================\n")

do {
    let sourceFileURL = URL(fileURLWithPath: #filePath)
    let sourceDirURL = sourceFileURL.deletingLastPathComponent()
    let projectRootURL = sourceDirURL.deletingLastPathComponent()
    
    // 1. Localizar artefacto binario IVF-CSR (.bin)
    let binURL: URL
    if CommandLine.arguments.count > 1 {
        binURL = URL(fileURLWithPath: CommandLine.arguments[1])
    } else {
        let candidates = [

            sourceDirURL.appendingPathComponent("Data/ag_news-384-ivf-csr.bin"),
            projectRootURL.appendingPathComponent("Data/ag_news-384-ivf-csr.bin")
        ]
        binURL = candidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) ?? candidates[0]
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
    
    // 2. Localizar modelo CoreML para el Apple Neural Engine (AegisNeuralRouter.mlpackage)
    let modelCandidates = [
        sourceDirURL.appendingPathComponent("Data/AegisNeuralRouter.mlpackage"),
        projectRootURL.appendingPathComponent("Data/AegisNeuralRouter.mlpackage")
    ]
    
    if let foundModelURL = modelCandidates.first(where: { FileManager.default.fileExists(atPath: $0.path) }) {
        try orchestrator.loadANERouter(from: foundModelURL)
    } else {
        print("[ANE-Notice] 'AegisNeuralRouter.mlpackage' aún no detectado en data/. Activando Nucleus Probing sobre Softmax-AMX (τ=15.0) como respaldo.")
    }
    
    let batchSize = min(10, dataset.queries.count)
    let topKSearch = 10
    let staticNprobe = min(16, dataset.numCentroids)
    
    // =========================================================================
    // CORRIDA A: BASELINE ESTÁTICO FASE 7.2 (AMX vDSP_mmul, nprobe fijo = 16)
    // =========================================================================
    print("\n---------------------------------------------------------")
    print(" [A] BASELINE FASE 7.2: AMX Estático (nprobe fijo = \(staticNprobe))")
    print("---------------------------------------------------------")
    
    var staticRecallSum = 0.0
    var staticTotalMsSum = 0.0
    var staticSiliconMsSum = 0.0
    
    for i in 0..<batchSize {
        let query = dataset.queries[i]
        let expectedGT = dataset.groundTruth[i]
        
        let t0 = CFAbsoluteTimeGetCurrent()
        let winningCentroids = try orchestrator.selectTopCentroids(queryVector: query, nprobe: staticNprobe)
        let t1 = CFAbsoluteTimeGetCurrent()
        let results = try orchestrator.executeSearch(queryVector: query, entryPoints: winningCentroids, topK: topKSearch)
        let t2 = CFAbsoluteTimeGetCurrent()
        
        let totalMs = (t2 - t0) * 1000.0
        staticTotalMsSum += totalMs
        staticSiliconMsSum += orchestrator.lastGpuSiliconMs
        
        let intersection = Set(results.map { $0.nodeId }).intersection(Set(expectedGT.prefix(topKSearch)))
        staticRecallSum += Double(intersection.count) / Double(topKSearch)
    }
    
    print(String(format: " ↳ Baseline Promedio | Latencia Total: %.4f ms (Silicio GPU: %.4f ms) | Recall@10: %.2f%%",
                 staticTotalMsSum / Double(batchSize),
                 staticSiliconMsSum / Double(batchSize),
                 (staticRecallSum / Double(batchSize)) * 100.0))
    
    // =========================================================================
    // CORRIDA B: FASE 8 ADAPTATIVA (ANE Router + Nucleus Probing + GPU Metal)
    // =========================================================================
    let routerLabel = orchestrator.isANERouterLoaded ? "ANE CoreML FP16" : "AMX-Softmax Emulation"
    print("\n---------------------------------------------------------")
    print(" [B] FASE 8 COGNITIVA: \(routerLabel) + Adaptive Nucleus Probing")
    print("---------------------------------------------------------")
    
    var adaptRecallSum = 0.0
    var adaptTotalMsSum = 0.0
    var adaptRouteMsSum = 0.0
    var adaptGpuWallMsSum = 0.0
    var adaptSiliconMsSum = 0.0
    var adaptProbesSum = 0
    
    for i in 0..<batchSize {
        let query = dataset.queries[i]
        let expectedGT = dataset.groundTruth[i]
        
        // Etapa 1 (ANE / Adaptive Nucleus Probing): Selección dinámica de celdas según incertidumbre
        let decision = try orchestrator.selectTopCentroidsAdaptive(
            queryVector: query,
            confidenceThreshold: 0.965,
            minProbe: 10,
            maxProbe: 28
        )
        
        // Etapa 2 (CPU Direct memcpy + GPU Compact SIMD)
        let gpuStart = CFAbsoluteTimeGetCurrent()
        let results = try orchestrator.executeSearch(
            queryVector: query,
            entryPoints: decision.selectedCentroids,
            topK: topKSearch
        )
        let gpuEnd = CFAbsoluteTimeGetCurrent()
        
        let routeMs = decision.latencyMs
        let gpuWallMs = (gpuEnd - gpuStart) * 1000.0
        let siliconMs = orchestrator.lastGpuSiliconMs
        let totalQueryMs = routeMs + gpuWallMs
        
        adaptRouteMsSum += routeMs
        adaptGpuWallMsSum += gpuWallMs
        adaptSiliconMsSum += siliconMs
        adaptTotalMsSum += totalQueryMs
        adaptProbesSum += decision.adaptiveNprobe
        
        let resultIds = results.map { $0.nodeId }
        let expectedTopK = Array(expectedGT.prefix(topKSearch))
        let intersection = Set(resultIds).intersection(Set(expectedTopK))
        let queryRecall = Double(intersection.count) / Double(topKSearch)
        adaptRecallSum += queryRecall
        
        print(String(format: " ↳ Query %02d | nprobe: %02d (H=%.2f) | Total: %6.3f ms (Route: %5.3f ms | GPU: %5.3f ms [Silicon: %5.3f ms]) | Recall: %d/%d (%.0f%%)",
                     i + 1,
                     decision.adaptiveNprobe,
                     decision.shannonEntropy,
                     totalQueryMs,
                     routeMs,
                     gpuWallMs,
                     siliconMs,
                     intersection.count,
                     topKSearch,
                     queryRecall * 100.0))
    }
    
    print("\n=========================================================")
    print(String(format: " Modo de Enrutamiento Activo   : %@", routerLabel))
    print(String(format: " nprobe Adaptativo Promedio    : %.2f celdas (Rango: 10..28)", Double(adaptProbesSum) / Double(batchSize)))
    print(String(format: " Latencia Router (ANE/Softmax) : %.4f ms", adaptRouteMsSum / Double(batchSize)))
    print(String(format: " Latencia Compact + GPU Wall   : %.4f ms", adaptGpuWallMsSum / Double(batchSize)))
    print(String(format: " Tiempo Puro de Silicio GPU    : %.4f ms", adaptSiliconMsSum / Double(batchSize)))
    print(String(format: " Latencia Total Promedio       : %.4f ms", adaptTotalMsSum / Double(batchSize)))
    print(String(format: " Recall@10 (Fase 7.2 -> Fase 8): %.2f%% -> %.2f%%",
                 (staticRecallSum / Double(batchSize)) * 100.0,
                 (adaptRecallSum / Double(batchSize)) * 100.0))
    print("=========================================================")
    
} catch {
    print("[CRITICAL FAILURE] \(error)")
}
