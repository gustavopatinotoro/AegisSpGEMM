//
//  SearchOrchestrator.swift
//  AegisSpGEMM
//
//  Created for Phase 6 - IVF-CSR Pipeline.
//  Strict Memory Standard: Facade Pattern for UMA Lifecycle and Multi-Seed GPU Dispatch.
//

import Foundation

public final class SearchOrchestrator {
    
    private let compiler: CSRTopologyCompiler
    private let gpuEngine: MetalSpGEMMEngine
    private var hardwareContext: CSRHardwareContext?
    
    public init() throws {
        self.compiler = try CSRTopologyCompiler()
        self.gpuEngine = try MetalSpGEMMEngine()
        self.hardwareContext = nil
    }
    
    public func ingestIndex(
        rowPtr: [Int32],
        colIdx: [Int32],
        denseVectors: [[Float]],
        vectorDim: Int,
        numCentroids: Int
    ) throws {
        print("[Aegis-Info] Inicializando puente UMA con CSR nativo...")
        let startTime = CFAbsoluteTimeGetCurrent()
        
        self.hardwareContext = try compiler.compileToUMA(
            rowPtr: rowPtr,
            colIdx: colIdx,
            denseVectors: denseVectors,
            vectorDim: vectorDim,
            numCentroids: numCentroids
        )
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        print(String(format: "[Aegis-Success] Traspaso Zero-Copy completado en %.4f ms.", elapsed * 1000.0))
        print("[Aegis-Telemetry] Vectores: \(self.hardwareContext!.numVectors) | Centroides: \(self.hardwareContext!.numCentroids) | Dim: \(self.hardwareContext!.vectorDim)")
    }
    
    public func executeSearch(queryVector: [Float], entryPoints: [Int], topK: Int = 10) throws -> [(nodeId: Int, score: Float)] {
        guard let context = self.hardwareContext else {
            throw MetalEngineError.pipelineCreationFailed("El índice no ha sido ingerido en la UMA.")
        }
        
        // Ejecución en Silicio de Apple (SpMV Scatter + Búsqueda Densa)
        let rawScores = try gpuEngine.executeSearchStep(context: context, queryVector: queryVector, entryPoints: entryPoints)
        
        var scoredNodes = [(nodeId: Int, score: Float)]()
        scoredNodes.reserveCapacity(context.numVectors)
        
        // Fase de post-procesamiento en CPU: Filtrar y ordenar los resultados Top-K.
        for (idx, score) in rawScores.enumerated() {
            if score > -Float.infinity {
                scoredNodes.append((nodeId: idx, score: score))
            }
        }
        
        scoredNodes.sort { $0.score > $1.score }
        
        return Array(scoredNodes.prefix(topK))
    }
}
