//
//  SearchOrchestrator.swift
//  AegisSpGEMM
//
//  Created for Phase 7.1 - Compacted Orchestrator.
//  Strict Memory Standard: CPU Posting List Union + Zero-Copy GPU Dispatch.
//

import Foundation
import Accelerate

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
        print("[Aegis-Info] Inicializando puente UMA con CSR y Scratchpad compacto...")
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
    
    public func selectTopCentroids(
        queryVector: [Float],
        centroids: [[Float]],
        nprobe: Int
    ) -> [Int] {
        let dim = vDSP_Length(queryVector.count)
        var centroidScores = [(index: Int, score: Float)]()
        centroidScores.reserveCapacity(centroids.count)
        
        queryVector.withUnsafeBufferPointer { qPtr in
            guard let qBase = qPtr.baseAddress else { return }
            for (cIdx, centroid) in centroids.enumerated() {
                var dot: Float = 0.0
                centroid.withUnsafeBufferPointer { cPtr in
                    if let cBase = cPtr.baseAddress {
                        vDSP_dotpr(qBase, 1, cBase, 1, &dot, dim)
                    }
                }
                centroidScores.append((index: cIdx, score: dot))
            }
        }
        
        centroidScores.sort { $0.score > $1.score }
        return centroidScores.prefix(nprobe).map { $0.index }
    }
    
    public func executeSearch(
        queryVector: [Float],
        entryPoints: [Int],
        topK: Int = 10
    ) throws -> [(nodeId: Int, score: Float)] {
        guard let context = self.hardwareContext else {
            throw MetalEngineError.pipelineCreationFailed("El índice no ha sido ingerido en la UMA.")
        }
        
        // 1. Unión de Posting Lists en CPU (O(nprobe * avg_cluster_size)) en microsegundos
        var candidateSet = Set<Int32>()
        candidateSet.reserveCapacity(entryPoints.count * 256)
        
        let rowPtr = context.rowPtr
        let colIdx = context.colIdx
        
        for cIdx in entryPoints {
            if cIdx >= 0 && (cIdx + 1) < rowPtr.count {
                let start = Int(rowPtr[cIdx])
                let end = Int(rowPtr[cIdx + 1])
                for idx in start..<end {
                    candidateSet.insert(colIdx[idx])
                }
            }
        }
        
        let candidateIndices = Array(candidateSet)
        
        // 2. Despacho Compacto a la GPU
        let scoresBuffer = try gpuEngine.executeCompactSearch(
            context: context,
            queryVector: queryVector,
            candidateIndices: candidateIndices
        )
        
        // 3. Post-procesamiento sobre el conjunto reducido de candidatos
        var scoredNodes = [(nodeId: Int, score: Float)]()
        scoredNodes.reserveCapacity(candidateIndices.count)
        
        if let basePtr = scoresBuffer.baseAddress {
            for i in 0..<candidateIndices.count {
                scoredNodes.append((nodeId: Int(candidateIndices[i]), score: basePtr[i]))
            }
        }
        
        scoredNodes.sort { $0.score > $1.score }
        return Array(scoredNodes.prefix(topK))
    }
}
