//
//  SearchOrchestrator.swift
//  AegisSpGEMM
//
//  Created for Phase 7.1 (Updated Phase 7.2 - AMX Matrix Routing & Direct UMA Slice Compaction).
//  Strict Memory Standard: Zero-Allocation Hot Loop (No Set<Int32>, Single-Pass vDSP_mmul).
//

import Foundation
import Accelerate
import Metal

public final class SearchOrchestrator {
    
    private let compiler: CSRTopologyCompiler
    private let gpuEngine: MetalSpGEMMEngine
    private var hardwareContext: CSRHardwareContext?
    
    /// Tiempo físico puro de silicio GPU registrado en la última consulta (ms).
    public private(set) var lastGpuSiliconMs: Double = 0.0
    
    public init() throws {
        self.compiler = try CSRTopologyCompiler()
        self.gpuEngine = try MetalSpGEMMEngine()
        self.hardwareContext = nil
    }
    
    public func ingestIndex(
        rowPtr: [Int32],
        colIdx: [Int32],
        flattenedVectors: [Float],
        flattenedCentroids: [Float],
        numVectors: Int,
        vectorDim: Int,
        numCentroids: Int
    ) throws {
        print("[Aegis-Info] Inicializando puente UMA con tensores planos y Scratchpad dinámico...")
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let context = try compiler.compileToUMA(
            rowPtr: rowPtr,
            colIdx: colIdx,
            flattenedVectors: flattenedVectors,
            flattenedCentroids: flattenedCentroids,
            numVectors: numVectors,
            vectorDim: vectorDim,
            numCentroids: numCentroids
        )
        self.hardwareContext = context
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        print(String(format: "[Aegis-Success] Traspaso Zero-Copy completado en %.4f ms.", elapsed * 1000.0))
        print("[Aegis-Telemetry] Vectores: \(context.numVectors) | Centroides: \(context.numCentroids) | Dim: \(context.vectorDim) | MaxCandidates UMA: \(context.maxCandidates)")
    }
    
    /// @brief Enrutamiento Grueso ejecutado en un único despacho matricial AMX (`vDSP_mmul`).
    public func selectTopCentroids(
        queryVector: [Float],
        nprobe: Int
    ) throws -> [Int] {
        guard let context = self.hardwareContext else {
            throw MetalEngineError.pipelineCreationFailed("El índice no ha sido ingerido en la UMA.")
        }
        
        let numCentroids = context.numCentroids
        let dim = context.vectorDim
        let centroidsPtr = context.centroidsBuffer.contents().bindMemory(to: Float.self, capacity: numCentroids * dim)
        let scoresPtr = context.centroidScoresBuffer.contents().bindMemory(to: Float.self, capacity: numCentroids)
        
        // Producto Matriz-Vector en un solo pase: [numCentroids x dim] * [dim x 1] = [numCentroids x 1]
        queryVector.withUnsafeBufferPointer { qPtr in
            guard let qBase = qPtr.baseAddress else { return }
            vDSP_mmul(
                centroidsPtr, 1,
                qBase, 1,
                scoresPtr, 1,
                vDSP_Length(numCentroids),
                1,
                vDSP_Length(dim)
            )
        }
        
        var centroidScores = [(index: Int, score: Float)]()
        centroidScores.reserveCapacity(numCentroids)
        for cIdx in 0..<numCentroids {
            centroidScores.append((index: cIdx, score: scoresPtr[cIdx]))
        }
        
        centroidScores.sort { $0.score > $1.score }
        return centroidScores.prefix(nprobe).map { $0.index }
    }
    
    /// @brief Compacta las listas invertidas directamente en UMA vía `memcpy` (sin `Set<Int32>`) y lanza el kernel SIMD.
    public func executeSearch(
        queryVector: [Float],
        entryPoints: [Int],
        topK: Int = 10
    ) throws -> [(nodeId: Int, score: Float)] {
        guard let context = self.hardwareContext else {
            throw MetalEngineError.pipelineCreationFailed("El índice no ha sido ingerido en la UMA.")
        }
        
        let rowPtr = context.rowPtr
        let maxCandidates = context.maxCandidates
        let destCandidatesPtr = context.candidateIndicesBuffer.contents().bindMemory(to: Int32.self, capacity: maxCandidates)
        
        var totalCandidates = 0
        
        // 1. Unión Disjunta Zero-Allocation: Copia directa de segmentos CSR contiguos hacia UMA
        context.colIdx.withUnsafeBufferPointer { colBuffer in
            guard let colBase = colBuffer.baseAddress else { return }
            for cIdx in entryPoints {
                if cIdx >= 0 && (cIdx + 1) < rowPtr.count {
                    let start = Int(rowPtr[cIdx])
                    let end = Int(rowPtr[cIdx + 1])
                    let count = end - start
                    if count > 0 {
                        let available = maxCandidates - totalCandidates
                        let copyCount = min(count, available)
                        if copyCount > 0 {
                            memcpy(
                                destCandidatesPtr.advanced(by: totalCandidates),
                                colBase.advanced(by: start),
                                copyCount * MemoryLayout<Int32>.stride
                            )
                            totalCandidates += copyCount
                        }
                        if totalCandidates >= maxCandidates {
                            break
                        }
                    }
                }
            }
        }
        
        // 2. Despacho Compacto a la GPU
        let (scoresBuffer, siliconMs) = try gpuEngine.executeCompactSearch(
            context: context,
            queryVector: queryVector,
            numCandidates: totalCandidates
        )
        self.lastGpuSiliconMs = siliconMs
        
        // 3. Extracción de Top-K leyendo directamente los punteros compartidos en UMA
        var scoredNodes = [(nodeId: Int, score: Float)]()
        scoredNodes.reserveCapacity(totalCandidates)
        
        if let baseScoresPtr = scoresBuffer.baseAddress {
            for i in 0..<totalCandidates {
                scoredNodes.append((nodeId: Int(destCandidatesPtr[i]), score: baseScoresPtr[i]))
            }
        }
        
        scoredNodes.sort { $0.score > $1.score }
        return Array(scoredNodes.prefix(topK))
    }
}
