//
//  SearchOrchestrator.swift
//  AegisSpGEMM
//
//  Created for Phase 7.1 (Updated Phase 8 - Heterogeneous ANE + AMX + Metal UMA Orchestrator).
//  Strict Memory Standard: Zero-Allocation Hot Loop & Adaptive Nucleus Probing.
//

import Foundation
import Accelerate
import Metal

public final class SearchOrchestrator {
    
    private let compiler: CSRTopologyCompiler
    private let gpuEngine: MetalSpGEMMEngine
    private var hardwareContext: CSRHardwareContext?
    private var aneRouter: ANENeuralRouter?
    
    /// Tiempo físico puro de silicio GPU registrado en la última consulta (ms).
    public private(set) var lastGpuSiliconMs: Double = 0.0
    
    /// Indica si el modelo CoreML está cargado y activo en el Apple Neural Engine.
    public var isANERouterLoaded: Bool {
        return aneRouter != nil
    }
    
    public init() throws {
        self.compiler = try CSRTopologyCompiler()
        self.gpuEngine = try MetalSpGEMMEngine()
        self.hardwareContext = nil
        self.aneRouter = nil
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
    
    /// @brief Conecta y compila el modelo CoreML (.mlpackage o .mlmodelc) en el Apple Neural Engine (ANE).
    public func loadANERouter(from modelURL: URL) throws {
        guard let context = self.hardwareContext else {
            throw MetalEngineError.pipelineCreationFailed("Debes ingerir el índice en la UMA antes de inicializar el ANE Router.")
        }
        print("[ANE-Info] Compilando e inyectando '\(modelURL.lastPathComponent)' en el Apple Neural Engine...")
        let start = CFAbsoluteTimeGetCurrent()
        self.aneRouter = try ANENeuralRouter(
            modelURL: modelURL,
            vectorDim: context.vectorDim,
            numCentroids: context.numCentroids
        )
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        print(String(format: "[ANE-Success] Enrutador Neuronal ANE online en %.2f ms (ComputeUnits: .cpuAndNeuralEngine).", elapsed))
    }
    
    /// @brief Enrutamiento Grueso Estático ejecutado en un único despacho matricial AMX (`vDSP_mmul`).
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
    
    /// @brief Enrutamiento Neuronal Adaptativo (Fase 8): Usa el ANE si el .mlpackage está presente,
    /// o ejecuta un respaldo de Softmax con Temperatura en el coprocesador AMX.
    public func selectTopCentroidsAdaptive(
        queryVector: [Float],
        confidenceThreshold: Float = 0.965,
        minProbe: Int = 10,
        maxProbe: Int = 28,
        fallbackTemperature: Float = 15.0
    ) throws -> AdaptiveRoutingDecision {
        
        // 1. Camino Primario: Inferencia Física en el Apple Neural Engine (ANE)
        if let router = self.aneRouter {
            return try router.routeAdaptively(
                queryVector: queryVector,
                confidenceThreshold: confidenceThreshold,
                minProbe: minProbe,
                maxProbe: maxProbe
            )
        }
        
        // 2. Camino de Respaldo (Si aún no se ha copiado AegisNeuralRouter.mlpackage):
        // Calcula logits en AMX (vDSP_mmul) y aplica Softmax con temperatura para distribución de probabilidad
        guard let context = self.hardwareContext else {
            throw MetalEngineError.pipelineCreationFailed("El índice no ha sido ingerido en la UMA.")
        }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let numCentroids = context.numCentroids
        let dim = context.vectorDim
        let centroidsPtr = context.centroidsBuffer.contents().bindMemory(to: Float.self, capacity: numCentroids * dim)
        let scoresPtr = context.centroidScoresBuffer.contents().bindMemory(to: Float.self, capacity: numCentroids)
        
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
        
        var maxLogit: Float = -Float.infinity
        for i in 0..<numCentroids {
            let scaled = scoresPtr[i] * fallbackTemperature
            scoresPtr[i] = scaled
            if scaled > maxLogit {
                maxLogit = scaled
            }
        }
        
        var probs = [Float](repeating: 0.0, count: numCentroids)
        var sumExp: Float = 0.0
        for i in 0..<numCentroids {
            let e = exp(scoresPtr[i] - maxLogit)
            probs[i] = e
            sumExp += e
        }
        
        let invSum = sumExp > 0 ? (1.0 / sumExp) : 1.0
        for i in 0..<numCentroids {
            probs[i] *= invSum
        }
        
        return ANENeuralRouter.computeNucleusProbing(
            probabilities: probs,
            confidenceThreshold: confidenceThreshold,
            minProbe: minProbe,
            maxProbe: maxProbe,
            startTime: startTime,
            usedHardwareANE: false
        )
    }
    
    /// @brief Compacta las listas invertidas directamente en UMA vía `memcpy` y lanza el kernel SIMD.
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
        
        let (scoresBuffer, siliconMs) = try gpuEngine.executeCompactSearch(
            context: context,
            queryVector: queryVector,
            numCandidates: totalCandidates
        )
        self.lastGpuSiliconMs = siliconMs
        
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
