//
//  MetalSpGEMMEngine.swift
//  AegisSpGEMM
//
//  Created for Phase 7.1 (Updated Phase 7.2 - Direct Scratchpad & Silicon Hardware Timers).
//  Strict Memory Standard: Direct command encoding over pre-filled UMA candidate streams.
//

import Foundation
import Metal

public enum MetalEngineError: Error {
    case libraryNotFound(String)
    case pipelineCreationFailed(String)
    case bufferAllocationFailed(String)
}

public final class MetalSpGEMMEngine {
    
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let compactSimilarityPipelineState: MTLComputePipelineState
    
    public init() throws {
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            throw MetalEngineError.libraryNotFound("Hardware Metal inalcanzable.")
        }
        self.device = defaultDevice
        
        guard let queue = device.makeCommandQueue() else {
            throw MetalEngineError.pipelineCreationFailed("No se pudo crear la Command Queue.")
        }
        self.commandQueue = queue
        
        guard let library = device.makeDefaultLibrary() else {
            throw MetalEngineError.libraryNotFound("No se encontró default.metallib.")
        }
        
        guard let funcCompact = library.makeFunction(name: "sparse_similarity_compact") else {
            throw MetalEngineError.pipelineCreationFailed("No se encontró sparse_similarity_compact.")
        }
        
        self.compactSimilarityPipelineState = try device.makeComputePipelineState(function: funcCompact)
    }
    
    /// @brief Ejecuta el kernel compacto sobre los candidatos ya depositados en `context.candidateIndicesBuffer`.
    /// @return Puntero Zero-Copy a los puntajes y tiempo puro de ejecución en silicio GPU (ms).
    public func executeCompactSearch(
        context: CSRHardwareContext,
        queryVector: [Float],
        numCandidates: Int
    ) throws -> (scores: UnsafeBufferPointer<Float>, gpuSiliconMs: Double) {
        
        guard queryVector.count == context.vectorDim else {
            throw MetalEngineError.pipelineCreationFailed("Dimensión de consulta incorrecta.")
        }
        
        let safeCandidates = min(numCandidates, context.maxCandidates)
        if safeCandidates <= 0 {
            return (UnsafeBufferPointer(start: nil, count: 0), 0.0)
        }
        
        let queryBytes = context.vectorDim * MemoryLayout<Float>.stride
        queryVector.withUnsafeBytes { src in
            if let base = src.baseAddress {
                memcpy(context.queryBuffer.contents(), base, queryBytes)
            }
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalEngineError.pipelineCreationFailed("Fallo en CommandBuffer.")
        }
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.pipelineCreationFailed("Fallo en ComputeEncoder.")
        }
        
        let threadWidth = compactSimilarityPipelineState.threadExecutionWidth
        let threadsPerTG = MTLSize(width: threadWidth, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (safeCandidates + threadWidth - 1) / threadWidth, height: 1, depth: 1)
        
        var vDim = UInt32(context.vectorDim)
        var nCand = UInt32(safeCandidates)
        
        encoder.setComputePipelineState(compactSimilarityPipelineState)
        encoder.setBuffer(context.queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(context.denseVectorsBuffer, offset: 0, index: 1)
        encoder.setBuffer(context.candidateIndicesBuffer, offset: 0, index: 2)
        encoder.setBuffer(context.resultsBuffer, offset: 0, index: 3)
        encoder.setBytes(&vDim, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&nCand, length: MemoryLayout<UInt32>.size, index: 5)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let gpuStart = commandBuffer.gpuStartTime
        let gpuEnd = commandBuffer.gpuEndTime
        let siliconMs = (gpuEnd > gpuStart && gpuStart > 0) ? (gpuEnd - gpuStart) * 1000.0 : 0.0
        
        let rawPtr = context.resultsBuffer.contents().bindMemory(to: Float.self, capacity: safeCandidates)
        return (UnsafeBufferPointer(start: rawPtr, count: safeCandidates), siliconMs)
    }
}
