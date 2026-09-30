//
//  MetalSpGEMMEngine.swift
//  AegisSpGEMM
//
//  Created for Phase 7.1 - Zero-Copy Compacted Engine.
//  Strict Memory Standard: Direct command encoding over compact candidate streams.
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
    
    public func executeCompactSearch(
        context: CSRHardwareContext,
        queryVector: [Float],
        candidateIndices: [Int32]
    ) throws -> UnsafeBufferPointer<Float> {
        
        guard queryVector.count == context.vectorDim else {
            throw MetalEngineError.pipelineCreationFailed("Dimensión de consulta incorrecta.")
        }
        
        let numCandidates = candidateIndices.count
        if numCandidates == 0 {
            return UnsafeBufferPointer(start: nil, count: 0)
        }
        
        let queryBytes = context.vectorDim * MemoryLayout<Float>.stride
        let candidateBytes = numCandidates * MemoryLayout<Int32>.stride
        
        // 1. Copiar consulta y candidatos compactos al scratchpad UMA
        queryVector.withUnsafeBytes { src in
            if let base = src.baseAddress {
                memcpy(context.queryBuffer.contents(), base, queryBytes)
            }
        }
        
        candidateIndices.withUnsafeBytes { src in
            if let base = src.baseAddress {
                memcpy(context.candidateIndicesBuffer.contents(), base, candidateBytes)
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
        let threadgroups = MTLSize(width: (numCandidates + threadWidth - 1) / threadWidth, height: 1, depth: 1)
        
        var vDim = UInt32(context.vectorDim)
        var nCand = UInt32(numCandidates)
        
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
        
        let rawPtr = context.resultsBuffer.contents().bindMemory(to: Float.self, capacity: numCandidates)
        return UnsafeBufferPointer(start: rawPtr, count: numCandidates)
    }
}
