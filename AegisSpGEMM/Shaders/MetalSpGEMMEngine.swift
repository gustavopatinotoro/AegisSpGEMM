//
//  MetalSpGEMMEngine.swift
//  AegisSpGEMM
//
//  Created for Phase 6 - IVF-CSR GPU Acceleration.
//  Strict Memory Standard: Asymmetric State Buffers and Gather/Scatter Metal Pipelines.
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
    private let spgemmPipelineState: MTLComputePipelineState
    private let similarityPipelineState: MTLComputePipelineState
    
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
        
        // Enlace de los nuevos kernels IVF
        guard let spgemmFunction = library.makeFunction(name: "spgemm_ivf_propagation"),
              let simFunction = library.makeFunction(name: "sparse_similarity") else {
            throw MetalEngineError.pipelineCreationFailed("No se encontraron las funciones en el archivo de Metal.")
        }
        
        self.spgemmPipelineState = try device.makeComputePipelineState(function: spgemmFunction)
        self.similarityPipelineState = try device.makeComputePipelineState(function: simFunction)
    }
    
    public func executeSearchStep(
        context: CSRHardwareContext,
        queryVector: [Float],
        entryPoints: [Int]
    ) throws -> [Float] {
        
        guard queryVector.count == context.vectorDim else {
            throw MetalEngineError.pipelineCreationFailed("Dimensión incorrecta.")
        }
        
        let centroidStateByteCount = context.numCentroids * MemoryLayout<Float>.stride
        let vectorStateByteCount = context.numVectors * MemoryLayout<Float>.stride
        
        guard let centroidStateBuffer = device.makeBuffer(length: centroidStateByteCount, options: .storageModeShared),
              let vectorStateBuffer = device.makeBuffer(length: vectorStateByteCount, options: .storageModeShared),
              let queryBuffer = device.makeBuffer(bytes: queryVector, length: queryVector.count * MemoryLayout<Float>.stride, options: .storageModeShared),
              let resultsBuffer = device.makeBuffer(length: vectorStateByteCount, options: .storageModeShared) else {
            throw MetalEngineError.bufferAllocationFailed("OOM: Fallo asignando tensores UMA.")
        }
        
        // Inicializar las múltiples semillas (nprobe)
        let centroidStatePointer = centroidStateBuffer.contents().bindMemory(to: Float.self, capacity: context.numCentroids)
        for ep in entryPoints {
            centroidStatePointer[ep] = 1.0
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalEngineError.pipelineCreationFailed("Fallo en CommandBuffer.")
        }
        
        let threadWidth = spgemmPipelineState.threadExecutionWidth
        let threadsPerThreadgroup = MTLSize(width: threadWidth, height: 1, depth: 1)
        
        // =========================================================
        // PIPELINE 1: SpGEMM Scatter (Centroides -> Vectores)
        // =========================================================
        guard let spgemmEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.pipelineCreationFailed("Fallo en ComputeEncoder 1.")
        }
        
        var cNodes = UInt32(context.numCentroids)
        let tgGridCentroids = MTLSize(width: (context.numCentroids + threadWidth - 1) / threadWidth, height: 1, depth: 1)
        
        spgemmEncoder.setComputePipelineState(spgemmPipelineState)
        spgemmEncoder.setBuffer(context.rowPointersBuffer, offset: 0, index: 0)
        spgemmEncoder.setBuffer(context.colIndicesBuffer, offset: 0, index: 1)
        spgemmEncoder.setBuffer(centroidStateBuffer, offset: 0, index: 2)
        spgemmEncoder.setBuffer(vectorStateBuffer, offset: 0, index: 3)
        spgemmEncoder.setBytes(&cNodes, length: MemoryLayout<UInt32>.size, index: 4)
        
        spgemmEncoder.dispatchThreadgroups(tgGridCentroids, threadsPerThreadgroup: threadsPerThreadgroup)
        spgemmEncoder.endEncoding()
        
        // =========================================================
        // PIPELINE 2: Búsqueda Densa en Subgrafo Activo
        // =========================================================
        guard let simEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.pipelineCreationFailed("Fallo en ComputeEncoder 2.")
        }
        
        var vNodes = UInt32(context.numVectors)
        var vDim = UInt32(context.vectorDim)
        let tgGridVectors = MTLSize(width: (context.numVectors + threadWidth - 1) / threadWidth, height: 1, depth: 1)
        
        simEncoder.setComputePipelineState(similarityPipelineState)
        simEncoder.setBuffer(queryBuffer, offset: 0, index: 0)
        simEncoder.setBuffer(context.denseVectorsBuffer, offset: 0, index: 1)
        simEncoder.setBuffer(vectorStateBuffer, offset: 0, index: 2)
        simEncoder.setBuffer(resultsBuffer, offset: 0, index: 3)
        simEncoder.setBytes(&vDim, length: MemoryLayout<UInt32>.size, index: 4)
        simEncoder.setBytes(&vNodes, length: MemoryLayout<UInt32>.size, index: 5)
        
        simEncoder.dispatchThreadgroups(tgGridVectors, threadsPerThreadgroup: threadsPerThreadgroup)
        simEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let rawPointer = resultsBuffer.contents().bindMemory(to: Float.self, capacity: context.numVectors)
        return Array(UnsafeBufferPointer(start: rawPointer, count: context.numVectors))
    }
}
