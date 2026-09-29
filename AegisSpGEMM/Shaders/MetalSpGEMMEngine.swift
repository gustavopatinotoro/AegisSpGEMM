//
//  MetalSpGEMMEngine.swift
//  AegisSpGEMM
//
//  Created for Phase 3 - GPU Acceleration.
//  Strict Memory Standard: Metal Command Encoder and Compute Pipeline State Management.
//

import Foundation
import Metal

/// @enum MetalEngineError
/// @brief Contratos de error para el orquestador GPU.
public enum MetalEngineError: Error {
    case libraryNotFound(String)
    case pipelineCreationFailed(String)
    case bufferAllocationFailed(String)
}

/// @class MetalSpGEMMEngine
/// @brief Despacha la ejecución paralela en Apple Silicon para SpGEMM y cálculos densos.
///
/// @invariant El `device`, `commandQueue` y los estados de tubería son instancias inmutables verificadas al inicializar.
public final class MetalSpGEMMEngine {
    
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let spgemmPipelineState: MTLComputePipelineState
    private let similarityPipelineState: MTLComputePipelineState
    
    /// @brief Inicializa el motor de Metal compilando los shaders.
    /// @pre El hardware debe soportar Metal. El archivo `.metal` debe estar compilado y visible en el Main Bundle.
    public init() throws {
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            throw MetalEngineError.libraryNotFound("Hardware Metal inalcanzable.")
        }
        self.device = defaultDevice
        
        guard let queue = device.makeCommandQueue() else {
            throw MetalEngineError.pipelineCreationFailed("No se pudo crear la Command Queue.")
        }
        self.commandQueue = queue
        
        // Carga la librería compilada de los shaders (.metal)
        guard let library = device.makeDefaultLibrary() else {
            throw MetalEngineError.libraryNotFound("No se encontró default.metallib. Revisa las Build Phases de Xcode.")
        }
        
        // Vincular kernels matemáticos
        guard let spgemmFunction = library.makeFunction(name: "spgemm_propagation"),
              let simFunction = library.makeFunction(name: "sparse_similarity") else {
            throw MetalEngineError.pipelineCreationFailed("No se encontraron las funciones en el archivo de Metal.")
        }
        
        self.spgemmPipelineState = try device.makeComputePipelineState(function: spgemmFunction)
        self.similarityPipelineState = try device.makeComputePipelineState(function: simFunction)
    }
    
    /// @brief Ejecuta un paso de navegación matemática en el grafo y extrae similitudes.
    ///
    /// @pre `queryVector.count` debe ser estrictamente igual a `context.vectorDim`.
    /// @pre `context` no debe contener búferes nulos.
    /// @post Devuelve un array con los *scores* de similitud para los nodos activados (los inactivos tendrán -INFINITY).
    public func executeSearchStep(
        context: CSRHardwareContext,
        queryVector: [Float],
        entryPointNode: Int
    ) throws -> [Float] {
        
        guard queryVector.count == context.vectorDim else {
            throw MetalEngineError.pipelineCreationFailed("Dimensión de consulta incorrecta. Esperada: \(context.vectorDim).")
        }
        
        // 1. Asignar búferes transaccionales Zero-Copy (UMA) para los vectores de estado
        let stateByteCount = context.numNodes * MemoryLayout<Float>.stride
        guard let currentStateBuffer = device.makeBuffer(length: stateByteCount, options: .storageModeShared),
              let nextStateBuffer = device.makeBuffer(length: stateByteCount, options: .storageModeShared),
              let queryBuffer = device.makeBuffer(bytes: queryVector, length: queryVector.count * MemoryLayout<Float>.stride, options: .storageModeShared),
              let resultsBuffer = device.makeBuffer(length: stateByteCount, options: .storageModeShared) else {
            throw MetalEngineError.bufferAllocationFailed("OOM: Fallo asignando tensores de estado SpGEMM.")
        }
        
        // 2. Inicializar el nodo semilla de entrada (Entry Point)
        let currentStatePointer = currentStateBuffer.contents().bindMemory(to: Float.self, capacity: context.numNodes)
        currentStatePointer[entryPointNode] = 1.0 // Activación matemática encendida
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalEngineError.pipelineCreationFailed("Fallo en CommandBuffer.")
        }
        
        // =========================================================
        // PIPELINE 1: SpGEMM (Ruteo Matemático en Grafo Disperso)
        // =========================================================
        guard let spgemmEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.pipelineCreationFailed("Fallo en ComputeEncoder.")
        }
        
        var totalNodes = UInt32(context.numNodes)
        
        spgemmEncoder.setComputePipelineState(spgemmPipelineState)
        spgemmEncoder.setBuffer(context.rowPointersBuffer, offset: 0, index: 0)
        spgemmEncoder.setBuffer(context.colIndicesBuffer, offset: 0, index: 1)
        spgemmEncoder.setBuffer(currentStateBuffer, offset: 0, index: 2)
        spgemmEncoder.setBuffer(nextStateBuffer, offset: 0, index: 3)
        spgemmEncoder.setBytes(&totalNodes, length: MemoryLayout<UInt32>.size, index: 4)
        
        // Optimización de Cuadrícula UMA (M-Series GPU)
        let threadWidth = spgemmPipelineState.threadExecutionWidth
        let threadsPerThreadgroup = MTLSize(width: threadWidth, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(width: (context.numNodes + threadWidth - 1) / threadWidth, height: 1, depth: 1)
        
        spgemmEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        spgemmEncoder.endEncoding()
        
        // =========================================================
        // PIPELINE 2: Búsqueda Densa en Subgrafo Activo
        // =========================================================
        guard let simEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalEngineError.pipelineCreationFailed("Fallo en ComputeEncoder.")
        }
        
        var vDim = UInt32(context.vectorDim)
        
        simEncoder.setComputePipelineState(similarityPipelineState)
        simEncoder.setBuffer(queryBuffer, offset: 0, index: 0)
        simEncoder.setBuffer(context.denseVectorsBuffer, offset: 0, index: 1)
        simEncoder.setBuffer(nextStateBuffer, offset: 0, index: 2) // Usamos el estado recién propagado
        simEncoder.setBuffer(resultsBuffer, offset: 0, index: 3)
        simEncoder.setBytes(&vDim, length: MemoryLayout<UInt32>.size, index: 4)
        simEncoder.setBytes(&totalNodes, length: MemoryLayout<UInt32>.size, index: 5)
        
        simEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        simEncoder.endEncoding()
        
        // Ejecución síncrona: CPU espera a que los núcleos GPU terminen el álgebra
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // 3. Extracción de Resultados UMA
        let rawPointer = resultsBuffer.contents().bindMemory(to: Float.self, capacity: context.numNodes)
        return Array(UnsafeBufferPointer(start: rawPointer, count: context.numNodes))
    }
}
