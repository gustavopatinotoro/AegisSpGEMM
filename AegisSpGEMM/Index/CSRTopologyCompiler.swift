//
//  CSRTopologyCompiler.swift
//  AegisSpGEMM
//
//  Created for Phase 7.1 - Compacted Candidate Compiler.
//  Strict Memory Standard: Native CSR bridging with optimized candidate scratchpads.
//

import Foundation
import Metal

public final class CSRTopologyCompiler {
    
    private let memoryManager: UMAMemoryManager
    private let device: MTLDevice
    
    public init() throws {
        self.memoryManager = UMAMemoryManager()
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            throw CSRCompilationError.hardwareBridgingFailed("Dispositivo Metal no disponible.")
        }
        self.device = defaultDevice
    }
    
    public func compileToUMA(
        rowPtr: [Int32],
        colIdx: [Int32],
        denseVectors: [[Float]],
        vectorDim: Int,
        numCentroids: Int
    ) throws -> CSRHardwareContext {
        
        let numVectors = denseVectors.count
        let totalEdges = colIdx.count
        
        guard numVectors > 0, numCentroids > 0, vectorDim > 0 else {
            throw CSRCompilationError.invalidGraphTopology("Dimensiones inválidas en el dataset IVF-CSR.")
        }
        
        var flattenedVectors = [Float]()
        flattenedVectors.reserveCapacity(numVectors * vectorDim)
        for vector in denseVectors {
            flattenedVectors.append(contentsOf: vector)
        }
        
        guard let rowPointersUMA = rowPtr.withUnsafeBufferPointer({ ptr in
            memoryManager.allocateCSRRowPointers(ptr.baseAddress!, count: UInt(rowPtr.count))
        }) else {
            throw CSRCompilationError.outOfMemory("Fallo al asignar rowPointers UMA.")
        }
        
        guard let colIndicesUMA = colIdx.withUnsafeBufferPointer({ ptr in
            memoryManager.allocateCSRColumnIndices(ptr.baseAddress!, count: UInt(colIdx.count))
        }) else {
            throw CSRCompilationError.outOfMemory("Fallo al asignar colIndices UMA.")
        }
        
        guard let vectorsUMA = flattenedVectors.withUnsafeBufferPointer({ ptr in
            memoryManager.allocateDenseVectors(ptr.baseAddress!, count: UInt(flattenedVectors.count))
        }) else {
            throw CSRCompilationError.outOfMemory("Fallo al asignar denseVectors UMA.")
        }
        
        // Scratchpad dimensionado para hasta 16,000 candidatos simultáneos en el peor caso
        let maxCandidates = 16_000
        let candidateBytes = maxCandidates * MemoryLayout<Int32>.stride
        let resultsBytes = maxCandidates * MemoryLayout<Float>.stride
        let queryBytes = vectorDim * MemoryLayout<Float>.stride
        
        guard let queryUMA = device.makeBuffer(length: queryBytes, options: .storageModeShared),
              let candidateIndicesUMA = device.makeBuffer(length: candidateBytes, options: .storageModeShared),
              let resultsUMA = device.makeBuffer(length: resultsBytes, options: .storageModeShared) else {
            throw CSRCompilationError.outOfMemory("Fallo al pre-asignar el Scratchpad compacto en UMA.")
        }
        
        return CSRHardwareContext(
            rowPointersBuffer: rowPointersUMA,
            colIndicesBuffer: colIndicesUMA,
            denseVectorsBuffer: vectorsUMA,
            rowPtr: rowPtr,
            colIdx: colIdx,
            queryBuffer: queryUMA,
            candidateIndicesBuffer: candidateIndicesUMA,
            resultsBuffer: resultsUMA,
            numVectors: numVectors,
            numCentroids: numCentroids,
            vectorDim: vectorDim,
            numEdges: totalEdges
        )
    }
}
