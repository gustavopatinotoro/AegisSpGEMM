//
//  CSRTopologyCompiler.swift
//  AegisSpGEMM
//
//  Created for Phase 6 - IVF-CSR Compiler.
//  Strict Memory Standard: Native CSR bridging to UMA without CPU pointer-chasing.
//

import Foundation
import Metal

/// @class CSRTopologyCompiler
/// @brief Motor de compilación *Ahead-of-Time* (AoT). Transfiere el CSR nativo a memoria unificada.
public final class CSRTopologyCompiler {
    
    private let memoryManager: UMAMemoryManager
    
    public init() throws {
        self.memoryManager = UMAMemoryManager()
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
        
        guard numVectors > 0, numCentroids > 0 else {
            throw CSRCompilationError.invalidGraphTopology("El dataset IVF está vacío.")
        }
        
        // Aplanar los vectores para la GPU (Contiguous Array)
        var flattenedVectors = [Float]()
        flattenedVectors.reserveCapacity(numVectors * vectorDim)
        for vector in denseVectors {
            flattenedVectors.append(contentsOf: vector)
        }
        
        // Traspaso Zero-Copy al Hardware (Vía Objective-C++)
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
        
        return CSRHardwareContext(
            rowPointersBuffer: rowPointersUMA,
            colIndicesBuffer: colIndicesUMA,
            denseVectorsBuffer: vectorsUMA,
            numVectors: numVectors,
            numCentroids: numCentroids,
            vectorDim: vectorDim,
            numEdges: totalEdges
        )
    }
}
