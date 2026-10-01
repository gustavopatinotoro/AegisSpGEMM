//
//  CSRTopologyCompiler.swift
//  AegisSpGEMM
//
//  Created for Phase 7.1 (Updated Phase 7.2 - Zero-Reallocation & Dynamic Scratchpad).
//  Strict Memory Standard: Direct Flat Bridging to Objective-C++ UMAMemoryManager.
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
        flattenedVectors: [Float],
        flattenedCentroids: [Float],
        numVectors: Int,
        vectorDim: Int,
        numCentroids: Int
    ) throws -> CSRHardwareContext {
        
        let totalEdges = colIdx.count
        
        guard numVectors > 0, numCentroids > 0, vectorDim > 0 else {
            throw CSRCompilationError.invalidGraphTopology("Dimensiones inválidas en el dataset IVF-CSR.")
        }
        guard flattenedVectors.count == numVectors * vectorDim else {
            throw CSRCompilationError.invalidGraphTopology("El tamaño de flattenedVectors no coincide con numVectors * vectorDim.")
        }
        guard flattenedCentroids.count == numCentroids * vectorDim else {
            throw CSRCompilationError.invalidGraphTopology("El tamaño de flattenedCentroids no coincide con numCentroids * vectorDim.")
        }
        guard rowPtr.count == numCentroids + 1 else {
            throw CSRCompilationError.invalidGraphTopology("rowPtr debe contener exactamente numCentroids + 1 elementos.")
        }
        
        // 1. Cálculo Dinámico de maxCandidates (Peor caso: unión de los 64 clústeres más densos)
        var clusterSizes = [Int]()
        clusterSizes.reserveCapacity(numCentroids)
        for c in 0..<numCentroids {
            let size = Int(rowPtr[c + 1] - rowPtr[c])
            clusterSizes.append(max(0, size))
        }
        clusterSizes.sort(by: >)
        
        let maxSupportedProbes = min(64, numCentroids)
        let worstCaseTop64 = clusterSizes.prefix(maxSupportedProbes).reduce(0, +)
        let dynamicMaxCandidates = min(numVectors, max(worstCaseTop64, min(numVectors, 16_384)))
        
        // 2. Traspaso Zero-Copy directo al gestor Objective-C++ (Sin re-aplanar arreglos en Swift)
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
        
        guard let centroidsUMA = flattenedCentroids.withUnsafeBufferPointer({ ptr in
            memoryManager.allocateDenseVectors(ptr.baseAddress!, count: UInt(flattenedCentroids.count))
        }) else {
            throw CSRCompilationError.outOfMemory("Fallo al asignar centroidsBuffer UMA.")
        }
        
        // 3. Scratchpad Transaccional Dimensionado Dinámicamente
        let candidateBytes = dynamicMaxCandidates * MemoryLayout<Int32>.stride
        let resultsBytes = dynamicMaxCandidates * MemoryLayout<Float>.stride
        let queryBytes = vectorDim * MemoryLayout<Float>.stride
        let centroidScoresBytes = numCentroids * MemoryLayout<Float>.stride
        
        guard let queryUMA = device.makeBuffer(length: queryBytes, options: .storageModeShared),
              let candidateIndicesUMA = device.makeBuffer(length: candidateBytes, options: .storageModeShared),
              let resultsUMA = device.makeBuffer(length: resultsBytes, options: .storageModeShared),
              let centroidScoresUMA = device.makeBuffer(length: centroidScoresBytes, options: .storageModeShared) else {
            throw CSRCompilationError.outOfMemory("Fallo al pre-asignar el Scratchpad compacto en UMA.")
        }
        
        return CSRHardwareContext(
            rowPointersBuffer: rowPointersUMA,
            colIndicesBuffer: colIndicesUMA,
            denseVectorsBuffer: vectorsUMA,
            centroidsBuffer: centroidsUMA,
            rowPtr: rowPtr,
            colIdx: colIdx,
            queryBuffer: queryUMA,
            candidateIndicesBuffer: candidateIndicesUMA,
            resultsBuffer: resultsUMA,
            centroidScoresBuffer: centroidScoresUMA,
            numVectors: numVectors,
            numCentroids: numCentroids,
            vectorDim: vectorDim,
            numEdges: totalEdges,
            maxCandidates: dynamicMaxCandidates
        )
    }
}
