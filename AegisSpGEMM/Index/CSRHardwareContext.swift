//
//  CSRHardwareContext.swift
//  AegisSpGEMM
//
//  Created for Phase 7.1 - Compacted Candidate Architecture.
//  Strict Memory Standard: Inmutable Data Container for UMA and Host Posting Lists.
//

import Foundation
import Metal

public enum CSRCompilationError: Error {
    case outOfMemory(String)
    case invalidGraphTopology(String)
    case hardwareBridgingFailed(String)
}

public struct CSRHardwareContext {
    
    // Búferes UMA Persistentes
    public let rowPointersBuffer: MTLBuffer
    public let colIndicesBuffer: MTLBuffer
    public let denseVectorsBuffer: MTLBuffer
    
    // Listas CSR en Host (CPU) para extracción O(1) de listas invertidas
    public let rowPtr: [Int32]
    public let colIdx: [Int32]
    
    // Scratchpad transaccional para consulta y resultados compactos
    public let queryBuffer: MTLBuffer
    public let candidateIndicesBuffer: MTLBuffer
    public let resultsBuffer: MTLBuffer
    
    public let numVectors: Int
    public let numCentroids: Int
    public let vectorDim: Int
    public let numEdges: Int
    
    public init(
        rowPointersBuffer: MTLBuffer,
        colIndicesBuffer: MTLBuffer,
        denseVectorsBuffer: MTLBuffer,
        rowPtr: [Int32],
        colIdx: [Int32],
        queryBuffer: MTLBuffer,
        candidateIndicesBuffer: MTLBuffer,
        resultsBuffer: MTLBuffer,
        numVectors: Int,
        numCentroids: Int,
        vectorDim: Int,
        numEdges: Int
    ) {
        self.rowPointersBuffer = rowPointersBuffer
        self.colIndicesBuffer = colIndicesBuffer
        self.denseVectorsBuffer = denseVectorsBuffer
        self.rowPtr = rowPtr
        self.colIdx = colIdx
        self.queryBuffer = queryBuffer
        self.candidateIndicesBuffer = candidateIndicesBuffer
        self.resultsBuffer = resultsBuffer
        self.numVectors = numVectors
        self.numCentroids = numCentroids
        self.vectorDim = vectorDim
        self.numEdges = numEdges
    }
}
