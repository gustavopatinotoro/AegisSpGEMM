//
//  CSRHardwareContext.swift
//  AegisSpGEMM
//
//  Created for Phase 7.1 (Updated Phase 7.2 - Universal Hardened Context).
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
    public let centroidsBuffer: MTLBuffer
    
    // Listas CSR en Host (CPU) para compactación O(nprobe) vía memcpy
    public let rowPtr: [Int32]
    public let colIdx: [Int32]
    
    // Scratchpad transaccional pre-asignado (Zero-Allocation en bucle caliente)
    public let queryBuffer: MTLBuffer
    public let candidateIndicesBuffer: MTLBuffer
    public let resultsBuffer: MTLBuffer
    public let centroidScoresBuffer: MTLBuffer
    
    // Invariantes Dimensionales y Guardarraíl de Candidatos
    public let numVectors: Int
    public let numCentroids: Int
    public let vectorDim: Int
    public let numEdges: Int
    public let maxCandidates: Int
    
    public init(
        rowPointersBuffer: MTLBuffer,
        colIndicesBuffer: MTLBuffer,
        denseVectorsBuffer: MTLBuffer,
        centroidsBuffer: MTLBuffer,
        rowPtr: [Int32],
        colIdx: [Int32],
        queryBuffer: MTLBuffer,
        candidateIndicesBuffer: MTLBuffer,
        resultsBuffer: MTLBuffer,
        centroidScoresBuffer: MTLBuffer,
        numVectors: Int,
        numCentroids: Int,
        vectorDim: Int,
        numEdges: Int,
        maxCandidates: Int
    ) {
        self.rowPointersBuffer = rowPointersBuffer
        self.colIndicesBuffer = colIndicesBuffer
        self.denseVectorsBuffer = denseVectorsBuffer
        self.centroidsBuffer = centroidsBuffer
        self.rowPtr = rowPtr
        self.colIdx = colIdx
        self.queryBuffer = queryBuffer
        self.candidateIndicesBuffer = candidateIndicesBuffer
        self.resultsBuffer = resultsBuffer
        self.centroidScoresBuffer = centroidScoresBuffer
        self.numVectors = numVectors
        self.numCentroids = numCentroids
        self.vectorDim = vectorDim
        self.numEdges = numEdges
        self.maxCandidates = maxCandidates
    }
}
