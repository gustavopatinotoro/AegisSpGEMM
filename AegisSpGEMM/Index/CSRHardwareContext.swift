//
//  CSRHardwareContext.swift
//  AegisSpGEMM
//
//  Created for Phase 2 - CSR Compiler (Updated Phase 6 - IVF-CSR).
//  Strict Memory Standard: Inmutable Data Container for UMA Buffers.
//

import Foundation
import Metal

/// @enum CSRCompilationError
/// @brief Define los estados de falla catastrófica durante la compilación topológica.
public enum CSRCompilationError: Error {
    case outOfMemory(String)
    case invalidGraphTopology(String)
    case hardwareBridgingFailed(String)
}

/// @struct CSRHardwareContext
/// @brief Contenedor seguro e inmutable de los tensores alojados en la Memoria Unificada (UMA).
///
/// @invariant Los búferes `rowPointersBuffer`, `colIndicesBuffer` y `denseVectorsBuffer` nunca son nulos tras la inicialización.
/// @invariant El ciclo de vida de estos búferes está atado al ciclo de vida de esta estructura (ARC de Swift).
public struct CSRHardwareContext {
    
    /// Búfer UMA que contiene el índice de inicio de cada fila (centroide) en el tensor disperso.
    public let rowPointersBuffer: MTLBuffer
    
    /// Búfer UMA que contiene los IDs de los vectores asignados a cada celda de Voronoi.
    public let colIndicesBuffer: MTLBuffer
    
    /// Búfer UMA que contiene los embeddings (vectores densos) aplanados.
    public let denseVectorsBuffer: MTLBuffer
    
    /// @brief Total de vectores base (documentos) en el índice.
    public let numVectors: Int
    
    /// @brief Total de centroides (celdas de Voronoi / filas CSR) en el índice.
    public let numCentroids: Int
    
    /// @brief Dimensionalidad del espacio latente (ej. 384, 768).
    public let vectorDim: Int
    
    /// @brief Total de asignaciones (conexiones centroide -> vector) en el grafo topológico.
    public let numEdges: Int
    
    public init(
        rowPointersBuffer: MTLBuffer,
        colIndicesBuffer: MTLBuffer,
        denseVectorsBuffer: MTLBuffer,
        numVectors: Int,
        numCentroids: Int,
        vectorDim: Int,
        numEdges: Int
    ) {
        self.rowPointersBuffer = rowPointersBuffer
        self.colIndicesBuffer = colIndicesBuffer
        self.denseVectorsBuffer = denseVectorsBuffer
        self.numVectors = numVectors
        self.numCentroids = numCentroids
        self.vectorDim = vectorDim
        self.numEdges = numEdges
    }
}
