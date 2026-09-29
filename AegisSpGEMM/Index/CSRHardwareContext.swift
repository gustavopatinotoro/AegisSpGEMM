//
//  CSRHardwareContext.swift
//  AegisSpGEMM
//
//  Created for Phase 2 - CSR Compiler.
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
    
    /// Búfer UMA que contiene el índice de inicio de cada fila (estado de activación) en el tensor disperso.
    public let rowPointersBuffer: MTLBuffer
    
    /// Búfer UMA que contiene las conexiones (columnas) para las aristas del grafo.
    public let colIndicesBuffer: MTLBuffer
    
    /// Búfer UMA que contiene los embeddings (vectores densos) aplanados.
    public let denseVectorsBuffer: MTLBuffer
    
    /// @brief Total de nodos (documentos) en el índice.
    public let numNodes: Int
    
    /// @brief Dimensionalidad del espacio latente (ej. 384, 768).
    public let vectorDim: Int
    
    /// @brief Total de aristas (conexiones) en el grafo topológico.
    public let numEdges: Int
}
