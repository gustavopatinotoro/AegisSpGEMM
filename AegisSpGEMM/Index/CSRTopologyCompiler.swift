//
//  CSRTopologyCompiler.swift
//  AegisSpGEMM
//
//  Created for Phase 2 - CSR Compiler.
//  Strict Memory Standard: Algorithmic transformation from Adjacency List to CSR format.
//

import Foundation

/// @class CSRTopologyCompiler
/// @brief Motor de compilación *Ahead-of-Time* (AoT). Traduce topologías de grafos (ej. HNSW Nivel 0) a formato CSR.
///
/// @invariant El compilador delega de forma segura la asignación a `UMAMemoryManager`.
public final class CSRTopologyCompiler {
    
    // Instancia del puente de memoria Objective-C++ creado en la Fase 1
    private let memoryManager: UMAMemoryManager
    
    /// @brief Inicializa el compilador y verifica el puente con la memoria de bajo nivel.
    /// @pre La clase UMAMemoryManager debe inicializarse sin fallos.
    public init() throws {
        self.memoryManager = UMAMemoryManager()
        // Nota: UMAMemoryManager volcará a stderr si no hay Metal UMA, pero continuará si es simulable.
    }
    
    /// @brief Compila listas de adyacencia y vectores en arreglos CSR y los aloja en la UMA de la GPU.
    ///
    /// @pre `adjacencyList` y `denseVectors` deben tener exactamente la misma cantidad de elementos (numNodes).
    /// @pre `adjacencyList` no debe contener referencias fuera de los límites (index >= numNodes).
    /// @post Devuelve un `CSRHardwareContext` inmutable con los datos cargados en la VRAM/RAM compartida.
    ///
    /// @param adjacencyList Un arreglo donde el índice es el Nodo ID, y su valor es un arreglo de Nodos ID vecinos.
    /// @param denseVectors Un arreglo de arreglos de flotantes (los embeddings).
    /// @param vectorDim La dimensionalidad de cada vector.
    public func compileToUMA(
        adjacencyList: [[Int32]],
        denseVectors: [[Float]],
        vectorDim: Int
    ) throws -> CSRHardwareContext {
        
        let numNodes = adjacencyList.count
        
        // 1. Verificación de Contratos de Entrada (Precondiciones)
        guard numNodes > 0 else {
            throw CSRCompilationError.invalidGraphTopology("El grafo está vacío. Se requieren nodos para compilar.")
        }
        guard numNodes == denseVectors.count else {
            throw CSRCompilationError.invalidGraphTopology("Inconsistencia: La cantidad de nodos en la lista de adyacencia (\(numNodes)) no coincide con la cantidad de vectores (\(denseVectors.count)).")
        }
        
        // 2. Pre-Cálculo de Capacidad para evitar fragmentación de memoria (Crítico para el límite de 8GB)
        var totalEdges = 0
        for neighbors in adjacencyList {
            totalEdges += neighbors.count
        }
        
        // 3. Inicialización de estructuras temporales en Swift (Contiguous Arrays)
        // Usamos [Int32] porque las GPUs operan más rápido con 32 bits y ahorramos el 50% de RAM vs 64 bits.
        var rowPointers = [Int32]()
        var colIndices = [Int32]()
        var flattenedVectors = [Float]()
        
        rowPointers.reserveCapacity(numNodes + 1)
        colIndices.reserveCapacity(totalEdges)
        flattenedVectors.reserveCapacity(numNodes * vectorDim)
        
        // 4. Transformación Algorítmica (Pointer-Chasing a Matemáticas CSR)
        var currentEdgeOffset: Int32 = 0
        rowPointers.append(currentEdgeOffset) // El primer nodo siempre empieza en el offset 0
        
        for (nodeId, neighbors) in adjacencyList.enumerated() {
            // Aplanar Vectores
            let vector = denseVectors[nodeId]
            guard vector.count == vectorDim else {
                throw CSRCompilationError.invalidGraphTopology("El nodo \(nodeId) tiene una dimensionalidad incorrecta. Esperado: \(vectorDim), Real: \(vector.count).")
            }
            flattenedVectors.append(contentsOf: vector)
            
            // Aplanar Topología
            colIndices.append(contentsOf: neighbors)
            currentEdgeOffset += Int32(neighbors.count)
            rowPointers.append(currentEdgeOffset)
        }
        
        // 5. Traspaso Zero-Copy al Hardware (Vía Objective-C++)
        // Usamos el prefijo `withUnsafeBufferPointer` para garantizar acceso directo a la memoria base de Swift.
        
        guard let rowPointersUMA = rowPointers.withUnsafeBufferPointer({ ptr in
            memoryManager.allocateCSRRowPointers(ptr.baseAddress!, count: UInt(rowPointers.count))
        }) else {
            throw CSRCompilationError.outOfMemory("El UMAMemoryManager falló al asignar rowPointers (\(rowPointers.count * 4) bytes).")
        }
        
        guard let colIndicesUMA = colIndices.withUnsafeBufferPointer({ ptr in
            memoryManager.allocateCSRColumnIndices(ptr.baseAddress!, count: UInt(colIndices.count))
        }) else {
            throw CSRCompilationError.outOfMemory("El UMAMemoryManager falló al asignar colIndices (\(colIndices.count * 4) bytes).")
        }
        
        guard let vectorsUMA = flattenedVectors.withUnsafeBufferPointer({ ptr in
            memoryManager.allocateDenseVectors(ptr.baseAddress!, count: UInt(flattenedVectors.count))
        }) else {
            throw CSRCompilationError.outOfMemory("El UMAMemoryManager falló al asignar los vectores densos (\(flattenedVectors.count * 4) bytes).")
        }
        
        // 6. Retorno de Invariantes (El estado final del sistema)
        return CSRHardwareContext(
            rowPointersBuffer: rowPointersUMA,
            colIndicesBuffer: colIndicesUMA,
            denseVectorsBuffer: vectorsUMA,
            numNodes: numNodes,
            vectorDim: vectorDim,
            numEdges: totalEdges
        )
    }
}
