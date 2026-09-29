//
//  SearchOrchestrator.swift
//  AegisSpGEMM
//
//  Created for Phase 4 - Hybrid Pipeline.
//  Strict Memory Standard: Facade Pattern for UMA Lifecycle and GPU Dispatch.
//

import Foundation

/// @class SearchOrchestrator
/// @brief Fachada principal que coordina el ciclo de vida del índice, la memoria y la ejecución GPU.
public final class SearchOrchestrator {
    
    private let compiler: CSRTopologyCompiler
    private let gpuEngine: MetalSpGEMMEngine
    private var hardwareContext: CSRHardwareContext?
    
    /// @brief Inicializa los componentes de compilación y los motores GPU.
    public init() throws {
        self.compiler = try CSRTopologyCompiler()
        self.gpuEngine = try MetalSpGEMMEngine()
        self.hardwareContext = nil
    }
    
    /// @brief Ingesta un grafo topológico y vectores, compilándolos y cargándolos en la Memoria Unificada (UMA).
    /// @pre Los nodos y vectores deben ser simétricos.
    /// @post El estado interno retiene el contexto de hardware. Operación pesada (AoT).
    public func ingestIndex(adjacencyList: [[Int32]], denseVectors: [[Float]], vectorDim: Int) throws {
        print("[Aegis-Info] Iniciando compilación topológica a formato CSR...")
        let startTime = CFAbsoluteTimeGetCurrent()
        
        self.hardwareContext = try compiler.compileToUMA(
            adjacencyList: adjacencyList,
            denseVectors: denseVectors,
            vectorDim: vectorDim
        )
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        print(String(format: "[Aegis-Success] Compilación finalizada en %.4f ms.", elapsed * 1000.0))
        print("[Aegis-Telemetry] Nodos: \(self.hardwareContext!.numNodes) | Aristas: \(self.hardwareContext!.numEdges) | Dimensión: \(self.hardwareContext!.vectorDim)")
    }
    
    /// @brief Ejecuta una búsqueda vectorial delegada totalmente a los núcleos de la GPU (SpGEMM + Dot Product).
    /// @pre `ingestIndex` debe haberse ejecutado exitosamente.
    /// @param queryVector El vector de la consulta (ej. generado por el ANE).
    /// @param entryPoint Nodo semilla desde donde el 'agua' empieza a fluir en el grafo.
    /// @param topK Cantidad de resultados deseados.
    public func executeSearch(queryVector: [Float], entryPoint: Int, topK: Int = 3) throws -> [(nodeId: Int, score: Float)] {
        guard let context = self.hardwareContext else {
            throw MetalEngineError.pipelineCreationFailed("El índice no ha sido ingerido en la UMA.")
        }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Ejecución en Silicio de Apple (SpMV + Similitud)
        let rawScores = try gpuEngine.executeSearchStep(context: context, queryVector: queryVector, entryPointNode: entryPoint)
        
        // Fase de post-procesamiento en CPU: Filtrar y ordenar los resultados Top-K.
        // Los nodos no activados tendrán -INFINITY y serán descartados orgánicamente.
        var scoredNodes = [(nodeId: Int, score: Float)]()
        scoredNodes.reserveCapacity(context.numNodes)
        
        for (idx, score) in rawScores.enumerated() {
            if score > -Float.infinity {
                scoredNodes.append((nodeId: idx, score: score))
            }
        }
        
        // Orden descendente (mayor similitud primero)
        scoredNodes.sort { $0.score > $1.score }
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        print(String(format: "[Aegis-Execution] SpGEMM Search Kernel ejecutado en %.4f ms.", elapsed * 1000.0))
        
        return Array(scoredNodes.prefix(topK))
    }
}
