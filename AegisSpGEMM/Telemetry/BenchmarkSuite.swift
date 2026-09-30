//
//  BenchmarkSuite.swift
//  AegisSpGEMM
//
//  Created for Phase 5 - Academic Telemetry Suite (Updated Phase 6 - IVF-CSR).
//  Strict Memory Standard: P95/P99 Latency, Recall, and Effective BW Profiling.
//

import Foundation

public struct BenchmarkReport {
    public let totalQueries: Int
    public let recallTopK: Double
    public let p95LatencyMs: Double
    public let p99LatencyMs: Double
    public let avgLatencyMs: Double
    public let effectiveBandwidthGBs: Double
    public let coarseRoutingAvgMs: Double
}

public final class BenchmarkSuite {
    
    private let orchestrator: SearchOrchestrator
    private let rawVectors: [[Float]]
    private let centroids: [[Float]]
    private let rowPtr: [Int32]
    
    public init(
        orchestrator: SearchOrchestrator,
        vectors: [[Float]],
        centroids: [[Float]],
        rowPtr: [Int32]
    ) {
        self.orchestrator = orchestrator
        self.rawVectors = vectors
        self.centroids = centroids
        self.rowPtr = rowPtr
    }
    
    /// @brief Ejecuta el protocolo de perfilado académico para la arquitectura híbrida IVF-CSR.
    /// @param queries Lote de vectores de consulta normalizados (L2).
    /// @param groundTruth Índices reales Top-K precalculados (opcional; si es nil usa fuerza bruta en CPU).
    /// @param nprobe Cantidad de celdas de Voronoi (centroides) a activar simultáneamente en la GPU.
    /// @param topK Cantidad de resultados finales a recuperar por consulta.
    public func runAcademicBenchmark(
        queries: [[Float]],
        groundTruth: [[Int]]? = nil,
        nprobe: Int = 16,
        topK: Int = 10
    ) throws -> BenchmarkReport {
        
        guard !queries.isEmpty, !rawVectors.isEmpty else {
            return BenchmarkReport(
                totalQueries: 0,
                recallTopK: 0.0,
                p95LatencyMs: 0.0,
                p99LatencyMs: 0.0,
                avgLatencyMs: 0.0,
                effectiveBandwidthGBs: 0.0,
                coarseRoutingAvgMs: 0.0
            )
        }
        
        let vectorDim = rawVectors[0].count
        var latencies = [Double]()
        latencies.reserveCapacity(queries.count)
        
        var totalRecall: Double = 0.0
        var recallSamplesCount: Int = 0
        var totalCoarseRoutingSeconds: Double = 0.0
        var totalBytesRead: Int = 0
        
        print("[Telemetry] Iniciando lote de \(queries.count) consultas para análisis estadístico (nprobe = \(nprobe))...")
        
        for (idx, query) in queries.enumerated() {
            // A. Fase 1 (CPU): Enrutamiento Grueso sobre Centroides
            let prepStart = CFAbsoluteTimeGetCurrent()
            let winningCentroids = selectTopCentroids(query: query, nprobe: nprobe)
            let prepEnd = CFAbsoluteTimeGetCurrent()
            totalCoarseRoutingSeconds += (prepEnd - prepStart)
            
            // Contabilizar aristas/candidatos activados (Scatter NNZ) en esta consulta
            var nnzTouched = 0
            for cIdx in winningCentroids {
                if cIdx >= 0 && (cIdx + 1) < rowPtr.count {
                    nnzTouched += Int(rowPtr[cIdx + 1] - rowPtr[cIdx])
                }
            }
            
            // B. Fase 2 (GPU): Medición de Ejecución en Silicio (SpMV Scatter + Dot Product)
            let execStart = CFAbsoluteTimeGetCurrent()
            let gpuResults = try orchestrator.executeSearch(
                queryVector: query,
                entryPoints: winningCentroids,
                topK: topK
            )
            let execEnd = CFAbsoluteTimeGetCurrent()
            
            let queryLatencyMs = (execEnd - execStart) * 1000.0
            latencies.append(queryLatencyMs)
            
            // Física de Memoria:
            // Lectura de índices CSR activos + máscara de estado + vectores densos de los candidatos iluminados
            let bytesThisQuery = (nnzTouched * MemoryLayout<Int32>.stride)
                + (rawVectors.count * MemoryLayout<Float>.stride)
                + (nnzTouched * vectorDim * MemoryLayout<Float>.stride)
            totalBytesRead += bytesThisQuery
            
            // C. Evaluación de Recall@K
            if let gt = groundTruth, idx < gt.count {
                let expectedTopK = Array(gt[idx].prefix(topK))
                totalRecall += calculateRecall(
                    gpuResults: gpuResults.map { $0.nodeId },
                    cpuResults: expectedTopK
                )
                recallSamplesCount += 1
            } else if idx % 20 == 0 {
                let cpuResults = computeGroundTruth(query: query, topK: topK)
                totalRecall += calculateRecall(
                    gpuResults: gpuResults.map { $0.nodeId },
                    cpuResults: cpuResults
                )
                recallSamplesCount += 1
            }
        }
        
        // 2. Procesamiento Estadístico (Percentiles P95 y P99 seguros)
        latencies.sort()
        let avgLatency = latencies.reduce(0, +) / Double(latencies.count)
        let p95Index = min(max(Int(Double(latencies.count) * 0.95), 0), latencies.count - 1)
        let p99Index = min(max(Int(Double(latencies.count) * 0.99), 0), latencies.count - 1)
        
        // 3. Física de Memoria (Ancho de Banda Efectivo en GB/s)
        let totalTimeSeconds = latencies.reduce(0, +) / 1000.0
        let effectiveBW = totalTimeSeconds > 0
            ? (Double(totalBytesRead) / 1_000_000_000.0) / totalTimeSeconds
            : 0.0
        
        let avgRecall = recallSamplesCount > 0
            ? (totalRecall / Double(recallSamplesCount))
            : 0.0
        
        return BenchmarkReport(
            totalQueries: queries.count,
            recallTopK: avgRecall * 100.0,
            p95LatencyMs: latencies[p95Index],
            p99LatencyMs: latencies[p99Index],
            avgLatencyMs: avgLatency,
            effectiveBandwidthGBs: effectiveBW,
            coarseRoutingAvgMs: (totalCoarseRoutingSeconds / Double(queries.count)) * 1000.0
        )
    }
    
    /// @brief Selecciona los `nprobe` centroides con mayor producto interno respecto a la consulta.
    private func selectTopCentroids(query: [Float], nprobe: Int) -> [Int] {
        var scores = [(index: Int, score: Float)]()
        scores.reserveCapacity(centroids.count)
        
        for (cIdx, centroid) in centroids.enumerated() {
            var dot: Float = 0.0
            for d in 0..<query.count {
                dot += query[d] * centroid[d]
            }
            scores.append((index: cIdx, score: dot))
        }
        
        scores.sort { $0.score > $1.score }
        return scores.prefix(nprobe).map { $0.index }
    }
    
    /// @brief Calcula el producto interno secuencial (Ground Truth en CPU) cuando no se provee desde el binario.
    private func computeGroundTruth(query: [Float], topK: Int) -> [Int] {
        var scores = [(nodeId: Int, score: Float)]()
        scores.reserveCapacity(rawVectors.count)
        
        for (id, vector) in rawVectors.enumerated() {
            var dotProduct: Float = 0.0
            for i in 0..<vector.count {
                dotProduct += query[i] * vector[i]
            }
            scores.append((nodeId: id, score: dotProduct))
        }
        
        scores.sort { $0.score > $1.score }
        return Array(scores.prefix(topK)).map { $0.nodeId }
    }
    
    /// @brief Compara intersecciones para calcular el Recall@K.
    private func calculateRecall(gpuResults: [Int], cpuResults: [Int]) -> Double {
        guard !cpuResults.isEmpty else { return 0.0 }
        let gpuSet = Set(gpuResults)
        let cpuSet = Set(cpuResults)
        let intersection = gpuSet.intersection(cpuSet)
        
        return Double(intersection.count) / Double(cpuResults.count)
    }
}
