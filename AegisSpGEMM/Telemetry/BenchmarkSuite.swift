//
//  BenchmarkSuite.swift
//  AegisSpGEMM
//
//  Created for Phase 7.1 (Updated Phase 8.1 - True End-to-End P95/P99 & Silicon Bandwidth Suite).
//  Strict Memory Standard: Exact Per-Query Total Latency Distribution & Adaptive Probe Profiling.
//

import Foundation
import Accelerate

public struct BenchmarkReport {
    public let totalQueries: Int
    public let recallTopK: Double
    public let p50TotalLatencyMs: Double
    public let p95TotalLatencyMs: Double
    public let p99TotalLatencyMs: Double
    public let avgTotalLatencyMs: Double
    public let coarseRoutingAvgMs: Double
    public let avgGpuWallMs: Double
    public let avgGpuSiliconMs: Double
    public let effectiveBandwidthGBs: Double
    public let siliconBandwidthGBs: Double
    public let avgNprobeUsed: Double
}

public final class BenchmarkSuite {
    
    private let orchestrator: SearchOrchestrator
    private let flattenedVectors: [Float]
    private let numVectors: Int
    private let vectorDim: Int
    private let rowPtr: [Int32]
    
    public init(
        orchestrator: SearchOrchestrator,
        flattenedVectors: [Float],
        numVectors: Int,
        vectorDim: Int,
        rowPtr: [Int32]
    ) {
        self.orchestrator = orchestrator
        self.flattenedVectors = flattenedVectors
        self.numVectors = numVectors
        self.vectorDim = vectorDim
        self.rowPtr = rowPtr
    }
    
    public func runAcademicBenchmark(
        queries: [[Float]],
        groundTruth: [[Int]]? = nil,
        nprobe: Int = 16,
        topK: Int = 10
    ) throws -> BenchmarkReport {
        
        guard !queries.isEmpty, numVectors > 0, vectorDim > 0 else {
            return BenchmarkReport(
                totalQueries: 0, recallTopK: 0.0,
                p50TotalLatencyMs: 0.0, p95TotalLatencyMs: 0.0, p99TotalLatencyMs: 0.0,
                avgTotalLatencyMs: 0.0, coarseRoutingAvgMs: 0.0, avgGpuWallMs: 0.0,
                avgGpuSiliconMs: 0.0, effectiveBandwidthGBs: 0.0, siliconBandwidthGBs: 0.0,
                avgNprobeUsed: 0.0
            )
        }
        
        var totalLatencies = [Double]()
        totalLatencies.reserveCapacity(queries.count)
        
        var totalRecall: Double = 0.0
        var recallSamplesCount: Int = 0
        var totalCoarseRoutingMs: Double = 0.0
        var totalGpuWallMs: Double = 0.0
        var totalGpuSiliconMs: Double = 0.0
        var totalBytesRead: Int = 0
        
        for (idx, query) in queries.enumerated() {
            let prepStart = CFAbsoluteTimeGetCurrent()
            let winningCentroids = try orchestrator.selectTopCentroids(
                queryVector: query,
                nprobe: nprobe
            )
            let prepEnd = CFAbsoluteTimeGetCurrent()
            let routeMs = (prepEnd - prepStart) * 1000.0
            totalCoarseRoutingMs += routeMs
            
            var nnzTouched = 0
            for cIdx in winningCentroids {
                if cIdx >= 0 && (cIdx + 1) < rowPtr.count {
                    nnzTouched += Int(rowPtr[cIdx + 1] - rowPtr[cIdx])
                }
            }
            
            let execStart = CFAbsoluteTimeGetCurrent()
            let gpuResults = try orchestrator.executeSearch(
                queryVector: query,
                entryPoints: winningCentroids,
                topK: topK
            )
            let execEnd = CFAbsoluteTimeGetCurrent()
            
            let gpuWallMs = (execEnd - execStart) * 1000.0
            let queryTotalMs = routeMs + gpuWallMs
            
            totalGpuWallMs += gpuWallMs
            totalGpuSiliconMs += orchestrator.lastGpuSiliconMs
            totalLatencies.append(queryTotalMs)
            
            let bytesThisQuery = (nnzTouched * MemoryLayout<Int32>.stride)
                + (nnzTouched * vectorDim * MemoryLayout<Float>.stride)
            totalBytesRead += bytesThisQuery
            
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
        
        totalLatencies.sort()
        let count = Double(totalLatencies.count)
        let avgTotal = totalLatencies.reduce(0, +) / count
        let p50Idx = min(max(Int(count * 0.50), 0), totalLatencies.count - 1)
        let p95Idx = min(max(Int(count * 0.95), 0), totalLatencies.count - 1)
        let p99Idx = min(max(Int(count * 0.99), 0), totalLatencies.count - 1)
        
        let totalWallSeconds = totalGpuWallMs / 1000.0
        let totalSiliconSeconds = totalGpuSiliconMs / 1000.0
        let totalGigabytes = Double(totalBytesRead) / 1_000_000_000.0
        
        let effectiveBW = totalWallSeconds > 0 ? totalGigabytes / totalWallSeconds : 0.0
        let siliconBW = totalSiliconSeconds > 0 ? totalGigabytes / totalSiliconSeconds : 0.0
        let avgRecall = recallSamplesCount > 0 ? (totalRecall / Double(recallSamplesCount)) : 0.0
        
        return BenchmarkReport(
            totalQueries: queries.count,
            recallTopK: avgRecall * 100.0,
            p50TotalLatencyMs: totalLatencies[p50Idx],
            p95TotalLatencyMs: totalLatencies[p95Idx],
            p99TotalLatencyMs: totalLatencies[p99Idx],
            avgTotalLatencyMs: avgTotal,
            coarseRoutingAvgMs: totalCoarseRoutingMs / count,
            avgGpuWallMs: totalGpuWallMs / count,
            avgGpuSiliconMs: totalGpuSiliconMs / count,
            effectiveBandwidthGBs: effectiveBW,
            siliconBandwidthGBs: siliconBW,
            avgNprobeUsed: Double(nprobe)
        )
    }
    
    /// @brief Ejecuta el benchmark estadístico con el Enrutador Neuronal Adaptativo (ANE / Nucleus Probing).
    public func runAdaptiveAcademicBenchmark(
        queries: [[Float]],
        groundTruth: [[Int]]? = nil,
        confidenceThreshold: Float = 0.985,
        minProbe: Int = 10,
        maxProbe: Int = 32,
        topK: Int = 10
    ) throws -> BenchmarkReport {
        
        guard !queries.isEmpty, numVectors > 0, vectorDim > 0 else {
            return BenchmarkReport(
                totalQueries: 0, recallTopK: 0.0,
                p50TotalLatencyMs: 0.0, p95TotalLatencyMs: 0.0, p99TotalLatencyMs: 0.0,
                avgTotalLatencyMs: 0.0, coarseRoutingAvgMs: 0.0, avgGpuWallMs: 0.0,
                avgGpuSiliconMs: 0.0, effectiveBandwidthGBs: 0.0, siliconBandwidthGBs: 0.0,
                avgNprobeUsed: 0.0
            )
        }
        
        var totalLatencies = [Double]()
        totalLatencies.reserveCapacity(queries.count)
        
        var totalRecall: Double = 0.0
        var recallSamplesCount: Int = 0
        var totalCoarseRoutingMs: Double = 0.0
        var totalGpuWallMs: Double = 0.0
        var totalGpuSiliconMs: Double = 0.0
        var totalBytesRead: Int = 0
        var totalProbesUsed: Int = 0
        
        for (idx, query) in queries.enumerated() {
            let decision = try orchestrator.selectTopCentroidsAdaptive(
                queryVector: query,
                confidenceThreshold: confidenceThreshold,
                minProbe: minProbe,
                maxProbe: maxProbe
            )
            let routeMs = decision.latencyMs
            totalCoarseRoutingMs += routeMs
            totalProbesUsed += decision.adaptiveNprobe
            
            var nnzTouched = 0
            for cIdx in decision.selectedCentroids {
                if cIdx >= 0 && (cIdx + 1) < rowPtr.count {
                    nnzTouched += Int(rowPtr[cIdx + 1] - rowPtr[cIdx])
                }
            }
            
            let execStart = CFAbsoluteTimeGetCurrent()
            let gpuResults = try orchestrator.executeSearch(
                queryVector: query,
                entryPoints: decision.selectedCentroids,
                topK: topK
            )
            let execEnd = CFAbsoluteTimeGetCurrent()
            
            let gpuWallMs = (execEnd - execStart) * 1000.0
            let queryTotalMs = routeMs + gpuWallMs
            
            totalGpuWallMs += gpuWallMs
            totalGpuSiliconMs += orchestrator.lastGpuSiliconMs
            totalLatencies.append(queryTotalMs)
            
            let bytesThisQuery = (nnzTouched * MemoryLayout<Int32>.stride)
                + (nnzTouched * vectorDim * MemoryLayout<Float>.stride)
            totalBytesRead += bytesThisQuery
            
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
        
        totalLatencies.sort()
        let count = Double(totalLatencies.count)
        let avgTotal = totalLatencies.reduce(0, +) / count
        let p50Idx = min(max(Int(count * 0.50), 0), totalLatencies.count - 1)
        let p95Idx = min(max(Int(count * 0.95), 0), totalLatencies.count - 1)
        let p99Idx = min(max(Int(count * 0.99), 0), totalLatencies.count - 1)
        
        let totalWallSeconds = totalGpuWallMs / 1000.0
        let totalSiliconSeconds = totalGpuSiliconMs / 1000.0
        let totalGigabytes = Double(totalBytesRead) / 1_000_000_000.0
        
        let effectiveBW = totalWallSeconds > 0 ? totalGigabytes / totalWallSeconds : 0.0
        let siliconBW = totalSiliconSeconds > 0 ? totalGigabytes / totalSiliconSeconds : 0.0
        let avgRecall = recallSamplesCount > 0 ? (totalRecall / Double(recallSamplesCount)) : 0.0
        
        return BenchmarkReport(
            totalQueries: queries.count,
            recallTopK: avgRecall * 100.0,
            p50TotalLatencyMs: totalLatencies[p50Idx],
            p95TotalLatencyMs: totalLatencies[p95Idx],
            p99TotalLatencyMs: totalLatencies[p99Idx],
            avgTotalLatencyMs: avgTotal,
            coarseRoutingAvgMs: totalCoarseRoutingMs / count,
            avgGpuWallMs: totalGpuWallMs / count,
            avgGpuSiliconMs: totalGpuSiliconMs / count,
            effectiveBandwidthGBs: effectiveBW,
            siliconBandwidthGBs: siliconBW,
            avgNprobeUsed: Double(totalProbesUsed) / count
        )
    }
    
    private func computeGroundTruth(query: [Float], topK: Int) -> [Int] {
        var allScores = [Float](repeating: 0.0, count: numVectors)
        
        flattenedVectors.withUnsafeBufferPointer { vPtr in
            query.withUnsafeBufferPointer { qPtr in
                guard let vBase = vPtr.baseAddress, let qBase = qPtr.baseAddress else { return }
                vDSP_mmul(
                    vBase, 1,
                    qBase, 1,
                    &allScores, 1,
                    vDSP_Length(numVectors),
                    1,
                    vDSP_Length(vectorDim)
                )
            }
        }
        
        var scored = [(nodeId: Int, score: Float)]()
        scored.reserveCapacity(numVectors)
        for id in 0..<numVectors {
            scored.append((nodeId: id, score: allScores[id]))
        }
        scored.sort { $0.score > $1.score }
        return Array(scored.prefix(topK)).map { $0.nodeId }
    }
    
    private func calculateRecall(gpuResults: [Int], cpuResults: [Int]) -> Double {
        guard !cpuResults.isEmpty else { return 0.0 }
        let gpuSet = Set(gpuResults)
        let cpuSet = Set(cpuResults)
        let intersection = gpuSet.intersection(cpuSet)
        return Double(intersection.count) / Double(cpuResults.count)
    }
}
