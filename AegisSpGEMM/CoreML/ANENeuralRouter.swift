//
//  ANENeuralRouter.swift
//  AegisSpGEMM
//
//  Created for Phase 8 - ANE Learned Coarse Router & Adaptive Nucleus Probing.
//  Strict Memory Standard: Pre-allocated MLMultiArray Scratchpad & .cpuAndNeuralEngine Isolation.
//

import Foundation
import CoreML
import Accelerate

public enum ANERouterError: Error {
    case modelNotFound(String)
    case invalidTensorAllocation
    case predictionFailed(String)
}

/// @struct AdaptiveRoutingDecision
/// @brief Resultado del enrutamiento adaptativo con telemetría de incertidumbre (Entropía de Shannon).
public struct AdaptiveRoutingDecision {
    public let selectedCentroids: [Int]
    public let adaptiveNprobe: Int
    public let cumulativeConfidence: Float
    public let shannonEntropy: Float
    public let latencyMs: Double
    public let usedHardwareANE: Bool
}

/// @class ANENeuralRouter
/// @brief Enrutador neuronal para el Apple Neural Engine (ANE) con selección dinámica de celdas (Nucleus Probing).
public final class ANENeuralRouter {
    
    private let model: MLModel
    private let inputTensor: MLMultiArray
    private let inputFeatureName: String
    private let outputFeatureName: String
    private let vectorDim: Int
    private let numCentroids: Int
    
    public init(
        modelURL: URL,
        vectorDim: Int,
        numCentroids: Int,
        inputFeatureName: String = "query_vector",
        outputFeatureName: String = "centroid_probabilities"
    ) throws {
        self.vectorDim = vectorDim
        self.numCentroids = numCentroids
        self.inputFeatureName = inputFeatureName
        self.outputFeatureName = outputFeatureName
        
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw ANERouterError.modelNotFound("No se encontró el modelo CoreML en: \(modelURL.path)")
        }
        
        // 1. Aislar el cómputo en el ANE (Prohibido tocar la GPU reservada para Metal SpGEMM)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        config.allowLowPrecisionAccumulationOnGPU = true
        
        // 2. Compilar .mlpackage en .mlmodelc si aún no está compilado
        let compiledURL: URL
        if modelURL.pathExtension == "mlmodelc" {
            compiledURL = modelURL
        } else {
            compiledURL = try MLModel.compileModel(at: modelURL)
        }
        
        self.model = try MLModel(contentsOf: compiledURL, configuration: config)
        
        // 3. Scratchpad MLMultiArray [1, vectorDim] pre-asignado (Zero-Allocation en bucle caliente)
        guard let multiArray = try? MLMultiArray(
            shape: [1, NSNumber(value: vectorDim)],
            dataType: .float32
        ) else {
            throw ANERouterError.invalidTensorAllocation
        }
        self.inputTensor = multiArray
    }
    
    /// @brief Ejecuta la inferencia en el ANE y aplica Nucleus Probing (masa acumulada >= confidenceThreshold).
    public func routeAdaptively(
        queryVector: [Float],
        confidenceThreshold: Float = 0.965,
        minProbe: Int = 8,
        maxProbe: Int = 32
    ) throws -> AdaptiveRoutingDecision {
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // 1. Inyección Zero-Allocation al puntero del MLMultiArray pre-asignado
        let dstPtr = inputTensor.dataPointer.bindMemory(to: Float.self, capacity: vectorDim)
        queryVector.withUnsafeBufferPointer { srcBuffer in
            if let srcBase = srcBuffer.baseAddress {
                memcpy(dstPtr, srcBase, vectorDim * MemoryLayout<Float>.stride)
            }
        }
        
        // 2. Despacho Síncrono al Apple Neural Engine (ANE)
        let provider = try MLDictionaryFeatureProvider(
            dictionary: [inputFeatureName: MLFeatureValue(multiArray: inputTensor)]
        )
        let prediction = try model.prediction(from: provider)
        
        guard let outputMultiArray = prediction.featureValue(for: outputFeatureName)?.multiArrayValue else {
            throw ANERouterError.predictionFailed("El modelo CoreML no devolvió el tensor '\(outputFeatureName)'.")
        }
        
        // 3. Lectura directa de probabilidades y aplicación de Nucleus Probing
        let count = min(numCentroids, outputMultiArray.count)
        var probabilities = [Float](repeating: 0.0, count: count)
        
        if outputMultiArray.dataType == .float32 {
            let probPtr = outputMultiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
            memcpy(&probabilities, probPtr, count * MemoryLayout<Float>.stride)
        } else {
            for i in 0..<count {
                probabilities[i] = outputMultiArray[i].floatValue
            }
        }
        
        let decision = Self.computeNucleusProbing(
            probabilities: probabilities,
            confidenceThreshold: confidenceThreshold,
            minProbe: minProbe,
            maxProbe: maxProbe,
            startTime: startTime,
            usedHardwareANE: true
        )
        
        return decision
    }
    
    /// @brief Algoritmo universal de selección adaptativa por masa acumulada y entropía de Shannon.
    public static func computeNucleusProbing(
        probabilities: [Float],
        confidenceThreshold: Float,
        minProbe: Int,
        maxProbe: Int,
        startTime: CFAbsoluteTime,
        usedHardwareANE: Bool
    ) -> AdaptiveRoutingDecision {
        
        let count = probabilities.count
        let safeMin = max(1, min(minProbe, count))
        let safeMax = max(safeMin, min(maxProbe, count))
        
        var indexedProbs = [(index: Int, prob: Float)]()
        indexedProbs.reserveCapacity(count)
        for idx in 0..<count {
            indexedProbs.append((index: idx, prob: probabilities[idx]))
        }
        
        indexedProbs.sort { $0.prob > $1.prob }
        
        var cumulativeMass: Float = 0.0
        var entropy: Float = 0.0
        var chosenCount = 0
        var selected = [Int]()
        selected.reserveCapacity(safeMax)
        
        for (i, item) in indexedProbs.enumerated() {
            let p = max(item.prob, 1e-9)
            cumulativeMass += item.prob
            entropy -= p * log(p)
            selected.append(item.index)
            chosenCount = i + 1
            
            if chosenCount >= safeMin && cumulativeMass >= confidenceThreshold {
                break
            }
            if chosenCount >= safeMax {
                break
            }
        }
        
        let elapsedMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
        
        return AdaptiveRoutingDecision(
            selectedCentroids: selected,
            adaptiveNprobe: chosenCount,
            cumulativeConfidence: min(cumulativeMass, 1.0),
            shannonEntropy: entropy,
            latencyMs: elapsedMs,
            usedHardwareANE: usedHardwareANE
        )
    }
}
