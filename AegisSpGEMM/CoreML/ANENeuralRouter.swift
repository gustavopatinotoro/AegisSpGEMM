//
//  ANENeuralRouter.swift
//  AegisSpGEMM
//
//  Created for Phase 8 (Updated Phase 8.1 - Zero-Allocation Scratchpad & Continuous Entropy Law).
//  Strict Memory Standard: Pre-allocated Input/Output Buffers & .cpuAndNeuralEngine Isolation.
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
/// @brief Enrutador neuronal para el Apple Neural Engine (ANE) con control híbrido Masa + Entropía.
public final class ANENeuralRouter {
    
    private let model: MLModel
    private let inputTensor: MLMultiArray
    private var probabilitiesScratchpad: [Float]
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
        self.probabilitiesScratchpad = [Float](repeating: 0.0, count: numCentroids)
        
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw ANERouterError.modelNotFound("No se encontró el modelo CoreML en: \(modelURL.path)")
        }
        
        // 1. Aislar el cómputo en el ANE (Dejar 100% de la GPU libre para Metal SpGEMM)
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
        
        // 3. Scratchpad MLMultiArray [1, vectorDim] pre-asignado (Zero-Allocation)
        guard let multiArray = try? MLMultiArray(
            shape: [1, NSNumber(value: vectorDim)],
            dataType: .float32
        ) else {
            throw ANERouterError.invalidTensorAllocation
        }
        self.inputTensor = multiArray
        
        // 4. Calentamiento Físico del Silicio ANE (Despierta el demonio 'aned' antes del benchmark)
        try performHardwareWarmup()
    }
    
    /// @brief Ejecuta 3 inferencias en vacío durante la carga para mapear los registros SRAM del ANE.
    private func performHardwareWarmup() throws {
        let ptr = inputTensor.dataPointer.bindMemory(to: Float.self, capacity: vectorDim)
        let normVal = 1.0 / Float(sqrt(Double(vectorDim)))
        for i in 0..<vectorDim {
            ptr[i] = normVal
        }
        let provider = try MLDictionaryFeatureProvider(
            dictionary: [inputFeatureName: MLFeatureValue(multiArray: inputTensor)]
        )
        for _ in 0..<3 {
            _ = try model.prediction(from: provider)
        }
    }
    
    /// @brief Ejecuta la inferencia en el ANE y aplica la Ley de Control Híbrida (Masa + Entropía).
    public func routeAdaptively(
        queryVector: [Float],
        confidenceThreshold: Float = 0.985,
        minProbe: Int = 10,
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
        
        // 3. Lectura directa sobre el Scratchpad pre-asignado (Cero asignaciones en Heap)
        let count = min(numCentroids, outputMultiArray.count)
        if outputMultiArray.dataType == .float32 {
            let probPtr = outputMultiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
            probabilitiesScratchpad.withUnsafeMutableBufferPointer { dstBuf in
                if let dstBase = dstBuf.baseAddress {
                    memcpy(dstBase, probPtr, count * MemoryLayout<Float>.stride)
                }
            }
        } else {
            for i in 0..<count {
                probabilitiesScratchpad[i] = outputMultiArray[i].floatValue
            }
        }
        
        return Self.computeNucleusProbing(
            probabilities: probabilitiesScratchpad,
            confidenceThreshold: confidenceThreshold,
            minProbe: minProbe,
            maxProbe: maxProbe,
            startTime: startTime,
            usedHardwareANE: true
        )
    }
    
    /// @brief Ley de Control Híbrida Fase 8.1: Combina Masa Acumulada con Rampa Continua por Entropía H_32.
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
        
        // 1. Calcular Entropía de Cola sobre las primeras 32 celdas (H_32)
        let entropyWindow = min(32, count)
        var tailEntropy: Float = 0.0
        for i in 0..<entropyWindow {
            let p = max(indexedProbs[i].prob, 1e-9)
            tailEntropy -= p * log(p)
        }
        
        // 2. Rampa Continua de Entropía:
        // - Consultas certeras (H_32 <= 0.25): Piso mínimo = safeMin (10 celdas).
        // - Consultas intermedias y de frontera (H_32 > 0.25): Crecimiento proporcional continuo.
        let entropyFloor: Int
        if tailEntropy <= 0.25 {
            entropyFloor = safeMin
        } else {
            let boost = Int(ceil((tailEntropy - 0.25) * 12.0))
            entropyFloor = min(safeMax, safeMin + boost)
        }
        
        // 3. Selección del núcleo acumulado respetando el piso de entropía
        var cumulativeMass: Float = 0.0
        var chosenCount = 0
        var selected = [Int]()
        selected.reserveCapacity(safeMax)
        
        for i in 0..<safeMax {
            let item = indexedProbs[i]
            cumulativeMass += item.prob
            selected.append(item.index)
            chosenCount = i + 1
            
            if chosenCount >= entropyFloor && cumulativeMass >= confidenceThreshold {
                break
            }
        }
        
        let elapsedMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
        
        return AdaptiveRoutingDecision(
            selectedCentroids: selected,
            adaptiveNprobe: chosenCount,
            cumulativeConfidence: min(cumulativeMass, 1.0),
            shannonEntropy: tailEntropy,
            latencyMs: elapsedMs,
            usedHardwareANE: usedHardwareANE
        )
    }
}
