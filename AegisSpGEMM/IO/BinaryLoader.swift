//
//  BinaryLoader.swift
//  AegisSpGEMM
//
//  Created for Phase 6 - IVF-CSR Ingestion.
//  Strict Memory Standard: mmap (Memory-Mapped File) for Zero-Copy I/O.
//

import Foundation

public struct ANNBenchmarkData {
    let numVectors: Int
    let numCentroids: Int
    let dimension: Int
    let numQueries: Int
    let topK: Int
    
    let centroids: [[Float]]
    let vectors: [[Float]]
    let rowPtr: [Int32]
    let colIdx: [Int32]
    let queries: [[Float]]
    let groundTruth: [[Int]]
}

public enum BinaryLoaderError: Error {
    case fileNotFound(String)
    case invalidHeader
}

public final class BinaryLoader {
    
    public static func loadAGNews(from url: URL) throws -> ANNBenchmarkData {
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        
        return try data.withUnsafeBytes { buffer -> ANNBenchmarkData in
            guard let baseAddress = buffer.baseAddress else {
                throw BinaryLoaderError.invalidHeader
            }
            
            // 1. Cabecera (28 bytes = 7 x int32)
            let headerPtr = baseAddress.bindMemory(to: Int32.self, capacity: 7)
            let numVectors = Int(headerPtr[0])
            let numCentroids = Int(headerPtr[1])
            let dimension = Int(headerPtr[2])
            let numQueries = Int(headerPtr[3])
            let topK = Int(headerPtr[4])
            let rowPtrSize = Int(headerPtr[5])
            let colIdxSize = Int(headerPtr[6])
            
            print("[BinaryLoader] IVF-CSR: \(numVectors) vectores, \(numCentroids) centroides, \(dimension) dims.")
            var byteOffset = 28
            
            // 2. Centroides (Nodos Hub)
            let centroidsCount = numCentroids * dimension
            let centroidsPtr = baseAddress.advanced(by: byteOffset).bindMemory(to: Float.self, capacity: centroidsCount)
            let centroidsBuffer = UnsafeBufferPointer(start: centroidsPtr, count: centroidsCount)
            var centroids = [[Float]]()
            centroids.reserveCapacity(numCentroids)
            for i in 0..<numCentroids {
                centroids.append(Array(centroidsBuffer[(i * dimension)..<(i * dimension + dimension)]))
            }
            byteOffset += centroidsCount * MemoryLayout<Float>.size
            
            // 3. Vectores Base
            let vectorsCount = numVectors * dimension
            let vectorsPtr = baseAddress.advanced(by: byteOffset).bindMemory(to: Float.self, capacity: vectorsCount)
            let vectorsBuffer = UnsafeBufferPointer(start: vectorsPtr, count: vectorsCount)
            var vectors = [[Float]]()
            vectors.reserveCapacity(numVectors)
            for i in 0..<numVectors {
                vectors.append(Array(vectorsBuffer[(i * dimension)..<(i * dimension + dimension)]))
            }
            byteOffset += vectorsCount * MemoryLayout<Float>.size
            
            // 4. CSR rowPtr
            let rowPtrPtr = baseAddress.advanced(by: byteOffset).bindMemory(to: Int32.self, capacity: rowPtrSize)
            let rowPtrBuffer = UnsafeBufferPointer(start: rowPtrPtr, count: rowPtrSize)
            let rowPtr = Array(rowPtrBuffer)
            byteOffset += rowPtrSize * MemoryLayout<Int32>.size
            
            // 5. CSR colIdx
            let colIdxPtr = baseAddress.advanced(by: byteOffset).bindMemory(to: Int32.self, capacity: colIdxSize)
            let colIdxBuffer = UnsafeBufferPointer(start: colIdxPtr, count: colIdxSize)
            let colIdx = Array(colIdxBuffer)
            byteOffset += colIdxSize * MemoryLayout<Int32>.size
            
            // 6. Consultas
            let queriesCount = numQueries * dimension
            let queriesPtr = baseAddress.advanced(by: byteOffset).bindMemory(to: Float.self, capacity: queriesCount)
            let queriesBuffer = UnsafeBufferPointer(start: queriesPtr, count: queriesCount)
            var queries = [[Float]]()
            queries.reserveCapacity(numQueries)
            for i in 0..<numQueries {
                queries.append(Array(queriesBuffer[(i * dimension)..<(i * dimension + dimension)]))
            }
            byteOffset += queriesCount * MemoryLayout<Float>.size
            
            // 7. Ground Truth (Normalizado Coseno)
            let gtCount = numQueries * topK
            let gtPtr = baseAddress.advanced(by: byteOffset).bindMemory(to: Int32.self, capacity: gtCount)
            let gtBuffer = UnsafeBufferPointer(start: gtPtr, count: gtCount)
            var groundTruth = [[Int]]()
            groundTruth.reserveCapacity(numQueries)
            for i in 0..<numQueries {
                groundTruth.append(Array(gtBuffer[(i * topK)..<(i * topK + topK)]).map { Int($0) })
            }
            
            print("[BinaryLoader] Memoria cargada: ~\(byteOffset / 1024 / 1024) MB.")
            
            return ANNBenchmarkData(
                numVectors: numVectors, numCentroids: numCentroids, dimension: dimension,
                numQueries: numQueries, topK: topK,
                centroids: centroids, vectors: vectors,
                rowPtr: rowPtr, colIdx: colIdx,
                queries: queries, groundTruth: groundTruth
            )
        }
    }
}
