//
//  BinaryLoader.swift
//  AegisSpGEMM
//
//  Created for Phase 6 - IVF-CSR Ingestion (Updated Phase 7.2 - Zero-Fragmentation Flat Loader).
//  Strict Memory Standard: mmap (Memory-Mapped File) to Contiguous Flat Arrays.
//

import Foundation

public struct ANNBenchmarkData {
    public let numVectors: Int
    public let numCentroids: Int
    public let dimension: Int
    public let numQueries: Int
    public let topK: Int
    
    /// Matriz continua de centroides [numCentroids * dimension] lista para AMX vDSP_mmul.
    public let flattenedCentroids: [Float]
    
    /// Matriz continua de vectores base [numVectors * dimension] lista para UMA Zero-Copy.
    public let flattenedVectors: [Float]
    
    public let rowPtr: [Int32]
    public let colIdx: [Int32]
    public let queries: [[Float]]
    public let groundTruth: [[Int]]
}

public enum BinaryLoaderError: Error {
    case fileNotFound(String)
    case invalidHeader
    case corruptedPayload(String)
}

public final class BinaryLoader {
    
    /// @brief Cargador universal para cualquier artefacto binario bajo el contrato IVF-CSR de 28 bytes.
    public static func loadDataset(from url: URL) throws -> ANNBenchmarkData {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw BinaryLoaderError.fileNotFound("No existe el archivo en: \(url.path)")
        }
        
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        let totalFileBytes = data.count
        
        return try data.withUnsafeBytes { buffer -> ANNBenchmarkData in
            guard let baseAddress = buffer.baseAddress, totalFileBytes >= 28 else {
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
            
            guard numVectors > 0, numCentroids > 0, dimension > 0, rowPtrSize == numCentroids + 1 else {
                throw BinaryLoaderError.invalidHeader
            }
            
            print("[BinaryLoader] IVF-CSR: \(numVectors) vectores, \(numCentroids) centroides, \(dimension) dims.")
            var byteOffset = 28
            
            func validateBounds(adding bytesToAdd: Int, section: String) throws {
                if byteOffset + bytesToAdd > totalFileBytes {
                    throw BinaryLoaderError.corruptedPayload("Desbordamiento leyendo sección \(section).")
                }
            }
            
            // 2. Centroides Contiguos (1 sola copia plana desde mmap)
            let centroidsCount = numCentroids * dimension
            let centroidsBytes = centroidsCount * MemoryLayout<Float>.size
            try validateBounds(adding: centroidsBytes, section: "Centroides")
            let centroidsPtr = baseAddress.advanced(by: byteOffset).bindMemory(to: Float.self, capacity: centroidsCount)
            let flattenedCentroids = Array(UnsafeBufferPointer(start: centroidsPtr, count: centroidsCount))
            byteOffset += centroidsBytes
            
            // 3. Vectores Base Contiguos (1 sola copia plana desde mmap, cero fragmentación [[Float]])
            let vectorsCount = numVectors * dimension
            let vectorsBytes = vectorsCount * MemoryLayout<Float>.size
            try validateBounds(adding: vectorsBytes, section: "Vectores Base")
            let vectorsPtr = baseAddress.advanced(by: byteOffset).bindMemory(to: Float.self, capacity: vectorsCount)
            let flattenedVectors = Array(UnsafeBufferPointer(start: vectorsPtr, count: vectorsCount))
            byteOffset += vectorsBytes
            
            // 4. CSR rowPtr
            let rowPtrBytes = rowPtrSize * MemoryLayout<Int32>.size
            try validateBounds(adding: rowPtrBytes, section: "rowPtr")
            let rowPtrPtr = baseAddress.advanced(by: byteOffset).bindMemory(to: Int32.self, capacity: rowPtrSize)
            let rowPtr = Array(UnsafeBufferPointer(start: rowPtrPtr, count: rowPtrSize))
            byteOffset += rowPtrBytes
            
            // 5. CSR colIdx
            let colIdxBytes = colIdxSize * MemoryLayout<Int32>.size
            try validateBounds(adding: colIdxBytes, section: "colIdx")
            let colIdxPtr = baseAddress.advanced(by: byteOffset).bindMemory(to: Int32.self, capacity: colIdxSize)
            let colIdx = Array(UnsafeBufferPointer(start: colIdxPtr, count: colIdxSize))
            byteOffset += colIdxBytes
            
            // 6. Consultas
            let queriesCount = numQueries * dimension
            let queriesBytes = queriesCount * MemoryLayout<Float>.size
            try validateBounds(adding: queriesBytes, section: "Queries")
            let queriesPtr = baseAddress.advanced(by: byteOffset).bindMemory(to: Float.self, capacity: queriesCount)
            let queriesBuffer = UnsafeBufferPointer(start: queriesPtr, count: queriesCount)
            var queries = [[Float]]()
            queries.reserveCapacity(numQueries)
            for i in 0..<numQueries {
                queries.append(Array(queriesBuffer[(i * dimension)..<(i * dimension + dimension)]))
            }
            byteOffset += queriesBytes
            
            // 7. Ground Truth (Normalizado Coseno)
            let gtCount = numQueries * topK
            let gtBytes = gtCount * MemoryLayout<Int32>.size
            try validateBounds(adding: gtBytes, section: "GroundTruth")
            let gtPtr = baseAddress.advanced(by: byteOffset).bindMemory(to: Int32.self, capacity: gtCount)
            let gtBuffer = UnsafeBufferPointer(start: gtPtr, count: gtCount)
            var groundTruth = [[Int]]()
            groundTruth.reserveCapacity(numQueries)
            for i in 0..<numQueries {
                groundTruth.append(Array(gtBuffer[(i * topK)..<(i * topK + topK)]).map { Int($0) })
            }
            byteOffset += gtBytes
            
            print("[BinaryLoader] Memoria cargada de forma contigua: ~\(byteOffset / 1024 / 1024) MB.")
            
            return ANNBenchmarkData(
                numVectors: numVectors,
                numCentroids: numCentroids,
                dimension: dimension,
                numQueries: numQueries,
                topK: topK,
                flattenedCentroids: flattenedCentroids,
                flattenedVectors: flattenedVectors,
                rowPtr: rowPtr,
                colIdx: colIdx,
                queries: queries,
                groundTruth: groundTruth
            )
        }
    }
    
    /// Alias de compatibilidad hacia atrás.
    public static func loadAGNews(from url: URL) throws -> ANNBenchmarkData {
        return try loadDataset(from: url)
    }
}
