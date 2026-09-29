//
//  AegisSpGEMMTests.swift
//  AegisSpGEMMTests
//
//  Created for Phase 2.5 - Integrity Validation.
//  Strict Memory Standard: XCTest suite for UMA Allocation and CSR Compilation.
//

import XCTest
import Metal // REQUERIDO: Para resolver MTLStorageMode

final class AegisSpGEMMTests: XCTestCase {

    // MARK: - Pruebas de Fase 1 (Memoria UMA)
    
    func testUMAMemoryManagerInitialization() {
        let manager = UMAMemoryManager()
        XCTAssertNotNil(manager, "El UMAMemoryManager no debería ser nulo si el hardware M-Series es compatible.")
    }
    
    func testUMAMemoryManagerAllocation() {
        let manager = UMAMemoryManager()
        let dummyData: [Int32] = [0, 1, 2, 3, 4]
        
        // Desempaquetado seguro para evitar el error "value of type 'T?'"
        guard let buffer = dummyData.withUnsafeBufferPointer({ ptr in
            manager.allocateCSRRowPointers(ptr.baseAddress!, count: UInt(dummyData.count))
        }) else {
            XCTFail("El búfer UMA retornó nulo en datos válidos.")
            return
        }
        
        XCTAssertEqual(buffer.length, dummyData.count * MemoryLayout<Int32>.size, "El tamaño del búfer en bytes es incorrecto.")
        // Uso explícito del tipo Enum de Metal
        XCTAssertEqual(buffer.storageMode, MTLStorageMode.shared, "El contrato exige MTLStorageModeShared.")
    }

    // MARK: - Pruebas de Fase 2 (Compilación Topológica CSR)
    
    func testCSRCompilerValidGraph() {
        do {
            let compiler = try CSRTopologyCompiler()
            
            let adjacencyList: [[Int32]] = [
                [1, 2],
                [2],
                []
            ]
            
            let vectorDim = 4
            let denseVectors: [[Float]] = [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2]
            ]
            
            let context = try compiler.compileToUMA(
                adjacencyList: adjacencyList,
                denseVectors: denseVectors,
                vectorDim: vectorDim
            )
            
            XCTAssertEqual(context.numNodes, 3)
            XCTAssertEqual(context.numEdges, 3)
            XCTAssertEqual(context.vectorDim, 4)
            
            XCTAssertNotNil(context.rowPointersBuffer)
            XCTAssertNotNil(context.colIndicesBuffer)
            XCTAssertNotNil(context.denseVectorsBuffer)
            
            let rowPtrs = context.rowPointersBuffer.contents().bindMemory(to: Int32.self, capacity: 4)
            XCTAssertEqual(rowPtrs[0], 0)
            XCTAssertEqual(rowPtrs[1], 2)
            XCTAssertEqual(rowPtrs[2], 3)
            XCTAssertEqual(rowPtrs[3], 3)
            
        } catch {
            XCTFail("Fallo inesperado: \(error)")
        }
    }
    
    func testCSRCompilerNodeCountMismatch() {
        do {
            let compiler = try CSRTopologyCompiler()
            let adjacencyList: [[Int32]] = [[1], [0]]
            let denseVectors: [[Float]] = [[0.1, 0.2]] // Vector faltante
            
            _ = try compiler.compileToUMA(adjacencyList: adjacencyList, denseVectors: denseVectors, vectorDim: 2)
            XCTFail("Se esperaba CSRCompilationError por inconsistencia.")
        } catch CSRCompilationError.invalidGraphTopology(let message) {
            XCTAssertTrue(message.contains("Inconsistencia"))
        } catch {
            XCTFail("Error inesperado: \(error)")
        }
    }
    
    func testCSRCompilerVectorDimensionMismatch() {
        do {
            let compiler = try CSRTopologyCompiler()
            let adjacencyList: [[Int32]] = [[0]]
            let denseVectors: [[Float]] = [[0.1, 0.2, 0.3]]
            
            _ = try compiler.compileToUMA(adjacencyList: adjacencyList, denseVectors: denseVectors, vectorDim: 2)
            XCTFail("Se esperaba CSRCompilationError por dimensionalidad.")
        } catch CSRCompilationError.invalidGraphTopology(let message) {
            XCTAssertTrue(message.contains("dimensionalidad incorrecta"))
        } catch {
            XCTFail("Error inesperado: \(error)")
        }
    }
}
