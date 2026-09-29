//
//  UMAMemoryManager.h
//  AegisSpGEMM
//
//  Created for Phase 1 - UMA Memory Core.
//  Strict Memory Standard: Objective-C++ Bridging for Apple Silicon Zero-Copy.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * @class UMAMemoryManager
 * @brief Gestor de memoria blindado para tensores CSR en arquitectura de Memoria Unificada (UMA).
 *
 * @invariant El dispositivo Metal subyacente (device) no debe ser nulo.
 * @invariant Todos los búferes creados deben utilizar MTLResourceStorageModeShared para evitar copias en PCIe.
 */
@interface UMAMemoryManager : NSObject

/**
 * @brief Inicializa el administrador de memoria vinculándolo al procesador gráfico predeterminado.
 * @pre Debe existir un hardware compatible con Metal (M-Series garantizado).
 * @post Se establece la conexión con el MTLDevice. Si falla, el entorno no es seguro para ejecución.
 */
- (instancetype)init;

/**
 * @brief Asigna memoria UMA de solo lectura para los arreglos de punteros de fila (CSR Row Pointers).
 * @pre El arreglo en C++ no debe ser nulo y el contador de elementos debe ser > 0.
 * @pre El tamaño total en bytes no debe exceder los márgenes de seguridad para un entorno de 8GB.
 * @post Retorna un MTLBuffer válido con storageModeShared, o nil si la RAM física se ha agotado.
 *
 * @param rowPointers Puntero al arreglo de enteros (32-bit).
 * @param count Cantidad total de elementos.
 */
- (nullable id<MTLBuffer>)allocateCSRRowPointers:(const int32_t *)rowPointers count:(NSUInteger)count;

/**
 * @brief Asigna memoria UMA de solo lectura para los arreglos de columnas dispersas (CSR Column Indices).
 * @pre El arreglo en C++ no debe ser nulo y el contador de elementos debe ser > 0.
 * @post Retorna un MTLBuffer válido con storageModeShared, o nil si la RAM física se ha agotado.
 *
 * @param colIndices Puntero al arreglo de enteros (32-bit).
 * @param count Cantidad total de elementos.
 */
- (nullable id<MTLBuffer>)allocateCSRColumnIndices:(const int32_t *)colIndices count:(NSUInteger)count;

/**
 * @brief Asigna memoria UMA para los vectores densos (Embeddings).
 * @pre El arreglo no debe ser nulo. vectorDim * numVectors * 4 bytes debe caber en RAM.
 * @post Retorna un MTLBuffer válido con storageModeShared.
 *
 * @param vectors Puntero flotante a la matriz linealizada.
 * @param count Cantidad de números flotantes totales (vectorDim * numVectors).
 */
- (nullable id<MTLBuffer>)allocateDenseVectors:(const float *)vectors count:(NSUInteger)count;

@end

NS_ASSUME_NONNULL_END
