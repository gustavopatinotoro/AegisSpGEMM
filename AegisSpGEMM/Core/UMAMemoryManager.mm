//
//  UMAMemoryManager.mm
//  AegisSpGEMM
//
//  Created for Phase 1 - UMA Memory Core.
//  Strict Memory Standard: Objective-C++ Bridging for Apple Silicon Zero-Copy.
//

#import "UMAMemoryManager.h"
#include <stdexcept>
#include <iostream>

@interface UMAMemoryManager ()
@property (nonatomic, strong) id<MTLDevice> device;
@end

@implementation UMAMemoryManager

- (instancetype)init {
    self = [super init];
    if (self) {
        // Inicialización crítica: Solicitar acceso al procesador primario.
        _device = MTLCreateSystemDefaultDevice();
        if (!_device) {
            std::cerr << "[CRITICAL ERROR] Fallo al adquirir el MTLDevice primario. Hardware incompatible." << std::endl;
            return nil;
        }
        
        // Verificación arquitectónica (Precondición de Hardware)
        if (![_device hasUnifiedMemory]) {
             std::cerr << "[WARNING] Arquitectura de memoria unificada no detectada. Rendimiento SpGEMM degradado." << std::endl;
        }
    }
    return self;
}

- (nullable id<MTLBuffer>)allocateCSRRowPointers:(const int32_t *)rowPointers count:(NSUInteger)count {
    if (rowPointers == nullptr || count == 0) {
        std::cerr << "[CONTRACT VIOLATION] rowPointers es nulo o el contador es 0." << std::endl;
        return nil;
    }
    
    size_t bufferSize = count * sizeof(int32_t);
    // FIXED: API Obj-C correcta 'newBufferWithBytes'
    id<MTLBuffer> buffer = [self.device newBufferWithBytes:rowPointers
                                                    length:bufferSize
                                                   options:MTLResourceStorageModeShared];
    if (!buffer) {
        std::cerr << "[OOM ERROR] Falla de asignación física (Out of Memory). Solicitados: " << bufferSize << " bytes." << std::endl;
    }
    return buffer;
}

- (nullable id<MTLBuffer>)allocateCSRColumnIndices:(const int32_t *)colIndices count:(NSUInteger)count {
    if (colIndices == nullptr || count == 0) {
        std::cerr << "[CONTRACT VIOLATION] colIndices es nulo o el contador es 0." << std::endl;
        return nil;
    }
    
    size_t bufferSize = count * sizeof(int32_t);
    // FIXED: API Obj-C correcta 'newBufferWithBytes'
    id<MTLBuffer> buffer = [self.device newBufferWithBytes:colIndices
                                                    length:bufferSize
                                                   options:MTLResourceStorageModeShared];
    if (!buffer) {
         std::cerr << "[OOM ERROR] Falla de asignación física (Out of Memory). Solicitados: " << bufferSize << " bytes." << std::endl;
    }
    return buffer;
}

- (nullable id<MTLBuffer>)allocateDenseVectors:(const float *)vectors count:(NSUInteger)count {
    if (vectors == nullptr || count == 0) {
        std::cerr << "[CONTRACT VIOLATION] vectors array es nulo o el contador es 0." << std::endl;
        return nil;
    }
    
    size_t bufferSize = count * sizeof(float);
    // FIXED: API Obj-C correcta 'newBufferWithBytes'
    id<MTLBuffer> buffer = [self.device newBufferWithBytes:vectors
                                                    length:bufferSize
                                                   options:MTLResourceStorageModeShared];
    if (!buffer) {
         std::cerr << "[OOM ERROR] Falla de asignación física (Out of Memory) para tensores densos. Solicitados: " << bufferSize << " bytes." << std::endl;
    }
    return buffer;
}

@end
