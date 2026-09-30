//
//  SpGEMMKernel.metal
//  AegisSpGEMM
//
//  Created for Phase 7.1 - Compacted Similarity Kernel.
//  Strict Memory Standard: Stream Compaction Kernel with zero thread divergence.
//

#include <metal_stdlib>
using namespace metal;

/// @kernel sparse_similarity_compact
/// @brief Calcula el producto interno estrictamente sobre los IDs candidatos proporcionados.
kernel void sparse_similarity_compact(device const float* query_vector [[buffer(0)]],
                                      device const float* dense_vectors [[buffer(1)]],
                                      device const int32_t* candidate_indices [[buffer(2)]],
                                      device float* results_scores [[buffer(3)]],
                                      constant uint& vector_dim [[buffer(4)]],
                                      constant uint& num_candidates [[buffer(5)]],
                                      uint id [[thread_position_in_grid]]) {
    
    // Contrato de seguridad: Si el hilo excede la cantidad real de candidatos, aborta instantáneamente.
    if (id >= num_candidates) return;
    
    int32_t node_id = candidate_indices[id];
    uint offset = (uint)node_id * vector_dim;
    
    float score = 0.0;
    
    // Cómputo denso altamente optimizado por unidades FMA de la GPU
    for (uint i = 0; i < vector_dim; ++i) {
        score += query_vector[i] * dense_vectors[offset + i];
    }
    
    results_scores[id] = score;
}
