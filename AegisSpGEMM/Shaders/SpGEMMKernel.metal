//
//  SpGEMMKernel.metal
//  AegisSpGEMM
//
//  Created for Phase 6 - IVF-CSR Acceleration.
//  Strict Memory Standard: Scatter-based topology propagation.
//

#include <metal_stdlib>
using namespace metal;

/// @kernel spgemm_ivf_propagation
/// @brief Propagación Scatter: Los centroides activos iluminan a todos los vectores de su celda de Voronoi.
kernel void spgemm_ivf_propagation(device const int32_t* row_pointers [[buffer(0)]],
                                   device const int32_t* col_indices [[buffer(1)]],
                                   device const float* centroid_state [[buffer(2)]],
                                   device float* vector_state [[buffer(3)]],
                                   constant uint& num_centroids [[buffer(4)]],
                                   uint id [[thread_position_in_grid]]) {
    
    if (id >= num_centroids) return;
    
    // Filtro topológico temprano: Si el centroide no fue seleccionado en la CPU, abortar hilo.
    if (centroid_state[id] <= 0.0) return;
    
    int start_offset = row_pointers[id];
    int end_offset = row_pointers[id + 1];
    
    // Activa masivamente en paralelo a todos los vectores hijos de este cluster
    for (int i = start_offset; i < end_offset; ++i) {
        int vector_id = col_indices[i];
        vector_state[vector_id] = 1.0;
    }
}

/// @kernel sparse_similarity
/// @brief Calcula el Producto Interno (Similitud) estrictamente sobre los nodos activos.
kernel void sparse_similarity(device const float* query_vector [[buffer(0)]],
                              device const float* dense_vectors [[buffer(1)]],
                              device const float* active_state [[buffer(2)]],
                              device float* results [[buffer(3)]],
                              constant uint& vector_dim [[buffer(4)]],
                              constant uint& num_nodes [[buffer(5)]],
                              uint id [[thread_position_in_grid]]) {
    
    if (id >= num_nodes) return;
    
    if (active_state[id] <= 0.0) {
        results[id] = -INFINITY;
        return;
    }
    
    float score = 0.0;
    uint offset = id * vector_dim;
    
    // ALU GPU Parallel Math (Dot Product optimizado por FMA)
    for(uint i = 0; i < vector_dim; ++i) {
        score += query_vector[i] * dense_vectors[offset + i];
    }
    
    results[id] = score;
}
