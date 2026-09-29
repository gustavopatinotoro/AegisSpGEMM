//
//  SpGEMMKernel.metal
//  AegisSpGEMM
//
//  Created for Phase 3 - GPU Acceleration.
//  Strict Memory Standard: Zero-branching Sparse Matrix-Vector Multiply (SpMV).
//

#include <metal_stdlib>
using namespace metal;

/// @kernel spgemm_propagation
/// @brief Propaga la activación topológica (rutas de navegación) usando álgebra matricial pura.
/// Reemplaza la navegación de grafos (saltar punteros) por una matriz dispersa multiplicando un vector de estado.
///
/// @param row_pointers Punteros CSR de inicio de fila.
/// @param col_indices Conexiones topológicas CSR (columnas dispersas).
/// @param current_state Vector [1 x N] con el estado actual de la búsqueda (1.0 = nodo activo, 0.0 = inactivo).
/// @param next_state Vector [1 x N] resultante (salida) con los nodos iluminados tras la difusión.
/// @param num_nodes Total de nodos en el sistema para evitar desbordamientos de hilos.
kernel void spgemm_propagation(device const int32_t* row_pointers [[buffer(0)]],
                               device const int32_t* col_indices [[buffer(1)]],
                               device const float* current_state [[buffer(2)]],
                               device float* next_state [[buffer(3)]],
                               constant uint& num_nodes [[buffer(4)]],
                               uint id [[thread_position_in_grid]]) {
    
    // Contrato de seguridad: Evitar acceso fuera de los límites de memoria
    if (id >= num_nodes) return;
    
    int start_offset = row_pointers[id];
    int end_offset = row_pointers[id + 1];
    
    float activation_sum = 0.0;
    
    // Operación SpMV (Sparse Matrix-Vector)
    // Calcula la energía que recibe el nodo actual desde todos sus vecinos conectados en la matriz de adyacencia.
    for (int i = start_offset; i < end_offset; ++i) {
        int neighbor_node = col_indices[i];
        // En una topología diferenciable, aquí se multiplicaría por los `edge_weights`.
        // Para esta prueba fundacional, asumimos un peso binario difuso de 1.0.
        activation_sum += current_state[neighbor_node];
    }
    
    next_state[id] = activation_sum;
}


/// @kernel sparse_similarity
/// @brief Calcula el Producto Interno (Similitud) estrictamente sobre los nodos activos.
///
/// @param query_vector El embedding de la consulta del usuario.
/// @param dense_vectors Los embeddings de los documentos aplanados.
/// @param active_state El vector de estado resultante de la propagación SpGEMM.
/// @param results Búfer de salida con las puntuaciones de similitud.
/// @param vector_dim Dimensionalidad latente (ej. 384).
/// @param num_nodes Total de nodos para seguridad del hilo.
kernel void sparse_similarity(device const float* query_vector [[buffer(0)]],
                              device const float* dense_vectors [[buffer(1)]],
                              device const float* active_state [[buffer(2)]],
                              device float* results [[buffer(3)]],
                              constant uint& vector_dim [[buffer(4)]],
                              constant uint& num_nodes [[buffer(5)]],
                              uint id [[thread_position_in_grid]]) {
    
    if (id >= num_nodes) return;
    
    // FILTRADO TOPOLÓGICO Y MATRICIAL:
    // Si el SpGEMM determinó que este nodo no está activo (valor <= 0.0), lo descartamos
    // sin gastar ciclos de GPU calculando similitudes innecesarias.
    if (active_state[id] <= 0.0) {
        results[id] = -INFINITY;
        return;
    }
    
    // Si sobrevivió al filtro topológico, calculamos la métrica densa de alta dimensionalidad.
    float score = 0.0;
    uint offset = id * vector_dim;
    
    // ALU GPU Parallel Math (Dot Product)
    for(uint i = 0; i < vector_dim; ++i) {
        score += query_vector[i] * dense_vectors[offset + i];
    }
    
    results[id] = score;
}


