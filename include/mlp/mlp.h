#ifndef _MLP_H_
#define _MLP_H_

#include <immintrin.h> // AVX, AVX2, FMA, AVX-512
#include <stdlib.h> // aligned_alloc, rand, size_t
#include <assert.h> // assert
#include <string.h> // memcpy

typedef float mlp_float_t;

#define VEC_BYTES 32
#define VEC_FLOATS (VEC_BYTES/sizeof(mlp_float_t))
#define MLP_CACHE_LINE 64
#define MLP_VALID_VEC_SZ(x) ((x) + ((x)%VEC_FLOATS))
#define MLP_VECS(x) ((x)/VEC_FLOATS + (size_t)(((x)%VEC_FLOATS > 0) & 1))

struct mlp;
struct mlp_layer;


void mlp_init(struct mlp *mlp, size_t const n_neurons_per_layer[], size_t n_layers); // Allocates memory, initializes it and set pointers
void mlp_load_input(struct mlp *mlp, mlp_float_t const input[]); // Memcpys de single input row into aligned address (batch size = 1 for the moment)
mlp_float_t const *mlp_forward(struct mlp *mlp); // The input goes into the first hidden layer and that output is the input of the next and so on. Return the pointer to the last layer output
void mlp_load_answer(struct mlp *mlp, mlp_float_t const answer[]); // Memcpys de single answer row into aligned address (batch size = 1 for the moment)
void mlp_backprog(struct mlp *mlp); // The cached output is compared to the answer and propagates backwards the change in the right direction based on the influence of each neuron


struct mlp{
    size_t n_layers;
    struct mlp_layer *layers;
    mlp_float_t *answer;
    mlp_float_t *aux;
};

struct mlp_layer{
    size_t n_outputs;
    mlp_float_t *weights;
    mlp_float_t *outputs;
    mlp_float_t *deltas;
};

#endif // _MLP_H_