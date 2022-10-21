#ifndef _MLP_H_
#define _MLP_H_

#include <immintrin.h> // AVX, AVX2, FMA, AVX-512
#include <stdlib.h> // aligned_alloc, rand, size_t


struct mlp;
struct mlp_layer;


void mlp_init(struct mlp *mlp, float const *neurons_per_layer, size_t layers); // Allocates memory, initializes it and set pointers
float const *mlp_forward(struct mlp *mlp, float const *input); // The input goes into the first hidden layer and that output is the input of the next and so on. Return the pointer to the last layer output
void mlp_backprog(struct mlp *mlp, float const *answer); // The cached output is compared to the answer and propagates backwards the change in the right direction based on the influence of each neuron


struct mlp{
    int n_layers;
    struct mlp_layer *layers;
};

struct mlp_layer{
    size_t n_outputs;
    float *weights;
    float *outputs;
    float *deltas;
};

#endif // _MLP_H_