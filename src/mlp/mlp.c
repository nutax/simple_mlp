#include "mlp/mlp.h"


// Allocates memory, initializes it and set pointers
void mlp_init(struct mlp *mlp, size_t n_vars_per_input, size_t const *n_neurons_per_hidden, size_t n_hiddens, size_t n_vars_per_output){

}

// Memcpys de single input row into aligned address (batch size = 1 for the moment)
void mlp_load_input(struct mlp *mlp, float_t const *input){

}

// The input goes into the first hidden layer and that output is the input of the next and so on. Return the pointer to the last layer output
float_t const *mlp_forward(struct mlp *mlp){

}

// Memcpys de single answer row into aligned address (batch size = 1 for the moment)
void mlp_load_answer(struct mlp *mlp, float_t const *answer){

}

// The cached output is compared to the answer and propagates backwards the change in the right direction based on the influence of each neuron
void mlp_backprog(struct mlp *mlp){

}
