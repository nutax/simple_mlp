#include "mlp/mlp.h"


// Allocates memory, initializes it and set pointers
void mlp_init(struct mlp *mlp, size_t const n_neurons_per_layer[], size_t n_layers){
    size_t i_layer, j_weight, n_outputs, n_inputs, n_weights;
    struct mlp_layer *layer;

    assert(mlp != NULL);
    assert(n_neurons_per_layer != NULL);
    assert(n_layers >= 3);
    
    mlp->n_layers = n_layers;

    mlp->layers = aligned_alloc(MLP_CACHE_LINE, sizeof(struct mlp_layer)*n_layers);
    assert(mlp->layers != NULL);


    i_layer = 0;
    n_outputs = n_neurons_per_layer[i_layer];
    layer = &(mlp->layers[i_layer]);
    layer->n_outputs = n_outputs;
    layer->outputs = aligned_alloc(MLP_CACHE_LINE, sizeof(mlp_float_t)*MLP_VALID_VEC_SZ(n_outputs));
    assert(layer->outputs != NULL);


    for(i_layer = 1; i_layer<n_layers; ++i_layer){
        n_inputs = n_neurons_per_layer[i_layer-1];
        n_outputs = n_neurons_per_layer[i_layer];
        n_weights = MLP_VALID_VEC_SZ(n_inputs + 1)*n_outputs;
        layer = &(mlp->layers[i_layer]);

        layer->n_outputs = n_outputs;

        layer->weights = aligned_alloc(MLP_CACHE_LINE, sizeof(mlp_float_t)*(n_weights));
        assert(layer->outputs != NULL);

        layer->outputs = aligned_alloc(MLP_CACHE_LINE, sizeof(mlp_float_t)*MLP_VALID_VEC_SZ(n_outputs));
        assert(layer->outputs != NULL);

        layer->deltas = aligned_alloc(MLP_CACHE_LINE, sizeof(mlp_float_t)*MLP_VALID_VEC_SZ(n_outputs));
        assert(layer->deltas != NULL);

        for(j_weight = 0; j_weight < n_weights; ++j_weight){
            layer->weights[j_weight] = ((mlp_float_t)rand())/((mlp_float_t)RAND_MAX);
        }
    }
}

// Memcpys de single input row into aligned address (batch size = 1 for the moment)
void mlp_load_input(struct mlp *mlp, mlp_float_t const input[]){
    struct mlp_layer *input_layer;

    input_layer = &(mlp->layers[0]);
    memcpy(input_layer->outputs, input, sizeof(mlp_float_t)*(input_layer->n_outputs));
}

// The input goes into the first hidden layer and that output is the input of the next and so on. Return the pointer to the last layer output
mlp_float_t const *mlp_forward(struct mlp *mlp){
    size_t i_layer, j_neuron, k_wvec, n_outputs, n_inputs, n_wvecs, n_layers, neuron_weights_padding;
    __m256 *inputs, *weights;
    mlp_float_t *outputs;
    struct mlp_layer *prev_layer, *layer;

    n_layers = mlp->n_layers;

    for(i_layer = 1; i_layer<n_layers; ++i_layer){
        prev_layer = &(mlp->layers[i_layer-1]);
        layer = &(mlp->layers[i_layer]);

        n_inputs = prev_layer->n_outputs;
        n_outputs = layer->n_outputs;
        n_wvecs = MLP_VECS(n_inputs+1);

        inputs = (__m256 *)(prev_layer->outputs);

        for(j_neuron = 0; j_neuron < n_outputs; ++j_neuron){
            weights = (__m256 *)(layer->weights + )
            for(k_wvec = 0; k_wvec<n_wvecs; ++k_wvec){
                
            }
        }
    }

    return mlp->layers[i_layer-1].outputs;
}

// Memcpys de single answer row into aligned address (batch size = 1 for the moment)
void mlp_load_answer(struct mlp *mlp, mlp_float_t const answer[]){

}

// The cached output is compared to the answer and propagates backwards the change in the right direction based on the influence of each neuron
void mlp_backprog(struct mlp *mlp){

}
