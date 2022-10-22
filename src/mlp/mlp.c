#include "mlp/mlp.h"

// Allocates memory, initializes it and set pointers
void mlp_init(struct mlp *mlp, size_t const n_neurons_per_layer[], size_t n_layers)
{
    size_t i_layer, j_neuron, k_weight, n_outputs, n_inputs, wvec_size, ovec_size, wmat_size;
    mlp_float_t *weights;
    struct mlp_layer *layer;

    assert(mlp != NULL);
    assert(n_neurons_per_layer != NULL);
    assert(n_layers >= 3);

    mlp->n_layers = n_layers;

    mlp->layers = aligned_alloc(MLP_CACHE_LINE, sizeof(struct mlp_layer) * n_layers);
    assert(mlp->layers != NULL);

    i_layer = 0;
    n_outputs = n_neurons_per_layer[i_layer];
    layer = &(mlp->layers[i_layer]);
    layer->n_outputs = n_outputs;
    ovec_size = MLP_VALID_VEC_SZ(n_outputs + 1);
    layer->outputs = aligned_alloc(MLP_CACHE_LINE, sizeof(mlp_float_t) * ovec_size);
    assert(layer->outputs != NULL);
    for (j_neuron = 0; j_neuron < ovec_size; ++j_neuron)
    {
        layer->outputs[j_neuron] = 0;
    }
    layer->outputs[n_outputs] = 1;

    for (i_layer = 1; i_layer < n_layers; ++i_layer)
    {
        n_inputs = n_neurons_per_layer[i_layer - 1];
        n_outputs = n_neurons_per_layer[i_layer];
        ovec_size = MLP_VALID_VEC_SZ(n_outputs + 1);
        wvec_size = MLP_VALID_VEC_SZ(n_inputs + 1);
        wmat_size = wvec_size * n_outputs;
        layer = &(mlp->layers[i_layer]);

        layer->n_outputs = n_outputs;

        layer->weights = aligned_alloc(MLP_CACHE_LINE, sizeof(mlp_float_t) * wmat_size);
        assert(layer->weights != NULL);
        for (k_weight = 0; k_weight < wmat_size; ++k_weight)
        {
            layer->weights[k_weight] = 0;
        }
        for (j_neuron = 0; j_neuron < n_outputs; ++j_neuron)
        {
            weights = (layer->weights) + j_neuron * wvec_size;
            for (k_weight = 0; k_weight < (n_inputs + 1); ++k_weight)
            {
                layer->weights[k_weight] = ((mlp_float_t)rand()) / ((mlp_float_t)RAND_MAX);
            }
        }

        layer->outputs = aligned_alloc(MLP_CACHE_LINE, sizeof(mlp_float_t) * ovec_size);
        assert(layer->outputs != NULL);
        for (j_neuron = 0; j_neuron < ovec_size; ++j_neuron)
        {
            layer->outputs[j_neuron] = 0;
        }
        layer->outputs[n_outputs] = 1;

        layer->deltas = aligned_alloc(MLP_CACHE_LINE, sizeof(mlp_float_t) * ovec_size);
        assert(layer->deltas != NULL);
        for (j_neuron = 0; j_neuron < ovec_size; ++j_neuron)
        {
            layer->deltas[j_neuron] = 0;
        }
    }

    mlp->answers = aligned_alloc(MLP_CACHE_LINE, sizeof(struct mlp_layer) * ovec_size);
    assert(mlp->answers != NULL);
    for (j_neuron = 0; j_neuron < ovec_size; ++j_neuron)
    {
        mlp->answers[j_neuron] = 0;
    }
}

// Memcpys de single input row into aligned address (batch size = 1 for the moment)
void mlp_load_input(struct mlp *mlp, mlp_float_t const input[])
{
    struct mlp_layer *input_layer;

    input_layer = &(mlp->layers[0]);
    memcpy(input_layer->outputs, input, sizeof(mlp_float_t) * (input_layer->n_outputs));
}

// The input goes into the first hidden layer and that output is the input of the next and so on. Return the pointer to the last layer output
mlp_float_t const *mlp_forward(struct mlp *mlp)
{
    size_t i_layer, j_neuron, k_wvec, n_outputs, n_inputs, n_wvecs, wvec_size, n_layers;
    struct mlp_layer *prev_layer, *layer;
    __m256 *inputs, *weights, sum;
    mlp_float_t *outputs;
    float net_energy;

    n_layers = mlp->n_layers;

    for (i_layer = 1; i_layer < n_layers; ++i_layer)
    {
        prev_layer = &(mlp->layers[i_layer - 1]);
        layer = &(mlp->layers[i_layer]);

        n_inputs = prev_layer->n_outputs;
        n_outputs = layer->n_outputs;
        wvec_size = MLP_VALID_VEC_SZ(n_inputs + 1);
        n_wvecs = wvec_size / VEC_FLOATS;
        printf("%ld\n", n_wvecs);

        inputs = (__m256 *)(prev_layer->outputs);
        outputs = layer->outputs;

        for (j_neuron = 0; j_neuron < n_outputs; ++j_neuron)
        {
            weights = (__m256 *)((layer->weights) + j_neuron * wvec_size);
            sum = _mm256_set1_ps(0);
            for (k_wvec = 0; k_wvec < n_wvecs; ++k_wvec)
            {
                printf("%ld\n", k_wvec);
                sum = _mm256_fmadd_ps(inputs[k_wvec], weights[k_wvec], sum);
                printf("%ld\n", k_wvec);
            }
            net_energy = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
            outputs[j_neuron] = (mlp_float_t)(1 / (1 + exp(-net_energy))); // sigmoid activation function
        }
    }

    return mlp->layers[i_layer - 1].outputs;
}

// Memcpys de single answer row into aligned address (batch size = 1 for the moment)
void mlp_load_answer(struct mlp *mlp, mlp_float_t const answer[])
{
    struct mlp_layer *output_layer;

    output_layer = &(mlp->layers[mlp->n_layers - 1]);
    memcpy(mlp->answers, answer, sizeof(mlp_float_t) * (output_layer->n_outputs));
}

// The cached output is compared to the answer and propagates backwards the change in the right direction based on the influence of each neuron
void mlp_backprog(struct mlp *mlp, mlp_float_t rate)
{
    size_t i_layer, j_neuron, k_vec, n_outputs, n_inputs, wvec_size, n_wvecs, n_layers, ovec_size, n_ovecs, next_neuron, n_next_outputs, next_wvec_size;
    struct mlp_layer *prev_layer, *layer, *next_layer;
    __m256 *inputs, *weights, *outputs, *answers, *deltas, aux1, aux2, ones;

    ones = _mm256_set1_ps(1);

    n_layers = mlp->n_layers;
    i_layer = n_layers - 1;

    layer = &(mlp->layers[i_layer]);
    prev_layer = &(mlp->layers[i_layer - 1]);

    n_inputs = prev_layer->n_outputs;
    n_outputs = layer->n_outputs;

    wvec_size = MLP_VALID_VEC_SZ(n_inputs + 1);
    ovec_size = MLP_VALID_VEC_SZ(n_outputs);

    n_wvecs = wvec_size / VEC_FLOATS;
    n_ovecs = ovec_size / VEC_FLOATS;

    inputs = (__m256 *)prev_layer->outputs;
    outputs = (__m256 *)layer->outputs;
    answers = (__m256 *)mlp->answers;
    deltas = (__m256 *)layer->deltas;

    for (k_vec = 0; k_vec < n_ovecs; ++k_vec)
    {
        // derivative sigmoid activation function
        aux1 = _mm256_sub_ps(ones, outputs[k_vec]);
        aux1 = _mm256_mul_ps(aux1, outputs[k_vec]);

        // delta
        deltas[k_vec] = _mm256_sub_ps(answers[k_vec], outputs[k_vec]);
        deltas[k_vec] = _mm256_mul_ps(deltas[k_vec], aux1);
    }

    for (j_neuron = 0; j_neuron < n_outputs; ++j_neuron)
    {
        weights = (__m256 *)((layer->weights) + j_neuron * wvec_size);
        aux1 = _mm256_set1_ps(layer->deltas[j_neuron] * rate);
        for (k_vec = 0; k_vec < n_wvecs; ++k_vec)
        {
            weights[k_vec] = _mm256_fmadd_ps(aux1, inputs[k_vec], weights[k_vec]);
        }
    }

    for (i_layer = n_layers - 2; i_layer > 0; --i_layer)
    {
        layer = &(mlp->layers[i_layer]);
        prev_layer = &(mlp->layers[i_layer - 1]);
        next_layer = &(mlp->layers[i_layer + 1]);

        n_inputs = prev_layer->n_outputs;
        n_outputs = layer->n_outputs;
        n_next_outputs = next_layer->n_outputs;

        next_wvec_size = MLP_VALID_VEC_SZ(n_outputs + 1);
        wvec_size = MLP_VALID_VEC_SZ(n_inputs + 1);
        ovec_size = MLP_VALID_VEC_SZ(n_outputs);

        n_wvecs = wvec_size / VEC_FLOATS;
        n_ovecs = ovec_size / VEC_FLOATS;

        inputs = (__m256 *)prev_layer->outputs;
        outputs = (__m256 *)layer->outputs;
        deltas = (__m256 *)layer->deltas;

        for (j_neuron = 0; j_neuron < n_outputs; ++j_neuron)
        {
            mlp_float_t error = 0;
            for (next_neuron = 0; next_neuron < n_next_outputs; ++next_neuron)
            {
                mlp_float_t const *const next_weights = next_layer->weights + next_neuron * next_wvec_size;
                error += next_layer->deltas[next_neuron] * next_weights[j_neuron];
            }
            layer->deltas[j_neuron] = error;
        }

        for (k_vec = 0; k_vec < n_ovecs; ++k_vec)
        {
            // derivative sigmoid activation function
            aux1 = _mm256_sub_ps(ones, outputs[k_vec]);
            aux1 = _mm256_mul_ps(aux1, outputs[k_vec]);

            // delta
            deltas[k_vec] = _mm256_mul_ps(deltas[k_vec], aux1);
        }

        for (j_neuron = 0; j_neuron < n_outputs; ++j_neuron)
        {
            weights = (__m256 *)((layer->weights) + j_neuron * wvec_size);
            aux1 = _mm256_set1_ps(layer->deltas[j_neuron] * rate);
            for (k_vec = 0; k_vec < n_wvecs; ++k_vec)
            {
                weights[k_vec] = _mm256_fmadd_ps(aux1, inputs[k_vec], weights[k_vec]);
            }
        }
    }
}

void mlp_free(struct mlp *mlp)
{
    size_t i, n_layers;
    n_layers = mlp->n_layers;
    free(mlp->layers[0].outputs);
    for (i = 1; i < n_layers; ++i)
    {
        free(mlp->layers[i].weights);
        free(mlp->layers[i].outputs);
        free(mlp->layers[i].deltas);
    }
    free(mlp->layers);
    free(mlp->answers);
}