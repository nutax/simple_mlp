#ifndef _MAIN_H_
#define _MAIN_H_

#include "hello/hello.h"
#include "mlp/mlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv);
void readIris();
void shuffle(int *array, int n);

#define epochs 10000
#define learning_rate 0.1
#define rand_seed 0
#define cols 4
#define out_cols 3
#define rows 150
#define train_rows 105
#define n_layers 3

int order[rows];
float feat[rows * cols];
float label[rows * out_cols];

size_t layers_size[n_layers] = {4, 5, 3};

struct mlp mlp;

#endif