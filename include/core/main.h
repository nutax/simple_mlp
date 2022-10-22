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

int const epochs = 1000000;
float const learning_rate = 0.1;
int const rand_seed = 0;
int const cols = 4;
int const out_cols = 3;
int const rows = 150;
int const train_rows = 105;

int order[rows];
float feat[rows * cols];
float label[rows * out_cols];

size_t const n_layers = 4;
size_t const layers_size[] = {4, 5, 4, 3};

struct mlp mlp;

#endif