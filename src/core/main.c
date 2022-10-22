#include "core/main.h"

int main(int argc, char **argv)
{
    srand(rand_seed);
    readIris();
    printf("ACA\n");
    mlp_init(&mlp, layers_size, n_layers);
    printf("ACA\n");

    for (int i = 0; i < rows; ++i)
        order[i] = i;
    shuffle(order, rows);

    for (int i = 0; i < epochs; i++)
    {
        shuffle(order, train_rows);
        for (int j = 0; j < train_rows; j++)
        {
            int row = order[j];
            mlp_load_input(&mlp, feat + row * cols);
            float const *prediction = mlp_forward(&mlp);
            mlp_load_answer(&mlp, label + row * out_cols);
            mlp_backprog(&mlp, learning_rate);
        }
    }

    int correct = 0;
    int incorrect = 0;

    for (int i = train_rows; i < rows; ++i)
    {
        int row = order[i];
        printf("ROW: %d\n", row);

        mlp_load_input(&mlp, feat + row * cols);
        float const *prediction = mlp_forward(&mlp);

        if (abs(prediction[0] - label[row * out_cols + 0]) < 0.1 &&
            abs(prediction[1] - label[row * out_cols + 1]) < 0.1 &&
            abs(prediction[2] - label[row * out_cols + 2]) < 0.1)
            correct++;
        else
            incorrect++;
        printf("Predicted: [%f, %f, %f] vs Answer: [ %f, %f, %f ]\n\n", prediction[0],
               prediction[1], prediction[2], label[row * out_cols + 0],
               label[row * out_cols + 1], label[row * out_cols + 2]);
    }

    printf("Total de predicciones correctas: %d\n", correct);
    printf("Total de predicciones incorrectas: %d\n", incorrect);

    return EXIT_SUCCESS;
}

void readIris()
{

    char *dataFileName = "iris.data";

    memset(feat, 0, rows * cols * sizeof(float));
    memset(label, 0, rows * out_cols * sizeof(float));

    printf("Obsns size is %d and feat size is %d.\n", rows, cols);

    FILE *fpDataFile = fopen(dataFileName, "r");

    if (!fpDataFile)
    {
        printf("Missing input file: %s\n", dataFileName);
        exit(1);
    }

    int index = 0;
    char line[1024];
    char flowerType[20];
    float l;

    while (fgets(line, 1024, fpDataFile))
    {
        if (5 == sscanf(line, "%f,%f,%f,%f,%f[^\n]", &feat[index * cols + 0],
                        &feat[index * cols + 1], &feat[index * cols + 2],
                        &feat[index * cols + 3], &l))
        {
            printf("%f,%f,%f,%f,%f\n", feat[index * cols + 0], feat[index * cols + 1],
                   feat[index * cols + 2], feat[index * cols + 3], l);
            label[index * out_cols + ((int)l)] = 1;
            index++;
        }
    }
    fclose(fpDataFile);
}

void shuffle(int *array, int n)
{
    if (n > 1)
    {
        int i;
        for (i = 0; i < n - 1; i++)
        {
            int j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}