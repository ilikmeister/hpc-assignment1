#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "conv2d.h"

float **alloc_2d(int H, int W) {
    float **arr = malloc(H * sizeof(float *));
    for (int i = 0; i < H; i++)
        arr[i] = calloc(W, sizeof(float));
    return arr;
}

void free_2d(float **arr, int H) {
    for (int i = 0; i < H; i++)
        free(arr[i]);
    free(arr);
}

void read_matrix(const char *filename, float ***arr, int *H, int *W) {
    FILE *f = fopen(filename, "r");
    fscanf(f, "%d %d", H, W);
    *arr = alloc_2d(*H, *W);
    for (int i = 0; i < *H; i++)
        for (int j = 0; j < *W; j++)
            fscanf(f, "%f", &((*arr)[i][j]));
    fclose(f);
}

void write_matrix(const char *filename, float **arr, int H, int W) {
    FILE *f = fopen(filename, "w");
    fprintf(f, "%d %d\n", H, W);
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            fprintf(f, "%.3f", arr[i][j]);
            if (j < W - 1) fprintf(f, " ");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void generate_random_matrix(float ***arr, int H, int W) {
    *arr = alloc_2d(H, W);
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            (*arr)[i][j] = (float)rand() / RAND_MAX;
}

void pad_input(float **f, int H, int W, float **fp, int padH, int padW) {
    for (int i = 0; i < H + 2 * padH; i++)
        for (int j = 0; j < W + 2 * padW; j++)
            fp[i][j] = 0.0;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            fp[i + padH][j + padW] = f[i][j];
}

void pad_input_asymmetric(float **f, int H, int W, float **fp, int padTop, int padBottom, int padLeft, int padRight) {
    for (int i = 0; i < H + padTop + padBottom; i++)
        for (int j = 0; j < W + padLeft + padRight; j++)
            fp[i][j] = 0.0;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            fp[i + padTop][j + padLeft] = f[i][j];
}

void conv2d_serial(float **f, int H, int W, float **g, int kH, int kW, float **out) {
    if (kH % 2 == 1 && kW % 2 == 1) {
        // Odd kernel size: use symmetric padding
        int padH = kH / 2, padW = kW / 2;
        float **fp = alloc_2d(H + 2 * padH, W + 2 * padW);
        pad_input(f, H, W, fp, padH, padW);

        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++) {
                float sum = 0.0;
                for (int u = 0; u < kH; u++)
                    for (int v = 0; v < kW; v++)
                        sum += fp[i + u][j + v] * g[u][v];
                out[i][j] = sum;
            }
        free_2d(fp, H + 2 * padH);
    } else {
        // Even kernel size: use asymmetric padding
        int padTop = 0, padBottom = kH - 1;
        int padLeft = 0, padRight = kW - 1;
        
        float **fp = alloc_2d(H + padTop + padBottom, W + padLeft + padRight);
        pad_input_asymmetric(f, H, W, fp, padTop, padBottom, padLeft, padRight);

        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++) {
                float sum = 0.0;
                for (int u = 0; u < kH; u++)
                    for (int v = 0; v < kW; v++)
                        sum += fp[i + u][j + v] * g[u][v];
                out[i][j] = sum;
            }
        free_2d(fp, H + padTop + padBottom);
    }
}

void conv2d_parallel(float **f, int H, int W, float **g, int kH, int kW, float **out) {
    if (kH % 2 == 1 && kW % 2 == 1) {
        // Odd kernel size: use symmetric padding
        int padH = kH / 2, padW = kW / 2;
        float **fp = alloc_2d(H + 2 * padH, W + 2 * padW);
        pad_input(f, H, W, fp, padH, padW);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++) {
                float sum = 0.0;
                for (int u = 0; u < kH; u++)
                    for (int v = 0; v < kW; v++)
                        sum += fp[i + u][j + v] * g[u][v];
                out[i][j] = sum;
            }
        free_2d(fp, H + 2 * padH);
    } else {
        // Even kernel size: use asymmetric padding
        int padTop = 0, padBottom = kH - 1;
        int padLeft = 0, padRight = kW - 1;
        
        float **fp = alloc_2d(H + padTop + padBottom, W + padLeft + padRight);
        pad_input_asymmetric(f, H, W, fp, padTop, padBottom, padLeft, padRight);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++) {
                float sum = 0.0;
                for (int u = 0; u < kH; u++)
                    for (int v = 0; v < kW; v++)
                        sum += fp[i + u][j + v] * g[u][v];
                out[i][j] = sum;
            }
        free_2d(fp, H + padTop + padBottom);
    }
}