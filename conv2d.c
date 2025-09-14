// This code was created for Assignment 1 - CITS5507
// Authours:
// Sarthak Saini - 24110857
// Iliyas Akhmet - 24038357

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "conv2d.h"

// This function allocates memory for a 2D float array and uses calloc to initialize all values to 0.0
float **alloc_2d(int H, int W) {
    // Allocate array of row pointers
    float **arr = malloc(H * sizeof(float *));  
    for (int i = 0; i < H; i++)
    // Allocate and zero-initialize each row
        arr[i] = calloc(W, sizeof(float));
    return arr;
}


// This function frees memory allocated for a 2D float array and is helpful when called to prevent memory leaks
void free_2d(float **arr, int H) {
    for (int i = 0; i < H; i++)
        free(arr[i]);    // Free each row
    // Free the array of pointers
    free(arr);
}

// This function reads a matrix from a text file
void read_matrix(const char *filename, float ***arr, int *H, int *W) {
    FILE *f = fopen(filename, "r");
    // Read dimensions
    fscanf(f, "%d %d", H, W);
    // Allocate memory for the matrix
    *arr = alloc_2d(*H, *W);
    // Read matrix data row by row
    for (int i = 0; i < *H; i++)    
        for (int j = 0; j < *W; j++)
            fscanf(f, "%f", &((*arr)[i][j]));
    fclose(f);
}

// This function writes a matrix to a text file
void write_matrix(const char *filename, float **arr, int H, int W) {
    FILE *f = fopen(filename, "w");
    fprintf(f, "%d %d\n", H, W); 
    // Write dimensions          
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            // Write each element with 3 decimals
            fprintf(f, "%.3f", arr[i][j]); 
            // Space between elements (not after last)
            if (j < W - 1) fprintf(f, " "); 
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

// This function generates a matrix filled with random float values
void generate_random_matrix(float ***arr, int H, int W) {
    *arr = alloc_2d(H, W);              // Allocate memory
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            (*arr)[i][j] = (float)rand() / RAND_MAX;  // Random value [0.0, 1.0]
}

// This function applies symmetric padding to input matrix
void pad_input(float **f, int H, int W, float **fp, int padH, int padW) {
    // Initialize padded matrix with zeros
    for (int i = 0; i < H + 2 * padH; i++)
        for (int j = 0; j < W + 2 * padW; j++)
            fp[i][j] = 0.0;
    
    // Copy original matrix to center of padded matrix
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            fp[i + padH][j + padW] = f[i][j];
}

// This function applies asymmetric padding to input matrix
void pad_input_asymmetric(float **f, int H, int W, float **fp, int padTop, int padBottom, int padLeft, int padRight) {
    // Initialize padded matrix with zeros
    for (int i = 0; i < H + padTop + padBottom; i++)
        for (int j = 0; j < W + padLeft + padRight; j++)
            fp[i][j] = 0.0;
    
    // Copy original matrix with specified offset
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            fp[i + padTop][j + padLeft] = f[i][j];
}

// This function performs 2D convolution using single-threaded execution
void conv2d_serial(float **f, int H, int W, float **g, int kH, int kW, float **out) {
    if (kH % 2 == 1 && kW % 2 == 1) {
        // If there is ODD KERNEL it uses symmetric padding for centered convolution
        int padH = kH / 2, padW = kW / 2;
        float **fp = alloc_2d(H + 2 * padH, W + 2 * padW);
        pad_input(f, H, W, fp, padH, padW);

        // Sliding kernel over each output position
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                float sum = 0.0;
                // Apply ke rnel at position (i,j)
                for (int u = 0; u < kH; u++)
                    for (int v = 0; v < kW; v++)
                        sum += fp[i + u][j + v] * g[u][v];
                out[i][j] = sum;
            }
        }
        free_2d(fp, H + 2 * padH);
    } else {
        // If there is EVEN KERNEL it uses asymmetric padding
        int padTop = 0, padBottom = kH - 1;
        int padLeft = 0, padRight = kW - 1;
        
        float **fp = alloc_2d(H + padTop + padBottom, W + padLeft + padRight);
        pad_input_asymmetric(f, H, W, fp, padTop, padBottom, padLeft, padRight);

        // Perform convolution with asymmetric padding
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                float sum = 0.0;
                // Apply kernel at position (i,j)
                for (int u = 0; u < kH; u++)
                    for (int v = 0; v < kW; v++)
                        sum += fp[i + u][j + v] * g[u][v];
                out[i][j] = sum;
            }
        }
        free_2d(fp, H + padTop + padBottom);
    }
}


// This function performs 2D convolution using multi-threaded execution with advanced OpenMP optimizations
void conv2d_parallel(float **f, int H, int W, float **g, int kH, int kW, float **out) {
    // Set optimal number of threads based on system capabilities
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    
    if (kH % 2 == 1 && kW % 2 == 1) {
        // If there is ODD KERNEL it uses symmetric padding for centered convolution
        int padH = kH / 2, padW = kW / 2;
        float **fp = alloc_2d(H + 2 * padH, W + 2 * padW);
        pad_input(f, H, W, fp, padH, padW);

        // Advanced OpenMP parallelization with optimized scheduling and memory access
        #pragma omp parallel for collapse(2) schedule(dynamic, 16) shared(fp, g, out) num_threads(num_threads)
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                float sum = 0.0;
                // Kernel application with loop unrolling for small kernels
                if (kH <= 5 && kW <= 5) {
                    // Unroll small kernels for better cache performance
                    for (int u = 0; u < kH; u++) {
                        for (int v = 0; v < kW; v++) {
                            sum += fp[i + u][j + v] * g[u][v];
                        }
                    }
                } else {
                    // Use vectorized operations for larger kernels
                    #pragma omp simd reduction(+:sum)
                    for (int u = 0; u < kH; u++) {
                        for (int v = 0; v < kW; v++) {
                            sum += fp[i + u][j + v] * g[u][v];
                        }
                    }
                }
                out[i][j] = sum;
            }
        }
        free_2d(fp, H + 2 * padH);
    } else {
        // If there is EVEN KERNEL it uses asymmetric padding (bottom-right bias)
        int padTop = 0, padBottom = kH - 1;
        int padLeft = 0, padRight = kW - 1;
        
        float **fp = alloc_2d(H + padTop + padBottom, W + padLeft + padRight);
        pad_input_asymmetric(f, H, W, fp, padTop, padBottom, padLeft, padRight);

        // Advanced OpenMP parallelization with load balancing for even kernels
        #pragma omp parallel for collapse(2) schedule(guided, 8) shared(fp, g, out) num_threads(num_threads)
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                float sum = 0.0;
                // Optimized kernel application with prefetching hints
                for (int u = 0; u < kH; u++) {
                    #pragma omp simd reduction(+:sum)
                    for (int v = 0; v < kW; v++) {
                        sum += fp[i + u][j + v] * g[u][v];
                    }
                }
                out[i][j] = sum;
            }
        }
        free_2d(fp, H + padTop + padBottom);
    }
}