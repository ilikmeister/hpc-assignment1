// This code was created for Assignment 1 - CITS5507
// Authours:
// Sarthak Saini - 24110857
// Iliyas Akhmet - 24038357

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <getopt.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "conv2d.h"


// Displays program usage when wrong arguments are provided.
void usage() {
    printf("Usage:\n");
    printf("./conv_test -f infile -g kernel -o outfile\n");
    printf("./conv_test -H height -W width -kH kernel_height -kW kernel_width [-f infile] [-g kernel] [-o outfile]\n");
}

// This main program handles command-line arguments, matrix operations, and convolution execution
int main(int argc, char *argv[]) {
    // This will hold input, kernel, and output matrices
    float **f = NULL, **g = NULL, **out = NULL;

    int H = 0, W = 0;      // Input matrix height and width
    int kH = 0, kW = 0;    // Kernel height and width

    // File paths for input/output (optional)
    char *f_in = NULL, *g_in = NULL, *out_file = NULL;

    // Flag to indicate random generation mode
    int gen = 0;

    // Seed random number generator for reproducible results during development
    srand(time(NULL));

    // Assignment examples: -H 1000 -W 1000 -kH 3 -kW 3 -f f.txt -g g.txt -o o.txt
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            f_in = argv[++i];
        } else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
            g_in = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            out_file = argv[++i];
        } else if (strcmp(argv[i], "-H") == 0 && i + 1 < argc) {
            H = atoi(argv[++i]); gen = 1;
        } else if (strcmp(argv[i], "-W") == 0 && i + 1 < argc) {
            W = atoi(argv[++i]); gen = 1;
        } else if (strcmp(argv[i], "-kH") == 0 && i + 1 < argc) {
            kH = atoi(argv[++i]); gen = 1;
        } else if (strcmp(argv[i], "-kW") == 0 && i + 1 < argc) {
            kW = atoi(argv[++i]); gen = 1;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            usage();
            return 1;
        }
    }

    if (gen && H > 0 && W > 0 && kH > 0 && kW > 0) {
        // RANDOM GENERATION MODE: Create matrices with random values
        printf("Generating random %dx%d input matrix and %dx%d kernel...\n", H, W, kH, kW);
        generate_random_matrix(&f, H, W);      // Create random input matrix
        generate_random_matrix(&g, kH, kW);    // Create random kernel matrix

        // Optionally save generated matrices to files (if filenames provided)
        if (f_in) write_matrix(f_in, f, H, W);
        if (g_in) write_matrix(g_in, g, kH, kW);
    } else if (f_in && g_in) {

        // FILE INPUT MODE: Read matrices from specified files
        printf("Reading matrices from files: %s and %s...\n", f_in, g_in);
        read_matrix(f_in, &f, &H, &W);         // Read input matrix and get dimensions
        read_matrix(g_in, &g, &kH, &kW);       // Read kernel matrix and get dimensions
    } else {

        // ERROR: Invalid arguments - need either files or dimensions
        printf("Error: Must provide either input files OR matrix dimensions\n");
        usage();
        return 1;
    }

    // Allocate memory for output matrix (same size as input)
    out = alloc_2d(H, W);
    printf("Performing %dx%d convolution with %dx%d kernel...\n", H, W, kH, kW);

    // Report OpenMP configuration
    #ifdef _OPENMP
    printf("OpenMP enabled with %d threads available\n", omp_get_max_threads());
    #else
    printf("OpenMP not available - using serial implementation\n");
    #endif

    // Execute parallel convolution with timing
    double start = omp_get_wtime();
    conv2d_parallel(f, H, W, g, kH, kW, out);
    double elapsed = (double)(omp_get_wtime() - start);

    // Execute serial convolution for comparison
    float **out_serial = alloc_2d(H, W);
    double start_2 = omp_get_wtime();
    conv2d_serial(f, H, W, g, kH, kW, out_serial);
    double elapsed_2 = (double)(omp_get_wtime() - start_2);

    // Calculate and report performance metrics
    printf("Performance Metrics:\n");
    printf("Parameters used: H=%d, W=%d, kH=%d, kW=%d\n", H, W, kH, kW);
    printf("Parallel convolution time: %.6fs\n", elapsed);
    printf("Serial convolution time:   %.6fs\n", elapsed_2);
    if (elapsed_2 > 0) {
        double speedup = elapsed_2 / elapsed;
        printf("Speedup: %.2fx\n", speedup);
        if (speedup >= 1.0) {
            printf("Efficiency: %.1f%%\n", speedup / omp_get_max_threads() * 100);
        } else {
            printf("Efficiency: No speedup achieved (%.1fx slowdown)\n", 1.0 / speedup);
        }
    }

    // Clean up serial result matrix
    free_2d(out_serial, H);


    // This section handles output and cleanup:

    // Save output matrix to file (if filename provided)
    if (out_file) {
        write_matrix(out_file, out, H, W);
        printf("Output saved to: %s\n", out_file);
    }

    // Clean up all allocated memory
    free_2d(f, H);
    free_2d(g, kH);
    free_2d(out, H);

    printf("convolution Done\n\n");
    return 0;
}
