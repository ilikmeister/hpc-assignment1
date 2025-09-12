/* 
 * This is the main driver program for testing 2D convolution implementations.
 * It supports two operation modes:
 * 
 * 1. FILE INPUT MODE: Read matrices from files and perform convolution
 *    Usage: ./conv_test -f input.txt -g kernel.txt -o output.txt
 * 
 * 2. RANDOM GENERATION MODE: Generate random matrices and perform convolution  
 *    Usage: ./conv_test -H 1000 -W 1000 -kH 3 -kW 3 [-f f.txt] [-g g.txt] [-o o.txt]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <getopt.h>
#include "conv2d.h"

/**
 * Displays program usage information
 * Shows both supported modes exactly as specified in assignment
 */
void usage() {
    printf("Usage:\n");
    printf("./conv_test -f infile -g kernel -o outfile\n");
    printf("./conv_test -H height -W width -kH kernel_height -kW kernel_width [-f infile] [-g kernel] [-o outfile]\n");
}

/**
 * This main program handles command-line arguments, matrix operations, and convolution execution
 */
int main(int argc, char *argv[]) {
    // This will hold input, kernel, and output matrices
    float **f = NULL, **g = NULL, **out = NULL;
    
    // Matrix dimensions
    int H = 0, W = 0;      // Input matrix height and width
    int kH = 0, kW = 0;    // Kernel height and width
    
    // File paths for input/output (optional)
    char *f_in = NULL, *g_in = NULL, *out_file = NULL;
    
    // Flag to indicate random generation mode
    int gen = 0;
    
    // Seed random number generator for reproducible results during development
    srand(time(NULL));

    // Command-line option definitions using getopt_long
    // Using SHORT OPTIONS ONLY as required by assignment
    static struct option long_options[] = {
        {0, 0, 0, 0}  // End marker - no long options needed
    };

    // Parse command-line arguments manually to support -kH and -kW as shown in assignment
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

    /*
     * This code implements matrix initialization with two modes:
     1. Random generation mode: Create matrices with specified dimensions
     2. File input mode: Read matrices from provided files
     */
    
    if (gen && H > 0 && W > 0 && kH > 0 && kW > 0) {
        // RANDOM GENERATION MODE: Create matrices with random values
        printf("Generating random %dx%d input matrix and %dx%d kernel...\n", H, W, kH, kW);
        generate_random_matrix(&f, H, W);      // Create random input matrix
        generate_random_matrix(&g, kH, kW);    // Create random kernel matrix
        
        // Optionally save generated matrices to files (if filenames provided)
        if (f_in) write_matrix(f_in, f, H, W);
        if (g_in) write_matrix(g_in, g, kH, kW);
    } else if (f_in && g_in) {
        // FILE INPUT MODE: Read matrices from files
        printf("Reading matrices from files: %s and %s...\n", f_in, g_in);
        read_matrix(f_in, &f, &H, &W);         // Read input matrix and get dimensions
        read_matrix(g_in, &g, &kH, &kW);       // Read kernel matrix and get dimensions
    } else {
        // ERROR: Invalid arguments - need either files or dimensions
        printf("Error: Must provide either input files OR matrix dimensions\n");
        usage();
        return 1;
    }

    /*
    This section performs the core convolution operation:
     1. Allocates memory for the output matrix
     2. Executes the parallel 2D convolution function
     3. Measures and prints the execution time 
     */
    
    // Allocate memory for output matrix (same size as input)
    out = alloc_2d(H, W);
    printf("Performing %dx%d convolution with %dx%d kernel...\n", H, W, kH, kW);

    // Execute parallel convolution with timing
    clock_t start = clock();
    conv2d_parallel(f, H, W, g, kH, kW, out);
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    // running serial second
    clock_t start_2 = clock();
    conv2d_serial(f, H, W, g, kH, kW, out);
    double elapsed_2 = (double)(clock() - start_2) / CLOCKS_PER_SEC;
    // printing results
    printf("Parameters used: H=%d, W=%d, kH=%d, kW=%d\n", H,W,kH,kW);
    printf("Parallel convolution time: %.6fs\n", elapsed);

    /* 
     * This section handles output and cleanup
     */
    
    // Save output matrix to file (if filename provided)
    if (out_file) {
        write_matrix(out_file, out, H, W);
        printf("Output saved to: %s\n", out_file);
    }

    // Clean up all allocated memory to prevent leaks
    free_2d(f, H);      // Free input matrix
    free_2d(g, kH);     // Free kernel matrix  
    free_2d(out, H);    // Free output matrix
    
    printf("Convolution completed successfully!\n");
    return 0;
}