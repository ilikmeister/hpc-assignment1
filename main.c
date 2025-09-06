#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <getopt.h>
#include "conv2d.h"

void usage() {
    printf("Usage:\n");
    printf("./conv_test [--f infile] [--g kernel] [--o outfile]\n");
    printf("./conv_test [--height H] [--width W] [--kH kh] [--kW kw] [--f infile] [--g kernel] [--o outfile]\n");
}

int main(int argc, char *argv[]) {
    float **f = NULL, **g = NULL, **out = NULL;
    int H = 0, W = 0, kH = 0, kW = 0;
    char *f_in = NULL, *g_in = NULL, *out_file = NULL;
    int gen = 0;
    srand(time(NULL));

    static struct option long_options[] = {
        {"f",      required_argument, 0, 'f'},
        {"g",      required_argument, 0, 'g'},
        {"o",      required_argument, 0, 'o'},
        {"height", required_argument, 0, 'H'},
        {"width",  required_argument, 0, 'W'},
        {"kH",     required_argument, 0, 'a'}, // unique short code for kernel height
        {"kW",     required_argument, 0, 'b'}, // unique short code for kernel width
        {0, 0, 0, 0}
    };

    int option_index = 0;
    int opt;
    while ((opt = getopt_long(argc, argv, "f:g:o:H:W:a:b:", long_options, &option_index)) != -1) {
        switch(opt) {
            case 'f': f_in = optarg; break;
            case 'g': g_in = optarg; break;
            case 'o': out_file = optarg; break;
            case 'H': H = atoi(optarg); gen = 1; break;
            case 'W': W = atoi(optarg); gen = 1; break;
            case 'a': kH = atoi(optarg); gen = 1; break;
            case 'b': kW = atoi(optarg); gen = 1; break;
            default: usage(); return 1;
        }
    }

    if (f_in && g_in) {
        read_matrix(f_in, &f, &H, &W);
        read_matrix(g_in, &g, &kH, &kW);
    } else if (gen) {
        generate_random_matrix(&f, H, W);
        generate_random_matrix(&g, kH, kW);
        if (f_in) write_matrix(f_in, f, H, W);
        if (g_in) write_matrix(g_in, g, kH, kW);
    } else {
        usage();
        return 1;
    }

    out = alloc_2d(H, W);

    clock_t start = clock();
    conv2d_parallel(f, H, W, g, kH, kW, out);
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("Parallel convolution time: %.6fs\n", elapsed);

    float **out_serial = alloc_2d(H, W);
    conv2d_serial(f, H, W, g, kH, kW, out_serial);

    if (out_file)
        write_matrix(out_file, out, H, W);

    free_2d(f, H);
    free_2d(g, kH);
    free_2d(out, H);
    free_2d(out_serial, H);
    return 0;
}