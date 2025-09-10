#ifndef CONV2D_H
#define CONV2D_H

void conv2d_serial(float **f, int H, int W, float **g, int kH, int kW, float **out);
void conv2d_parallel(float **f, int H, int W, float **g, int kH, int kW, float **out);

float **alloc_2d(int H, int W);
void free_2d(float **arr, int H);
void pad_input(float **f, int H, int W, float **fp, int padH, int padW);
void pad_input_asymmetric(float **f, int H, int W, float **fp, int padTop, int padBottom, int padLeft, int padRight);

void read_matrix(const char *filename, float ***arr, int *H, int *W);
void write_matrix(const char *filename, float **arr, int H, int W);
void generate_random_matrix(float ***arr, int H, int W);

#endif