#ifndef CONV2D_H
#define CONV2D_H
// This is a header file for 2D convolution operations with both serial and parallel implementations

// Convolution function for single threaded convolution
void conv2d_serial(float **f, int H, int W, float **g, int kH, int kW, float **out);

// Convolution function for multi-threaded convolution
void conv2d_parallel(float **f, int H, int W, float **g, int kH, int kW, float **out);

// Memory management
float **alloc_2d(int H, int W);      // Allocate 2D float array
void free_2d(float **arr, int H);    // Free 2D float array

// Padding functions if there is symmetric padding needed
void pad_input(float **f, int H, int W, float **fp, int padH, int padW); 

// Padding function if there is asymmetric padding needed
void pad_input_asymmetric(float **f, int H, int W, float **fp, int padTop, int padBottom, int padLeft, int padRight);  

// File I/O operations functions
void read_matrix(const char *filename, float ***arr, int *H, int *W);     // Function for reading matrix from file
void write_matrix(const char *filename, float **arr, int H, int W);       // Function for writing matrix to file
void generate_random_matrix(float ***arr, int H, int W);                  // Function for generating random matrix

#endif 
