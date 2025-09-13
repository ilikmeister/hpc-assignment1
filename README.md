# CITS5507 - Assignment 1

## Description

This project implements fast, parallel 2D convolution in C using OpenMP.  
It supports two modes:

1. **File Input Mode:** Reads matrices from files and performs convolution.
2. **Random Generation Mode:** Generates random matrices and performs convolution.

## Files

- `Makefile` – For building the project on Kaya.
- `main.c` – Main driver program.
- `conv2d.c`, `conv2d.h` – convolution implementation.
- `test-files/` – Example/test input files.
- `script.slurm` – Example SLURM batch script for Kaya.

## Compilation

On Kaya:

```bash
make
```

## Usage

### 1. **File Input Mode**
Read matrices from files and perform convolution.  
Files must follow the text file specification: first line is dimensions, followed by rows of space-separated floats.

```bash
./conv_test -f input.txt -g kernel.txt -o output.txt
```

### 2. **Random Generation Mode**
Generate random input and kernel, optionally saving them to files:
```bash
./conv_test -H <height> -W <width> -kH <kernel_height> -kW <kernel_width> \
[-f input.txt] [-g kernel.txt] [-o output.txt]
```
Example:
```bash
./conv_test -H 1000 -W 1000 -kH 3 -kW 3 -f input.txt -g kernel.txt -o output.txt
```

## SLURM Batch Submission (Kaya)

Example script (``script.slurm``):

```bash
#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=1:00:00
#SBATCH --partition=cits3402

./conv_test --H 1000 --W 2000 --kH 3 --kW 5
```

Submit with:
```bash
sbatch script.slurm
```

You should receive the message, where N is the number of the job:

```bash
Submitted batch job N
```

See the output with:

```bash
cat slurm-N.out
```

## Cleaning Up

To remove compiled files:
```bash
make clean
```

**Authors:**  

- Sarthak Saini - 24110857

- Iliyas Akhmet - 24038357