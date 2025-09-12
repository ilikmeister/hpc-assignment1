# Makefile for 2D Convolution Assignment

# Compiler and flags
CC = clang
CFLAGS = -Wall -std=c99 -O2 -Xpreprocessor -fopenmp -I/opt/homebrew/Cellar/libomp/21.1.0/include  # Include OpenMP support for macOS
LIBS = -L/opt/homebrew/Cellar/libomp/21.1.0/lib -lomp -lm  # Link OpenMP and math libraries

# Target executable and source files
TARGET = conv_test
SOURCES = main.c conv2d.c
OBJECTS = main.o conv2d.o

# Default target - build the executable
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

# Compile source files to object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Remove build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)
