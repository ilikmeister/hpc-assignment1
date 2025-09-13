# Compiler and flags
CC = gcc
CFLAGS = -Wall -std=c99 -O2 -fopenmp    # Standard OpenMP on GCC
LIBS = -lm                              # Link math library

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
