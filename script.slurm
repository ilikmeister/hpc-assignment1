#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=1:00:00
#SBATCH --partition=cits3402

./conv_test --H 1000 --W 2000 --kH 3 --kW 5
./conv_test --H 1000 --W 2000 --kH 7 --kW 10
./conv_test --H 1000 --W 2000 --kH 10 --kW 12
./conv_test --H 1000 --W 2000 --kH 15 --kW 19