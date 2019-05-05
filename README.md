# NQueens Parallel Computing
A C/C++ program to solve the N-Queens problem using parallel computing

## Requirements
* MPI enabled computing cluster
* g++ compiler

## Compiling
* Compile 'Main.cpp' program
  * `mpiCC Main.cpp -o gn`

## Executing
* Run program on 4 slots (1 node)
* `mpirun --hostfile /etc/mpi_nodes --host [your_MPI_host] gn` 
