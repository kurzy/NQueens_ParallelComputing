////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Filename: Main.cpp
// Project: NQueens Problem using Distributed Computing. 3801ICT Scientific Computing Assignment.
// Author: Jack Kearsley, April 2018.
// Description: Genetic search for solving the n-Queens problem.
// Uses the MPI library for distributed computing.
// Runs in serial mode if executed on a single processor.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//===================== Includes =======================
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>
// If compiling in Visual Studio, use the included 'mpi.h' file.
#if defined(_MSC_VER)
#include "libs/MPI/Include/mpi.h"
// Otherwise, if compiling on the 'shrek' cluster, 'mpi.h' is in a different location.
#else
#include <mpi.h>
#endif

//======================= Defines and Global Variables ============================
// The number of queens, and the dimensions of the square chess board.
#define NQUEENS 1000
// The number of boards in the population.
#define NUM_BOARDS 150
// Boards in the population that are ranked above POPULATION_THRESHOLD will be selected for reproduction.
int POPULATION_THRESHOLD = (NUM_BOARDS / 2);
// Chance of an offspring being mutated out of 100 (percentage).
int GN_MUTATION_RATE = 15;
// Maximum number of reproductions per generation.
int MAX_REPRODUCTIONS = (NUM_BOARDS / 3) - 2;

// 'avg_time' is used to measure the average time taken for multiple searches.
double avg_time = 0.0;
// If 'print_debug' is true, when each solution is found, the board and time elapsed are printed.
bool print_debug = 0;
// If 'print_master_progress' is true, the master process prints the best cost found so far, throughout the search.
bool print_master_progress = 0;

// MPI Tags & Variables
// Tells slave processes to shutdown.
#define MPITAG_SHUTDOWN 500

// 'BOARDS_PER_SLAVE' defines how many boards are sent out to each slave process by the master.
#define BOARDS_PER_SLAVE 2
// 'SLAVE_ITERATIONS_PER_CYCLE' defines how many attempts slave processes have at creating a fitter board, each send/receive cycle.
// Each slave sends back the best board it can come up with.
#define SLAVE_ITERATIONS_PER_CYCLE 8
// MUTATIONS_PER_SLAVE_ITERATION: How many mutations to try out on new offspring. This is performed by the slave processes.
#define MUTATIONS_PER_SLAVE_ITERATION 8

//=============================== Typedefs and Structs ===========================================
// 'BOARD' represents the chess board.
// A queen's column is represented by its index in the 'q' array.
// A queen's row is represented by an integer in the 'q' array, for a given column (array index).
// When a board is created the queens are placed in separate columns so a 'column' integer is redundant.
// 'BOARD' also stores its total cost, or how many possible conflicts/attacks can occur between queens.
struct BOARD {
	int q[NQUEENS];
	int totalCost;
};

//====================== Function Prototypes ============================
int cmpBoardCosts(const void * a, const void * b);
BOARD geneticSearchSerial();
void geneticSearchParallel();
void computeCosts(BOARD * board);
void initialise(BOARD * board);
void printBoard(const BOARD * board);
void mutateQueen(BOARD * board);
void crossover(BOARD * p1, BOARD * p2, BOARD * c1, BOARD * c2);
void printParameters(void);

//==================== Function Implementations ======================
// Program entry point.
int main(int argc, char * argv[]) {
	// Initialise MPI.
	MPI_Init(NULL, NULL);
	// Get the number of processes.
	int num_processes;
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

	// If running the program on a single machine, execute the search in serial mode, then exit.
	if (num_processes == 1) {
		MPI_Finalize();
		printf("Only 1 process detected, running the genetic search in serial mode...\n");
		BOARD serial_solution = geneticSearchSerial();
		return 0;
	}
	// Otherwise, if multiple processes were detected, run in parallel mode.
	// Get the rank of the process.
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// If Master (0) process, call the geneticSearchParallel() function.
	if (world_rank == 0) {
		printf("Master: %d processes detected, running the genetic search in parallel mode.\n", num_processes);
	}
	geneticSearchParallel();
	MPI_Finalize();
	return 0;
}

// Execute the genetic search on a serial machine (one computer).
// Returns the solution 'BOARD'.
BOARD geneticSearchSerial() {
	// Keep track of the time elapsed.
	srand((unsigned int)time(NULL));
	clock_t clk_start = clock();
	// The population of chess boards.
	BOARD population[NUM_BOARDS];
	// Fill each board with queens.
	// initialise() also calls computeCost() to calculate each BOARD's totalCost attribute.
	for (int i = 0; i < NUM_BOARDS; i++) {
		initialise(&population[i]);
	}
	// If a solution is found, exits the while-loop.
	bool solution_found = false;
	// Keep searching until a solution is found.
	while (!solution_found) {
		// Sort the boards in ascending order of their totalCost.
		qsort(population, NUM_BOARDS, sizeof(*population), cmpBoardCosts);
		// For a number of reproductions.
		for (int m = 0; m < MAX_REPRODUCTIONS; m++) {
			// Select two parents randomly from the fittest portion of the population.
			// The fittest portion is defined by 'POPULATION_THRESHOLD'.
			BOARD * p1 = &population[rand() % (POPULATION_THRESHOLD)];
			BOARD * p2 = &population[rand() % (POPULATION_THRESHOLD)];
			BOARD childOne, childTwo;

			// Crossover the two parents.
			// Results in two offspring, which are stored in childOne and childTwo.
			crossover(p1, p2, &childOne, &childTwo);

			// Apply random mutations based on a probability (GN_MUTATION_RATE).
			if ((rand() % 100) > GN_MUTATION_RATE) {
				mutateQueen(&childOne);
			}
			if ((rand() % 100) > GN_MUTATION_RATE) {
				mutateQueen(&childTwo);
			}
			// Recompute the new costs for the child boards.
			computeCosts(&childOne);
			computeCosts(&childTwo);

			// If one of the child boards is a solution (cost = 0), then end the search, and print the board.
			if (childOne.totalCost == 0) {
				// Print time elapsed.
				double t_elapsed = (double)(clock() - clk_start) / CLOCKS_PER_SEC;
				avg_time += t_elapsed;
				printBoard(&childOne);
				printf("\n" "Solution found\n");
				printf("Time elapsed: %.3lfs\n", t_elapsed);
				printParameters();
				solution_found = true;
				return childOne;
			}
			if (childTwo.totalCost == 0) {
				// Print time elapsed.
				double t_elapsed = (double)(clock() - clk_start) / CLOCKS_PER_SEC;
				avg_time += t_elapsed;
				printBoard(&childTwo);
				printf("\n" "Solution found\n");
				printf("Time elapsed: %.3lfs\n", t_elapsed);
				printParameters();
				solution_found = true;
				return childTwo;
			}
			// Add the two child boards back into the population.
			// They replace the least fit boards at the bottom-end of the sorted 'population' array.
			population[NUM_BOARDS - (2 * m) - 1] = childOne;
			population[NUM_BOARDS - (2 * m) - 2] = childTwo;
		}///// for(max-number-of-generations) /////

	}///// while(!solution_found) /////

	// Board with a 0 cost is sorted to the front of array (index [0]).
	return population[0];

}///// geneticSearchSerial() /////

// Execute the genetic search in parallel (across multiple computers).
// This function uses the MPI library to send and receive messages.
// Returns the solution BOARD.
void geneticSearchParallel() {
	// Get the world rank of the process.
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	// If the Master (0) process.
	if (world_rank == 0) {
		// Get the total world size.
		int world_size;
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);	
		// Keep track of the time elapsed.
		srand((unsigned int)time(NULL));
		clock_t clk_start = clock();
		// The population of chess boards.
		BOARD population[NUM_BOARDS];
		// Fill each board with queens.
		// initialise() also calls computeCost() to calculate each BOARD's totalCost attribute.
		for (int i = 0; i < NUM_BOARDS; i++) {
			initialise(&population[i]);
		}		
		// Send out some boards to the slave processes.
		// For each slave process (1..world_size).
		for (int i = 1; i < world_size; i++) {
			// Send out 'BOARDS_PER_SLAVE' boards.
			for (int b = 0; b < BOARDS_PER_SLAVE; b++) {
				int b_index = (i + b) % NUM_BOARDS;
				MPI_Send(population[b_index].q, NQUEENS, MPI_INT, i, 0, MPI_COMM_WORLD);
				if (print_debug) printf("Master: Sent a board to slave %d\n", i);
			}
		}

		// Keep track of the best cost so far.
		int best_cost = INT_MAX;
		// Keep looping until solution board is found.
		while (true) {
			// Sort the current population in ascending order of cost.
			qsort(population, NUM_BOARDS, sizeof(*population), cmpBoardCosts);
			// If a solution board exists, exit the loop.
			if (population[0].totalCost == 0) {
				break;
			}
			// Keep track of the best cost so far.
			if (population[0].totalCost < best_cost) {
				best_cost = population[0].totalCost;
				if (print_master_progress) printf("Master: Best board cost so far: \t\t\t\t%d\n", best_cost);
			}
			// Receive a board back from any slave.
			BOARD recv_board;
			MPI_Status status;
			if (print_debug) printf("Master: Waiting to receive a board from a slave\n");
			MPI_Recv(&recv_board.q, NQUEENS, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			if (print_debug) printf("Master: Received a board from S%d\n", status.MPI_SOURCE);
			// Send back a number of boards to the same slave so it can continue working.
			if (print_debug) printf("Master: Sending %d new boards to S%d...\n", BOARDS_PER_SLAVE, status.MPI_SOURCE);
			for (int i = 0; i < BOARDS_PER_SLAVE; i++) {
				MPI_Send(&population[rand() % POPULATION_THRESHOLD].q, NQUEENS, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
			}
			if (print_debug) printf("Master: %d new boards sent to S%d\n", BOARDS_PER_SLAVE, status.MPI_SOURCE);
			// Calculate the cost of the received board.
			computeCosts(&recv_board);
			// If the board has a lesser cost than the worst of the current population, replace the worst-cost board with 'recv_board'. 
			if (recv_board.totalCost < population[NUM_BOARDS - 1].totalCost) {
				population[NUM_BOARDS - 1] = recv_board;
			}
		} ////// while(solution not found) ///////
		
		// Print the solution board, and time elapsed.
		printBoard(&population[0]);
		printf("\n\nSolution found after %.3lfs.\n", NQUEENS, (double)(clock() - clk_start) / CLOCKS_PER_SEC);
		printParameters();
		// Shutdown all of the slave processes.
		// The slave is waiting to send and receive a board, so send each slave a dummy board, with a 'MPITAG_SHUTDOWN' tag.
		for (int i = 1; i < world_size; i++) {
			BOARD * recv_board = (BOARD *)calloc(1, sizeof(BOARD));
			MPI_Status status;
			MPI_Recv(recv_board->q, NQUEENS, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
			MPI_Send(recv_board->q, NQUEENS, MPI_INT, i, MPITAG_SHUTDOWN, MPI_COMM_WORLD);
			free(recv_board);
		}

	}/////// if (master process) ////////

	// Otherwise, if slave process.
	else if (world_rank != 0) {
		// Store the best-fitness board so far.
		BOARD bestBoard;
		bestBoard.totalCost = INT_MAX;
		// Create an array to hold the boards received from the master.
		BOARD slaveBoards[BOARDS_PER_SLAVE];
		// Run in a loop until ordered to shutdown by the master.
		while (true) {
			// Receive the boards from the master.
			for (int i = 0; i < BOARDS_PER_SLAVE; i++) {
				MPI_Status status;
				if (print_debug) printf("S%d: Waiting to receive a board from master.\n", world_rank);
				MPI_Recv(slaveBoards[i].q, NQUEENS, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				if (status.MPI_TAG == MPITAG_SHUTDOWN) {
					if (print_debug) printf("S%d received shutdown tag from master.\n", world_rank);
					return;
				}
				if (print_debug) printf("S%d: Received a board from master\n", world_rank);
				// Compute the received board's cost, and compare it with the current 'bestBoard'.
				computeCosts(&slaveBoards[i]);
				if (slaveBoards[i].totalCost < bestBoard.totalCost) {
					bestBoard = slaveBoards[i];
				}
			}
			// For a maximum number of iterations, try and generate a board of a lesser cost than the one stored in 'bestBoard'.
			// This is achieved through multiple mutation and crossover attempts.
			for (int i = 0; i < SLAVE_ITERATIONS_PER_CYCLE; i++) {
				BOARD childOne, childTwo;
				// For each board in slaveBoards.
				for (int b = 0; b < BOARDS_PER_SLAVE; b++) {
					// Do a crossover operation.
					int ind2 = (b + 1) % BOARDS_PER_SLAVE;
					crossover(&slaveBoards[b], &slaveBoards[ind2], &childOne, &childTwo);
					computeCosts(&childOne);
					computeCosts(&childTwo);
					// If an offspring is fitter than 'bestBoard', it becomes the new 'bestBoard'.
					if (childOne.totalCost < bestBoard.totalCost) {
						bestBoard = childOne;
					}
					if (childTwo.totalCost < bestBoard.totalCost) {
						bestBoard = childTwo;
					}
					// Perform some mutations and see if they improve the board's cost.
					for (int m = 0; m < MUTATIONS_PER_SLAVE_ITERATION; m++) {
						mutateQueen(&childOne);
						mutateQueen(&childTwo);
						computeCosts(&childOne);
						computeCosts(&childTwo);
						// If an offspring is fitter than 'bestBoard', it becomes the new 'bestBoard'.
						if (childOne.totalCost < bestBoard.totalCost) {
							bestBoard = childOne;
						}
						if (childTwo.totalCost < bestBoard.totalCost) {
							bestBoard = childTwo;
						}
					}///// for (number of mutations) ///////

				}////// for (each board in 'slaveBoards') ///////

			}////// for (the number of slave iterations) //////

			// After the slave has performed all of its iterations, it sends back the fittest board it has (bestBoard) to the master.
			if (print_debug) printf("S%d: Sending best board to master...\n", world_rank);
			MPI_Send(&bestBoard.q, NQUEENS, MPI_INT, 0, 0, MPI_COMM_WORLD);
			if (print_debug) printf("S%d: Sent best board to master.\n", world_rank);

		}////// while (true) ///////

	}////// else (slave process) //////

}///// geneticSearchParallel() /////

// A crossover function for making two offspring (c1, c2) from two parents (p1, p2).
// Uses the PMX crossover method (partially matched crossover).
void crossover(BOARD * p1, BOARD * p2, BOARD * c1, BOARD * c2) {
	// Get two random crossover points.
	int pos1 = rand() % NQUEENS;
	int pos2 = rand() % NQUEENS;
	while (pos2 == pos1) {
		pos2 = rand() % NQUEENS;
	}
	// Get the minimum and maximum of the crossover points.
	int min = (pos1 > pos2) ? pos2 : pos1;
	int max = (pos1 > pos2) ? pos1 : pos2;
	// Set uninitialised child values to -1
	for (int i = 0; i < NQUEENS; i++) {
		c1->q[i] = -1;
		c2->q[i] = -1;
	}
	// Child 1 gets a portion from parent 1.
	for (int i = min; i < max; i++) {
		c1->q[i] = p1->q[i];
	}
	// Child 2 gets a portion from parent 2.
	for (int i = min; i < max; i++) {
		c2->q[i] = p2->q[i];
	}
	////////////////////////// Check for duplicates in c1 ////////////////////////////////
	// Find values in p2 that are not already in c1.
	for (int i = min; i < max; i++) {
		bool in_child = false;
		int index = 0;
		for (int j = min; j < max; j++) {
			if (p2->q[i] == c1->q[j]) {
				in_child = true;
			}
			index = i;
		}
		// If we find a value that is not in c1.
		if (!in_child) {
			int orig_index = index;
			bool num_added = false;
			// While looking for a number to add to c1.
			while (!num_added) {
				// Get the number from p1 at the same index.
				int num = p1->q[index];
				int p2ind = 0;
				// Now find the position of that number in p2, store it in 'p2ind'.
				for (int k = 0; k < NQUEENS; k++) {
					if (p2->q[k] == num) {
						p2ind = k;
						break;
					}
				}
				// If p2ind is outside of the min..max range, add the original number (at p2[orig_index]) to c1.
				// And then move onto the next number within min..max from p2 (next 'i' iteration in for-loop).
				if (p2ind < min || p2ind >= max) {
					c1->q[p2ind] = p2->q[orig_index];
					num_added = true;
				}
				// Otherwise, repeat the above steps until an index outside of min..max is found.
				else {
					index = p2ind;
				}
			}
		}//////if(not in child)//////

	}////for(each sqaure in p1 within min..max)//////

	// Set unused squares to p2 values.
	for (int i = 0; i < NQUEENS; i++) {
		if (c1->q[i] == -1) {
			c1->q[i] = p2->q[i];
		}
	}

	////////////////////////// Check for duplicates in c2 ////////////////////////////////
	// Find values in p1 that are not already in c2.
	for (int i = min; i < max; i++) {
		bool in_child = false;
		int index = 0;
		for (int j = min; j < max; j++) {
			if (p1->q[i] == c2->q[j]) {
				in_child = true;
			}
			index = i;
		}
		// If we find a value that is not in c2.
		if (!in_child) {
			int orig_index = index;
			bool num_added = false;
			// While looking for a number to add to c2.
			while (!num_added) {
				// Get the number from p2 at the same index.
				int num = p2->q[index];
				int p1ind = 0;
				// Now find the position of that number in p1, store it in 'p1ind'.
				for (int k = 0; k < NQUEENS; k++) {
					if (p1->q[k] == num) {
						p1ind = k;
						break;
					}
				}
				// If p1ind is outside of the min..max range, add the original number (at p1[orig_index]) to c2.
				// And then move onto the next number within min..max from p1 (next 'i' iteration in for-loop).
				if (p1ind < min || p1ind >= max) {
					c2->q[p1ind] = p1->q[orig_index];
					num_added = true;
				}
				// Otherwise, repeat the above steps until an index outside of min..max is found.
				else {
					index = p1ind;
				}
			}

		}//////if(not in child)//////

	}////for(each sqaure in p1 within min..max)//////

	 // Set unused squares to p1 values.
	for (int i = 0; i < NQUEENS; i++) {
		if (c2->q[i] == -1) {
			c2->q[i] = p1->q[i];
		}
	}

}

// The cost function computes the 'totalCost' attribute for 'board'.
// A board's cost is the total number of conflicts or 'attacks' that can occur between queens on the board.
void computeCosts(BOARD * board) {
	// Create counters for left and right diagonals, and set them all to 0.
	int left_diagonal[(2 * NQUEENS) - 1];
	int right_diagonal[(2 * NQUEENS) - 1];
	for (int i = 0; i < (2 * NQUEENS) - 1; i++) {
		left_diagonal[i] = 0;
		right_diagonal[i] = 0;
	}
	// Increment each diagonal counter if a queen is present.
	for (int i = 0; i < NQUEENS; i++) {
		left_diagonal[i + board->q[i]]++;
		right_diagonal[NQUEENS - 1 - i + board->q[i]]++;
	}
	// If a diagonal-counter's value is greater than 1, there are at least two queens on that diagonal,
	// and therefore, a conflict exists between them, so increment 'sum' appropriately.
	int sum = 0;
	for (int i = 0; i < (2 * NQUEENS) - 1; i++) {
		int counter = 0;
		if (left_diagonal[i] > 1) {
			counter += left_diagonal[i] - 1;
		}
		if (right_diagonal[i] > 1) {
			counter += right_diagonal[i] - 1;
		}
		sum += counter;
	}
	board->totalCost = sum;
}

// Comparsion function for qsort(). Compares two boards' 'totalCost' attributes.
int cmpBoardCosts(const void * a, const void * b) {
	BOARD * ab = (BOARD *)a;
	BOARD * bb = (BOARD *)b;
	return (ab->totalCost) - (bb->totalCost);
}

// Places queens on a new board.
// Then calls computeCost() to fill each board's 'cost' attribute.
void initialise(BOARD * board) {
	// Fill an array with unique y (row) positions.
	int array[NQUEENS];
	for (int i = 0; i < NQUEENS; i++) {
		array[i] = i;
	}
	// Shuffle the array.
	for (int i = 0; i < NQUEENS; i++) {
		int temp = array[i];
		int randomIndex = rand() % NQUEENS;
		array[i] = array[randomIndex];
		array[randomIndex] = temp;
	}
	// Set the queens' positions y (row) positions.
	// A queen's x (column) position is defined as its index in the 'q' array.
	// This method ensures there are no row or column conflicts.
	for (int i = 0; i < NQUEENS; i++) {
		int y = array[i];
		board->q[i] = y;
	}
	computeCosts(board);
}

// Print the locations of queens on 'board'. Print a newline every 5 queens.
void printBoard(const BOARD * board) {
	printf("Queens: {\n");
	for (int i = 0; i < NQUEENS; i++) {
		printf("%d", board->q[i]);
		if (i == NQUEENS - 1) continue;
		printf(", ");
		if (i != 0 && i % 10 == 0) putchar('\n');
	}
	printf("\n}\n");
}

// Swaps two queens in 'board' at random.
// Equivalent to swapping the queen's columns.
void mutateQueen(BOARD * board) {
	// Pick two queens at random.
	int index1 = rand() % NQUEENS;
	int index2 = rand() % NQUEENS;
	while (index2 == index1) {
		index2 = rand() % NQUEENS;
	}
	// Swap them.
	int temp = board->q[index1];
	board->q[index1] = board->q[index2];
	board->q[index2] = temp;
}

// Prints the parameters used for the search.
void printParameters(void) {
	printf("Parameters:\n"
			"NQUEENS: %d\n" "NUM_BOARDS: %d\n" "POPULATION_THRESHOLD: %d\n"
			"Serial Search:\n" "GN_MUTATION_RATE: %d\n" "MAX_REPRODUCTIONS: %d\n"
			"Parallel Search:\n" "BOARDS_PER_SLAVE: %d\n" "SLAVE_ITERATIONS_PER_CYCLE: %d\n" "MUTATIONS_PER_SLAVE_ITERATION: %d\n\n",
			NQUEENS, NUM_BOARDS, POPULATION_THRESHOLD, GN_MUTATION_RATE, MAX_REPRODUCTIONS,
			BOARDS_PER_SLAVE, SLAVE_ITERATIONS_PER_CYCLE, MUTATIONS_PER_SLAVE_ITERATION);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// END OF FILE
///////////////////////////////////////////////////////////////////////////////////////////////////