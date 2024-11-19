# Genetic Algorithm - THE KNAPSACK PROBLEM
- Sean Hulseman
## Project Overview
This project implements a genetic algorithm to solve the **backpacking problem**, where the objective is to maximize the value of items packed in a backpack without exceeding a specified weight limit. The backpack can hold a maximum weight of 250 units, and each item has a defined weight and importance value.

## Problem Definition
Given a set of items, each with a specific weight and value, the goal is to select a combination of items to maximize the total value in the backpack while ensuring that the total weight does not exceed 250 units.

## Genome Definition
In this genetic algorithm, the genome is represented as a binary list, where:
- `1` indicates that an item is included in the backpack.
- `0` indicates that an item is not included.

Each genome corresponds to a potential solution to the backpacking problem.

## Fringe Operations
The following operations are defined to support the genetic algorithm:

1. **Fitness Evaluation**:
   - Calculates the total weight and total value of the items represented by the genome.
   - Ensures that the total weight does not exceed the maximum weight (250).

2. **Population Initialization**:
   - Generates an initial population of genomes (potential solutions).

3. **Selection**:
   - Cull the population by retaining the top 50% of genomes based on their fitness scores.

4. **Recombination**:
   - Perform single-point crossover between two parent genomes to produce offspring.

5. **Mutation**:
   - Introduces random changes to the genomes to maintain diversity within the population.

## Algorithm Execution
The genetic algorithm operates through multiple generations, applying selection, recombination, and mutation iteratively. The process continues until a specified number of generations is reached.

## Requirements
- Python 3.x
- NumPy
- Matplotlib (for visualization)

