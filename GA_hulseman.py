import numpy as np
import matplotlib.pyplot as plt
items = [
    {"value": 6, "weight": 20},
    {"value": 5, "weight": 30},
    {"value": 8, "weight": 60},
    {"value": 7, "weight": 90},
    {"value": 6, "weight": 50},
    {"value": 9, "weight": 70},
    {"value": 4, "weight": 30},
    {"value": 5, "weight": 30},
    {"value": 4, "weight": 70},
    {"value": 9, "weight": 20},
    {"value": 2, "weight": 20},
    {"value": 1, "weight": 60}
]

max_weight = 250
population_size = 100
mutation_rate = .01
num_generations = 100


# Method to sum all the items' weights. Can take a genome (the 1's are items you tried to pack in the bag) and calculate based on phenotypes* present
def total_weight(items, genome=[1,1,1,1,1,1,1,1,1,1,1,1]):
    return np.sum([item['weight'] * gene for item, gene in zip(items, genome)])

# Method to sum all the items' values. Same functionality as total_weight()
def total_value(items, genome=[1,1,1,1,1,1,1,1,1,1,1,1]):
    return np.sum([item['value'] * gene for item, gene in zip(items, genome)])

# combining the two above and zeroing out overweight bags to evaluate fitness
def evaluate_fitness(items, genome, max_weight=250):
    # Calculate the total weight and value of the genome
    total_wt = total_weight(items, genome)
    total_val = total_value(items, genome)
    
    # If the total weight is within the limit, return the total value, otherwise return 0 (or penalize)
    if total_wt <= max_weight:
        return total_val
    else:
        return 0  # or penalize with some negative value if desired

# method to make an individual. 1's and 0's randomly 
def pack_bag(items):
    return [np.random.randint(0,2) for _ in items] 
    
# pack_bag(items)
def make_population_bags(items = items, pop_size = 100):
    return [pack_bag(items) for _ in range(pop_size)]

def recombine_genomes(parent1, parent2):
    # Ensure both parents have the same genome length
    assert len(parent1) == len(parent2), "Genomes must be of the same length"
    
    # Select a random crossover point (not including the last index)
    crossover_point = np.random.randint(1, len(parent1) - 1)
    
    # Create offspring by combining parts from each parent
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    
    return offspring1, offspring2
def mutate_genome(genome, mutation_rate=0.01):
    # Copy the genome to avoid modifying the original directly
    mutated_genome = genome[:]
    
    # Apply mutation based on mutation rate
    if np.random.rand() < mutation_rate:
        # Choose a random point for mutation
        mutation_point = np.random.randint(len(genome))
        
        # Flip the bit at the mutation point
        mutated_genome[mutation_point] = 1 - mutated_genome[mutation_point]
    
    return mutated_genome

def genetic_algorithm(items, pop_size=100, max_weight=250, mutation_rate=0.01, num_generations=100):
    # Step 1: Initialize population
    population = make_population_bags(items, pop_size)
    # Initialize lists to track average fitness and weight per generation
    avg_fitness_per_gen = []
    avg_weight_per_gen = []
    # Evolutionary loop
    for generation in range(num_generations):
        # Step 2: Evaluate fitness for each genome
        fitness_scores = [evaluate_fitness(items, genome, max_weight) for genome in population]
        weights = [total_weight(items, genome) for genome in population]
        
        # Record the average fitness and weight of the current population
        avg_fitness = np.mean(fitness_scores)
        avg_weight = np.mean(weights)
        avg_fitness_per_gen.append(avg_fitness)
        avg_weight_per_gen.append(avg_weight)
        

        # Step 3: Sort and select top 50% based on fitness
        sorted_population = [genome for _, genome in sorted(zip(fitness_scores, population), reverse=True)]
        population = sorted_population[:pop_size // 2]  # Keep top 50%
        
        # Step 4: Reproduce to refill the population
        next_generation = []
        while len(next_generation) < pop_size:
            # Select two random parents from the top 50%
            # Select two random parents by their indices
            parent_indices = np.random.choice(len(population), 2, replace=False)
            parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]
            
            # Create offspring via recombination
            offspring1, offspring2 = recombine_genomes(parent1, parent2)
            
            # Apply mutation
            offspring1 = mutate_genome(offspring1, mutation_rate)
            offspring2 = mutate_genome(offspring2, mutation_rate)
            
            # Add offspring to the next generation
            next_generation.extend([offspring1, offspring2])
        
        # Replace the old population with the new generation
        population = next_generation[:pop_size]  # Trim to exact pop_size if necessary
        
        # Optional: Print the best fitness score of the generation
        best_fitness = max([evaluate_fitness(items, genome, max_weight) for genome in population])
        print(f"Generation {generation+1}: Best Fitness = {best_fitness} Avg Fitness = {avg_fitness}, Avg Weight = {avg_weight}")
    
    # Final best solution
    best_genome = max(population, key=lambda genome: evaluate_fitness(items, genome, max_weight))
    best_fitness = evaluate_fitness(items, best_genome, max_weight)
    
    # Plotting average fitness and weight per generation
    plt.figure(figsize=(12, 6))
    
    # Plot average fitness
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_generations + 1), avg_fitness_per_gen, label="Average Fitness", color="blue")
    plt.xlabel("Generation")
    plt.ylabel("Average Fitness")
    plt.title("Average Fitness per Generation")
    plt.legend()
    
    # Plot average weight
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_generations + 1), avg_weight_per_gen, label="Average Weight", color="orange")
    plt.xlabel("Generation")
    plt.ylabel("Average Weight")
    plt.title("Average Weight per Generation")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    return best_genome, best_fitness

# Run the genetic algorithm
best_genome, best_fitness = genetic_algorithm(items, pop_size=100, max_weight=250, mutation_rate=0.01, num_generations=100)
winner_weight = total_weight(items, best_genome)
print("Best Genome:", best_genome)
print("Best Fitness:", best_fitness)
print('Weight of winner: ', winner_weight)