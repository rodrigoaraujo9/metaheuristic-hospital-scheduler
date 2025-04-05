import random
import copy
import time
import math
from utils import calculate_cost

class GeneticAlgorithmScheduler:
    def __init__(self, instance_data, population_size=100, generations=200, 
                 crossover_rate=0.85, mutation_rate=0.3, elitism_count=2):
        self.instance_data = instance_data
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        
        # Metrics
        self.runtime = 0
        self.best_solution = None
        self.best_cost = float('inf')
        self.cost_history = []
        
        # Generate truly random initial population
        print("Generating initial population...")
        self.population = self.generate_random_population(use_worse_initial=True)
        
        # Evaluate initial population
        self.evaluate_population()
    
    def generate_random_population(self, use_worse_initial=False):
        """Generate a random initial population with controlled quality."""
        population = []
        
        for _ in range(self.population_size):
            # Create completely random solution
            solution = {}
            
            for patient in self.instance_data['patients']:
                patient_data = self.instance_data['patients'][patient]
                earliest = patient_data['earliest_admission']
                latest = patient_data['latest_admission']
                
                # For worse initial solutions, bias toward later days when possible
                if use_worse_initial and latest > earliest:
                    # With 70% probability choose a later day
                    if random.random() < 0.7:
                        midpoint = earliest + (latest - earliest) // 2
                        day = random.randint(midpoint, latest)
                    else:
                        day = random.randint(earliest, latest)
                else:
                    day = random.randint(earliest, latest)
                
                # Random ward assignment (completely random, not optimized)
                ward = random.choice(list(self.instance_data['wards'].keys()))
                
                solution[patient] = {'ward': ward, 'day': day}
            
            population.append(solution)
        
        return population
    
    def evaluate_population(self):
        """Evaluate entire population and update best solution."""
        population_costs = []
        for solution in self.population:
            try:
                cost = calculate_cost(self.instance_data, solution)
                population_costs.append(cost)
            except:
                population_costs.append(float('inf'))
        
        if population_costs:
            min_cost = min(population_costs)
            min_index = population_costs.index(min_cost)
            
            if min_cost < self.best_cost:
                self.best_cost = min_cost
                self.best_solution = copy.deepcopy(self.population[min_index])
        
        return population_costs
    
    def tournament_selection(self, costs=None, tournament_size=3):
        """Tournament selection with configurable tournament size."""
        if costs is None:
            costs = [calculate_cost(self.instance_data, sol) for sol in self.population]
        
        # Select random individuals
        competitors_indices = random.sample(range(len(self.population)), tournament_size)
        
        # Return the best one
        best_competitor_idx = min(competitors_indices, key=lambda i: costs[i])
        return self.population[best_competitor_idx]
    
    def uniform_crossover(self, parent1, parent2):
        """Uniform crossover with 50% chance of selecting genes from each parent."""
        child1, child2 = {}, {}
        
        for patient in parent1:
            if random.random() < 0.5:
                child1[patient] = copy.deepcopy(parent1[patient])
                child2[patient] = copy.deepcopy(parent2[patient])
            else:
                child1[patient] = copy.deepcopy(parent2[patient])
                child2[patient] = copy.deepcopy(parent1[patient])
        
        return child1, child2
    
    def two_point_crossover(self, parent1, parent2):
        """Two-point crossover for better preservation of good sequences."""
        patients = list(parent1.keys())
        
        if len(patients) <= 2:
            return self.uniform_crossover(parent1, parent2)
            
        # Select two crossover points
        point1, point2 = sorted(random.sample(range(len(patients)), 2))
        
        # Create children
        child1, child2 = {}, {}
        
        for i, patient in enumerate(patients):
            # Middle section from opposite parent, ends from same parent
            if i >= point1 and i < point2:
                child1[patient] = copy.deepcopy(parent2[patient])
                child2[patient] = copy.deepcopy(parent1[patient])
            else:
                child1[patient] = copy.deepcopy(parent1[patient])
                child2[patient] = copy.deepcopy(parent2[patient])
        
        return child1, child2
    
    def crossover(self, parent1, parent2):
        """Apply crossover with different strategies."""
        # Choose crossover method randomly
        if random.random() < 0.5:
            child1, child2 = self.uniform_crossover(parent1, parent2)
        else:
            child1, child2 = self.two_point_crossover(parent1, parent2)
        
        self.repair_solution(child1)
        self.repair_solution(child2)
        return child1, child2
    
    def repair_solution(self, solution):
        """Basic solution repair to ensure validity."""
        for patient, data in solution.items():
            patient_data = self.instance_data['patients'].get(patient)
            if patient_data is None:
                continue
                
            # Fix invalid ward
            if data.get('ward') is None or data['ward'] not in self.instance_data['wards']:
                data['ward'] = random.choice(list(self.instance_data['wards'].keys()))
            
            # Fix invalid day
            earliest = patient_data['earliest_admission']
            latest = patient_data['latest_admission']
            
            if data.get('day') is None or data['day'] < earliest or data['day'] > latest:
                data['day'] = random.randint(earliest, latest)
    
    def mutate(self, solution, mutation_rate):
        """Enhanced mutation with different strategies."""
        mutated = False
        
        for patient in solution:
            if random.random() < mutation_rate:
                mutated = True
                patient_data = self.instance_data['patients'].get(patient)
                if not patient_data:
                    continue
                
                # Pick mutation strategy
                mutation_strategy = random.choices(
                    ["ward", "day", "both", "shift_day", "explore"],
                    weights=[0.3, 0.3, 0.2, 0.1, 0.1]
                )[0]
                
                earliest = patient_data['earliest_admission']
                latest = patient_data['latest_admission']
                
                if mutation_strategy == "ward":
                    # Change ward
                    solution[patient]['ward'] = random.choice(list(self.instance_data['wards'].keys()))
                
                elif mutation_strategy == "day":
                    # Change day (within valid range)
                    solution[patient]['day'] = random.randint(earliest, latest)
                
                elif mutation_strategy == "both":
                    # Change both ward and day
                    solution[patient]['ward'] = random.choice(list(self.instance_data['wards'].keys()))
                    solution[patient]['day'] = random.randint(earliest, latest)
                
                elif mutation_strategy == "shift_day":
                    # Shift day by +/-1 (if possible)
                    current_day = solution[patient]['day']
                    shift = random.choice([-1, 1])
                    new_day = current_day + shift
                    
                    if earliest <= new_day <= latest:
                        solution[patient]['day'] = new_day
                
                elif mutation_strategy == "explore":
                    # Radical change for exploration
                    solution[patient]['ward'] = random.choice(list(self.instance_data['wards'].keys()))
                    solution[patient]['day'] = random.randint(earliest, latest)
        
        return solution, mutated
    
    def scramble_mutation(self, solution, intensity=0.3):
        """Scramble mutation that randomly reassigns groups of patients."""
        patients = list(solution.keys())
        
        # Select a subset of patients to scramble
        scramble_count = max(1, int(len(patients) * intensity))
        scramble_patients = random.sample(patients, scramble_count)
        
        # Scramble them
        for patient in scramble_patients:
            patient_data = self.instance_data['patients'].get(patient)
            if not patient_data:
                continue
                
            earliest = patient_data['earliest_admission']
            latest = patient_data['latest_admission']
            
            # Completely rerandomize this patient
            solution[patient]['ward'] = random.choice(list(self.instance_data['wards'].keys()))
            solution[patient]['day'] = random.randint(earliest, latest)
            
        return solution
    
    def generate_restart_population(self, best_solution, divergence=0.7):
        """Generate a partially new population centered around the best solution."""
        new_population = []
        
        # Keep the best solution
        new_population.append(copy.deepcopy(best_solution))
        
        # Generate new solutions based on the best
        for _ in range(self.population_size - 1):
            # Create a new solution by mutating the best solution
            new_solution = copy.deepcopy(best_solution)
            
            # Apply heavy mutation to diverge from best solution
            if random.random() < divergence:
                new_solution = self.scramble_mutation(new_solution, intensity=0.5)
            else:
                # Light mutation to stay near best solution
                new_solution, _ = self.mutate(new_solution, self.mutation_rate * 3)
            
            new_population.append(new_solution)
            
        return new_population
    
    def run(self):
        """Run the genetic algorithm."""
        print(f"Starting genetic algorithm with population size {self.population_size} for {self.generations} generations")
        
        start_time = time.time()
        
        # Track improvement for logging
        prev_best_cost = self.best_cost
        stagnation_counter = 0
        restart_counter = 0
        
        # Temporary variables
        current_mutation_rate = self.mutation_rate
        
        for gen in range(self.generations):
            try:
                # Evaluate current population
                population_costs = self.evaluate_population()
                
                if population_costs:
                    gen_best_cost = min(population_costs)
                    self.cost_history.append(gen_best_cost)
                    
                    # Log progress with improvement percentage
                    elapsed = time.time() - start_time
                    improvement = 0
                    if prev_best_cost != float('inf'):
                        improvement = (prev_best_cost - self.best_cost) / prev_best_cost * 100
                        
                    print(f"Generation {gen+1}/{self.generations} - Best cost: {self.best_cost:.2f} - " + 
                          f"Improvement: {improvement:.4f}% - Time: {elapsed:.2f}s")
                    
                    # Check for stagnation
                    if abs(prev_best_cost - self.best_cost) < 0.001:
                        stagnation_counter += 1
                    else:
                        stagnation_counter = 0
                        prev_best_cost = self.best_cost
                    
                    # Create next generation
                    new_population = []
                    
                    # Elitism: preserve best solutions
                    sorted_indices = sorted(range(len(population_costs)), key=lambda i: population_costs[i])
                    for idx in sorted_indices[:self.elitism_count]:
                        new_population.append(copy.deepcopy(self.population[idx]))
                    
                    # Dynamic adjustment based on stagnation
                    if stagnation_counter > 15:
                        print("Stagnation detected - increasing mutation rate temporarily")
                        current_mutation_rate = min(0.8, self.mutation_rate * 2)
                        stagnation_counter = 0
                    elif stagnation_counter > 10:
                        current_mutation_rate = min(0.6, self.mutation_rate * 1.5)
                    else:
                        current_mutation_rate = self.mutation_rate
                    
                    # Population restart if long-term stagnation detected
                    if stagnation_counter > 20:
                        restart_counter += 1
                        if restart_counter >= 3:
                            print("Major stagnation detected - restarting population with diversity")
                            self.population = self.generate_restart_population(self.best_solution)
                            restart_counter = 0
                            stagnation_counter = 0
                            continue
                    
                    # Generate new individuals
                    while len(new_population) < self.population_size:
                        try:
                            # Selection
                            parent1 = self.tournament_selection(population_costs, tournament_size=4)
                            parent2 = self.tournament_selection(population_costs, tournament_size=4)
                            
                            # Crossover
                            if random.random() < self.crossover_rate:
                                child1, child2 = self.crossover(parent1, parent2)
                            else:
                                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                            
                            # Mutation
                            child1, _ = self.mutate(child1, current_mutation_rate)
                            child2, _ = self.mutate(child2, current_mutation_rate)
                            
                            # Add to new population
                            new_population.append(child1)
                            if len(new_population) < self.population_size:
                                new_population.append(child2)
                                
                        except Exception as e:
                            print(f"Error generating children: {str(e)}")
                    
                    # Update population
                    self.population = new_population[:self.population_size]
                    
            except Exception as e:
                print(f"Error in generation {gen+1}: {str(e)}")
                continue
        
        # Calculate total runtime
        self.runtime = time.time() - start_time
        print(f"Genetic algorithm completed in {self.runtime:.2f} seconds")
        print(f"Best cost found: {self.best_cost:.2f}")
        
        # Calculate total improvement
        initial_best = self.cost_history[0] if self.cost_history else float('inf')
        if initial_best != float('inf') and self.best_cost != float('inf'):
            total_improvement = (initial_best - self.best_cost) / initial_best * 100
            print(f"Total improvement: {total_improvement:.2f}%")
        
        return self.best_solution