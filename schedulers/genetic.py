import random
import copy
import time
from utils import calculate_cost

class GeneticAlgorithmScheduler:
    def __init__(self, instance_data, population_size=150, generations=100, 
                 crossover_rate=0.8, mutation_rate=0.3, elitism_count=2):
        self.instance_data = instance_data
        self.population_size = population_size  # População maior
        self.generations = generations  # Mais gerações
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        
        # Métricas para exibição
        self.runtime = 0
        self.best_solution = None
        self.best_cost = float('inf')
        self.cost_history = []
        self.final_cost = None      # Atributo para exibição no app
        self.iterations = None      # Atributo para exibição no app (usado se 'iterations' não existir, usa 'generations')
        
        print("Generating initial population...")
        self.population = self.generate_random_population()
        self.evaluate_population()
    
    def generate_random_population(self):
        """Gera população 100% aleatória."""
        population = []
        for _ in range(self.population_size):
            solution = {}
            for patient in self.instance_data['patients']:
                patient_data = self.instance_data['patients'][patient]
                earliest = patient_data['earliest_admission']
                latest = patient_data['latest_admission']
                day = random.randint(earliest, latest)
                ward = random.choice(list(self.instance_data['wards'].keys()))
                solution[patient] = {'ward': ward, 'day': day}
            population.append(solution)
        return population
    
    def evaluate_population(self):
        """Avalia toda a população."""
        population_costs = []
        for solution in self.population:
            try:
                cost = calculate_cost(self.instance_data, solution)
                population_costs.append(cost)
            except Exception as e:
                population_costs.append(float('inf'))
        
        if population_costs:
            min_cost = min(population_costs)
            min_index = population_costs.index(min_cost)
            if min_cost < self.best_cost:
                self.best_cost = min_cost
                self.best_solution = copy.deepcopy(self.population[min_index])
        
        return population_costs
    
    def selection(self, costs):
        """Seleção simples por torneio."""
        tournament_size = 3
        competitors = random.sample(range(len(self.population)), tournament_size)
        best_idx = min(competitors, key=lambda i: costs[i])
        return self.population[best_idx]
    
    def crossover(self, parent1, parent2):
        """Crossover uniforme simples."""
        child = {}
        for patient in parent1:
            if random.random() < 0.5:
                child[patient] = copy.deepcopy(parent1[patient])
            else:
                child[patient] = copy.deepcopy(parent2[patient])
        return child
    
    def mutate(self, solution):
        """Mutação agressiva focada em early_day."""
        for patient in solution:
            if random.random() < 0.4:
                patient_data = self.instance_data['patients'].get(patient)
                if not patient_data:
                    continue
                earliest = patient_data['earliest_admission']
                latest = patient_data['latest_admission']
                if random.random() < 0.8:
                    solution[patient]['day'] = earliest
                else:
                    solution[patient]['day'] = random.randint(earliest, latest)
                solution[patient]['ward'] = random.choice(list(self.instance_data['wards'].keys()))
        return solution
    
    def create_random_solution(self):
        """Cria uma solução 100% aleatória."""
        solution = {}
        for patient in self.instance_data['patients']:
            patient_data = self.instance_data['patients'][patient]
            earliest = patient_data['earliest_admission']
            latest = patient_data['latest_admission']
            solution[patient] = {
                'day': random.randint(earliest, latest),
                'ward': random.choice(list(self.instance_data['wards'].keys()))
            }
        return solution
    
    def run(self):
        """Motor do algoritmo genético simplificado."""
        print(f"Starting genetic algorithm with population size {self.population_size} for {self.generations} generations")
        start_time = time.time()
        stagnation_counter = 0
        previous_best = self.best_cost
        
        for gen in range(self.generations):
            try:
                # Avaliar população atual
                population_costs = self.evaluate_population()
                if population_costs:
                    gen_best_cost = min(population_costs)
                    self.cost_history.append(gen_best_cost)
                    
                    elapsed = time.time() - start_time
                    improvement = 0
                    if previous_best != float('inf'):
                        improvement = (previous_best - self.best_cost) / previous_best * 100
                    print(f"Generation {gen+1}/{self.generations} - Best: {self.best_cost:.2f} - Improvement: {improvement:.4f}% - Time: {elapsed:.2f}s")
                    
                    # Verificar estagnação
                    if abs(previous_best - self.best_cost) < 0.01:
                        stagnation_counter += 1
                    else:
                        stagnation_counter = 0
                        previous_best = self.best_cost
                    
                    # Nova população
                    new_population = []
                    
                    # Elitismo - manter as melhores soluções
                    sorted_indices = sorted(range(len(population_costs)), key=lambda i: population_costs[i])
                    for idx in sorted_indices[:self.elitism_count]:
                        new_population.append(copy.deepcopy(self.population[idx]))
                    
                    # Estratégia de restart parcial
                    if stagnation_counter >= 7 or gen % 10 == 0:
                        best_solution = copy.deepcopy(self.population[sorted_indices[0]])
                        new_population = [best_solution]
                        for _ in range(self.population_size * 3 // 10):
                            variant = copy.deepcopy(best_solution)
                            for patient in variant:
                                if random.random() < 0.7:
                                    patient_data = self.instance_data['patients'].get(patient)
                                    if patient_data:
                                        variant[patient]['day'] = patient_data['earliest_admission']
                            new_population.append(variant)
                        while len(new_population) < self.population_size:
                            new_population.append(self.create_random_solution())
                        stagnation_counter = 0
                        print(f"Generation {gen+1}: Restarting population for better exploration")
                        self.population = new_population
                        continue
                    
                    # Gerar novos indivíduos
                    while len(new_population) < self.population_size:
                        parent1 = self.selection(population_costs)
                        parent2 = self.selection(population_costs)
                        if random.random() < self.crossover_rate:
                            child = self.crossover(parent1, parent2)
                        else:
                            child = copy.deepcopy(parent1)
                        child = self.mutate(child)
                        new_population.append(child)
                    
                    self.population = new_population
            except Exception as e:
                print(f"Error in generation {gen+1}: {str(e)}")
                continue
        
        self.runtime = time.time() - start_time
        # Definir atributos para compatibilidade com o app
        self.final_cost = self.best_cost
        self.iterations = self.generations
        
        print(f"Genetic algorithm completed in {self.runtime:.2f} seconds")
        print(f"Best cost found: {self.best_cost:.2f}")
        initial_best = self.cost_history[0] if self.cost_history else float('inf')
        if initial_best != float('inf') and self.best_cost != float('inf'):
            total_improvement = (initial_best - self.best_cost) / initial_best * 100
            print(f"Total improvement: {total_improvement:.2f}%")
        
        return self.best_solution
