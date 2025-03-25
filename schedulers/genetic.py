import random
import copy
from utils import generate_initial_solution, calculate_cost

class GeneticAlgorithmScheduler:
    def __init__(self, instance_data, population_size=50, generations=100, crossover_rate=0.8, mutation_rate=0.1):
        self.instance_data = instance_data
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        # Inicializa a população
        self.population = [generate_initial_solution(instance_data) for _ in range(population_size)]
        self.best_solution = None
        self.best_cost = float('inf')
        self.cost_history = []

    def selection(self):
        """Seleção por torneio: escolhe dois indivíduos aleatórios e retorna o melhor."""
        a, b = random.sample(self.population, 2)      
        return a if calculate_cost(self.instance_data, a) < calculate_cost(self.instance_data, b) else b

    def crossover(self, parent1, parent2):
        """Crossover uniforme: para cada paciente, escolhe a atribuição de um dos pais."""
        child1, child2 = {}, {}
        for patient in self.instance_data['patients']:
            if random.random() < 0.5:
                child1[patient] = copy.deepcopy(parent1[patient])
                child2[patient] = copy.deepcopy(parent2[patient])
            else:
                child1[patient] = copy.deepcopy(parent2[patient])
                child2[patient] = copy.deepcopy(parent1[patient])
        return child1, child2

    def mutate(self, solution):
        """Mutação: altera aleatoriamente a atribuição de um paciente com certa probabilidade."""
        for patient in solution:
            if random.random() < self.mutation_rate:
                solution[patient]['ward'] = random.choice(list(self.instance_data['wards'].keys()))
                solution[patient]['day'] = random.randint(
                    self.instance_data['patients'][patient]['earliest_admission'],
                    self.instance_data['patients'][patient]['latest_admission']
                )
        return solution

    def run(self):
        for _ in range(self.generations):
            new_population = []
            population_costs = [calculate_cost(self.instance_data, sol) for sol in self.population]
            gen_best_cost = min(population_costs)
            self.cost_history.append(gen_best_cost)
            if gen_best_cost < self.best_cost:
                self.best_cost = gen_best_cost
                self.best_solution = copy.deepcopy(self.population[population_costs.index(gen_best_cost)])

            while len(new_population) < self.population_size:
                parent1 = self.selection()
                parent2 = self.selection()
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            self.population = new_population[:self.population_size]
        return self.best_solution
