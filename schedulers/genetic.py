import random
import copy
from utils import generate_initial_solution, calculate_cost, make_neighbor_solution

class GeneticAlgorithmScheduler:
    def __init__(self, instance_data, population_size=50, generations=100, crossover_rate=0.8, mutation_rate=0.1, elitism_count=2, local_search_rate=0.3):
        self.instance_data = instance_data
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate  # Taxa de mutação inicial
        self.elitism_count = elitism_count  # Número de indivíduos preservados por geração
        self.local_search_rate = local_search_rate  # Probabilidade de aplicar busca local em um filho

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

    def mutate(self, solution, mutation_rate):
        """Mutação: altera aleatoriamente a atribuição de um paciente com certa probabilidade."""
        for patient in solution:
            if random.random() < mutation_rate:
                solution[patient]['ward'] = random.choice(list(self.instance_data['wards'].keys()))
                solution[patient]['day'] = random.randint(
                    self.instance_data['patients'][patient]['earliest_admission'],
                    self.instance_data['patients'][patient]['latest_admission']
                )
        return solution

    def local_search(self, solution, iterations=5):
        """Realiza uma busca local simples para refinar a solução."""
        best_solution = solution
        best_cost = calculate_cost(self.instance_data, solution)
        for _ in range(iterations):
            neighbor = make_neighbor_solution(self.instance_data, best_solution)
            neighbor_cost = calculate_cost(self.instance_data, neighbor)
            if neighbor_cost < best_cost:
                best_solution = neighbor
                best_cost = neighbor_cost
        return best_solution

    def run(self):
        for gen in range(self.generations):
            new_population = []
            # Taxa de mutação adaptativa: diminui conforme as gerações avançam
            current_mutation_rate = self.mutation_rate * (1 - (gen / self.generations))
            
            # Avalia a população e atualiza o melhor indivíduo
            population_costs = [calculate_cost(self.instance_data, sol) for sol in self.population]
            gen_best_cost = min(population_costs)
            self.cost_history.append(gen_best_cost)
            if gen_best_cost < self.best_cost:
                self.best_cost = gen_best_cost
                self.best_solution = copy.deepcopy(self.population[population_costs.index(gen_best_cost)])
            
            # Elitismo: preserva os melhores indivíduos
            elites = sorted(self.population, key=lambda sol: calculate_cost(self.instance_data, sol))[:self.elitism_count]

            # Cria nova população
            while len(new_population) < self.population_size - self.elitism_count:
                parent1 = self.selection()
                parent2 = self.selection()
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                child1 = self.mutate(child1, current_mutation_rate)
                child2 = self.mutate(child2, current_mutation_rate)
                # Aplica busca local com certa probabilidade
                if random.random() < self.local_search_rate:
                    child1 = self.local_search(child1)
                if random.random() < self.local_search_rate:
                    child2 = self.local_search(child2)
                new_population.extend([child1, child2])
            
            self.population = elites + new_population
            self.population = self.population[:self.population_size]
            print(f"Generation {gen+1}/{self.generations} - Best cost: {self.best_cost}")
        return self.best_solution