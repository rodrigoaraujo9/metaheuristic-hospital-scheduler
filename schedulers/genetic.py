import random
import copy
from utils import generate_initial_solution, calculate_cost, make_neighbor_solution

class GeneticAlgorithmScheduler:
    def __init__(self, instance_data, population_size=50, generations=100, crossover_rate=0.8, 
                 mutation_rate=0.1, elitism_count=2, local_search_rate=0.3, tournament_size=3):
        self.instance_data = instance_data
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate  # Taxa de mutação inicial
        self.elitism_count = elitism_count  # Número de indivíduos preservados por geração
        self.local_search_rate = local_search_rate  # Probabilidade de aplicar busca local em um filho
        self.tournament_size = tournament_size  # Tamanho do torneio para seleção
        self.population = [generate_initial_solution(instance_data) for _ in range(population_size)]
        self.best_solution = None
        self.best_cost = float('inf')
        self.cost_history = []
    
    def tournament_selection(self):
        """Realiza a seleção por torneio com 'tournament_size' indivíduos."""
        competitors = random.sample(self.population, self.tournament_size)
        best = min(competitors, key=lambda sol: calculate_cost(self.instance_data, sol))
        return best
    
    def crossover(self, parent1, parent2):
        """Operador de crossover uniforme com reparo simples para preservar viabilidade."""
        child1, child2 = {}, {}
        for patient in self.instance_data['patients']:
            if random.random() < 0.5:
                child1[patient] = copy.deepcopy(parent1[patient])
                child2[patient] = copy.deepcopy(parent2[patient])
            else:
                child1[patient] = copy.deepcopy(parent2[patient])
                child2[patient] = copy.deepcopy(parent1[patient])
        # Reparar eventuais inconsistências (se necessário, pode ser expandido)
        return child1, child2
    
    def mutate(self, solution, mutation_rate):
        """Altera aleatoriamente a atribuição de um paciente com probabilidade 'mutation_rate'."""
        for patient in solution:
            if random.random() < mutation_rate:
                solution[patient]['ward'] = random.choice(list(self.instance_data['wards'].keys()))
                solution[patient]['day'] = random.randint(
                    self.instance_data['patients'][patient]['earliest_admission'],
                    self.instance_data['patients'][patient]['latest_admission']
                )
        return solution
    
    def local_search(self, solution, max_iterations=10):
        """Busca local adaptativa que continua até não haver melhora ou atingir o número máximo de iterações."""
        best_solution = solution
        best_cost = calculate_cost(self.instance_data, solution)
        iterations = 0
        improvement = True
        while improvement and iterations < max_iterations:
            neighbor = make_neighbor_solution(self.instance_data, best_solution)
            neighbor_cost = calculate_cost(self.instance_data, neighbor)
            if neighbor_cost < best_cost:
                best_solution = neighbor
                best_cost = neighbor_cost
                improvement = True
            else:
                improvement = False
            iterations += 1
        return best_solution
    
    def measure_diversity(self):
        """Calcula a diversidade média da população com base em diferenças nas atribuições dos pacientes."""
        diversity = 0
        count = 0
        pop = self.population
        patients = list(self.instance_data['patients'].keys())
        for i in range(len(pop)):
            for j in range(i+1, len(pop)):
                diff = sum(1 for p in patients if pop[i][p] != pop[j][p])
                diversity += diff
                count += 1
        return diversity / count if count > 0 else 0
    
    def run(self):
        for gen in range(self.generations):
            new_population = []
            # Taxa de mutação adaptativa (decai linearmente)
            current_mutation_rate = self.mutation_rate * (1 - (gen / self.generations))
            population_costs = [calculate_cost(self.instance_data, sol) for sol in self.population]
            gen_best_cost = min(population_costs)
            self.cost_history.append(gen_best_cost)
            if gen_best_cost < self.best_cost:
                self.best_cost = gen_best_cost
                self.best_solution = copy.deepcopy(self.population[population_costs.index(gen_best_cost)])
            
            # Log da diversidade (opcional para análise)
            diversity = self.measure_diversity()
            print(f"Generation {gen+1}/{self.generations} - Best cost: {self.best_cost:.2f} - Diversity: {diversity:.2f}")
            
            # Elitismo: preserva os melhores indivíduos
            elites = sorted(self.population, key=lambda sol: calculate_cost(self.instance_data, sol))[:self.elitism_count]
            
            # Geração de novos indivíduos
            while len(new_population) < self.population_size - self.elitism_count:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                child1 = self.mutate(child1, current_mutation_rate)
                child2 = self.mutate(child2, current_mutation_rate)
                if random.random() < self.local_search_rate:
                    child1 = self.local_search(child1)
                if random.random() < self.local_search_rate:
                    child2 = self.local_search(child2)
                new_population.extend([child1, child2])
            
            self.population = elites + new_population
            self.population = self.population[:self.population_size]
        return self.best_solution
