import time
import random
import math
import copy
from utils import generate_initial_solution, calculate_cost, make_neighbor_solution

class HybridSimulatedAnnealingScheduler:
    def __init__(self, instance_data, initial_temp=1000, cooling_rate=0.995, min_temp=1,
                 local_search_interval=100, local_search_iterations=50):
        self.instance_data = instance_data
        self.temperature = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.current_solution = generate_initial_solution(instance_data)
        self.best_solution = copy.deepcopy(self.current_solution)
        self.iterations = 0
        self.runtime = 0
        self.final_cost = None

        # Parâmetros da busca local
        self.local_search_interval = local_search_interval
        self.local_search_iterations = local_search_iterations

        # Para plotagem e análise do comportamento
        self.temperature_history = []
        self.cost_history = []

    def local_search(self, solution):
        """
        Aplica uma busca local simples (hill climbing) usando a estratégia 'smart'
        para um número definido de iterações.
        """
        best_sol = copy.deepcopy(solution)
        best_cost = calculate_cost(self.instance_data, best_sol)
        for _ in range(self.local_search_iterations):
            candidate = make_neighbor_solution(self.instance_data, best_sol, strategy="smart")
            candidate_cost = calculate_cost(self.instance_data, candidate)
            if candidate_cost < best_cost:
                best_sol = candidate
                best_cost = candidate_cost
        return best_sol

    def run(self, max_time=None):
        start_time = time.time()
        while self.temperature > self.min_temp:
            # Check for timeout
            if max_time is not None and time.time() - start_time > max_time:
                print(f"Timeout reached at iteration {self.iterations}.")
                break

            self.temperature_history.append(self.temperature)
            current_cost = calculate_cost(self.instance_data, self.current_solution)
            self.cost_history.append(current_cost)

            new_solution = make_neighbor_solution(self.instance_data, self.current_solution)
            new_cost = calculate_cost(self.instance_data, new_solution)

            if new_cost < current_cost or random.uniform(0, 1) < math.exp((current_cost - new_cost) / self.temperature):
                self.current_solution = new_solution
                if new_cost < calculate_cost(self.instance_data, self.best_solution):
                    self.best_solution = copy.deepcopy(new_solution)

            if self.iterations % self.local_search_interval == 0:
                improved_solution = self.local_search(self.current_solution)
                improved_cost = calculate_cost(self.instance_data, improved_solution)
                if improved_cost < current_cost:
                    self.current_solution = improved_solution
                    if improved_cost < calculate_cost(self.instance_data, self.best_solution):
                        self.best_solution = copy.deepcopy(improved_solution)

            self.temperature *= self.cooling_rate
            self.iterations += 1

        self.runtime = time.time() - start_time
        self.final_cost = calculate_cost(self.instance_data, self.best_solution)
        return self.best_solution
