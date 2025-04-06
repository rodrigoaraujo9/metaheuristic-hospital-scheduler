import time
import random
import copy
from utils import generate_initial_solution, calculate_cost, make_neighbor_solution

class TabuSearchScheduler:
    def __init__(self, instance_data, max_iterations=1000, tabu_list_size=50, neighbor_count=10):
        self.instance_data = instance_data
        self.max_iterations = max_iterations
        self.tabu_list_size = tabu_list_size
        self.neighbor_count = neighbor_count
        self.current_solution = generate_initial_solution(instance_data)
        self.best_solution = copy.deepcopy(self.current_solution)
        self.iterations = 0
        self.runtime = 0
        self.final_cost = None

        # Para monitoramento
        self.cost_history = []
        self.tabu_list = []

    def run(self, max_time=None):
        start_time = time.time()
        for _ in range(self.max_iterations):
            if max_time is not None and time.time() - start_time > max_time:
                print(f"Timeout reached at iteration {self.iterations}.")
                break

            self.iterations += 1
            current_cost = calculate_cost(self.instance_data, self.current_solution)
            self.cost_history.append(current_cost)

            neighbors = [make_neighbor_solution(self.instance_data, self.current_solution)
                        for _ in range(self.neighbor_count)]
            valid_neighbors = []
            for neighbor in neighbors:
                sol_repr = tuple(sorted((p, d['ward'], d['day']) for p, d in neighbor.items()))
                if sol_repr not in self.tabu_list:
                    valid_neighbors.append((neighbor, calculate_cost(self.instance_data, neighbor)))
            if not valid_neighbors:
                continue

            best_neighbor, best_neighbor_cost = min(valid_neighbors, key=lambda x: x[1])
            sol_repr = tuple(sorted((p, d['ward'], d['day']) for p, d in best_neighbor.items()))
            self.tabu_list.append(sol_repr)
            if len(self.tabu_list) > self.tabu_list_size:
                self.tabu_list.pop(0)

            self.current_solution = best_neighbor
            if best_neighbor_cost < calculate_cost(self.instance_data, self.best_solution):
                self.best_solution = copy.deepcopy(best_neighbor)
        
        self.runtime = time.time() - start_time
        self.final_cost = calculate_cost(self.instance_data, self.best_solution)
        return self.best_solution

