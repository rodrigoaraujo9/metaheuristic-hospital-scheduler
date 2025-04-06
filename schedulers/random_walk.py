import random
import copy
import time
from utils import calculate_cost, generate_initial_solution, make_neighbor_solution

class RandomWalkScheduler:
    def __init__(self, instance_data, iterations=2000):
        self.instance_data = instance_data
        self.iterations = iterations
        
        # Metrics for visualization and display
        self.runtime = 0
        self.best_solution = None
        self.best_cost = float('inf')
        self.cost_history = []
        self.final_cost = None
        
        print("Initializing Random Walk scheduler...")
    
    def generate_smart_initial_solution(self):
        """Generate a smarter initial solution using the utility function."""
        return generate_initial_solution(self.instance_data)
    
    def run(self, max_time=None):
        """Main random walk algorithm engine."""
        print(f"Starting Random Walk algorithm for {self.iterations} iterations")
        start_time = time.time()
        
        current_solution = self.generate_smart_initial_solution()
        current_cost = calculate_cost(self.instance_data, current_solution)
        
        self.best_solution = copy.deepcopy(current_solution)
        self.best_cost = current_cost
        self.cost_history.append(current_cost)
        
        strategies = ["ward", "day", "both", "smart", "ot_balancing"]
        
        for i in range(self.iterations):
            if max_time is not None and time.time() - start_time > max_time:
                print(f"Timeout reached at iteration {i+1}.")
                break

            strategy = random.choice(strategies)
            neighbor = make_neighbor_solution(self.instance_data, current_solution, strategy=strategy)
            neighbor_cost = calculate_cost(self.instance_data, neighbor)
            
            current_solution = copy.deepcopy(neighbor)
            current_cost = neighbor_cost
            
            if current_cost < self.best_cost:
                self.best_solution = copy.deepcopy(current_solution)
                self.best_cost = current_cost
                print(f"Iteration {i+1}: New best solution found with cost {self.best_cost:.2f}")
            
            self.cost_history.append(self.best_cost)
            
            if (i+1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {i+1}/{self.iterations} - Best: {self.best_cost:.2f} - Time: {elapsed:.2f}s")
                if (i+1) % 200 == 0:
                    for patient in current_solution:
                        if random.random() < 0.3:
                            earliest = self.instance_data['patients'][patient]['earliest_admission']
                            current_solution[patient]['day'] = earliest
                    current_cost = calculate_cost(self.instance_data, current_solution)
                    if current_cost < self.best_cost:
                        self.best_solution = copy.deepcopy(current_solution)
                        self.best_cost = current_cost
                        print(f"Guided move: New best solution found with cost {self.best_cost:.2f}")
        
        self.runtime = time.time() - start_time
        self.final_cost = self.best_cost
        print(f"Random Walk completed in {self.runtime:.2f} seconds")
        print(f"Best cost found: {self.best_cost:.2f}")
        return self.best_solution
