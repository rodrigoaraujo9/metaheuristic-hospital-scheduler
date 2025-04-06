import random
import copy
import time
from utils import calculate_cost, generate_initial_solution, make_neighbor_solution

class HillClimbingScheduler:
    def __init__(self, instance_data, iterations=1000, restart_after=100):
        self.instance_data = instance_data
        self.iterations = iterations
        self.restart_after = restart_after  # Restart if no improvement after this many iterations
        
        # Metrics for visualization and display
        self.runtime = 0
        self.best_solution = None
        self.best_cost = float('inf')
        self.cost_history = []
        self.final_cost = None
        
        print("Initializing Hill Climbing scheduler...")
    
    def generate_smart_initial_solution(self):
        """Generate a smarter initial solution using the utility function."""
        return generate_initial_solution(self.instance_data)
    
    def run(self):
        """Main hill climbing algorithm engine with restarts."""
        print(f"Starting Hill Climbing algorithm for {self.iterations} iterations")
        start_time = time.time()
        
        # Generate initial solution
        current_solution = self.generate_smart_initial_solution()
        current_cost = calculate_cost(self.instance_data, current_solution)
        
        self.best_solution = copy.deepcopy(current_solution)
        self.best_cost = current_cost
        self.cost_history.append(current_cost)
        
        # Variables to track progress
        iterations_without_improvement = 0
        restart_count = 0
        
        for i in range(self.iterations):
            # Generate a neighbor solution
            neighbor = make_neighbor_solution(self.instance_data, current_solution, strategy="mixed")
            neighbor_cost = calculate_cost(self.instance_data, neighbor)
            
            # If neighbor is better, move to it
            if neighbor_cost < current_cost:
                current_solution = copy.deepcopy(neighbor)
                current_cost = neighbor_cost
                iterations_without_improvement = 0
                
                # Update best solution if needed
                if current_cost < self.best_cost:
                    self.best_solution = copy.deepcopy(current_solution)
                    self.best_cost = current_cost
                    print(f"Iteration {i+1}: New best solution found with cost {self.best_cost:.2f}")
            else:
                iterations_without_improvement += 1
            
            # Consider restart if stuck
            if iterations_without_improvement >= self.restart_after:
                restart_count += 1
                print(f"Iteration {i+1}: Restarting search (restart #{restart_count})")
                
                # Generate new solution with priority on earliest admission
                new_solution = self.generate_smart_initial_solution()
                new_cost = calculate_cost(self.instance_data, new_solution)
                
                # Always move to the new solution for diversity
                current_solution = copy.deepcopy(new_solution)
                current_cost = new_cost
                iterations_without_improvement = 0
            
            # Record cost history for visualization
            self.cost_history.append(self.best_cost)
            
            # Progress reporting
            if (i+1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {i+1}/{self.iterations} - Best: {self.best_cost:.2f} - Time: {elapsed:.2f}s")
        
        self.runtime = time.time() - start_time
        self.final_cost = self.best_cost
        
        print(f"Hill Climbing completed in {self.runtime:.2f} seconds")
        print(f"Best cost found: {self.best_cost:.2f}")
        
        return self.best_solution