# Hospital Scheduling Optimization

This project implements and compares different optimization algorithms for patient admission scheduling in hospitals, focusing on efficient resource utilization and cost minimization.

## Requirements

- Python 3.8 or higher
- Required packages: listed in `requirements.txt`

## Installation

1. Clone this repository or unzip the provided files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

- `main.py`: Entry point for command-line execution
- `app.py`: Interactive Streamlit web application
- `parser.py`: Functions for parsing instance data files
- `utils.py`: Utility functions for solution generation and evaluation
- `schedulers/`: Folder containing all optimization algorithms:
  - `genetic.py`: Genetic Algorithm implementation
  - `hillclimbing.py`: Hill Climbing implementation
  - `randomwalk.py`: Random Walk implementation
  - `simulated.py`: Simulated Annealing implementation (if available)
  - `tabu.py`: Tabu Search implementation (if available)
- `data/instances/`: Folder containing problem instances
- `results/`: Folder where results are saved
- `images/`: Folder where visualization images are saved

## Running the Program

### Command-Line Interface

To run the optimization with default settings:

```bash
python main.py
```

By default, this will use the Genetic Algorithm. To change the algorithm, modify the `optimization_method` variable in `main.py`.

### Interactive Web Application

To run the interactive Streamlit application:

```bash
streamlit run app.py
```

This will open a web browser with the application interface where you can:
1. Select the optimization algorithm (GA, Hill Climbing, Random Walk, etc.)
2. Choose which problem instances to solve
3. Set algorithm-specific parameters
4. Visualize and analyze the results

## Using the Application

1. In the sidebar, select your preferred algorithm:
   - `ga`: Genetic Algorithm
   - `hc`: Hill Climbing
   - `rw`: Random Walk
   - `sa`: Simulated Annealing (if implemented)
   - `tabu`: Tabu Search (if implemented)

2. Select one or more problem instances to solve, or check "Use all instances"

3. Adjust algorithm-specific parameters (if available):
   - For Genetic Algorithm: population size, generations, etc.
   - For Hill Climbing: iterations, restart threshold
   - For Random Walk: iterations

4. Click the "Run" button to start the optimization

5. View results:
   - General statistics (runtime, cost, allocation percentage)
   - Solution visualizations (ward occupancy, surgeries per day, etc.)
   - Cost breakdown analysis
   - Algorithm performance metrics

## Evaluating Solutions

Solutions are evaluated based on multiple criteria, including:
- Bed capacity constraints
- Surgery scheduling conflicts
- Operating time utilization
- Patient admission delays

Lower cost values indicate better solutions.

## Extending the Project

To add a new algorithm:
1. Create a new file in the `schedulers/` directory
2. Implement a class with at least `__init__` and `run()` methods
3. Ensure the class tracks metrics like `runtime`, `best_solution`, and `cost_history`
4. Update the `get_scheduler()` function in `main.py` and `app.py`