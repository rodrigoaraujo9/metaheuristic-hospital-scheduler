# Hospital Patient Admission Scheduling Optimization

This project implements and compares different metaheuristic algorithms for patient admission scheduling in hospitals, focusing on efficient resource utilization, cost minimization, and workload balancing across hospital wards.

## Problem Overview

Hospital scheduling involves balancing multiple competing objectives:
- Minimizing patient admission delays
- Optimizing operating theater (OT) utilization
- Respecting bed capacity constraints
- Balancing workload across wards and over time
- Matching patients with appropriate wards based on specialization

Each patient has:
- An admission time window (earliest to latest day)
- A surgical specialty requirement
- A length of stay (LOS) that occupies bed capacity
- A surgery duration (in minutes)
- A workload vector representing required nursing attention

Each ward has:
- A major specialization and optional minor specializations
- A fixed bed capacity
- Potential carryover patients already present
- Workload capacity limitations

## Requirements

- Python 3.8 or higher
- Required packages:
  - streamlit
  - matplotlib
  - seaborn
  - pandas
  - numpy
  - (All detailed in `requirements.txt`)

## Installation

1. Clone this repository or unzip the provided files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

- `main.py`: Entry point for command-line execution with visualization generation
- `app.py`: Interactive Streamlit web application
- `parser.py`: Functions for parsing instance data files (.dat format)
- `utils.py`: Utility functions for solution generation, evaluation, and neighborhood creation
- `schedulers/`: Folder containing all optimization algorithms:
  - `genetic.py`: Genetic Algorithm implementation
  - `hill_climbing.py`: Hill Climbing with restart mechanism
  - `random_walk.py`: Random Walk with occasional guided moves
  - `simulated.py`: Simulated Annealing implementation
  - `hybrid_simulated.py`: Enhanced SA with local search intensification
  - `tabu.py`: Tabu Search implementation
- `data/instances/`: Folder containing problem instances
- `results/`: Folder where algorithm results are saved
- `images/`: Folder where visualization images are saved

## Running the Program

### Command-Line Interface

To run the optimization with default settings:

```bash
python main.py
```

By default, this will use the Genetic Algorithm. To change the algorithm, modify the `optimization_method` variable in `main.py`.

The command-line interface will:
1. Process all instance files in the `data/instances` directory
2. Generate visualizations in the `images` directory
3. Save results to CSV files in the `results` directory
4. Display summary statistics in the console

### Interactive Web Application

To run the interactive Streamlit application:

```bash
streamlit run app.py
```

This will open a web browser with the application interface where you can:
1. Select the optimization algorithm
2. Choose which problem instances to solve
3. Set algorithm-specific parameters
4. Visualize and analyze the results in real-time

## Implemented Algorithms

### Hill Climbing (HC)
- **Approach**: Greedy local search that only accepts improving moves
- **Key Features**: Fast execution, restart mechanism to escape local optima
- **Parameters**: Iterations, restart threshold
- **Best For**: Quick solutions, simple instances, initial exploration

### Random Walk (RW)
- **Approach**: Exploration through random neighbor selection
- **Key Features**: Diverse solution space coverage, occasional guided moves
- **Parameters**: Iterations
- **Best For**: Solution space exploration, finding unexpected solutions

### Simulated Annealing (SA)
- **Approach**: Temperature-based probabilistic acceptance of moves
- **Key Features**: Balances exploration and exploitation, can escape local optima
- **Parameters**: Initial temperature, cooling rate, minimum temperature
- **Best For**: Medium complexity problems, balanced exploration-exploitation

### Hybrid Simulated Annealing (Hybrid SA)
- **Approach**: Combines SA with periodic local search intensification
- **Key Features**: Enhanced exploitation of promising regions while maintaining exploration
- **Parameters**: SA parameters plus local search interval and iterations
- **Best For**: Complex instances with many local optima

### Tabu Search (TS)
- **Approach**: Memory-based search preventing revisits to recent solutions
- **Key Features**: Avoids cycling, efficient navigation of complex spaces
- **Parameters**: Max iterations, tabu list size, number of neighbors
- **Best For**: Problems with many similar local optima

### Genetic Algorithm (GA)
- **Approach**: Population-based evolutionary search
- **Key Features**: Parallel exploration, solution recombination, diversity management
- **Parameters**: Population size, generations, crossover rate, mutation rate
- **Best For**: Large, complex problems requiring wide exploration

## Using the Application

1. In the sidebar, select your preferred algorithm:
   - `ga`: Genetic Algorithm
   - `hc`: Hill Climbing
   - `rw`: Random Walk
   - `sa`: Simulated Annealing
   - `hybrid_sa`: Hybrid Simulated Annealing
   - `tabu`: Tabu Search

2. Select one or more problem instances to solve, or check "Use all instances"

3. Adjust algorithm-specific parameters (if available):
   - For Genetic Algorithm: population size, generations
   - For Hill Climbing: iterations, restart threshold
   - For Random Walk: iterations
   - For SA/Hybrid SA: temperature parameters appear in code (not in UI)
   - For Tabu Search: parameters appear in code (not in UI)

4. Click the "Run" button to start the optimization

5. View results:
   - General statistics (runtime, iterations, allocation percentage, final cost)
   - Cost breakdown pie chart
   - Surgery distribution by day
   - Ward occupancy heatmap
   - Cost evolution chart
   - Algorithm-specific visualizations (e.g., temperature for SA)

## Understanding the Cost Function

Solutions are evaluated based on multiple criteria:

- **Delay Cost**: Penalty for each day a patient is admitted after their earliest possible day
- **OT Overtime Cost**: Penalty when surgeries exceed available operating theater time
- **Bed Capacity Cost**: Penalty for exceeding ward capacity
- **Surgery Conflict Cost**: Penalty for too many surgeries in a single day

Lower total cost values indicate better solutions.

## Visualization Capabilities

The application generates several visualizations to help analyze solutions:

1. **Cost Evolution**: Line chart showing how solution cost changes over iterations
2. **Cost Breakdown**: Pie chart of different cost components
3. **Ward Occupancy**: Heatmap showing bed utilization by ward and day
4. **Surgery Distribution**: Bar chart of surgeries scheduled per day
5. **OT Utilization**: Charts showing operating theater utilization rates
6. **Algorithm-Specific**: Temperature cooling for SA, diversity for GA, etc.

## Example Results Interpretation

- **High Allocation Percentage**: More patients successfully scheduled
- **Low Final Cost**: Better overall solution quality
- **Ward Occupancy Near Capacity**: Efficient resource utilization
- **Balanced Surgery Distribution**: Avoids overloading specific days
- **OT Utilization Near 100%**: Optimal use of operating theater time

## Extending the Project

To add a new algorithm:
1. Create a new file in the `schedulers/` directory
2. Implement a class with at least `__init__` and `run()` methods
3. Ensure the class tracks metrics like `runtime`, `best_solution`, and `cost_history`
4. Update the `get_scheduler()` function in `main.py` and `app.py`

To add new problem instances:
1. Create a .dat file in the `data/instances/` directory following the format of existing instances
2. The parser will automatically handle loading the new instance

## Authors

- Rodrigo Miranda - up202204916
- Eduardo Cunha - up202207126
- Rodrigo Araújo - up202205515

## License

This project is licensed under [Your License] - see the LICENSE file for details.

## Acknowledgments

- Based on the hospital patient scheduling optimization problem in healthcare operations research
- Implements various metaheuristic algorithms for combinatorial optimization