# Hospital Scheduling Optimizer

> AI-powered patient admission scheduling using advanced metaheuristic algorithms

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Academic Excellence](https://img.shields.io/badge/FEUP%20AI%20Course-20%2F20-green.svg)](https://sigarra.up.pt/feup/)

A comprehensive optimization suite that tackles the complex challenge of hospital patient admission scheduling through cutting-edge metaheuristic algorithms. This project demonstrates how AI can revolutionize healthcare operations by efficiently balancing resource utilization, minimizing costs, and optimizing patient care delivery.

## Key Features

- **Six Advanced Algorithms**: Genetic Algorithm, Simulated Annealing, Tabu Search, Hill Climbing, Random Walk, and Hybrid approaches
- **Interactive Web Interface**: Real-time visualization and parameter tuning via Streamlit
- **Multi-Objective Optimization**: Balances patient delays, resource utilization, and operational costs
- **Comprehensive Analytics**: Rich visualizations including cost evolution, ward occupancy heatmaps, and performance metrics
- **Extensible Architecture**: Easy to add new algorithms and problem instances

## Problem Overview

Healthcare scheduling is a complex optimization challenge that involves:

| Challenge | Solution Approach |
|-----------|------------------|
| **Patient Scheduling** | Minimize admission delays while respecting time windows |
| **Resource Management** | Optimize operating theater utilization and bed capacity |
| **Workload Balancing** | Distribute nursing workload across wards and time |
| **Specialty Matching** | Align patients with appropriate ward specializations |

### Patient Attributes
- **Admission Windows**: Earliest to latest possible admission days
- **Specialty Requirements**: Surgical specialty needed
- **Length of Stay**: Duration affecting bed capacity
- **Surgery Duration**: Operating theater time requirements
- **Workload Vector**: Nursing attention requirements

### Ward Constraints
- **Specializations**: Major and minor medical specialties
- **Bed Capacity**: Fixed maximum patient capacity
- **Carryover Patients**: Existing patients affecting availability
- **Workload Limits**: Nursing capacity constraints

## Quick Start

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation

```bash
git clone https://github.com/yourusername/hospital-scheduling-optimizer.git
cd hospital-scheduling-optimizer
pip install -r requirements.txt
```

### Run the Web Application

```bash
streamlit run app.py
```

### Command Line Execution

```bash
python main.py
```

## Optimization Algorithms

### Genetic Algorithm (GA)
**Best for**: Large, complex problems requiring diverse exploration
- Population-based evolutionary search
- Solution crossover and mutation
- Diversity management and elitism

### Simulated Annealing (SA)
**Best for**: Medium complexity with balanced exploration
- Temperature-controlled probabilistic acceptance
- Gradual cooling schedule
- Escape from local optima

### Tabu Search (TS)
**Best for**: Problems with many similar local optima
- Memory-based search with forbidden moves
- Aspiration criteria and intensification
- Efficient navigation of solution space

### Hill Climbing (HC)
**Best for**: Quick solutions and initial exploration
- Greedy local search with restarts
- Fast execution and simple implementation
- Multiple restart mechanism

### Random Walk (RW)
**Best for**: Solution space exploration
- Stochastic neighbor selection
- Occasional guided moves
- Diverse coverage of search space

### Hybrid Simulated Annealing
**Best for**: Complex instances with multiple optima
- Combines SA with local search intensification
- Enhanced exploitation of promising regions
- Maintains exploration capability

## Project Structure

```
hospital-scheduling-optimizer/
├── app.py                      # Interactive Streamlit web app
├── main.py                     # Command-line interface
├── parser.py                   # Instance data parser
├── utils.py                    # Core utilities and evaluation
├── schedulers/                 # Algorithm implementations
│   ├── genetic.py
│   ├── simulated.py
│   ├── tabu.py
│   ├── hill_climbing.py
│   ├── random_walk.py
│   └── hybrid_simulated.py
├── data/instances/             # Problem instances
├── results/                    # Algorithm outputs
├── images/                     # Generated visualizations
└── requirements.txt
```

## Using the Interactive Application

1. **Algorithm Selection**: Choose from six different optimization methods
2. **Instance Selection**: Pick specific problems or run all instances
3. **Parameter Tuning**: Adjust algorithm-specific parameters
4. **Execution**: Click run and watch real-time progress
5. **Analysis**: Explore comprehensive results and visualizations

## Visualization Suite

| Visualization | Purpose |
|---------------|---------|
| **Cost Evolution** | Track optimization progress over time |
| **Cost Breakdown** | Analyze different penalty components |
| **Ward Occupancy Heatmap** | Visualize resource utilization |
| **Surgery Distribution** | Balance surgical scheduling |
| **OT Utilization** | Monitor operating theater efficiency |
| **Algorithm Metrics** | Temperature curves, diversity measures |

## Understanding Results

### Key Performance Indicators
- **Allocation Percentage**: Successfully scheduled patients
- **Final Cost**: Overall solution quality (lower is better)
- **Ward Occupancy**: Resource utilization efficiency
- **Surgery Balance**: Workload distribution across days
- **OT Utilization**: Operating theater efficiency

### Cost Components
- **Delay Penalties**: Patient admission delays
- **OT Overtime**: Surgical time overruns
- **Capacity Violations**: Bed limit breaches
- **Surgery Conflicts**: Day overloading penalties

## Extending the Framework

### Adding New Algorithms

```python
# Create schedulers/your_algorithm.py
class YourAlgorithm:
    def __init__(self, instance):
        self.instance = instance
        
    def run(self):
        # Your optimization logic
        return best_solution
```

### Adding Problem Instances
Simply add `.dat` files to `data/instances/` following the existing format.

## Contributors

- **[Rodrigo Miranda](https://github.com/h0leee)** - up202204916
- **[Eduardo Cunha](https://github.com/educunhA04)** - up202207126  
- **Rodrigo Araújo** - [GitHub Profile](https://github.com/rodrigoaraujo-up) - up202205515

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Rodrigo Miranda, Eduardo Cunha, Rodrigo Araújo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Academic Excellence

This project achieved a **perfect score of 20/20** in the Artificial Intelligence course at [FEUP](https://sigarra.up.pt/feup/) (Faculdade de Engenharia da Universidade do Porto), demonstrating excellence in:
- Algorithm implementation and optimization
- Problem modeling and constraint handling  
- Software engineering and user experience
- Comprehensive analysis and visualization

## Acknowledgments

- Inspired by real-world healthcare operations research challenges
- Implements state-of-the-art metaheuristic optimization techniques
- Built upon established principles in combinatorial optimization
- Developed as part of advanced AI coursework at FEUP

---

**If this project helped you, please give it a star!**

[Report Bug](../../issues) • [Request Feature](../../issues) • [Documentation](../../wiki)
