import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from parser import parse_instance_file
from utils import calculate_cost

def get_scheduler(algorithm, instance_data):
    """
    Returns the scheduler instance based on the chosen algorithm.
    Available algorithms: 'sa', 'hybrid_sa', 'tabu', 'ga', 'hc', 'rw'.
    """
    if algorithm == "sa":
        from schedulers.simulated import SimulatedAnnealingScheduler
        return SimulatedAnnealingScheduler(instance_data)
    elif algorithm == "hybrid_sa":
        from schedulers.hybrid_simulated import HybridSimulatedAnnealingScheduler
        return HybridSimulatedAnnealingScheduler(instance_data)
    elif algorithm == "tabu":
        from schedulers.tabu import TabuSearchScheduler
        return TabuSearchScheduler(instance_data)
    elif algorithm == "ga":
        from schedulers.genetic import GeneticAlgorithmScheduler
        return GeneticAlgorithmScheduler(instance_data)
    elif algorithm == "hc":
        from schedulers.hill_climbing import HillClimbingScheduler
        return HillClimbingScheduler(instance_data)
    elif algorithm == "rw":
        from schedulers.random_walk import RandomWalkScheduler
        return RandomWalkScheduler(instance_data)
    else:
        raise ValueError("Unknown algorithm. Use 'sa', 'hybrid_sa', 'tabu', 'ga', 'hc', or 'rw'.")

def analyze_cost_components(instance_data, solution):
    """
    Analyzes and breaks down the total cost into its components.
    Returns a dictionary with the values of each component.
    """
    # Simply calls the unified function
    return calculate_cost(instance_data, solution, use_same_weights=True, return_components=True)

def generate_visualization(scheduler, instance_name, instance_data, output_dir, algorithm="ga"):
    """
    Generates visualizations for the scheduling algorithm.
    Works with GA, HC, RW, SA, and Tabu.
    """
    # Initial setup
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.style.use('ggplot')
    
    # 1. Cost evolution visualization
    generate_cost_evolution_visualization(scheduler, instance_name, output_dir, algorithm)
    
    # 2. Ward occupancy visualization
    generate_ward_occupancy_visualization(scheduler, instance_name, instance_data, output_dir)
    
    # 3. Cost breakdown visualization
    cost_components = analyze_cost_components(instance_data, scheduler.best_solution)
    generate_cost_breakdown_visualization(cost_components, instance_name, output_dir)
    
    # 4. Surgeries per day visualization
    generate_surgeries_visualization(scheduler, instance_name, instance_data, output_dir)
    
    # 5. OT utilization visualization
    generate_ot_utilization_visualization(scheduler, instance_name, instance_data, output_dir)
    
    # Additional algorithm-specific visualizations
    if algorithm == "sa":
        generate_temperature_visualization(scheduler, instance_name, output_dir)
    elif algorithm == "ga":
        # If the GA scheduler has these attributes
        if hasattr(scheduler, 'diversity_history') and scheduler.diversity_history:
            generate_diversity_visualization(scheduler, instance_name, output_dir)
        
        if hasattr(scheduler, 'cost_history') and len(scheduler.cost_history) > 1:
            generate_improvements_visualization(scheduler, instance_name, output_dir)

def generate_cost_evolution_visualization(scheduler, instance_name, output_dir, algorithm="ga"):
    """Generates visualization showing cost evolution."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Primary axis for cost
    color = 'tab:blue'
    if algorithm == "ga":
        x_label = 'Generation'
    else:
        x_label = 'Iteration'
        
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Cost', color=color)
    ax1.plot(scheduler.cost_history, color=color, linewidth=2, label='Cost')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add algorithm-specific metrics if available
    if algorithm == "ga" and hasattr(scheduler, 'diversity_history') and scheduler.diversity_history:
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Diversity/Mutation Rate', color=color)
        
        # Plot diversity
        ax2.plot(scheduler.diversity_history, color=color, linestyle='-', alpha=0.7, label='Diversity')
        
        # Plot mutation rate if available
        if hasattr(scheduler, 'mutation_rate_history') and scheduler.mutation_rate_history:
            ax2.plot(scheduler.mutation_rate_history, color='tab:green', linestyle='--', alpha=0.7, label='Mutation Rate')
        
        ax2.set_ylim(0, 1.1)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend()
    
    plt.title(f'Cost Evolution - {algorithm.upper()} - {instance_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{algorithm}_cost_evolution.png'), dpi=300)
    plt.close()

def generate_temperature_visualization(scheduler, instance_name, output_dir):
    """Generates visualization of temperature evolution for Simulated Annealing."""
    if hasattr(scheduler, 'temperature_history') and scheduler.temperature_history:
        plt.figure(figsize=(12, 6))
        plt.plot(scheduler.temperature_history, color='tab:orange', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Temperature')
        plt.title(f'Temperature Evolution - SA - {instance_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sa_temperature.png'), dpi=300)
        plt.close()

def generate_diversity_visualization(scheduler, instance_name, output_dir):
    """Generates visualization of population diversity for Genetic Algorithm."""
    plt.figure(figsize=(12, 6))
    plt.plot(scheduler.diversity_history, color='tab:red', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.title(f'Population Diversity - GA - {instance_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ga_diversity.png'), dpi=300)
    plt.close()

def generate_ward_occupancy_visualization(scheduler, instance_name, instance_data, output_dir):
    """Generates visualization of ward occupancy."""
    # Calculate occupancy by day for each ward
    days = instance_data['days']
    wards = list(instance_data['wards'].keys())
    
    # Initialize occupancy matrix
    daily_occupancy = {ward: [0] * days for ward in wards}
    
    # Calculate occupancy by day
    for patient, data in scheduler.best_solution.items():
        if data['ward'] is None or data['day'] < 0:
            continue
            
        ward = data['ward']
        admission_day = data['day']
        length_of_stay = instance_data['patients'][patient]['length_of_stay']
        
        for day_offset in range(length_of_stay):
            day = admission_day + day_offset
            if day < days:
                daily_occupancy[ward][day] += 1
    
    # Create heatmap
    plt.figure(figsize=(12, 6))
    
    # Prepare data
    heatmap_data = []
    for ward in wards:
        for day in range(days):
            occupancy = daily_occupancy[ward][day]
            capacity = instance_data['wards'][ward]['bed_capacity']
            
            heatmap_data.append({
                'Ward': ward,
                'Day': f'Day {day+1}',
                'Occupancy': occupancy,
                'Capacity': capacity,
                'OccupancyRate': occupancy / capacity if capacity > 0 else 0
            })
    
    df_heatmap = pd.DataFrame(heatmap_data)
    
    # Create pivot table for the heatmap
    pivot = df_heatmap.pivot(index='Ward', columns='Day', values='Occupancy')
    
    # Generate absolute occupancy heatmap
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu', 
               linewidths=0.5)
    
    plt.title(f'Ward Occupancy - {instance_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ward_occupancy_heatmap.png'), dpi=300)
    plt.close()
    
    # Generate occupancy rate heatmap
    plt.figure(figsize=(12, 6))
    pivot_rate = df_heatmap.pivot(index='Ward', columns='Day', values='OccupancyRate')
    
    # Create custom colormap to highlight overload
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    
    sns.heatmap(pivot_rate, annot=True, fmt='.0%', cmap=cmap, 
               linewidths=0.5, vmin=0, vmax=1.2, center=0.6)
    
    plt.title(f'Ward Occupancy Rate - {instance_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ward_occupancy_rate_heatmap.png'), dpi=300)
    plt.close()

def generate_cost_breakdown_visualization(cost_components, instance_name, output_dir):
    """Generates visualization showing the cost breakdown."""
    # Data for the pie chart
    labels = ['Bed Capacity', 'Surgery Conflicts', 'Admission Delay', 'OT Usage']
    values = [
        cost_components.get('bed_capacity_cost', 0), 
        cost_components.get('surgery_conflict_cost', 0),
        cost_components.get('delay_cost', 0),
        cost_components.get('ot_cost', 0)
    ]
    
    # Check consistency of total values
    total = sum(values)
    reported_total = cost_components.get('total_cost', total)
    
    if abs(total - reported_total) > 0.1:
        print(f"WARNING: Discrepancy in total cost. Visualization: {total}, Reported: {reported_total}")
        # Adjust values to match reported total cost
        scaling_factor = reported_total / total if total > 0 else 1
        values = [v * scaling_factor for v in values]
    
    # Remove components with zero value
    non_zero_labels = []
    non_zero_values = []
    for label, value in zip(labels, values):
        if value > 0:
            non_zero_labels.append(label)
            non_zero_values.append(value)
    
    if not non_zero_values:  # If there are no positive values
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "All cost components are zero", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'cost_breakdown.png'), dpi=300)
        plt.close()
        return
    
    # Colors
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    colors = colors[:len(non_zero_labels)]
    
    # Generate pie chart
    fig, ax = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax.pie(
        non_zero_values, 
        labels=non_zero_labels, 
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )
    
    # Configure text properties
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('black')
    
    ax.axis('equal')  # Ensure the chart is a circle
    plt.title(f'Cost Breakdown - {instance_name}', fontsize=14)
    
    # Add legend with absolute values
    legend_labels = []
    for label, value in zip(non_zero_labels, non_zero_values):
        legend_labels.append(f'{label}: {value:.1f}')
    
    if legend_labels:
        plt.legend(wedges, legend_labels, title="Cost Components", 
                 loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_breakdown.png'), dpi=300)
    plt.close()

def generate_surgeries_visualization(scheduler, instance_name, instance_data, output_dir):
    """Generates visualization of scheduled surgeries."""
    days = instance_data['days']
    
    # Count surgeries by day
    surgeries_by_day = [0] * days
    for patient, data in scheduler.best_solution.items():
        if data['ward'] is not None and data['day'] >= 0 and data['day'] < days:
            surgeries_by_day[data['day']] += 1
    
    # Count surgeries by specialization and day
    spec_surgeries = {}
    for patient, data in scheduler.best_solution.items():
        if data['ward'] is not None and data['day'] >= 0 and data['day'] < days:
            spec = instance_data['patients'][patient]['specialization']
            if spec not in spec_surgeries:
                spec_surgeries[spec] = [0] * days
            spec_surgeries[spec][data['day']] += 1
    
    # Visualization: Total surgeries by day
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(1, days+1), surgeries_by_day, color='#1f77b4')
    
    # Add line for surgery capacity
    capacity = len(instance_data['specializations'])
    plt.axhline(y=capacity, color='red', linestyle='--', label=f'Capacity ({capacity})')
    
    plt.title(f'Surgeries Scheduled by Day - {instance_name}', fontsize=14)
    plt.xlabel('Day')
    plt.ylabel('Number of Surgeries')
    plt.xticks(range(1, days+1))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add labels
    for i, count in enumerate(surgeries_by_day):
        color = 'red' if count > capacity else 'black'
        plt.text(i+1, count, str(count), ha='center', va='bottom', color=color)
    
    plt.savefig(os.path.join(output_dir, 'surgeries_by_day.png'), dpi=300)
    plt.close()

def generate_improvements_visualization(scheduler, instance_name, output_dir):
    """Generates visualization of improvements by generation."""
    if len(scheduler.cost_history) > 1:
        # Calculate absolute improvements
        improvements = [0]
        for i in range(1, len(scheduler.cost_history)):
            improvement = max(0, scheduler.cost_history[i-1] - scheduler.cost_history[i])
            improvements.append(improvement)
        
        # Calculate relative improvements (%)
        relative_improvements = [0]
        for i in range(1, len(scheduler.cost_history)):
            if scheduler.cost_history[i-1] > 0:  # Avoid division by zero
                rel_improvement = 100 * (scheduler.cost_history[i-1] - scheduler.cost_history[i]) / scheduler.cost_history[i-1]
                relative_improvements.append(rel_improvement)
            else:
                relative_improvements.append(0)
        
        # Absolute improvements graph
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(range(len(improvements)), improvements, color='#2ca02c', alpha=0.7)
        plt.xlabel('Generation')
        plt.ylabel('Absolute Improvement')
        plt.title(f'Absolute Improvement per Generation - {instance_name}', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Relative improvements graph
        plt.subplot(1, 2, 2)
        plt.bar(range(len(relative_improvements)), relative_improvements, color='#d62728', alpha=0.7)
        plt.xlabel('Generation')
        plt.ylabel('Relative Improvement (%)')
        plt.title(f'Relative Improvement per Generation - {instance_name}', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'improvements.png'), dpi=300)
        plt.close()

def generate_ot_utilization_visualization(scheduler, instance_name, instance_data, output_dir):
    """Generates visualization of operating time (OT) utilization."""
    days = instance_data['days']
    specs = list(instance_data['specializations'].keys())
    
    # Calculate OT usage by day and specialization
    ot_usage = {spec: [0] * days for spec in specs}
    ot_available = {spec: [0] * days for spec in specs}
    
    for patient, data in scheduler.best_solution.items():
        if data['ward'] is None or data['day'] < 0:
            continue
            
        spec = instance_data['patients'][patient]['specialization']
        day = data['day']
        
        if day < days and spec in ot_usage:
            surgery_duration = instance_data['patients'][patient]['surgery_duration']
            ot_usage[spec][day] += surgery_duration
    
    # Get available OT
    for spec in specs:
        for day in range(min(len(instance_data['specializations'][spec]['available_ot']), days)):
            ot_available[spec][day] = instance_data['specializations'][spec]['available_ot'][day]
    
    # Create aggregate OT utilization visualization
    plt.figure(figsize=(12, 6))
    
    # Calculate utilization rate by day
    utilization_rates = []
    
    for day in range(days):
        day_usage = sum(ot_usage[spec][day] for spec in specs)
        day_available = sum(ot_available[spec][day] for spec in specs if day < len(ot_available[spec]))
        
        if day_available > 0:
            utilization_rates.append(day_usage / day_available * 100)
        else:
            utilization_rates.append(0)
    
    # Create bars
    bars = plt.bar(range(1, days+1), utilization_rates, color='skyblue')
    
    # Add 100% line
    plt.axhline(y=100, color='red', linestyle='--', label='Ideal Utilization')
    
    # Color bars based on rate
    for i, rate in enumerate(utilization_rates):
        if rate > 110:  # Significant overtime
            bars[i].set_color('crimson')
        elif rate > 95:  # Near ideal
            bars[i].set_color('limegreen')
        elif rate < 70:  # Significant undertime
            bars[i].set_color('orange')
    
    plt.xlabel('Day')
    plt.ylabel('Utilization Rate (%)')
    plt.title(f'Operating Time Utilization by Day - {instance_name}', fontsize=14)
    plt.xticks(range(1, days+1))
    plt.grid(axis='y', alpha=0.3)
    
    # Add labels
    for i, rate in enumerate(utilization_rates):
        plt.text(i+1, rate+2, f"{rate:.1f}%", ha='center')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ot_utilization_rate.png'), dpi=300)
    plt.close()
    
    # Create individual visualizations per specialization
    for spec in specs:
        plt.figure(figsize=(10, 6))
        
        # Prepare data for bars
        days_range = range(1, min(days+1, len(ot_available[spec])+1))
        usage_values = [ot_usage[spec][d-1] for d in days_range]
        available_values = [ot_available[spec][d-1] for d in days_range]
        
        # Calculate utilization rate
        util_rates = [usage/avail*100 if avail > 0 else 0 for usage, avail in zip(usage_values, available_values)]
        
        # Plot utilization vs availability bars
        x = np.arange(len(days_range))
        width = 0.35
        
        plt.bar(x - width/2, usage_values, width, label='OT Used', color='skyblue')
        plt.bar(x + width/2, available_values, width, label='OT Available', color='lightgray')
        
        # Add utilization rate line
        ax2 = plt.twinx()
        ax2.plot(x, util_rates, 'r-', label='Utilization Rate', linewidth=2)
        ax2.set_ylim(0, max(max(util_rates)*1.1, 110))
        ax2.set_ylabel('Utilization Rate (%)')
        
        # Configure axes and labels
        plt.xlabel('Day')
        plt.ylabel('Operating Time (minutes)')
        plt.title(f'OT Utilization - {spec} - {instance_name}', fontsize=14)
        plt.xticks(x, days_range)
        
        # Add second legend
        lines, labels = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'ot_utilization_{spec}.png'), dpi=300)
        plt.close()

def main():
    optimization_method = "ga"  # Change to "sa", "tabu", "ga", "hc", or "rw" as desired

    # Define base directory for results
    results_dir = "./results"
    # Create directory for chosen algorithm, e.g.: ./results/ga
    algorithm_results_dir = os.path.join(results_dir, optimization_method)
    if not os.path.exists(algorithm_results_dir):
        os.makedirs(algorithm_results_dir)

    instance_dir = "./data/instances"
    schedule_results = []  # Store results DataFrames
    metrics_list = []      # Store performance metrics
    images_base_dir = "./images"
    if not os.path.exists(images_base_dir):
        os.makedirs(images_base_dir)

    # Process each instance file
    for filename in os.listdir(instance_dir):
        if filename.endswith(".dat"):
            filepath = os.path.join(instance_dir, filename)
            print(f"Processing instance: {filename}")
            instance_data = parse_instance_file(filepath)

            scheduler = get_scheduler(optimization_method, instance_data)
            best_schedule = scheduler.run()
            
            # Calculate final cost using calculate_cost function
            # Use use_same_weights=True to ensure consistency with analyze_cost_components
            final_cost = calculate_cost(instance_data, best_schedule, use_same_weights=True)

            df_results = pd.DataFrame(best_schedule).T.reset_index()
            df_results.rename(columns={'index': 'patient'}, inplace=True)
            df_results['instance'] = filename
            schedule_results.append(df_results)

            total_patients = len(instance_data['patients'])
            allocated = sum(1 for patient in best_schedule if best_schedule[patient]['ward'] is not None)
            pct_allocated = (allocated / total_patients) * 100

            # Determine algorithm-specific iteration term
            if optimization_method == "ga":
                iterations_or_generations = getattr(scheduler, "generations", "N/A")
                iteration_term = "generations"
            else:
                iterations_or_generations = getattr(scheduler, "iterations", "N/A")
                iteration_term = "iterations"

            instance_metrics = {
                "instance": filename,
                iteration_term: iterations_or_generations,
                "runtime_seconds": getattr(scheduler, "runtime", "N/A"),
                "final_cost": final_cost,
                "pct_allocated": pct_allocated
            }
            metrics_list.append(instance_metrics)

            # Create directory for instance plots
            instance_name = os.path.splitext(filename)[0]
            instance_image_dir = os.path.join(images_base_dir, instance_name)
            if not os.path.exists(instance_image_dir):
                os.makedirs(instance_image_dir)

            # Generate visualizations based on algorithm
            generate_visualization(scheduler, instance_name, instance_data, instance_image_dir, optimization_method)
            
            # Analyze and display cost components
            cost_components = analyze_cost_components(instance_data, best_schedule)
            print("\nFinal cost components:")
            print(f"  Bed Capacity: {cost_components.get('bed_capacity_cost', 0):.2f}")
            print(f"  Surgery Conflicts: {cost_components.get('surgery_conflict_cost', 0):.2f}")
            print(f"  Admission Delay: {cost_components.get('delay_cost', 0):.2f}")
            print(f"  OT Usage: {cost_components.get('ot_cost', 0):.2f}")
            print(f"  Total: {cost_components.get('total_cost', 0):.2f}")

            print(df_results)
            print("\n" + "="*50 + "\n")

    # Save combined results and metrics
    all_schedules_df = pd.concat(schedule_results, ignore_index=True)
    all_schedules_df.to_csv(os.path.join(algorithm_results_dir, "best_schedules.csv"), index=False)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(os.path.join(algorithm_results_dir, "metrics_results.csv"), index=False)

    # Aggregate graph: Histogram of final costs
    plt.figure(figsize=(8, 6))
    plt.hist(metrics_df["final_cost"], bins=20, edgecolor='black')
    plt.xlabel("Final Cost")
    plt.ylabel("Frequency")
    plt.title("Distribution of Final Costs Across Instances")
    plt.grid()
    plt.savefig(os.path.join(algorithm_results_dir, "final_cost_histogram.png"))
    plt.close()

    # Aggregate graph: Final Cost vs Allocation Percentage
    plt.figure(figsize=(8, 6))
    plt.scatter(metrics_df["pct_allocated"], metrics_df["final_cost"])
    plt.xlabel("Allocation Percentage")
    plt.ylabel("Final Cost")
    plt.title("Final Cost vs Allocation Percentage")
    plt.grid()
    plt.savefig(os.path.join(algorithm_results_dir, "final_cost_vs_allocation.png"))
    plt.close()

    print("Processing complete. Results saved in 'best_schedules.csv', 'metrics_results.csv',")
    print("individual plots in './images' folder and aggregate graphs in", algorithm_results_dir)

if __name__ == "__main__":
    main()