import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # For percentile calculations

from parser import parse_instance_file
from utils import analyze_solution, calculate_cost

st.set_page_config(page_title="Hospital Scheduling", layout="wide")

def compute_average_occupancy(stats):
    """
    Computes the average occupancy rate (as a fraction) across all wards and days.
    stats['ward_occupancy'] is expected to be a dictionary:
       ward -> { 'capacity': X, 'daily_occupancy': [occ1, occ2, ...] }
    """
    total_rate = 0
    count = 0
    for ward, occupancy in stats['ward_occupancy'].items():
        capacity = occupancy['capacity']
        for occ in occupancy['daily_occupancy']:
            rate = occ / capacity if capacity > 0 else 0
            total_rate += rate
            count += 1
    return total_rate / count if count > 0 else 0

def compute_outlier_percentage(stats):
    """
    Computes the percentage of outliers in occupancy rates using the IQR method.
    Outliers are defined as occupancy rates that fall outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    Returns the percentage (0 to 100) of occupancy values that are considered outliers.
    """
    rates = []
    for ward, occupancy in stats['ward_occupancy'].items():
        capacity = occupancy['capacity']
        for occ in occupancy['daily_occupancy']:
            rate = occ / capacity if capacity > 0 else 0
            rates.append(rate)
    if not rates:
        return 0
    total = len(rates)
    q1 = np.percentile(rates, 25)
    q3 = np.percentile(rates, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [r for r in rates if r < lower_bound or r > upper_bound]
    return (len(outliers) / total) * 100

def get_scheduler(algorithm, instance_data, params=None):
    """
    Returns the scheduler instance based on the chosen algorithm and optional parameters.
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
        if params:
            return GeneticAlgorithmScheduler(
                instance_data,
                population_size=params.get("population_size", 150),
                generations=params.get("generations", 100)
            )
        else:
            return GeneticAlgorithmScheduler(instance_data)
    elif algorithm == "hc":
        from schedulers.hill_climbing import HillClimbingScheduler
        if params:
            return HillClimbingScheduler(
                instance_data,
                iterations=params.get("iterations", 1000),
                restart_after=params.get("restart_after", 100)
            )
        else:
            return HillClimbingScheduler(instance_data)
    elif algorithm == "rw":
        from schedulers.random_walk import RandomWalkScheduler
        if params:
            return RandomWalkScheduler(
                instance_data,
                iterations=params.get("iterations", 2000)
            )
        else:
            return RandomWalkScheduler(instance_data)
    else:
        raise ValueError("Unknown algorithm. Use 'sa', 'hybrid_sa', 'tabu', 'ga', 'hc', or 'rw'.")

def save_visualizations_to_folder(scheduler, instance_data, output_folder, instance_name, algorithm):
    """
    Saves visualization images to a 'visualizations' subfolder within the algorithm folder.
    
    Structure:
    tests/
    ├── s0m3/
    │   ├── ga/
    │   │   ├── final_solution.csv
    │   │   ├── metrics.csv
    │   │   └── visualizations/
    │   │       └── [images]
    │   └── ...
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # Create visualizations subfolder
    viz_folder = os.path.join(output_folder, "visualizations")
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)
    
    # 1. Cost evolution visualization
    plt.figure(figsize=(10, 6))
    plt.plot(scheduler.cost_history, color='tab:blue', linewidth=2)
    
    if algorithm == "ga":
        x_label = 'Generation'
    else:
        x_label = 'Iteration'
        
    plt.xlabel(x_label)
    plt.ylabel('Cost')
    plt.title(f'Cost Evolution - {algorithm.upper()} - {instance_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_folder, f'cost_evolution.png'), dpi=300)
    plt.close()
    
    # 2. Temperature Evolution (for SA)
    if algorithm == "sa" and hasattr(scheduler, 'temperature_history'):
        plt.figure(figsize=(10, 6))
        plt.plot(scheduler.temperature_history, color='tab:orange', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Temperature')
        plt.title(f'Temperature Evolution - SA - {instance_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_folder, 'temperature_evolution.png'), dpi=300)
        plt.close()
    
    # 3. Ward occupancy heatmap
    # Get stats for visualization
    from utils import analyze_solution
    stats = analyze_solution(instance_data, scheduler.best_solution)
    
    # Ward occupancy visualization
    ward_data = []
    for ward, occupancy in stats['ward_occupancy'].items():
        capacity = occupancy['capacity']
        for day, occ in enumerate(occupancy['daily_occupancy']):
            ward_data.append({
                'Ward': ward,
                'Day': f'Day {day+1}',
                'Occupancy': occ,
                'Capacity': capacity,
                'Rate': occ/capacity if capacity > 0 else 0
            })
            
    if ward_data:
        df_ward = pd.DataFrame(ward_data)
        
        # Absolute occupancy
        pivot_abs = df_ward.pivot(index='Ward', columns='Day', values='Occupancy')
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_abs, annot=True, fmt='.0f', cmap='YlGnBu', linewidths=0.5)
        plt.title(f'Ward Occupancy - {instance_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_folder, 'ward_occupancy.png'), dpi=300)
        plt.close()
        
        # Occupancy rate
        pivot_rate = df_ward.pivot(index='Ward', columns='Day', values='Rate')
        plt.figure(figsize=(12, 6))
        cmap = sns.diverging_palette(10, 133, as_cmap=True)
        sns.heatmap(pivot_rate, annot=True, fmt='.0%', cmap=cmap, 
                   linewidths=0.5, vmin=0, vmax=1.2, center=0.6)
        plt.title(f'Ward Occupancy Rate - {instance_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_folder, 'ward_occupancy_rate.png'), dpi=300)
        plt.close()
    
    # 4. Surgery distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(stats['surgery_per_day'])+1), stats['surgery_per_day'])
    plt.axhline(y=len(instance_data['specializations']), color='red', linestyle='--', 
                label=f'Capacity ({len(instance_data["specializations"])})')
    plt.xlabel('Day')
    plt.ylabel('Number of Surgeries')
    plt.title(f'Surgeries Scheduled by Day - {instance_name}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(viz_folder, 'surgeries_by_day.png'), dpi=300)
    plt.close()
    
    # 5. Cost breakdown
    cost_data = stats['cost_breakdown']
    if cost_data:
        # Remove total from pie chart
        labels = list(cost_data.keys())
        if 'total_estimate' in labels:
            labels.remove('total_estimate')
        values = [cost_data[k] for k in labels]
        
        if values:
            plt.figure(figsize=(10, 6))
            wedges, texts, autotexts = plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title(f'Cost Breakdown - {instance_name}', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_folder, 'cost_breakdown.png'), dpi=300)
            plt.close()
            
    # 6. OT utilization visualization
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
    plt.savefig(os.path.join(viz_folder, 'ot_utilization_rate.png'), dpi=300)
    plt.close()

page = st.sidebar.selectbox("Choose an Option:", ["Single Algorithms", "Compare Algorithms"])

# --------------------------------------------------------------------------
# Single Algorithms Page
# --------------------------------------------------------------------------
if page == "Single Algorithms":
    st.sidebar.title("Configuration")
    algoritmo = st.sidebar.selectbox("Choose algorithm", ["sa", "hybrid_sa", "tabu", "ga", "hc", "rw"])
    
    instancias_path = "./data/instances"
    arquivos = [f for f in os.listdir(instancias_path) if f.endswith(".dat")]
    
    usar_todas = st.sidebar.checkbox("Use all instances", value=False)
    instancias_escolhidas = arquivos if usar_todas else st.sidebar.multiselect("Select instances", arquivos)
    
    # Algorithm-specific parameters for the single algorithm page.
    algo_params = {}
    if algoritmo == "hc":
        algo_params["iterations"] = st.sidebar.slider("Iterations", 500, 5000, 1000, 100)
        algo_params["restart_after"] = st.sidebar.slider("Restart after (iterations)", 50, 500, 100, 10)
    elif algoritmo == "rw":
        algo_params["iterations"] = st.sidebar.slider("Iterations", 500, 10000, 2000, 100)
    elif algoritmo == "ga":
        algo_params["population_size"] = st.sidebar.slider("Population size", 50, 300, 150, 10)
        algo_params["generations"] = st.sidebar.slider("Generations", 50, 500, 100, 10)
    
    if st.sidebar.button("Run"):
        from results_manager import create_test_directory_structure, save_solution_csv
        
        results_summary = []
        # Define a lista de instâncias (sem extensão) para criar a estrutura de pastas.
        if usar_todas:
            instances = [os.path.splitext(f)[0] for f in arquivos]
        else:
            instances = [os.path.splitext(f)[0] for f in instancias_escolhidas]
        # Em Single Algorithms, há apenas um algoritmo selecionado.
        algorithms = [algoritmo]
        test_folder, folder_structure = create_test_directory_structure(instances, algorithms)
        st.markdown(f"### Results will be saved in: {test_folder}")
        
        for filename in instancias_escolhidas:
            instance_name = os.path.splitext(filename)[0]
            # Obtém a pasta de saída para esta instância e algoritmo.
            output_folder = folder_structure[instance_name][algoritmo]
            
            with st.expander(f"Instance: {instance_name}", expanded=True):
                filepath = os.path.join(instancias_path, filename)
                dados = parse_instance_file(filepath)
                scheduler = get_scheduler(algoritmo, dados, algo_params)
                solucao = scheduler.run()
                
                # Converter solução para DataFrame e exibir.
                solution_df = pd.DataFrame.from_dict(solucao, orient="index").reset_index()
                solution_df.rename(columns={"index": "Patient"}, inplace=True)
                solution_df = solution_df[["Patient", "ward", "day"]]
                st.write("#### Final Solution")
                st.dataframe(solution_df)
                
                # Analisar a solução.
                stats = analyze_solution(dados, solucao)
                total_pacientes = stats['total_patients']
                alocados = stats['allocated_patients']
                pct_alocados = (alocados / total_pacientes) * 100
                custo = getattr(scheduler, "final_cost", None)
                if custo is None:
                    custo = calculate_cost(dados, solucao)
                tempo = getattr(scheduler, "runtime", "N/A")
                if algoritmo == "ga":
                    iteracoes = getattr(scheduler, "generations", "N/A")
                    iteracoes_label = "Generations"
                else:
                    iteracoes = getattr(scheduler, "iterations", "N/A")
                    iteracoes_label = "Iterations"
                
                # Compute average occupancy and percentage of outliers.
                avg_occupancy = compute_average_occupancy(stats) * 100
                outlier_percent = compute_outlier_percentage(stats)
                
                st.markdown("### General Statistics")
                metrics = {
                    iteracoes_label: iteracoes,
                    "Time (s)": round(tempo, 2) if isinstance(tempo, (int, float)) else tempo,
                    "% Allocated": round(pct_alocados, 2),
                    "Final Cost": round(custo, 2) if isinstance(custo, (int, float)) else custo,
                    "Avg Occupancy (%)": round(avg_occupancy, 2),
                    "Outlier (%)": round(outlier_percent, 2)
                }
                st.write(metrics)
                
                st.markdown("### Solution Analysis")
                # Cost Breakdown
                st.markdown("#### Cost Breakdown")
                cost_data = stats['cost_breakdown']
                if cost_data:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    labels = list(cost_data.keys())
                    if 'total_estimate' in labels:
                        labels.remove('total_estimate')
                    values = [cost_data[k] for k in labels]
                    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig)
                
                # Surgery Distribution
                st.markdown("#### Surgery Distribution")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(range(1, len(stats['surgery_per_day'])+1), stats['surgery_per_day'])
                ax.set_xlabel('Day')
                ax.set_ylabel('Number of Surgeries')
                ax.set_title('Surgeries per Day')
                st.pyplot(fig)
                
                # Ward Occupancy Heatmap
                st.markdown("#### Ward Occupancy")
                ward_data = []
                for ward, occupancy in stats['ward_occupancy'].items():
                    capacity = occupancy['capacity']
                    for day, occ in enumerate(occupancy['daily_occupancy']):
                        ward_data.append({
                            'Ward': ward,
                            'Day': f'Day {day+1}',
                            'Occupancy': occ,
                            'Capacity': capacity,
                            'Rate': occ/capacity if capacity > 0 else 0
                        })
                if ward_data:
                    df_ward = pd.DataFrame(ward_data)
                    pivot = df_ward.pivot(index='Ward', columns='Day', values='Rate')
                    fig, ax = plt.subplots(figsize=(12, 4))
                    sns.heatmap(pivot, annot=True, fmt='.0%', cmap='YlGnBu', linewidths=0.5, vmin=0, vmax=1.2, center=0.6)
                    plt.title('Ward Occupancy Rate')
                    st.pyplot(fig)
                
                # Algorithm Performance: Cost Evolution.
                st.markdown("### Algorithm Performance")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(scheduler.cost_history, label="Cost")
                ax.set_title("Cost Evolution")
                ax.set_xlabel(iteracoes_label)
                ax.set_ylabel("Cost")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
                
                # Temperature Evolution (for Simulated Annealing)
                if algoritmo == "sa" and hasattr(scheduler, "temperature_history"):
                    st.markdown("#### Temperature Evolution")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(scheduler.temperature_history, label="Temperature", color="orange")
                    ax.set_title("Temperature Evolution")
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Temperature")
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                
                # Salvar a solução e as métricas na pasta do algoritmo.
                save_solution_csv(solucao, metrics, output_folder)
                st.success(f"Results saved in: {output_folder}")
                
                # Save visualizations to the output folder
                save_visualizations_to_folder(scheduler, dados, output_folder, instance_name, algoritmo)
                
                results_summary.append({
                    "Instance": filename,
                    "Algorithm": algoritmo,
                    iteracoes_label: iteracoes,
                    "Time (s)": round(tempo, 2) if isinstance(tempo, (int, float)) else tempo,
                    "% Allocated": round(pct_alocados, 2),
                    "Final Cost": round(custo, 2) if isinstance(custo, (int, float)) else custo,
                    "Avg Occupancy (%)": round(avg_occupancy, 2),
                    "Outlier (%)": round(outlier_percent, 2)
                })
        
        st.markdown("## Summary Statistics")
        df_results = pd.DataFrame(results_summary)
        st.dataframe(df_results)
        
        # Comparison charts across instances (if more than one)
        if len(results_summary) > 1:
            st.markdown("## Instance Comparisons")
            # Final Cost Comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(df_results['Instance'], df_results['Final Cost'])
            ax.set_xlabel('Instance')
            ax.set_ylabel('Final Cost')
            ax.set_title('Cost Comparison Across Instances')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            
            # Runtime Comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(df_results['Instance'], df_results['Time (s)'])
            ax.set_xlabel('Instance')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Runtime Comparison Across Instances')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            
            # Average Occupancy Comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(df_results['Instance'], df_results['Avg Occupancy (%)'])
            ax.set_xlabel('Instance')
            ax.set_ylabel('Average Occupancy (%)')
            ax.set_title('Average Occupancy Comparison Across Instances')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
            
            # Outlier Percentage Comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(df_results['Instance'], df_results['Outlier (%)'])
            ax.set_xlabel('Instance')
            ax.set_ylabel('Outlier (%)')
            ax.set_title('Outlier Percentage Comparison Across Instances')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

# --------------------------------------------------------------------------
# Compare Algorithms Page
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Compare Algorithms Page
# --------------------------------------------------------------------------
elif page == "Compare Algorithms":
    st.sidebar.title("Comparison Configuration")
    # Allow user to select multiple algorithms.
    selected_algorithms = st.sidebar.multiselect(
        "Select Algorithms",
        options=["sa", "hybrid_sa", "tabu", "ga", "hc", "rw"],
        default=["sa", "ga"]
    )
    
    # Instance selection.
    instancias_path = "./data/instances"
    arquivos = [f for f in os.listdir(instancias_path) if f.endswith(".dat")]
    usar_todas = st.sidebar.checkbox("Use all instances", value=False, key="compare_use_all")
    instancias_escolhidas = arquivos if usar_todas else st.sidebar.multiselect("Select instances", arquivos, key="compare_instances")
    
    # Prepare algorithm-specific parameters.
    algo_params = {}
    if "hc" in selected_algorithms:
        with st.sidebar.expander("Hill Climbing (hc) Parameters"):
            hc_iterations = st.slider("Iterations", 500, 5000, 1000, 100, key="compare_hc_iterations")
            hc_restart_after = st.slider("Restart after (iterations)", 50, 500, 100, 10, key="compare_hc_restart")
            algo_params["hc"] = {"iterations": hc_iterations, "restart_after": hc_restart_after}
    if "rw" in selected_algorithms:
        with st.sidebar.expander("Random Walk (rw) Parameters"):
            rw_iterations = st.slider("Iterations", 500, 10000, 2000, 100, key="compare_rw_iterations")
            algo_params["rw"] = {"iterations": rw_iterations}
    if "ga" in selected_algorithms:
        with st.sidebar.expander("Genetic Algorithm (ga) Parameters"):
            ga_population_size = st.slider("Population size", 50, 300, 150, 10, key="compare_ga_population")
            ga_generations = st.slider("Generations", 50, 500, 100, 10, key="compare_ga_generations")
            algo_params["ga"] = {"population_size": ga_population_size, "generations": ga_generations}
    
    if st.sidebar.button("Run Comparisons"):
        from results_manager import create_test_directory_structure, save_solution_csv
        
        results_summary = []  # Overall summary across instances and algorithms.
        
        # Cria a estrutura de pastas para todas as instâncias selecionadas e para todos os algoritmos escolhidos.
        if usar_todas:
            instances = [os.path.splitext(f)[0] for f in arquivos]
        else:
            instances = [os.path.splitext(f)[0] for f in instancias_escolhidas]
        test_folder, folder_structure = create_test_directory_structure(instances, selected_algorithms)
        st.markdown(f"### Results will be saved in: {test_folder}")
        
        for filename in instancias_escolhidas:
            instance_name = os.path.splitext(filename)[0]
            filepath = os.path.join(instancias_path, filename)
            dados = parse_instance_file(filepath)
            
            st.markdown(f"## Instance: {instance_name}")
            # Create a tab for each selected algorithm.
            tabs = st.tabs(selected_algorithms)
            for i, algorithm in enumerate(selected_algorithms):
                with tabs[i]:
                    st.markdown(f"### Algorithm: {algorithm}")
                    params = algo_params.get(algorithm, {})
                    scheduler = get_scheduler(algorithm, dados, params)
                    solucao = scheduler.run()
                    
                    # Final solution DataFrame.
                    solution_df = pd.DataFrame.from_dict(solucao, orient="index").reset_index()
                    solution_df.rename(columns={"index": "Patient"}, inplace=True)
                    solution_df = solution_df[["Patient", "ward", "day"]]
                    st.write("#### Final Solution")
                    st.dataframe(solution_df)
                    
                    # Analyze solution.
                    stats = analyze_solution(dados, solucao)
                    total_pacientes = stats['total_patients']
                    alocados = stats['allocated_patients']
                    pct_alocados = (alocados / total_pacientes) * 100
                    custo = getattr(scheduler, "final_cost", None)
                    if custo is None:
                        custo = calculate_cost(dados, solucao)
                    tempo = getattr(scheduler, "runtime", "N/A")
                    if algorithm == "ga":
                        iteracoes = getattr(scheduler, "generations", "N/A")
                        iteracoes_label = "Generations"
                    else:
                        iteracoes = getattr(scheduler, "iterations", "N/A")
                        iteracoes_label = "Iterations"
                    
                    # Compute average occupancy and percentage of outliers.
                    avg_occupancy = compute_average_occupancy(stats) * 100
                    outlier_percent = compute_outlier_percentage(stats)
                    
                    st.markdown("#### General Statistics")
                    st.write({
                        iteracoes_label: iteracoes,
                        "Time (s)": round(tempo, 2) if isinstance(tempo, (int, float)) else tempo,
                        "% Allocated": round(pct_alocados, 2),
                        "Final Cost": round(custo, 2) if isinstance(custo, (int, float)) else custo,
                        "Avg Occupancy (%)": round(avg_occupancy, 2),
                        "Outlier (%)": round(outlier_percent, 2)
                    })
                    
                    # Cost Breakdown Pie Chart.
                    st.markdown("#### Cost Breakdown")
                    cost_data = stats['cost_breakdown']
                    if cost_data:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        labels = list(cost_data.keys())
                        if 'total_estimate' in labels:
                            labels.remove('total_estimate')
                        values = [cost_data[k] for k in labels]
                        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                        ax.axis('equal')
                        st.pyplot(fig)
                    
                    # Surgery Distribution Bar Chart.
                    st.markdown("#### Surgery Distribution")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.bar(range(1, len(stats['surgery_per_day'])+1), stats['surgery_per_day'])
                    ax.set_xlabel('Day')
                    ax.set_ylabel('Number of Surgeries')
                    ax.set_title('Surgeries per Day')
                    st.pyplot(fig)
                    
                    # Ward Occupancy Heatmap.
                    st.markdown("#### Ward Occupancy")
                    ward_data = []
                    for ward, occupancy in stats['ward_occupancy'].items():
                        capacity = occupancy['capacity']
                        for day, occ in enumerate(occupancy['daily_occupancy']):
                            ward_data.append({
                                'Ward': ward,
                                'Day': f'Day {day+1}',
                                'Occupancy': occ,
                                'Capacity': capacity,
                                'Rate': occ/capacity if capacity > 0 else 0
                            })
                    if ward_data:
                        df_ward = pd.DataFrame(ward_data)
                        pivot = df_ward.pivot(index='Ward', columns='Day', values='Rate')
                        fig, ax = plt.subplots(figsize=(12, 4))
                        sns.heatmap(pivot, annot=True, fmt='.0%', cmap='YlGnBu', linewidths=0.5, vmin=0, vmax=1.2, center=0.6)
                        plt.title('Ward Occupancy Rate')
                        st.pyplot(fig)
                    
                    # Algorithm Performance: Cost Evolution.
                    st.markdown("#### Algorithm Performance")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(scheduler.cost_history, label="Cost")
                    ax.set_title("Cost Evolution")
                    ax.set_xlabel(iteracoes_label)
                    ax.set_ylabel("Cost")
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Temperature Evolution (for SA)
                    if algorithm == "sa" and hasattr(scheduler, "temperature_history"):
                        st.markdown("#### Temperature Evolution")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(scheduler.temperature_history, label="Temperature", color="orange")
                        ax.set_title("Temperature Evolution")
                        ax.set_xlabel("Iteration")
                        ax.set_ylabel("Temperature")
                        ax.grid(True)
                        ax.legend()
                        st.pyplot(fig)
                    
                    # Salva a solução e as métricas na pasta do algoritmo.
                    output_folder = folder_structure[instance_name][algorithm]
                    metrics_to_save = {
                        iteracoes_label: iteracoes,
                        "Time (s)": round(tempo, 2) if isinstance(tempo, (int, float)) else tempo,
                        "% Allocated": round(pct_alocados, 2),
                        "Final Cost": round(custo, 2) if isinstance(custo, (int, float)) else custo,
                        "Avg Occupancy (%)": round(avg_occupancy, 2),
                        "Outlier (%)": round(outlier_percent, 2)
                    }
                    save_solution_csv(solucao, metrics_to_save, output_folder)
                    st.success(f"Results saved in: {output_folder}")
                    
                    # Save visualizations to the output folder
                    save_visualizations_to_folder(scheduler, dados, output_folder, instance_name, algorithm)
                    
                    results_summary.append({
                        "Instance": instance_name,
                        "Algorithm": algorithm,
                        iteracoes_label: iteracoes,
                        "Time (s)": round(tempo, 2) if isinstance(tempo, (int, float)) else tempo,
                        "% Allocated": round(pct_alocados, 2),
                        "Final Cost": round(custo, 2) if isinstance(custo, (int, float)) else custo,
                        "Avg Occupancy (%)": round(avg_occupancy, 2),
                        "Outlier (%)": round(outlier_percent, 2)
                    })
        
        st.markdown("## Overall Summary Statistics")
        df_results = pd.DataFrame(results_summary)
        st.dataframe(df_results)
        
        # Multi-level grouped comparison charts.
        if not df_results.empty:
            st.markdown("### Final Cost Comparison")
            cost_pivot = df_results.pivot(index="Instance", columns="Algorithm", values="Final Cost")
            fig, ax = plt.subplots(figsize=(10, 6))
            cost_pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("Final Cost")
            ax.set_title("Cost Comparison Across Instances and Algorithms")
            st.pyplot(fig)
            
            st.markdown("### Runtime Comparison")
            time_pivot = df_results.pivot(index="Instance", columns="Algorithm", values="Time (s)")
            fig, ax = plt.subplots(figsize=(10, 6))
            time_pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("Time (s)")
            ax.set_title("Runtime Comparison Across Instances and Algorithms")
            st.pyplot(fig)
            
            st.markdown("### Percentage of Allocated Patients")
            alloc_pivot = df_results.pivot(index="Instance", columns="Algorithm", values="% Allocated")
            fig, ax = plt.subplots(figsize=(10, 6))
            alloc_pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("% Allocated")
            ax.set_title("Patient Allocation Across Instances and Algorithms")
            st.pyplot(fig)
            
            st.markdown("### Average Occupancy Comparison")
            occupancy_pivot = df_results.pivot(index="Instance", columns="Algorithm", values="Avg Occupancy (%)")
            fig, ax = plt.subplots(figsize=(10, 6))
            occupancy_pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("Avg Occupancy (%)")
            ax.set_title("Average Occupancy Across Instances and Algorithms")
            st.pyplot(fig)
            
            st.markdown("### Outlier Percentage Comparison")
            outlier_pivot = df_results.pivot(index="Instance", columns="Algorithm", values="Outlier (%)")
            fig, ax = plt.subplots(figsize=(10, 6))
            outlier_pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("Outlier (%)")
            ax.set_title("Outlier Percentage Across Instances and Algorithms")
            st.pyplot(fig)