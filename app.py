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
        results_summary = []
        for filename in instancias_escolhidas:
            instance_name = os.path.splitext(filename)[0]
            with st.expander(f"Instance: {instance_name}", expanded=True):
                filepath = os.path.join(instancias_path, filename)
                dados = parse_instance_file(filepath)
                scheduler = get_scheduler(algoritmo, dados, algo_params)
                solucao = scheduler.run()
                
                # Convert solution (a dict) into a DataFrame.
                solution_df = pd.DataFrame.from_dict(solucao, orient="index").reset_index()
                solution_df.rename(columns={"index": "Patient"}, inplace=True)
                solution_df = solution_df[["Patient", "ward", "day"]]
                st.write("#### Final Solution")
                st.dataframe(solution_df)
                
                # Analyze the solution.
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
                st.write({
                    iteracoes_label: iteracoes,
                    "Time (s)": round(tempo, 2) if isinstance(tempo, (int, float)) else tempo,
                    "% Allocated": round(pct_alocados, 2),
                    "Final Cost": round(custo, 2) if isinstance(custo, (int, float)) else custo,
                    "Avg Occupancy (%)": round(avg_occupancy, 2),
                    "Outlier (%)": round(outlier_percent, 2)
                })
                
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
        results_summary = []  # Overall summary across instances and algorithms.
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
