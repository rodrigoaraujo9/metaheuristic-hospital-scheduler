import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from parser import parse_instance_file
from utils import analyze_solution, calculate_cost

st.set_page_config(page_title="Hospital Scheduling", layout="wide")
page = st.sidebar.selectbox("Choose an Option:", ["Single Algorithms", "Compare Algorithms"])

def get_scheduler_compare(algorithm, instance_data, params):
    """
    Returns the scheduler instance based on the chosen algorithm and provided parameters.
    Available algorithms: 'sa' (Simulated Annealing), 'hybrid_sa' (Hybrid SA), 
    'tabu' (Tabu Search), 'ga' (Genetic Algorithm), 'hc' (Hill Climbing), 'rw' (Random Walk).
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
            return GeneticAlgorithmScheduler(instance_data, 
                                             population_size=params.get("population_size"),
                                             generations=params.get("generations"))
        else:
            return GeneticAlgorithmScheduler(instance_data)
    elif algorithm == "hc":
        from schedulers.hill_climbing import HillClimbingScheduler
        return HillClimbingScheduler(instance_data, 
                                     iterations=params.get("iterations", 1000),
                                     restart_after=params.get("restart_after", 100))
    elif algorithm == "rw":
        from schedulers.random_walk import RandomWalkScheduler
        return RandomWalkScheduler(instance_data, 
                                   iterations=params.get("iterations", 2000))
    else:
        raise ValueError("Unknown algorithm.")

if page == "Single Algorithms":
    # (The code for Single Algorithms remains unchanged.)
    st.sidebar.title("Configuration")
    algoritmo = st.sidebar.selectbox("Choose algorithm", ["sa", "hybrid_sa", "tabu", "ga", "hc", "rw"])
    
    instancias_path = "./data/instances"
    arquivos = [f for f in os.listdir(instancias_path) if f.endswith(".dat")]
    
    usar_todas = st.sidebar.checkbox("Use all instances", value=False)
    instancias_escolhidas = arquivos if usar_todas else st.sidebar.multiselect("Select instances", arquivos)
    
    # Algorithm-specific parameters
    if algoritmo == "hc":
        iterations = st.sidebar.slider("Iterations", 500, 5000, 1000, 100)
        restart_after = st.sidebar.slider("Restart after (iterations)", 50, 500, 100, 10)
    elif algoritmo == "rw":
        iterations = st.sidebar.slider("Iterations", 500, 10000, 2000, 100)
    elif algoritmo == "ga":
        population_size = st.sidebar.slider("Population size", 50, 300, 150, 10)
        generations = st.sidebar.slider("Generations", 50, 500, 100, 10)
        
    def get_scheduler(algorithm, instance_data):
        """
        Returns the scheduler instance for the given algorithm.
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
            if 'population_size' in st.session_state and 'generations' in st.session_state:
                return GeneticAlgorithmScheduler(
                    instance_data, 
                    population_size=st.session_state.population_size,
                    generations=st.session_state.generations
                )
            else:
                return GeneticAlgorithmScheduler(instance_data)
        elif algorithm == "hc":
            from schedulers.hill_climbing import HillClimbingScheduler
            return HillClimbingScheduler(
                instance_data,
                iterations=iterations,
                restart_after=restart_after
            )
        elif algorithm == "rw":
            from schedulers.random_walk import RandomWalkScheduler
            return RandomWalkScheduler(
                instance_data,
                iterations=iterations
            )
        else:
            raise ValueError("Unknown algorithm. Use 'sa', 'hybrid_sa', 'tabu', 'ga', 'hc', or 'rw'.")
    
    if st.sidebar.button("Run"):
        if algoritmo == "ga":
            st.session_state.population_size = population_size
            st.session_state.generations = generations
        
        resultados = []
        for filename in instancias_escolhidas:
            instance_name = os.path.splitext(filename)[0]
            with st.expander(f"Instance: {instance_name}", expanded=True):
                filepath = os.path.join(instancias_path, filename)
                dados = parse_instance_file(filepath)
                scheduler = get_scheduler(algoritmo, dados)
                solucao = scheduler.run()
                
                # Process solution and display results
                solution_df = pd.DataFrame.from_dict(solucao, orient="index").reset_index()
                solution_df.rename(columns={"index": "Patient"}, inplace=True)
                solution_df = solution_df[["Patient", "ward", "day"]]
                
                st.write("#### Final Solution")
                st.dataframe(solution_df)
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
                
                st.markdown("### General Statistics")
                st.write({
                    iteracoes_label: iteracoes,
                    "Time (s)": round(tempo, 2) if isinstance(tempo, (int, float)) else tempo,
                    "% Allocated": round(pct_alocados, 2),
                    "Final Cost": round(custo, 2) if isinstance(custo, (int, float)) else custo
                })
                
                st.markdown("### Solution Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Cost Breakdown:**")
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
                
                with col2:
                    st.write("**Surgery Distribution:**")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.bar(range(1, len(stats['surgery_per_day'])+1), stats['surgery_per_day'])
                    ax.set_xlabel('Day')
                    ax.set_ylabel('Number of Surgeries')
                    ax.set_title('Surgeries per Day')
                    st.pyplot(fig)
                
                st.write("**Ward Occupancy:**")
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
                    sns.heatmap(pivot, annot=True, fmt='.0%', cmap='YlGnBu', 
                                linewidths=0.5, vmin=0, vmax=1.2, center=0.6)
                    plt.title('Ward Occupancy Rate')
                    st.pyplot(fig)
                
                st.markdown("### Algorithm Performance")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(scheduler.cost_history, label="Cost")
                ax.set_title("Cost Evolution")
                ax.set_xlabel(iteracoes_label)
                ax.set_ylabel("Cost")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
                
                if algoritmo == "sa" and hasattr(scheduler, "temperature_history"):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(scheduler.temperature_history, label="Temperature", color="orange")
                    ax.set_title("Temperature Evolution")
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Temperature")
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                
                resultados.append({
                    "Instance": filename,
                    "Algorithm": algoritmo,
                    iteracoes_label: iteracoes,
                    "Time (s)": round(tempo, 2) if isinstance(tempo, (int, float)) else tempo,
                    "% Allocated": round(pct_alocados, 2),
                    "Final Cost": round(custo, 2) if isinstance(custo, (int, float)) else custo
                })

        st.markdown("## Summary Statistics")
        df_resultados = pd.DataFrame(resultados)
        st.dataframe(df_resultados)
        
        if len(resultados) > 1:
            st.markdown("## Instance Comparisons")
            # For simplicity, create bar charts per metric aggregated across instances.
            # Cost Comparison
            cost_pivot = df_resultados.pivot(index="Instance", columns="Algorithm", values="Final Cost")
            fig, ax = plt.subplots(figsize=(10, 6))
            cost_pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("Final Cost")
            ax.set_title("Cost Comparison Across Instances and Algorithms")
            st.pyplot(fig)
            
            # Runtime Comparison
            time_pivot = df_resultados.pivot(index="Instance", columns="Algorithm", values="Time (s)")
            fig, ax = plt.subplots(figsize=(10, 6))
            time_pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("Time (s)")
            ax.set_title("Runtime Comparison Across Instances and Algorithms")
            st.pyplot(fig)

elif page == "Compare Algorithms":
    st.sidebar.title("Comparison Configuration")
    # Allow user to select multiple algorithms.
    selected_algorithms = st.sidebar.multiselect(
        "Select Algorithms", 
        options=["sa", "hybrid_sa", "tabu", "ga", "hc", "rw"],
        default=["sa", "ga"]
    )
    
    instancias_path = "./data/instances"
    arquivos = [f for f in os.listdir(instancias_path) if f.endswith(".dat")]
    usar_todas = st.sidebar.checkbox("Use all instances", value=False, key="compare_use_all")
    instancias_escolhidas = arquivos if usar_todas else st.sidebar.multiselect("Select instances", arquivos, key="compare_instances")
    
    # Prepare a dictionary to hold algorithm-specific parameters.
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
        results_summary = []
        # Loop over each selected instance.
        for filename in instancias_escolhidas:
            instance_name = os.path.splitext(filename)[0]
            filepath = os.path.join(instancias_path, filename)
            dados = parse_instance_file(filepath)
            
            st.markdown(f"## Instance: {instance_name}")
            # Create a tab for each selected algorithm
            tabs = st.tabs(selected_algorithms)
            for i, algorithm in enumerate(selected_algorithms):
                with tabs[i]:
                    st.markdown(f"### Algorithm: {algorithm}")
                    params = algo_params.get(algorithm, {})
                    scheduler = get_scheduler_compare(algorithm, dados, params)
                    solucao = scheduler.run()
                    
                    # Convert solution to DataFrame and display.
                    solution_df = pd.DataFrame.from_dict(solucao, orient="index").reset_index()
                    solution_df.rename(columns={"index": "Patient"}, inplace=True)
                    solution_df = solution_df[["Patient", "ward", "day"]]
                    st.write("#### Final Solution")
                    st.dataframe(solution_df)
                    
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
                    
                    st.markdown("#### General Statistics")
                    st.write({
                        iteracoes_label: iteracoes,
                        "Time (s)": round(tempo, 2) if isinstance(tempo, (int, float)) else tempo,
                        "% Allocated": round(pct_alocados, 2),
                        "Final Cost": round(custo, 2) if isinstance(custo, (int, float)) else custo
                    })
                    
                    st.markdown("#### Algorithm Performance")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(scheduler.cost_history, label="Cost")
                    ax.set_title("Cost Evolution")
                    ax.set_xlabel(iteracoes_label)
                    ax.set_ylabel("Cost")
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                    
                    if algorithm == "sa" and hasattr(scheduler, "temperature_history"):
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
                        "Final Cost": round(custo, 2) if isinstance(custo, (int, float)) else custo
                    })
        
        st.markdown("## Summary Statistics Across Algorithms")
        df_results = pd.DataFrame(results_summary)
        st.dataframe(df_results)
        
        # Generate grouped bar charts for overall cost and runtime comparison.
        if not df_results.empty:
            st.markdown("### Cost Comparison")
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
