import streamlit as st
import os
import time
import pandas as pd
from parser import parse_instance_file
from utils import calculate_cost

# Map algorithm names to their scheduler constructors.
algorithm_constructors = {
    "sa": lambda instance_data: __import__("schedulers.simulated", fromlist=["SimulatedAnnealingScheduler"]).SimulatedAnnealingScheduler(instance_data),
    "hybrid_sa": lambda instance_data: __import__("schedulers.hybrid_simulated", fromlist=["HybridSimulatedAnnealingScheduler"]).HybridSimulatedAnnealingScheduler(instance_data),
    "tabu": lambda instance_data: __import__("schedulers.tabu", fromlist=["TabuSearchScheduler"]).TabuSearchScheduler(instance_data),
    "ga": lambda instance_data: __import__("schedulers.genetic", fromlist=["GeneticAlgorithmScheduler"]).GeneticAlgorithmScheduler(instance_data),
    "hc": lambda instance_data: __import__("schedulers.hill_climbing", fromlist=["HillClimbingScheduler"]).HillClimbingScheduler(instance_data),
    "rw": lambda instance_data: __import__("schedulers.random_walk", fromlist=["RandomWalkScheduler"]).RandomWalkScheduler(instance_data)
}

st.set_page_config(page_title="Algorithm Comparison", layout="wide")
st.title("Algorithm Comparison")
st.sidebar.header("Comparison Settings")

# Sidebar: Select algorithms (limit to 3)
selected_algs = st.sidebar.multiselect(
    "Select up to 3 algorithms to compare:",
    options=list(algorithm_constructors.keys()),
    default=["sa"],
    key="compare_selected_algs"
)
if len(selected_algs) > 3:
    st.sidebar.error("Select at most 3 algorithms.")

# Sidebar: Execution time configuration per algorithm
same_time = st.sidebar.checkbox(
    "Use the same maximum execution time for all algorithms", 
    value=True,
    key="compare_same_time"
)
execution_times = {}
if same_time:
    time_all = st.sidebar.slider(
        "Maximum execution time (s) for all:",
        min_value=10, max_value=600, value=120, step=10,
        key="compare_time_all"
    )
    for alg in selected_algs:
        execution_times[alg] = time_all
else:
    for alg in selected_algs:
        execution_times[alg] = st.sidebar.slider(
            f"Maximum execution time (s) for {alg}:",
            min_value=10, max_value=600, value=120, step=10,
            key=f"compare_time_{alg}"
        )

# Sidebar: Instance selection
instances_path = "./data/instances"
instance_files = [f for f in os.listdir(instances_path) if f.endswith(".dat")]
use_all_instances = st.sidebar.checkbox(
    "Use all instances", 
    value=False,
    key="compare_all_instances"
)
if use_all_instances:
    selected_instances = instance_files
else:
    selected_instances = st.sidebar.multiselect(
        "Select instances:",
        options=instance_files,
        key="compare_instance_files"
    )

# Use the button's return value to trigger the comparison.
compare_pressed = st.sidebar.button("Compare", key="compare_button")

if compare_pressed:
    st.write("## Comparison Results")
    results = []
    # Loop over selected instances
    for instance_file in selected_instances:
        instance_name = os.path.splitext(instance_file)[0]
        filepath = os.path.join(instances_path, instance_file)
        instance_data = parse_instance_file(filepath)
        st.write(f"### Instance: {instance_name}")
        total_patients = len(instance_data['patients'])
        # Loop over each selected algorithm
        for alg in selected_algs:
            st.write(f"**Algorithm: {alg}**")
            scheduler = algorithm_constructors[alg](instance_data)
            t0 = time.time()
            try:
                solution = scheduler.run(max_time=execution_times[alg])
            except TypeError:
                solution = scheduler.run()
            elapsed = time.time() - t0

            # Get iterations or generations if available.
            iterations = getattr(scheduler, "iterations", None) or getattr(scheduler, "generations", None)
            # Get final cost (or calculate it if not available).
            cost = getattr(scheduler, "final_cost", None)
            if cost is None:
                cost = calculate_cost(instance_data, solution)
            # Compute allocation percentage.
            allocated = sum(1 for d in solution.values() if d['ward'] is not None)
            allocation_pct = allocated / total_patients * 100 if total_patients > 0 else 0

            st.write(f"Time: {elapsed:.2f} s, Iterations: {iterations}, Final Cost: {cost:.2f}, Allocation: {allocation_pct:.2f}%")
            results.append({
                "Instance": instance_name,
                "Algorithm": alg,
                "Time (s)": round(elapsed, 2),
                "Iterations": iterations,
                "Final Cost": round(cost, 2) if isinstance(cost, (int, float)) else cost,
                "Allocation (%)": round(allocation_pct, 2)
            })
    
    if results:
        df_results = pd.DataFrame(results)
        st.write("### Comparison Table")
        st.dataframe(df_results)

        # Aggregated metrics: Average Final Cost, Runtime, and Allocation per Algorithm.
        st.write("### Aggregated Metrics")
        avg_cost = df_results.groupby("Algorithm")["Final Cost"].mean()
        avg_time = df_results.groupby("Algorithm")["Time (s)"].mean()
        avg_alloc = df_results.groupby("Algorithm")["Allocation (%)"].mean()

        st.write("Average Final Cost per Algorithm")
        st.bar_chart(avg_cost)
        st.write("Average Runtime (s) per Algorithm")
        st.bar_chart(avg_time)
        st.write("Average Allocation (%) per Algorithm")
        st.bar_chart(avg_alloc)
