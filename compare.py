import streamlit as st
import os
import pandas as pd
import time
from parser import parse_instance_file

# Map algorithm names to their scheduler constructors.
algorithm_constructors = {
    "sa": lambda instance_data: __import__("schedulers.simulated", fromlist=["SimulatedAnnealingScheduler"]).SimulatedAnnealingScheduler(instance_data),
    "hybrid_sa": lambda instance_data: __import__("schedulers.hybrid_simulated", fromlist=["HybridSimulatedAnnealingScheduler"]).HybridSimulatedAnnealingScheduler(instance_data),
    "tabu": lambda instance_data: __import__("schedulers.tabu", fromlist=["TabuSearchScheduler"]).TabuSearchScheduler(instance_data),
    "ga": lambda instance_data: __import__("schedulers.genetic", fromlist=["GeneticAlgorithmScheduler"]).GeneticAlgorithmScheduler(instance_data),
    "hc": lambda instance_data: __import__("schedulers.hill_climbing", fromlist=["HillClimbingScheduler"]).HillClimbingScheduler(instance_data),
    "rw": lambda instance_data: __import__("schedulers.random_walk", fromlist=["RandomWalkScheduler"]).RandomWalkScheduler(instance_data)
}

st.title("Algorithm Comparison")

st.sidebar.header("Comparison Settings")

# Use unique keys for sidebar widgets to avoid state conflicts.
selected_algs = st.sidebar.multiselect(
    "Select up to 3 algorithms to compare:",
    options=list(algorithm_constructors.keys()),
    default=["sa"],
    key="compare_selected_algs"
)
if len(selected_algs) > 3:
    st.sidebar.error("Select at most 3 algorithms.")

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

# Use a session state flag to persist that the compare button was pressed.
if st.sidebar.button("Compare", key="compare_button"):
    st.session_state.compare = True

if "compare" in st.session_state and st.session_state.compare:
    st.write("## Comparison Results")
    results = []
    for instance_file in selected_instances:
        instance_name = os.path.splitext(instance_file)[0]
        filepath = os.path.join(instances_path, instance_file)
        instance_data = parse_instance_file(filepath)
        st.write(f"### Instance: {instance_name}")
        for alg in selected_algs:
            st.write(f"**Algorithm: {alg}**")
            scheduler = algorithm_constructors[alg](instance_data)
            start_time = time.time()
            # Try running with max_time; if not supported, fall back to run() without the parameter.
            try:
                solution = scheduler.run(max_time=execution_times[alg])
            except TypeError:
                solution = scheduler.run()
            elapsed = time.time() - start_time
            cost = getattr(scheduler, "final_cost", None)
            iterations = getattr(scheduler, "iterations", None)
            st.write(f"Time: {elapsed:.2f} s, Iterations: {iterations}, Final Cost: {cost:.2f}")
            results.append({
                "Instance": instance_name,
                "Algorithm": alg,
                "Time (s)": round(elapsed, 2),
                "Iterations": iterations,
                "Final Cost": cost
            })
    if results:
        df_results = pd.DataFrame(results)
        st.write("### Comparison Table")
        st.dataframe(df_results)
