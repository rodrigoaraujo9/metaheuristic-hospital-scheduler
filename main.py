import os
import pandas as pd
import matplotlib.pyplot as plt
from parser import parse_instance_file
from utils import calculate_cost

def get_scheduler(algorithm, instance_data):
    """
    Retorna a instância do scheduler com base no algoritmo escolhido.
    Algoritmos disponíveis: 'sa' (Simulated Annealing), 'tabu' (Tabu Search) e 'ga' (Genetic Algorithm).
    """
    if algorithm == "sa":
        from schedulers.simulated import SimulatedAnnealingScheduler
        return SimulatedAnnealingScheduler(instance_data)
    elif algorithm == "tabu":
        from schedulers.tabu import TabuSearchScheduler
        return TabuSearchScheduler(instance_data)
    elif algorithm == "ga":
        from schedulers.genetic import GeneticAlgorithmScheduler
        return GeneticAlgorithmScheduler(instance_data)
    else:
        raise ValueError("Algoritmo desconhecido. Use 'sa', 'tabu' ou 'ga'.")

def main():
    optimization_method = "ga"  # Altere para "sa", "tabu" ou "ga" conforme desejado

    # Define o diretório base para os resultados
    results_dir = "./results"
    # Cria um diretório para o algoritmo escolhido, ex: ./results/ga
    algorithm_results_dir = os.path.join(results_dir, optimization_method)
    if not os.path.exists(algorithm_results_dir):
        os.makedirs(algorithm_results_dir)

    instance_dir = "./data/instances"
    schedule_results = []  # Armazenar DataFrames de resultados
    metrics_list = []      # Armazenar métricas de desempenho
    images_base_dir = "./images"
    if not os.path.exists(images_base_dir):
        os.makedirs(images_base_dir)

    # Processa cada arquivo de instância
    for filename in os.listdir(instance_dir):
        if filename.endswith(".dat"):
            filepath = os.path.join(instance_dir, filename)
            print(f"Processando instância: {filename}")
            instance_data = parse_instance_file(filepath)

            scheduler = get_scheduler(optimization_method, instance_data)
            best_schedule = scheduler.run()
            
            # Calcula o custo final utilizando a função calculate_cost
            final_cost = calculate_cost(instance_data, best_schedule)

            df_results = pd.DataFrame(best_schedule).T.reset_index()
            df_results.rename(columns={'index': 'patient'}, inplace=True)
            df_results['instance'] = filename
            schedule_results.append(df_results)

            total_patients = len(instance_data['patients'])
            allocated = sum(1 for patient in best_schedule if best_schedule[patient]['ward'] is not None)
            pct_allocated = (allocated / total_patients) * 100

            instance_metrics = {
                "instance": filename,
                "iterations_or_generations": getattr(scheduler, "iterations", getattr(scheduler, "generations", "N/A")),
                "runtime_seconds": getattr(scheduler, "runtime", "N/A"),
                "final_cost": final_cost,
                "pct_allocated": pct_allocated
            }
            metrics_list.append(instance_metrics)

            # Cria diretório para as plots da instância
            instance_name = os.path.splitext(filename)[0]
            instance_image_dir = os.path.join(images_base_dir, instance_name)
            if not os.path.exists(instance_image_dir):
                os.makedirs(instance_image_dir)

            # Plot da evolução do custo
            plt.figure(figsize=(10, 5))
            if optimization_method == "sa":
                plt.plot(scheduler.cost_history, label="Custo")
                plt.xlabel("Iteração")
                plt.title(f"Evolução do Custo (SA) - {instance_name}")
            elif optimization_method == "tabu":
                plt.plot(scheduler.cost_history, label="Custo")
                plt.xlabel("Iteração")
                plt.title(f"Evolução do Custo (Tabu Search) - {instance_name}")
            elif optimization_method == "ga":
                plt.plot(scheduler.cost_history, label="Melhor Custo por Geração")
                plt.xlabel("Geração")
                plt.title(f"Evolução do Custo (Algoritmo Genético) - {instance_name}")
            plt.ylabel("Custo")
            plt.legend()
            plt.grid()
            cost_plot_path = os.path.join(instance_image_dir, "cost_evolution.png")
            plt.savefig(cost_plot_path)
            plt.close() 

            # Plot da evolução da temperatura (para SA)
            if optimization_method == "sa":
                plt.figure(figsize=(10, 5))
                plt.plot(scheduler.temperature_history, label="Temperatura")
                plt.xlabel("Iteração")
                plt.ylabel("Temperatura")
                plt.title(f"Evolução da Temperatura (SA) - {instance_name}")
                plt.legend()
                plt.grid()
                temp_plot_path = os.path.join(instance_image_dir, "temperature_evolution.png")
                plt.savefig(temp_plot_path)
                plt.close()

            print(df_results)
            print("\n" + "="*50 + "\n")

    # Salva resultados combinados e métricas
    all_schedules_df = pd.concat(schedule_results, ignore_index=True)
    all_schedules_df.to_csv("best_schedules.csv", index=False)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv("metrics_results.csv", index=False)

    # Gráfico agregado: Histograma dos custos finais
    plt.figure(figsize=(8, 6))
    plt.hist(metrics_df["final_cost"], bins=20, edgecolor='black')
    plt.xlabel("Custo Final")
    plt.ylabel("Frequência")
    plt.title("Distribuição dos Custos Finais entre as Instâncias")
    plt.grid()
    plt.savefig(os.path.join(algorithm_results_dir, "final_cost_histogram.png"))
    plt.close()

    # Gráfico agregado: Custo Final vs Percentual de Alocação
    plt.figure(figsize=(8, 6))
    plt.scatter(metrics_df["pct_allocated"], metrics_df["final_cost"])
    plt.xlabel("Percentual de Alocação")
    plt.ylabel("Custo Final")
    plt.title("Custo Final vs Percentual de Alocação")
    plt.grid()
    plt.savefig(os.path.join(algorithm_results_dir, "final_cost_vs_allocation.png"))
    plt.close()

    print("Processamento concluído. Resultados salvos em 'best_schedules.csv', 'metrics_results.csv',")
    print("plots individuais na pasta './images' e gráficos agregados em", algorithm_results_dir)

if __name__ == "__main__":
    main()
