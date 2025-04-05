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

def analyze_cost_components(instance_data, solution):
    """
    Analisa e decompõe o custo total em seus componentes.
    Retorna um dicionário com os valores de cada componente.
    """
    # Simplesmente chama a função unificada
    return calculate_cost(instance_data, solution, use_same_weights=True, return_components=True)

def generate_ga_visualizations(scheduler, instance_name, instance_data, output_dir):
    """
    Gera visualizações específicas para o algoritmo genético.
    """
    # Configuração inicial
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.style.use('ggplot')
    
    # 1. Visualização combinada: Custo, Diversidade e Taxa de Mutação
    generate_convergence_visualization(scheduler, instance_name, output_dir)
    
    # 2. Visualização de ocupação das enfermarias
    generate_ward_occupancy_visualization(scheduler, instance_name, instance_data, output_dir)
    
    # 3. Visualização da decomposição do custo
    cost_components = analyze_cost_components(instance_data, scheduler.best_solution)
    generate_cost_breakdown_visualization(cost_components, instance_name, output_dir)
    
    # 4. Visualização adicional: Cirurgias por dia
    generate_surgeries_visualization(scheduler, instance_name, instance_data, output_dir)
    
    # 5. Visualização adicional: Melhorias por geração
    generate_improvements_visualization(scheduler, instance_name, output_dir)
    
    # 6. Nova visualização: Utilização de OT
    generate_ot_utilization_visualization(scheduler, instance_name, instance_data, output_dir)

def generate_convergence_visualization(scheduler, instance_name, output_dir):
    """Gera visualização combinada mostrando convergência."""
    # 1. Gráfico combinado: Custo e Diversidade/Mutação em um único gráfico
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Eixo primário para custo
    color = 'tab:blue'
    ax1.set_xlabel('Geração')
    ax1.set_ylabel('Custo', color=color)
    ax1.plot(scheduler.cost_history, color=color, linewidth=2, label='Custo')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Eixo secundário para diversidade
    if hasattr(scheduler, 'diversity_history') and scheduler.diversity_history:
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Diversidade/Taxa de Mutação', color=color)
        
        # Plotar diversidade
        ax2.plot(scheduler.diversity_history, color=color, linestyle='-', alpha=0.7, label='Diversidade')
        
        # Plotar taxa de mutação
        if hasattr(scheduler, 'mutation_rate_history') and scheduler.mutation_rate_history:
            ax2.plot(scheduler.mutation_rate_history, color='tab:green', linestyle='--', alpha=0.7, label='Taxa de Mutação')
        
        ax2.set_ylim(0, 1.1)  # Ambos diversidade e mutação estão entre 0 e 1
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Legenda combinada
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend()
    
    plt.title(f'Evolução do Algoritmo Genético - {instance_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ga_convergence.png'), dpi=300)
    plt.close()

def generate_ward_occupancy_visualization(scheduler, instance_name, instance_data, output_dir):
    """Gera visualização da ocupação das enfermarias."""
    # Calcular ocupação por dia para cada enfermaria
    days = instance_data['days']
    wards = list(instance_data['wards'].keys())
    
    # Inicializar matriz de ocupação
    daily_occupancy = {ward: [0] * days for ward in wards}
    
    # Calcular ocupação por dia
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
    
    # Criar heatmap
    plt.figure(figsize=(12, 6))
    
    # Preparar dados
    heatmap_data = []
    for ward in wards:
        for day in range(days):
            occupancy = daily_occupancy[ward][day]
            capacity = instance_data['wards'][ward]['bed_capacity']
            
            heatmap_data.append({
                'Ward': ward,
                'Day': f'Dia {day+1}',
                'Occupancy': occupancy,
                'Capacity': capacity,
                'OccupancyRate': occupancy / capacity if capacity > 0 else 0
            })
    
    df_heatmap = pd.DataFrame(heatmap_data)
    
    # Criar pivot table para o heatmap
    pivot = df_heatmap.pivot(index='Ward', columns='Day', values='Occupancy')
    
    # Gerar heatmap de ocupação absoluta
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu', 
               linewidths=0.5)
    
    plt.title(f'Ocupação das Enfermarias - {instance_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ward_occupancy_heatmap.png'), dpi=300)
    plt.close()
    
    # Gerar heatmap de taxa de ocupação
    plt.figure(figsize=(12, 6))
    pivot_rate = df_heatmap.pivot(index='Ward', columns='Day', values='OccupancyRate')
    
    # Criar mapa de cores personalizado para destacar sobrecarga
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    
    sns.heatmap(pivot_rate, annot=True, fmt='.0%', cmap=cmap, 
               linewidths=0.5, vmin=0, vmax=1.2, center=0.6)
    
    plt.title(f'Taxa de Ocupação das Enfermarias - {instance_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ward_occupancy_rate_heatmap.png'), dpi=300)
    plt.close()

def generate_cost_breakdown_visualization(cost_components, instance_name, output_dir):
    """Gera visualização mostrando a decomposição do custo."""
    # Dados para o gráfico de pizza
    labels = ['Capacidade das\nEnfermarias', 'Conflitos de\nCirurgia', 'Atraso de\nAdmissão', 'Uso de OT']
    values = [
        cost_components['bed_capacity_cost'], 
        cost_components['surgery_conflict_cost'],
        cost_components['delay_cost'],
        cost_components['ot_cost']
    ]
    
    # Verificar a consistência dos valores totais
    total = sum(values)
    reported_total = cost_components['total_cost']
    
    if abs(total - reported_total) > 0.1:
        print(f"AVISO: Discrepância no custo total. Visualização: {total}, Relatado: {reported_total}")
        # Ajustar os valores para corresponder ao custo total reportado
        scaling_factor = reported_total / total if total > 0 else 1
        values = [v * scaling_factor for v in values]
    
    # Remover componentes com valor zero
    non_zero_labels = []
    non_zero_values = []
    for label, value in zip(labels, values):
        if value > 0:
            non_zero_labels.append(label)
            non_zero_values.append(value)
    
    if not non_zero_values:  # Se não houver valores positivos
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Todos os componentes de custo são zero", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'cost_breakdown.png'), dpi=300)
        plt.close()
        return
    
    # Cores
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    colors = colors[:len(non_zero_labels)]
    
    # Gerar gráfico de pizza
    fig, ax = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax.pie(
        non_zero_values, 
        labels=non_zero_labels, 
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )
    
    # Configurar propriedades do texto
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('black')
    
    ax.axis('equal')  # Garantir que o gráfico seja um círculo
    plt.title(f'Decomposição do Custo - {instance_name}', fontsize=14)
    
    # Adicionar legenda com valores absolutos
    legend_labels = []
    for label, value in zip(non_zero_labels, non_zero_values):
        legend_labels.append(f'{label}: {value:.1f}')
    
    if legend_labels:
        plt.legend(wedges, legend_labels, title="Componentes do Custo", 
                 loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_breakdown.png'), dpi=300)
    plt.close()

def generate_surgeries_visualization(scheduler, instance_name, instance_data, output_dir):
    """Gera visualização das cirurgias agendadas."""
    days = instance_data['days']
    
    # Contar cirurgias por dia
    surgeries_by_day = [0] * days
    for patient, data in scheduler.best_solution.items():
        if data['ward'] is not None and data['day'] >= 0 and data['day'] < days:
            surgeries_by_day[data['day']] += 1
    
    # Contar cirurgias por especialização e dia
    spec_surgeries = {}
    for patient, data in scheduler.best_solution.items():
        if data['ward'] is not None and data['day'] >= 0 and data['day'] < days:
            spec = instance_data['patients'][patient]['specialization']
            if spec not in spec_surgeries:
                spec_surgeries[spec] = [0] * days
            spec_surgeries[spec][data['day']] += 1
    
    # Visualização: Total de cirurgias por dia
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(1, days+1), surgeries_by_day, color='#1f77b4')
    
    # Adicionar linha para capacidade de cirurgias
    capacity = len(instance_data['specializations'])
    plt.axhline(y=capacity, color='red', linestyle='--', label=f'Capacidade ({capacity})')
    
    plt.title(f'Agendamento de Cirurgias por Dia - {instance_name}', fontsize=14)
    plt.xlabel('Dia')
    plt.ylabel('Número de Cirurgias')
    plt.xticks(range(1, days+1))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Adicionar rótulos
    for i, count in enumerate(surgeries_by_day):
        color = 'red' if count > capacity else 'black'
        plt.text(i+1, count, str(count), ha='center', va='bottom', color=color)
    
    plt.savefig(os.path.join(output_dir, 'surgeries_by_day.png'), dpi=300)
    plt.close()

def generate_improvements_visualization(scheduler, instance_name, output_dir):
    """Gera visualização das melhorias por geração."""
    if len(scheduler.cost_history) > 1:
        # Calcular melhorias absolutas
        improvements = [0]
        for i in range(1, len(scheduler.cost_history)):
            improvement = max(0, scheduler.cost_history[i-1] - scheduler.cost_history[i])
            improvements.append(improvement)
        
        # Calcular melhorias relativas (%)
        relative_improvements = [0]
        for i in range(1, len(scheduler.cost_history)):
            if scheduler.cost_history[i-1] > 0:  # Evitar divisão por zero
                rel_improvement = 100 * (scheduler.cost_history[i-1] - scheduler.cost_history[i]) / scheduler.cost_history[i-1]
                relative_improvements.append(rel_improvement)
            else:
                relative_improvements.append(0)
        
        # Gráfico de melhorias absolutas
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(range(len(improvements)), improvements, color='#2ca02c', alpha=0.7)
        plt.xlabel('Geração')
        plt.ylabel('Melhoria Absoluta')
        plt.title(f'Melhoria Absoluta por Geração - {instance_name}', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Gráfico de melhorias relativas
        plt.subplot(1, 2, 2)
        plt.bar(range(len(relative_improvements)), relative_improvements, color='#d62728', alpha=0.7)
        plt.xlabel('Geração')
        plt.ylabel('Melhoria Relativa (%)')
        plt.title(f'Melhoria Relativa por Geração - {instance_name}', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'improvements.png'), dpi=300)
        plt.close()

def generate_ot_utilization_visualization(scheduler, instance_name, instance_data, output_dir):
    """Gera visualização da utilização de tempo operatório (OT)."""
    days = instance_data['days']
    specs = list(instance_data['specializations'].keys())
    
    # Calcular utilização de OT por dia e especialização
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
    
    # Obter OT disponível
    for spec in specs:
        for day in range(min(len(instance_data['specializations'][spec]['available_ot']), days)):
            ot_available[spec][day] = instance_data['specializations'][spec]['available_ot'][day]
    
    # Criar visualização agregada de utilização de OT
    plt.figure(figsize=(12, 6))
    
    # Calcular taxa de utilização por dia
    utilization_rates = []
    
    for day in range(days):
        day_usage = sum(ot_usage[spec][day] for spec in specs)
        day_available = sum(ot_available[spec][day] for spec in specs if day < len(ot_available[spec]))
        
        if day_available > 0:
            utilization_rates.append(day_usage / day_available * 100)
        else:
            utilization_rates.append(0)
    
    # Criar barras
    bars = plt.bar(range(1, days+1), utilization_rates, color='skyblue')
    
    # Adicionar linha de 100%
    plt.axhline(y=100, color='red', linestyle='--', label='Utilização Ideal')
    
    # Colorir barras baseado na taxa
    for i, rate in enumerate(utilization_rates):
        if rate > 110:  # Overtime significativo
            bars[i].set_color('crimson')
        elif rate > 95:  # Próximo do ideal
            bars[i].set_color('limegreen')
        elif rate < 70:  # Undertime significativo
            bars[i].set_color('orange')
    
    plt.xlabel('Dia')
    plt.ylabel('Taxa de Utilização (%)')
    plt.title(f'Utilização de Tempo Operatório por Dia - {instance_name}', fontsize=14)
    plt.xticks(range(1, days+1))
    plt.grid(axis='y', alpha=0.3)
    
    # Adicionar rótulos
    for i, rate in enumerate(utilization_rates):
        plt.text(i+1, rate+2, f"{rate:.1f}%", ha='center')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ot_utilization_rate.png'), dpi=300)
    plt.close()
    
    # Criar visualizações individuais por especialização
    for spec in specs:
        plt.figure(figsize=(10, 6))
        
        # Preparar dados para as barras
        days_range = range(1, min(days+1, len(ot_available[spec])+1))
        usage_values = [ot_usage[spec][d-1] for d in days_range]
        available_values = [ot_available[spec][d-1] for d in days_range]
        
        # Calcular taxa de utilização
        util_rates = [usage/avail*100 if avail > 0 else 0 for usage, avail in zip(usage_values, available_values)]
        
        # Plotar barras de utilização vs disponibilidade
        x = np.arange(len(days_range))
        width = 0.35
        
        plt.bar(x - width/2, usage_values, width, label='OT Utilizado', color='skyblue')
        plt.bar(x + width/2, available_values, width, label='OT Disponível', color='lightgray')
        
        # Adicionar linha de taxas de utilização
        ax2 = plt.twinx()
        ax2.plot(x, util_rates, 'r-', label='Taxa de Utilização', linewidth=2)
        ax2.set_ylim(0, max(max(util_rates)*1.1, 110))
        ax2.set_ylabel('Taxa de Utilização (%)')
        
        # Configurar eixos e labels
        plt.xlabel('Dia')
        plt.ylabel('Tempo Operatório (minutos)')
        plt.title(f'Utilização de OT - {spec} - {instance_name}', fontsize=14)
        plt.xticks(x, days_range)
        
        # Adicionar segunda legenda
        lines, labels = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'ot_utilization_{spec}.png'), dpi=300)
        plt.close()

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
            # Usar use_same_weights=True para garantir consistência com analyze_cost_components
            final_cost = calculate_cost(instance_data, best_schedule, use_same_weights=True)

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
            
            # Gerar visualizações adicionais para o algoritmo genético
            if optimization_method == "ga":
                generate_ga_visualizations(scheduler, instance_name, instance_data, instance_image_dir)
                
                # Analise e exiba componentes do custo
                cost_components = analyze_cost_components(instance_data, best_schedule)
                print("\nComponentes do custo final:")
                print(f"  Capacidade das Enfermarias: {cost_components['bed_capacity_cost']:.2f} ({cost_components['pct_bed']:.1f}%)")
                print(f"  Conflitos de Cirurgia: {cost_components['surgery_conflict_cost']:.2f} ({cost_components['pct_surgery']:.1f}%)")
                print(f"  Atraso de Admissão: {cost_components['delay_cost']:.2f} ({cost_components['pct_delay']:.1f}%)")
                print(f"  Uso de OT: {cost_components['ot_cost']:.2f} ({cost_components['pct_ot']:.1f}%)")
                print(f"  Total: {cost_components['total_cost']:.2f}")

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