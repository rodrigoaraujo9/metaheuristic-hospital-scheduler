import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def generate_advanced_analysis(instance_data, solution, output_dir):
    """
    Gera análises avançadas e visualizações para uma solução.
    
    Parâmetros:
    - instance_data: Dados da instância
    - solution: Solução a ser analisada
    - output_dir: Diretório para salvar as visualizações
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Análise de ocupação por especialização
    analyze_occupancy_by_specialization(instance_data, solution, output_dir)
    
    # 2. Análise da distribuição das admissões
    analyze_admission_distribution(instance_data, solution, output_dir)
    
    # 3. Análise do equilíbrio de carga
    analyze_workload_balance(instance_data, solution, output_dir)
    
    # 4. Análise da utilização de OT
    analyze_ot_utilization(instance_data, solution, output_dir)

def analyze_occupancy_by_specialization(instance_data, solution, output_dir):
    """Analisa a ocupação de leitos por especialização."""
    # Inicializar estruturas de dados
    specs = list(instance_data['specializations'].keys())
    days = instance_data['days']
    
    # Ocupação por especialização e dia
    spec_occupancy = {spec: [0] * days for spec in specs}
    
    # Calcular ocupação
    for patient, data in solution.items():
        if data['ward'] is None or data['day'] < 0:
            continue
            
        spec = instance_data['patients'][patient]['specialization']
        admission_day = data['day']
        length_of_stay = instance_data['patients'][patient]['length_of_stay']
        
        for day_offset in range(length_of_stay):
            day = admission_day + day_offset
            if day < days and spec in spec_occupancy:
                spec_occupancy[spec][day] += 1
    
    # Criar DataFrame para visualização
    df_data = []
    for spec in specs:
        for day in range(days):
            df_data.append({
                'Specialization': spec,
                'Day': day + 1,
                'Occupancy': spec_occupancy[spec][day]
            })
    
    df = pd.DataFrame(df_data)
    
    # Criar heatmap
    plt.figure(figsize=(12, 8))
    pivot = df.pivot(index='Specialization', columns='Day', values='Occupancy')
    
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu', linewidths=0.5)
    plt.title('Ocupação por Especialização ao Longo do Tempo', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'specialization_occupancy_heatmap.png'), dpi=300)
    plt.close()
    
    # Criar gráfico de linhas
    plt.figure(figsize=(12, 6))
    for spec in specs:
        plt.plot(range(1, days+1), spec_occupancy[spec], label=spec, marker='o')
    
    plt.xlabel('Dia')
    plt.ylabel('Número de Pacientes')
    plt.title('Evolução da Ocupação por Especialização', fontsize=14)
    plt.legend(title='Especialização')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, days+1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'specialization_occupancy_line.png'), dpi=300)
    plt.close()

def analyze_admission_distribution(instance_data, solution, output_dir):
    """Analisa a distribuição das admissões ao longo do horizonte de planejamento."""
    days = instance_data['days']
    
    # Contar admissões por dia
    admissions = [0] * days
    postponed_admissions = [0] * days
    
    for patient, data in solution.items():
        if data['ward'] is None or data['day'] < 0:
            continue
            
        # Contar admissão
        day = data['day']
        if day < days:
            admissions[day] += 1
            
            # Verificar se foi adiada
            earliest = instance_data['patients'][patient]['earliest_admission']
            if day > earliest:
                postponed_admissions[day] += 1
    
    # Criar gráfico combinado
    plt.figure(figsize=(12, 6))
    
    # Barras para todas as admissões
    bars = plt.bar(range(1, days+1), admissions, alpha=0.7, label='Total Admissões')
    
    # Barras para admissões adiadas
    postponed_bars = plt.bar(range(1, days+1), postponed_admissions, alpha=0.5, color='orange', label='Admissões Adiadas')
    
    plt.xlabel('Dia')
    plt.ylabel('Número de Admissões')
    plt.title('Distribuição das Admissões ao Longo do Tempo', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, days+1))
    
    # Adicionar valores
    for i, (total, postponed) in enumerate(zip(admissions, postponed_admissions)):
        plt.text(i+1, total + 0.3, str(total), ha='center')
        if postponed > 0:
            plt.text(i+1, postponed/2, str(postponed), ha='center', color='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'admission_distribution.png'), dpi=300)
    plt.close()
    
    # Análise de atrasos
    delays = []
    for patient, data in solution.items():
        if data['ward'] is None or data['day'] < 0:
            continue
            
        earliest = instance_data['patients'][patient]['earliest_admission']
        actual = data['day']
        delay = actual - earliest
        
        if delay > 0:
            delays.append(delay)
    
    if delays:
        plt.figure(figsize=(10, 6))
        plt.hist(delays, bins=range(max(delays)+2), alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Dias de Atraso')
        plt.ylabel('Número de Pacientes')
        plt.title('Distribuição dos Atrasos de Admissão', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(max(delays)+1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'admission_delay_distribution.png'), dpi=300)
        plt.close()

def analyze_workload_balance(instance_data, solution, output_dir):
    """Analisa o equilíbrio de carga de trabalho entre enfermarias."""
    days = instance_data['days']
    wards = list(instance_data['wards'].keys())
    
    # Calcular carga de trabalho por enfermaria e dia
    ward_workloads = {ward: [0] * days for ward in wards}
    
    for patient, data in solution.items():
        if data['ward'] is None or data['day'] < 0:
            continue
            
        ward = data['ward']
        admission_day = data['day']
        patient_data = instance_data['patients'][patient]
        workload_values = patient_data['workload']
        
        for day_offset, workload in enumerate(workload_values):
            day = admission_day + day_offset
            if day < days:
                ward_workloads[ward][day] += workload
    
    # Adicionar carryover workload
    for ward in wards:
        if 'carryover_workload' in instance_data['wards'][ward]:
            for day, workload in enumerate(instance_data['wards'][ward]['carryover_workload']):
                if day < days:
                    ward_workloads[ward][day] += workload
    
    # Criar DataFrame para visualização
    df_data = []
    for ward in wards:
        capacity = instance_data['wards'][ward]['workload_capacity']
        for day in range(days):
            workload = ward_workloads[ward][day]
            utilization = workload / capacity if capacity > 0 else 0
            
            df_data.append({
                'Ward': ward,
                'Day': day + 1,
                'Workload': workload,
                'Capacity': capacity,
                'Utilization': utilization
            })
    
    df = pd.DataFrame(df_data)
    
    # Heatmap de carga absoluta
    plt.figure(figsize=(12, 6))
    pivot_abs = df.pivot(index='Ward', columns='Day', values='Workload')
    
    sns.heatmap(pivot_abs, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=0.5)
    plt.title('Carga de Trabalho por Enfermaria', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'workload_heatmap.png'), dpi=300)
    plt.close()
    
    # Heatmap de utilização (%)
    plt.figure(figsize=(12, 6))
    pivot_util = df.pivot(index='Ward', columns='Day', values='Utilization')
    
    # Criar mapa de cores personalizado
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    
    sns.heatmap(pivot_util, annot=True, fmt='.0%', cmap=cmap, 
               linewidths=0.5, vmin=0, vmax=1.2, center=0.6)
    
    plt.title('Taxa de Utilização por Enfermaria', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'workload_utilization_heatmap.png'), dpi=300)
    plt.close()
    
    # Gráfico de linhas da carga média por dia
    plt.figure(figsize=(12, 6))
    
    for ward in wards:
        capacity = instance_data['wards'][ward]['workload_capacity']
        utilization = [workload / capacity if capacity > 0 else 0 for workload in ward_workloads[ward]]
        plt.plot(range(1, days+1), utilization, label=f"{ward} (Cap: {capacity})", marker='o')
    
    plt.axhline(y=1.0, color='red', linestyle='--', label='Capacidade Máxima')
    plt.xlabel('Dia')
    plt.ylabel('Taxa de Utilização')
    plt.title('Evolução da Utilização de Carga por Enfermaria', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, days+1))
    plt.ylim(0, max(1.2, plt.ylim()[1]))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'workload_utilization_line.png'), dpi=300)
    plt.close()

def analyze_ot_utilization(instance_data, solution, output_dir):
    """Analisa a utilização de tempo operatório (OT)."""
    days = instance_data['days']
    specs = list(instance_data['specializations'].keys())
    
    # Calcular uso de OT por especialização e dia
    ot_usage = {spec: [0] * days for spec in specs}
    
    for patient, data in solution.items():
        if data['ward'] is None or data['day'] < 0:
            continue
            
        spec = instance_data['patients'][patient]['specialization']
        day = data['day']
        
        if day < days and spec in ot_usage:
            surgery_duration = instance_data['patients'][patient]['surgery_duration']
            ot_usage[spec][day] += surgery_duration
    
    # Gráficos individuais por especialização
    for spec in specs:
        available_ot = instance_data['specializations'][spec]['available_ot']
        max_days = min(days, len(available_ot))
        
        plt.figure(figsize=(12, 6))
        
        # Barras empilhadas: usado e disponível
        used = [min(ot_usage[spec][d], available_ot[d]) if d < len(available_ot) else 0 for d in range(max_days)]
        unused = [max(0, available_ot[d] - ot_usage[spec][d]) if d < len(available_ot) else 0 for d in range(max_days)]
        overtime = [max(0, ot_usage[spec][d] - available_ot[d]) if d < len(available_ot) else 0 for d in range(max_days)]
        
        plt.bar(range(1, max_days+1), used, color='#66b3ff', label='OT Utilizado')
        plt.bar(range(1, max_days+1), unused, bottom=used, color='#c2c2d6', label='OT Disponível Não Utilizado')
        
        if any(ot > 0 for ot in overtime):
            plt.bar(range(1, max_days+1), overtime, bottom=[available_ot[d] if d < len(available_ot) else 0 for d in range(max_days)], 
                    color='#ff6666', label='OT Excedido (Overtime)')
        
        plt.axhline(y=sum(available_ot[:max_days])/max_days if max_days > 0 else 0, 
                   color='red', linestyle='--', label='Média de OT Disponível')
        
        plt.xlabel('Dia')
        plt.ylabel('Tempo Operatório (minutos)')
        plt.title(f'Utilização de Tempo Operatório - {spec}', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, max_days+1))
        
        # Adicionar percentuais de utilização
        for d in range(max_days):
            if available_ot[d] > 0:
                utilization = ot_usage[spec][d] / available_ot[d] * 100
                y_pos = used[d] / 2 if used[d] > 0 else 10
                plt.text(d+1, y_pos, f"{utilization:.0f}%", ha='center', 
                        color='white' if utilization > 50 else 'black')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'ot_utilization_{spec}.png'), dpi=300)
        plt.close()
    
    # Gráfico agregado
    plt.figure(figsize=(14, 8))
    
    # Preparar dados agregados
    df_data = []
    for spec in specs:
        available_ot = instance_data['specializations'][spec]['available_ot']
        for day in range(min(days, len(available_ot))):
            used = ot_usage[spec][day]
            available = available_ot[day]
            utilization = used / available if available > 0 else 0
            
            df_data.append({
                'Specialization': spec,
                'Day': day + 1,
                'Used': used,
                'Available': available,
                'Utilization': utilization,
                'Status': 'Overtime' if used > available else 'Undertime' if used < available else 'Balanced'
            })
    
    df = pd.DataFrame(df_data)
    
    # Gráfico de calor da utilização
    plt.figure(figsize=(12, 8))
    pivot_util = df.pivot(index='Specialization', columns='Day', values='Utilization')
    
    # Criar mapa de cores divergente
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    
    sns.heatmap(pivot_util, annot=True, fmt='.0%', cmap=cmap, 
               linewidths=0.5, vmin=0, vmax=2.0, center=1.0)
    
    plt.title('Taxa de Utilização de Tempo Operatório', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ot_utilization_heatmap.png'), dpi=300)
    plt.close()
    
    # Gráfico de barras da média de utilização por especialização
    avg_util = df.groupby('Specialization')['Utilization'].mean()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(avg_util.index, avg_util.values, color=[
        '#66b3ff' if val <= 0.8 else '#66ff66' if val <= 1.1 else '#ff6666' for val in avg_util.values
    ])
    
    plt.axhline(y=1.0, color='red', linestyle='--', label='Utilização Ideal')
    plt.xlabel('Especialização')
    plt.ylabel('Taxa Média de Utilização')
    plt.title('Utilização Média de Tempo Operatório por Especialização', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Adicionar valores
    for i, v in enumerate(avg_util.values):
        plt.text(i, v + 0.05, f"{v:.0%}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ot_utilization_by_spec.png'), dpi=300)
    plt.close()
    
    # Distribuição por status (Overtime/Undertime/Balanced)
    status_counts = df['Status'].value_counts()
    
    plt.figure(figsize=(8, 8))
    plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%',
            colors=['#ff6666', '#66ff66', '#66b3ff'])
    plt.title('Distribuição dos Status de Utilização de OT', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ot_status_distribution.png'), dpi=300)
    plt.close()

def analyze_and_compare_solutions(instance_data, solutions, names, output_dir):
    """
    Analisa e compara múltiplas soluções.
    
    Parâmetros:
    - instance_data: Dados da instância
    - solutions: Lista de soluções a serem comparadas
    - names: Lista de nomes para as soluções
    - output_dir: Diretório para salvar as visualizações
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Verificar se o número de soluções e nomes é consistente
    if len(solutions) != len(names):
        raise ValueError("O número de soluções e nomes deve ser igual")
    
    # Comparar custos
    from utils import calculate_cost
    from main import analyze_cost_components
    
    costs = []
    cost_components = []
    
    for i, solution in enumerate(solutions):
        try:
            total_cost = calculate_cost(instance_data, solution, use_same_weights=True)
            components = analyze_cost_components(instance_data, solution)
            
            costs.append({
                'Solution': names[i],
                'Cost': total_cost
            })
            
            cost_components.append({
                'Solution': names[i],
                'Bed Capacity Cost': components['bed_capacity_cost'],
                'Surgery Conflict Cost': components['surgery_conflict_cost'],
                'Delay Cost': components['delay_cost'],
                'OT Cost': components['ot_cost'],
                'Total': components['total_cost']
            })
        except Exception as e:
            print(f"Erro ao calcular custo da solução {names[i]}: {str(e)}")
    
    # Se não conseguiu calcular nenhum custo, sair
    if not costs:
        print("Não foi possível calcular o custo para nenhuma solução")
        return
    
    # Criar gráfico de barras para custos totais
    df_costs = pd.DataFrame(costs)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_costs['Solution'], df_costs['Cost'])
    
    # Colorir a melhor solução
    best_idx = df_costs['Cost'].idxmin()
    bars[best_idx].set_color('green')
    
    plt.xlabel('Solução')
    plt.ylabel('Custo Total')
    plt.title('Comparação de Custos entre Soluções', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Rotacionar labels se houver muitas soluções
    if len(names) > 3:
        plt.xticks(rotation=45)
    
    # Adicionar valores
    for i, v in enumerate(df_costs['Cost']):
        plt.text(i, v + 0.01 * max(df_costs['Cost']), f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_comparison.png'), dpi=300)
    plt.close()
    
    # Comparar componentes de custo
    df_components = pd.DataFrame(cost_components)
    
    # Gráfico de barras empilhadas
    plt.figure(figsize=(12, 8))
    
    bottom = np.zeros(len(names))
    for component in ['Bed Capacity Cost', 'Surgery Conflict Cost', 'Delay Cost', 'OT Cost']:
        plt.bar(names, df_components[component], bottom=bottom, label=component)
        bottom += df_components[component].values
    
    plt.xlabel('Solução')
    plt.ylabel('Custo')
    plt.title('Componentes de Custo por Solução', fontsize=14)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Rotacionar labels se houver muitas soluções
    if len(names) > 3:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_components_comparison.png'), dpi=300)
    plt.close()
    
    # Gráfico de distribuição percentual
    plt.figure(figsize=(15, 8))
    
    for i, solution in enumerate(names):
        plt.subplot(1, len(names), i+1)
        
        values = [
            df_components.loc[i, 'Bed Capacity Cost'],
            df_components.loc[i, 'Surgery Conflict Cost'],
            df_components.loc[i, 'Delay Cost'],
            df_components.loc[i, 'OT Cost']
        ]
        
        labels = ['Capacity', 'Conflicts', 'Delay', 'OT']
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        # Filtrar componentes zero
        non_zero_values = []
        non_zero_labels = []
        non_zero_colors = []
        
        for j, (val, lbl, col) in enumerate(zip(values, labels, colors)):
            if val > 0:
                non_zero_values.append(val)
                non_zero_labels.append(lbl)
                non_zero_colors.append(col)
        
        if non_zero_values:
            plt.pie(non_zero_values, labels=non_zero_labels, autopct='%1.1f%%', colors=non_zero_colors)
        else:
            plt.text(0.5, 0.5, "Todos os componentes são zero", ha='center', va='center')
            plt.axis('off')
        
        plt.title(solution)
    
    plt.suptitle('Distribuição dos Componentes de Custo', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_distribution_comparison.png'), dpi=300)
    plt.close()
    
    # Comparar alocação de pacientes
    allocation_stats = []
    
    for i, solution in enumerate(solutions):
        allocated = sum(1 for patient, data in solution.items() if data['ward'] is not None)
        total = len(instance_data['patients'])
        allocation_stats.append({
            'Solution': names[i],
            'Allocated': allocated,
            'Total': total,
            'Percentage': allocated / total * 100 if total > 0 else 0
        })
    
    df_allocation = pd.DataFrame(allocation_stats)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_allocation['Solution'], df_allocation['Percentage'])
    
    # Colorir barras com base na taxa de alocação
    for i, bar in enumerate(bars):
        if df_allocation.iloc[i]['Percentage'] >= 95:
            bar.set_color('green')
        elif df_allocation.iloc[i]['Percentage'] >= 80:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.xlabel('Solução')
    plt.ylabel('Taxa de Alocação (%)')
    plt.title('Comparação da Taxa de Alocação de Pacientes', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 105)  # Garantir que 100% seja visível
    
    # Rotacionar labels se houver muitas soluções
    if len(names) > 3:
        plt.xticks(rotation=45)
    
    # Adicionar valores
    for i, v in enumerate(df_allocation['Percentage']):
        allocated = int(df_allocation.iloc[i]['Allocated'])
        total = int(df_allocation.iloc[i]['Total'])
        plt.text(i, v + 2, f"{v:.1f}%\n({allocated}/{total})", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'allocation_comparison.png'), dpi=300)
    plt.close()