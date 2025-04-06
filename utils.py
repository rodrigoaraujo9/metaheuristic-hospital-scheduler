import random
import copy
import math

def generate_initial_solution(instance_data):
    """
    Gera uma solução inicial mais inteligente considerando especialidades e capacidades.
    """
    solution = {}
    
    # Mapear wards por especialização
    specialization_to_wards = {}
    for ward_id, ward_data in instance_data['wards'].items():
        major_spec = ward_data['major_specialization']
        if major_spec not in specialization_to_wards:
            specialization_to_wards[major_spec] = []
        specialization_to_wards[major_spec].append(ward_id)
        
        for minor_spec in ward_data['minor_specializations']:
            if minor_spec != "NONE" and minor_spec:
                if minor_spec not in specialization_to_wards:
                    specialization_to_wards[minor_spec] = []
                specialization_to_wards[minor_spec].append(ward_id)
    
    # Ordenar pacientes por flexibilidade de admissão (janela de tempo mais curta primeiro)
    patients_by_flexibility = sorted(
        instance_data['patients'].keys(),
        key=lambda p: (
            instance_data['patients'][p]['latest_admission'] - 
            instance_data['patients'][p]['earliest_admission']
        )
    )
    
    # Manter o controle da ocupação dos wards por dia
    ward_occupancy = {
        ward: [0] * instance_data['days'] 
        for ward in instance_data['wards']
    }
    
    # Manter o controle das cirurgias agendadas por dia
    surgeries_per_day = [0] * instance_data['days']
    
    # Manter o controle do uso de OT por especialização e dia
    spec_ot_usage = {
        spec: [0] * instance_data['days'] 
        for spec in instance_data['specializations']
    }
    
    # Atribuir pacientes
    for patient in patients_by_flexibility:
        patient_data = instance_data['patients'][patient]
        spec = patient_data['specialization']
        
        # Tentar atribuir ao ward compatível
        compatible_wards = specialization_to_wards.get(spec, list(instance_data['wards'].keys()))
        if not compatible_wards:
            compatible_wards = list(instance_data['wards'].keys())
        
        # Calcular pontuação para cada possível atribuição (ward, dia)
        best_score = float('inf')
        best_ward = None
        best_day = None
        
        for ward in compatible_wards:
            ward_capacity = instance_data['wards'][ward]['bed_capacity']
            
            for day in range(
                patient_data['earliest_admission'],
                min(patient_data['latest_admission'] + 1, instance_data['days'])
            ):
                # Verificar se o ward tem capacidade para este paciente em todos os dias necessários
                can_fit = True
                for d in range(day, min(day + patient_data['length_of_stay'], instance_data['days'])):
                    if ward_occupancy[ward][d] >= ward_capacity:
                        can_fit = False
                        break
                
                if can_fit:
                    # Calcular pontuação (menor é melhor)
                    # Considerar cirurgias no dia
                    surgery_score = surgeries_per_day[day]
                    
                    # Considerar utilização de OT
                    # Esta é uma nova melhoria para balancear melhor o uso de OT
                    if spec in spec_ot_usage and day < len(spec_ot_usage[spec]):
                        current_ot = spec_ot_usage[spec][day]
                        available_ot = instance_data['specializations'][spec]['available_ot'][day] if day < len(instance_data['specializations'][spec]['available_ot']) else 0
                        
                        # Preferir dias com menor diferença entre OT disponível e utilizado
                        ot_score = abs(available_ot - (current_ot + patient_data['surgery_duration']))
                        ot_score /= (available_ot + 1)  # Normalizar por capacidade
                    else:
                        ot_score = 0
                    
                    # Preferência para especialidade principal
                    spec_penalty = 0 if spec == instance_data['wards'][ward]['major_specialization'] else 5
                    
                    # Penalizar dias mais tardios (preferir admissão mais cedo)
                    day_penalty = (day - patient_data['earliest_admission']) * 0.5
                    
                    total_score = surgery_score + spec_penalty + day_penalty + ot_score * 3
                    
                    if total_score < best_score:
                        best_score = total_score
                        best_ward = ward
                        best_day = day
        
        # Se não encontrar uma atribuição viável, escolher a menos ruim
        if best_ward is None:
            best_ward = random.choice(compatible_wards) if compatible_wards else random.choice(list(instance_data['wards'].keys()))
            best_day = random.randint(
                patient_data['earliest_admission'],
                patient_data['latest_admission']
            )
        
        # Atualizar solução
        solution[patient] = {'ward': best_ward, 'day': best_day}
        
        # Atualizar contadores
        for d in range(best_day, min(best_day + patient_data['length_of_stay'], instance_data['days'])):
            ward_occupancy[best_ward][d] += 1
        
        surgeries_per_day[best_day] += 1
        
        # Atualizar uso de OT
        if best_day < instance_data['days'] and spec in spec_ot_usage:
            spec_ot_usage[spec][best_day] += patient_data['surgery_duration']
    
    return solution

def calculate_cost(instance_data, solution, use_same_weights=False, return_components=False):
    """
    Função de custo simplificada focada apenas nos dois objetivos principais:
    1. Minimizar dia de admissão
    2. Minimizar OT overtime
    """
    # Inicializar componentes de custo
    delay_cost = 0
    ot_overtime_cost = 0
    
    # Para utilização de tempo de cirurgia
    specialization_surgeries = {spec: [0] * instance_data['days'] for spec in instance_data['specializations']}
    
    # Calcular ocupação e carga de trabalho
    for patient, data in solution.items():
        if data['ward'] is None or data['day'] < 0:
            continue  # Pular pacientes não alocados
            
        admission_day = data['day']
        patient_data = instance_data['patients'][patient]
        spec = patient_data['specialization']
        
        # Adicionar à utilização de tempo de cirurgia
        if admission_day < instance_data['days']:
            surgery_duration = patient_data['surgery_duration']
            specialization_surgeries[spec][admission_day] += surgery_duration
        
        # Calcular atraso na admissão
        earliest = patient_data['earliest_admission']
        delay = admission_day - earliest
        if delay > 0:
            delay_cost += delay * 10  # Peso maior para day minimization
    
    # Calcular penalidades apenas para OT overtime (não undertime)
    for spec in instance_data['specializations']:
        for day in range(min(len(instance_data['specializations'][spec]['available_ot']), instance_data['days'])):
            used_ot = specialization_surgeries[spec][day]
            available_ot = instance_data['specializations'][spec]['available_ot'][day]
            
            # Penalidade por overtime
            if used_ot > available_ot:
                ot_overtime_cost += (used_ot - available_ot) * 8  # Peso para overtime
    
    # Custo total (apenas os dois componentes solicitados)
    total_cost = delay_cost + ot_overtime_cost
    
    if return_components:
        return {
            'total': total_cost,
            'delay': delay_cost,
            'ot_overtime': ot_overtime_cost
        }
    
    return total_cost
    
def make_neighbor_solution(instance_data, current_solution, strategy="mixed"):
    """
    Gera uma solução vizinha alterando a atribuição de um paciente.
    """
    new_solution = copy.deepcopy(current_solution)
    
    # Se não houver pacientes na solução, retornar a solução sem alterações
    if not new_solution:
        return new_solution
    
    # Definir a estratégia de vizinhança se for "mixed"
    if strategy == "mixed":
        strategy = random.choice(["ward", "day", "both", "smart", "ot_balancing"])
    
    # Selecionar um paciente aleatoriamente para modificar
    patient = random.choice(list(new_solution.keys()))
    
    # Obter dados do paciente
    patient_data = instance_data['patients'].get(patient)
    if not patient_data:
        return new_solution
    
    # Especialização do paciente
    spec = patient_data['specialization']
    
    # Estratégia "ot_balancing": focar em balancear o tempo operatório
    if strategy == "ot_balancing":
        surgery_duration = patient_data['surgery_duration']
        
        # Calcular uso atual de OT por dia
        ot_usage = {day: 0 for day in range(instance_data['days'])}
        ot_available = {day: 0 for day in range(instance_data['days'])}
        
        for p, d in current_solution.items():
            if p != patient and d['ward'] is not None and d['day'] >= 0:
                p_spec = instance_data['patients'][p]['specialization']
                if p_spec == spec:
                    p_duration = instance_data['patients'][p]['surgery_duration']
                    if d['day'] < instance_data['days']:
                        ot_usage[d['day']] += p_duration
        
        # Obter OT disponível
        for day in range(min(len(instance_data['specializations'][spec]['available_ot']), instance_data['days'])):
            ot_available[day] = instance_data['specializations'][spec]['available_ot'][day]
        
        # Identificar dias com melhor equilíbrio
        best_days = []
        for day in range(patient_data['earliest_admission'], patient_data['latest_admission'] + 1):
            if day < instance_data['days']:
                # Calcular desequilíbrio atual e futuro
                current_imbalance = abs(ot_available.get(day, 0) - ot_usage.get(day, 0))
                future_imbalance = abs(ot_available.get(day, 0) - (ot_usage.get(day, 0) + surgery_duration))
                
                # Se adicionar este paciente melhora o equilíbrio
                if future_imbalance < current_imbalance:
                    best_days.append(day)
        
        if best_days:
            new_solution[patient]['day'] = random.choice(best_days)
        else:
            # Se não encontrou dias que melhoram o equilíbrio, escolher aleatoriamente
            earliest = patient_data['earliest_admission']
            latest = patient_data['latest_admission']
            new_solution[patient]['day'] = random.randint(earliest, latest)
    
    # Estratégia "smart": tentar atribuir ao ward compatível com a especialização
    elif strategy == "smart":
        # Encontrar wards compatíveis com a especialização
        compatible_wards = []
        for ward_id, ward_data in instance_data['wards'].items():
            if spec == ward_data['major_specialization'] or spec in ward_data['minor_specializations']:
                compatible_wards.append(ward_id)
        
        if compatible_wards:
            new_solution[patient]['ward'] = random.choice(compatible_wards)
        else:
            new_solution[patient]['ward'] = random.choice(list(instance_data['wards'].keys()))
        
        # Ajustar dia de admissão com preferência para dias mais cedo
        earliest = patient_data['earliest_admission']
        latest = patient_data['latest_admission']
        if latest > earliest and random.random() < 0.7:  # 70% de chance de preferir dias mais cedo
            new_solution[patient]['day'] = random.randint(earliest, earliest + (latest - earliest) // 2)
        else:
            new_solution[patient]['day'] = random.randint(earliest, latest)
    
    # Estratégias básicas
    else:
        if strategy == "ward" or strategy == "both":
            new_solution[patient]['ward'] = random.choice(list(instance_data['wards'].keys()))
        
        if strategy == "day" or strategy == "both":
            earliest = patient_data['earliest_admission']
            latest = patient_data['latest_admission']
            new_solution[patient]['day'] = random.randint(earliest, latest)
    
    return new_solution

def analyze_solution(instance_data, solution):
    """
    Analisa uma solução e retorna estatísticas úteis.
    """
    stats = {
        'total_patients': len(instance_data['patients']),
        'allocated_patients': sum(1 for p, data in solution.items() if data['ward'] is not None),
        'ward_occupancy': {},
        'surgery_per_day': [0] * instance_data['days'],
        'cost_breakdown': {},
        'ot_utilization': {}
    }
    
    # Calcular ocupação por ward e por dia
    ward_occupancy = {ward: [0] * instance_data['days'] for ward in instance_data['wards']}
    
    # Calcular utilização de OT por especialização e dia
    ot_usage = {spec: [0] * instance_data['days'] for spec in instance_data['specializations']}
    
    for patient, data in solution.items():
        if data['ward'] is None or data['day'] < 0:
            continue
        
        ward = data['ward']
        day = data['day']
        length_of_stay = instance_data['patients'][patient]['length_of_stay']
        spec = instance_data['patients'][patient]['specialization']
        
        # Contar cirurgia
        stats['surgery_per_day'][day] += 1
        
        # Contar uso de OT
        if day < instance_data['days'] and spec in ot_usage:
            surgery_duration = instance_data['patients'][patient]['surgery_duration']
            ot_usage[spec][day] += surgery_duration
        
        # Contar ocupação
        for d in range(day, min(day + length_of_stay, instance_data['days'])):
            ward_occupancy[ward][d] += 1
    
    # Calcular estatísticas de ocupação por ward
    for ward in instance_data['wards']:
        capacity = instance_data['wards'][ward]['bed_capacity']
        avg_occupancy = sum(ward_occupancy[ward]) / len(ward_occupancy[ward])
        max_occupancy = max(ward_occupancy[ward])
        overload_days = sum(1 for occ in ward_occupancy[ward] if occ > capacity)
        
        stats['ward_occupancy'][ward] = {
            'capacity': capacity,
            'avg_occupancy': avg_occupancy,
            'max_occupancy': max_occupancy,
            'occupancy_rate': avg_occupancy / capacity if capacity > 0 else 0,
            'overload_days': overload_days,
            'daily_occupancy': ward_occupancy[ward]
        }
    
    # Calcular estatísticas de utilização de OT
    for spec in instance_data['specializations']:
        available_ot = instance_data['specializations'][spec]['available_ot']
        usage = ot_usage[spec]
        
        # Calcular estatísticas apenas para dias disponíveis
        valid_days = min(len(available_ot), instance_data['days'])
        
        if valid_days > 0:
            avg_usage = sum(usage[:valid_days]) / valid_days
            avg_available = sum(available_ot[:valid_days]) / valid_days
            
            if avg_available > 0:
                utilization_rate = avg_usage / avg_available
            else:
                utilization_rate = 0
                
            overtime_days = 0
            undertime_days = 0
            
            for d in range(valid_days):
                if usage[d] > available_ot[d]:
                    overtime_days += 1
                else:
                    undertime_days += 1
            
            stats['ot_utilization'][spec] = {
                'avg_usage': avg_usage,
                'avg_available': avg_available,
                'utilization_rate': utilization_rate,
                'overtime_days': overtime_days,
                'undertime_days': undertime_days,
                'daily_usage': usage[:valid_days],
                'daily_available': available_ot[:valid_days]
            }
    
    # Componentes de custo simplificados (não precisa ser o cálculo completo)
    bed_capacity_cost = sum(
        max(0, stats['ward_occupancy'][ward]['max_occupancy'] - instance_data['wards'][ward]['bed_capacity'])
        for ward in instance_data['wards']
    ) * 10
    
    surgery_conflict_cost = sum(
        max(0, day_surgeries - len(instance_data['specializations']))
        for day_surgeries in stats['surgery_per_day']
    ) * 5
    
    stats['cost_breakdown'] = {
        'bed_capacity_cost': bed_capacity_cost,
        'surgery_conflict_cost': surgery_conflict_cost,
        'total_estimate': bed_capacity_cost + surgery_conflict_cost
    }
    
    return stats