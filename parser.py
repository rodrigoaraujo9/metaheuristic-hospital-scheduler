def parse_instance_file(filepath):
    """
    Faz o parsing de um arquivo de instância e retorna os dados em um dicionário estruturado.
    """
    with open(filepath, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]

    instance_data = {}
    current_line = 0

    # Parsing do seed e especializações menores
    instance_data['seed'] = int(lines[current_line].split(":")[1].strip())
    current_line += 1
    instance_data['minor_specializations_per_ward'] = int(lines[current_line].split(":")[1].strip())
    current_line += 1

    # Parsing dos pesos
    instance_data['weight_overtime'] = float(lines[current_line].split(":")[1].strip())
    current_line += 1
    instance_data['weight_undertime'] = float(lines[current_line].split(":")[1].strip())
    current_line += 1
    instance_data['weight_delay'] = float(lines[current_line].split(":")[1].strip())
    current_line += 1

    # Parsing do período de planejamento
    instance_data['days'] = int(lines[current_line].split(":")[1].strip())
    current_line += 1

    # Parsing das especializações
    specialization_count = int(lines[current_line].split(":")[1].strip())
    current_line += 1
    specializations = {}
    for _ in range(specialization_count):
        parts = lines[current_line].split()
        spec_id = parts[0]
        scaling_factor = float(parts[1])
        available_ot = [float(t) for t in parts[2].split(';')]
        specializations[spec_id] = {
            'scaling_factor': scaling_factor,
            'available_ot': available_ot,
            'total_available_ot': sum(available_ot)  # Novo: soma total de OT disponível
        }
        current_line += 1
    instance_data['specializations'] = specializations

    # Parsing dos wards
    ward_count = int(lines[current_line].split(":")[1].strip())
    current_line += 1
    wards = {}
    for _ in range(ward_count):
        parts = lines[current_line].split()
        ward_id = parts[0]
        bed_capacity = int(parts[1])
        workload_capacity = float(parts[2])
        major_spec = parts[3]
        minor_specs = [] if parts[4] == "NONE" else parts[4].split(',')
        carryover_patients = [int(x) for x in parts[5].split(';')]
        carryover_workload = [float(x) for x in parts[6].split(';')]
        wards[ward_id] = {
            'bed_capacity': bed_capacity,
            'workload_capacity': workload_capacity,
            'major_specialization': major_spec,
            'minor_specializations': minor_specs,
            'carryover_patients': carryover_patients,
            'carryover_workload': carryover_workload
        }
        current_line += 1
    instance_data['wards'] = wards

    # Parsing dos pacientes
    patient_count = int(lines[current_line].split(":")[1].strip())
    current_line += 1
    patients = {}
    
    # Pré-processamento para estatísticas
    spec_patient_counts = {spec: 0 for spec in specializations}
    total_surgery_duration = 0
    
    for _ in range(patient_count):
        parts = lines[current_line].split()
        patient_id = parts[0]
        spec = parts[1]
        earliest = int(parts[2])
        latest = int(parts[3])
        length_of_stay = int(parts[4])
        surgery_duration = int(parts[5])
        workload = [float(x) for x in parts[6].split(';')]
        
        # Atualizar estatísticas
        if spec in spec_patient_counts:
            spec_patient_counts[spec] += 1
        total_surgery_duration += surgery_duration
        
        patients[patient_id] = {
            'specialization': spec,
            'earliest_admission': earliest,
            'latest_admission': latest,
            'length_of_stay': length_of_stay,
            'surgery_duration': surgery_duration,
            'workload': workload
        }
        current_line += 1
    
    instance_data['patients'] = patients
    
    # Calcular e adicionar estatísticas (novidade)
    # Estatísticas de utilização de OT
    total_available_ot = 0
    for spec, data in specializations.items():
        total_available_ot += data['total_available_ot']
    
    # Se OT disponível for zero, evitar divisão por zero
    if total_available_ot > 0:
        ot_utilization = total_surgery_duration / total_available_ot
    else:
        ot_utilization = 0
    
    # Adicionar à instância
    instance_data['total_surgery_duration'] = total_surgery_duration
    instance_data['total_available_ot'] = total_available_ot
    instance_data['ot_utilization'] = ot_utilization
    instance_data['patients_per_specialization'] = spec_patient_counts
    
    # Ajustar pesos com base nas características da instância
    if ot_utilization < 0.8:
        # Se há muita capacidade ociosa, reduzir peso do undertime
        instance_data['weight_undertime'] *= 0.8
    elif ot_utilization > 1.1:
        # Se há sobrecarga de capacidade, aumentar peso do overtime
        instance_data['weight_overtime'] *= 1.2
    
    # Informações de debug
    print(f"Instância carregada: {filepath}")
    print(f"  Pacientes: {patient_count}")
    print(f"  Especializações: {len(specializations)}")
    print(f"  Enfermarias: {len(wards)}")
    print(f"  Dias de planejamento: {instance_data['days']}")
    print(f"  Taxa de utilização de OT potencial: {ot_utilization:.2f}")
    
    return instance_data