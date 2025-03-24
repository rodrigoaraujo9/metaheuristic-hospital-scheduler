import random
import copy
import math

def generate_initial_solution(instance_data):
    """
    Gera uma solução inicial aleatória para todos os pacientes.
    """
    solution = {}
    wards = list(instance_data['wards'].keys())
    for patient in instance_data['patients']:
        assigned_ward = random.choice(wards)
        assigned_day = random.randint(
            instance_data['patients'][patient]['earliest_admission'],
            instance_data['patients'][patient]['latest_admission']
        )
        solution[patient] = {'ward': assigned_ward, 'day': assigned_day}
    return solution

def calculate_cost(instance_data, solution):
    """
    Calcula o custo de uma solução, penalizando sobrecarga dos wards e conflitos de agendamento.
    """
    cost = 0
    # Penaliza sobrecarga nos wards
    ward_loads = {ward: 0 for ward in instance_data['wards']}
    for patient, data in solution.items():
        ward_loads[data['ward']] += 1

    for ward, load in ward_loads.items():
        if load > instance_data['wards'][ward]['bed_capacity']:
            cost += (load - instance_data['wards'][ward]['bed_capacity']) * 10

    # Penaliza conflitos no agendamento de cirurgias
    scheduled_surgeries = {}
    for patient, data in solution.items():
        day = data['day']
        scheduled_surgeries[day] = scheduled_surgeries.get(day, 0) + 1

    for day, count in scheduled_surgeries.items():
        if count > len(instance_data['specializations']):
            cost += (count - len(instance_data['specializations'])) * 5

    return cost

def make_neighbor_solution(instance_data, current_solution):
    """
    Gera uma solução vizinha alterando aleatoriamente a atribuição de um paciente.
    """
    new_solution = copy.deepcopy(current_solution)
    patient = random.choice(list(new_solution.keys()))
    new_solution[patient]['ward'] = random.choice(list(instance_data['wards'].keys()))
    new_solution[patient]['day'] = random.randint(
        instance_data['patients'][patient]['earliest_admission'],
        instance_data['patients'][patient]['latest_admission']
    )
    return new_solution
