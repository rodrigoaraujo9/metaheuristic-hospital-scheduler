import random
import copy
import time
import math
from utils import generate_initial_solution, calculate_cost, make_neighbor_solution, analyze_solution

class GeneticAlgorithmScheduler:
    def __init__(self, instance_data, population_size=100, generations=200, crossover_rate=0.85, 
                 mutation_rate=0.2, elitism_count=5, local_search_rate=0.4, tournament_size=5):
        self.instance_data = instance_data
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.local_search_rate = local_search_rate
        self.tournament_size = tournament_size
        
        # Inicialização de métricas
        self.runtime = 0
        self.best_solution = None
        self.best_cost = float('inf')
        self.cost_history = []
        self.diversity_history = []
        self.best_cost_history = []  # Para rastrear melhoras ao longo do tempo
        
        # Parâmetros adaptativos avançados
        self.adaptive_mutation = True
        self.min_mutation_rate = 0.05
        self.max_mutation_rate = 0.5
        self.mutation_rate_history = []
        
        # Controle de diversidade e estagnação
        self.iterations_without_improvement = 0
        self.reset_threshold = 30  # Reiniciar parcialmente após N iterações sem melhora
        self.diversity_target = 0.6  # Nível desejado de diversidade
        
        # Métodos de crossover disponíveis
        self.crossover_methods = ["uniform", "one_point", "two_point", "position_based"]
        self.active_crossover_method = "uniform"
        
        print("Gerando população inicial...")
        start_time = time.time()
        
        # Gerar população inicial mais diversificada
        self.population = []
        for _ in range(population_size // 2):  # Metade com geração inteligente
            solution = generate_initial_solution(instance_data)
            self.population.append(solution)
        
        for _ in range(population_size - len(self.population)):  # Metade com perturbações aleatórias
            base_solution = copy.deepcopy(self.population[random.randint(0, len(self.population)-1)])
            for _ in range(random.randint(5, 15)):  # Aplicar 5-15 perturbações aleatórias
                base_solution = make_neighbor_solution(instance_data, base_solution)
            self.population.append(base_solution)
            
        end_time = time.time()
        print(f"População inicial gerada em {end_time - start_time:.2f} segundos")
        
        # Avaliar população inicial
        self.evaluate_population()
    
    def evaluate_population(self):
        """Avalia toda a população e atualiza a melhor solução."""
        population_costs = []
        for solution in self.population:
            try:
                cost = calculate_cost(self.instance_data, solution)
                population_costs.append(cost)
            except Exception as e:
                print(f"Erro ao calcular custo: {str(e)}")
                population_costs.append(float('inf'))
        
        if population_costs:
            min_cost = min(population_costs)
            min_index = population_costs.index(min_cost)
            
            if min_cost < self.best_cost:
                self.best_cost = min_cost
                self.best_solution = copy.deepcopy(self.population[min_index])
                self.iterations_without_improvement = 0
            else:
                self.iterations_without_improvement += 1
        
        return population_costs
    
    def tournament_selection(self, costs=None):
        """Realiza a seleção por torneio com 'tournament_size' indivíduos."""
        if costs is None:
            costs = [calculate_cost(self.instance_data, sol) for sol in self.population]
        
        competitors_indices = random.sample(range(len(self.population)), min(self.tournament_size, len(self.population)))
        best_competitor_idx = min(competitors_indices, key=lambda i: costs[i])
        return self.population[best_competitor_idx]
    
    def crossover(self, parent1, parent2):
        """
        Realiza crossover entre dois pais usando diferentes métodos.
        """
        method = self.active_crossover_method
        
        if method == "uniform":
            return self._uniform_crossover(parent1, parent2)
        elif method == "one_point":
            return self._one_point_crossover(parent1, parent2)
        elif method == "two_point":
            return self._two_point_crossover(parent1, parent2)
        elif method == "position_based":
            return self._position_based_crossover(parent1, parent2)
        else:
            return self._uniform_crossover(parent1, parent2)  # Padrão
    
    def _uniform_crossover(self, parent1, parent2):
        """Crossover uniforme, misturando atribuições de pacientes aleatoriamente."""
        child1, child2 = {}, {}
        
        for patient in parent1:
            if random.random() < 0.5:
                child1[patient] = copy.deepcopy(parent1[patient])
                child2[patient] = copy.deepcopy(parent2[patient])
            else:
                child1[patient] = copy.deepcopy(parent2[patient])
                child2[patient] = copy.deepcopy(parent1[patient])
        
        self.repair_solution(child1)
        self.repair_solution(child2)
        return child1, child2
    
    def _one_point_crossover(self, parent1, parent2):
        """Crossover de um ponto, dividindo a lista de pacientes em um ponto."""
        patients = list(parent1.keys())
        point = random.randint(1, len(patients) - 1)
        
        child1, child2 = {}, {}
        
        for i, patient in enumerate(patients):
            if i < point:
                child1[patient] = copy.deepcopy(parent1[patient])
                child2[patient] = copy.deepcopy(parent2[patient])
            else:
                child1[patient] = copy.deepcopy(parent2[patient])
                child2[patient] = copy.deepcopy(parent1[patient])
        
        self.repair_solution(child1)
        self.repair_solution(child2)
        return child1, child2
    
    def _two_point_crossover(self, parent1, parent2):
        """Crossover de dois pontos, trocando a seção do meio."""
        patients = list(parent1.keys())
        if len(patients) <= 2:
            return self._uniform_crossover(parent1, parent2)
            
        points = sorted(random.sample(range(1, len(patients)), 2))
        p1, p2 = points
        
        child1, child2 = {}, {}
        
        for i, patient in enumerate(patients):
            if i < p1 or i >= p2:
                child1[patient] = copy.deepcopy(parent1[patient])
                child2[patient] = copy.deepcopy(parent2[patient])
            else:
                child1[patient] = copy.deepcopy(parent2[patient])
                child2[patient] = copy.deepcopy(parent1[patient])
        
        self.repair_solution(child1)
        self.repair_solution(child2)
        return child1, child2
    
    def _position_based_crossover(self, parent1, parent2):
        """
        Crossover baseado em posição, que mapeia as atribuições dos pacientes
        com base na sua "posição" (ward e dia) na solução.
        """
        # Agrupar pacientes por ward
        ward_mapping = {}
        for patient, data in parent1.items():
            ward = data['ward']
            if ward not in ward_mapping:
                ward_mapping[ward] = []
            ward_mapping[ward].append(patient)
        
        # Criar filhos
        child1, child2 = {}, {}
        
        # Para cada ward, aplicar crossover
        for ward in ward_mapping:
            patients_in_ward = ward_mapping[ward]
            if not patients_in_ward:
                continue
                
            # Decidir se troca atribuições para este ward
            if random.random() < 0.5:
                for patient in patients_in_ward:
                    child1[patient] = copy.deepcopy(parent1[patient])
                    child2[patient] = copy.deepcopy(parent2[patient])
            else:
                for patient in patients_in_ward:
                    child1[patient] = copy.deepcopy(parent2[patient])
                    child2[patient] = copy.deepcopy(parent1[patient])
        
        # Adicionar pacientes que não estão em nenhum ward
        for patient in parent1:
            if patient not in child1:
                if random.random() < 0.5:
                    child1[patient] = copy.deepcopy(parent1[patient])
                    child2[patient] = copy.deepcopy(parent2[patient])
                else:
                    child1[patient] = copy.deepcopy(parent2[patient])
                    child2[patient] = copy.deepcopy(parent1[patient])
        
        self.repair_solution(child1)
        self.repair_solution(child2)
        return child1, child2
    
    def repair_solution(self, solution):
        """Repara a solução para garantir que seja válida."""
        for patient, data in solution.items():
            patient_data = self.instance_data['patients'].get(patient)
            if patient_data is None:
                continue
                
            # Corrigir ward inválido
            if data.get('ward') is None or data['ward'] not in self.instance_data['wards']:
                data['ward'] = random.choice(list(self.instance_data['wards'].keys()))
            
            # Corrigir day inválido
            earliest = patient_data['earliest_admission']
            latest = patient_data['latest_admission']
            
            if data.get('day') is None or data['day'] < earliest or data['day'] > latest:
                data['day'] = random.randint(earliest, latest)
    
    def mutate(self, solution, current_mutation_rate):
        """
        Aplica mutação com uma taxa adaptativa.
        Usa diferentes tipos de mutação dependendo do contexto.
        """
        mutated = False
        
        for patient in solution:
            if random.random() < current_mutation_rate:
                mutated = True
                
                # Decidir tipo de mutação com base na estagnação
                if self.iterations_without_improvement > 10:
                    # Se estagnado, aplicar mutações mais agressivas
                    strategy = random.choice(["both", "smart"])
                else:
                    # Caso contrário, mutações mais suaves
                    strategy = random.choice(["ward", "day", "smart"])
                
                # Aplicar mutação
                patient_data = self.instance_data['patients'].get(patient)
                if not patient_data:
                    continue
                    
                if strategy == "ward" or strategy == "both":
                    # Tentar atribuir a wards compatíveis preferencialmente
                    spec = patient_data['specialization']
                    compatible_wards = []
                    
                    for ward in self.instance_data['wards']:
                        ward_data = self.instance_data['wards'][ward]
                        if spec == ward_data['major_specialization'] or spec in ward_data['minor_specializations']:
                            compatible_wards.append(ward)
                    
                    if compatible_wards and random.random() < 0.7:  # 70% de preferir compatíveis
                        solution[patient]['ward'] = random.choice(compatible_wards)
                    else:
                        solution[patient]['ward'] = random.choice(list(self.instance_data['wards'].keys()))
                
                if strategy == "day" or strategy == "both":
                    earliest = patient_data['earliest_admission']
                    latest = patient_data['latest_admission']
                    
                    if self.iterations_without_improvement > 15 and random.random() < 0.5:
                        # Se muito estagnado, tentar dias extremos
                        if random.random() < 0.5:
                            solution[patient]['day'] = earliest
                        else:
                            solution[patient]['day'] = latest
                    else:
                        # Caso contrário, qualquer dia válido
                        solution[patient]['day'] = random.randint(earliest, latest)
                
                if strategy == "smart":
                    # Mutação inteligente baseada na capacidade do ward
                    current_ward = solution[patient]['ward']
                    ward_stats = {}
                    
                    # Calcular ocupação por ward
                    for ward in self.instance_data['wards']:
                        ward_stats[ward] = {
                            'count': 0,
                            'capacity': self.instance_data['wards'][ward]['bed_capacity']
                        }
                    
                    for p, d in solution.items():
                        if p != patient and d['ward'] is not None:
                            ward_stats[d['ward']]['count'] += 1
                    
                    # Encontrar wards menos ocupados
                    available_wards = []
                    for ward, stats in ward_stats.items():
                        if stats['count'] < stats['capacity'] * 0.9:  # Menos de 90% ocupado
                            available_wards.append(ward)
                    
                    if available_wards:
                        solution[patient]['ward'] = random.choice(available_wards)
                    
                    # Ajustar dia para reduzir conflitos de cirurgia
                    surgery_days = [d['day'] for p, d in solution.items() if p != patient]
                    day_counts = {}
                    for day in surgery_days:
                        day_counts[day] = day_counts.get(day, 0) + 1
                    
                    earliest = patient_data['earliest_admission']
                    latest = patient_data['latest_admission']
                    
                    best_days = []
                    for day in range(earliest, latest + 1):
                        if day_counts.get(day, 0) < len(self.instance_data['specializations']):
                            best_days.append(day)
                    
                    if best_days:
                        solution[patient]['day'] = random.choice(best_days)
        
        return solution, mutated
    
    def local_search(self, solution, max_iterations=15):
        """
        Busca local melhorada que tenta diversas estratégias de vizinhança.
        """
        best_solution = copy.deepcopy(solution)
        best_cost = calculate_cost(self.instance_data, best_solution)
        
        improved = True
        iterations = 0
        strategies = ["ward", "day", "both", "smart"]
        
        # Se estagnado, aumentar a intensidade da busca
        if self.iterations_without_improvement > 20:
            max_iterations = 25
        
        while improved and iterations < max_iterations:
            improved = False
            
            # Tentar cada estratégia em ordem
            for strategy in strategies:
                try:
                    neighbor = make_neighbor_solution(self.instance_data, best_solution, strategy=strategy)
                    neighbor_cost = calculate_cost(self.instance_data, neighbor)
                    
                    if neighbor_cost < best_cost:
                        best_solution = neighbor
                        best_cost = neighbor_cost
                        improved = True
                        break
                except Exception as e:
                    continue
            
            # Se não melhorou com estratégias individuais, tentar modificar múltiplos pacientes
            if not improved and iterations > 5:
                # Selecionar alguns pacientes aleatoriamente
                patients = list(best_solution.keys())
                selected = random.sample(patients, min(3, len(patients)))
                
                temp_solution = copy.deepcopy(best_solution)
                for patient in selected:
                    # Modificar cada paciente
                    try:
                        patient_data = self.instance_data['patients'].get(patient)
                        if patient_data:
                            # Modificar ward e dia
                            temp_solution[patient]['ward'] = random.choice(list(self.instance_data['wards'].keys()))
                            earliest = patient_data['earliest_admission']
                            latest = patient_data['latest_admission']
                            temp_solution[patient]['day'] = random.randint(earliest, latest)
                    except Exception:
                        continue
                
                try:
                    temp_cost = calculate_cost(self.instance_data, temp_solution)
                    if temp_cost < best_cost:
                        best_solution = temp_solution
                        best_cost = temp_cost
                        improved = True
                except Exception:
                    pass
            
            iterations += 1
        
        return best_solution
    
    def measure_diversity(self):
        """Mede a diversidade da população."""
        if not self.population or len(self.population) < 2:
            return 0
        
        diversity_sum = 0
        comparisons = 0
        
        # Amostragem para eficiência
        sample_size = min(len(self.population), 10)
        sample_indices = random.sample(range(len(self.population)), sample_size)
        
        for i in range(len(sample_indices)):
            for j in range(i+1, len(sample_indices)):
                idx1, idx2 = sample_indices[i], sample_indices[j]
                sol1, sol2 = self.population[idx1], self.population[idx2]
                
                # Contar diferenças
                differences = 0
                total = 0
                
                for patient in sol1:
                    if patient in sol2:
                        total += 1
                        if sol1[patient]['ward'] != sol2[patient]['ward'] or sol1[patient]['day'] != sol2[patient]['day']:
                            differences += 1
                
                if total > 0:
                    diversity_sum += differences / total
                    comparisons += 1
        
        return diversity_sum / comparisons if comparisons > 0 else 0
    
    def adjust_parameters(self, gen, diversity, current_mutation_rate):
        """
        Ajusta os parâmetros do algoritmo com base na diversidade e estagnação.
        """
        # Ajustar taxa de mutação
        if diversity < self.diversity_target - 0.1:  # Diversidade muito baixa
            new_mutation_rate = min(self.max_mutation_rate, current_mutation_rate * 1.2)
        elif diversity > self.diversity_target + 0.1:  # Diversidade muito alta
            new_mutation_rate = max(self.min_mutation_rate, current_mutation_rate * 0.9)
        else:
            # Na faixa alvo, ajustar com base na estagnação
            if self.iterations_without_improvement > 10:
                # Aumentar a taxa se estagnado
                new_mutation_rate = min(self.max_mutation_rate, current_mutation_rate * 1.1)
            else:
                new_mutation_rate = current_mutation_rate
        
        # Ajustar método de crossover
        if gen % 20 == 0 or self.iterations_without_improvement > 15:  # Mudar periodicamente ou se estagnado
            self.active_crossover_method = random.choice(self.crossover_methods)
        
        return new_mutation_rate
    
    def reset_population_partially(self):
        """
        Reinicia parte da população para escapar de ótimos locais.
        """
        print("Reiniciando parcialmente a população devido à estagnação...")
        
        # Manter os melhores indivíduos
        keep_count = self.population_size // 4
        population_costs = [calculate_cost(self.instance_data, sol) for sol in self.population]
        sorted_indices = sorted(range(len(population_costs)), key=lambda i: population_costs[i])
        
        elite_solutions = [copy.deepcopy(self.population[i]) for i in sorted_indices[:keep_count]]
        
        # Gerar novos indivíduos
        new_population = []
        new_population.extend(elite_solutions)
        
        # Adicionar soluções completamente novas
        for _ in range(self.population_size // 4):
            new_population.append(generate_initial_solution(self.instance_data))
        
        # Adicionar variações dos elites
        for i in range(len(elite_solutions)):
            for _ in range(2):  # 2 variações por elite
                if len(new_population) < self.population_size:
                    modified = copy.deepcopy(elite_solutions[i])
                    # Aplicar várias mutações
                    for _ in range(random.randint(5, 15)):
                        modified = make_neighbor_solution(self.instance_data, modified, strategy="smart")
                    new_population.append(modified)
        
        # Preencher o restante com soluções aleatórias
        while len(new_population) < self.population_size:
            new_population.append(generate_initial_solution(self.instance_data))
        
        self.population = new_population
        self.iterations_without_improvement = 0
    
    def run(self):
        """Executa o algoritmo genético."""
        print(f"Iniciando algoritmo genético com população de {self.population_size} e {self.generations} gerações")
        
        start_time = time.time()
        current_mutation_rate = self.mutation_rate
        
        for gen in range(self.generations):
            try:
                # Avaliar população atual
                population_costs = self.evaluate_population()
                
                if population_costs:
                    gen_best_cost = min(population_costs)
                    gen_best_idx = population_costs.index(gen_best_cost)
                    
                    self.cost_history.append(gen_best_cost)
                    self.best_cost_history.append(self.best_cost)
                    
                    # Medir diversidade
                    diversity = self.measure_diversity()
                    self.diversity_history.append(diversity)
                    
                    # Ajustar parâmetros
                    current_mutation_rate = self.adjust_parameters(gen, diversity, current_mutation_rate)
                    self.mutation_rate_history.append(current_mutation_rate)
                    
                    # Verificar estagnação e reset parcial se necessário
                    if self.iterations_without_improvement >= self.reset_threshold:
                        self.reset_population_partially()
                    
                    # Log de progresso
                    elapsed = time.time() - start_time
                    print(f"Geração {gen+1}/{self.generations} - Melhor custo: {self.best_cost:.2f} - Diversidade: {diversity:.2f} - Taxa de mutação: {current_mutation_rate:.3f} - Tempo: {elapsed:.2f}s")
                    
                    # Criar próxima geração
                    new_population = []
                    
                    # Elitismo: preservar as melhores soluções
                    elites = []
                    sorted_indices = sorted(range(len(population_costs)), key=lambda i: population_costs[i])
                    for idx in sorted_indices[:self.elitism_count]:
                        elites.append(copy.deepcopy(self.population[idx]))
                    
                    # Gerar novos indivíduos
                    children_generated = 0
                    max_attempts = 100  # Evitar loops infinitos
                    attempts = 0
                    
                    while len(new_population) < self.population_size - len(elites) and attempts < max_attempts:
                        try:
                            # Seleção
                            parent1 = self.tournament_selection(population_costs)
                            parent2 = self.tournament_selection(population_costs)
                            
                            # Crossover
                            if random.random() < self.crossover_rate:
                                child1, child2 = self.crossover(parent1, parent2)
                            else:
                                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                            
                            # Mutação
                            child1, mutated1 = self.mutate(child1, current_mutation_rate)
                            child2, mutated2 = self.mutate(child2, current_mutation_rate)
                            
                            # Busca local
                            if mutated1 and random.random() < self.local_search_rate:
                                child1 = self.local_search(child1)
                            if mutated2 and random.random() < self.local_search_rate:
                                child2 = self.local_search(child2)
                            
                            # Adicionar à nova população
                            new_population.append(child1)
                            if len(new_population) < self.population_size - len(elites):
                                new_population.append(child2)
                            
                            children_generated += 2
                        except Exception as e:
                            print(f"Erro na geração de filhos: {str(e)}")
                        
                        attempts += 1
                    
                    # Se não conseguiu gerar filhos suficientes, completar com cópias dos elites
                    while len(new_population) < self.population_size - len(elites):
                        random_elite = random.choice(elites)
                        modified_elite = copy.deepcopy(random_elite)
                        modified_elite, _ = self.mutate(modified_elite, current_mutation_rate * 2)  # Mutação mais forte
                        new_population.append(modified_elite)
                    
                    # Atualizar população
                    self.population = elites + new_population
                    
                    # Garantir tamanho da população
                    self.population = self.population[:self.population_size]
            except Exception as e:
                print(f"Erro na geração {gen+1}: {str(e)}")
                continue
        
        # Final local search intensivo na melhor solução
        print("Aplicando busca local intensiva na melhor solução...")
        improved = True
        iterations = 0
        max_iterations = 100
        
        while improved and iterations < max_iterations:
            improved = False
            strategies = ["ward", "day", "both", "smart"]
            
            for strategy in strategies:
                try:
                    neighbor = make_neighbor_solution(self.instance_data, self.best_solution, strategy=strategy)
                    neighbor_cost = calculate_cost(self.instance_data, neighbor)
                    
                    if neighbor_cost < self.best_cost:
                        self.best_solution = neighbor
                        self.best_cost = neighbor_cost
                        improved = True
                        print(f"Melhoria na busca local final: {self.best_cost:.2f}")
                        break
                except Exception:
                    continue
            
            iterations += 1
        
        # Calcular tempo total
        self.runtime = time.time() - start_time
        print(f"Algoritmo genético concluído em {self.runtime:.2f} segundos")
        print(f"Melhor custo encontrado: {self.best_cost:.2f}")
        
        # Analisar a melhor solução
        best_stats = analyze_solution(self.instance_data, self.best_solution)
        print(f"Estatísticas da melhor solução:")
        print(f"  Pacientes alocados: {best_stats['allocated_patients']}/{best_stats['total_patients']} ({best_stats['allocated_patients']/best_stats['total_patients']*100:.1f}%)")
        
        # Mostrar ocupação das enfermarias
        print("  Ocupação das enfermarias:")
        for ward, stats in best_stats['ward_occupancy'].items():
            print(f"    {ward}: {stats['avg_occupancy']:.1f}/{stats['capacity']} ({stats['occupancy_rate']*100:.1f}%) - Dias em sobrecarga: {stats['overload_days']}")
        
        return self.best_solution