import os
import copy
import random
import math

# -------------------------------
# Core Classes
# -------------------------------

class Hospital:
    """Represents a hospital with wards and operating theaters for scheduling patient admissions."""
    def __init__(self):
        self.wards = []
        self.operating_theaters = []
    
    def add_ward(self, ward):
        self.wards.append(ward)
    
    def add_operating_theater(self, theater):
        self.operating_theaters.append(theater)
    
    def schedule_patient(self, patient):
        """
        Attempt to schedule a patient.
        First, try to assign the patient to a ward that exactly matches the required specialization.
        Then, if that fails, try any ward with capacity.
        For operating theater, first choose one whose name matches the patient's required specialization,
        and if none exists, choose the first available theater.
        """
        # Try to assign to a matching ward first.
        for ward in self.wards:
            if (ward.specialization == patient.required_specialization or
                (hasattr(ward, 'minor_specializations') and patient.required_specialization in ward.minor_specializations)) \
                and ward.can_admit(patient):
                # Choose operating theater that matches patient's spec if possible.
                chosen_theater = None
                for theater in self.operating_theaters:
                    if theater.name == patient.required_specialization:
                        chosen_theater = theater
                        break
                if not chosen_theater and self.operating_theaters:
                    chosen_theater = self.operating_theaters[0]
                if chosen_theater:
                    slot = patient.find_feasible_surgery_slot(chosen_theater.available_slots)
                    if slot is not None:
                        ward.admit_patient(patient)
                        chosen_theater.schedule_surgery(patient, slot)
                        patient.assigned_ward = ward
                        patient.assigned_theater_slot = (chosen_theater, slot)
                        return True
        # Fallback: assign to any ward with capacity.
        for ward in self.wards:
            if ward.can_admit(patient):
                chosen_theater = self.operating_theaters[0] if self.operating_theaters else None
                if chosen_theater:
                    slot = patient.find_feasible_surgery_slot(chosen_theater.available_slots)
                    if slot is not None:
                        ward.admit_patient(patient)
                        chosen_theater.schedule_surgery(patient, slot)
                        patient.assigned_ward = ward
                        patient.assigned_theater_slot = (chosen_theater, slot)
                        return True
        return False

class Ward:
    """Represents a ward with a specialization, capacity, and current load."""
    def __init__(self, name, specialization, capacity, minor_specializations=None):
        self.name = name
        self.specialization = specialization
        self.capacity = capacity
        self.current_load = 0
        self.minor_specializations = minor_specializations if minor_specializations else []
    
    def can_admit(self, patient):
        return self.current_load < self.capacity
    
    def admit_patient(self, patient):
        if not self.can_admit(patient):
            raise Exception(f"Ward {self.name} is at full capacity for patient {patient.name}.")
        self.current_load += 1

class Patient:
    """Represents a patient needing admission with scheduling requirements."""
    def __init__(self, name, required_specialization, required_procedure, admission_window, workload_impact):
        self.name = name
        self.required_specialization = required_specialization
        self.required_procedure = required_procedure
        self.admission_window = admission_window  # tuple (start, end)
        self.workload_impact = workload_impact
        self.assigned_ward = None
        self.assigned_theater_slot = None
    
    def find_feasible_surgery_slot(self, available_slots):
        # Try to find a slot within the admission window.
        for slot in available_slots:
            if self.admission_window[0] <= slot <= self.admission_window[1]:
                return slot
        # If none falls within, relax and choose the smallest available slot.
        if available_slots:
            return min(available_slots)
        return None

class OperatingTheater:
    """Represents an operating theater with available slots and a scheduling dictionary."""
    def __init__(self, name, available_slots):
        self.name = name
        self.original_slots = list(available_slots)  # keep original copy for resetting
        self.available_slots = list(available_slots)
        self.schedule = {slot: None for slot in self.available_slots}
    
    def is_slot_available(self, slot):
        return slot in self.schedule and self.schedule[slot] is None
    
    def schedule_surgery(self, patient, slot):
        if self.is_slot_available(slot):
            self.schedule[slot] = patient
            if slot in self.available_slots:
                self.available_slots.remove(slot)
            return True
        return False

# -------------------------------
# Optimization Model (Greedy)
# -------------------------------

class OptimizationModel:
    """
    A simple greedy optimization model.
    Loops over all patients and calls hospital.schedule_patient().
    """
    def __init__(self, hospital, patients):
        self.hospital = hospital
        self.patients = patients
    
    def build_model(self):
        self.model = "Greedy scheduling model built"
    
    def solve(self):
        for patient in self.patients:
            if patient.assigned_ward is None:
                self.hospital.schedule_patient(patient)
        return "Greedy scheduling complete"
    
    def get_solution(self):
        result = {}
        for patient in self.patients:
            result[patient.name] = (
                patient.assigned_ward.name if patient.assigned_ward else None,
                patient.assigned_theater_slot[0].name if patient.assigned_theater_slot else None,
                patient.assigned_theater_slot[1] if patient.assigned_theater_slot else None
            )
        return result

class SolutionApproach:
    """A stub for additional solution processing (e.g., Pareto front extraction)."""
    def __init__(self, optimization_model):
        self.optimization_model = optimization_model
        self.pareto_solutions = []
    
    def solve_bi_objective(self):
        solution = self.optimization_model.solve()
        self.pareto_solutions.append(solution)
        return self.pareto_solutions
    
    def get_pareto_front(self):
        sol_mapping = self.optimization_model.get_solution()
        return [sol_mapping]

# -------------------------------
# Heuristic Implementations
# -------------------------------

# Heuristic 1: Tabu Search
class HeuristicScheduler1:
    """
    Implements a Tabu Search heuristic.
    The solution is represented by an ordering (permutation) of patient indices.
    """
    def __init__(self, hospital, patients):
        self.hospital = hospital
        self.patients = patients
        self.original_hospital = copy.deepcopy(hospital)
        self.original_patients = copy.deepcopy(patients)
    
    def _reset_state(self, hosp, pats):
        for patient in pats:
            patient.assigned_ward = None
            patient.assigned_theater_slot = None
        for ward in hosp.wards:
            ward.current_load = 0
        for theater in hosp.operating_theaters:
            theater.available_slots = list(theater.original_slots)
            theater.schedule = {slot: None for slot in theater.available_slots}
    
    def _evaluate_solution(self, ordering):
        hosp_copy = copy.deepcopy(self.original_hospital)
        pats_copy = copy.deepcopy(self.original_patients)
        self._reset_state(hosp_copy, pats_copy)
        ordered_patients = [pats_copy[i] for i in ordering]
        for patient in ordered_patients:
            hosp_copy.schedule_patient(patient)
        score = sum(1 for patient in pats_copy if patient.assigned_ward is not None)
        return score
    
    def schedule(self, max_iter=50, tabu_tenure=5):
        n = len(self.patients)
        current_solution = list(range(n))
        best_solution = current_solution[:]
        best_score = self._evaluate_solution(current_solution)
        current_iter = 0
        tabu_list = {}
        while current_iter < max_iter:
            neighborhood = []
            for i in range(n):
                for j in range(i+1, n):
                    move = (i, j)
                    if move in tabu_list and tabu_list[move] > current_iter:
                        continue
                    neighbor = current_solution[:]
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    score = self._evaluate_solution(neighbor)
                    neighborhood.append((move, neighbor, score))
            if not neighborhood:
                break
            move_chosen, best_neighbor, neighbor_score = max(neighborhood, key=lambda x: x[2])
            current_solution = best_neighbor
            if neighbor_score > best_score:
                best_solution = best_neighbor
                best_score = neighbor_score
            tabu_list[move_chosen] = current_iter + tabu_tenure
            tabu_list = {move: expiry for move, expiry in tabu_list.items() if expiry > current_iter}
            current_iter += 1
        
        self.hospital = copy.deepcopy(self.original_hospital)
        self.patients = copy.deepcopy(self.original_patients)
        self._reset_state(self.hospital, self.patients)
        for i in best_solution:
            self.hospital.schedule_patient(self.patients[i])
        
        print("\n[Tabu Search Heuristic] Final Assignments:")
        analysis = self.hospital_schedule_analysis()
        print(analysis)
        print(f"Total patients scheduled: {best_score}")
    
    def hospital_schedule_analysis(self):
        analysis = ""
        allocated = sum(1 for p in self.patients if p.assigned_ward is not None)
        total = len(self.patients)
        percent = (allocated / total) * 100
        for patient in self.patients:
            ward = patient.assigned_ward.name if patient.assigned_ward else "None"
            theater = patient.assigned_theater_slot[0].name if patient.assigned_theater_slot else "None"
            slot = patient.assigned_theater_slot[1] if patient.assigned_theater_slot else "None"
            analysis += f"  Patient {patient.name}: Ward {ward}, Theater {theater}, Slot {slot}\n"
        analysis += f"Allocated {allocated}/{total} patients ({percent:.1f}%)\n"
        return analysis

# Heuristic 2: Simulated Annealing
class HeuristicScheduler2:
    """
    Implements a Simulated Annealing heuristic.
    The solution is represented by an ordering (permutation) of patient indices.
    """
    def __init__(self, hospital, patients):
        self.hospital = hospital
        self.patients = patients
        self.original_hospital = copy.deepcopy(hospital)
        self.original_patients = copy.deepcopy(patients)
    
    def _reset_state(self, hosp, pats):
        for patient in pats:
            patient.assigned_ward = None
            patient.assigned_theater_slot = None
        for ward in hosp.wards:
            ward.current_load = 0
        for theater in hosp.operating_theaters:
            theater.available_slots = list(theater.original_slots)
            theater.schedule = {slot: None for slot in theater.available_slots}
    
    def _evaluate_solution(self, ordering):
        hosp_copy = copy.deepcopy(self.original_hospital)
        pats_copy = copy.deepcopy(self.original_patients)
        self._reset_state(hosp_copy, pats_copy)
        ordered_patients = [pats_copy[i] for i in ordering]
        for patient in ordered_patients:
            hosp_copy.schedule_patient(patient)
        score = sum(1 for patient in pats_copy if patient.assigned_ward is not None)
        return score
    
    def schedule(self, max_iter=100, initial_temp=100.0, cooling_rate=0.95):
        n = len(self.patients)
        current_solution = list(range(n))
        current_score = self._evaluate_solution(current_solution)
        best_solution = current_solution[:]
        best_score = current_score
        temperature = initial_temp
        
        for iteration in range(max_iter):
            neighbor = current_solution[:]
            i, j = random.sample(range(n), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbor_score = self._evaluate_solution(neighbor)
            delta = neighbor_score - current_score
            if delta >= 0 or random.random() < math.exp(delta / temperature):
                current_solution = neighbor
                current_score = neighbor_score
                if current_score > best_score:
                    best_solution = current_solution[:]
                    best_score = current_score
            temperature *= cooling_rate
        
        self.hospital = copy.deepcopy(self.original_hospital)
        self.patients = copy.deepcopy(self.original_patients)
        self._reset_state(self.hospital, self.patients)
        for i in best_solution:
            self.hospital.schedule_patient(self.patients[i])
        
        print("\n[Simulated Annealing Heuristic] Final Assignments:")
        analysis = self.hospital_schedule_analysis()
        print(analysis)
        print(f"Total patients scheduled: {best_score}")
    
    def hospital_schedule_analysis(self):
        analysis = ""
        allocated = sum(1 for p in self.patients if p.assigned_ward is not None)
        total = len(self.patients)
        percent = (allocated / total) * 100
        for patient in self.patients:
            ward = patient.assigned_ward.name if patient.assigned_ward else "None"
            theater = patient.assigned_theater_slot[0].name if patient.assigned_theater_slot else "None"
            slot = patient.assigned_theater_slot[1] if patient.assigned_theater_slot else "None"
            analysis += f"  Patient {patient.name}: Ward {ward}, Theater {theater}, Slot {slot}\n"
        analysis += f"Allocated {allocated}/{total} patients ({percent:.1f}%)\n"
        return analysis

# -------------------------------
# Instance File Parser Functions
# -------------------------------

def parse_instance_file(filepath):
    """
    Parse an instance file with labeled lines.
    Expected format:
      Line 1: "Seed: <integer>"
      Line 2: "M: <integer>"
      Line 3: "Weight_overtime: <float>"
      Line 4: "Weight_undertime: <float>"
      Line 5: "Weight_delay: <float>"
      Line 6: "Days: <integer>"
      Line 7: "Specialisms: <integer>"
      Next S lines: each with "[SpecID] [ScalingFactor] [AvailableOTTime1;AvailableOTTime2;...]"
      Next line: "Wards: <integer>"
      Next that many lines: each with "[WardID] [BedCapacity] [WorkloadCapacity] [MajorSpecialization] [MinorSpecializations or NONE] ..."
      Next line: "Patients: <integer>"
      Next that many lines: each with "[PatientID] [Specialization] [EarliestAdmission] [LatestAdmission] [LengthOfStay] [SurgeryDuration] [WorkloadPerDay (semicolon separated)]"
    """
    with open(filepath, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]
    
    instance_data = {}
    current_line = 0

    instance_data['seed'] = int(lines[current_line].split(":")[1].strip())
    current_line += 1
    instance_data['minor_specializations_per_ward'] = int(lines[current_line].split(":")[1].strip())
    current_line += 1

    instance_data['weight_overtime'] = float(lines[current_line].split(":")[1].strip())
    current_line += 1
    instance_data['weight_undertime'] = float(lines[current_line].split(":")[1].strip())
    current_line += 1
    instance_data['weight_delay'] = float(lines[current_line].split(":")[1].strip())
    current_line += 1

    instance_data['days'] = int(lines[current_line].split(":")[1].strip())
    current_line += 1

    specialization_count = int(lines[current_line].split(":")[1].strip())
    current_line += 1
    specializations = {}
    for _ in range(specialization_count):
        parts = lines[current_line].split()
        spec_id = parts[0]
        scaling_factor = float(parts[1])
        available_ot = [float(t) for t in parts[2].split(';')]
        specializations[spec_id] = {'scaling_factor': scaling_factor, 'available_ot': available_ot}
        current_line += 1
    instance_data['specializations'] = specializations

    ward_count = int(lines[current_line].split(":")[1].strip())
    current_line += 1
    wards = {}
    for _ in range(ward_count):
        parts = lines[current_line].split()
        ward_id = parts[0]
        bed_capacity = int(parts[1])
        major_spec = parts[3]
        minor_specs = [] if parts[4] == "NONE" else parts[4].split(',')
        wards[ward_id] = {'bed_capacity': bed_capacity, 'major_specialization': major_spec, 'minor_specializations': minor_specs}
        current_line += 1
    instance_data['wards'] = wards

    patient_count = int(lines[current_line].split(":")[1].strip())
    current_line += 1
    patients = {}
    for _ in range(patient_count):
        parts = lines[current_line].split()
        patient_id = parts[0]
        spec = parts[1]
        earliest = int(parts[2])
        latest = int(parts[3])
        length_of_stay = int(parts[4])
        surgery_duration = int(parts[5])
        workload = [float(x) for x in parts[6].split(';')]
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

    return instance_data

def build_hospital_instance(instance_data):
    hospital = Hospital()
    for ward_id, data in instance_data['wards'].items():
        ward = Ward(name=ward_id,
                    specialization=data['major_specialization'],
                    capacity=data['bed_capacity'],
                    minor_specializations=data.get('minor_specializations', []))
        hospital.add_ward(ward)
    # Create one operating theater per specialism.
    for spec, spec_data in instance_data['specializations'].items():
        theater = OperatingTheater(name=spec, available_slots=spec_data['available_ot'])
        hospital.add_operating_theater(theater)
    patients = []
    for patient_id, data in instance_data['patients'].items():
        patient = Patient(name=patient_id,
                          required_specialization=data['specialization'],
                          required_procedure="Procedure_" + data['specialization'],
                          admission_window=(data['earliest_admission'], data['latest_admission']),
                          workload_impact=data['workload'][0])
        patients.append(patient)
    return hospital, patients

# -------------------------------
# Main Execution
# -------------------------------

def analyze_allocations(solution):
    total = len(solution)
    allocated = sum(1 for alloc in solution.values() if alloc[0] is not None)
    percent = (allocated / total) * 100
    analysis = f"Allocated {allocated}/{total} patients ({percent:.1f}%).\n"
    return analysis

def main():
    instance_path = './data/instances/s0m0.dat'
    if os.path.exists(instance_path):
        print("Using instance file:", instance_path)
        instance_data = parse_instance_file(instance_path)
        hospital, patients = build_hospital_instance(instance_data)
    else:
        print("Instance file not found, using generated test instance.")
        from_time = 1
        to_time = 110
        num_patients = 100
        hospital = Hospital()
        ward = Ward(name="W1", specialization="Cardiology", capacity=num_patients)
        hospital.add_ward(ward)
        theater = OperatingTheater(name="Cardiology", available_slots=list(range(1, num_patients+1)))
        hospital.add_operating_theater(theater)
        patients = []
        for i in range(1, num_patients+1):
            patient = Patient(name=f"patient{i}",
                              required_specialization="Cardiology",
                              required_procedure="Procedure_Cardiology",
                              admission_window=(from_time, to_time),
                              workload_impact=1)
            patients.append(patient)
    
    print("\nHospital Wards:")
    for ward in hospital.wards:
        print(f"  Ward: {ward.name}, Specialization: {ward.specialization}, Capacity: {ward.capacity}")
    print("\nOperating Theaters:")
    for theater in hospital.operating_theaters:
        print(f"  Theater: {theater.name}, Available Slots: {theater.available_slots}")
    print("\nPatients:")
    for patient in patients:
        print(f"  Patient: {patient.name}, Specialization: {patient.required_specialization}, Admission Window: {patient.admission_window}")
    
    # Run Greedy Optimization Model
    for patient in patients:
        patient.assigned_ward = None
        patient.assigned_theater_slot = None
    opt_model = OptimizationModel(hospital, patients)
    opt_model.build_model()
    opt_model.solve()
    print("\n[Greedy Optimization] Scheduling complete.")
    allocations = opt_model.get_solution()
    print("Final Allocations from Greedy Optimization Model:")
    for patient_name, allocation in allocations.items():
        ward_name, theater_name, slot = allocation
        print(f"  Patient {patient_name}: Ward {ward_name}, Theater {theater_name}, Slot {slot}")
    print(analyze_allocations(allocations))
    
    # Run Tabu Search Heuristic
    heuristic1 = HeuristicScheduler1(hospital, patients)
    heuristic1.schedule(max_iter=50, tabu_tenure=5)
    
    # Run Simulated Annealing Heuristic
    heuristic2 = HeuristicScheduler2(hospital, patients)
    heuristic2.schedule(max_iter=100, initial_temp=100.0, cooling_rate=0.95)
    
    sol_approach = SolutionApproach(opt_model)
    sol_approach.solve_bi_objective()
    print("\nPareto Front Solutions:")
    for idx, pareto in enumerate(sol_approach.get_pareto_front()):
        print(f"  Solution {idx+1}:")
        for patient_name, allocation in pareto.items():
            ward_name, theater_name, slot = allocation
            print(f"    Patient {patient_name}: Ward {ward_name}, Theater {theater_name}, Slot {slot}")
        print(analyze_allocations(pareto))
    
    print("\n--- Finished scheduling ---\n")

if __name__ == "__main__":
    main()
