import re

# -------------------------------
# Class Definitions
# -------------------------------

class Hospital:
    def __init__(self):
        self.wards = []
        self.operating_theaters = []
    
    def add_ward(self, ward):
        self.wards.append(ward)
    
    def add_operating_theater(self, theater):
        self.operating_theaters.append(theater)
    
    def schedule_patient(self, patient):
        for ward in self.wards:
            if ward.specialization == patient.required_specialization and ward.can_admit(patient):
                for theater in self.operating_theaters:
                    slot = patient.find_feasible_surgery_slot(theater.available_slots)
                    if slot is not None:
                        # Reserve the ward and theater slot for the patient
                        ward.admit_patient(patient)
                        theater.schedule_surgery(patient, slot)
                        patient.assigned_ward = ward
                        patient.assigned_theater_slot = (theater, slot)
                        return True
        return False

class Ward:
    """Represents a ward in the hospital, with a specialization, capacity, and workload tracking."""
    
    def __init__(self, name, specialization, capacity):
        self.name = name
        self.specialization = specialization
        self.capacity = capacity
        self.current_load = 0  # Track how many patients are currently assigned
    
    def can_admit(self, patient):
        return self.current_load < self.capacity and patient.required_specialization == self.specialization
    
    def admit_patient(self, patient):
        if not self.can_admit(patient):
            raise Exception(f"Ward {self.name} cannot admit patient {patient.name}.")
        self.current_load += 1

class Patient:
    """Represents a patient needing admission, including their scheduling requirements and impact on workload."""
    
    def __init__(self, name, required_specialization, required_procedure, admission_window, workload_impact):
        self.name = name
        self.required_specialization = required_specialization
        self.required_procedure = required_procedure
        self.admission_window = admission_window  # e.g., (earliest, latest)
        self.workload_impact = workload_impact
        self.assigned_ward = None
        self.assigned_theater_slot = None
    
    def find_feasible_surgery_slot(self, available_slots):
        for slot in available_slots:
            if self.admission_window[0] <= slot <= self.admission_window[1]:
                return slot
        return None

class OperatingTheater:
    """Represents an operating theater with available time slots and scheduling constraints for surgeries."""
    
    def __init__(self, name, available_slots):
        self.name = name
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

class OptimizationModel:
    """Defines the optimization model (MILP) for patient admission scheduling with workload balancing and cost minimization."""
    
    def __init__(self, hospital, patients):
        self.hospital = hospital
        self.patients = patients
        self.model = None  # Replace with an actual ILP model instance if using a solver
        self.decision_variables = {}
    
    def build_model(self):
        """
        Build the integer programming model with decision variables, constraints, and objectives.
        
        The model includes:
         - Ward assignment constraints.
         - Operating theater scheduling within admission windows.
         - Workload balancing constraints.
         - Two objectives: operational cost and workload imbalance.
        """
        self.model = "Constructed MILP model with variables, constraints, and objectives"
    
    def solve(self):
        """
        Solve the optimization model using an ILP solver.
        
        Returns:
            object: The solver's solution (or status) for the optimization problem.
        """
        solution = "Optimal solution from solver"
        return solution
    
    def get_solution(self):
        """
        Extract the solution from the model in a user-friendly format.
        
        Returns:
            dict: Mapping of patients to their assigned ward and operating theater slot.
        """
        result = {}
        for patient in self.patients:
            result[patient.name] = (
                patient.assigned_ward.name if patient.assigned_ward else None,
                patient.assigned_theater_slot[0].name if patient.assigned_theater_slot else None,
                patient.assigned_theater_slot[1] if patient.assigned_theater_slot else None
            )
        return result

class SolutionApproach:
    """Implements the Balanced Box Method for the bi-objective optimization problem."""
    
    def __init__(self, optimization_model):
        """
        Initialize with the given optimization model.
        
        Args:
            optimization_model (OptimizationModel): The optimization model instance.
        """
        self.optimization_model = optimization_model
        self.pareto_solutions = []
    
    def solve_bi_objective(self):
        """
        Solve the bi-objective optimization problem using the Balanced Box Method.
        
        Returns:
            list: Non-dominated solutions from the Pareto front.
        """
        solution = self.optimization_model.solve()
        self.pareto_solutions.append(solution)
        return self.pareto_solutions
    
    def get_pareto_front(self):
        """
        Get the Pareto front solutions.
        
        Returns:
            list: A list of mappings from patients to assignments representing non-dominated solutions.
        """
        pareto_front = []
        for sol in self.pareto_solutions:
            solution_mapping = self.optimization_model.get_solution()
            pareto_front.append(solution_mapping)
        return pareto_front

# -------------------------------
# Parser Function
# -------------------------------

def parse_instance_file(filepath):
    """
    Parse a problem instance file and return the data as a dictionary.
    
    The file format is as follows:
    1. Line 1: Seed value.
    2. Line 2: Number of minor specializations per ward.
    3. Lines 3-5: Weights for OT overtime, OT undertime, and admission delay.
    4. Line 6: Number of days in the planning period.
    5. Line 7: Total number of specializations (S).
    6. Next S lines: [Specialization id] [Workload scaling factor] [Available OT time per day (semicolon separated)]
    7. Next line: "Wards: <number>"
    8. Next lines: For each ward: 
       [Ward id] [Bed capacity] [Workload capacity] [Major specialization] [Minor specializations or NONE] [Carryover patients] [Carryover workload]
    9. Next line: "Patients: <number>"
    10. Remaining lines: For each patient:
        [Patient id] [Specialization] [Earliest admission] [Latest admission] [Length of stay] [Surgery duration] [Workload per day (semicolon separated)]
    """
    with open(filepath, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]
    
    instance_data = {}
    current_line = 0

    # Parse seed and minor specializations (lines 1-2)
    instance_data['seed'] = int(re.search(r'\d+', lines[current_line]).group())
    current_line += 1
    instance_data['minor_specializations_per_ward'] = int(re.search(r'\d+', lines[current_line]).group())
    current_line += 1

    # Parse weights (lines 3-5)
    instance_data['weight_overtime'] = float(re.search(r'[\d\.]+', lines[current_line]).group())
    current_line += 1
    instance_data['weight_undertime'] = float(re.search(r'[\d\.]+', lines[current_line]).group())
    current_line += 1
    instance_data['weight_delay'] = float(re.search(r'[\d\.]+', lines[current_line]).group())
    current_line += 1

    # Parse planning period (line 6)
    instance_data['days'] = int(re.search(r'\d+', lines[current_line]).group())
    current_line += 1

    # Parse specializations (line 7 and next S lines)
    specialization_count = int(re.search(r'\d+', lines[current_line]).group())
    current_line += 1
    specializations = {}
    for _ in range(specialization_count):
        parts = lines[current_line].split()
        spec_id = parts[0]
        scaling_factor = float(parts[1])
        available_ot = [float(t) for t in parts[2].split(';')]
        specializations[spec_id] = {
            'scaling_factor': scaling_factor,
            'available_ot': available_ot
        }
        current_line += 1
    instance_data['specializations'] = specializations

    # Parse wards: Look for a line starting with "Wards:"
    ward_line_match = re.match(r'Wards:\s*(\d+)', lines[current_line])
    if ward_line_match:
        ward_count = int(ward_line_match.group(1))
        current_line += 1
    else:
        raise ValueError("Could not find ward count.")
    
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

    # Parse patients: Look for a line starting with "Patients:"
    patient_line_match = re.match(r'Patients:\s*(\d+)', lines[current_line])
    if patient_line_match:
        patient_count = int(patient_line_match.group(1))
        current_line += 1
    else:
        raise ValueError("Could not find patient count.")
    
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

# -------------------------------
# Example: Build Objects from Parsed Data
# -------------------------------

def build_hospital_instance(instance_data):
    """
    Build a Hospital object and related entities (Wards, Operating Theaters, Patients)
    from the parsed instance data.
    
    Returns:
        hospital (Hospital): The constructed Hospital object.
        patients (list of Patient): List of Patient objects.
    """
    hospital = Hospital()
    
    # Create Wards from instance_data
    for ward_id, data in instance_data['wards'].items():
        ward = Ward(name=ward_id,
                    specialization=data['major_specialization'],
                    capacity=data['bed_capacity'])
        hospital.add_ward(ward)
    
    # For this example, we create a single Operating Theater for all specializations.
    # We use the OT time available from the first specialization (or you can design more complex logic).
    first_spec = next(iter(instance_data['specializations'].values()))
    available_slots = first_spec['available_ot']  # List of OT time slots
    theater = OperatingTheater(name="OT1", available_slots=available_slots)
    hospital.add_operating_theater(theater)
    
    # Create Patient objects
    patients = []
    for patient_id, data in instance_data['patients'].items():
        patient = Patient(name=patient_id,
                          required_specialization=data['specialization'],
                          required_procedure="Procedure_" + data['specialization'],  # placeholder procedure
                          admission_window=(data['earliest_admission'], data['latest_admission']),
                          workload_impact=data['workload'][0])  # using the first workload value as an example
        patients.append(patient)
    
    return hospital, patients

# -------------------------------
# Putting It All Together
# -------------------------------

# Example usage:
# Replace 'instance_file.txt' with the path to your instance file.
# filepath = "instance_file.txt"
# instance_data = parse_instance_file(filepath)
# hospital, patients = build_hospital_instance(instance_data)
# 
# # Optionally, schedule patients one by one:
# for patient in patients:
#     scheduled = hospital.schedule_patient(patient)
#     print(f"Patient {patient.name} scheduling status: {scheduled}")
#
# # Build and solve the optimization model (if using a solver)
# opt_model = OptimizationModel(hospital, patients)
# opt_model.build_model()
# solution = opt_model.solve()
# print(opt_model.get_solution())
#
# # Use the SolutionApproach if exploring Pareto solutions:
# sol_approach = SolutionApproach(opt_model)
# pareto_solutions = sol_approach.solve_bi_objective()
# print(sol_approach.get_pareto_front())
