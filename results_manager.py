import os
import pandas as pd

def create_test_directory_structure(instances, algorithms):
    """
    Creates a directory structure for saving test results.
    
    Structure:
    tests/
    ├── s0m3/
    │   ├── ga/
    │   │   ├── solution.csv
    │   │   ├── metrics.csv
    │   │   └── visualizations/
    │   │       └── [images]
    │   └── ...
    └── ...
    
    Parameters:
    - instances: List of instance names (without extension)
    - algorithms: List of algorithm names
    
    Returns:
    - test_folder: Path to the test folder
    - folder_structure: Dictionary mapping instance names to algorithm folders
    """
    import os
    
    # Always use a fixed 'tests' folder instead of timestamp-based folders
    test_folder = "./tests"
    
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    
    folder_structure = {}
    
    for instance in instances:
        instance_folder = os.path.join(test_folder, instance)
        if not os.path.exists(instance_folder):
            os.makedirs(instance_folder)
        
        folder_structure[instance] = {}
        
        for algorithm in algorithms:
            algorithm_folder = os.path.join(instance_folder, algorithm)
            if not os.path.exists(algorithm_folder):
                os.makedirs(algorithm_folder)
            
            folder_structure[instance][algorithm] = algorithm_folder
    
    return test_folder, folder_structure

def save_solution_csv(solution, metrics, output_folder):
    """
    Salva a solução final e as métricas em arquivos CSV na pasta especificada.
    
    Parâmetros:
      solution: dicionário com a solução final (ex.: { patient: {'ward': ..., 'day': ...}, ... })
      metrics: dicionário com as métricas calculadas
      output_folder: pasta onde os CSVs serão salvos
    """
    # Converter a solução em DataFrame e salvar
    df_solution = pd.DataFrame.from_dict(solution, orient="index").reset_index()
    df_solution.rename(columns={"index": "Patient"}, inplace=True)
    solution_csv = os.path.join(output_folder, "final_solution.csv")
    df_solution.to_csv(solution_csv, index=False)
    
    # Converter as métricas em DataFrame e salvar
    df_metrics = pd.DataFrame([metrics])
    metrics_csv = os.path.join(output_folder, "metrics.csv")
    df_metrics.to_csv(metrics_csv, index=False)
