import os
import pandas as pd

def create_test_directory_structure(instances, algorithms, base_dir="."):
    """
    Cria a estrutura de pastas para salvar os resultados.
    
    Parâmetros:
      instances: lista de nomes de instâncias (ex.: ["instancia1", "instancia2"])
      algorithms: lista com os nomes dos algoritmos (ex.: ["sa", "ga", "hc", "rw", "hybrid_sa", "tabu"])
      base_dir: diretório base onde procurar/criar os testes (padrão é o diretório atual)
      
    Retorna:
      test_folder: caminho da pasta do teste (ex.: "./teste_3")
      structure: dicionário com a estrutura criada, onde structure[instancia][algoritmo] é o caminho da pasta para aquele algoritmo.
    """
    # Procura pastas que começam com "teste_" no diretório base
    test_dirs = [d for d in os.listdir(base_dir)
                 if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("teste_")]
    max_n = 0
    for d in test_dirs:
        try:
            n = int(d.split("_")[1])
            if n > max_n:
                max_n = n
        except:
            continue
    next_test = max_n + 1
    test_folder = os.path.join(base_dir, f"teste_{next_test}")
    os.makedirs(test_folder, exist_ok=True)
    
    structure = {}
    for inst in instances:
        instance_folder = os.path.join(test_folder, inst)
        os.makedirs(instance_folder, exist_ok=True)
        structure[inst] = {}
        for alg in algorithms:
            alg_folder = os.path.join(instance_folder, alg)
            os.makedirs(alg_folder, exist_ok=True)
            structure[inst][alg] = alg_folder
            
    return test_folder, structure

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
