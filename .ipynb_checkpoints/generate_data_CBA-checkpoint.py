import os
import json
from sage.all import GF, vector, matrix
# Importamos la función de simulación directamente desde utils.py
from utils import simulate_and_analyze_leakage

# --- Configuración de rutas de directorios ---
input_dir = "generated_data"
results_dir = "data_for_seeds"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

target_files = [
    # "instances_n7_k3_r2_m4.json"
    "instances_n33_k15_r10_m31.json"
]
beta_values = [0.03, 0.05, 0.1, 0.15, 0.2]

print("--- Starting Multi-Beta JSON Data Generation Pipeline ---")
print(f"Target Output folder: '{results_dir}/'\n")

for filename in target_files:
    input_path = os.path.join(input_dir, filename)
    if not os.path.exists(input_path):
        print(f"[WARNING] Source file not found: {input_path}. Skipping.")
        continue
        
    print(f"Loading parameter dataset: {filename}")
    with open(input_path, "r") as f:
        dataset = json.load(f)
        
    params = dataset["parameters"]
    q_val, n, k, r, m = params["q"], params["n"], params["k"], params["r"], params["m"]
    
    # Inicializar cuerpos y espacios vectoriales de SageMath para la reconstrucción
    Fq = GF(q_val)
    Fqm = GF(q_val**m, name='z')
    V_Fq = Fqm.vector_space(map=False)

    # 1. Pre-cargar y reconstruir las estructuras algebraicas desde las listas serializadas del JSON
    parsed_instances = []
    for instance in dataset["instances"]:
        instance_id = instance["instance_id"]
        
        # Reconstruir el vector e_secret original algebraico y su correspondiente matriz E
        e_secret = vector(Fqm, [Fqm(V_Fq(vector(Fq, instance["e"][j]))) for j in range(n)])
        E = matrix(Fq, [V_Fq(el) for el in e_secret]).transpose()
        
        parsed_instances.append({
            "instance_id": instance_id,
            "e_original_bits": instance["e"],
            "E_original": E,
            "e_secret": e_secret
        })

    total_instances = len(parsed_instances)

    # 2. Simular un JSON independiente por cada valor de beta usando la misma E original
    for beta in beta_values:
        output_filename = f"simulation_data_n{n}_k{k}_r{r}_m{m}_beta_{beta}.json"
        output_path = os.path.join(results_dir, output_filename)
        
        # --- CARGA DEL CHECKPOINT EXISTENTE ---
        processed_instances = {}
        if os.path.exists(output_path):
            print(f"\n   [CHECKPOINT] Existing log detected for beta={beta}. Parsing progress...")
            try:
                with open(output_path, "r") as pf:
                    existing_data = json.load(pf)
                    for record in existing_data.get("results", []):
                        processed_instances[record["instance_id"]] = record
                print(f"   [CHECKPOINT] Resuming execution. {len(processed_instances)}/{total_instances} instances already completed.")
            except Exception as e:
                print(f"   [CHECKPOINT ERROR] Failed to safely parse checkpoint file ({e}). Starting fresh for this beta.")
                processed_instances = {}

        results_payload = {
            "meta_parameters": {
                "q": q_val, "m": m, "n": n, "k": k, "r": r,
                "alpha": 0.01, "beta": beta
            },
            "results": list(processed_instances.values())
        }
        
        print(f"\n   -> Simulating leakage for Beta = {beta}:")
        
        for idx, instance_obj in enumerate(parsed_instances):
            instance_id = instance_obj["instance_id"]
            
            # Omitir si ya fue procesada bajo esta configuración
            if instance_id in processed_instances:
                continue
                
            print(f"      [Instance {idx + 1}/{total_instances}] Processing Instance ID: {instance_id} ... ", end="", flush=True)
            
            E = instance_obj["E_original"]
            e_secret = instance_obj["e_secret"]
            
            # Recibimos los 5 argumentos retornados por la función (incluyendo posterior_matrix)
            E_tilde_vector, E_tilde_matrix, posteriors, sorted_cols, posterior_matrix = simulate_and_analyze_leakage(
                E, Fqm, r, q_val, m, n, alpha=0.01, beta=beta
            )
            
            # Formatear la matriz E original para el volcado de JSON
            E_original_serialized = [list(row) for row in E.rows()]
            # Convertir las variables calculadas de SageMath a listas serializables standard
            E_tilde_matrix_serialized = [list(row) for row in E_tilde_matrix.rows()]
            E_tilde_vector_serialized = [list(V_Fq(el)) for el in E_tilde_vector]
            
            # --- SERIALIZACIÓN PARA POSTERIORS DICCIONARIO ---
            posteriors_serialized = {str(j): list(val) for j, val in posteriors.items()}
                
            # --- SERIALIZACIÓN PARA SORTED_COLS ---
            sorted_cols_serialized = [int(col_id) for col_id in sorted_cols]

            
            instance_record = {
                "instance_id": instance_id,
                "beta": beta,
                "e_original": instance_obj["e_original_bits"],
                "E_original": [[int(val) for val in row] for row in E_original_serialized],
                "E_tilde_vector": [[int(val) for val in row] for row in E_tilde_vector_serialized],
                "E_tilde_matrix": [[int(val) for val in row] for row in E_tilde_matrix_serialized],
                "posteriors": posteriors_serialized,
                "sorted_cols": sorted_cols_serialized,
                "posterior_matrix": posterior_matrix  # Guardamos la matriz del mismo tamaño que E (m x n)
            }
            
            # Añadir a nuestra memoria interna de progreso para checkpoints
            processed_instances[instance_id] = instance_record
            results_payload["results"] = list(processed_instances.values())
            
            print("Done!")
            
            # --- GUARDADO PERIÓDICO (Cada 10 instancias o al final) ---
            if (idx + 1) % 10 == 0 or (idx + 1) == total_instances:
                print(f"      >> [SAVING CHECKPOINT] Storing progress up to index {idx + 1}...")
                with open(output_path, "w") as out_f:
                    json.dump(results_payload, out_f, indent=4)
            
        print(f"   -> Finished simulation of Beta = {beta}!")
        
    print(f"\nSuccessfully processed all beta configurations for {filename}!\n")

print("\n[SUCCESS] Multi-Beta generation pipeline completed. JSON files stored in 'data_for_seeds/'.")