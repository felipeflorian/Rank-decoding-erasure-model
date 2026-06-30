import os
import json
import time
import random
from sage.all import ceil, GF, vector
from utils import execute_and_verify_ap_instance

input_dir = "generated_data"
results_dir = os.path.join("evaluation_results", "true_columns")
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

target_files = [
    "instances_n7_k3_r2_m4.json",
    "instances_n15_k7_r5_m15.json",
    "instances_n33_k15_r10_m31.json"
]

print("--- Starting AP-Decoder Evaluation Pipeline (Random True Columns) ---")
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
    
    hAP = (r + 1) * (k + 1) - 1 - n
    fixed_j_size = int(ceil(hAP / (r + 1)))
    print(f" -> Config attributes: n={n}, k={k}, r={r}, m={m} | Fixed Erasure Size |J| = {fixed_j_size}")
    
    output_filename = f"results_n{n}_k{k}_r{r}_m{m}.json"
    output_path = os.path.join(results_dir, output_filename)
    
    processed_instances = {}
    if os.path.exists(output_path):
        print(f"    [CHECKPOINT] Existing log detected for {output_filename}. Parsing progress...")
        try:
            with open(output_path, "r") as pf:
                existing_data = json.load(pf)
                for record in existing_data.get("results", []):
                    processed_instances[record["instance_id"]] = record
            print(f"    [CHECKPOINT] Resuming execution. {len(processed_instances)}/20 instances already completed.")
        except Exception as e:
            print(f"    [CHECKPOINT ERROR] Failed to safely parse checkpoint file ({e}). Starting fresh.")
            processed_instances = {}

    results_payload = {
        "meta_parameters": {"q": q_val, "m": m, "n": n, "k": k, "r": r, "fixed_erasure_size": fixed_j_size},
        "results": list(processed_instances.values())
    }

    # Initialize fields to safely extract field objects out of JSON arrays
    Fq = GF(q_val)
    Fqm = GF(q_val**m, name='z')
    V_Fq = Fqm.vector_space(map=False)

    for idx, instance in enumerate(dataset["instances"]):
        instance_id = instance["instance_id"]
        
        if instance_id in processed_instances:
            continue
            
        print(f"     -> Processing Instance ID: {instance_id}/20 ... ", end="", flush=True)
        instance["parameters"] = params
        
        # Reconstruct algebraic e vector to pick correct field elements for a_J
        e_secret = vector(Fqm, [Fqm(V_Fq(vector(Fq, instance["e"][j]))) for j in range(n)])
        
        # Randomly choose J indices of size fixed_j_size
        all_column_indices = list(range(n))
        J = sorted(random.sample(all_column_indices, fixed_j_size))
        
        # Unpack the absolute true coordinates from the secret error vector directly
        a_J = [e_secret[j] for j in J]
        
        start_time = time.perf_counter()
        original_e_bits = instance["e"]
        recovered_e_bits = None
        
        try:
            # Pass our manually selected indices J and true column values a_J explicitly
            summary = execute_and_verify_ap_instance(instance, J=J, a_J=a_J)
            success_flag = bool(summary["success"])
            recovered_e_bits = summary["recovered_e"]
        except Exception as error:
            print(f"\n    [ERROR] Crash on instance {instance_id}: {error}")
            success_flag = False
            
        execution_time = time.perf_counter() - start_time
        print(f"Done! (Time: {execution_time:.4f}s | Decoded: {success_flag})")
        
        instance_record = {
            "instance_id": instance_id,
            "execution_time_seconds": execution_time,
            "decoded_successfully": success_flag,
            "original_e": original_e_bits,
            "recovered_e": recovered_e_bits
        }
        
        processed_instances[instance_id] = instance_record
        results_payload["results"] = list(processed_instances.values())
        
        with open(output_path, "w") as out_f:
            json.dump(results_payload, out_f, indent=4)
            
    print(f"Successfully finalized log: '{output_filename}'\n")

print("\n[SUCCESS] Pipeline completed successfully. Data stored in 'evaluation_results/true_columns/'.")