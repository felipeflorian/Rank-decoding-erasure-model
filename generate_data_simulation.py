import os
import json
import random
from sage.all import GF, matrix, vector, VectorSpace

# --- Tailored Parameter Configurations ---
q = 2  # Fixed binary base field F_2
configurations = [
    {"n": 7,  "k": 3,  "r": 2,  "m": 4},
    {"n": 15, "k": 7,  "r": 5,  "m": 15},
    {"n": 33, "k": 15, "r": 10, "m": 31}
]
instances_per_config = 50
max_attempts = 300  # Bounded iteration threshold to prevent infinite loops

# Target output directory setup (Volvemos a generated_data)
output_dir = "generated_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

Fq = GF(q)

print(f"--- Starting Portable JSON Pipeline: Writing 20 Instances per Config to '{output_dir}/' ---")

for config in configurations:
    n, k, r, m = config["n"], config["k"], config["r"], config["m"]
    n_minus_k = n - k
    
    print(f"\nProcessing configuration: n={n}, k={k}, r={r} over extension field m={m}")
    
    # Instantiate large field environments
    Fqm = GF(q**m, name='z')
    V_Fq = Fqm.vector_space(map=False)  # map=False suppresses Sage free_module warnings
    VS_base = VectorSpace(Fq, m)
    
    config_instances_payload = []
    
    for instance_idx in range(instances_per_config):
        H_mat = None
        e_secret = None
        
        # 1. Sample a Full-Row-Rank Parity-Check Matrix H
        sampled_H = False
        for _ in range(max_attempts):
            test_H = matrix(Fqm, n_minus_k, n, [Fqm.random_element() for _ in range(n_minus_k * n)])
            if test_H.rank() == n_minus_k:
                H_mat = test_H
                sampled_H = True
                break
        if not sampled_H:
            raise RuntimeError(f"Critical Error: Failed to find full rank H after {max_attempts} attempts.")

        # 2. Construct a Secret Error Vector e with Exact Rank Weight r
        sampled_e = False
        for _ in range(max_attempts):
            support_basis = []
            basis_attempts = 0
            
            # Select r linearly independent extension field elements over Fq
            while len(support_basis) < r and basis_attempts < max_attempts:
                basis_attempts += 1
                elem = Fqm.random_element()
                if elem != 0 and V_Fq(elem) not in VS_base.subspace([V_Fq(b) for b in support_basis]):
                    support_basis.append(elem)
            
            if len(support_basis) < r:
                continue
            
            # Populate error vector via random Fq combinations of the support basis elements
            e_list = []
            for _ in range(n):
                coeffs = [Fq.random_element() for _ in range(r)]
                entry = sum(int(coeffs[i]) * support_basis[i] for i in range(r))
                e_list.append(entry)
                
            test_e = vector(Fqm, e_list)
            
            # Double check exact target rank verification check
            E_matrix = matrix(Fq, [V_Fq(el) for el in test_e]).transpose()
            if E_matrix.rank() == r:
                e_secret = test_e
                sampled_e = True
                break
                
        if not sampled_e:
            raise RuntimeError(f"Critical Error: Failed to build valid rank-{r} error vector after {max_attempts} attempts.")
            
        # 3. Compute associated Syndrome vector s
        s_vec = H_mat * e_secret
        
        # --- Serialization Helpers (Converts extension elements to portable binary arrays) ---
        def serialize_element(el):
            return [int(bit) for bit in V_Fq(el)]
            
        H_serialized = [[serialize_element(H_mat[i, j]) for j in range(n)] for i in range(n_minus_k)]
        s_serialized = [serialize_element(s_vec[i]) for i in range(n_minus_k)]
        e_serialized = [serialize_element(e_secret[j]) for j in range(n)]
        
        config_instances_payload.append({
            "instance_id": instance_idx + 1,
            "H": H_serialized,
            "s": s_serialized,
            "e": e_serialized
        })
    
    # 4. Save the 20 instances collected under this configuration to its unique JSON file
    file_name = f"instances_n{n}_k{k}_r{r}_m{m}.json"
    file_path = os.path.join(output_dir, file_name)
    
    output_payload = {
        "parameters": {"q": q, "m": m, "n": n, "k": k, "r": r},
        "total_instances": len(config_instances_payload),
        "instances": config_instances_payload
    }
    
    with open(file_path, "w") as f:
        json.dump(output_payload, f, indent=4)
        
    print(f"[SAVED] Created '{file_name}' containing {instances_per_config} instances.")

print(f"\n[SUCCESS] Completed generation. All clean cryptographic matrices stored inside '{output_dir}/'.")