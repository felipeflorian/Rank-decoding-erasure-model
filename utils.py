from sage.matrix.matrix_space import MatrixSpace
from sage.all import GF,  vector, identity_matrix, ceil, VectorSpace, matrix
import itertools
from sage.all import GF,  vector, identity_matrix, ceil, VectorSpace, matrix
from sage.matrix.matrix_space import MatrixSpace
import random 

def full_entry_ap_decoder_module(H, s, J, a_J, q, m, n, k, B_hyb=1):
    """
    Implements the complete Full-Entry Erasure-Aware AP Decoder Module (Algorithm 5).
    """
    Fq = GF(q)
    Fqm = H.base_ring()
    V_Fq = Fqm.vector_space()
    
    # -------------------------------------------------------------
    # Input Normalization & Feasibility Check
    # -------------------------------------------------------------
    G = H.right_kernel().basis_matrix()
    y = H.solve_right(s)
    
    # Reconstruct target rank weight boundary dynamically
    r = int(H.ncols() - G.nrows())
    
    if len(J) > 0 and len(a_J) > 0:
        erasure_subspace = VectorSpace(Fq, m).subspace([V_Fq(aj) for aj in a_J])
        rho_J = erasure_subspace.dimension()
        
        # Explicitly build the field power basis via generator powers
        field_basis = [Fqm.gen()**i for i in range(m)]
        A_J_basis = []
        for b in erasure_subspace.basis():
            field_el = sum(b[i] * field_basis[i] for i in range(m))
            A_J_basis.append(Fqm(field_el))
    else:
        rho_J = 0
        A_J_basis = []
        
    if rho_J > r:
        return [] 
        
    # -------------------------------------------------------------
    # Affine Space Reduction
    # -------------------------------------------------------------
    if len(J) == 0:
        u0 = vector(Fqm, [0]*k)
        R = identity_matrix(Fqm, k)
        h_J = 0
    else:
        G_J = matrix(Fqm, [G.column(j) for j in J]).transpose()
        y_J = vector(Fqm, [y[j] for j in J])
        a_J_vec = vector(Fqm, a_J)
        try:
            u0 = G_J.solve_left(y_J - a_J_vec)
        except ValueError:
            return [] 
        h_J = G_J.rank()
        R = G_J.left_kernel().basis_matrix()
        
    y_prime = y - u0 * G
    G_prime = R * G
    
    # -------------------------------------------------------------
    # System Deficit Evaluation & Hybridization Loop
    # -------------------------------------------------------------
    hAP = (r + 1) * (k + 1) - 1 - n
    d_AP_J = hAP + len(J) - rho_J - (r + 1) * h_J
    t_res = max(0, int(ceil(d_AP_J / r)))
    
    if t_res > min(k - h_J, n - len(J)):
        return [] 
        
    if t_res == 0:
        B_hyb = 1
    else:
        B_hyb = int(ceil(q**(r * t_res)))
    
    E_AP = [] 
    Jc = [j for j in range(n) if j not in J]
    n_prime = len(Jc)
    
    # Execute the Hybridization Trial Search Loop
    for b in range(1, B_hyb + 1):
        if t_res == 0:
            y_hat = y_prime
            G_hat = G_prime
            k_reduced = k - h_J
            active_columns = Jc
            Q_inv = None
        else:
            # Sample an F_q linear coordinate transformation matrix
            while True:
                Q_U = matrix(Fq, n_prime, n_prime, [Fq.random_element() for _ in range(n_prime**2)])
                if Q_U.is_invertible():
                    break
            
            # Embed Q_U into a block diagonal structure Q
            Q = identity_matrix(Fqm, n)
            for r_idx in range(n_prime):
                for c_idx in range(n_prime):
                    Q[Jc[r_idx], Jc[c_idx]] = Fqm(Q_U[r_idx, c_idx])
            
            # Mix the coordinate systems
            y_hat = y_prime * Q
            G_hat = G_prime * Q
            
            # Hypothesize that the last t_res mixed positions in Jc evaluate to 0
            zero_target_cols = Jc[-t_res:]
            G_zero = matrix(Fqm, [G_hat.column(col) for col in zero_target_cols]).transpose()
            y_zero = vector(Fqm, [y_hat[col] for col in zero_target_cols])
            
            # Local geometric consistency compatibility check
            try:
                u_zero = G_zero.solve_left(y_zero)
            except ValueError:
                continue 
                
            # Perform intermediate matrix shrinkage using local null-space
            R_zero = G_zero.left_kernel().basis_matrix()
            y_hat = y_hat - u_zero * G_hat
            G_hat = R_zero * G_hat
            k_reduced = G_hat.nrows() 
            active_columns = [j for j in Jc if j not in zero_target_cols]
            Q_inv = Q.inverse() 
            
        # -------------------------------------------------------------
        # Algebraic Assembly & Solving
        # -------------------------------------------------------------
        num_vars = k_reduced + r + (r * k_reduced)
        eqs_matrix = []
        eqs_rhs = []
        
        # Construct Linearized AP equations for unfixed columns
        for j in active_columns:
            g_j_hat = G_hat.column(j)
            y_j_hat = y_hat[j]
            
            row = [Fqm(0)] * num_vars
            for l in range(k_reduced):
                row[l] = -(g_j_hat[l])**(q**r)
            for i in range(r):
                row[k_reduced + i] = y_j_hat**(q**i)
            idx_v = k_reduced + r
            for i in range(r):
                for l in range(k_reduced):
                    row[idx_v] = -(g_j_hat[l])**(q**i)
                    idx_v += 1
                    
            eqs_matrix.append(row)
            eqs_rhs.append(-(y_j_hat**(q**r)))
            
        # Inject Known-Root Equations from erasure spans
        for c in A_J_basis:
            row = [Fqm(0)] * num_vars
            for i in range(r):
                row[k_reduced + i] = c**(q**i)
            eqs_matrix.append(row)
            eqs_rhs.append(-(c**(q**r)))
            
        # Solve Assembled System (Verify list is not empty explicitly)
        if len(eqs_matrix) == 0:
            continue 
            
        M_sys = matrix(Fqm, eqs_matrix)
        V_sys = vector(Fqm, eqs_rhs)
        
        try:
            sol_flat = M_sys.solve_right(V_sys)
        except ValueError:
            continue 
            
        # -------------------------------------------------------------
        # Filtering & Reconstruction
        # -------------------------------------------------------------
        U_part = vector(Fqm, sol_flat[0:k_reduced])
        P_part = vector(Fqm, sol_flat[k_reduced:k_reduced+r])
        V_flat = sol_flat[k_reduced+r:]
        
        V_part = []
        idx = 0
        for i in range(r):
            sub_v = []
            for l in range(k_reduced):
                sub_v.append(V_flat[idx])
                idx += 1
            V_part.append(vector(Fqm, sub_v))
            
        # Componentwise Inverse Frobenius execution to unpack variable w
        w = vector(Fqm, [val**(q**(m - r)) for val in U_part])
        
        # Local Consistency Verification
        is_consistent = True
        for i in range(r):
            calculated_V = P_part[i] * vector(Fqm, [el**(q**i) for el in w])
            if calculated_V != V_part[i]:
                is_consistent = False
                break
                
        if not is_consistent:
            continue 
            
        # Reconstruct & Lift to Full Representation
        e_candidate_mixed = y_hat - w * G_hat
        
        # Undo residual hybrid mixing base mapping, if activated
        if Q_inv is not None:
            e_candidate = e_candidate_mixed * Q_inv
        else:
            e_candidate = e_candidate_mixed
            
        # Explicitly pin original known erasure coordinates back
        for idx_j, real_j in enumerate(J):
            e_candidate[real_j] = a_J[idx_j]
            
        if e_candidate not in E_AP:
            E_AP.append(e_candidate)
            
    return E_AP

def compute_bitwise_posteriors(obs_bit, rho, alpha=0.01, beta=0.20):
    """
    Calculates bit-level posterior probabilities exactly as written in 
    the provided formula image using Bayes' rule under a uniform coordinate prior.
    """
    if obs_bit == 0:
        # Top half of the image formulas (If x_tilde_i = 0)
        denom = (1.0 - alpha) * (1.0 - rho) + beta * rho
        p_0 = ((1.0 - alpha) * (1.0 - rho)) / denom
        p_1 = (beta * rho) / denom
    else:
        # Bottom half of the image formulas (If x_tilde_i = 1)
        denom = alpha * (1.0 - rho) + (1.0 - beta) * rho
        p_0 = (alpha * (1.0 - rho)) / denom
        p_1 = ((1.0 - beta) * rho) / denom
        
    return p_0, p_1


def simulate_and_analyze_leakage(E, Fqm, r, q, m, n, alpha=0.01, beta=0.20):
    """
    Simulates asymmetric cold-boot bit leakage over the coordinates of the matrix E
    and computes full extension-field entry posteriors using bitwise factorization.
    
    Arguments:
    ----------
    E : An m x n Matrix over the base field Fq (The true secret error matrix)
    """
    V_Fq = Fqm.vector_space()
    Fq = GF(q)
    
    # Calculate global prior rho = target_rank_weight / code_length as specified
    rho = float(r / n)
    
    all_elements = list(Fqm)
    element_to_bits = {el: tuple(V_Fq(el)) for el in all_elements}
    
    # --- Step A: Simulate Asymmetric Cold-Boot Physical Noise ---
    e_tilde_list = []
    bit_leakage_matrix = [] 
    
    # Process the matrix column-by-column (E.column(j))
    for j in range(n):
        true_bits = E.column(j)
        observed_bits = []
        for bit in true_bits:
            if bit == 1:
                obs = Fq(1) if random.random() >= alpha else Fq(0)
            else:
                obs = Fq(0) if random.random() >= beta else Fq(1)
            observed_bits.append(obs)
            
        bit_leakage_matrix.append(observed_bits)
        e_tilde_list.append(Fqm(observed_bits))
        
    E_tilde_vector = vector(Fqm, e_tilde_list)
    # Correctly build the m x n base field matrix from the corrupted rows
    E_tilde_matrix = matrix(Fq, bit_leakage_matrix).transpose()
    
    # --- Step B: Calculate Entry-Wise Posteriors and Sort Columns ---
    posteriors = []
    for j in range(n):
        column_posteriors = {}
        
        for element in all_elements:
            candidate_bits = element_to_bits[element]
            column_weight = 1.0
            
            for l in range(m):
                obs_b = int(bit_leakage_matrix[j][l])
                cand_b = int(candidate_bits[l])
                
                p_0, p_1 = compute_bitwise_posteriors(obs_b, rho, alpha, beta)
                column_weight *= p_1 if cand_b == 1 else p_0
                
            column_posteriors[element] = column_weight
            
        posteriors.append(column_posteriors)
        
    # Evaluate Maximum A Posteriori (MAP) confidence profiles
    column_scores = [(j, max(posteriors[j].values())) for j in range(n)]
    sorted_cols = sorted(column_scores, key=lambda x: x[1], reverse=True)
    
    # Returns exactly 4 values to match your unpacked sequence
    return E_tilde_vector, E_tilde_matrix, posteriors, sorted_cols