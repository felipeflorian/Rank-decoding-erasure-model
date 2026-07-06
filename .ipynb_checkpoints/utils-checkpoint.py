from sage.matrix.matrix_space import MatrixSpace
from sage.all import GF, vector, identity_matrix, ceil, VectorSpace, matrix, PolynomialRing, Integer
import itertools
import random 

def serialize_element_to_bits(el, V_Fq):
    """
    Global helper function to map an Fqm extension element into its 
    standard Fq coefficient vector bit list.
    """
    return [int(bit) for bit in V_Fq(el)]

def full_entry_ap_decoder_module(H, s, J, a_J, q, m, n, k, B_hyb=1):
    """
    Implements the complete Full-Entry Erasure-Aware AP Decoder Module (Algorithm 5).
    """
    Fq = GF(q)
    Fqm = H.base_ring()
    V_Fq = Fqm.vector_space(map=False)
    
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
    Calculates bit-level posterior probabilities matching the mathematical 
    formulas derived under Bayes' rule under a uniform coordinate prior.
    """
    if obs_bit == 0:
        denom = (1.0 - alpha) * (1.0 - rho) + beta * rho
        p_0 = ((1.0 - alpha) * (1.0 - rho)) / denom
        p_1 = (beta * rho) / denom
    else:
        denom = alpha * (1.0 - rho) + (1.0 - beta) * rho
        p_0 = (alpha * (1.0 - rho)) / denom
        p_1 = ((1.0 - beta) * rho) / denom
        
    return p_0, p_1

def simulate_and_analyze_leakage(E, Fqm, r, q, m, n, alpha=0.01, beta=0.20):
    """
    Simulates asymmetric cold-boot bit leakage over the coordinates of the matrix E
    and computes full extension-field entry posteriors using bitwise factorization.
    """
    V_Fq = Fqm.vector_space(map=False)
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
    
    return E_tilde_vector, E_tilde_matrix, posteriors, sorted_cols

def grs_scalar_erasures(q, m, n, d, support_basis, Fqm, H, s, T, a_T, r, B_rec=128):
    
    """
        GRS Scalar Erasures Decoder (Algorithm 2: GRS Reconstruction with Scalar Erasures).
    
        This module builds a base-field linear system of equations combining extension-field 
        syndrome constraints and side-channel scalar erasures, verifies system consistency, 
        reconstructs potential solutions, and applies local rank filtering.
    
        Parameters:
        -----------
        q : int
            The size of the base field GF(q).
        m : int
            The extension degree defining the extension field GF(q^m) over GF(q).
        n : int
            The block length of the rank-metric code.
        d : int
            The dimension of the candidate super-support space.
        support_basis : list
            A list of d elements from GF(q^m) forming a basis for the super-support space.
        Fqm : FiniteField
            The SageMath extension field object instance representing GF(q^m).
        H : Matrix
            The (n - k) x n parity-check matrix of the code over GF(q^m).
        s : Vector
            The syndrome vector of length (n - k) over GF(q^m).
        T : list of tuples
            The set of leaked/guessed side-channel coordinate positions (column_index, bit_position).
        a_T : dict
            A dictionary mapping each coordinate pair in T to its respective leaked base-field bit value.
        r : int
            The target rank weight parameter used for filtering final candidates.
        B_rec : int, optional
            The maximum solution reconstruction budget cap for enumeration (default: 128).
    
        Returns:
        --------
        E_F : list
            A list containing all reconstructed error candidate vectors matching rank r.
        equations : list
            A list of compiled linear polynomial expressions expanded over the base field.
        beta_matrix : list of lists
            The structured matrix holding pointers to the multivariate polynomial variables.
        R_poly : PolynomialRing
            The underlying multivariate polynomial ring over GF(q^m) used for symbol tracking.
    """
    
    E_F = []
    
    # Introduce algebraic variables over Fqm to facilitate direct multiplication
    var_names = [f"beta_{lambda_}_{j}" for lambda_ in range(1, d + 1) for j in range(1, n + 1)]
    R_poly = PolynomialRing(Fqm, var_names)
    betas = R_poly.gens()
    
    # Map flat variables to a d x n list structure matching \beta_{\lambda, j}
    beta_matrix = [[betas[(lambda_ * n) + j] for j in range(n)] for lambda_ in range(d)]
    
    # define: e_j = \sum \beta_{\lambda, j} * f_{\lambda}
    e = []
    for j in range(n):
        e_j = sum(beta_matrix[lambda_][j] * support_basis[lambda_] for lambda_ in range(d))
        e.append(e_j)
        
    equations = []
    Fq = GF(q)
    V_Fq = Fqm.vector_space(map=False)
    
    # Compute H * e(\beta)^\top - s^\top = 0
    n_minus_k = H.nrows()
    for i in range(n_minus_k):
        
        # Compute syndrome equations over extension field F_q^m
        syndrome_eq = sum(H[i, j] * e[j] for j in range(n)) - s[i]

        # splits the syndrome equation in the extension field into m separate F_q equations
        for pos in range(m): 
            fq_expr = R_poly(0) #initialize polynomial as 0
            
            for lambda_ in range(d):
                for j in range(n):
                    
                    # split the polynomial into vector components
                    # extract the extension field coefficient of the variable
                    coeff = syndrome_eq.coefficient(beta_matrix[lambda_][j]) 
                    if coeff != 0:
                        # converts coefficient into a vector over F_q
                        fq_coeff = Fq(V_Fq(Fqm(coeff))[pos])
                        fq_expr += fq_coeff * beta_matrix[lambda_][j]
            
            # Extract base field part of the constant term
            const_term = syndrome_eq.constant_coefficient()
            if const_term != 0:
                
                # extract the field component of the syndrome constant
                fq_expr += Fq(V_Fq(Fqm(const_term))[pos])
                
            equations.append(fq_expr) 

            
    for (j, ell) in T:
        val_erasure = R_poly(0)
        for lambda_ in range(d):
            val = Fq(V_Fq(support_basis[lambda_])[ell])
            val_erasure += val * beta_matrix[lambda_][j] 
            
        # Subtract the target leakage hint to set the expression equal to 0
        hint = Fq(a_T[(j, ell)])
        val_erasure -= hint
        
        equations.append(val_erasure)

    
    # Convert the polynomial equations into an explicit matrix system A * beta = b
    num_vars = d * n
    A_list = []
    b_list = []
    
    for eq in equations:
        row_coeffs = []
        for lambda_ in range(d):
            for j in range(n):
                
                # Extract numerical coefficient from polynomial
                row_coeffs.append(Fq(eq.coefficient(beta_matrix[lambda_][j])))
        A_list.append(row_coeffs)
        
        # val syndrome
        b_list.append(-Fq(eq.constant_coefficient()))
        
    A = matrix(Fq, A_list)
    b = vector(Fq, b_list)
    

    matrix_aug = A.augment(b)
    if A.rank() != matrix_aug.rank(): 
        print("System is inconsistent. Returning empty list E_F.")
        return E_F, equations, beta_matrix, R_poly


    # Find a particular solution and the kernel
    particular_sol = A.solve_right(b)
    kernel_basis = A.right_kernel().basis() #
    num_free_vars = len(kernel_basis)
    
    # q^num_free_vars total of solutions
    total_solutions = min(q^num_free_vars, B_rec) 
    print(f"[+] Step 8: System has {num_free_vars} free variables. Enumerating up to {total_solutions} solutions.")
    
    raw_candidates = []
    
    # Loop over binary representation of integers up to total_solutions to mix kernel combinations
    for idx in range(total_solutions):
        
        current_sol = particular_sol
        
        # Convert integer to base-q digit representations to pick free variable coefficients
        digits = Integer(idx).digits(base=q)
        digits += [0] * (num_free_vars - len(digits))
        
        for k_idx, scalar in enumerate(digits):
            current_sol += Fq(scalar) * kernel_basis[k_idx]
            
        sol_e = []
        for j in range(n):
            # Compute e_j = \sum \beta_{\lambda, j} * f_{\lambda} using the numerical solution
            e_j_val = sum(current_sol[lambda_ * n + j] * support_basis[lambda_] for lambda_ in range(d))
            sol_e.append(e_j_val)
            
        raw_candidates.append(vector(Fqm, sol_e))
        
    # Filter the candidates by those who has rank r
    for candidate_vector in raw_candidates:
        
        # Convert the extension field elements into columns of a base-field matrix
        matrix_representation = matrix(Fq, [V_Fq(element) for element in candidate_vector]).transpose()
        if matrix_representation.rank() == r:
            E_F.append(candidate_vector)
            
    return E_F, equations, beta_matrix, R_poly

def verify_decoder_candidates(E_F, H, s, r, Fqm):
    """
    Validates all candidate error vectors recovered by the decoder of GRS scalar erasure.
    Checks syndrome consistency and global rank metrics over the base field field.
    """
    Fq = H.base_ring().base_ring()
    V_Fq = Fqm.vector_space(map=False)
    valid_solutions = []

    
    for idx, candidate in enumerate(E_F):
        syndrome_check = (H * candidate == s)
        
        matrix_rep = matrix(Fq, [V_Fq(element) for element in candidate]).transpose()
        rank_check = (matrix_rep.rank() == r)
        
        if syndrome_check and rank_check:
            valid_solutions.append(candidate)
        else:
            reasons = []
            if not syndrome_check: reasons.append("Syndrome Mismatch")
            if not rank_check: reasons.append(f"Rank Weight is {matrix_rep.rank()} (Expected {r})")
            
    return valid_solutions

def execute_and_verify_ap_instance(instance_data, J, a_J):
    """
    Orchestrates structural unpacking, runs Algorithm 5 explicitly given 
    the picked indices J and known column field elements a_J, and validates 
    results using explicit structural rank and syndrome checks.
    """
    params = instance_data["parameters"]
    q, m, n, k, r = params["q"], params["m"], params["n"], params["k"], params["r"]
    n_minus_k = n - k
    
    Fq = GF(q)
    Fqm = GF(q**m, name='z')
    V_Fq = Fqm.vector_space(map=False)
    
    # Rebuild H Matrix from stored bit representation
    H_rows = []
    for i in range(n_minus_k):
        row_elements = [Fqm(V_Fq(vector(Fq, instance_data["H"][i][j]))) for j in range(n)]
        H_rows.append(row_elements)
    H = matrix(Fqm, H_rows)
    
    # Rebuild Syndrome Vector s
    s = vector(Fqm, [Fqm(V_Fq(vector(Fq, instance_data["s"][i]))) for i in range(n_minus_k)])
    
    # Rebuild True Error Vector for validation verification
    e_secret = vector(Fqm, [Fqm(V_Fq(vector(Fq, instance_data["e"][j]))) for j in range(n)])
    
    # Trigger Decoder Execution using the passed J and a_J arguments
    candidate_list = full_entry_ap_decoder_module(H, s, J, a_J, q, m, n, k)
    print(f"Number of candidates returned: {len(candidate_list)}")

    # Final Verification
    success = False
    recovered_e_serialized = None
    
    for idx, candidate in enumerate(candidate_list):
        syndrome_check = (H * candidate == s)
        cand_matrix = matrix(Fq, [V_Fq(el) for el in candidate]).transpose()
        rank_check = (cand_matrix.rank() == r)
        
        if syndrome_check and rank_check:
            print(f"[+] Candidate #{idx+1} passes validation filters.")
            if candidate == e_secret:
                print("Reconstructed vector matches the secret error exactly!")
                success = True
                # Serialize the successfully recovered candidate vector back into binary bits
                recovered_e_serialized = [[int(bit) for bit in V_Fq(el)] for el in candidate]
                break
                
    return {
        "success": success,
        "parameters": {"n": n, "k": k, "r": r, "m": m},
        "erasures_used": J,
        "candidates_found": len(candidate_list),
        "recovered_e": recovered_e_serialized,
        "original_e": instance_data["e"]
    }