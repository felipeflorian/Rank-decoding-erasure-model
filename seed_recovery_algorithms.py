import math
from collections import defaultdict
from functools import lru_cache
from itertools import product
from bitarray import bitarray
from candidate import ChunkCandidate
from okeanode import initialize

def build_posteriors_from_tilde(s_tilde, alpha, beta):

    """
        Construct the posterior matrix P according to the leakage s_tilde

        Input:
            - s_tilde: list of bits
            - alpha: Probability of 0 flipping to 1
            - beta: Probability of 1 flipping to 0
        
        Output:
            - P matrix with the posterior probabilities for each bit
    """

    P = []
    for obs_bit in s_tilde:
        if obs_bit == 0:
            denom = (1 - alpha) + beta
            P.append([
                (1 - alpha) / denom,  # Pr(s*_j = 0 | s̃_j = 0)
                beta / denom          # Pr(s*_j = 1 | s̃_j = 0)
            ])
        else:  # obs_bit == 1
            denom = alpha + (1 - beta)
            P.append([
                alpha / denom,        # Pr(s*_j = 0 | s̃_j = 1)
                (1 - beta) / denom    # Pr(s*_j = 1 | s̃_j = 1)
            ])
    return P

def extract_chunk(seed: bitarray, start: int, end: int) -> bitarray:
    return seed[start:end]

@lru_cache(maxsize=4096)
def safe_log(x):
    return math.log(x)

@lru_cache(maxsize=32)
def get_bit_combinations(w):
    """
        Returns all possible bit combinations of length w.
        Uses a cache to avoid recomputing them if they have been calculated before.

        Input:
            - w: int defining the bit width of a chunk to expand

        Output:
            - A list of tuples containing all possible binary value combinations of length w
    """
    return list(product([0, 1], repeat=w))


def generate_candidates(P, W, w, eta, mu, scale=10000.0):

    """
        (Algorithm 1) Generates scored and locally aggregated chunk candidates from raw bitwise 
        posterior probabilities using an optimal key enumeration layout.

        Input:
            - P: list of lists of floats representing the posterior matrix of dimension W x 2
            - W: int defining the absolute length of the secret seed in bits
            - w: int defining the bit width of an individual chunk
            - eta: int defining the number of chunks grouped into a single block
            - mu: int defining the maximum number of top candidates retained per block group

        Output:
            - list of lists containing Candidate objects representing the top mu scored candidates per block
    """

    if W % w != 0:
        raise ValueError("W debe ser divisible por w")
    N = W // w  # Total number of chunks

    if N % eta != 0:
        raise ValueError("Número de chunks no divisible por η")
    xi = N // eta  # Number of blocks

    chunk_lists = []

    # Step 1: Generate candidates for each chunk
    for i in range(N):
        start = i * w
        end = start + w
        P_chunk = P[start:end]
        logs = [(safe_log(float(p0)), safe_log(float(p1))) for (p0, p1) in P_chunk]

        candidates = []
        for bits in get_bit_combinations(w):
            score = -sum(logs[j][bit] for j, bit in enumerate(bits))
            ba = bitarray(bits)
            candidates.append(ChunkCandidate(score, ba))

        candidates.sort(key=lambda c: c.score)
        chunk_lists.append(candidates)

    # Step 2: Group chunks by blocks using xi
    blocks = []
    for i in range(xi):  # use of xi
        start = i * eta
        chunk_group = chunk_lists[start:start+eta]
        okea_tree = initialize(chunk_group, 0, eta - 1, scale=scale)

        block_candidates = []
        for j in range(mu):
            cand = okea_tree.getCandidate(j)
            if cand is None:
                break
            block_candidates.append(cand)

        blocks.append(block_candidates)

        del okea_tree

    return blocks

def create(L, B1, B2, W, w, eta, mu, scale=10000):

    """
        (Algorithm 2) Constructs the dynamic programming counting matrix B, which tracks the path 
        combinations that can satisfy the target quantized boundary.

        Input:
            - L: list of lists containing Candidate objects from Algorithm 1
            - B1: int representing the lower bound of the targeted quantized score band (inclusive)
            - B2: int representing the upper bound of the targeted quantized score band (exclusive)
            - W: int defining the absolute length of the secret seed in bits
            - w: int defining the chunk width in bits
            - eta: int defining the number of chunks grouped into a single block
            - mu: int defining the maximum number of top candidates retained per block group

        Output:
            - list of lists of ints representing the prefix score accumulation matrix B
    """

    N = W // w
    xi = N // eta

    # Internal structure with efficient access
    B_sparse = [defaultdict(int) for _ in range(xi)]

    # Precalculate integer weights per block
    weights_by_block = [
        [
            cand.to_weight(
                scale=10 ** (-math.floor(math.log10(abs(cand.score))))
            )
            for cand in block
        ]
        for block in L
    ]

    # --- Base: last block (i = xi - 1) ---
    i = xi - 1
    for b in range(B2):
        for r in weights_by_block[i]:
            if B1 - b <= r < B2 - b:
                B_sparse[i][b] += 1

    # --- Recursion: previous blocks ---
    for i in reversed(range(xi - 1)):
        for b in range(B2):
            total = 0
            for r in weights_by_block[i]:
                next_b = b + r
                if next_b in B_sparse[i + 1]:
                    total += B_sparse[i + 1][next_b]
            if total > 0:
                B_sparse[i][b] = total

    # --- Conversion to list of lists ---
    B = [[0] * B2 for _ in range(xi)]
    for i in range(xi):
        for b, count in B_sparse[i].items():
            B[i][b] = count

    return B


def rank(L, B1, B2, W, w, eta, mu):
    
    """
        (Algorithm 3) Computes the total number of full-length seed candidates whose cumulative 
        scores fall within the quantized score interval [B1, B2).

        Input:
            - L: list of lists containing Candidate objects generated by Algorithm 1
            - B1: int representing the lower bound of the targeted score band (inclusive)
            - B2: int representing the upper bound of the targeted score band (exclusive)
            - W: int defining the total secret seed length in bits
            - w: int defining the bit-width of an individual chunk
            - eta: int defining the number of chunks grouped into a single block
            - mu: int defining the maximum number of top candidates retained per block group

        Output:
            - An integer representing the total count of valid full-length seed combinations
    """

    B = create(L, B1, B2, W, w, eta, mu)
    return B[0][0]


def getSeed(L, B, B1, B2, W, w, eta, mu, r):

    """
        (Algorithm 4) Reconstructs the exact r-th full-length seed candidate whose total quantized 
        score falls within the range [B1, B2).

        Input:
            - L: list of lists containing precomputed chunk candidate items from Algorithm 1
            - B: matrix (list of lists of ints) generated via Algorithm 2 (create)
            - B1: int representing the lower score threshold of the quantized target band
            - B2: int representing the upper score threshold of the quantized target band
            - W: int defining the absolute length of the complete secret seed in bits
            - w: int defining the chunk width in bits
            - eta: int defining the number of chunks per block group
            - mu: int defining the number of elements capped per block group
            - r: int defining the targeted candidate rank index to reconstruct (1-indexed)

        Output:
            - A string representing the reconstructed full-length seed candidate bitstring, 
              or None if the requested rank 'r' exceeds available candidates
    """

    N = W // w # Total number of chunks
    xi = N // eta # Number of blocks
    
    if r > B[0][0]: # Check if the requested rank is valid
        return None
    
    seed_r = "" # Initialize the reconstructed seed string
    b = 0 # Current prefix score

    # Traverse the DP matrix backwards from the second-to-last block
    for i in range(xi-1):
        for j in range(mu):
            scale = 10 ** (-math.floor(math.log10(abs(L[i][j].score))))
            sc = L[i][j].to_weight(scale=scale) # Get the score of the current chunk candidate
            
            # print(f"este es sc primer for: {sc}")
            # print(L[i][j].score)
            if r <= B[i+1][b + sc]: # Check if the current rank falls within the current candidate's range
                seed_r += L[i][j].bits.to01() # Append the current candidate's bitstring to the seed
                b += sc
                break
            r -= B[i+1][b + sc]

    # --- Base: last block (i = xi - 1) ---
    i_new = xi - 1
    for j in range(mu):
        scale = 10 ** (-math.floor(math.log10(abs(L[i_new][j].score))))
        sc = L[i_new][j].to_weight(scale=scale) # Get the score of the current chunk candidate
        x = 1 if B1-b <= sc < B2-b else 0 # Check if the current rank falls within the current candidate's range
        if r <= x: # Check if the current rank falls within the current candidate's range
            seed_r += L[i_new][j].bits.to01()
            break
        
        r = r - x

    # Return the reconstructed seed string
    return seed_r
