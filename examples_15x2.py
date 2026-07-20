from okeanode import random_chunk_candidates, initialize


# Generate 15 chunk lists, each with 2 candidates of 4 bits
chunk_lists = [random_chunk_candidates(2, 4) for _ in range(8)]

print(chunk_lists)
# Build OKEA tree
root = initialize(chunk_lists, 0, 7)

print("Enumerating all full key candidates:")
count = 0
total=2**2
while True:
    cand = root.next_candidate()
    if cand is None:
        break
    count += 1
    print(cand.score)
    # print(cand.to_weight())
    print(cand.bits)
    # print(f"{count}: weight={{cand.to_weight()}}, score={{cand.score:.4f}}, bits={{cand.bits.to01()}}")

if count < total:
    print(f"⚠️ Only {{count}} candidates generated")
else:
    print("✅ All candidates generated successfully.")
