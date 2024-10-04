from sentence_transformers import SentenceTransformer, util
import json
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def is_similar(str_a, str_b):
    embedding_a = model.encode(str_a, convert_to_tensor=True)
    embedding_b = model.encode(str_b, convert_to_tensor=True)

    cosine_score = util.cos_sim(embedding_a, embedding_b)
    cosine_score = cosine_score.cpu().numpy()
    #print(cosine_score[0][0])
    if cosine_score[0][0] >= 0.8:
        return True
    else:
        return False

def lcs(X, Y, m, n, dp):

    if (m == 0 or n == 0):
        return 0

    if (dp[m][n] != -1):
        return dp[m][n]

    if is_similar(X[m-1], Y[n-1]):
        dp[m][n] = 1 + lcs(X, Y, m - 1, n - 1, dp)
        return dp[m][n]

    dp[m][n] = max(lcs(X, Y, m, n - 1, dp) , lcs(X, Y, m - 1, n, dp))
    return dp[m][n]

def calculate_len(filein):
    lcs_len = []
    lcs_norm = []
    for i, line in enumerate(open(filein)):
        if i % 10 == 0:
            print(i)
        data = json.loads(line.strip())
        candidate = data["candidate"].split("\n")
        reference = data["reference"].split("\n")
        m = len(candidate)
        n = len(reference)
        lcs_mat = np.zeros((m+1, n+1)) - 1.
        lcs_value = lcs(candidate, reference, m, n, lcs_mat)
        lcs_len.append(lcs_value)
        lcs_norm_value = lcs_value / max(m, n)
        lcs_norm.append(lcs_norm_value)

    print(lcs_len)
    print(lcs_norm)
    print(np.mean(lcs_len))
    print(np.mean(lcs_norm))
    return lcs_len

if __name__ == "__main__":
    #calculate_len("gpt-4v_pairs.jsonl")
    calculate_len("gemini_new_pairs.jsonl")
    #calculate_len("claude_v3_haiku_pairs.jsonl")
    #calculate_len("claude_v3_sonnet_pairs.jsonl")

