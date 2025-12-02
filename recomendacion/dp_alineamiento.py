# dp_alineamiento.py
def needleman_wunsch_score(seq1, seq2, match=2, mismatch=-1, gap=-1):
    """
    devuelve un score normalizado entre 0 y 1 indicando similitud de dos secuencias (generos).
    implementa una version simple del algoritmo needleman-wunsch.
    """
    if not seq1 or not seq2:
        return 0.0

    m = len(seq1)
    n = len(seq2)
    # matriz de dp
    dp = [[0]*(n+1) for _ in range(m+1)]

    # inicializa
    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] + gap
    for j in range(1, n+1):
        dp[0][j] = dp[0][j-1] + gap

    # rellena
    for i in range(1, m+1):
        for j in range(1, n+1):
            if seq1[i-1] == seq2[j-1]:
                score = match
            else:
                score = mismatch
            dp[i][j] = max(
                dp[i-1][j-1] + score,
                dp[i-1][j] + gap,
                dp[i][j-1] + gap
            )

    raw_score = dp[m][n]

    # normaliza: maximo posible seria match * min(m,n) (si todo coincide)
    max_possible = match * min(m,n)
    if max_possible <= 0:
        return 0.0
    normalized = (raw_score + abs(mismatch)*max(m,n)) / (max_possible + abs(mismatch)*max(m,n))
    # retorna en [0,1]
    normalized = max(0.0, min(1.0, normalized))
    return normalized
