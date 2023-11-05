import numpy as np

def entropy(s):
    
    #implementation manually to see if I could do it
    # counts = []
    # seen = [False for i in range(len(s))]
    # for i in range(len(s)):
    #     if (seen[i] == True):
    #         continue
    #     count = 1
    #     for j in range(i + 1, len(s), 1):
    #         if (s[i] == s[j]):
    #             seen[j] = True
    #             count += 1
    #     counts.append(count)
    # countArr = np.asarray(counts)
    # probabilities = countArr / len(s)
    # entropy = 0
    
    count = np.bincount(s)
    probabilities = count/len(s)
    
    for probability in probabilities:
        if probability > 0:
            entropy += probability*np.log2(probability)
    return - entropy
