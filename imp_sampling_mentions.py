import numpy as np
import numpy.linalg as la
import pandas as pd
import numpy.random as rn
import matplotlib.pyplot as plt
import scipy.stats as spst

def main():
    rn.seed(0)
    a, b, c = [], [], []
    num_sims = 100
    for _ in range(num_sims):
        num_pairs = 100
        k = 3
        imbalance = 0.8
        pair_labeling_budget = 50
        # NOTE: drop any pairs with fewer than k mentions before the analysis
        num_mentions_by_pair = rn.randint(k, 8, size=num_pairs)
        precisions_by_pair = [rn.rand(m) ** (0.5 if m >= 5 else 1) / np.arange(1, m+1)
                              for m in num_mentions_by_pair]
        correct_by_pair = [(rn.rand(len(p)) < p) for p in precisions_by_pair]
        precision_at_k = np.mean([sum(c[:k])/len(c[:k]) for c in correct_by_pair
                                  if len(c) >= k])
        pair_p = np.array([imbalance if m >= 5 else 1-imbalance for m in num_mentions_by_pair])
        max_index_unnormalized_p = np.array([2**(-i) for i in range(1000)])
        max_index_p = [max_index_unnormalized_p[:m] / max_index_unnormalized_p[:m].sum()
                       for m in num_mentions_by_pair]
        selected_pair_idxs = rn.choice(num_pairs, size=pair_labeling_budget, p=pair_p / sum(pair_p), replace=True)
        selected_max_mention_by_pair = [1 + rn.choice(num_mentions_by_pair[idx], p=max_index_p[idx])
                                        for idx in selected_pair_idxs]
        selected_correct_by_pair = [correct_by_pair[idx][:m]
                                    for idx, m in zip(selected_pair_idxs, selected_max_mention_by_pair)]
        empirical_precision_at_k = np.mean([sum(c[:k]) / len(c[:k]) for c in selected_correct_by_pair
                                            if len(c) >= k])
        selected_pair_weights = [1/len(pair_p) / (pair_p[idx] / sum(pair_p)) for idx in selected_pair_idxs]
        w_precision_at_k = np.mean([w * sum(c[:k]) / len(c[:k])
                                    for w, c in zip(selected_pair_weights, selected_correct_by_pair)
                                    if len(c) >= k])
        a.append(empirical_precision_at_k)
        b.append(w_precision_at_k)
        c.append(precision_at_k)
    print(np.mean(a), np.mean(b), np.mean(c))
    plt.hist(a, alpha=0.4, label='empirical', bins=30)
    plt.hist(b, alpha=0.4, label='weighted', bins=30)
    plt.hist(c, alpha=0.4, label='complete data', bins=30)
    plt.legend()
    plt.show()


if __name__ == "__main__": main()
