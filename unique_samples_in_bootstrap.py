import numpy as np
import numpy.linalg as la
import pandas as pd
import numpy.random as rn
import matplotlib.pyplot as plt
from sympy.functions.combinatorial.numbers import stirling
from sympy import ff

def p(n, a, k):
    return ff(n, k) / n**a * stirling(a, k)

def main():
    n_sim = 1000
    mark_size = 100
    sample_size = 100
    num_pokemon = 300
    mark_info = np.zeros((n_sim, num_pokemon))
    mark_info[:, :mark_size] = 1
    sample = rn.multinomial(sample_size,
                            np.ones(num_pokemon) / num_pokemon,
                            size=n_sim)
    mark_info * (sample > 0)
    selections = []
    for sim in sample:
        s = (sim > 0).sum()
        _p, i = max((p(i, batch_size, s), i)
                    for i in range(s, 30))
        selections.append(i)



if __name__ == "__main__": main()
