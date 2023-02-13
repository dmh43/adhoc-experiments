import numpy as np
import numpy.linalg as la
import pandas as pd
import numpy.random as rn
import matplotlib.pyplot as plt
import matplotlib

def calc_coverage(var, est, val):
    ci_l = est - 1.96 * np.sqrt(var)
    ci_r = est + 1.96 * np.sqrt(var)
    coverage = ci_l < val
    coverage = coverage & (ci_r > val)
    return coverage.mean()

def main():
    num_sim = 1000
    max_num_visits = 10000
    b_ctr = 0.0048
    v_ctr = 0.005
    b_coverages = []
    d_coverages = []
    e_coverages = []
    v = np.geomspace(100, max_num_visits, dtype=int)
    for num_visits in v:
        b_clicks = rn.binomial(num_visits, b_ctr, num_sim)
        v_clicks = rn.binomial(num_visits, v_ctr, num_sim)
        b_est = b_clicks / num_visits
        v_est = v_clicks / num_visits
        b_asymp_var = b_est * (1 - b_est) / num_visits
        v_asymp_var = v_est * (1 - v_est) / num_visits
        frac_est = (v_est - b_est) / b_est
        frac_d_var = 1/b_est**2 * v_asymp_var + v_est ** 2 / b_est**4 * b_asymp_var
        frac_e_var = (v_asymp_var + b_asymp_var) / b_est**2
        boot_b_est = rn.binomial(num_visits, b_est, (num_visits, num_sim)).T
        boot_v_est = rn.binomial(num_visits, v_est, (num_visits, num_sim)).T
        boot_frac_est = (boot_v_est - boot_b_est) / boot_b_est
        frac_b_var = boot_frac_est.var(1)
        b_coverages.append(calc_coverage(frac_b_var,
                                         frac_est,
                                         (v_ctr - b_ctr) / b_ctr))
        e_coverages.append(calc_coverage(frac_e_var,
                                         frac_est,
                                         (v_ctr - b_ctr) / b_ctr))
        d_coverages.append(calc_coverage(frac_d_var,
                                         frac_est,
                                         (v_ctr - b_ctr) / b_ctr))

    fig, ax = plt.subplots()
    ax.plot(v, b_coverages, label='bootstrap')
    ax.plot(v, e_coverages, label='ratio')
    ax.plot(v, d_coverages, label='delta_method')
    ax.set_xscale('log')
    ax.legend()

if __name__ == "__main__": main()
