import numpy as np
import numpy.linalg as la
import pandas as pd
import numpy.random as rn
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

def calc_bootstrap_ci(samples):
  res = [bootstrap((s,), np.mean, confidence_level=0.95, method='basic', n_resamples=1000).confidence_interval
         for s in samples]
  return np.array([a.low for a in res]), np.array([a.high for a in res])

def main():
  n = []
  b = []
  n_sim = 1000
  for n_samples in np.geomspace(10, 100):
    n_samples = int(n_samples)
    samples = rn.normal(size=(n_sim, n_samples))
    normal_ci_coverage = ((samples.mean(1) - 1.96*samples.std(1) / np.sqrt(n_samples) < 0) * (0 < samples.mean(1) + 1.96*samples.std(1) / np.sqrt(n_samples))).mean()
    bootstrap_ci_l, bootstrap_ci_r = calc_bootstrap_ci(samples)
    bootstrap_ci_coverage = ((bootstrap_ci_l < 0) * (0 < bootstrap_ci_r)).mean()
    n.append(normal_ci_coverage)
    b.append(bootstrap_ci_coverage)

  m = []
  x = []
  n_sim = 1000
  for n_samples in np.geomspace(100, 10000, 10):
    samples = rn.normal(size=(n_sim, int(n_samples)))
    res = [bootstrap((s,), np.median, confidence_level=0.95, method='basic', n_resamples=1000).confidence_interval
           for s in samples]
    l, r = np.array([a.low for a in res]), np.array([a.high for a in res])
    m_coverage = ((l < 0) * (0 < r)).mean()
    m.append(m_coverage)
    x.append(n_samples)

if __name__ == "__main__": main()
