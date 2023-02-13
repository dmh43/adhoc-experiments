import numpy as np
import numpy.linalg as la
import pandas as pd
import numpy.random as rn
import matplotlib.pyplot as plt
import pywt

from rpca import R_pca
from wavelet import coef_pyramid_plot

def main():
    num_users = 100
    # num_time_points = 100
    num_time_points = 2**11
    n_click_params = rn.uniform(0, 3, size=(num_time_points, num_users))
    novelty_mask = np.geomspace(1, 1e-4, num_time_points)
    novelty_views = rn.poisson(n_click_params * novelty_mask[:, np.newaxis])
    background_pageviews = rn.poisson(1, size=(num_time_points, num_users))
    obs = background_pageviews + novelty_views

    # plt.scatter(np.arange(num_time_points), obs[:, 0])
    # plt.hlines(np.mean(background_pageviews[:, 0]), 0, num_time_points, colors='r')
    # plt.plot(np.arange(num_time_points), np.convolve(obs[:, 0], np.ones(100), 'same')/100)

    bcoefs = pywt.wavedec(background_pageviews.sum(1), 'haar', level=11, mode='per')
    ocoefs = pywt.wavedec(obs.sum(1), 'haar', level=11, mode='per')

    # sigma = np.std(ocoefs[-1])
    sigma = np.median(np.abs(ocoefs[-1] - np.median(ocoefs[-1])))/0.6745
    uthresh = sigma*np.sqrt(2*np.log(num_time_points))

    denoised = ocoefs[:]

    denoised[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in denoised[1:])
    signal = pywt.waverec(denoised, 'haar', mode='per')

    plt.plot(obs.sum(1), label='observed')
    # plt.plot(background_pageviews.sum(1), label='background_pageviews')
    # plt.plot(np.convolve(obs.sum(1), np.ones(10), 'save')/10, label='10ma')
    # plt.plot(np.convolve(obs.sum(1), np.ones(1000), 'save')/1000, label='1kma')
    plt.plot(signal, label='result')
    plt.plot(1 * num_users + novelty_mask, linestyle='--', label='mean')
    plt.legend()

    coef_pyramid_plot(denoised[1:])
    coef_pyramid_plot(bcoefs[1:])
    coef_pyramid_plot(ocoefs[1:])


    rpca = R_pca(obs.T)
    L, S = [m.T for m in rpca.fit(max_iter=10000, tol=1e-7)]

    print(((L - background_pageviews)**2).mean())
    print(((obs - background_pageviews)**2).mean())
    print(((S - novelty_views)**2).mean())

if __name__ == "__main__": main()
