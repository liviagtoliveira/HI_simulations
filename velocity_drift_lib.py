### Methods of calculating the velocity drift

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
from scipy.optimize import curve_fit


def Gaussian(x, A, x0, sigma, C):
    '''
    '''
    return A * np.exp(-(x-x0)**2/sigma**2) + C


def cross_correlation(arr_V, stacked1, stacked2):
    '''
    '''
    dv = arr_V[1] - arr_V[0]
    ccf = correlate(stacked2, stacked1)
    lags = correlation_lags(len(stacked2), len(stacked1))

    p0 = [np.max(ccf), lags[np.argmax(ccf)]*dv, 10*dv, 0]
    popt, pcov = curve_fit(Gaussian, lags*dv, ccf, p0=p0) # fits Gaussian to CCF
    A_fit, x0_fit, sigma_fit, C_fit = popt # fitted parameters
    A_err, x0_err, sigma_err, C_err = np.sqrt(np.diag(pcov)) # uncertainties of the parameters
    offset, sigma_offset = x0_fit, x0_err
    
    plt.plot(lags*dv, ccf, label='CCF')
    plt.plot(lags*dv, Gaussian(lags*dv, A_fit, x0_fit, sigma_fit, C_fit), color='black', label='Gaussian fit')
    plt.axvline(offset, color='black', linestyle=':', label=f'Best offset = {offset:.2f} km/s')
    plt.legend()
    plt.xlabel('Velocity offset [km/s]')
    plt.ylabel('CCF')
    plt.title('Cross-Correlation Function')
    plt.show()
    
    return offset, sigma_offset


def direct_fitting(arr_V, stacked1, stacked2, plot_distributions=False):
    p0_1 = [np.max(stacked1), arr_V[np.argmax(stacked1)], 60.0, np.median(stacked1)]
    popt1, pcov1 = curve_fit(Gaussian, arr_V, stacked1, p0=p0_1) # fits Gaussian to stacked spectrum of the 1st epoch
    A1, x01, sigma1, C1 = popt1 # fitted parameters
    A1_err, x01_err, sigma1_err, C1_err = np.sqrt(np.diag(pcov1)) # uncertainties of the parameters

    #p0_2 = [A1, x01+100, sigma1, C1]
    popt2, pcov2 = curve_fit(Gaussian, arr_V, stacked2, p0=popt1) # fits Gaussian to stacked spectrum of the 2nd epoch
    A2, x02, sigma2, C2 = popt2 # fitted parameters
    A2_err, x02_err, sigma2_err, C2_err = np.sqrt(np.diag(pcov2)) # uncertainties of the parameters

    offset = x02 - x01
    sigma_offset = np.sqrt(x01_err**2 + x02_err**2)

    plt.plot(arr_V, stacked1, linewidth=0.5, label='1st survey')
    plt.plot(arr_V, stacked2, linewidth=0.5, label='2nd survey')
    plt.plot(arr_V, Gaussian(arr_V, A1, x01, sigma1, C1), color='tab:blue', label='Gaussian fit for the 1st survey')
    plt.plot(arr_V, Gaussian(arr_V, A2, x02, sigma2, C2), color='tab:orange', label='Gaussian fit for the 2nd survey')
    plt.axvline(x01, color='tab:blue', linestyle=':', label=f'1st survey center = {x01:.1f} km/s')
    plt.axvline(x02, color='tab:orange', linestyle=':', label=f'2nd survey center = {x02:.1f} km/s')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Velocity [km/s]')
    plt.ylabel('Flux [Jy]')
    plt.title('Direct fitting method')
    plt.tight_layout()
    plt.show()

    return offset, sigma_offset


def difference(arr_V, stacked1, stacked2):
    p_cygni = stacked2 - stacked1
    
    plt.plot(arr_V, p_cygni)
    plt.xlabel('Velocity [km/s]')
    plt.ylabel('Flux [Jy]')
    plt.title('Difference method')
    plt.show()