### Create HI lines

import numpy as np

def HI_emission_line(arr_V, line_flux, psi_0, psi_max, w_20, w_50, w_peak):
    '''
    '''
    k1 = 1/4 * (np.log(0.5) * (w_peak**2 - w_20**2) + np.log(0.2) * (w_50**2 - w_peak**2)) / (np.log(0.5) * (w_peak - w_20) + np.log(0.2) * (w_50 - w_peak))
    k2 = (0.25 * (w_peak**2 - w_50**2) + k1 * (w_50 - w_peak)) / np.log(0.5)
    k3 = psi_max * np.exp((2*k1 - w_peak)**2 / (4*k2))
    k4 = 1/4 * (w_peak**2 * psi_max**2) / (psi_max**2 - psi_0**2)
    k5 = psi_0 * np.sqrt(k4)

    arr_line = np.zeros_like(arr_V)
    mask_out = abs(arr_V) >= w_peak/2
    value_out = k3 * np.exp(-(abs(arr_V) - k1)**2 / k2)
    value_in = k5 / np.sqrt(k4 - arr_V**2)
    arr_line = np.where(mask_out, value_out, value_in)
    arr_line[abs(arr_V) >= k1] = min(arr_line) # flattens the end of the wings

    pure_line = line_flux*arr_line

    spectral_shape = 0.01*np.max(pure_line) + (0.00001*np.max(pure_line))*arr_V
    
    return pure_line + spectral_shape

# SKADS
# Continuum