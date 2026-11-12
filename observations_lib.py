### Observational systematic errors

import numpy as np
np.random.seed(123)


def add_calibration_drift(arr_V, V_shift): # shift in km/s
    '''
    '''
    return arr_V - V_shift


def add_bandpass_gain_variations(arr_V, arr_line, gain_percentage, tilt_percentage):
    '''
    '''
    gain = 1.0 + gain_percentage/100
    arr_V_norm = (arr_V - np.min(abs(arr_V))) / ((np.max(arr_V) - np.min(arr_V))/2) # normalize velocity (-1 to 1)
    tilt = 1.0 + tilt_percentage/100 * arr_V_norm 
    return arr_line * gain * tilt


def add_baseline_ripples(arr_V, arr_line, amp_ripple, lmbd_ripple, phase_ripple):
    '''
    '''
    ripple = amp_ripple * np.sin(2*np.pi/lmbd_ripple * arr_V + phase_ripple)
    return arr_line + ripple


def add_flags(arr_line, N_flags):
    '''
    '''
    idx_flags = np.random.randint(0, len(arr_line), size=N_flags)
    arr_line[idx_flags] = np.nan
    return arr_line


def add_noise(arr_line, rms_noise):
    '''
    '''
    noise = np.random.normal(0.0, rms_noise, size=len(arr_line))
    return arr_line + noise


###################### SKA noise ######################

def SEFD(D_dish):
    '''
    Returns the SEFD of a dish [W m⁻² Hz⁻¹]
    
    D_dish = diameter of the dish [m]
    '''

    kB       = 1.380649e-23 # J/K
    T_sys    = 23
    eta_dish = 1.
    return 2*kB*T_sys / (eta_dish*np.pi*D_dish**2/4)

def S_rms_SKA(t_obs, N_ant, Dnu):
    '''
    Returns the RMS noise of the array [𝜇Jy]

    t_obs = observation time (per pointing) [s]
    N_ant = total number of antennas
    Dnu   = channel width [Hz]
    '''

    eta_sys = 1.
    n_13_5 = 64
    n_15 = N_ant - 64
    SEFD_arr = (n_13_5*(n_13_5-1)/SEFD(13.5)**2 + 2*n_13_5*n_15/(SEFD(13.5)*SEFD(15.)) + n_15*(n_15-1)/SEFD(15.)**2)**-.5
    return SEFD_arr / (eta_sys*np.sqrt(2*Dnu*t_obs)) * 1e26 # W m⁻² Hz⁻¹ to Jy