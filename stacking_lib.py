### Stacking spectra

import numpy as np
import matplotlib.pyplot as plt

def estimate_noise(spectrum):
    '''
    '''
    edge_width = int(0.15 * len(spectrum))
    edges = np.concatenate((spectrum[:edge_width], spectrum[-edge_width:]))
    return np.nanstd(edges)

def mean_stack(galaxy_catalog, z_arr, bins, survey1, survey2):
    '''
    '''
    bin_indices = np.digitize(z_arr, bins)
    z_bin, N_bin = np.unique(sorted(bin_indices), return_counts=True) # gets number of spectra in each redshift bin
    stacked_spectra1 = np.zeros((len(bins)-1, len(survey1[0])))
    stacked_spectra2 = np.zeros((len(bins)-1, len(survey2[0])))
    for gal_idx in range(len(galaxy_catalog)):
        bin_idx = bin_indices[gal_idx] - 1
        stacked_spectra1[bin_idx] += np.nan_to_num(survey1[gal_idx], nan=0.)
        stacked_spectra2[bin_idx] += np.nan_to_num(survey2[gal_idx], nan=0.)
    return stacked_spectra1/N_bin.reshape(-1,1), stacked_spectra2/N_bin.reshape(-1,1)

def weighted_mean_stack(galaxy_catalog, z_arr, bins, survey1, survey2):
    '''
    '''
    bin_indices = np.digitize(z_arr, bins)
    z_bin, N_bin = np.unique(sorted(bin_indices), return_counts=True) # gets number of spectra in each redshift bin
    stacked_spectra1 = np.zeros((len(bins)-1, len(survey1[0])))
    stacked_spectra2 = np.zeros((len(bins)-1, len(survey2[0])))
    sum_weights1 = np.zeros(len(bins)-1)
    sum_weights2 = np.zeros(len(bins)-1)
    for gal_idx in range(len(galaxy_catalog)):
        bin_idx = bin_indices[gal_idx] - 1
        w1_gal = (estimate_noise(survey1[gal_idx]))**(-2)
        w2_gal = (estimate_noise(survey2[gal_idx]))**(-2)
        stacked_spectra1[bin_idx] += np.nan_to_num(survey1[gal_idx], nan=0.)*w1_gal
        stacked_spectra2[bin_idx] += np.nan_to_num(survey2[gal_idx], nan=0.)*w2_gal
        sum_weights1[bin_idx] += w1_gal
        sum_weights2[bin_idx] += w2_gal    
    return stacked_spectra1/sum_weights1.reshape(-1,1), stacked_spectra2/sum_weights2.reshape(-1,1)