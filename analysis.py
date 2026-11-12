### Create surveys and get velocity drift

import numpy as np
import matplotlib.pyplot as plt
import HI_line_lib as HIlib
import observations_lib as obslib
import stacking_lib as stacklib
import velocity_drift_lib as driftlib
np.random.seed(123)

test_galaxy_catalog = [
    # Redshift Bin 0.0 - 0.1
    {"id": "Gal_01_01", "z": 0.012, "s_int": 3.850, "central_dens": 0.62, "peak_dens": 1.0, "w20": 380, "w50": 310, "w_peak": 210},
    {"id": "Gal_01_02", "z": 0.015, "s_int": 3.400, "central_dens": 0.70, "peak_dens": 1.0, "w20": 280, "w50": 230, "w_peak": 150},
    {"id": "Gal_01_03", "z": 0.018, "s_int": 3.100, "central_dens": 0.58, "peak_dens": 1.0, "w20": 420, "w50": 340, "w_peak": 230},
    {"id": "Gal_01_04", "z": 0.021, "s_int": 2.900, "central_dens": 0.82, "peak_dens": 1.0, "w20": 190, "w50": 150, "w_peak": 95},
    {"id": "Gal_01_05", "z": 0.024, "s_int": 2.650, "central_dens": 0.65, "peak_dens": 1.0, "w20": 320, "w50": 260, "w_peak": 175},
    {"id": "Gal_01_06", "z": 0.026, "s_int": 2.450, "central_dens": 0.75, "peak_dens": 1.0, "w20": 240, "w50": 195, "w_peak": 130},
    {"id": "Gal_01_07", "z": 0.029, "s_int": 2.200, "central_dens": 0.60, "peak_dens": 1.0, "w20": 390, "w50": 320, "w_peak": 215},
    {"id": "Gal_01_08", "z": 0.031, "s_int": 2.050, "central_dens": 0.85, "peak_dens": 1.0, "w20": 160, "w50": 130, "w_peak": 80},
    {"id": "Gal_01_09", "z": 0.034, "s_int": 1.850, "central_dens": 0.68, "peak_dens": 1.0, "w20": 300, "w50": 245, "w_peak": 165},
    {"id": "Gal_01_10", "z": 0.036, "s_int": 1.700, "central_dens": 0.72, "peak_dens": 1.0, "w20": 260, "w50": 210, "w_peak": 140},
    {"id": "Gal_01_11", "z": 0.039, "s_int": 1.550, "central_dens": 0.55, "peak_dens": 1.0, "w20": 410, "w50": 335, "w_peak": 225},
    {"id": "Gal_01_12", "z": 0.041, "s_int": 1.420, "central_dens": 0.78, "peak_dens": 1.0, "w20": 210, "w50": 170, "w_peak": 110},
    {"id": "Gal_01_13", "z": 0.043, "s_int": 1.300, "central_dens": 0.64, "peak_dens": 1.0, "w20": 340, "w50": 275, "w_peak": 185},
    {"id": "Gal_01_14", "z": 0.045, "s_int": 1.180, "central_dens": 0.80, "peak_dens": 1.0, "w20": 180, "w50": 145, "w_peak": 90},
    {"id": "Gal_01_15", "z": 0.048, "s_int": 1.050, "central_dens": 0.66, "peak_dens": 1.0, "w20": 310, "w50": 250, "w_peak": 170},
    {"id": "Gal_01_16", "z": 0.050, "s_int": 0.980, "central_dens": 0.73, "peak_dens": 1.0, "w20": 250, "w50": 200, "w_peak": 135},
    {"id": "Gal_01_17", "z": 0.052, "s_int": 0.910, "central_dens": 0.59, "peak_dens": 1.0, "w20": 370, "w50": 300, "w_peak": 200},
    {"id": "Gal_01_18", "z": 0.055, "s_int": 0.840, "central_dens": 0.83, "peak_dens": 1.0, "w20": 170, "w50": 135, "w_peak": 85},
    {"id": "Gal_01_19", "z": 0.057, "s_int": 0.780, "central_dens": 0.67, "peak_dens": 1.0, "w20": 290, "w50": 235, "w_peak": 160},
    {"id": "Gal_01_20", "z": 0.059, "s_int": 0.720, "central_dens": 0.75, "peak_dens": 1.0, "w20": 230, "w50": 185, "w_peak": 125},
    {"id": "Gal_01_21", "z": 0.061, "s_int": 0.670, "central_dens": 0.61, "peak_dens": 1.0, "w20": 350, "w50": 285, "w_peak": 190},
    {"id": "Gal_01_22", "z": 0.063, "s_int": 0.620, "central_dens": 0.81, "peak_dens": 1.0, "w20": 195, "w50": 155, "w_peak": 100},
    {"id": "Gal_01_23", "z": 0.066, "s_int": 0.570, "central_dens": 0.69, "peak_dens": 1.0, "w20": 275, "w50": 220, "w_peak": 150},
    {"id": "Gal_01_24", "z": 0.068, "s_int": 0.530, "central_dens": 0.76, "peak_dens": 1.0, "w20": 220, "w50": 175, "w_peak": 115},
    {"id": "Gal_01_25", "z": 0.070, "s_int": 0.490, "central_dens": 0.63, "peak_dens": 1.0, "w20": 330, "w50": 265, "w_peak": 180},
    {"id": "Gal_01_26", "z": 0.072, "s_int": 0.455, "central_dens": 0.84, "peak_dens": 1.0, "w20": 155, "w50": 125, "w_peak": 75},
    {"id": "Gal_01_27", "z": 0.074, "s_int": 0.420, "central_dens": 0.70, "peak_dens": 1.0, "w20": 285, "w50": 230, "w_peak": 155},
    {"id": "Gal_01_28", "z": 0.076, "s_int": 0.390, "central_dens": 0.74, "peak_dens": 1.0, "w20": 240, "w50": 190, "w_peak": 130},
    {"id": "Gal_01_29", "z": 0.078, "s_int": 0.360, "central_dens": 0.57, "peak_dens": 1.0, "w20": 395, "w50": 320, "w_peak": 215},
    {"id": "Gal_01_30", "z": 0.080, "s_int": 0.335, "central_dens": 0.79, "peak_dens": 1.0, "w20": 200, "w50": 160, "w_peak": 105},
    {"id": "Gal_01_31", "z": 0.082, "s_int": 0.310, "central_dens": 0.65, "peak_dens": 1.0, "w20": 315, "w50": 255, "w_peak": 170},
    {"id": "Gal_01_32", "z": 0.084, "s_int": 0.285, "central_dens": 0.82, "peak_dens": 1.0, "w20": 165, "w50": 135, "w_peak": 85},
    {"id": "Gal_01_33", "z": 0.085, "s_int": 0.265, "central_dens": 0.68, "peak_dens": 1.0, "w20": 295, "w50": 235, "w_peak": 160},
    {"id": "Gal_01_34", "z": 0.087, "s_int": 0.245, "central_dens": 0.72, "peak_dens": 1.0, "w20": 255, "w50": 205, "w_peak": 140},
    {"id": "Gal_01_35", "z": 0.088, "s_int": 0.225, "central_dens": 0.60, "peak_dens": 1.0, "w20": 360, "w50": 290, "w_peak": 195},
    {"id": "Gal_01_36", "z": 0.090, "s_int": 0.210, "central_dens": 0.80, "peak_dens": 1.0, "w20": 185, "w50": 150, "w_peak": 95},
    {"id": "Gal_01_37", "z": 0.091, "s_int": 0.195, "central_dens": 0.66, "peak_dens": 1.0, "w20": 325, "w50": 260, "w_peak": 175},
    {"id": "Gal_01_38", "z": 0.093, "s_int": 0.180, "central_dens": 0.75, "peak_dens": 1.0, "w20": 235, "w50": 190, "w_peak": 125},
    {"id": "Gal_01_39", "z": 0.094, "s_int": 0.165, "central_dens": 0.62, "peak_dens": 1.0, "w20": 345, "w50": 280, "w_peak": 190},
    {"id": "Gal_01_40", "z": 0.095, "s_int": 0.155, "central_dens": 0.83, "peak_dens": 1.0, "w20": 175, "w50": 140, "w_peak": 90},
    {"id": "Gal_01_41", "z": 0.096, "s_int": 0.145, "central_dens": 0.70, "peak_dens": 1.0, "w20": 270, "w50": 215, "w_peak": 145},
    {"id": "Gal_01_42", "z": 0.097, "s_int": 0.135, "central_dens": 0.77, "peak_dens": 1.0, "w20": 215, "w50": 170, "w_peak": 115},
    {"id": "Gal_01_43", "z": 0.097, "s_int": 0.125, "central_dens": 0.58, "peak_dens": 1.0, "w20": 405, "w50": 330, "w_peak": 220},
    {"id": "Gal_01_44", "z": 0.098, "s_int": 0.118, "central_dens": 0.81, "peak_dens": 1.0, "w20": 190, "w50": 150, "w_peak": 100},
    {"id": "Gal_01_45", "z": 0.098, "s_int": 0.110, "central_dens": 0.64, "peak_dens": 1.0, "w20": 305, "w50": 245, "w_peak": 165},
    {"id": "Gal_01_46", "z": 0.099, "s_int": 0.105, "central_dens": 0.73, "peak_dens": 1.0, "w20": 245, "w50": 195, "w_peak": 130},
    {"id": "Gal_01_47", "z": 0.099, "s_int": 0.100, "central_dens": 0.67, "peak_dens": 1.0, "w20": 280, "w50": 225, "w_peak": 150},
    {"id": "Gal_01_48", "z": 0.099, "s_int": 0.095, "central_dens": 0.76, "peak_dens": 1.0, "w20": 205, "w50": 165, "w_peak": 110},
    {"id": "Gal_01_49", "z": 0.011, "s_int": 0.090, "central_dens": 0.61, "peak_dens": 1.0, "w20": 335, "w50": 270, "w_peak": 180},
    {"id": "Gal_01_50", "z": 0.013, "s_int": 0.085, "central_dens": 0.85, "peak_dens": 1.0, "w20": 150, "w50": 120, "w_peak": 75},
    {"id": "Gal_01_51", "z": 0.014, "s_int": 3.700, "central_dens": 0.63, "peak_dens": 1.0, "w20": 360, "w50": 295, "w_peak": 200},
    {"id": "Gal_01_52", "z": 0.016, "s_int": 3.250, "central_dens": 0.72, "peak_dens": 1.0, "w20": 270, "w50": 220, "w_peak": 145},
    {"id": "Gal_01_53", "z": 0.019, "s_int": 2.950, "central_dens": 0.59, "peak_dens": 1.0, "w20": 410, "w50": 330, "w_peak": 225},
    {"id": "Gal_01_54", "z": 0.022, "s_int": 2.750, "central_dens": 0.81, "peak_dens": 1.0, "w20": 185, "w50": 145, "w_peak": 92},
    {"id": "Gal_01_55", "z": 0.025, "s_int": 2.500, "central_dens": 0.66, "peak_dens": 1.0, "w20": 315, "w50": 255, "w_peak": 170},
    {"id": "Gal_01_56", "z": 0.027, "s_int": 2.300, "central_dens": 0.74, "peak_dens": 1.0, "w20": 235, "w50": 190, "w_peak": 125},
    {"id": "Gal_01_57", "z": 0.030, "s_int": 2.100, "central_dens": 0.61, "peak_dens": 1.0, "w20": 385, "w50": 315, "w_peak": 210},
    {"id": "Gal_01_58", "z": 0.032, "s_int": 1.950, "central_dens": 0.84, "peak_dens": 1.0, "w20": 165, "w50": 135, "w_peak": 85},
    {"id": "Gal_01_59", "z": 0.035, "s_int": 1.750, "central_dens": 0.69, "peak_dens": 1.0, "w20": 295, "w50": 240, "w_peak": 160},
    {"id": "Gal_01_60", "z": 0.037, "s_int": 1.600, "central_dens": 0.71, "peak_dens": 1.0, "w20": 255, "w50": 205, "w_peak": 135},
    {"id": "Gal_01_61", "z": 0.040, "s_int": 1.450, "central_dens": 0.56, "peak_dens": 1.0, "w20": 405, "w50": 330, "w_peak": 220},
    {"id": "Gal_01_62", "z": 0.042, "s_int": 1.350, "central_dens": 0.77, "peak_dens": 1.0, "w20": 215, "w50": 175, "w_peak": 115},
    {"id": "Gal_01_63", "z": 0.044, "s_int": 1.250, "central_dens": 0.65, "peak_dens": 1.0, "w20": 335, "w50": 270, "w_peak": 180},
    {"id": "Gal_01_64", "z": 0.046, "s_int": 1.120, "central_dens": 0.79, "peak_dens": 1.0, "w20": 175, "w50": 140, "w_peak": 88},
    {"id": "Gal_01_65", "z": 0.049, "s_int": 1.000, "central_dens": 0.67, "peak_dens": 1.0, "w20": 305, "w50": 245, "w_peak": 165},
    {"id": "Gal_01_66", "z": 0.051, "s_int": 0.940, "central_dens": 0.72, "peak_dens": 1.0, "w20": 245, "w50": 195, "w_peak": 130},
    {"id": "Gal_01_67", "z": 0.053, "s_int": 0.880, "central_dens": 0.58, "peak_dens": 1.0, "w20": 375, "w50": 305, "w_peak": 205},
    {"id": "Gal_01_68", "z": 0.056, "s_int": 0.810, "central_dens": 0.82, "peak_dens": 1.0, "w20": 165, "w50": 130, "w_peak": 82},
    {"id": "Gal_01_69", "z": 0.058, "s_int": 0.750, "central_dens": 0.68, "peak_dens": 1.0, "w20": 285, "w50": 230, "w_peak": 155},
    {"id": "Gal_01_70", "z": 0.060, "s_int": 0.690, "central_dens": 0.76, "peak_dens": 1.0, "w20": 225, "w50": 180, "w_peak": 120},
    {"id": "Gal_01_71", "z": 0.062, "s_int": 0.640, "central_dens": 0.62, "peak_dens": 1.0, "w20": 345, "w50": 280, "w_peak": 185},
    {"id": "Gal_01_72", "z": 0.064, "s_int": 0.590, "central_dens": 0.80, "peak_dens": 1.0, "w20": 190, "w50": 150, "w_peak": 98},
    {"id": "Gal_01_73", "z": 0.067, "s_int": 0.540, "central_dens": 0.70, "peak_dens": 1.0, "w20": 265, "w50": 215, "w_peak": 145},
    {"id": "Gal_01_74", "z": 0.069, "s_int": 0.500, "central_dens": 0.75, "peak_dens": 1.0, "w20": 210, "w50": 170, "w_peak": 110},
    {"id": "Gal_01_75", "z": 0.071, "s_int": 0.460, "central_dens": 0.64, "peak_dens": 1.0, "w20": 320, "w50": 260, "w_peak": 175},
    {"id": "Gal_01_76", "z": 0.073, "s_int": 0.430, "central_dens": 0.83, "peak_dens": 1.0, "w20": 150, "w50": 120, "w_peak": 72},
    {"id": "Gal_01_77", "z": 0.075, "s_int": 0.400, "central_dens": 0.71, "peak_dens": 1.0, "w20": 275, "w50": 220, "w_peak": 150},
    {"id": "Gal_01_78", "z": 0.077, "s_int": 0.370, "central_dens": 0.73, "peak_dens": 1.0, "w20": 235, "w50": 185, "w_peak": 125},
    {"id": "Gal_01_79", "z": 0.079, "s_int": 0.345, "central_dens": 0.59, "peak_dens": 1.0, "w20": 390, "w50": 315, "w_peak": 210},
    {"id": "Gal_01_80", "z": 0.081, "s_int": 0.320, "central_dens": 0.78, "peak_dens": 1.0, "w20": 195, "w50": 155, "w_peak": 102},
    {"id": "Gal_01_81", "z": 0.083, "s_int": 0.295, "central_dens": 0.66, "peak_dens": 1.0, "w20": 310, "w50": 250, "w_peak": 168},
    {"id": "Gal_01_82", "z": 0.085, "s_int": 0.270, "central_dens": 0.81, "peak_dens": 1.0, "w20": 160, "w50": 130, "w_peak": 82},
    {"id": "Gal_01_83", "z": 0.086, "s_int": 0.250, "central_dens": 0.69, "peak_dens": 1.0, "w20": 290, "w50": 230, "w_peak": 158},
    {"id": "Gal_01_84", "z": 0.088, "s_int": 0.230, "central_dens": 0.73, "peak_dens": 1.0, "w20": 250, "w50": 200, "w_peak": 135},
    {"id": "Gal_01_85", "z": 0.089, "s_int": 0.215, "central_dens": 0.61, "peak_dens": 1.0, "w20": 355, "w50": 285, "w_peak": 192},
    {"id": "Gal_01_86", "z": 0.091, "s_int": 0.200, "central_dens": 0.79, "peak_dens": 1.0, "w20": 180, "w50": 145, "w_peak": 92},
    {"id": "Gal_01_87", "z": 0.092, "s_int": 0.185, "central_dens": 0.67, "peak_dens": 1.0, "w20": 320, "w50": 255, "w_peak": 172},
    {"id": "Gal_01_88", "z": 0.093, "s_int": 0.172, "central_dens": 0.74, "peak_dens": 1.0, "w20": 230, "w50": 185, "w_peak": 122},
    {"id": "Gal_01_89", "z": 0.094, "s_int": 0.160, "central_dens": 0.63, "peak_dens": 1.0, "w20": 340, "w50": 275, "w_peak": 185},
    {"id": "Gal_01_90", "z": 0.095, "s_int": 0.150, "central_dens": 0.82, "peak_dens": 1.0, "w20": 170, "w50": 135, "w_peak": 88},
    {"id": "Gal_01_91", "z": 0.096, "s_int": 0.140, "central_dens": 0.71, "peak_dens": 1.0, "w20": 265, "w50": 210, "w_peak": 142},
    {"id": "Gal_01_92", "z": 0.097, "s_int": 0.130, "central_dens": 0.76, "peak_dens": 1.0, "w20": 210, "w50": 165, "w_peak": 110},
    {"id": "Gal_01_93", "z": 0.097, "s_int": 0.120, "central_dens": 0.59, "peak_dens": 1.0, "w20": 400, "w50": 325, "w_peak": 215},
    {"id": "Gal_01_94", "z": 0.098, "s_int": 0.114, "central_dens": 0.80, "peak_dens": 1.0, "w20": 185, "w50": 145, "w_peak": 96},
    {"id": "Gal_01_95", "z": 0.098, "s_int": 0.108, "central_dens": 0.65, "peak_dens": 1.0, "w20": 300, "w50": 240, "w_peak": 162},
    {"id": "Gal_01_96", "z": 0.099, "s_int": 0.102, "central_dens": 0.72, "peak_dens": 1.0, "w20": 240, "w50": 190, "w_peak": 128},
    {"id": "Gal_01_97", "z": 0.099, "s_int": 0.097, "central_dens": 0.68, "peak_dens": 1.0, "w20": 275, "w50": 220, "w_peak": 148},
    {"id": "Gal_01_98", "z": 0.099, "s_int": 0.092, "central_dens": 0.77, "peak_dens": 1.0, "w20": 200, "w50": 160, "w_peak": 105},
    {"id": "Gal_01_99", "z": 0.010, "s_int": 0.087, "central_dens": 0.60, "peak_dens": 1.0, "w20": 330, "w50": 265, "w_peak": 178},
    {"id": "Gal_01_100", "z": 0.015, "s_int": 0.082, "central_dens": 0.84, "peak_dens": 1.0, "w20": 145, "w50": 115, "w_peak": 72}
    ]


# Create 1st survey

def first_survey(arr_V, params, noise, sys_err=False):
    '''
    '''
    lines_1st = np.zeros((len(params), len(arr_V)))
    for i, parameters in enumerate(params):
        id_gal, z, line_flux, psi_0, psi_max, w_20, w_50, w_peak = parameters
        line = HIlib.HI_emission_line(arr_V, line_flux, psi_0, psi_max, w_20, w_50, w_peak)

        if sys_err:
            # baseline ripples
            amp_ripple = np.random.uniform(0.001, 0.01)*line_flux
            lmbd_ripple = np.random.uniform(50, 200)
            line = obslib.add_baseline_ripples(arr_V, line, amp_ripple, lmbd_ripple, 0)

            # interference
            N_flags = np.random.randint(0, 10)
            line = obslib.add_flags(line, N_flags)

        # noise
        line = obslib.add_noise(line, noise)
        lines_1st[i] = line

    return lines_1st


# Create 2nd survey

def second_survey(arr_V, params, noise, V_offset, sys_err=False):
    '''
    '''
    arr_V = arr_V - V_offset

    V_shift = np.random.uniform(0.01, 0.1)
    gain_percentage = np.random.uniform(0.05, 0.2)
    tilt_percentage = np.random.uniform(0.2, 0.8)
    
    lines_2nd = np.zeros((len(params), len(arr_V)))
    for i, parameters in enumerate(params):
        id_gal, z, line_flux, psi_0, psi_max, w_20, w_50, w_peak = parameters

        if sys_err:
            line = HIlib.HI_emission_line(obslib.add_calibration_drift(arr_V, V_shift), line_flux, psi_0, psi_max, w_20, w_50, w_peak) # line with shifted arr_V
            
            # bandpass and gain variations
            line = obslib.add_bandpass_gain_variations(arr_V, line, gain_percentage, tilt_percentage)

            # baseline ripples
            amp_ripple = np.random.uniform(0.001, 0.01)*line_flux
            lmbd_ripple = np.random.uniform(50, 200)
            phase_ripple = np.random.uniform(0, 2*np.pi)
            line = obslib.add_baseline_ripples(arr_V, line, amp_ripple, lmbd_ripple, phase_ripple)

            # interference
            N_flags = np.random.randint(0, 10)
            line = obslib.add_flags(line, N_flags)

        else:
            line = HIlib.HI_emission_line(arr_V, line_flux, psi_0, psi_max, w_20, w_50, w_peak)

        # noise
        line = obslib.add_noise(line, noise)
        lines_2nd[i] = line

    return lines_2nd


def compute_drift(noise, N=1000, offset=100, sys_err=False, weights=False, plot_stacked=False, analyse_noise=False, plot_RMSvsN=False):
    '''
    Prints the velocity shifts and their uncertainties computed with both the cross-correlation and with direct fitting
    Stacking is done with a weighted mean if weights=True, otherwise it is a normal mean
    Plots the stacked spectra for each redshift bin if plot_stacked=True
    Compares the estimated and theorical noise of the stacked spectra if analyse_noise=True

    noise = RMS noise of observations [Jy]
    N = number of velocity channels
    offset = velocity offset of the second epoch [km/s]
    '''

    V_val = np.linspace(-500, 500, N)

    test_galaxy_params = [tuple(d.values()) for d in test_galaxy_catalog]
    # noise = obslib.S_rms_SKA(100, 144, 0.001)
    survey1 = first_survey(V_val, test_galaxy_params, noise, sys_err)
    survey2 = second_survey(V_val, test_galaxy_params, noise, offset, sys_err)

    z_arr = np.array([d['z'] for d in test_galaxy_catalog])
    bins = np.arange(0, max(z_arr)+0.1, 0.1)
    if weights:
        stacked_spectra1, stacked_spectra2 = stacklib.weighted_mean_stack(test_galaxy_catalog, z_arr, bins, survey1, survey2)
    else:
        stacked_spectra1, stacked_spectra2 = stacklib.mean_stack(test_galaxy_catalog, z_arr, bins, survey1, survey2)

    if plot_stacked:
        for j in range(len(bins)-1):
            plt.plot(V_val, stacked_spectra1[j], label='1st survey')
            plt.plot(V_val, stacked_spectra2[j], label='2nd survey')
            plt.legend()
            plt.xlabel('Velocity [km/s]')
            plt.ylabel('Flux [Jy]')
            plt.title(f'Stacked spectra for z in [{bins[j]:.1f}, {bins[j+1]:.1f})')
            plt.show()

    if analyse_noise:
        stacked_noise = stacklib.estimate_noise(stacked_spectra1[0])
        print(f"Estimated noise in the stacked spectrum: {stacked_noise:.1f} Jy")
        print(f"Theorical noise in the stacked spectrum: {noise/np.sqrt(100):.1f} Jy")

    if plot_RMSvsN:
        bin_indices = np.digitize(z_arr, bins)
        z_bin, N_bin = np.unique(sorted(bin_indices), return_counts=True) # gets number of spectra in each redshift bin
        for i in range(len(bins)-1):        
            N_spectra = N_bin[i]
            stacked_noises = np.zeros(N_spectra)
            weighted_stacked_noises = np.zeros(N_spectra)
            for n in range(N_spectra):
                stacked_spectra = np.zeros(len(survey1[0]))
                weighted_stacked_spectra = np.zeros(len(survey1[0]))
                sum_weights = 0
                for gal_idx in range(len(test_galaxy_catalog[:n+1])):
                    w_gal = (stacklib.estimate_noise(survey1[gal_idx]))**(-2)
                    stacked_spectra += np.nan_to_num(survey1[gal_idx], nan=0.)
                    weighted_stacked_spectra += np.nan_to_num(survey1[gal_idx], nan=0.)*w_gal
                    sum_weights += w_gal
                stacked_spectra /= (n+1)
                weighted_stacked_spectra /= sum_weights
                stacked_noises[n] = stacklib.estimate_noise(stacked_spectra)
                weighted_stacked_noises[n] = stacklib.estimate_noise(weighted_stacked_spectra)
            N_val = np.arange(1,N_spectra+1)
            plt.plot(N_val, stacked_noises, label='Unweighted stacking')
            plt.plot(N_val, weighted_stacked_noises, label='Weighted stacking')
            plt.plot(N_val, noise*(N_val)**-0.5, label='Theorical')
            plt.legend()
            plt.xlabel('Number of stacked spectra')
            plt.ylabel('RMS noise of the final spectrum')
            plt.title(f'RMS noise of the stacked spectrum with N spectra for z in [{bins[i]:.1f}, {bins[i+1]:.1f})')
            plt.show()

    V_offset_recovered_ccf, sigmaV_offset_recovered_ccf = driftlib.cross_correlation(V_val, stacked_spectra1[0], stacked_spectra2[0])
    print(f'\nCCF: ∆v = ({V_offset_recovered_ccf} ± {sigmaV_offset_recovered_ccf}) km/s\n')
    V_offset_recovered_fitting, sigmaV_offset_recovered_fitting = driftlib.direct_fitting(V_val, stacked_spectra1[0], stacked_spectra2[0])
    print(f'Direct Fitting: ∆v = ({V_offset_recovered_fitting} ± {sigmaV_offset_recovered_fitting}) km/s\n')


compute_drift(5)