### Create surveys and get velocity drift

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import HI_line_lib as HIlib
import observations_lib as obslib
import stacking_lib as stacklib
import velocity_drift_lib as driftlib
import query_data_lib as datalib
np.random.seed(123)

test_galaxy_catalog = [
    {
        "id": "Gal_01_01",
        "z": 0.0144856,
        "s_int": 0.601,
        "central_dens": 0.006,
        "peak_dens": 0.0104,
        "w20": 139.0,
        "w50": 128.0,
        "w_peak": 100.0,
    },
    {
        "id": "Gal_01_02",
        "z": 0.0144913,
        "s_int": 0.44,
        "central_dens": 0.00723,
        "peak_dens": 0.0132,
        "w20": 124.0,
        "w50": 112.0,
        "w_peak": 87.1,
    },
    {
        "id": "Gal_01_03",
        "z": 0.0147245,
        "s_int": 5.23,
        "central_dens": 0.00317,
        "peak_dens": 0.00696,
        "w20": 235.0,
        "w50": 223.0,
        "w_peak": 196.0,
    },
    {
        "id": "Gal_01_04",
        "z": 0.0148712,
        "s_int": 0.0378,
        "central_dens": 0.0095,
        "peak_dens": 0.0153,
        "w20": 97.7,
        "w50": 86.7,
        "w_peak": 61.9,
    },
    {
        "id": "Gal_01_05",
        "z": 0.0144363,
        "s_int": 0.244,
        "central_dens": 0.00642,
        "peak_dens": 0.011,
        "w20": 130.0,
        "w50": 117.0,
        "w_peak": 89.6,
    },
    {
        "id": "Gal_01_06",
        "z": 0.0146445,
        "s_int": 0.629,
        "central_dens": 0.00469,
        "peak_dens": 0.00885,
        "w20": 171.0,
        "w50": 159.0,
        "w_peak": 134.0,
    },
    {
        "id": "Gal_01_07",
        "z": 0.0146147,
        "s_int": 0.547,
        "central_dens": 0.00433,
        "peak_dens": 0.00802,
        "w20": 185.0,
        "w50": 173.0,
        "w_peak": 145.0,
    },
    {
        "id": "Gal_01_08",
        "z": 0.0145425,
        "s_int": 0.845,
        "central_dens": 0.00267,
        "peak_dens": 0.00659,
        "w20": 279.0,
        "w50": 268.0,
        "w_peak": 238.0,
    },
    {
        "id": "Gal_01_09",
        "z": 0.0145497,
        "s_int": 2.37,
        "central_dens": 0.00329,
        "peak_dens": 0.00693,
        "w20": 232.0,
        "w50": 220.0,
        "w_peak": 190.0,
    },
    {
        "id": "Gal_01_10",
        "z": 0.0147948,
        "s_int": 0.608,
        "central_dens": 0.0125,
        "peak_dens": 0.0222,
        "w20": 79.2,
        "w50": 68.3,
        "w_peak": 46.0,
    },
    {
        "id": "Gal_01_11",
        "z": 0.014926,
        "s_int": 1.14,
        "central_dens": 0.00248,
        "peak_dens": 0.00564,
        "w20": 307.0,
        "w50": 296.0,
        "w_peak": 269.0,
    },
    {
        "id": "Gal_01_12",
        "z": 0.0147545,
        "s_int": 0.371,
        "central_dens": 0.00908,
        "peak_dens": 0.0158,
        "w20": 102.0,
        "w50": 90.2,
        "w_peak": 68.5,
    },
    {
        "id": "Gal_01_13",
        "z": 0.0146834,
        "s_int": 0.054,
        "central_dens": 0.00706,
        "peak_dens": 0.012,
        "w20": 124.0,
        "w50": 113.0,
        "w_peak": 87.5,
    },
    {
        "id": "Gal_01_14",
        "z": 0.0150795,
        "s_int": 0.0585,
        "central_dens": 0.00653,
        "peak_dens": 0.0101,
        "w20": 131.0,
        "w50": 119.0,
        "w_peak": 92.2,
    },
    {
        "id": "Gal_01_15",
        "z": 0.0151515,
        "s_int": 0.288,
        "central_dens": 0.00829,
        "peak_dens": 0.0159,
        "w20": 110.0,
        "w50": 98.8,
        "w_peak": 74.6,
    },
    {
        "id": "Gal_01_16",
        "z": 0.0148115,
        "s_int": 0.0529,
        "central_dens": 0.00448,
        "peak_dens": 0.00926,
        "w20": 175.0,
        "w50": 162.0,
        "w_peak": 134.0,
    },
    {
        "id": "Gal_01_17",
        "z": 0.0144382,
        "s_int": 0.549,
        "central_dens": 0.00948,
        "peak_dens": 0.0159,
        "w20": 94.8,
        "w50": 83.6,
        "w_peak": 59.8,
    },
    {
        "id": "Gal_01_18",
        "z": 0.0228072,
        "s_int": 0.111,
        "central_dens": 0.00854,
        "peak_dens": 0.0132,
        "w20": 103.0,
        "w50": 91.7,
        "w_peak": 64.7,
    },
    {
        "id": "Gal_01_19",
        "z": 0.02295,
        "s_int": 0.163,
        "central_dens": 0.00833,
        "peak_dens": 0.0135,
        "w20": 105.0,
        "w50": 93.2,
        "w_peak": 68.0,
    },
    {
        "id": "Gal_01_20",
        "z": 0.0233515,
        "s_int": 0.272,
        "central_dens": 0.00566,
        "peak_dens": 0.0101,
        "w20": 146.0,
        "w50": 134.0,
        "w_peak": 105.0,
    },
    {
        "id": "Gal_01_21",
        "z": 0.0234579,
        "s_int": 0.057,
        "central_dens": 0.00965,
        "peak_dens": 0.0154,
        "w20": 93.3,
        "w50": 82.3,
        "w_peak": 60.2,
    },
    {
        "id": "Gal_01_22",
        "z": 0.023413,
        "s_int": 0.207,
        "central_dens": 0.00439,
        "peak_dens": 0.00866,
        "w20": 184.0,
        "w50": 172.0,
        "w_peak": 146.0,
    },
    {
        "id": "Gal_01_23",
        "z": 0.0226952,
        "s_int": 0.0988,
        "central_dens": 0.00543,
        "peak_dens": 0.0094,
        "w20": 152.0,
        "w50": 140.0,
        "w_peak": 112.0,
    },
    {
        "id": "Gal_01_24",
        "z": 0.0226484,
        "s_int": 0.893,
        "central_dens": 0.00392,
        "peak_dens": 0.00813,
        "w20": 195.0,
        "w50": 183.0,
        "w_peak": 155.0,
    },
    {
        "id": "Gal_01_25",
        "z": 0.0232155,
        "s_int": 0.214,
        "central_dens": 0.00722,
        "peak_dens": 0.0125,
        "w20": 118.0,
        "w50": 106.0,
        "w_peak": 84.1,
    },
    {
        "id": "Gal_01_26",
        "z": 0.0238029,
        "s_int": 0.0286,
        "central_dens": 0.00441,
        "peak_dens": 0.00885,
        "w20": 180.0,
        "w50": 168.0,
        "w_peak": 140.0,
    },
    {
        "id": "Gal_01_27",
        "z": 0.0240842,
        "s_int": 0.0509,
        "central_dens": 0.00896,
        "peak_dens": 0.0133,
        "w20": 101.0,
        "w50": 89.5,
        "w_peak": 61.0,
    },
    {
        "id": "Gal_01_28",
        "z": 0.024348,
        "s_int": 0.292,
        "central_dens": 0.00466,
        "peak_dens": 0.00887,
        "w20": 174.0,
        "w50": 162.0,
        "w_peak": 134.0,
    },
    {
        "id": "Gal_01_29",
        "z": 0.0252715,
        "s_int": 0.0605,
        "central_dens": 0.00767,
        "peak_dens": 0.0151,
        "w20": 110.0,
        "w50": 98.9,
        "w_peak": 75.9,
    },
    {
        "id": "Gal_01_30",
        "z": 0.0258166,
        "s_int": 0.0968,
        "central_dens": 0.00902,
        "peak_dens": 0.0153,
        "w20": 102.0,
        "w50": 91.1,
        "w_peak": 64.7,
    },
    {
        "id": "Gal_01_31",
        "z": 0.0255958,
        "s_int": 0.0312,
        "central_dens": 0.00569,
        "peak_dens": 0.0111,
        "w20": 145.0,
        "w50": 134.0,
        "w_peak": 109.0,
    },
    {
        "id": "Gal_01_32",
        "z": 0.0238523,
        "s_int": 0.367,
        "central_dens": 0.00723,
        "peak_dens": 0.0135,
        "w20": 117.0,
        "w50": 106.0,
        "w_peak": 84.1,
    },
    {
        "id": "Gal_01_33",
        "z": 0.0250415,
        "s_int": 0.137,
        "central_dens": 0.00756,
        "peak_dens": 0.0126,
        "w20": 114.0,
        "w50": 103.0,
        "w_peak": 77.0,
    },
    {
        "id": "Gal_01_34",
        "z": 0.0259596,
        "s_int": 0.18,
        "central_dens": 0.00566,
        "peak_dens": 0.01,
        "w20": 144.0,
        "w50": 133.0,
        "w_peak": 105.0,
    },
    {
        "id": "Gal_01_35",
        "z": 0.0251861,
        "s_int": 0.011,
        "central_dens": 0.0352,
        "peak_dens": 0.0499,
        "w20": 46.4,
        "w50": 36.2,
        "w_peak": 15.0,
    },
    {
        "id": "Gal_01_36",
        "z": 0.0258826,
        "s_int": 0.407,
        "central_dens": 0.0105,
        "peak_dens": 0.0189,
        "w20": 88.8,
        "w50": 77.7,
        "w_peak": 56.5,
    },
    {
        "id": "Gal_01_37",
        "z": 0.0259939,
        "s_int": 1.84,
        "central_dens": 0.00498,
        "peak_dens": 0.0107,
        "w20": 160.0,
        "w50": 148.0,
        "w_peak": 123.0,
    },
    {
        "id": "Gal_01_38",
        "z": 0.0261686,
        "s_int": 0.232,
        "central_dens": 0.0136,
        "peak_dens": 0.0204,
        "w20": 74.5,
        "w50": 63.0,
        "w_peak": 38.2,
    },
    {
        "id": "Gal_01_39",
        "z": 0.025979,
        "s_int": 0.0227,
        "central_dens": 0.00756,
        "peak_dens": 0.0124,
        "w20": 111.0,
        "w50": 100.0,
        "w_peak": 75.3,
    },
    {
        "id": "Gal_01_40",
        "z": 0.0373645,
        "s_int": 0.101,
        "central_dens": 0.0064,
        "peak_dens": 0.0111,
        "w20": 131.0,
        "w50": 120.0,
        "w_peak": 94.8,
    },
    {
        "id": "Gal_01_41",
        "z": 0.0389333,
        "s_int": 0.0365,
        "central_dens": 0.0103,
        "peak_dens": 0.0161,
        "w20": 89.5,
        "w50": 78.2,
        "w_peak": 52.1,
    },
    {
        "id": "Gal_01_42",
        "z": 0.0396538,
        "s_int": 0.174,
        "central_dens": 0.00476,
        "peak_dens": 0.0102,
        "w20": 164.0,
        "w50": 152.0,
        "w_peak": 126.0,
    },
    {
        "id": "Gal_01_43",
        "z": 0.0383343,
        "s_int": 0.0392,
        "central_dens": 0.0149,
        "peak_dens": 0.0251,
        "w20": 73.8,
        "w50": 63.1,
        "w_peak": 41.0,
    },
    {
        "id": "Gal_01_44",
        "z": 0.0383243,
        "s_int": 0.00872,
        "central_dens": 0.00867,
        "peak_dens": 0.0151,
        "w20": 107.0,
        "w50": 95.6,
        "w_peak": 72.0,
    },
    {
        "id": "Gal_01_45",
        "z": 0.0394438,
        "s_int": 0.0389,
        "central_dens": 0.00325,
        "peak_dens": 0.00542,
        "w20": 253.0,
        "w50": 220.0,
        "w_peak": 166.0,
    },
    {
        "id": "Gal_01_46",
        "z": 0.0397129,
        "s_int": 0.0232,
        "central_dens": 0.00982,
        "peak_dens": 0.0151,
        "w20": 95.6,
        "w50": 84.2,
        "w_peak": 58.4,
    },
    {
        "id": "Gal_01_47",
        "z": 0.0394789,
        "s_int": 0.00858,
        "central_dens": 0.00725,
        "peak_dens": 0.0114,
        "w20": 122.0,
        "w50": 110.0,
        "w_peak": 83.0,
    },
    {
        "id": "Gal_01_48",
        "z": 0.038939,
        "s_int": 0.184,
        "central_dens": 0.00627,
        "peak_dens": 0.0127,
        "w20": 140.0,
        "w50": 129.0,
        "w_peak": 87.1,
    },
    {
        "id": "Gal_01_49",
        "z": 0.0390861,
        "s_int": 0.0454,
        "central_dens": 0.00695,
        "peak_dens": 0.0116,
        "w20": 122.0,
        "w50": 111.0,
        "w_peak": 86.1,
    },
    {
        "id": "Gal_01_50",
        "z": 0.0387037,
        "s_int": 0.032,
        "central_dens": 0.0476,
        "peak_dens": 0.0499,
        "w20": 42.9,
        "w50": 32.6,
        "w_peak": 12.5,
    },
    {
        "id": "Gal_01_51",
        "z": 0.0391026,
        "s_int": 0.0293,
        "central_dens": 0.00235,
        "peak_dens": 0.00501,
        "w20": 326.0,
        "w50": 299.0,
        "w_peak": 249.0,
    },
    {
        "id": "Gal_01_52",
        "z": 0.0385597,
        "s_int": 0.079,
        "central_dens": 0.00661,
        "peak_dens": 0.0119,
        "w20": 125.0,
        "w50": 113.0,
        "w_peak": 90.6,
    },
    {
        "id": "Gal_01_53",
        "z": 0.0391772,
        "s_int": 0.0269,
        "central_dens": 0.0063,
        "peak_dens": 0.0104,
        "w20": 134.0,
        "w50": 122.0,
        "w_peak": 93.7,
    },
    {
        "id": "Gal_01_54",
        "z": 0.0393884,
        "s_int": 0.0367,
        "central_dens": 0.00682,
        "peak_dens": 0.0122,
        "w20": 121.0,
        "w50": 110.0,
        "w_peak": 84.7,
    },
    {
        "id": "Gal_01_55",
        "z": 0.0392842,
        "s_int": 0.113,
        "central_dens": 0.0084,
        "peak_dens": 0.0152,
        "w20": 108.0,
        "w50": 96.9,
        "w_peak": 72.4,
    },
    {
        "id": "Gal_01_56",
        "z": 0.0392985,
        "s_int": 0.103,
        "central_dens": 0.00691,
        "peak_dens": 0.0118,
        "w20": 122.0,
        "w50": 110.0,
        "w_peak": 84.1,
    },
    {
        "id": "Gal_01_57",
        "z": 0.0390245,
        "s_int": 0.0326,
        "central_dens": 0.0118,
        "peak_dens": 0.0183,
        "w20": 84.3,
        "w50": 73.2,
        "w_peak": 48.4,
    },
    {
        "id": "Gal_01_58",
        "z": 0.0396766,
        "s_int": 0.483,
        "central_dens": 0.00403,
        "peak_dens": 0.00833,
        "w20": 194.0,
        "w50": 182.0,
        "w_peak": 155.0,
    },
    {
        "id": "Gal_01_59",
        "z": 0.0392311,
        "s_int": 0.0986,
        "central_dens": 0.0147,
        "peak_dens": 0.0253,
        "w20": 72.1,
        "w50": 61.5,
        "w_peak": 39.9,
    },
    {
        "id": "Gal_01_60",
        "z": 0.0399404,
        "s_int": 0.0604,
        "central_dens": 0.00625,
        "peak_dens": 0.0109,
        "w20": 131.0,
        "w50": 119.0,
        "w_peak": 92.5,
    },
    {
        "id": "Gal_01_61",
        "z": 0.0416082,
        "s_int": 0.686,
        "central_dens": 0.00505,
        "peak_dens": 0.00974,
        "w20": 157.0,
        "w50": 146.0,
        "w_peak": 117.0,
    },
    {
        "id": "Gal_01_62",
        "z": 0.0417236,
        "s_int": 0.0311,
        "central_dens": 0.0114,
        "peak_dens": 0.0177,
        "w20": 84.2,
        "w50": 73.3,
        "w_peak": 51.6,
    },
    {
        "id": "Gal_01_63",
        "z": 0.0430264,
        "s_int": 0.15,
        "central_dens": 0.00905,
        "peak_dens": 0.0166,
        "w20": 99.7,
        "w50": 88.7,
        "w_peak": 64.2,
    },
    {
        "id": "Gal_01_64",
        "z": 0.0429141,
        "s_int": 0.0712,
        "central_dens": 0.00567,
        "peak_dens": 0.0104,
        "w20": 147.0,
        "w50": 136.0,
        "w_peak": 109.0,
    },
    {
        "id": "Gal_01_65",
        "z": 0.0432112,
        "s_int": 0.0178,
        "central_dens": 0.00973,
        "peak_dens": 0.0147,
        "w20": 97.6,
        "w50": 86.1,
        "w_peak": 59.4,
    },
    {
        "id": "Gal_01_66",
        "z": 0.0432512,
        "s_int": 0.0169,
        "central_dens": 0.00821,
        "peak_dens": 0.0138,
        "w20": 105.0,
        "w50": 93.8,
        "w_peak": 69.4,
    },
    {
        "id": "Gal_01_67",
        "z": 0.0437268,
        "s_int": 0.144,
        "central_dens": 0.00738,
        "peak_dens": 0.0154,
        "w20": 122.0,
        "w50": 111.0,
        "w_peak": 88.3,
    },
    {
        "id": "Gal_01_68",
        "z": 0.0438012,
        "s_int": 0.016,
        "central_dens": 0.00627,
        "peak_dens": 0.00991,
        "w20": 135.0,
        "w50": 123.0,
        "w_peak": 92.2,
    },
    {
        "id": "Gal_01_69",
        "z": 0.0462197,
        "s_int": 0.0705,
        "central_dens": 0.012,
        "peak_dens": 0.0206,
        "w20": 82.3,
        "w50": 71.2,
        "w_peak": 47.7,
    },
    {
        "id": "Gal_01_70",
        "z": 0.0465307,
        "s_int": 0.0266,
        "central_dens": 0.00917,
        "peak_dens": 0.0143,
        "w20": 101.0,
        "w50": 89.8,
        "w_peak": 62.9,
    },
    {
        "id": "Gal_01_71",
        "z": 0.0467837,
        "s_int": 0.0808,
        "central_dens": 0.00343,
        "peak_dens": 0.00734,
        "w20": 227.0,
        "w50": 215.0,
        "w_peak": 187.0,
    },
    {
        "id": "Gal_01_72",
        "z": 0.0468288,
        "s_int": 0.0326,
        "central_dens": 0.00915,
        "peak_dens": 0.0153,
        "w20": 99.5,
        "w50": 88.3,
        "w_peak": 64.5,
    },
    {
        "id": "Gal_01_73",
        "z": 0.0471374,
        "s_int": 0.0114,
        "central_dens": 0.0119,
        "peak_dens": 0.0229,
        "w20": 84.1,
        "w50": 73.3,
        "w_peak": 50.8,
    },
    {
        "id": "Gal_01_74",
        "z": 0.0468449,
        "s_int": 0.0222,
        "central_dens": 0.00447,
        "peak_dens": 0.00931,
        "w20": 171.0,
        "w50": 159.0,
        "w_peak": 132.0,
    },
    {
        "id": "Gal_01_75",
        "z": 0.0468877,
        "s_int": 0.0249,
        "central_dens": 0.0499,
        "peak_dens": 0.0499,
        "w20": 39.7,
        "w50": 29.6,
        "w_peak": 10.1,
    },
    {
        "id": "Gal_01_76",
        "z": 0.0469264,
        "s_int": 0.0461,
        "central_dens": 0.00608,
        "peak_dens": 0.0105,
        "w20": 134.0,
        "w50": 122.0,
        "w_peak": 96.2,
    },
    {
        "id": "Gal_01_77",
        "z": 0.0462188,
        "s_int": 0.0149,
        "central_dens": 0.004,
        "peak_dens": 0.00742,
        "w20": 198.0,
        "w50": 180.0,
        "w_peak": 144.0,
    },
    {
        "id": "Gal_01_78",
        "z": 0.047559,
        "s_int": 0.0425,
        "central_dens": 0.00511,
        "peak_dens": 0.00965,
        "w20": 154.0,
        "w50": 142.0,
        "w_peak": 120.0,
    },
    {
        "id": "Gal_01_79",
        "z": 0.0465763,
        "s_int": 0.0201,
        "central_dens": 0.00588,
        "peak_dens": 0.0102,
        "w20": 140.0,
        "w50": 128.0,
        "w_peak": 102.0,
    },
    {
        "id": "Gal_01_80",
        "z": 0.0461555,
        "s_int": 0.00241,
        "central_dens": 0.00601,
        "peak_dens": 0.0104,
        "w20": 140.0,
        "w50": 129.0,
        "w_peak": 101.0,
    },
    {
        "id": "Gal_01_81",
        "z": 0.0468281,
        "s_int": 0.00137,
        "central_dens": 0.004,
        "peak_dens": 0.00913,
        "w20": 195.0,
        "w50": 183.0,
        "w_peak": 156.0,
    },
    {
        "id": "Gal_01_82",
        "z": 0.0466516,
        "s_int": 0.0173,
        "central_dens": 0.00652,
        "peak_dens": 0.0116,
        "w20": 126.0,
        "w50": 114.0,
        "w_peak": 90.2,
    },
    {
        "id": "Gal_01_83",
        "z": 0.0467319,
        "s_int": 0.0821,
        "central_dens": 0.00164,
        "peak_dens": 0.00526,
        "w20": 432.0,
        "w50": 412.0,
        "w_peak": 376.0,
    },
    {
        "id": "Gal_01_84",
        "z": 0.0472384,
        "s_int": 0.0048,
        "central_dens": 0.00728,
        "peak_dens": 0.0151,
        "w20": 120.0,
        "w50": 109.0,
        "w_peak": 85.1,
    },
    {
        "id": "Gal_01_85",
        "z": 0.0474365,
        "s_int": 0.0197,
        "central_dens": 0.027,
        "peak_dens": 0.0448,
        "w20": 52.0,
        "w50": 41.6,
        "w_peak": 20.8,
    },
    {
        "id": "Gal_01_86",
        "z": 0.0468607,
        "s_int": 0.0315,
        "central_dens": 0.00559,
        "peak_dens": 0.0112,
        "w20": 145.0,
        "w50": 133.0,
        "w_peak": 108.0,
    },
    {
        "id": "Gal_01_87",
        "z": 0.0469282,
        "s_int": 0.00302,
        "central_dens": 0.00267,
        "peak_dens": 0.00515,
        "w20": 295.0,
        "w50": 267.0,
        "w_peak": 216.0,
    },
    {
        "id": "Gal_01_88",
        "z": 0.04621,
        "s_int": 0.0253,
        "central_dens": 0.00697,
        "peak_dens": 0.0113,
        "w20": 122.0,
        "w50": 110.0,
        "w_peak": 87.4,
    },
    {
        "id": "Gal_01_89",
        "z": 0.0474891,
        "s_int": 0.0101,
        "central_dens": 0.0179,
        "peak_dens": 0.0333,
        "w20": 65.1,
        "w50": 54.6,
        "w_peak": 34.5,
    },
    {
        "id": "Gal_01_90",
        "z": 0.0468401,
        "s_int": 0.0508,
        "central_dens": 0.0081,
        "peak_dens": 0.0134,
        "w20": 107.0,
        "w50": 96.0,
        "w_peak": 70.9,
    },
    {
        "id": "Gal_01_91",
        "z": 0.0473098,
        "s_int": 0.0488,
        "central_dens": 0.00836,
        "peak_dens": 0.0138,
        "w20": 104.0,
        "w50": 92.3,
        "w_peak": 66.7,
    },
    {
        "id": "Gal_01_92",
        "z": 0.0469007,
        "s_int": 0.0644,
        "central_dens": 0.00625,
        "peak_dens": 0.0107,
        "w20": 132.0,
        "w50": 120.0,
        "w_peak": 91.7,
    },
    {
        "id": "Gal_01_93",
        "z": 0.0473233,
        "s_int": 0.0284,
        "central_dens": 0.00833,
        "peak_dens": 0.0132,
        "w20": 107.0,
        "w50": 95.2,
        "w_peak": 69.9,
    },
    {
        "id": "Gal_01_94",
        "z": 0.0468988,
        "s_int": 0.188,
        "central_dens": 0.00344,
        "peak_dens": 0.00739,
        "w20": 227.0,
        "w50": 215.0,
        "w_peak": 189.0,
    },
    {
        "id": "Gal_01_95",
        "z": 0.0471317,
        "s_int": 0.0864,
        "central_dens": 0.0103,
        "peak_dens": 0.0142,
        "w20": 90.4,
        "w50": 77.9,
        "w_peak": 47.4,
    },
    {
        "id": "Gal_01_96",
        "z": 0.0468741,
        "s_int": 0.0141,
        "central_dens": 0.0499,
        "peak_dens": 0.0499,
        "w20": 30.1,
        "w50": 20.2,
        "w_peak": 1.31,
    },
    {
        "id": "Gal_01_97",
        "z": 0.0471009,
        "s_int": 0.0514,
        "central_dens": 0.0111,
        "peak_dens": 0.0183,
        "w20": 85.8,
        "w50": 74.6,
        "w_peak": 50.4,
    },
    {
        "id": "Gal_01_98",
        "z": 0.0475921,
        "s_int": 0.076,
        "central_dens": 0.0129,
        "peak_dens": 0.022,
        "w20": 77.6,
        "w50": 66.9,
        "w_peak": 46.1,
    },
    {
        "id": "Gal_01_99",
        "z": 0.0470828,
        "s_int": 0.0405,
        "central_dens": 0.00767,
        "peak_dens": 0.0126,
        "w20": 111.0,
        "w50": 99.2,
        "w_peak": 71.3,
    },
    {
        "id": "Gal_01_100",
        "z": 0.0476039,
        "s_int": 0.00288,
        "central_dens": 0.00319,
        "peak_dens": 0.00673,
        "w20": 251.0,
        "w50": 227.0,
        "w_peak": 188.0,
    },
]
test_galaxy_params = [tuple(d.values()) for d in test_galaxy_catalog]

N_sources = 100_000
rphitheta, vpec, zapp, profileval, incl = datalib.getsaxdata(limit=N_sources)
# profileval: # apparent redshift, hiintflux, hilumcenter, hilumpeak, hidiwthpeak, hiwdith50, hiwdith20
print(f"Successfully queried {N_sources} galaxies from the database!")


# Create 1st survey

def first_survey(arr_V, params, noise, sys_err=False):
    '''
    '''
    lines_1st = np.zeros((len(params), len(arr_V)))
    for i, parameters in enumerate(params):
        z, line_flux, psi_0, psi_max, w_peak, w_50, w_20 = parameters
        try:
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
        except ZeroDivisionError:
            print(f"Warning: Skipping Galaxy {i} (survey 1) due to division by zero (psi_max = {psi_max}, psi_0 = {psi_0})")

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
        z, line_flux, psi_0, psi_max, w_peak, w_50, w_20 = parameters
        try:

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

        except ZeroDivisionError:
            print(f"Warning: Skipping Galaxy {i} (survey 2) due to division by zero (psi_max = {psi_max}, psi_0 = {psi_0})")

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

    # noise = obslib.S_rms_SKA(100, 144, 0.001)
    survey1 = first_survey(V_val, profileval, noise, sys_err)
    survey2 = second_survey(V_val, profileval, noise, offset, sys_err)

    z_arr = np.array([profile[0] for profile in profileval])
    bins = np.arange(0, max(z_arr)+0.1, 0.1)
    bin_indices = np.digitize(z_arr, bins)
    z_bin, N_bin = np.unique(sorted(bin_indices), return_counts=True) # gets number of spectra in each redshift bin
    if weights:
        stacked_spectra1, stacked_spectra2 = stacklib.weighted_mean_stack(z_arr, bins, survey1, survey2)
    else:
        stacked_spectra1, stacked_spectra2 = stacklib.mean_stack(z_arr, bins, survey1, survey2)

    for j in range(len(bins)-1):
        N_spectra = N_bin[j]
        print(f"========= There are {N_spectra} galaxies for z in [{bins[j]:.1f}, {bins[j+1]:.1f}) =========")

        if plot_stacked:
            plt.plot(V_val, stacked_spectra1[j], label='1st survey')
            plt.plot(V_val, stacked_spectra2[j], label='2nd survey')
            plt.legend()
            plt.xlabel('Velocity [km/s]')
            plt.ylabel('Flux [Jy]')
            plt.title(f'Stacked spectra for z in [{bins[j]:.1f}, {bins[j+1]:.1f})')
            plt.show()

    
        if analyse_noise:
            stacked_noise, stacked_SNR = stacklib.estimate_noise(stacked_spectra1[j])
            print(f"\nEstimated noise in the stacked spectrum for z in [{bins[j]:.1f}, {bins[j+1]:.1f}): {stacked_noise:.1e} Jy")
            print(f"Theorical noise in the stacked spectrum for z in [{bins[j]:.1f}, {bins[j+1]:.1f}): {noise/np.sqrt(100):.1e} Jy")
            print(f"Estimated SNR in the stacked spectrum for z in [{bins[j]:.1f}, {bins[j+1]:.1f}): {stacked_SNR:.1f}")

        V_offset_recovered_ccf, sigmaV_offset_recovered_ccf = driftlib.cross_correlation(V_val, stacked_spectra1[j], stacked_spectra2[j])
        print(f'\nCCF: ∆v = ({V_offset_recovered_ccf} ± {sigmaV_offset_recovered_ccf}) km/s\n')
        V_offset_recovered_fitting, sigmaV_offset_recovered_fitting = driftlib.direct_fitting(V_val, stacked_spectra1[j], stacked_spectra2[j])
        print(f'Direct Fitting: ∆v = ({V_offset_recovered_fitting} ± {sigmaV_offset_recovered_fitting}) km/s\n')

        if plot_RMSvsN:
            stacked_noises = np.zeros(N_spectra)
            weighted_stacked_noises = np.zeros(N_spectra)
            for n in range(N_spectra):
                stacked_spectra = np.zeros(len(survey1[0]))
                weighted_stacked_spectra = np.zeros(len(survey1[0]))
                sum_weights = 0
                for gal_idx in range(len(survey1[:n+1])):
                    w_gal = (stacklib.estimate_noise(survey1[gal_idx])[0])**(-2)
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
            plt.title(f'RMS noise of the stacked spectrum with N spectra for z in [{bins[j]:.1f}, {bins[j+1]:.1f})')
            plt.show()


compute_drift(0.001, plot_stacked=True, analyse_noise=True)


def FWHM_centroid_distributions(noise, N=1000, offset=10, sys_err=False, verify_fits=False):
    V_val = np.linspace(-500, 500, N)
    
    # noise = obslib.S_rms_SKA(100, 144, 0.001)
    survey1 = first_survey(V_val, profileval, noise, sys_err)
    survey2 = second_survey(V_val, profileval, noise, offset, sys_err)

    N_spectra = len(survey1)
    Dx0_val = np.zeros(N_spectra)
    Dsigma_val = np.zeros(N_spectra)
    for i in range(N_spectra):

        try:
            p0_1 = [survey1[i,np.argmin(abs(V_val))], 0, 30, np.median(survey1[i])]
            popt1, pcov1 = curve_fit(driftlib.Gaussian, V_val, survey1[i], p0=p0_1) # fits Gaussian to stacked spectrum of the 1st epoch
            A1, x01, sigma1, C1 = popt1 # fitted parameters

            p0_2 = [A1, offset, sigma1, C1]
            popt2, pcov2 = curve_fit(driftlib.Gaussian, V_val, survey2[i], p0=p0_2) # fits Gaussian to stacked spectrum of the 2nd epoch
            A2, x02, sigma2, C2 = popt2 # fitted parameters
            
            Dx0_val[i] = x02 - x01
            Dsigma_val[i] = sigma2 - sigma1

        except RuntimeError:
            Dx0_val[i] = np.nan
            Dsigma_val[i] = np.nan

        if np.all(survey1[i] == 0.):
            Dx0_val[i] = np.nan
            Dsigma_val[i] = np.nan

    if verify_fits:
        mask = ~np.isnan(Dsigma_val)
        for i in np.argsort(abs(Dsigma_val[mask]))[-5:]: # verify the 10 fittings with highest Dsigma
            plt.plot(V_val, survey1[mask][i], linewidth=0.5, label='1st survey')
            plt.plot(V_val, survey2[mask][i], linewidth=0.5, label='2nd survey')
            p0_1 = [survey1[i,np.argmin(abs(V_val))], 0, 30, np.median(survey1[i])]
            popt1, pcov1 = curve_fit(driftlib.Gaussian, V_val, survey1[mask][i], p0=p0_1) # fits Gaussian to stacked spectrum of the 1st epoch
            A1, x01, sigma1, C1 = popt1 # fitted parameters
            p0_2 = [A1, offset, sigma1, C1]
            popt2, pcov2 = curve_fit(driftlib.Gaussian, V_val, survey2[mask][i], p0=p0_2) # fits Gaussian to stacked spectrum of the 2nd epoch
            A2, x02, sigma2, C2 = popt2 # fitted parameters
            print(x01, x02, sigma1, sigma2)
            print(Dx0_val[mask][i], Dsigma_val[mask][i])
            plt.plot(V_val, driftlib.Gaussian(V_val, A1, x01, sigma1, C1), label='Gaussian fit for the 1st survey')
            plt.plot(V_val, driftlib.Gaussian(V_val, A2, x02, sigma2, C2), label='Gaussian fit for the 2nd survey')
            plt.axvline(x01, linestyle=':', color='tab:green', label=f'1st survey center = {x01:.1f} km/s')
            plt.axvline(x02, linestyle=':', color='tab:red', label=f'2nd survey center = {x02:.1f} km/s')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xlabel('Velocity [km/s]')
            plt.ylabel('Flux [Jy]')
            plt.tight_layout()
            plt.show()

    plt.plot(Dx0_val, Dsigma_val, 'o')
    plt.xlabel(r'$\Delta x_0$')
    plt.ylabel(r'$\Delta \sigma$')
    plt.grid()
    plt.show()

#FWHM_centroid_distributions(0, offset=100, verify_fits=True)