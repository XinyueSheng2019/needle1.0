# RANDOM SEED
SEED = 443


# exception paths
EXCEPTION_IMG_PATH = '../image/failed_objs.txt'
EXCEPTION_LC_PATH = '../light_curve/failed_objs.txt'

SCALING_DATA_PATH = '../info/global_scaling_data_hosted_new.json'

# data paths
OBJ_INFO_PATH = '../info/ztf_train_valid_set.csv'

DEFAULT_DATA_PATH = '/Users/xinyuesheng/Documents/astro_projects/data/' 
MAG_OUTPUT_PATH = '/Users/xinyuesheng/Documents/astro_projects/data/mag_sets_v4'
HOST_DATA_PATH = '/Users/xinyuesheng/Documents/astro_projects/data/host_info_r5_ext_new'


UNTOUCHED_2025_PATH = '../untouched_2025/'
UNTOUCHED_2025_INFO_PATH = UNTOUCHED_2025_PATH+'20240225_20250603.csv'
UNTOUCHED_2025_INPUT_IMG_PATH = UNTOUCHED_2025_PATH + 'images/'
UNTOUCHED_2025_IMG_OUTPUT_PATH = UNTOUCHED_2025_PATH + 'image_preprocessing_output/'
UNTOUCHED_2025_UNMASKED_IMG_OUTPUT_PATH = UNTOUCHED_2025_PATH + 'image_unmasked_output/'
UNTOUCHED_2025_MAG_OUTPUT_PATH = UNTOUCHED_2025_PATH + 'mags/'
UNTOUCHED_2025_HOST_PATH = UNTOUCHED_2025_PATH + 'hosts_ext/'
UNTOUCHED_2025_LC_OUTPUT_PATH = UNTOUCHED_2025_PATH + 'light_curve_upsampling_output/'


# output paths
PHOTO_OUTPUT_PATH = '../light_curve/photo_processing_output_new'
IMG_OUTPUT_PATH = '../image/image_preprocessing_output'
UNMASKED_IMG_OUTPUT_PATH = '../image/image_unmasked_output'
NEEDLE_SET_PATH = '../needle_inputs'

# labels
RAW_LABEL_DICT = {
                    "3-class":
                    {'SLSN-II': 0, 
                    'SN Ib/c': 0, 
                    'SN Ib-pec': 0, 
                    'SN IIP': 0, 
                    'SN Ia': 0, 
                    'SN Ic-pec': 0, 
                    'SN II': 0, 
                    'SN II-pec': 0, 
                    'SN Ia-CSM': 0,  
                    'SN Ibn': 0, 
                    'SN Ic-BL': 0, 
                    'SN IIb': 0, 
                    'SN Iax': 0, 
                    'SN Ic': 0, 
                    'SN Ia-91T': 0, 
                    'SN Ia-91bg': 0, 
                    'SN Ib': 0, 
                    'SN IIn': 0, 
                    'SN Ia-SC': 0, 
                    'SN Ia-pec': 0, 
                    'SN Icn': 0, 
                    'SLSN-I': 1, 
                    'TDE': 2,
                    'TDE-He': 2, 
                    'SN Ca-rich-Ca': 3,
                    'LRN': 3, 
                    'LBV': 3,
                    'nova': 3,
                    'Gap': 3}
                    , 
                    "2-class":{
                    'SLSN-II': 0, 
                    'SN Ib/c': 0, 
                    'SN Ib-pec': 0, 
                    'SN IIP': 0, 
                    'SN Ia': 0, 
                    'SN Ic-pec': 0, 
                    'SN II': 0, 
                    'SN II-pec': 0, 
                    'SN Ia-CSM': 0,  
                    'SN Ibn': 0, 
                    'SN Ic-BL': 0, 
                    'SN IIb': 0, 
                    'SN Iax': 0, 
                    'SN Ic': 0, 
                    'SN Ia-91T': 0, 
                    'SN Ia-91bg': 0, 
                    'SN Ib': 0, 
                    'SN IIn': 0, 
                    'SN Ia-SC': 0, 
                    'SN Ia-pec': 0, 
                    'SN Icn': 0, 
                    'SLSN-I': 1, 
                    'TDE': 0,
                    'TDE-He': 0, 
                    'SN Ca-rich-Ca': 0,
                    'LRN': 0, 
                    'LBV': 0,
                    'nova': 0,
                    'Gap': 0
                    },
                    "classify":{
                        "Ia":0, 
                        "II":0, 
                        "Stripped Envelope":0, 
                        "Ic":0, 
                        "Interacting SN":0, 
                        "SLSN-I":1, 
                        "TDE":2, 
                        "Non-SN":3, 
                        "Other":4
                    },
                    "reverse_label":{
                        "0":"SN",
                        "1":"SLSN-I",
                        "2":"TDE",
                        "3":"Non-SN",
                        "4":"Other"
                    },
                    "label-hosted":{
                        "SN":0,
                        "SLSN-I":1,
                        "TDE":2
                    },
                    "label-hostless":{
                        "SN":0,
                        "SLSN-I":1
                    },
                    "test_num":{
                        "SN":15,
                        "SLSN-I":15,
                        "TDE":15
                    }
                    }

LABEL_DICT = {'SN': 0, 'SLSN-I': 1, 'TDE': 2}

LABEL_DICT_HOSTED = {'SN': 0, 'SLSN-I': 1, 'TDE': 2} 
LABEL_DICT_HOSTLESS = {'Non-SLSN': 0, 'SLSN-I': 1} 

LABEL_DICT_SLSN = {'SN': 0, 'SLSN-I': 1, 'TDE': 0} # for binary classification
LABEL_DICT_TDE = {'SN': 0, 'SLSN-I': 0, 'TDE': 1} # for binary classification


FEATURE_LIMIT_DICT = {
    'host_u': [0,23.3],
    'host_g': [0,23.2],
    'host_r': [0,23.1],
    'host_i': [0,22.3],
    'host_z': [0,21.4],
    't_g_minus_r': [-50, 50],
    'ratio_recent': [0, 100],


}