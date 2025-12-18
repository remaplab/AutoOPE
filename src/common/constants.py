import os

REAL_DATASETS_FOLDER_PATH = os.path.join("../../..", "real_datasets")
OBD_FOLDER_PATH = os.path.join(REAL_DATASETS_FOLDER_PATH, "open_bandit_dataset")
CIFAR10_FOLDER_PATH = os.path.join(os.path.join(REAL_DATASETS_FOLDER_PATH, "cifar10"), "cifar-10-batches-py")
LOGS_FOLDER_PATH = os.path.join("../../..", "logs")
EXPERIMENTS_LOGS_FOLDER_NAME = "log_experiments_OPERA_0.8"
FIG_EXTENSION = 'pdf'
TIME_LOGS_ROOT_FOLDER_NAME = "log_time_experiment"

ECOLI_ID = 39
GLASS_IDENTIFICATION_ID = 42
LETTER_RECOGNITION_ID = 59
OPTICAL_DIGIT_RECOGNITION_ID = 80
PAGE_BLOCK__ID = 78
PEN_DIGIT_RECOGNITION_ID = 81
STATLOG_SATELLITE_ID = 146
STATLOG_VEHICLE_ID = 149
YEAST_ID = 110
WISCONSIN_BREAST_CANCER_ID = 17

CLASSIFICATION_DATA_ID_MAP = {
    'ecoli': ECOLI_ID,
    'glass': GLASS_IDENTIFICATION_ID,
    'letter': LETTER_RECOGNITION_ID,
    'optdigits': OPTICAL_DIGIT_RECOGNITION_ID,
    'page-blocks': PAGE_BLOCK__ID,
    'pendigits': PEN_DIGIT_RECOGNITION_ID,
    'satimage': STATLOG_SATELLITE_ID,
    'vehicle': STATLOG_VEHICLE_ID,
    'yeast': YEAST_ID,
    'breast-cancer': WISCONSIN_BREAST_CANCER_ID,
    'cifar10': None
}

ALL_REAL_WORLD_DATASETS = list(CLASSIFICATION_DATA_ID_MAP.keys()) + ['obd']

OBD_CAMPAIGNS = ['all', 'men', 'women']
OBD_POLICIES = ['bts', 'random']
