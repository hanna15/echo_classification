_val_ids = [38,41,47,54,82]
VAL_PATIENT_IDS_CV = [f'{i}CV' for i in _val_ids]
VAL_PATIENT_IDS = VAL_PATIENT_IDS_CV
# Include strong lesion and some healthy patients in test set, mask of sample 84 contains error
_test_ids_ano = [55,63,66,84,86,226]
_test_ids_healthy = [70,88,95,104,199]
_test_ids = _test_ids_healthy+_test_ids_ano
HEALTHY_TEST_PATIENT_IDS_CV = [f'{i}CV' for i in _test_ids_healthy]
ANO_TEST_PATIENT_IDS_CV = [f'{i}CV' for i in _test_ids_ano]
TEST_PATIENT_IDS = HEALTHY_TEST_PATIENT_IDS_CV+ANO_TEST_PATIENT_IDS_CV

# Exclude minor lesions from training
_minor_lesions_cv = [30, 33, 34, 35, 47, 48, 54, 62, 72, 80, 82, 90,
                 97, 98, 99, 103, 107, 111, 113, 116, 118, 124,
                 132,133,137,142,143,144,145,146,147,148, 151, 153,
                 154,155,156,157,159,160,161,162,163,165,168,171,
                 172,173,177,178,179,182,184,186,187,190,191,192,
                 193,194,195,196,197,198,200,201,202,204,205,207,
                 209,211,213,214,215,216,217,220,221,222,224,229,
                 234,236,239,241,243]
MINOR_ANO_TEST_PATIENT_IDS_CV = [f'{x}CV' for x in _minor_lesions_cv]

# Train on all patients that are not in validation or test set
TRAIN_PATIENT_IDS_CV = [f'{i}CV' for i in range(30,244) if
                        i not in _val_ids and
                        i not in _test_ids and
                        i not in _minor_lesions_cv]
TRAIN_PATIENT_IDS = TRAIN_PATIENT_IDS_CV

# PH labels
_no_ph_ids = [30,31,32,33,34,36,37,38,39,40,41,44,46,48,49,50,51,52,54,56,57,58,59,60,61,64,67,68,69,70,71,72,73,75,77,78,79,81,82,83,85,87,89,91,92,93,95,96,97,99,100,101,102,105,106,108,109,110,112,114,115,116,117,118,119,120,123,125,126,128,129,130,131,135,136,139,142,143,144,145,147,148,149,150,151,153,154,155,156,157,161,162,164,165,166,167,168,169,170,171,173,174,175,176,178,180,181,184,185,187,189,191,193,199,203,204,206,209,210,211,213,215,216,218,220,223,225,226,228,233,235,239,240,242,243,244]
_mild_ph_ids = [35,42,43,45,47,62,66,74,80,84,88,94,121,122,124,127,132,158,160,163,182,186,188,190,192,194,198,200,205,212,214,219,227,230,231,232,238,241]
_clear_ph_ids = [53,55,76,86,98,104,107,111,113,133,137,138,146,152,159,172,177,179,183,195,196,197,201,202,207,208,217,221,222,224,229,234,236,237,]
_distinctive_ph_ids = [63,90,103,134,141]

PH_HEALTHY = {}
PH_MILD = {}
PH_CLEAR = {}
PH_DISTINCTIVE = {}
PH_ANOMALIES = {}

PH_HEALTHY['KAPAP'] = [f'{i}KAPAP' for i in _no_ph_ids]
PH_MILD['KAPAP'] = [f'{i}KAPAP' for i in _mild_ph_ids]
PH_CLEAR['KAPAP'] = [f'{i}KAPAP' for i in _clear_ph_ids]
PH_DISTINCTIVE['KAPAP'] = [f'{i}KAPAP' for i in _distinctive_ph_ids]
PH_ANOMALIES['KAPAP'] = PH_MILD['KAPAP']+PH_CLEAR['KAPAP']+PH_DISTINCTIVE['KAPAP']

PH_HEALTHY['KAAP'] = [f'{i}KAAP' for i in _no_ph_ids]
PH_MILD['KAAP'] = [f'{i}KAAP' for i in _mild_ph_ids]
PH_CLEAR['KAAP'] = [f'{i}KAAP' for i in _clear_ph_ids]
PH_DISTINCTIVE['KAAP'] = [f'{i}KAAP' for i in _distinctive_ph_ids]
PH_ANOMALIES['KAAP'] = PH_MILD['KAAP']+PH_CLEAR['KAAP']+PH_DISTINCTIVE['KAAP']

PH_HEALTHY['LA'] = [f'{i}LA' for i in _no_ph_ids]
PH_MILD['LA'] = [f'{i}LA' for i in _mild_ph_ids]
PH_CLEAR['LA'] = [f'{i}LA' for i in _clear_ph_ids]
PH_DISTINCTIVE['LA'] = [f'{i}LA' for i in _distinctive_ph_ids]
PH_ANOMALIES['LA'] = PH_MILD['LA']+PH_CLEAR['LA']+PH_DISTINCTIVE['LA']