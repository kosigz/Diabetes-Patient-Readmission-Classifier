categorical_features = set(['diabetesMed', 'chlorpropamide', 'weight', 'repaglinide',\
    'medical_specialty', 'rosiglitazone', 'miglitol',\
    'glipizide', 'acetohexamide', 'admission_source_id', 'glipizide-metformin',\
    'glyburide', 'metformin', 'tolbutamide', 'pioglitazone', 'glimepiride-pioglitazone', \
    'glimepiride', 'glyburide-metformin', 'A1Cresult', 'troglitazone',\
    'metformin-rosiglitazone', 'max_glu_serum', 'acarbose', 'metformin-pioglitazone', \
    'payer_code', 'discharge_disposition_id', 'change', 'gender', 'age',\
    'nateglinide', 'tolazamide', 'race', 'insulin',\
    'admission_type_id', 'examide', 'citoglipton', 'diag_2', 'diag_3', 'diag_1'])

numerical_features = set(['number_diagnoses', 'time_in_hospital', 'number_inpatient',\
    'number_emergency', 'num_procedures', 'num_medications', 'number_outpatient',\
    'num_lab_procedures'])

# these are id features
irrelevant = set(['patient_nbr', 'encounter_id'])

# only one feature value over the entire data set
no_signal = set(['examide', 'citoglipton'])
