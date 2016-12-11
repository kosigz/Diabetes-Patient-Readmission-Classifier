feature_types = {
    # features with continuous numeric values
    "continuous": [
        "number_diagnoses",
        "time_in_hospital",
        "number_inpatient",
        "number_emergency",
        "num_procedures",
        "num_medications",
        "num_lab_procedures"],

    # features which describe buckets of a continuous-valued feature
    "range": ["age", "weight"],

    # features which take one of two or more non-continuous values
    "categorical": [
        "diabetesMed",
        "chlorpropamide",
        "repaglinide",
        "medical_specialty",
        "rosiglitazone",
        "miglitol",
        "glipizide",
        "acetohexamide",
        "admission_source_id",
        "glipizide-metformin",
        "glyburide",
        "metformin",
        "tolbutamide",
        "pioglitazone",
        "glimepiride-pioglitazone",
        "glimepiride",
        "glyburide-metformin",
        "A1Cresult",
        "troglitazone",
        "metformin-rosiglitazone",
        "max_glu_serum",
        "acarbose",
        "metformin-pioglitazone",
        "payer_code",
        "discharge_disposition_id",
        "change",
        "gender",
        "nateglinide",
        "tolazamide",
        "race",
        "number_outpatient",
        "insulin",
        "admission_type_id",
        "diag_1", "diag_2", "diag_3"],

    # features which are either unique or constant for all samples
    "constant": ["patient_nbr", "encounter_id", "examide", "citoglipton"]}
