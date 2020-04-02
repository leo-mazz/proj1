import pandas as pd
import os

MIMIC_ROOT = os.path.expanduser('~/Data/mimic')

def table_path(table_name):
    return os.path.join(MIMIC_ROOT, table_name)

def compile_to_craneware():
    admissions = pd.read_csv(table_path("ADMISSIONS.csv"))[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','INSURANCE']]
    PATIENTS = pd.read_csv(table_path("PATIENTS.csv"))[['SUBJECT_ID','GENDER','DOB']]
    DIAGNOSES_ICD = pd.read_csv(table_path("DIAGNOSES_ICD.csv")).astype({"ICD9_CODE": str})[['SUBJECT_ID','HADM_ID','SEQ_NUM','ICD9_CODE']]
    DRGCODES = pd.read_csv(table_path("DRGCODES.csv")).astype({"DRG_CODE": str})[['SUBJECT_ID','HADM_ID','DRG_TYPE','DRG_CODE']]
    PROCEDURES_ICD = pd.read_csv(table_path("PROCEDURES_ICD.csv")).astype({"ICD9_CODE": str})[['SUBJECT_ID','HADM_ID','SEQ_NUM','ICD9_CODE']]
    CPTEVENTS = pd.read_csv(table_path("CPTEVENTS.csv")).astype({"CPT_CD": str})[[ 'SUBJECT_ID', 'HADM_ID', 'COSTCENTER', 'CPT_CD',                                                                             'TICKET_ID_SEQ']]

    data_view = admissions.merge(PATIENTS,left_on=['SUBJECT_ID'],right_on=['SUBJECT_ID'], how='inner').merge(DIAGNOSES_ICD,left_on=['SUBJECT_ID','HADM_ID'],right_on=['SUBJECT_ID','HADM_ID'],how='left').merge(PROCEDURES_ICD,left_on=['SUBJECT_ID','HADM_ID'],right_on=['SUBJECT_ID','HADM_ID'],how='left').merge(CPTEVENTS,left_on=['SUBJECT_ID','HADM_ID'],right_on=['SUBJECT_ID','HADM_ID'],how='left')

    data_view.columns = ['subject_id', 'adm_id', 'adm_time', 'disch_time', 'insurance', 'gender', 'date_of_birth', 'diagnosis_seq', 'diagnosis_code', 'procedure_seq', 'procedure_code', 'cost_center', 'cost_code', 'cost_seq']

    return data_view

def write_craneware():
    data_view = compile_to_craneware()
    data_view.to_csv(index=False, path_or_buf=table_path('craneware.csv'))

def look_up_diagnosis(code):
    diagnoses_dic = pd.read_csv(table_path("D_ICD_DIAGNOSES.csv")).astype({"ICD9_CODE": str})[['SHORT_TITLE','LONG_TITLE','ICD9_CODE']]

    return diagnoses_dic[diagnoses_dic.ICD9_CODE==code].loc[:, 'LONG_TITLE'].tolist()[0]

def look_up_procedure(code):
    procedures_dic = pd.read_csv(table_path("D_ICD_PROCEDURES.csv")).astype({"ICD9_CODE": str})[['SHORT_TITLE','LONG_TITLE','ICD9_CODE']]

    return procedures_dic[procedures_dic.ICD9_CODE==code].loc[:, 'LONG_TITLE'].tolist()[0]

def read_craneware():
    data_view = pd.read_csv(table_path('craneware.csv'), dtype={
        'subject_id': int,
        'adm_id': int,
        'adm_time': str,
        'disch_time': str,
        'insurance': str,
        'gender': str,
        'date_of_birth': str,
        'diagnosis_seq': float,
        'diagnosis_code': str,
        'procedure_seq': float,
        'procedure_code': str,
        'cost_center': str,
        'cost_code': str,
        'cost_seq': float
    }, parse_dates=['adm_time', 'disch_time', 'date_of_birth'])

    return data_view

"""
subject_id | admission_id | admit_time | discharge_time | insurance(government/private/medicare/medicaid) | date_of_birth

diagnoses_sequence_no | diagnoses_code(mapping)

[drg_type (a handful) | drg_code (no mapping) should be together!]

procedures_sequence_no | procedures_code (mapping)

cost_center (icu/resp) | current_procedural_terminology_code (very convenient mapping!) |  ticket_sequence_no

"""

def get_notes():
    data = pd.read_csv(table_path('NOTEEVENTS.csv'))
    return data


def get_admissions():
    data = pd.read_csv(table_path('ADMISSIONS.csv'))
    return data

def get_procedures():
    procedures = pd.read_csv(table_path("PROCEDURES_ICD.csv")).astype(
        {"ICD9_CODE": str})[['SUBJECT_ID','HADM_ID','SEQ_NUM','ICD9_CODE']]
    
    return procedures

def get_patients():
    data = pd.read_csv(table_path('PATIENTS.csv'))
    return data

def get_patient_demographics():
    adm = get_admissions()
    patients = get_patients()
    patients = adm.merge(patients,left_on=['SUBJECT_ID'],right_on=['SUBJECT_ID'], how='inner')
    patients = patients.drop_duplicates(subset=['SUBJECT_ID'])

    patients['DOB'] = pd.to_datetime(patients['DOB']).dt.year

    return patients

    
def make_list(df):
    """ Turn a pandas dataframe into a list of records """
    return df.fillna('?').values.tolist()