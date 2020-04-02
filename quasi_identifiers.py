class QuasiIdentifier():
    def __init__(self, idx, value, gen_rule):
        self.idx = idx
        self.value = value
        self.gen_rule = gen_rule
        self.generalization = 0

    def generalize(self, steps=1):
        if self.generalization + steps > self.gen_rule[1]: raise ValueError('cannot generalize that much')
        gen_value = self.value
        gen_level = self.generalization
        for _ in range(steps):
            gen_level += 1
            gen_value = self.gen_rule[0](gen_value, gen_level)
        return gen_value, gen_level

# Observation 1: it's useful to have all values being the same type, regardless of generalization
# Observation 2: it's convenient to infer the current generalization level from the value itself and instead pass the generalization level one wants to achieve
# Observation 3: 0 should be a valid generalization level

class GeneralizationRule():
    def __init__(self, apply, max_level):
        self.apply = apply
        self.max_level = max_level

def generalize_gender(value, gen_level):
    if gen_level == 0 and (value == 'Male' or value == 'Female' or value == 'Other'):
        return value
    if gen_level == 1 and (value == 'Male' or value == 'Female' or value == 'Other'):
        return 'Unknown'
    
    raise ValueError(value, gen_level)

generalize_gender_rule = GeneralizationRule(generalize_gender, 1)


def generalize_age(value, gen_level):
    if isinstance(value, str):
        value = int(value)
    if isinstance(value, int):
        value = (value,value)
    if not isinstance(value, tuple): raise ValueError
    if value[0] > value[1]: raise ValueError

    ages = range(130)
    # range_per_gen_level = [1, 5, 10, 20, 50, 130]
    range_per_gen_level = [1, 10, 50, 130]

    if gen_level > len(range_per_gen_level)-1: raise ValueError

    a = list(range(0, ages[-2], range_per_gen_level[gen_level]))
    b = list(range(range_per_gen_level[gen_level]-1, ages[-1]+1, range_per_gen_level[gen_level]))
    
    for r in zip(a,b):
        if value[0] >= r[0] and value[1] <= r[1]:
            return r
    
    raise ValueError(value, gen_level)

generalize_age_rule = GeneralizationRule(generalize_age, 3)


def adult_generalize_country(value, gen_level):
    south_america = ['Columbia', 'Peru', 'Ecuador', 'Trinadad&Tobago']
    north_america = ['United-States', 'Canada', 'Mexico']
    central_america = ['Puerto-Rico', 'Jamaica', 'Guatemala', 'El-Salvador', 'Cuba', 'Dominican-Republic', 'Haiti', 'Honduras', 'Nicaragua']
    eastern_europe = ['Hungary', 'Yugoslavia', 'Poland']
    southern_europe = ['Greece', 'Portugal', 'Italy']
    western_europe = ['Scotland', 'Germany', 'Ireland', 'England', 'Holand-Netherlands', 'France']
    southern_africa = ['South']
    east_asia = ['China', 'Japan', 'Thailand', 'Cambodia', 'Philippines', 'Taiwan', 'India', 'Hong', 'Vietnam', 'Laos', 'Outlying-US(Guam-USVI-etc)']
    middle_east = ['Iran']

    if gen_level == 0:
        return value

    if gen_level == 1:
        if value == '?':
            return '?'
        if value in south_america:
            return 'South-America'
        if value in north_america:
            return 'North-America'
        if value in central_america:
            return 'Central-America'
        if value in eastern_europe:
            return 'Eastern-Europe'
        if value in southern_europe:
            return 'Southern-Europe'
        if value in western_europe:
            return 'Western-Europe'
        if value in southern_africa:
            return 'Southern-Africa'
        if value in east_asia:
            return 'East-Asia'
        if value in middle_east:
            return 'Middle-East'
    
    if gen_level == 2:
        if value == '?':
            return '?'
        if value in south_america or value in central_america or value in north_america:
            return 'America'
        if value in eastern_europe or value in western_europe or value in southern_europe:
            return 'Europe'
        if value in southern_africa:
            return 'Africa'
        if value in east_asia or value in middle_east:
            return 'Asia'
    
    if gen_level == 3:
        return 'World'

    raise ValueError(value, gen_level)

adult_generalize_country_rule = GeneralizationRule(adult_generalize_country, 3)

def adult_generalize_occupation(value, gen_level):
    if gen_level == 0:
        return value

    if gen_level == 1:
        if value == '?':
            return '?'
        if value in ['Tech-support', 'Exec-managerial', 'Adm-clerical', 'Handlers-cleaners', 'Other-service', 'Prof-specialty', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']:
            return 'Tertiary-Sector'
        if value in ['Craft-repair', 'Machine-op-inspct', 'Sales']:
            return 'Secondary-Sector'
        if value in ['Farming-fishing', 'Transport-moving']:
            return 'Primary-Sector'
    
    if gen_level == 2:
        return 'Profession'
    
    raise ValueError(value, gen_level)

adult_generalize_occupation_rule = GeneralizationRule(adult_generalize_occupation, 2)

def adult_generalize_education(value, gen_level):
    if gen_level == 0:
        return value

    if gen_level == 1:
        if value == '?':
            return '?'
        if value in ['Some-college', 'Bachelors', 'Masters', 'Doctorate']:
            return 'College'
        if value in ['HS-grad', '9th', '10th', '11th', '12th']:
            return 'High-School'
        if value in ['7th-8th', '5th-6th']:
            return 'Middle-School'
        if value in ['1st-4th', 'Preschool']:
            return 'Elementary-School'
        if value in ['Prof-school', 'Assoc-acdm', 'Assoc-voc']:
            return 'Professional-Dev'
    
    if gen_level == 2:
        return 'Instruction'
    
    raise ValueError(value, gen_level)

adult_generalize_education_rule = GeneralizationRule(adult_generalize_education, 2)

def adult_generalize_marital_status(value, gen_level):
    if gen_level == 0:
        return value

    if gen_level == 1:
        if value == '?':
            return '?'
        if value in ['Married-civ-spouse', 'Married-AF-spouse', 'Married-spouse-absent']:
            return 'Has-Spouse'
        if value in ['Divorced', 'Separated', 'Widowed' ]:
            return 'Had-Spouse'
        if value in ['Never-married']:
            return 'No-Spouse'

    if gen_level == 2:
        return 'Human'
    
    raise ValueError(value, gen_level)

adult_generalize_marital_status_rule = GeneralizationRule(adult_generalize_marital_status, 2)


def adult_generalize_workclass(value, gen_level):
    if gen_level == 0:
        return value

    if gen_level == 1:
        if value == '?':
            return '?'
        if value in ['Self-emp-not-inc', 'Self-emp-inc']:
            return 'Self-Employed'
        if value in ['Federal-gov', 'Local-gov', 'State-gov']:
            return 'Government'
        if value in ['Without-pay', 'Never-worked']:
            return 'Not-Employed'
        if value in ['Private']:
            return 'Private'

    if gen_level == 2:
        return 'Workforce'
    
    raise ValueError(value, gen_level)

adult_generalize_workclass_rule = GeneralizationRule(adult_generalize_workclass, 2)

def adult_generalize_relationship(value, gen_level):
    if gen_level == 0:
        return value
    if gen_level == 1:
        if value in ['Wife', 'Own-child', 'Husband', 'Other-relative', 'Unmarried']:
            return 'Family'
        if value in ['Not-in-family']:
            return 'Alone'
    if gen_level == 2:
        return 'Relationship'
    
    raise ValueError(value, gen_level)

adult_generalize_relationship_rule = GeneralizationRule(adult_generalize_relationship, 2)

def suppress(value, gen_level):
    if gen_level == 0:
        return value
    if gen_level == 1:
        return None
    
    raise ValueError(value, gen_level)

suppress_rule = GeneralizationRule(suppress, 1)

def mimic_generalize_insurance(value, gen_level):
    if gen_level == 0:
        return value
    if gen_level == 1:
        if value in ['Private', 'Self Pay']:
            return 'Not Public'
        if value in ['Medicare', 'Medicaid', 'Government']:
            return 'Public'
        if value in ['?']: 
            return '?'

    if gen_level == 2:
        return 'Insurance'

    raise ValueError(value, gen_level)

mimic_generalize_insurance_rule = GeneralizationRule(mimic_generalize_insurance, 2)

def mimic_generalize_dob(value, gen_level):
    if gen_level == 0:
        return value
    if gen_level == 1:
        if value <= 1950:
            return 'quite old'
        if value > 1950 and value <= 1980:
            return 'a bit old'
        if value > 1980 and value <= 2000:
            return 'young'
        if value > 2000:
            return 'very young'

    if gen_level == 2:
        return 'dob'

    raise ValueError(value, gen_level)

mimic_generalize_dob_rule = GeneralizationRule(mimic_generalize_dob, 2)

def mimic_generalize_proc(value, gen_level):
    if gen_level == 0:
        return value
    if gen_level == 1:
        # first digit in ICD9 code
        return value[0]

    if gen_level == 2:
        return 'proc'

    raise ValueError(value, gen_level)

mimic_generalize_proc_rule = GeneralizationRule(mimic_generalize_proc, 2)

def mimic_generalize_marital_status(value, gen_level):
    if gen_level == 0:
        return value
    if gen_level == 1:
        if value in ['MARRIED', 'LIFE PARTNER']:
            return 'HAS PARTNER'
        if value in ['SINGLE', 'DIVORCED', 'WIDOWED', 'SEPARATED']:
            return 'NO PARTNER'
        if value in ['?', 'UNKNOWN (DEFAULT)']: 
            return '?'

    if gen_level == 2:
        return 'Marital status'

    raise ValueError(value, gen_level)

mimic_generalize_marital_status_rule = GeneralizationRule(mimic_generalize_marital_status, 2)
