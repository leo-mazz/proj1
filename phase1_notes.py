import numpy as np
import pandas as pd
import dataset.mimic as mimic


notes = mimic.get_notes()

def part1():
    """ Identify patients from nurse notes """
    nurse_notes = notes.loc[notes['CATEGORY'] == 'Nursing']

    texts = {}

    for _, n in nurse_notes.iterrows():
        t = n['TEXT'][0:20]
        subject = n['SUBJECT_ID']

        if t in texts.keys():
            texts[t].add(subject)
        else:
            texts[t] = set([subject])

    lengths = [len(subjects)for subjects in texts.values()]

    max = np.max(lengths)
    print('max', max)
    print('min', np.min(lengths))
    print('mean', np.mean(lengths))

    freq_lenghts = {}
    for i in range(max+1):
        freq_lenghts[i] = lengths.count(i)

    print('freq_lenghts', freq_lenghts)

    print('num_subjects', np.sum(lengths))


def part2():
    """ Look for quasi-identifier values in free text """
    def count_strings_in_notes(*strings):
        finds = pd.concat(
            [notes[notes['TEXT'].str.contains(s, case=False)] for s in strings]
        ,ignore_index=True).drop_duplicates().reset_index(drop=True)

        return len(finds)

    print('age', count_strings_in_notes('year old', 'year-old', 'years old', 'years-old'))
    print('sex', count_strings_in_notes('she', 'he', 'male', 'female', 'man', 'woman'))
    print('ethnicity', count_strings_in_notes('white man', 'white male', 'white female', 'white woman' 'black male', 'black man', 'black female', 'black woman', 'african american', 'native american', 'asian male', 'asian man', 'asian female', 'asian woman', 'caucasian', 'hispanic', 'indian male', 'indian man', 'indian female', 'indian woman', 'arabic man', 'arabic woman', 'arabic male', 'arabic female'))
    print('religion', count_strings_in_notes('religion', 'catholic', 'protestant', 'muslim', 'jewish', 'buddhist', 'hindu'))
    print('sex_orientation', count_strings_in_notes('heterosexual', 'cisgender', 'homosexual', 'gay', 'lesbian'))

    print('total_notes', len(notes))


part1()
part2()