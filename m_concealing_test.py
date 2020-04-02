import unittest

import m_concealing as mc

class TestGroupRelease(unittest.TestCase):
    def test_one(self):
        records = [
            ['Black', 'Male', 'Rich', 'Diabetes'],
            ['Black', 'Male', 'Rich', 'Diabetes'],
            ['Black', 'Male', 'Poor', 'Diabetes'],
            ['White', 'Woman', 'Rich', 'Diabetes'],
            ['White', 'Woman', 'Poor', 'Diabetes'],
        ]
        ks = [(True, False, False, False), (True, False, True, False)]
        result = mc.group_release(records, ks)

        expected = {
            (True, False, False, False): {
                ('Black',): 
                    [
                        ['Black', 'Male', 'Rich', 'Diabetes'],
                        ['Black', 'Male', 'Rich', 'Diabetes'],
                        ['Black', 'Male', 'Poor', 'Diabetes'],
                    ],
                ('White',):
                    [
                        ['White', 'Woman', 'Rich', 'Diabetes'],
                        ['White', 'Woman', 'Poor', 'Diabetes'],
                    ]
            },
            (True, False, True, False): {
                ('Black', 'Rich'): 
                    [
                        ['Black', 'Male', 'Rich', 'Diabetes'],
                        ['Black', 'Male', 'Rich', 'Diabetes'],
                    ],
                ('Black', 'Poor'): 
                    [
                        ['Black', 'Male', 'Poor', 'Diabetes'],
                    ],
                ('White', 'Rich'): 
                    [
                        ['White', 'Woman', 'Rich', 'Diabetes'],
                    ],
                ('White', 'Poor'): 
                    [
                        ['White', 'Woman', 'Poor', 'Diabetes'],
                    ],
            },
        }
        self.assertEqual(result, expected)


class TestKnowledgeStates(unittest.TestCase):
    def test(self):
        states = mc.knowledge_states(5, [2,3], [1,4])
        expected = [
            (False, True, True, True, True),
            (False, True, True, True, False),
            (False, False, True, True, True),
            (False, False, True, True, False),
        ]

        self.assertEqual(states, expected)

class TestPKnowledgeState(unittest.TestCase):
    def test(self):
        k = [False, False, True, True, True]
        probs_knowing = [0, 0.01, 1, 1, 0.2]
        result = mc.p_knowledge_state(k, probs_knowing)

        self.assertEqual(result, 0.198)

class TestColValues(unittest.TestCase):
    def test(self):
        k = [False, False, True, True, True]
        record = ['Diabetes', 'HighIncome', 'White', 'Female', 'High Street 10']
        result = mc.col_values(record, k)
        expected = ('White', 'Female', 'High Street 10')

        self.assertEqual(result, expected)

class TestPReidInState(unittest.TestCase):
    def test(self):
        k = [True, False, True, True, False]
        record = ['Diabetes', 'HighIncome', 'White', 'Female', 'High Street 10']
        grouped_release = {
            ('Diabetes', 'White', 'Female'): [
                ['Diabetes', 'MiddleIncome', 'White', 'Female', 'Pleasance 20'],
                record,
                ['Diabetes', 'HighIncome', 'White', 'Female', 'Brighton Street 8'],
                ['Diabetes', 'LowIncome', 'White', 'Female', 'Lauriston Place 9'],
            ],
            ('BrokenLeg', 'White', 'Female'): [
                ['BrokenLeg', 'MiddleIncome', 'White', 'Female', 'Falcon Avenue 13'],
            ]
        }
        p_inclusion = 0.5
        result = mc.p_reid_in_state(record, p_inclusion, grouped_release, k)

        self.assertEqual(result, 1/4*p_inclusion)

class TestPKnowingEachAttribute(unittest.TestCase):
    def test(self):
        gen_rules = {2: None, 3: None}
        probs_knowing_sa = {1: 0.4, 4: 0.01}
        result = mc.p_knowing_each_attribute(5, gen_rules, probs_knowing_sa)
        expected = [0, 0.4, 1, 1, 0.01]

        self.assertEqual(result, expected)

class TestProbReid(unittest.TestCase):
    def test_first_ec(self):
        release = [
            ['Black', 'Male', 'Rich', 'Diabetes'],
            ['Black', 'Male', 'Rich', 'Diabetes'],
            ['Black', 'Male', 'Rich', 'Diabetes'],
            ['Black', 'Male', 'Poor', 'Diabetes'],
            ['Black', 'Male', 'Poor', 'Diabetes'],

            ['Black', 'Female', 'Rich', 'Diabetes'],
            ['Black', 'Female', 'Rich', 'Diabetes'],
            ['Black', 'Female', 'Rich', 'Diabetes'],
            ['Black', 'Female', 'Rich', 'Diabetes'],
            ['Black', 'Female', 'Rich', 'Diabetes'],
        ]
        probs_knowing_sa = {2: 0.1}
        gen_rules = {0: None, 1: None}
        result, _ = mc.prob_reid(release, 1, probs_knowing_sa, gen_rules)

        self.assertEqual(result, 0.23)

    def test_second_ec(self):
        release = [
            ['Black', 'Male', 'Rich', 'Diabetes'],
            ['Black', 'Male', 'Rich', 'Diabetes'],
            ['Black', 'Male', 'Rich', 'Diabetes'],
            ['Black', 'Male', 'Rich', 'Diabetes'],
            ['Black', 'Male', 'Rich', 'Diabetes'],

            ['Black', 'Female', 'Rich', 'Diabetes'],
            ['Black', 'Female', 'Rich', 'Diabetes'],
            ['Black', 'Female', 'Rich', 'Diabetes'],
            ['Black', 'Female', 'Poor', 'Diabetes'],
            ['Black', 'Female', 'Poor', 'Diabetes'],
        ]
        probs_knowing_sa = {2: 0.1}
        gen_rules = {0: None, 1: None}
        result, _ = mc.prob_reid(release, 1, probs_knowing_sa, gen_rules)

        self.assertEqual(result, 0.23)

if __name__ == '__main__':
    unittest.main()