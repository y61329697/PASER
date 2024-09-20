class IEMOCAP_Meter:
    """Computes and stores the current best value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.WA = 0.
        self.UA = 0.
        self.WA_male = 0.
        self.UA_male = 0.
        self.test_WA_male = 0.
        self.test_UA_male = 0.
        self.WA_female = 0.
        self.UA_female = 0.
        self.test_WA_female = 0.
        self.test_UA_female = 0.

    def update(self, WA, UA, WA_male, UA_male, WA_female, UA_female):

        if UA > self.UA:
            self.UA = UA
        if WA > self.WA:
            self.WA = WA
            
        if UA_male > self.UA_male:
            self.UA_male = UA_male
            # when the performance of UA on the one speaker is the best
            # test on the other speaker
            self.test_UA_female = UA_female
            self.test_WA_female = WA_female
            
        if WA_male > self.WA_male:
            self.WA_male = WA_male
            
        if UA_female > self.UA_female:
            self.UA_female = UA_female
            # when the performance of UA on the one speaker is the best
            # test on the other speaker
            self.test_UA_male = UA_male
            self.test_WA_male = WA_male
        if WA_female > self.WA_female:
            self.WA_female = WA_female