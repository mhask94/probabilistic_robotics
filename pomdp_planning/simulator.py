import numpy as np
from numpy.random import rand

class Simulator():
    def __init__(self, prob1, prob2, pz_prob, pu3_prob):
        self.prob1, self.prob2 = prob1, prob2
        self.state = 1      # face lava initially
        self.belief = 0.6   # initial p1
        self.pz_prob = pz_prob
        self.pu3_prob = pu3_prob

    def run(self):
        self.observe()
        self.act()

    def observe(self):
        z = rand() < self.pz_prob
        if z:
            self.belief = self.pz_prob*self.belief / ()
