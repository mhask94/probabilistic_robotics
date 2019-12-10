import numpy as np
from numpy.random import rand

class Simulator():
    def __init__(self, prob1, prob2, pz, pu3, lava_f, lava_b, door, u3_cost):
        self.prob1, self.prob2 = prob1, prob2
        self.state = 1      # face lava initially
        self.belief = 0.6   # initial p1
        self.pz = pz
        self.pu3 = pu3
        self.running = True
        self.num_turns = 0
        self.reward = 0
        self.lava_f = lava_f
        self.lava_b = lava_b
        self.door = door
        self.u3_cost = u3_cost

    def run(self):
        while self.running:
            self.observe()
            self.act()
        return self.win

    def observe(self):
        z_prob = rand() < self.pz
        if z_prob: z = self.state
        else: z = 1 - self.state
        if z:
            self.belief = self.pz * self.belief / ((2*self.pz - 1) * \
                    self.belief + (1-self.pz))
        else:
            self.belief = (1-self.pz) * self.belief / (self.pz - \
                    (2*self.pz - 1) * self.belief)

    def act(self):
        if self.belief < self.prob1: # forward -> u1
            if self.state:
                self.msg = 'Drove forwards into LAVA :('
                self.win = False
                self.reward += self.lava_f
            else:
                self.msg = 'Drove forwards into GOAL :)'
                self.win = True
                self.reward += self.door
            self.running = False
        elif self.belief > self.prob2: # backward -> u2
            if self.state:
                self.msg = 'Drove backward into GOAL :)'
                self.win = True
                self.reward += self.door
            else:
                self.msg = 'Drove backward into LAVA :('
                self.win = False
                self.reward += self.lava_b
            self.running = False
        else: # turn -> u3
            turn = rand() < self.pu3
            self.belief = self.pu3 - (2*self.pu3 - 1)*self.belief
            if turn:
                self.state = 1 - self.state
            self.num_turns += 1
            self.reward += self.u3_cost
