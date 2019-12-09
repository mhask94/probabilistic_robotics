import numpy as np
from itertools import permutations as perms

class POMDP():
    def __init__(self, pz, pu3, lava_f, lava_b, door, act, samples, prune=True):
        self.meas_p1 = np.diag([pz, 1-pz])
        self.meas_p2 = self.meas_p1[::-1,::-1]
        self.prediction = np.array([[1-pu3, pu3],[pu3, 1-pu3]])
        step = 1 / samples
        prob = np.arange(0,1+step,step)
        self.prob = np.block([[prob],[prob[::-1]]])
        self.value_lines = np.zeros((1,2))
        self.cost_lines = np.array([[lava_f, door],[door, lava_b]])
        self.value_fn = np.zeros(samples+1)
        self.act_cost = act
        self.prune_on = prune

    def prune(self):
        res = self.value_lines @ self.prob
        self.value_fn = np.max(res, axis=0)
        ind = np.argmax(res, axis=0)
        uniq = np.unique(ind)
        self.value_lines = self.value_lines[uniq]

        if len(uniq) > 2:
            self.prob_turn_start = self.prob[0,np.where(ind!=ind[0])[0][0]]
            self.prob_turn_end = self.prob[0,np.where(ind!=ind[-1])[0][-1]]

    def observe(self):
        num = len(self.value_lines)
        res1 = self.value_lines @ self.meas_p1
        res2 = self.value_lines @ self.meas_p2
        pad = [(ii,ii) for ii in range(num)]
        ind = np.array(list(perms(range(num), 2)) + pad).T
        self.value_lines = res1[ind[0]] + res2[ind[1]]
        if self.prune_on:
            self.prune()
        return self.value_lines

    def predict(self):
        temp = self.value_lines @ self.prediction
        temp += self.act_cost
        self.value_lines = np.array([*self.cost_lines, *temp])
        if self.prune_on:
            self.prune()
        return self.value_lines
