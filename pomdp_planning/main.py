import numpy as np
from itertools import permutations as perms
import matplotlib.pyplot as plt
from pomdp import POMDP

from IPython.core.debugger import set_trace

# params
time_horizon = 20

pz_true = 0.7
pu3_true = 0.8

lava_cost_f = -100
lava_cost_b = -50 
door_cost = 100
u3_cost = -1

samples = 1000

planner = POMDP(pz_true, pu3_true, lava_cost_f, lava_cost_b, door_cost, u3_cost,
        samples, prune=True)

#meas_p1 = np.diag([pz_true, 1-pz_true])
#meas_p2 = np.diag([1-pz_true, pz_true])
#prediction = np.array([[1-pu3_true, pu3_true], [pu3_true, 1-pu3_true]])
#
#samples = 1000
#step = 1 / samples
#
#prob = np.empty((2, samples+1))
#prob[0] = np.arange(0, 1+step, step)
#prob[1] = prob[0,::-1]
#
#value_lines = np.zeros((1,2))
#cost_lines = np.array([[lava_cost_f, door_cost], [door_cost, lava_cost_b]])
#
#val_fn = np.zeros(samples+1)
#
#def prune(lines_in):
#    res = lines_in @ prob
#    value_function = np.max(res, axis=0)
#    ind = np.argmax(res, axis=0)
#    uniq = np.unique(ind)
#    lines_out = lines_in[uniq]
#    return lines_out, value_function, ind
#
#def observe(eq_in):
#    num = len(eq_in)
#    res1 = eq_in @ meas_p1
#    res2 = eq_in @ meas_p2
#    pad = [(ii, ii) for ii in range(num)]
#    ind = np.array(list(perms(range(num), 2)) + pad).T
#    res = res1[ind[0]] + res2[ind[1]]
#    return res
#
#def predict(eq_in):
#    temp = eq_in @ prediction
#    temp += u3_cost
#    eq_out = np.array([*eq_u1_u2, *temp])
#    return eq_out

for T in range(time_horizon):
    obs = planner.observe()
#    if prune_on:
#        obs, val_fn, ind = prune(obs)
    pred = planner.predict()
#    if prune_on:
#        eq, val_fn, ind = prune(pred)

print(pred)

book_lines = np.array([[100, -100],[-50, 100], [-1, -1]])
plt.figure()
plt.plot([0, 1], book_lines.T, 'r:')
plt.plot(planner.prob[0], planner.value_fn, 'b')
plt.title('Value Function')
plt.xlabel('Probability of $x_1$')
plt.ylabel('Value')
plt.axis([0, 1, -100, 100])
plt.pause(0.00001)
plt.show()
