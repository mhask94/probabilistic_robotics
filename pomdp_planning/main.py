import numpy as np
from itertools import permutations as perms
import matplotlib.pyplot as plt

from IPython.core.debugger import set_trace

# params
time_horizon = 2
prune_on = True

pz_true = 0.7
pz_false = 1 - pz_true
pu3_true = 0.8
pu3_false = 1 - pu3_true

lava_cost_f = -100
lava_cost_b = -50 
door_cost = 100
action_cost = -1

meas_p1 = np.diag([pz_true, pz_false])
meas_p2 = np.diag([pz_false, pz_true])
prediction = np.array([[pu3_false, pu3_true], [pu3_true, pu3_false]])

samples = 1000
step = 1 / samples

prob = np.empty((2, samples+1))
prob[0] = np.arange(0, 1+step, step)
prob[1] = prob[0,::-1]

eq = np.zeros((1,2))
eq_u1_u2 = np.array([[lava_cost_f, door_cost], [door_cost, lava_cost_b]])

val_fn = np.zeros(samples+1)

def prune(eq_in):
    res = eq_in @ prob
    val_fn = np.max(res, axis=0)
    ind = np.argmax(res, axis=0)
    uniq = np.unique(ind)
    eq_out = eq_in[uniq]
    return eq_out, val_fn, ind

def observe(eq_in):
    num = len(eq_in)
    res1 = eq_in @ meas_p1
    res2 = eq_in @ meas_p2
    pad = [(ii, ii) for ii in range(num)]
    ind = np.array(list(perms(range(num), 2)) + pad).T
    res = res1[ind[0]] + res2[ind[1]]
    return res

def predict(eq_in):
    temp = eq_in @ prediction
    temp += action_cost
    eq_out = np.array([*eq_u1_u2, *temp])
    return eq_out

for T in range(time_horizon):
    obs = observe(eq)
    if prune_on:
        obs, val_fn, ind = prune(obs)
    pred = predict(obs)
    if prune_on:
        eq, val_fn, ind = prune(pred)

book_lines = np.array([[100, -100],[-50, 100], [-1, -1]])
plt.figure()
plt.plot([0, 1], book_lines.T, 'r')
plt.plot(prob[0], val_fn, 'b')
plt.title('Value Function')
plt.xlabel('Probability of $x_1$')
plt.ylabel('Value')
plt.legend()
plt.axis([0, 1, -100, 100])
plt.pause(0.00001)
plt.show()
