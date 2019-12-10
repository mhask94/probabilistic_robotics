import numpy as np
import matplotlib.pyplot as plt
from pomdp import POMDP
from simulator import Simulator

time_horizon = 20

pz = 0.7
pu3 = 0.8

lava_cost_f = -100
lava_cost_b = -50 
door_cost = 100
u3_cost = -1

samples = 1000

planner = POMDP(pz, pu3, lava_cost_f, lava_cost_b, door_cost, u3_cost, samples, 
        prune=True)

for T in range(time_horizon):
    obs = planner.observe()
    pred = planner.predict()

#print(pred)

book_lines = np.array([[door_cost, lava_cost_f],[lava_cost_b, door_cost]])
plt.figure()
plt.plot([0, 1], book_lines.T, 'r--')
plt.plot([planner.prob_turn_start, planner.prob_turn_start], [-500, 500], 'g:')
plt.plot([planner.prob_turn_end, planner.prob_turn_end], [-500, 500], 'g:')
plt.plot(planner.prob[0], planner.value_fn, 'b')
plt.title('Value Function')
plt.xlabel('Probability of $x_1$')
plt.ylabel('Value')
plt.axis([0, 1, -100, 100])
plt.pause(0.00001)

### simulate 10 times
wins = 0
for T in range(10):
    sim = Simulator(planner.prob_turn_start, planner.prob_turn_end, pz, pu3, 
            lava_cost_f, lava_cost_b, door_cost, u3_cost)
    wins += sim.run()
    print('Sim {}: {}\tactions:{}\treward: {}'.format(T, sim.msg, sim.num_turns,
        sim.reward))
print('Won {}/10 times'.format(wins))

plt.show()
