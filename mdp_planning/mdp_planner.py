import numpy as np

from IPython.core.debugger import set_trace

class MDPPlanner():
    def __init__(self, walls, obs, goal, ww=-100, wo=-5000, wg=1e5, wm=-2):
        self.V = (ww*walls) + (wo*obs) + (wg*goal)
        self.wm = wm
        self.absolve = (walls + obs + goal) != 0
        self.pi = np.zeros_like(obs)
        self.count = 0
        self.gam = 1#0.99
        r,c = obs.shape
        self.rewards = np.empty((4, *obs[2:-2,2:-2].shape))

    def update(self):
        residual = 0

        for j in range(2, len(self.V[0])-2):
            for i in range(2, len(self.V)-2):
                if self.absolve[i,j]:
                    continue
                else:
                    set_trace()
                    dirs = np.empty(self.rewards.shape)
                    dirs[0] = self.V[1:-3, 2:-2] # north
                    dirs[1] = self.V[2:-2, 3:-1] # east
                    dirs[2] = self.V[3:-1, 2:-2] # south
                    dirs[3] = self.V[2:-2, 1:-3] # west

                    weights = np.array([0.1, 0.8, 0.1]).reshape(-1,1,1) 
                    
                    # north
                    idx = np.array([3,0,1])
                    self.rewards[0] = np.sum(weights * dirs[idx], 0) + self.wm*\
                            ~self.absolve[1:-3, 2:-2]
                    # east
                    idx = np.array([0,1,2])
                    self.rewards[1] = np.sum(weights * dirs[idx], 0) + self.wm*\
                            ~self.absolve[2:-2, 3:-1]
                    # south
                    idx = np.array([1,2,3])
                    self.rewards[2] = np.sum(weights * dirs[idx], 0) + self.wm*\
                            ~self.absolve[3:-1, 2:-2]
                    # west 
                    idx = np.array([2,3,0])
                    self.rewards[3] = np.sum(weights * dirs[idx], 0) + self.wm*\
                            ~self.absolve[2:-2, 1:-3]

                    v = np.max(self.rewards, axis=0)

                    mask = ~self.absolve[2:-2, 2:-2]
                    residual = np.sum(np.abs(self.V[2:-2, 2:-2] - v)[mask])

                    self.V[2:-2, 2:-2][mask] = self.gam * v[mask]
                    self.pi[2:-2, 2:-2][mask] = np.argmax(self.rewards, \
                            axis=0)[mask]

#                    rewards = np.empty(4)
#                    weights = np.array([0.1, 0.8, 0.1]).reshape(-1,1,1) 
#                    # move north action
#                    ri = np.array([i,   i-1, i])
#                    ci = np.array([j-1, j,   j+1])
#                    rewards[0] = np.sum(self.V[ri,ci]*weights) + \
#                            self.wm * (not self.absolve[i-1,j])
#                    # move east action
#                    ri = np.array([i-1, i,   i+1])
#                    ci = np.array([j,   j+1, j])
#                    rewards[1] = np.sum(self.V[ri,ci]*weights) + \
#                            self.wm * (not self.absolve[i,j+1])
#                    # move south action
#                    ri = np.array([i,   i+1, i])
#                    ci = np.array([j+1, j,   j-1])
#                    rewards[2] = np.sum(self.V[ri,ci]*weights) + \
#                            self.wm * (not self.absolve[i+1,j])
#                    # move west action
#                    ri = np.array([i+1, i,   i-1])
#                    ci = np.array([j,   j-1, j])
#                    rewards[3] = np.sum(self.V[ri,ci]*weights) + \
#                            self.wm * (not self.absolve[i,j-1])
#                    
#                    v = np.max(rewards)
#                    residual += np.abs(self.V[i,j] - v)
#
#                    self.V[i,j] = self.gam * v
#                    self.pi[i,j] = np.argmax(rewards, axis=0)

        self.count += 1
        return residual
