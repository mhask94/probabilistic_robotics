import numpy as np

class MDPPlanner():
    def __init__(self, walls, obs, goal, ww=-100, wo=-5000, wg=1e5, wm=-2):
        self.V = (ww*walls) + (wo*obs) + (wg*goal)
        self.wm = wm
        self.absolve = (walls + obs + goal) != 0
        self.pi = np.zeros_like(obs)
        self.count = 0
        self.gam = 0.99

    def update(self):
        residual = 0

        for j in range(2, len(self.V[0])-2):
            for i in range(2, len(self.V)-2):
                if self.absolve[i,j]:
                    continue
                else:
                    rewards = np.empty(4)
                    weights = np.array([0.1, 0.8, 0.1]) # left, straight, right
                    # move north action
                    ri = np.array([i,   i-1, i])
                    ci = np.array([j-1, j,   j+1])
                    rewards[0] = np.sum(self.V[ri,ci]*weights) + \
                            self.wm * (not self.absolve[i-1,j])
                    # move east action
                    ri = np.array([i-1, i,   i+1])
                    ci = np.array([j,   j+1, j])
                    rewards[1] = np.sum(self.V[ri,ci]*weights) + \
                            self.wm * (not self.absolve[i,j+1])
                    # move south action
                    ri = np.array([i,   i+1, i])
                    ci = np.array([j+1, j,   j-1])
                    rewards[2] = np.sum(self.V[ri,ci]*weights) + \
                            self.wm * (not self.absolve[i+1,j])
                    # move west action
                    ri = np.array([i+1, i,   i-1])
                    ci = np.array([j,   j-1, j])
                    rewards[3] = np.sum(self.V[ri,ci]*weights) + \
                            self.wm * (not self.absolve[i,j-1])
                    
                    v = np.max(rewards)
                    residual += np.abs(self.V[i,j] - v)

                    self.V[i,j] = self.gam * v
                    self.pi[i,j] = np.argmax(rewards)

        self.count += 1
        return residual
