import numpy as np

class MDPPlanner():
    def __init__(self, walls, obs, goal):
        self.V = (-100*walls) + (-5000*obs) + (1e5*goal)
        self.absolve = (walls + obs + goal) != 0
        self.pi = np.zeros_like(obs)
        self.count = 0

    def update(self):
        residual = 0
        for i in range(2, len(self.V)-2):
            for j in range(2, len(self.V[0])-2):
                if self.absolve[i,j]:
                    continue
                else:
                    rewards = np.empty(4)
                    weights = np.array([0.1, 0.8, 0.1]) # left, straight, right
                    # move north action
                    ri = np.array([i,   i-1, i])
                    ci = np.array([j-1, j,   j+1])
                    rewards[0] = np.sum(self.V[ri,ci] * weights)
                    # move east action
                    ri = np.array([i-1, i,   i+1])
                    ci = np.array([j,   j+1, j])
                    rewards[1] = np.sum(self.V[ri,ci] * weights)
                    # move south action
                    ri = np.array([i,   i+1, i])
                    ci = np.array([j+1, j,   j-1])
                    rewards[2] = np.sum(self.V[ri,ci] * weights)
                    # move west action
                    ri = np.array([i+1, i,   i-1])
                    ci = np.array([j,   j-1, j])
                    rewards[3] = np.sum(self.V[ri,ci] * weights)
                    
                    v = np.max(rewards)
                    residual += np.abs(self.V[i,j] - v)

                    self.V[i,j] = v
                    self.pi[i,j] = np.argmax(rewards)

        self.count += 1
        return residual
