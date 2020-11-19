import argparse
import pdb
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import nearest_points

plt.ion()

class Env:
    def __init__(self, vertices=None, seed=None):
        if vertices:
            self.vertices = vertices
        else:
            self.vertices = np.array([
                [0, 0],
                [0, 1],
                [1, 1],
                [1, 0.35],
                [0.7, 0.35],
                [0.7, 0],
                [0, 0]
            ])

        self.polygon = Polygon(self.vertices)

        self.seed = seed
        self.state = None
        self.B = None
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self.state = np.array([[0.5], [0.5], [0]])
        self.B = np.diag(np.random.rand(3))
    
    def step(self, u):
        self.state = self.state + self.B @ u
        self.state[2] = self.state[2] % 2 * np.pi
        return self._measure()

    def render(self):
        fig, ax = plt.subplots()
        ax.plot(*self.polygon.exterior.xy)        
        ax.plot(self.state[0], self.state[1], "ro")
        ax.plot([self.state[0], self.state[0] + np.cos(self.state[2])], [self.state[1], self.state[1] + np.sin(self.state[2])])
        plt.show()

    def _measure(self, max_dist=100):
        x, y, theta = self.state.flatten()
        edges = LineString(self.polygon.exterior.coords)
        vision = LineString([(x, y), (x + max_dist * np.cos(theta), y + max_dist * np.sin(theta))])
        return np.array(edges.intersection(vision))



def main():
    env = Env()
    print(env.B)
    print(env.state)
    env.render()
    foo = env.step([[0.1], [0.1], [.1]])
    print(env.state)
    env.render()
    pdb.set_trace()

if __name__ == "__main__":
    main()