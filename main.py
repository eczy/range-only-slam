import argparse
import pdb
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import nearest_points

plt.ion()


class Env:
    def __init__(self, vertices=None, seed=None, dyn_awgn=None, obs_awgn=None):
        if vertices:
            self.vertices = vertices
        else:
            self.vertices = np.array(
                [[0, 0], [0, 1], [1, 1], [1, 0.35], [0.7, 0.35], [0.7, 0], [0, 0]]
            )

        self.polygon = Polygon(self.vertices)
        self.dyn_awgn = dyn_awgn
        self.obs_awgn = obs_awgn

        self.seed = seed
        self.perceived_state = None
        self.actual_state = None
        self.observation = None
        self.B = None
        self.reset()

    def reset(self):
        np.random.seed(self.seed)
        self.perceived_state = np.array([[0.5], [0.5], [0]])
        self.actual_state = np.array([[0.5], [0.5], [0]])
        self.observation = self._measure()
        self.B = np.diag((np.random.normal(1, 0.05, 3)))

    def step(self, u):
        self.perceived_state = self.perceived_state + self.B @ u
        if self.dyn_awgn:
            self.perceived_state += np.random.normal(*self.dyn_awgn, size=(2, 1))
        self.actual_state = self.actual_state + u
        for state in [self.perceived_state, self.actual_state]:
            state[2] = state[-1] % (2 * np.pi)
        self.observation = self._measure()

    def render(self, ax=None):
        if not ax:
            fig, ax = plt.subplots()
        ax.plot(*self.polygon.exterior.xy, "b-")
        ax.plot(self.perceived_state[0], self.perceived_state[1], "ro")
        ax.plot(
            [
                self.perceived_state[0],
                self.perceived_state[0] + np.cos(self.perceived_state[2]),
            ],
            [
                self.perceived_state[1],
                self.perceived_state[1] + np.sin(self.perceived_state[2]),
            ],
        )

        ax.plot(self.actual_state[0], self.actual_state[1], "bo")
        ax.plot(
            [self.actual_state[0], self.actual_state[0] + np.cos(self.actual_state[2])],
            [self.actual_state[1], self.actual_state[1] + np.sin(self.actual_state[2])],
        )
        plt.show()

    def _measure(self, max_dist=100):
        x, y, theta = self.actual_state.flatten()
        edges = LineString(self.polygon.exterior.coords)
        vision = LineString(
            [(x, y), (x + max_dist * np.cos(theta), y + max_dist * np.sin(theta))]
        )
        actual_intercepts = np.array(edges.intersection(vision))
        x, y, theta = self.perceived_state.flatten()
        edges = LineString(self.polygon.exterior.coords)
        vision = LineString(
            [(x, y), (x + max_dist * np.cos(theta), y + max_dist * np.sin(theta))]
        )
        perceived_intercepts = np.array(edges.intersection(vision))
        if self.obs_awgn:
            self.perceived_intercepts += np.random.normal(*self.obs_awgn, size=2)
        return actual_intercepts, perceived_intercepts


def sweep(env, n_points=100):
    interval = (np.pi * 2) / n_points
    print(interval)
    actual_observations = []
    perceived_observations = []
    for _ in range(n_points):
        actual_observations.append(env.observation[0])
        perceived_observations.append(env.observation[1])
        env.step(np.array([0, 0, interval]).reshape(-1, 1))
    return np.stack(actual_observations), np.stack(perceived_observations)


def plot_sweep(env, n_points=100):
    actual_observations, perceived_observations = sweep(env, n_points)
    fig, ax = plt.subplots()
    env.render(ax)
    ax.plot(actual_observations[:, 0], actual_observations[:, 1], "bo", alpha=0.5)
    ax.plot(perceived_observations[:, 0], perceived_observations[:, 1], "ro", alpha=0.5)
    plt.show()
    fig, ax = plt.subplots()
    actual_distances = [
        np.linalg.norm(x - env.actual_state[:2].flatten()) for x in actual_observations
    ]
    ax.plot(np.arange(len(actual_distances)), actual_distances, "b-", label="actual")
    perceived_distances = [
        np.linalg.norm(x - env.perceived_state[:2].flatten())
        for x in perceived_observations
    ]
    ax.plot(
        np.arange(len(perceived_distances)),
        perceived_distances,
        "r-",
        label="perceived",
    )
    plt.show()


def main():
    env = Env()
    plot_sweep(env)
    pdb.set_trace()


if __name__ == "__main__":
    main()
