from sys import getsizeof as size

from .config import *
from .powers import *
from .weights import *


class Cell:
    def __init__(self):
        self.times_chosen = 0
        self.times_chosen_since_new = 0
        self.times_seen = 0
        self.recompute_score()

    def __sizeof__(self):
        return (
            size(self.times_chosen)
            + size(self.times_chosen_since_new)
            + size(self.times_seen)
            + size(self.score)
            + size(self.ram)
            + size(self.reward)
            + size(self.length)
        )

    def cntscore(self, a):
        w = getattr(Weights, a)
        p = getattr(Powers, a)
        v = getattr(self, a)
        return w / (v + e1) ** p + e2

    def cellscore(self):
        return (
            self.cntscore("times_chosen")
            + self.cntscore("times_chosen_since_new")
            + self.cntscore("times_seen")
            + 1
        )

    def recompute_score(self):
        self.score = self.cellscore()

    def visit(self):
        self.times_seen += 1
        self.recompute_score()
        return self.times_seen == 1

    def choose(self):
        self.times_chosen += 1
        self.times_chosen_since_new += 1
        return self.ram, self.reward, self.length

    def beats(self, cell):
        return (
            self.reward > cell.reward
            or self.reward == cell.reward
            and self.length < cell.length
        )

    def setstate(self, state):
        self.ram, self.reward, self.length = state
        self.reset_selection_counts()

    def led_to_improvement(self):
        self.reset_selection_counts()

    def reset_selection_counts(self):
        self.times_chosen = 0
        self.times_chosen_since_new = 0
        self.recompute_score()

    def save(self):
        return {
            "reward": self.reward,
            "selection": {
                "score": self.score,
                "counts": {
                    "times chosen": self.times_chosen,
                    "times chosen since new": self.times_chosen_since_new,
                    "times seen": self.times_seen,
                },
            },
        }

    def load(self, info):
        self.reward = info["reward"]
        self.score = info["selection"]["score"]

        self.times_seen = info["selection"]["counts"]["times seen"]
        self.times_chosen = info["selection"]["counts"]["times chosen"]
        self.times_chosen_since_new = info["selection"]["counts"][
            "times chosen since new"
        ]
