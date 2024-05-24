from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TimeRemainingColumn,
)
from rich.traceback import install

from .archive import *
from .cell import *
from .config import *
from .exceptions import *
from .powers import *
from .termination import *
from .tree import *
from .utils import *
from .weights import *
from .wrappers import *

install()

import gzip
import inspect
import io
import json
import marshal
import os
import shutil
import sys
import tarfile
import types
from collections import defaultdict
from sys import getsizeof
from time import sleep

import numpy as np


class GoExplore:
    """
    GoExplore algorithm in the form of a class implementation.

    Parameters
    ----------
    env : GymSpecialWrapper
        Environment
    hashseed : int
        Python random seed for :func:`~hash()` function
    """

    metadata = {"method": ["ram", "trajectory"]}

    def __init__(self, env, hashseed=42):
        self.env = env
        self.report = (
            lambda: "Iterations: %d, Cells: %d, Frames: %d, Max Reward: %d"
            % (self.iterations, len(self.archive), self.frames, self.highscore)
        )
        self.status = (
            lambda delimiter=" ", separator=True: "Archive: %s, Trajectory: %s"
            % (
                prettysize(
                    self.archive,
                    delimiter=delimiter,
                    separator=separator,
                    sizefn=getsizeof,
                ),
                prettysize(
                    self.trajectory, delimiter=delimiter, separator=separator
                ),
            )
        )
        self.setseed(hashseed)

    @property
    def __dict__(self):
        return {
            "env": self.env.env_id,
            "seed": self.seed,
            "method": self.method,
            "repeat": self.repeat,
            "nsteps": self.nsteps,
            "highscore": self.highscore,
            "frames": self.frames,
            "iterations": self.iterations,
            "cellfn": inspect.getsource(self.cellfn),
            "hashfn": inspect.getsource(self.hashfn),
        }

    def setseed(self, seed):
        """Set PYTHONHASHSEED environment variable

        Parameters
        ----------
        seed : int
            Python random seed for :func:`~hash()` function
        """
        self.hashseed = seed
        if not os.environ.get("PYTHONHASHSEED"):
            os.environ["PYTHONHASHSEED"] = str(seed)
            os.execv(sys.executable, ["python3"] + sys.argv)

    def archivesize(self):
        """Returns the size of the archive

        Returns
        -------
        int
            Size of archive (number of cells)
        """
        return len(self.archive)

    def log(self, verbose, console=Console(), delimeter=" ", separator=True):
        """Logs basic information to the console

        * 0: no printout
        * 1: iterations/cells/frames/highscore
        * 2: 1 + archive memory size/trajectory memory size

        Parameters
        ----------
        verbose : int
            Verbosity of the printout, ranges from 0-2
        """
        if verbose == 1:
            console.print(self.report())
        elif verbose == 2:
            console.print(
                self.report() + ", " + self.status(delimeter, separator)
            )

    def ram(self):
        """Returns the RAM state of the environment emulator

        Returns
        -------
        numpy.ndarray
            RAM state

        """
        return self.env.unwrapped.em.get_state()

    def restore(self, cell):
        """Restores to a chosen cell

        Parameters
        ----------
        cell : Cell
            Cell for returning
        """
        ram, reward, length = cell.choose()
        self.reward = reward
        self.length = length
        self.trajectory.set(cell.node)
        self.env.reset(seed=self.seed)

        if self.method == "ram":
            self.env.unwrapped.em.set_state(ram)
        else:
            trajectory = self.trajectory.get_trajectory()
            while trajectory:
                action = trajectory.pop()
                observation, reward, *_ = self.env.step(action)
                yield observation

    def random(self):
        """Select an action randomly

        Returns
        -------
        int
            Random action selected from the environment action space
        """
        return self.env.action_space.sample()

    def getstate(self):
        return (
            self.ram() if self.method == "ram" else None,
            self.reward,
            self.length,
        )

    def initialize(
        self,
        cellfn=cellfn,
        hashfn=hashfn,
        repeat=0.95,
        nsteps=100,
        seed=42,
        method="ram",
        saveobs=False,
        **kwargs,
    ):
        """Initialize the algorithm.

        Parameters
        ----------
        cellfn : callable
            Function that returns cell given an observation.
        hashfn : callable
            Function that returns hash code given a cell.
        repeat : float
            Probability of repeating the previous action.
        nsteps : int
            Maximum duration of each iteration of exploration in emulator steps.
        seed : int
            Environment seed.
        method : str
            Method for return. Either 'ram' (default) or 'trajectory'.
        saveobs : bool
            Whether to save observations associated with each cell.
        """
        self.saveobs = saveobs
        self.cellfn = cellfn
        self.hashfn = hashfn
        self.repeat = repeat
        self.nsteps = nsteps
        self.method = method
        self.seed = seed

        ensure_type(repeat, float, "repeat", "action repeat probability")
        ensure_range(
            repeat,
            float,
            "repeat",
            "action repeat probability",
            0,
            1,
        )

        ensure_type(nsteps, int, "nsteps", "max explore duration")
        ensure_range(nsteps, int, "nsteps", "max explore duration", minn=1)
        ensure_from(method, self.metadata["method"], "method", "return method")

        observation, info = self.env.reset(seed=seed)

        cell = self.cellfn(observation)
        code = self.hashfn(cell)

        self.archive = Archive()
        self.reward = 0
        self.action = 0
        self.length = 0
        self.frames = 0
        self.highscore = 0
        self.discovered = 0
        self.iterations = 0
        self.trajectory = LinkedTree()

        cell = self.archive[code]

        if saveobs:
            cell.observation = observation

        cell.node = self.trajectory.node
        self.trajectory.node.assign(code)
        cell.visit()
        cell.setstate(self.getstate())
        self.restore_code = code

    def update(self, cell):
        """Determines whether or not to update a cell.

        Parameters
        ----------
        cell : Cell
            A preexisting cell from the archive.

        Returns
        -------
        bool
            Indication of whether or not the current emulator state improves upon the existing cell state.
        """
        new = cell.visit()
        return (
            new
            or self.reward > cell.reward
            or self.reward == cell.reward
            and self.length < cell.length
        )

    def act(self, render=False):
        """Perform one emulator step.

        Parameters
        ----------
        render : bool
            Whether or not to render the frame following the emulator step.

        Returns
        -------
        numpy.ndarray
            observation
        float
            reward
        bool:
            terminal
        dict:
            info
        """
        if np.random.random() > self.repeat:
            self.action = self.random()

        observation, reward, terminated, truncated, info = self.env.step(
            self.action
        )
        done = terminated or truncated
        self.reward += reward
        self.length += 1
        self.frames += 1
        self.highscore = max(self.highscore, self.reward)

        if render:
            self.env.render()

        if done:
            return observation, reward, terminated, truncated, info

        cell = self.cellfn(observation)
        code = self.hashfn(cell)
        cell = self.archive[code]

        self.trajectory.act(self.action)

        if self.update(cell):
            if hasattr(cell, "code"):
                cell.node.remove()

            cell.node = self.trajectory.node
            self.trajectory.node.assign(code)

            if self.saveobs:
                cell.observation = observation

            cell.setstate(self.getstate())
            self.discovered += 1

        return observation, reward, terminated, truncated, info

    def run(
        self,
        render=False,
        debug=False,
        delay=0.01,
        return_states=False,
        max_frames=np.inf,
        return_traj=False,
    ):
        """Run a full iteration of the algorithm.

        Parameters
        ----------
        render : bool
            Whether or not to render this iteration.
        debug : bool
            Whether or not to apply a delay to slow down rendering
        delay : float
            Delay in seconds between frames rendering in debug mode
        return_states : bool
            Whether to return the observations throughout this iteration
        return_traj : bool
            Whether to also return the frames emulated when returning in 'trajectory' mode

        Returns
        -------
        list
            list of numpy.ndarray observations, if return_states or return_traj is True
        NoneType
            otherwise
        """
        self.discovered = 0

        if return_states:
            observations = []

        for _ in range(self.nsteps):
            observation, reward, terminated, truncated, info = self.act(render)
            done = terminated or truncated

            if return_states:
                observations.append(observation)

            if done or self.frames == max_frames:
                break

            if debug:
                sleep(delay)

        if self.discovered:
            self.archive[self.restore_code].led_to_improvement()

        self.iterations += 1

        traj = self.go()

        if return_traj:
            return observations + traj

        if return_states:
            return observations

    def go(self):
        codes = [*self.archive]
        probs = np.array([cell.score for cell in self.archive.values()])
        probs = probs / probs.sum()

        restore_code = np.random.choice(codes, p=probs)
        restore_cell = self.archive[restore_code]
        self.restore_code = restore_code

        return list(self.restore(restore_cell))

    def run_for(
        self,
        duration,
        verbose=1,
        units="iterations",
        renderfn=lambda iteration: False,
        delimeter=" ",
        separator=True,
        debug=False,
        delay=0.01,
        desc="Running",
        **kwargs,
    ):
        """Run algorithm for a certain duration

        Parameters
        ----------
        duration : int
            Duration to run algorithm.
        verbose : int
            Verbosity of printout.
        units : str
            Units indicating how to interpret duration, ie. 'iterations' or 'frames'
        renderfn : callable
            Function that takes the current iteration and returns a boolean indicating whether to render.
        debug : bool
            Whether or not to apply a delay to slow down rendering
        delay : float
            Delay in seconds between frames rendering in debug mode
        desc : str
            Description for progress bar. Defaults to 'Running'.
        """
        ensure_type(verbose, int, "verbose", "logging verbosity")
        ensure_range(verbose, int, "verbose", "logging verbosity", 0, 2)
        ensure_from(
            units, ["iterations", "frames"], "units", "units of duration"
        )

        progress = Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            "{task.completed} of {task.total} %s" % units,
        )

        if units == "iterations":
            max_frames = np.inf
        else:
            max_frames = duration

        with progress:
            task = progress.add_task(desc, total=duration)
            iteration = 0

            while not progress.finished:
                render = renderfn(iteration)
                self.run(
                    render, debug=debug, delay=delay, max_frames=max_frames
                )

                if units == "iterations":
                    progress.advance(task)
                else:
                    progress.update(task, completed=self.frames, refresh=True)

                self.log(verbose, progress.console, delimeter, separator)
                iteration += 1

    def save(self, path):
        """Save all data representing the current state of the algorithm.

        * RAM states
        * PYTHONHASHSEED
        * Trajectories
        * Algorithm variables
        * Cell metadata, ie. counts and the like

        All of this data is stored as efficiently as possible,
        compressed in .tar.gz archives, and can be restored using
        the :func:`~load()` function.

        Parameters
        ----------
        path : str
            Path to save information. Creates the directory if it does not already exist.

        Returns
        -------
        bool
            Whether the operation was successful.
        """
        try:
            # Make the save directory if it does not already exist
            if not os.path.exists(path):
                os.mkdir(path)

            # Write PYTHONHASHSEED to file
            with open(os.path.join(path, ".PYTHONHASHSEED"), "w") as f:
                f.write(str(self.hashseed))

            states = os.path.join(path, "ram")
            trajectories = os.path.join(path, "trajectory")

            # Make directories
            os.mkdir(states)
            os.mkdir(trajectories)

            # Save emulator ram states and action trajectories to compressed .npy.gz archives
            # Collect archive data
            data = {}
            for code, cell in self.archive.items():
                with gzip.GzipFile(
                    os.path.join(states, f"{code}.npy.gz"), "w"
                ) as f:
                    np.save(f, cell.ram)

                with gzip.GzipFile(
                    os.path.join(trajectories, f"{code}.npy.gz"), "w"
                ) as f:
                    np.save(
                        f, np.array(self.trajectory.get_trajectory(cell.node))
                    )
                data[code] = cell.save()

            # Save archive data
            with open(os.path.join(path, "exploration.json"), "w") as f:
                json.dump(data, f)

            # Save vars
            with open(os.path.join(path, "vars.json"), "w") as f:
                json.dump(vars(self), f)

            # Make .tar.gz archives for directories
            shutil.make_archive(os.path.join(path, "ram"), "gztar", states)
            shutil.make_archive(
                os.path.join(path, "trajectory"), "gztar", trajectories
            )

            # Remove directories
            shutil.rmtree(states)
            shutil.rmtree(trajectories)

            return True
        except:
            return False

    def load(self, path, overwrite=True):
        """Restores data saved with :func:`~save()`

        Parameters
        ----------
        path : str
            Path to save information. Creates the directory if it does not already exist.
        overwrite : bool
            Whether to overwrite completely or merge with existing data

        Returns
        -------
        bool
            Whether the operation was successful.
        """
        if overwrite:
            self.archive = Archive()
            self.trajectory = LinkedTree()

        try:
            # Load archive data
            with open(os.path.join(path, "exploration.json"), "r") as f:
                archive = json.load(f)

            # Load vars
            with open(os.path.join(path, "vars.json"), "r") as f:
                meta = json.load(f)

                # NOTE: cellfn and hashfn loading disable because I really
                # don't know how to do this properly
                # self.cellfn = types.FunctionType(
                #     marshal.loads(meta["cellfn"].encode()), globals(), "cellfn"
                # )
                # self.hashfn = types.FunctionType(
                #     marshal.loads(meta["hashfn"]), globals(), "hashfn"
                # )

                self.repeat = meta["repeat"]
                self.nsteps = meta["nsteps"]
                self.frames = meta["frames"]
                self.method = meta["method"]

                self.iterations = meta["iterations"]
                self.highscore = meta["highscore"]

                if self.env.env_id != meta["env"]:
                    self.close()
                    self.env = name2env[meta["env"]]

                # NOTE: this used the old gym API, we set seed at reset now
                # self.env.seed(meta["seed"])
                self.seed = meta["seed"]

            # Update archive with file data
            for code, info in archive.items():
                self.archive[int(code)].load(info)

            # Update PYTHONHASHSEED
            with open(os.path.join(path, ".PYTHONHASHSEED"), "r") as f:
                self.setseed(int(f.read()))

            # Load emulator ram states to their corresponding cells
            with tarfile.open(os.path.join(path, "ram.tar.gz"), "r:gz") as tar:
                for member in tar.getmembers():
                    f = tar.extractfile(member)
                    if f is not None:
                        code = int(member.name[2:-7])
                        data = gzip.decompress(f.read())
                        stream = io.BytesIO(data)
                        ram = np.load(stream)
                        self.archive[code].ram = ram

            # Build trajectory tree
            with tarfile.open(
                os.path.join(path, "trajectory.tar.gz"), "r:gz"
            ) as tar:
                for member in tar.getmembers():
                    f = tar.extractfile(member)
                    if f is not None:
                        code = int(member.name[2:-7])
                        data = gzip.decompress(f.read())
                        stream = io.BytesIO(data)
                        trajectory = np.load(stream)
                        self.archive[code].length = len(trajectory)
                        self.archive[code].node = self.trajectory.add(
                            trajectory, code
                        )

            return True
        except:
            return False

    def refresh(self, method="ram"):
        """Refresh the archive"""
        ensure_from(
            method, ["ram", "obs"], "method", "observation fetching method"
        )
        if method == "obs" and not self.saveobs:
            raise ValueError(
                "Cannot use 'obs' observation fetching method when saveobs is False"
            )

        new = Archive()
        for old_code, cell in self.archive.items():
            if method == "ram":
                self.env.restore_full_state(cell.ram)
                new_code = self.hashfn(self.cellfn(self.env.step(0)[0]))
            elif method == "obs":
                new_code = self.hashfn(self.cellfn(cell.observation))
            if new_code in new:
                if cell.beats(new[new_code]):
                    cell.node.assign(new_code)
                    new[new_code] = cell
            else:
                cell.node.assign(new_code)
                new[new_code] = cell
        self.archive = new
        self.go()

    def close(self):
        """Close the environment"""
        self.env.close()
