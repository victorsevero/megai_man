import numpy as np
from env import make_env

EXPLORATION_PARAM = 0.005


class Node:
    def __init__(self, value=-np.inf, children=None):
        self.value = value
        self.visits = 0
        self.children = {} if children is None else children

    def __repr__(self):
        return "<Node value=%f visits=%d len(children)=%d>" % (
            self.value,
            self.visits,
            len(self.children),
        )


def select_actions(root, action_space, max_episode_steps, rng):
    node = root

    acts = []
    steps = 0
    while steps < max_episode_steps:
        if node is None:
            # we've fallen off the explored area of the tree, just select random actions
            act = action_space.sample()
        else:
            epsilon = EXPLORATION_PARAM / np.log(node.visits + 2)
            if rng.random() < epsilon:
                # random action
                act = action_space.sample()
            else:
                # greedy action
                act_value = {}
                for act in range(action_space.n):
                    if node is not None and act in node.children:
                        act_value[act] = node.children[act].value
                    else:
                        act_value[act] = -np.inf
                best_value = max(act_value.values())
                best_acts = [
                    act
                    for act, value in act_value.items()
                    if value == best_value
                ]
                act = rng.choice(best_acts)

            if act in node.children:
                node = node.children[act]
            else:
                node = None

        acts.append(act)
        steps += 1

    return acts


def rollout(env, acts):
    total_rew = 0
    env.reset()
    steps = 0
    for act in acts:
        _, rew, terminated, truncated, _ = env.step(act)
        steps += 1
        total_rew += rew
        if terminated or truncated:
            break

    return steps, total_rew


def update_tree(root, executed_acts, total_rew):
    root.value = max(total_rew, root.value)
    root.visits += 1
    new_nodes = 0

    node = root
    for act in executed_acts:
        if act not in node.children:
            node.children[act] = Node()
            new_nodes += 1
        node = node.children[act]
        node.value = max(total_rew, node.value)
        node.visits += 1

    return new_nodes


class Brute:
    def __init__(self, env, max_episode_steps, seed=666):
        self.node_count = 1
        self._root = Node()
        self._env = env
        self._max_episode_steps = max_episode_steps
        self._rng = np.random.default_rng(seed)

    def run(self):
        acts = select_actions(
            self._root,
            self._env.action_space,
            self._max_episode_steps,
            self._rng,
        )
        steps, total_rew = rollout(self._env, acts)
        executed_acts = acts[:steps]
        self.node_count += update_tree(self._root, executed_acts, total_rew)
        return executed_acts, total_rew


def main(max_episode_steps=4500, timestep_limit=100_000_000):
    env_kwargs = {
        "state": "CutMan",
        "frameskip": 4,
        "truncate_if_no_improvement": True,
        "obs_space": "screen",
        "action_space": "discrete",
        "crop_img": False,
        "invincible": True,
        "render_mode": None,
        "damage_terminate": False,
        "fixed_damage_punishment": 1,
        "forward_factor": 0.1,
        "backward_factor": 0.11,
    }
    env = make_env(**env_kwargs)

    brute = Brute(env, max_episode_steps=max_episode_steps)
    timesteps = 0
    best_rew = float("-inf")
    while True:
        acts, rew = brute.run()
        timesteps += len(acts)

        if rew > best_rew:
            print(f"new best reward {best_rew} => {rew}")
            best_rew = rew
            env.unwrapped.record_movie("brute.bk2")
            env.reset()
            for act in acts:
                env.step(act)
            env.unwrapped.stop_record()

        if timesteps > timestep_limit:
            print("timestep limit exceeded")
            break


if __name__ == "__main__":
    main()
