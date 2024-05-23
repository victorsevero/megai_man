from pathlib import Path

from callbacks import StageLoggingCallback
from env import make_venv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


def train():
    n_envs = 8
    multi_input = True
    n_envs = 8
    frameskip = 4
    frame_stack = 2
    time_punishment_factor = 0
    env_kwargs = {
        "n_envs": n_envs,
        "state": "CutMan",
        # "state": "NightmarePit",
        "screen": None,
        "frameskip": frameskip,
        "frame_stack": frame_stack,
        "truncate_if_no_improvement": True,
        "obs_space": "screen",
        "action_space": "discrete",
        "crop_img": True,
        "invincible": False,
        "no_enemies": False,
        "render_mode": None,
        "fixed_damage_punishment": 1,
        "forward_factor": 0.5,
        "backward_factor": 0.55,
        "time_punishment_factor": time_punishment_factor,
        "multi_input": multi_input,
        "curriculum": False,
        "_enforce_subproc": True,
    }
    venv = make_venv(**env_kwargs)

    model_kwargs = {
        "buffer_size": max(n_envs // 4, 1) * 100_000,
        "learning_rate": 1e-4,
        "batch_size": 256,
        "gamma": 0.999,
        "learning_starts": 10_000,
        "target_update_interval": 10_000,
        "train_freq": 4,
        "gradient_steps": 1,
        "exploration_fraction": 0,
        "exploration_final_eps": 0.01,
    }

    model_name = (
        "dqn"
        # "_NIGHTMAREPIT"
        f"_envs{n_envs}"
        # f"_{'all' if env_kwargs['screen'] is None else env_kwargs['screen']}"
        f"_buff{model_kwargs['buffer_size']}_batch{model_kwargs['batch_size']}"
        f"_gamma{model_kwargs['gamma']}"
        f"_lr{model_kwargs['learning_rate']:.1e}"
        # "_lrLin4e-4"
        f"_lstart{model_kwargs['learning_starts']}"
        f"_updinterval{model_kwargs['target_update_interval']}"
        f"_freq{model_kwargs['train_freq']}"
        f"gradsteps{model_kwargs['gradient_steps']}"
        f"_explrfrac{model_kwargs['exploration_fraction']}"
        f"_explreps{model_kwargs['exploration_final_eps']}"
        "_"  # for separating env parameters
        f"_fs{frameskip}"
        f"_stack{frame_stack}"
        "_crop224"
        "common_rews"
        f"_timepun{time_punishment_factor}"
        # "_trunc6min"
        "_trunc60snoprog"
        "_spikefix6"
        "_scen3"
        # "_actionskipB"
        "_multinput2"
        # "_recurrent"
        # "_NIGHTMAREREW"
        # "_RE3"
        # "_curriculum500k"
        # "_INVINCIBLE"
        # "_NO_ENEMIES"
    )
    tensorboard_log = "logs/cutman"

    if Path(f"models/{model_name}.zip").exists():
        model = DQN.load(
            f"models/{model_name}",
            env=venv,
            tensorboard_log=tensorboard_log,
        )
    else:
        model = DQN(
            policy="CnnPolicy" if not multi_input else "MultiInputPolicy",
            env=venv,
            tensorboard_log=tensorboard_log,
            verbose=0,
            seed=666,
            device="cuda",
            **model_kwargs,
        )
    total_timesteps = 10_000_000
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000 // n_envs,
        save_path="checkpoints/",
        name_prefix=model_name,
    )
    eval_venv = make_venv(
        **{
            **env_kwargs,
            "n_envs": 1,
            "screen": None,
            "render_mode": None,
        }
    )
    eval_callback = EvalCallback(
        # same env, just replacing n_envs with 1
        eval_venv,
        n_eval_episodes=1,
        eval_freq=250_000 // n_envs,
        best_model_save_path=f"models/{model_name}_best",
        verbose=0,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[StageLoggingCallback(), checkpoint_callback, eval_callback],
        log_interval=1,
        tb_log_name=model_name,
        reset_num_timesteps=False,
    )
    model.save(f"models/{model_name}.zip")


if __name__ == "__main__":
    train()
