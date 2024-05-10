from pathlib import Path

from callbacks import MinDistanceCallback
from env import make_venv
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from megai_man.policy import CustomMultiInputLstmPolicy, CustomMultiInputPolicy


def train():
    AlgoClass = RecurrentPPO
    multi_input = True
    n_envs = 8
    frameskip = 4
    frame_stack = 1
    time_punishment_factor = 0
    env_kwargs = {
        "n_envs": n_envs,
        "state": "CutMan",
        "frameskip": frameskip,
        "frame_stack": frame_stack,
        "truncate_if_no_improvement": True,
        "obs_space": "screen",
        "action_space": "multi_discrete",
        "crop_img": True,
        "invincible": False,
        "render_mode": None,
        "fixed_damage_punishment": 1,
        "forward_factor": 0.5,
        "backward_factor": 0.55,
        "time_punishment_factor": time_punishment_factor,
        "multi_input": multi_input,
        "_enforce_subproc": True,
    }
    venv = make_venv(**env_kwargs)
    n_steps = 2048
    batch_size = n_envs * n_steps
    # batch_size = 512
    # lr = lambda x: 5e-4 * x
    lr = 1e-3

    model_kwargs = {
        "learning_rate": lr,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": 4,
        # future_horizon = frame_skip * frame_time / (1 - gamma) =
        # 4 * (1 / 60) / (1 - 0.995) = 13.3 seconds in real game time
        # that's the future horizon that our agent is capable of planning for
        "gamma": 0.99,
        "gae_lambda": 0.9,
        "clip_range": 0.2,
        "normalize_advantage": True,
        "ent_coef": 1e-3,
        # "policy_kwargs": {
        # "features_extractor_class": WideNatureCNN,
        # "features_extractor_kwargs": {"features_dim": 1024},
        # },
    }

    model_name = (
        "sevs"
        f"_steps{n_steps}_batch{batch_size}"
        f"_lr{model_kwargs['learning_rate']:.1e}"
        # "_lrLin5e-4"
        f"_epochs{model_kwargs['n_epochs']}"
        f"_clip{model_kwargs['clip_range']}"
        f"_ecoef{model_kwargs['ent_coef']:.0e}"
        f"_gamma{model_kwargs['gamma']}"
        # "_wide"
        # "_featdim1024"
        "_"  # for separating env parameters
        f"_fs{frameskip}"
        f"_stack{frame_stack}"
        "_crop224"
        "_small_rewards"
        f"_time_punishment{time_punishment_factor}"
        # "_trunc6min"
        "_trunc60snoprog"
        "_spikefix6"
        "_scen3"
        "_actionskip"
        "_multinput"
        "_recurrent"
        # "_INVINCIBLE"
    )
    tensorboard_log = "logs/cutman"
    if Path(f"models/{model_name}.zip").exists():
        model = AlgoClass.load(
            f"models/{model_name}",
            env=venv,
            device="cuda",
            tensorboard_log=tensorboard_log,
        )
    else:
        model = AlgoClass(
            # policy="CnnPolicy" if not multi_input else CustomMultiInputPolicy,
            policy="CnnLstmPolicy"
            if not multi_input
            else CustomMultiInputLstmPolicy,
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
    eval_callback = EvalCallback(
        # same env, just replacing n_envs with 1
        make_venv(**{**env_kwargs, "n_envs": 1}),
        n_eval_episodes=1,
        eval_freq=250_000 // n_envs,
        best_model_save_path=f"models/{model_name}",
        verbose=0,
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            MinDistanceCallback(),
            checkpoint_callback,
            eval_callback,
        ],
        log_interval=1,
        tb_log_name=model_name,
        reset_num_timesteps=False,
    )
    model.save(f"models/{model_name}.zip")


if __name__ == "__main__":
    train()
