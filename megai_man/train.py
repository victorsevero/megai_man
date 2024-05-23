from pathlib import Path

from callbacks import (
    CurriculumCallback,
    RE3Callback,
    StageLoggingCallback,
    TrainingStatsLoggerCallback,
)
from env import make_venv
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

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
        # "state": "CutMan",
        "state": "NightmarePit",
        "screen": None,
        "frameskip": frameskip,
        "frame_stack": frame_stack,
        "truncate_if_no_improvement": True,
        "obs_space": "screen",
        "action_space": "multi_discrete",
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
    # venv = VecNormalize(
    #     venv,
    #     training=True,
    #     norm_obs=False,
    #     norm_reward=True,
    #     gamma=0.995,
    #     clip_reward=10,
    # )
    n_steps = 512
    # batch_size = n_envs * n_steps
    batch_size = 128
    # lr = lambda x: 4e-4 * x
    lr = 1e-4

    model_kwargs = {
        "learning_rate": lr,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": 4,
        # future_horizon = frame_skip * frame_time / (1 - gamma) =
        # 4 * (1 / 60) / (1 - 0.995) = 13.3 seconds in real game time
        # that's the future horizon that our agent is capable of planning for
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.1,
        "normalize_advantage": True,
        "ent_coef": 1e-3,
        # "policy_kwargs": dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
        # "policy_kwargs": {
        # "features_extractor_class": WideNatureCNN,
        # "features_extractor_kwargs": {"features_dim": 1024},
        # },
    }

    model_name = (
        "sevs"
        "_NIGHTMAREPIT"
        f"_{'all' if env_kwargs['screen'] is None else env_kwargs['screen']}"
        f"_steps{n_steps}_batch{batch_size}"
        f"_lr{model_kwargs['learning_rate']:.1e}"
        # "_lrLin4e-4"
        f"_epochs{model_kwargs['n_epochs']}"
        f"_clip{model_kwargs['clip_range']}"
        f"_ecoef{model_kwargs['ent_coef']:.0e}"
        f"_gamma{model_kwargs['gamma']}"
        # "_wide_pivf"
        # "_rewnorm2"
        # "_featdim1024"
        "_"  # for separating env parameters
        f"_fs{frameskip}"
        f"_stack{frame_stack}"
        "_crop224"
        "common_rews"
        f"_time_punishment{time_punishment_factor}"
        # "_trunc6min"
        "_trunc60snoprog"
        "_spikefix6"
        "_scen3"
        "_actionskipB"
        "_multinput2"
        "_recurrent"
        # "_NIGHTMAREREW"
        # "_RE3"
        # "_curriculum500k"
        # "_INVINCIBLE"
        # "_NO_ENEMIES"
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
    total_timesteps = 2_000_000
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
    # eval_venv = VecNormalize(
    #     eval_venv,
    #     training=False,
    #     norm_obs=False,
    #     norm_reward=True,
    #     gamma=0.995,
    #     clip_reward=10,
    # )
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
        callback=[
            # RE3Callback(),
            StageLoggingCallback(),
            # CurriculumCallback(freq=500_000 // n_envs),
            TrainingStatsLoggerCallback(),
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
