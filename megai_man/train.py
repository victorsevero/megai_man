from pathlib import Path

from callbacks import (
    CuriosityCallback,
    CurriculumCallback,
    StageLoggingCallback,
    TrainingStatsLoggerCallback,
)
from env import make_venv
from rllte.xplore.reward.icm import ICM
from rllte.xplore.reward.rnd import RND
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# import wandb
from megai_man.policy import CustomMultiInputLstmPolicy, CustomMultiInputPolicy

# from wandb.integration.sb3 import WandbCallback


def train():
    AlgoClass = PPO
    multi_input = True
    n_envs = 8
    frameskip = 4
    frame_stack = 3
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
        "action_space": "multi_discrete",
        "crop_img": False,
        "invincible": False,
        "no_enemies": False,
        "render_mode": None,
        "damage_terminate": False,
        "fixed_damage_punishment": 0.05,
        "forward_factor": 0.05,
        "backward_factor": 0.055,
        "time_punishment_factor": time_punishment_factor,
        "multi_input": multi_input,
        "curriculum": False,
        "screen_rewards": False,
        "score_reward": 0,
        "distance_only_on_ground": True,
        "term_back_screen": True,
        "_enforce_subproc": False,
    }
    venv = make_venv(**env_kwargs)
    n_steps = 1024
    batch_size = 128
    model_kwargs = {
        "learning_rate": 2.5e-4,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": 4,
        # future_horizon = frame_skip * frame_time / (1 - gamma) =
        # 4 * (1 / 60) / (1 - 0.995) = 13.3 seconds in real game time
        # that's the future horizon that our agent is capable of planning for
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 1e-3,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {
            "share_features_extractor": False,
            # "lstm_hidden_size": 128,
        }
        # "policy_kwargs": dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
        # "policy_kwargs": {
        # "features_extractor_class": WideNatureCNN,
        # "features_extractor_kwargs": {"features_dim": 1024},
        # },
    }

    model_name = (
        f"sevs{n_envs if n_envs != 8 else ''}"
        # "_NIGHTMAREPIT"
        # f"_{'all' if env_kwargs['screen'] is None else env_kwargs['screen']}"
        f"_steps{n_steps}_batch{batch_size}"
        f"_lr{model_kwargs['learning_rate']:.1e}"
        # "_lrLin4e-4"
        f"_epochs{model_kwargs['n_epochs']}"
        f"_clip{model_kwargs['clip_range']}"
        f"_ecoef{model_kwargs['ent_coef']:.0e}"
        f"_gamma{model_kwargs['gamma']}"
        f"_vf{model_kwargs['vf_coef']}"
        f"_maxgrad{model_kwargs['max_grad_norm']}"
        f"{'_twoFEs' if not model_kwargs['policy_kwargs']['share_features_extractor'] else ''}"
        # "_RND"
        # "_ICM"
        # "_wide_pivf"
        # "_rewnorm2"
        # "_featdim1024"
        "_"  # for separating env parameters
        f"_fs{frameskip}"
        f"_stack{frame_stack}"
        # "_lstm128"
        # "_crop224"
        # "_hw224"
        # "_Near"
        "_rews0.05+screen1"
        f"_scorerew{env_kwargs['score_reward']}"
        f"_dmg{env_kwargs['fixed_damage_punishment']}"
        # "screenrews"
        # f"_time_punish{time_punishment_factor}"
        "_groundonly"
        # "_trunc6min"
        "_termbackscreen2"
        # "_dmgterm"
        # "_trunc60snoprog"
        # fmt: off
        "_spikefix6"
        "_scen5multi"
        "_skipB"
        # fmt: on
        "_multinput5_default"
        # "_recurrent"
        # "_contmap"
        # "_NIGHTMAREREW"
        # "_RE3"
        # "_curriculum500k"
        # "_INVINCIBLE"
        "_NO_ENEMIES2"
        # "_editROM3"
        "_visible"
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
            policy="CnnPolicy" if not multi_input else "MultiInputPolicy",
            # policy="CnnLstmPolicy"
            # if not multi_input
            # else CustomMultiInputLstmPolicy,
            # else "MultiInputLstmPolicy",
            env=venv,
            tensorboard_log=tensorboard_log,
            verbose=0,
            seed=666,
            device="cuda",
            **model_kwargs,
        )
    total_timesteps = 20_000_000
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
    # wandb.init(project="mega-man-1")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            # CuriosityCallback(ICM),
            StageLoggingCallback(),
            # CurriculumCallback(freq=500_000 // n_envs),
            TrainingStatsLoggerCallback(),
            checkpoint_callback,
            eval_callback,
            # WandbCallback(),
        ],
        log_interval=1,
        tb_log_name=model_name,
        reset_num_timesteps=False,
    )
    model.save(f"models/{model_name}.zip")


if __name__ == "__main__":
    train()
