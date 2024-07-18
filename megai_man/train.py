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
from sb3_contrib import MaskablePPO, RecurrentPPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# import wandb
from megai_man.policy import CustomMultiInputLstmPolicy, CustomMultiInputPolicy

# from wandb.integration.sb3 import WandbCallback


def train():
    AlgoClass = MaskablePPO
    multi_input = False
    n_envs = 8
    frameskip = 4
    frame_stack = 3
    time_punishment_factor = 0
    env_kwargs = {
        "n_envs": n_envs,
        "state": "CutMan",
        "screen": None,
        "frameskip": frameskip,
        "frame_stack": frame_stack,
        "truncate_if_no_improvement": True,
        "obs_space": "screen",
        "action_space": "multi_discrete",
        "crop_img": False,
        "no_enemies": True,
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
        "ent_coef": 1e-4,
        "vf_coef": 1,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"share_features_extractor": False},
    }

    model_name = (
        f"sevs{n_envs if n_envs != 8 else ''}"
        f"_steps{n_steps}_batch{batch_size}"
        f"_lr{model_kwargs['learning_rate']:.1e}"
        f"_epochs{model_kwargs['n_epochs']}"
        f"_clip{model_kwargs['clip_range']}"
        f"_ecoef{model_kwargs['ent_coef']:.0e}"
        f"_gamma{model_kwargs['gamma']}"
        f"_vf{model_kwargs['vf_coef']}"
        f"_maxgrad{model_kwargs['max_grad_norm']}"
        f"{'_twoFEs' if not model_kwargs['policy_kwargs']['share_features_extractor'] else ''}"
        "_"  # for separating env parameters
        f"_fs{frameskip}"
        f"_stack{frame_stack}"
        f"_rews{env_kwargs['forward_factor']}+scrn1"
        f"_scorerew{env_kwargs['score_reward']}"
        f"_dmg{env_kwargs['fixed_damage_punishment']}"
        "_ground"
        "_termbackscrn"
        # fmt: off
        "_spikefix7"
        "_Vscrnfix2"
        "_scen5multnoB"
        # fmt: on
        f"{'_multin6def' if multi_input else ''}"
        f"{'_mask' if AlgoClass == MaskablePPO else ''}"
        "_NO_ENEM2"
        "_vsbl"
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
            env=venv,
            tensorboard_log=tensorboard_log,
            verbose=0,
            seed=666,
            device="cuda",
            **model_kwargs,
        )
    total_timesteps = 19_000_000
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
    eval_callback = MaskableEvalCallback(
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
            StageLoggingCallback(),
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
