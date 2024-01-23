from pathlib import Path

import numpy as np
import optuna
import yaml
from callbacks import MinDistanceCallback
from env import make_venv
from optuna.samplers import RandomSampler
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from train import model_name_from


def sample_params(trial: optuna.Trial, n_envs: int, n_steps: int):
    batch_size_exp = trial.suggest_int(
        "batch_size_power",
        6,
        int(np.log2(n_envs * n_steps)),
    )
    batch_size = 2**batch_size_exp
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    vf_coef = trial.suggest_float("vf_coef", 0.0, 1.0)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-1, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0, log=True)
    n_epochs = trial.suggest_int("n_epochs", 1, 10)
    gamma = trial.suggest_categorical("gamma", [0.99, 0.995, 0.999])
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1)
    features_dim_exp = trial.suggest_int("features_dim_power", 8, 10)
    features_dim = 2**features_dim_exp

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "clip_range": clip_range,
        "vf_coef": vf_coef,
        "ent_coef": ent_coef,
        "gae_lambda": gae_lambda,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "features_extractor_kwargs": {
                "features_dim": features_dim,
            },
        },
    }


def model_name_from(kwargs: dict):
    kwargs = kwargs.copy()
    kwargs["features_dim"] = kwargs["policy_kwargs"][
        "features_extractor_kwargs"
    ]["features_dim"]
    del kwargs["policy_kwargs"]

    string = ""
    for k, v in kwargs.items():
        if k == "n_steps":
            continue
        if string:
            string += "_"
        prefix = "".join([x[0] for x in k.split("_")])
        if k in [
            "learning_rate",
            "vf_coef",
            "ent_coef",
            "gae_lambda",
            "max_grad_norm",
        ]:
            value = f"{v:.2e}"
        else:
            value = v
        string += f"{prefix}{value}"

    return string


def optimizer(tensorboard_log: str, sample_fn, timesteps_per_trial=1_000_000):
    def optimize_agent(trial):
        n_envs = 8
        n_steps = 8192
        sticky_prob = 0.0
        damage_terminate = False
        damage_factor = 1 / 10
        truncate_if_no_improvement = True

        venv = make_venv(
            n_envs=n_envs,
            sticky_prob=sticky_prob,
            damage_terminate=damage_terminate,
            damage_factor=damage_factor,
            truncate_if_no_improvement=truncate_if_no_improvement,
            render_mode=None,
        )

        model_params = sample_fn(trial, n_envs, n_steps)
        model = PPO(
            policy="CnnPolicy",
            env=venv,
            tensorboard_log=tensorboard_log,
            verbose=0,
            seed=666,
            device="cuda",
            **model_params,
        )
        model_name = model_name_from(model_params)
        model.learn(
            timesteps_per_trial,
            callback=[MinDistanceCallback()],
            log_interval=1,
            tb_log_name=model_name,
        )
        venv.close()

        venv = make_venv(
            n_envs=1,
            sticky_prob=sticky_prob,
            damage_terminate=damage_terminate,
            damage_factor=damage_factor,
            truncate_if_no_improvement=truncate_if_no_improvement,
            render_mode=None,
        )
        reward, _ = evaluate_policy(
            model,
            venv,
            n_eval_episodes=1,
            deterministic=True,
        )
        venv.close()

        return reward

    return optimize_agent


def tune(sample_fn, name, n_trials=500, timesteps_per_trial=1_000_000):
    db_path = f"studies/{name}.db"
    Path(db_path).touch(exist_ok=True)

    study = optuna.create_study(
        storage=f"sqlite:///{db_path}",
        sampler=RandomSampler(seed=666),
        study_name=name,
        direction="maximize",
        load_if_exists=True,
    )

    n_trials -= len([x for x in study.trials if x.state.name == "COMPLETE"])

    study.optimize(
        optimizer(f"logs/{name}", sample_fn, timesteps_per_trial),
        n_trials=n_trials,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    with open(f"{name}_opt.yml", "w") as fp:
        yaml.safe_dump(study.best_params, fp)


if __name__ == "__main__":
    tune(
        sample_fn=sample_params,
        name="cutman_random_searcher",
        n_trials=100,
        timesteps_per_trial=1_000_000,
    )
