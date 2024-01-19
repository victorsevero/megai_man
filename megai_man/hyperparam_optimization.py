from pathlib import Path

import optuna
import yaml
from env import make_venv
from optuna.visualization import plot_optimization_history
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def sample_ppo_params(trial: optuna.Trial, n_envs: int):
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    n_steps = trial.suggest_categorical(
        "n_steps",
        [64, 128, 256, 512, 1024, 2048],
    )
    batch_size = trial.suggest_categorical(
        "batch_size",
        [64, 128, 256, 512],
    )

    if (n_envs * n_steps) % batch_size > 0:
        raise optuna.TrialPruned()

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical(
        "gae_lambda",
        [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0],
    )
    max_grad_norm = trial.suggest_categorical(
        "max_grad_norm",
        [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5],
    )
    vf_coef = trial.suggest_float("vf_coef", 0, 1)

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
    }


def optimize_agent(trial):
    n_envs = 8
    damage_terminate = True

    venv = make_venv(
        n_envs=n_envs,
        sticky_prob=0.25,
        damage_terminate=damage_terminate,
        render_mode=None,
    )

    model_params = sample_ppo_params(trial, n_envs)
    model = PPO(
        policy="CnnPolicy",
        env=venv,
        tensorboard_log="logs/cutman_optimization_dmg_terminate",
        verbose=0,
        seed=666,
        device="cuda",
        **model_params,
    )
    model.learn(100_000, log_interval=1)

    venv.close()
    venv = make_venv(
        n_envs=n_envs,
        sticky_prob=0.0,
        damage_terminate=damage_terminate,
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


if __name__ == "__main__":
    name = "cutman_dmg_terminate"

    db_path = f"studies/{name}.db"
    Path(db_path).touch(exist_ok=True)

    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        load_if_exists=True,
    )

    n_trials = 100
    n_trials -= len([x for x in study.trials if x.state.name == "COMPLETE"])

    try:
        study.optimize(
            optimize_agent,
            n_trials=n_trials,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=True,
        )
        best_params = study.best_params

        with open(f"{name}_opt.yml", "w") as fp:
            yaml.safe_dump(study.best_params, fp)
    finally:
        fig = plot_optimization_history(study)
        fig.write_html(f"studies/{name}.html")
