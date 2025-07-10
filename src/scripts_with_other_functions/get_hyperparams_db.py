import optuna


def get_all_trials(db_path):
    """
    Extracts hyperparameters from a database file.

    Args:
        db_path (str): Path to the database file.
    Returns:
        dict: A dictionary containing hyperparameters.

    """

    study = optuna.load_study(study_name="study", storage=f"sqlite:///{db_path}")
    trials = study.trials
    hyperparams = {}
    for trial in trials:
        hyperparams[trial.number] = {
            "params": trial.params,
            "value": trial.value,
            "state": trial.state,
        }

    with open(f"models/hyperparams_{db_path.split('/')[-1]}.json", "w") as f:
        import json
        json.dump(hyperparams, f, indent=4)

    print(f"Hyperparameters saved to models/hyperparams_{db_path.split('/')[-1]}.json with {len(hyperparams)} trials.")


def get_best_trial(db_path):
    """
    Extracts the best trial from a database file.

    Args:
        db_path (str): Path to the database file.
    Returns:
        dict: A dictionary containing the best trial's hyperparameters.

    """

    study = optuna.load_study(study_name="study", storage=f"sqlite:///{db_path}")
    best_trial = study.best_trial
    best_hyperparams = {
        "params": best_trial.params,
        "value": best_trial.value,
        "state": best_trial.state,
    }

    with open(f"models/best_hyperparams_{db_path.split('/')[-1]}.json", "w") as f:
        import json
        json.dump(best_hyperparams, f, indent=4)

    print(f"Best hyperparameters saved to models/best_hyperparams_{db_path.split('/')[-1]}.json.")
    return best_hyperparams


