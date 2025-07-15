import optuna
import json

def get_all_trials(db_path, study_name=None):
    """
    Extracts hyperparameters from a database file.

    Args:
        db_path (str): Path to the database file.
        study_name (str): Name of the Optuna study to load. If not specified, the name is the base name of the database file.
    Returns:
        dict: A dictionary containing hyperparameters.

    """

    if not study_name:
        study_name = db_path.split("/")[-1].split(".")[0]
        print(f"[INFO] Using study name: {study_name}")
    study = optuna.load_study(study_name= study_name, storage=f"sqlite:///{db_path}")
    trials = study.trials
    hyperparams = {}
    for trial in trials:
        hyperparams[trial.number] = {
            "params": trial.params,
            "value": trial.value,
            "state": trial.state,
        }
    with open(f"models/hyperparams_{db_path.split('/')[-1].split('.')[0]}.json", "w") as f:
        json.dump(hyperparams, f, indent=4)

    print(f"[INFO] Hyperparameters saved to models/hyperparams_{db_path.split('/')[-1].split('.')[0]}.json")
    return hyperparams

def get_best_trial(db_path, study_name=None):
    """
    Extracts the best trial from a database file.

    Args:
        db_path (str): Path to the database file.
    Returns:
        dict: A dictionary containing the best trial's hyperparameters.

    """

    if not study_name:
        study_name = db_path.split("/")[-1].split(".")[0]
        print(f"[INFO] Using study name: {study_name}")
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
    best_trial = study.best_trial
    best_hyperparams = {
        "params": best_trial.params,
        "value": best_trial.value,
        "state": best_trial.state,
    }

    with open(f"models/best_hyperparams_{db_path.split('/')[-1]}.json", "w") as f:
        import json
        json.dump(best_hyperparams, f, indent=4)

    print(f"[INFO] Best hyperparameters saved to models/best_hyperparams_{db_path.split('/')[-1]}.json")
    return best_hyperparams


#get_all_trials("models/cnn_affinity_022.db", study_name="cnn_affinity")