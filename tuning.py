import pandas as pd
import numpy as np
import tensorflow as tf
from mlp import generate_mlp
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

import optuna
from numba import cuda
import gc

from tensorflow.keras import layers, models

from model_factory_bird import *




if __name__ == "__main__":
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=hyp_calls, gc_after_trial=True)
    # Get the best parameters
    best_params = study.best_params
    print("Best Parameters:", best_params)

    # Best objective value achieved
    best_value = study.best_value
    print("Best Objective Value:", best_value)
    for trial in study.trials:
        print("Trial Number:", trial.number)
        print("Params:", trial.params)
        print("Value:", trial.value)

    from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_contour
    plot_optimization_history(study)
    plot_param_importances(study)
    plot_slice(study)
    plot_contour(study)
