

import site
site.addsitedir("D:\\atr\\AI4Water")

import math
import os

import numpy as np
import matplotlib.pyplot as plt
from ai4water import Model
from ai4water.models import MLP
from ai4water.utils.utils import jsonize, dateandtime_now, TrainTestSplit
from ai4water.hyperopt import Categorical, Real, Integer, HyperOpt
from SeqMetrics import RegressionMetrics

from utils import get_dataset, evaluate_model

#%%%
dataset, _, _, _, _ = get_dataset(encode=True)
X_train, y_train = dataset.training_data()
X_test, y_test = dataset.test_data()

y_train = np.log(y_train)
y_test = np.log(y_test)

#%%%
model = Model(
    model=MLP(),
    epochs=500,
    input_features=dataset.input_features,
    output_features=dataset.output_features,
    verbosity=0,
            )


#%%%%
model.fit(X_train,y_train, validation_data=(X_test, y_test))

#%% Training evaluation
train_p = model.predict(x=X_train)

evaluate_model(y_train, train_p)

#%% Test evaluation
test_p = model.predict(x=X_test,)
evaluate_model(y_test, test_p)

#%%%
PREFIX = f"hpo_mlp_{dateandtime_now()}"

#%%%
num_iterations = 100
ITER = 0
#%%%
SEP = os.sep
#%%%
MONITOR = {"mse": [], "r2_score": [], "r2": []}

#%%%


#%%
spliter = TrainTestSplit()
train_x, val_x, train_y, val_y = spliter.split_by_random(X_train, y_train)

#%%%
def objective_fn(
        prefix: str = None,
        return_model: bool = False,
        epochs: int = 400,
        fit_on_all_data: bool = False,
        **suggestions
):
    suggestions = jsonize(suggestions)
    global ITER

    # build model
    _model = Model(
        model=MLP(units=suggestions['units'],
                  num_layers=suggestions['num_layers'],
                  activation=suggestions['activation']),
        batch_size=suggestions["batch_size"],
        lr=suggestions["lr"],
        prefix=prefix or PREFIX,
        split_random=True,
        epochs=epochs,
        input_features=dataset.input_features,
        output_features=dataset.output_features,
        verbosity=0)

    # train the model
    if fit_on_all_data:
        _model.fit(X_train, y_train, validation_data=(X_test, y_test))
    else:
        _model.fit(train_x, train_y, validation_data=(val_x, val_y))


    # evaluate the model

    t, p = _model.predict(val_x, val_y, return_true=True,
                                            process_results=False)
    metrics = RegressionMetrics(t, p)
    val_score = metrics.mse()

    for metric in MONITOR.keys():
        val = getattr(metrics, metric)()
        MONITOR[metric].append(val)
    # here we are evaluating model with respect to mse, therefore
    #we don't need to subtract it from 1
    if not math.isfinite(val_score):
        val_score = 999999

    print(f"{ITER} {val_score}")
    ITER +=1

    if fit_on_all_data:
        _model.predict(X_train, y_train)
        _model.predict(X_test, y_test)

    if return_model:
        return _model

    return val_score

#%%%
# parameter space
param_space = [
    Integer(10, 50, name="units"),
    Integer(1, 4, name= "num_layers"),
    Categorical(["relu", "elu", "tanh", "sigmoid"], name="activation"),
    Real(0.00001, 0.01, name="lr"),
    Categorical([4, 8, 12, 16, 24, 32, 48, 64], name="batch_size")
                ]

x0 = [15, 1, "relu", 0.001, 8]

#%%%
optimizer = HyperOpt(
    algorithm="bayes",
    objective_fn=objective_fn,
    param_space=param_space,
    x0=x0,
    num_iterations=num_iterations,
    process_results=True,
    # we can turn it False if we want post-processesing of results
    opt_path=f"results{SEP}{PREFIX}"
)
#%%

results = optimizer.fit()
#
import matplotlib.pyplot as plot
optimizer._plot_convergence()
plt.show()
#
# optimizer._plot_evaluations()
# plt.tight_layout()
# plt.show()
#
# optimizer.plot_importance()
# plt.tight_layout()
# plt.show()
#
# optimizer._plot_parallel_coords(figsize=(14, 8))
# plt.tight_layout()
# plt.show()
#
# from skopt.plots import plot_objective
# _ = plot_objective(optimizer.gpmin_results)
# plt.show()


#%%%
model = objective_fn(prefix=f"{PREFIX}{SEP}best",
                     return_model=True,
                     epochs=500,
                     fit_on_all_data=True,
                     **optimizer.best_paras())


print(model.evaluate(X_train, y_train, metrics=['r2', 'nse']))
print(model.evaluate(X_test, y_test, metrics=['r2', 'nse']))


