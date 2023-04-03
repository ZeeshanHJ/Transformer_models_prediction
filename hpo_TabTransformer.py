import math
import os

import numpy as np
from ai4water.models.utils import gen_cat_vocab
from utils import make_data
import matplotlib.pyplot as plt
from ai4water import Model
from ai4water.models import MLP
from ai4water.utils.utils import jsonize, dateandtime_now, TrainTestSplit
from ai4water.models import TabTransformer
from ai4water.hyperopt import Categorical, Real, Integer, HyperOpt
from SeqMetrics import RegressionMetrics

from utils import get_dataset, evaluate_model




data = make_data()

# %%
ITER = 0

#%%%
PREFIX = f"hpo_mlp_{dateandtime_now()}"

NUMERIC_FEATURES = ["Pyrolysis_temp", "Heating rate (oC)",
       "Pyrolysis_time (min)", "C", "H", "O", "N", "Ash", "H/C", "O/C", "N/C",
       "(O+N/C)", "Surface area", "Pore volume", "Average pore size",
        "Adsorption_time (min)", "Ci", "solution pH", "rpm",
       "Volume (L)", "loading (g)", "adsorption_temp",
                    "Ion Concentration (M)", "DOM"]
CAT_FEATURES = ["Adsorbent", "Feedstock", "inorganics", "Anion_type"]
LABEL = "qe"

splitter = TrainTestSplit(seed=1000)

data[NUMERIC_FEATURES] = data[NUMERIC_FEATURES].astype(float)
data[CAT_FEATURES] = data[CAT_FEATURES].astype(str)
data['qe'] = data['qe'].astype(float)

train_data, test_data, _, _ = splitter.split_by_random(data)


TrainY = np.log(train_data[LABEL].values)
test_y = np.log(test_data[LABEL].values)

# create vocabulary of unique values of categorical features
cat_vocabulary = gen_cat_vocab(data)


X_train = [train_data[NUMERIC_FEATURES].values, train_data[CAT_FEATURES].values]
test_x = [test_data[NUMERIC_FEATURES].values, test_data[CAT_FEATURES].values]

# model = Model (
#     model=TabTransformer(len(NUMERIC_FEATURES),
#                          cat_vocabulary=cat_vocabulary,
#                          hidden_units=16, depth=3,
#                          final_mlp_units= [84, 42])
# )
#
# model.fit(x=X_train, y= train_data[LABEL].values,
#               validation_data=(test_x, test_data[LABEL].values),
#               epochs=500)
#
# train_p = model.predict(x=X_train,)
# evaluate_model(train_data[LABEL].values, train_p)
#
#
# test_p = model.predict(x=test_x,)
# evaluate_model(test_data[LABEL].values, test_p)


# %%
splitter = TrainTestSplit()
train_x, val_x, train_y, val_y = splitter.split_by_random(X_train, TrainY)

#%%%
MONITOR = {"mse": [], "r2_score": [], "r2": []}

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
            model=TabTransformer(len(NUMERIC_FEATURES),
                         cat_vocabulary=cat_vocabulary,
                         hidden_units=suggestions['hidden_units'],
                                 depth=suggestions['depth'],
                                 num_heads=suggestions['num_heads'],
                         final_mlp_units= [84, 42]),
            batch_size=suggestions["batch_size"],
            lr=suggestions["lr"],
            prefix=prefix or PREFIX,
            split_random=True,
            epochs=epochs,
            verbosity=0)

    # train the model
    if fit_on_all_data:
        _model.fit(X_train, TrainY,
                   validation_data=(test_x, test_y))
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
        _model.predict(X_train, TrainY)
        _model.predict(test_x, test_y)

    if return_model:
        return _model

    return val_score

#%%%
# parameter space
param_space = [
    Integer(10, 50, name="hidden_units"),
    Integer(1, 8, name= "num_heads"),
    Integer(1, 8, name= "depth"),
    Real(0.00001, 0.01, name="lr"),
    Categorical([4, 8, 12, 16, 24, 32, 48, 64], name="batch_size")
                ]

x0 = [16, 3, 4, 0.001, 8]


# %%
SEP = os.sep
num_iterations = 100

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

optimizer._plot_evaluations()
plt.tight_layout()
plt.show()

optimizer.plot_importance()
plt.tight_layout()
plt.show()

optimizer._plot_parallel_coords(figsize=(14, 8))
plt.tight_layout()
plt.show()

from skopt.plots import plot_objective
_ = plot_objective(optimizer.gpmin_results)
plt.show()


#%%%
model = objective_fn(prefix=f"{PREFIX}{SEP}best",
                     return_model=True,
                     epochs=500,
                     fit_on_all_data=True,
                     **optimizer.best_paras())


model.evaluate(X_train, TrainY, metrics=['r2', 'nse'])
model.evaluate(test_x, test_y, metrics=['r2', 'nse'])