
import site
site.addsitedir("D:\\atr\\AI4Water")

import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ai4water.utils import edf_plot

from ai4water.models.utils import gen_cat_vocab
from ai4water.functional import Model
from ai4water.utils.utils import TrainTestSplit

from utils import make_data, get_dataset


mlp_path = r'D:\Papers drafts\HMI adsorption data prof. Cho\modelling\scripts\results\hpo_mlp_20230201_164049\best\20230201_174327'
mlp_model = Model.from_config_file(config_path=os.path.join(mlp_path, "config.json"))
mlp_wpath = os.path.join(mlp_path, 'weights', 'weights_224_0.31723.hdf5')
mlp_model.update_weights(mlp_wpath)

tab_path = r'D:\Papers drafts\HMI adsorption data prof. Cho\modelling\scripts\results\hpo_tab_20230130_191852\best\20230131_025331'
tab_model = Model.from_config_file(config_path=os.path.join(tab_path, "config.json"))
wpath = os.path.join(tab_path, 'weights', 'weights_194_0.20955.hdf5')
tab_model.update_weights(wpath)

ftt_path = r'D:\Papers drafts\HMI adsorption data prof. Cho\modelling\scripts\results\hpo_ftt_20230131_090916\best\20230131_133340'
ftt_model = Model.from_config_file(config_path=os.path.join(ftt_path, "config.json"))
wpath = os.path.join(ftt_path, 'weights', 'weights_080_0.10067.hdf5')
ftt_model.update_weights(wpath)

# %%

NUMERIC_FEATURES = ["Pyrolysis_temp", "Heating rate (oC)",
       "Pyrolysis_time (min)", "C", "H", "O", "N", "Ash", "H/C", "O/C", "N/C",
       "(O+N/C)", "Surface area", "Pore volume", "Average pore size",
        "Adsorption_time (min)", "Ci", "solution pH", "rpm",
       "Volume (L)", "loading (g)", "adsorption_temp",
                    "Ion Concentration (M)", "DOM"]
CAT_FEATURES = ["Adsorbent", "Feedstock", "inorganics", "Anion_type"]
LABEL = "qe"
data, *_ = make_data()

splitter = TrainTestSplit(seed=1000)

data[NUMERIC_FEATURES] = data[NUMERIC_FEATURES].astype(float)
data[CAT_FEATURES] = data[CAT_FEATURES].astype(str)
data['qe'] = data['qe'].astype(float)


train_data, test_data, _, _ = splitter.split_by_random(data)
cat_vocabulary = gen_cat_vocab(data)

X_train = [train_data[NUMERIC_FEATURES].values, train_data[CAT_FEATURES].values]
test_x = [test_data[NUMERIC_FEATURES].values, test_data[CAT_FEATURES].values]

TrainY = np.log(train_data[LABEL].values)
test_y = np.log(test_data[LABEL].values)

train_pred_ftt = ftt_model.predict(X_train, process_results=False)
test_pred_ftt = ftt_model.predict(test_x, process_results=False).reshape(-1,)
print(ftt_model.evaluate(x=test_x, y=test_y, metrics=["r2", "r2_score", "mse"]))


train_pred_tab = tab_model.predict(X_train, process_results=False).reshape(-1,)
test_pred_tab = tab_model.predict(test_x, process_results=False).reshape(-1,)
print(tab_model.evaluate(x=test_x, y=test_y, metrics=["r2", "r2_score", "mse"]))

dataset, *_ = get_dataset(encode=True)
train_x_mlp, train_y_mlp = dataset.training_data()
test_x_mlp, test_y_mlp = dataset.test_data()
train_y_mlp = np.log(train_y_mlp)
test_y_mlp = np.log(test_y_mlp)
train_pred_mlp = mlp_model.predict(train_x_mlp, process_results=False)
test_pred_mlp = mlp_model.predict(test_x_mlp, process_results=False)
print(mlp_model.evaluate(x=train_x_mlp, y=train_y_mlp, metrics=["r2", "r2_score", "mse"]))
print(mlp_model.evaluate(x=test_x_mlp, y=test_y_mlp, metrics=["r2", "r2_score", "mse"]))


legend_properties = {'weight':'bold',
                     'size': 14}

_, ax = plt.subplots()

edf_plot(np.abs(test_y_mlp - test_pred_mlp),
         c=np.array([68, 178, 205])/255, linewidth=2.5,
         label='ANN', ax=ax, show=False,
         ax_kws=dict(grid=True, xlabel='Absolute error'))

edf_plot(np.abs(np.log(test_data[LABEL].values) - test_pred_tab),
         c=np.array([200, 49, 40]) / 255,
         linewidth=2.5,
         label='Tab-Transformer', ax=ax, show=False,
         ax_kws=dict(grid=True, xlabel='Absolute error'))


edf_plot(np.abs(np.log(test_data[LABEL].values) - test_pred_ftt),
         c='#287271', linewidth=2.5,
         label='FT-Transformer', ax=ax, show=False,
         ax_kws=dict(grid=True, xlabel='Absolute error'))

ax.set_ylabel(ylabel= 'Commulative Probabilty', fontsize=14, weight='bold')
ax.set_xlabel(xlabel='Absolute Error', fontsize=14, weight='bold')
ax.set_xticklabels(ax.get_xticks().astype(int), size=12, weight='bold')
ax.set_yticklabels(ax.get_yticks().round(2), size=12, weight='bold')
ax.legend(prop=legend_properties)
ax.grid(ls='--', color='lightgrey')
plt.title("Empirical Distribution Function Plot",fontweight="bold")
plt.savefig("results/figures/edf.png", dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()
