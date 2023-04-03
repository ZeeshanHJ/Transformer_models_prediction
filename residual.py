
import site
site.addsitedir("D:\\atr\\AI4Water")

import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

train_pred_ftt = ftt_model.predict(X_train, process_results=False).reshape(-1,)
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
train_pred_mlp = mlp_model.predict(train_x_mlp, process_results=False).reshape(-1,)
test_pred_mlp = mlp_model.predict(test_x_mlp, process_results=False).reshape(-1,)
print(mlp_model.evaluate(x=train_x_mlp, y=train_y_mlp, metrics=["r2", "r2_score", "mse"]))
print(mlp_model.evaluate(x=test_x_mlp, y=test_y_mlp, metrics=["r2", "r2_score", "mse"]))

# %%

train_er = pd.DataFrame((np.exp(train_y_mlp.reshape(-1,)) - np.exp(train_pred_mlp)), columns=['Error'])
train_er['prediction'] = np.exp(train_pred_mlp)
train_er['hue'] = 'Training'
test_er = pd.DataFrame((np.exp(test_y_mlp.reshape(-1,)) - np.exp(test_pred_mlp)), columns=['Error'])
test_er['prediction'] = np.exp(test_pred_mlp)
test_er['hue'] = 'Test'

df_er = pd.concat([train_er, test_er], axis=0)

legend_properties = {'weight':'bold',
                     'size': 14,}

g = sns.jointplot(data=df_er, x="prediction",
                     y="Error",
                  joint_kws=dict(edgecolors='black', linewidth=0.5, alpha=0.5,
                                 edgecolor="black"),
              hue='hue', palette='flare')

ax = g.ax_joint

ax.set_ylabel(ylabel= 'Residuals', fontsize=14, weight='bold')
ax.set_xlabel(xlabel='Prediction', fontsize=14, weight='bold')
ax.set_xticklabels(ax.get_xticks().astype(int), size=12, weight='bold')
ax.set_yticklabels(ax.get_yticks().astype(int), size=12, weight='bold')
ax.axhline(0.0, color="k")
ax.legend(prop=legend_properties)
plt.savefig("results/figures/res_ann.png", dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()


# %%

train_er = pd.DataFrame((train_data[LABEL].values - np.exp(train_pred_tab)), columns=['Error'])
train_er['prediction'] = np.exp(train_pred_tab)
train_er['hue'] = 'Training'
test_er = pd.DataFrame((test_data[LABEL].values - np.exp(test_pred_tab)), columns=['Error'])
test_er['prediction'] = np.exp(test_pred_tab)
test_er['hue'] = 'Test'

df_er = pd.concat([train_er, test_er], axis=0)

legend_properties = {'weight':'bold',
                     'size': 14,}

g = sns.jointplot(data=df_er, x="prediction",
                     y="Error",
                  joint_kws=dict(edgecolors='black', linewidth=0.5, alpha=0.5,
                                 edgecolor="black"),
              hue='hue', palette='flare')

ax = g.ax_joint

ax.set_ylabel(ylabel= 'Residuals', fontsize=14, weight='bold')
ax.set_xlabel(xlabel='Prediction', fontsize=14, weight='bold')
ax.set_xticklabels(ax.get_xticks().astype(int), size=12, weight='bold')
ax.set_yticklabels(ax.get_yticks().astype(int), size=12, weight='bold')
ax.axhline(0.0, color="k")
ax.legend(prop=legend_properties)
plt.savefig("results/figures/res_tab.png", dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()


# %%

train_er = pd.DataFrame((train_data[LABEL].values - np.exp(train_pred_ftt)), columns=['Error'])
train_er['prediction'] = np.exp(train_pred_ftt)
train_er['hue'] = 'Training'
test_er = pd.DataFrame((test_data[LABEL].values - np.exp(test_pred_ftt)), columns=['Error'])
test_er['prediction'] = np.exp(test_pred_ftt)
test_er['hue'] = 'Test'

df_er = pd.concat([train_er, test_er], axis=0)

legend_properties = {'weight':'bold',
                     'size': 14,}

g = sns.jointplot(data=df_er, x="prediction",
                     y="Error",
                  joint_kws=dict(edgecolors='black', linewidth=0.5, alpha=0.5,
                                 edgecolor="black"),
              hue='hue', palette='flare')

ax = g.ax_joint

ax.set_ylabel(ylabel= 'Residuals', fontsize=14, weight='bold')
ax.set_xlabel(xlabel='Prediction', fontsize=14, weight='bold')
ax.set_xticklabels(ax.get_xticks().astype(int), size=12, weight='bold')
ax.set_yticklabels(ax.get_yticks().astype(int), size=12, weight='bold')
ax.axhline(0.0, color="k")
ax.legend(prop=legend_properties)
plt.savefig("results/figures/res_ftt.png", dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()
