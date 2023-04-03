
import site
site.addsitedir("D:\\atr\\AI4Water")

# import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()

import os
import numpy as np
import pandas as pd

import shap
from shap import KernelExplainer
from shap import Explanation
from utils import box_violin
import matplotlib.pyplot as plt
from easy_mpl.utils import create_subplots
from shap.plots import beeswarm, heatmap
from easy_mpl import imshow
from sklearn.preprocessing import LabelEncoder

from ai4water import Model
from ai4water.models.utils import gen_cat_vocab
from ai4water.utils.utils import TrainTestSplit

from easy_mpl import bar_chart

from utils import make_data, get_dataset

NUMERIC_FEATURES = ["Pyrolysis_temp", "Heating rate (oC)",
                    'Pyrolysis_time (min)', "C", "H", "O", "N", "Ash", "H/C", "O/C", "N/C",
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

ads_encoder = LabelEncoder()
ads_le = ads_encoder.fit_transform(train_data["Adsorbent"].values)
fs_encoder = LabelEncoder()
fs_le = fs_encoder.fit_transform(train_data["Feedstock"].values)
ino_encoder = LabelEncoder()
ino_le = ino_encoder.fit_transform(train_data["inorganics"].values)
ant_encoder = LabelEncoder()
ant_le = ant_encoder.fit_transform(train_data["Anion_type"].values)

ads_encoder_test = LabelEncoder()
ads_le_test = ads_encoder_test.fit_transform(test_data["Adsorbent"].values)
fs_encoder_test = LabelEncoder()
fs_le_test = fs_encoder_test.fit_transform(test_data["Feedstock"].values)
ino_encoder_test = LabelEncoder()
ino_le_test = ino_encoder_test.fit_transform(test_data["inorganics"].values)
ant_encoder_test = LabelEncoder()
ant_le_test = ant_encoder_test.fit_transform(test_data["Anion_type"].values)

cat_le_test = np.column_stack([ads_le_test, fs_le_test, ino_le_test, ant_le_test])

cat_le = np.column_stack([ads_le, fs_le, ino_le, ant_le])

tabt_path = r'D:\Papers drafts\HMI adsorption data prof. Cho\modelling\scripts\results\hpo_tab_20230130_191852\best\20230131_025331'
tabt_model = Model.from_config_file(config_path=os.path.join(tabt_path, "config.json"))
wpath = os.path.join(tabt_path, 'weights', 'weights_194_0.20955.hdf5')
tabt_model.update_weights(wpath)


class SingleInputModel:

    def predict(self, X):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=NUMERIC_FEATURES + CAT_FEATURES)

        # decode categorical features
        ads = ads_encoder.inverse_transform(X.loc[:, 'Adsorbent'].values.astype(int))
        fs = fs_encoder.inverse_transform(X.loc[:, 'Feedstock'].values.astype(int))
        ino = ino_encoder.inverse_transform(X.loc[:, 'inorganics'].values.astype(int))
        ant = ant_encoder.inverse_transform(X.loc[:, 'Anion_type'].values.astype(int))

        cat_x = pd.DataFrame(
            np.column_stack([ads, fs, ino, ant]),
            columns=CAT_FEATURES, dtype=str)

        num_x = X.loc[:, NUMERIC_FEATURES].astype(float)

        X = [num_x.values, cat_x.values]

        return np.exp(tabt_model.predict(x=X).reshape(-1,))


x_train_all = pd.DataFrame(
    np.column_stack([train_data[NUMERIC_FEATURES].values, cat_le]),
    columns=NUMERIC_FEATURES + CAT_FEATURES)

# %%

x_test_all = pd.DataFrame(
    np.column_stack([test_data[NUMERIC_FEATURES].values, cat_le_test]),
    columns=NUMERIC_FEATURES + CAT_FEATURES)

# %%

fpath = os.path.join(os.getcwd(), "results", "figures", "kernel_test_tabt.csv")

if os.path.exists(fpath):
    sv_df = pd.read_csv(fpath, index_col="Unnamed: 0")
    sv = sv_df.values
else:

    exp = KernelExplainer(SingleInputModel().predict, x_train_all)

    sv = exp.shap_values(x_test_all, nsamples=200)
    sv_df = pd.DataFrame(sv, columns=NUMERIC_FEATURES + CAT_FEATURES)
    sv_df.to_csv(fpath)

# %%

shap_values_exp = Explanation(
    sv,
    data=X_train,
    feature_names=NUMERIC_FEATURES + CAT_FEATURES
)

# %%

sv_bar = np.mean(np.abs(shap_values_exp.values), axis=0)

colors = {'Adsorption Experimental Conditions': '#60AB7B',
          'Physical Properties': '#F9B234',
          'Synthesis Conditions': '#E91B23',
          'Adsorbent Composition': 'k',
          }

classes = ['Synthesis Conditions',
           'Synthesis Conditions',
           'Synthesis Conditions',
           'Adsorbent Composition',
           'Adsorbent Composition',
           'Adsorbent Composition',
           'Adsorbent Composition',
           'Adsorbent Composition',
           'Adsorbent Composition',
           'Adsorbent Composition',
           'Adsorbent Composition',
           'Adsorbent Composition',
            'Physical Properties',
           'Physical Properties',
           'Physical Properties',
           'Adsorption Experimental Conditions',
           'Adsorption Experimental Conditions',
           'Adsorption Experimental Conditions',
           'Adsorption Experimental Conditions',
           'Adsorption Experimental Conditions',
           'Adsorption Experimental Conditions',
           'Adsorption Experimental Conditions',
           'Adsorption Experimental Conditions',
           'Adsorption Experimental Conditions',
           'Synthesis Conditions',
           'Synthesis Conditions',
           'Adsorption Experimental Conditions',
           'Adsorption Experimental Conditions'
           ]

df = pd.DataFrame({'features': shap_values_exp.feature_names, 'classes': classes})

# %%

ax = bar_chart(sv_bar, shap_values_exp.feature_names,
          # bar_labels=sv_bar,
               bar_label_kws={'label_type':'edge',
                                            'fontsize': 10,
                                            'weight': 'bold'},
          show=False, sort=True, color=[
        '#E91B23', '#E91B23', '#E91B23',
        'k', 'k', 'k',
        'k', 'k', 'k',
        'k', 'k', 'k',
        '#F9B234', '#F9B234', '#F9B234',
        '#60AB7B', '#60AB7B', '#60AB7B',
        '#60AB7B', '#60AB7B', '#60AB7B',
        '#60AB7B', '#60AB7B', '#60AB7B',
        '#E91B23', '#E91B23', '#60AB7B',
        '#60AB7B'
    ])

ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel(xlabel='mean(|SHAP value|)', fontsize=14, weight='bold')
ax.set_xticklabels(ax.get_xticks(), size=12, weight='bold')
ax.set_yticklabels(ax.get_yticklabels(), size=12, weight='bold')

labels = df['classes'].unique()
handles = [plt.Rectangle((0,0),1,1, color=colors[l]) for l in labels]
plt.legend(handles, labels, loc='lower right')

plt.tight_layout()
#plt.savefig("results/figures/kernel_tabt_bar.png", dpi=600, bbox_inches='tight')
plt.show()

# %%
from easy_mpl import pie
sv_norm = sv_bar / np.sum(sv_bar)

synthesis = np.sum(sv_norm[[0, 1, 2, 24]])
physical = np.sum(sv_norm[[12, 13, 14]])
experimental = np.sum(sv_norm[[15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27]])
composition = np.sum(sv_norm[[3, 4, 5, 6, 7, 8, 9, 10, 11]])

f, ax = plt.subplots()
ax.pie([experimental, physical, synthesis, composition],
       labels=[
           f"{round(experimental, 2)} %",
           f"{round(physical, 2)} %",
           f"{round(synthesis, 2)} %",
           f"{round(composition, 2)} %"],
            colors=colors.values(),
         textprops={"fontsize":14, "weight":"bold"},
         )
ax.legend(labels=["Experimental", "Physical", "Synthesis", "Composition"],
          fontsize=14,
          loc=(0.9, 0.9))
plt.tight_layout()
#plt.savefig("results/figures/kernel_tabt_pie.png", dpi=600, bbox_inches='tight')
plt.show()

# %%
from sklearn.manifold import TSNE
from easy_mpl import scatter


tsne = TSNE(n_components=2, random_state=313)
sv_2D = tsne.fit_transform(sv_df.values)

color = 'Surface area'

axes, s = scatter(
    sv_2D[:, 0],
    sv_2D[:, 1],
    c=test_data[color],
    cmap="Spectral",
    s=5,
    show=False
)
plt.gca().set_aspect('equal', 'datalim')

cbar = plt.colorbar(s)
cbar.ax.spines[:].set_visible(False)
cbar.ax.set_ylabel(color, fontsize=14, weight="bold")

plt.savefig(f'results/figures/shap_tabt_{color}.png', dpi=600, bbox_inches="tight")
plt.show()