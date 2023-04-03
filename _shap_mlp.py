
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
from ai4water.utils.utils import TrainTestSplit


from utils import get_dataset

dataset, *_ = get_dataset(encode=True)

splitter = TrainTestSplit(seed=1000)

X_train, y_train = dataset.training_data()
X_test, y_test = dataset.test_data()

ann_path = r'D:\Papers drafts\HMI adsorption data prof. Cho\modelling\scripts\results\hpo_mlp_20230201_164049\best\20230201_174327'
ann_model = Model.from_config_file(config_path=os.path.join(ann_path, "config.json"))
wpath = os.path.join(ann_path, 'weights', 'weights_224_0.31723.hdf5')
ann_model.update_weights(wpath)


NUMERIC_FEATURES = ["Pyrolysis_temp", "Heating rate (oC)",
                    'Pyrolysis_time (min)', "C", "H", "O", "N", "Ash", "H/C", "O/C", "N/C",
       "(O+N/C)", "Surface area", "Pore volume", "Average pore size",
        "Adsorption_time (min)", "Ci", "solution pH", "rpm",
       "Volume (L)", "loading (g)", "adsorption_temp",
                    "Ion Concentration (M)", "DOM"]
CAT_FEATURES = ["Adsorbent", "Feedstock", "inorganics", "Anion_type"]
LABEL = "qe"


class SingleInputModel:

    def predict(self, X):

        return np.exp(ann_model.predict(x=X).reshape(-1,))

sv_fpath = os.path.join(os.getcwd(), "results", "figures", "kernel_ann1.csv")

if os.path.exists(sv_fpath):
    sv_df1 = pd.read_csv(sv_fpath, index_col="Unnamed: 0")
else:
    exp = KernelExplainer(SingleInputModel().predict, X_train)

    sv = exp.shap_values(X_test, nsamples=200)

    # # %%
    sv_df = pd.DataFrame(sv, columns=ann_model.input_features)
    sv_ads = sv_df.loc[:, ['Adsorbent_0', 'Adsorbent_1', 'Adsorbent_2', 'Adsorbent_3',
                         'Adsorbent_4', 'Adsorbent_5', 'Adsorbent_6', 'Adsorbent_7',
                         'Adsorbent_8', 'Adsorbent_9', 'Adsorbent_10', 'Adsorbent_11',
                         'Adsorbent_12', 'Adsorbent_13', 'Adsorbent_14', 'Adsorbent_15'
                           ]].values.sum(axis=1)
    sv_fs = sv_df.loc[:, ['Feedstock_0', 'Feedstock_1', 'Feedstock_2', 'inorganics_0']].values.sum(axis=1)
    sv_ing = sv_df.loc[:, ['inorganics_1', 'inorganics_2', 'inorganics_3', 'inorganics_4',
                    'inorganics_5', 'inorganics_6']].values.sum(axis=1)
    sv_anion = sv_df.loc[:, ['Anion_type_0', 'Anion_type_1']].values.sum(axis=1)

    sv_df1 = pd.DataFrame(
        np.column_stack([sv_df.loc[:, NUMERIC_FEATURES].values,
                         sv_ads,
                         sv_fs,
                         sv_ing,
                         sv_anion
                         ]),
        columns=NUMERIC_FEATURES+CAT_FEATURES)
    sv_df1.to_csv("results/figures/kernel_ann1.csv")

# # %%
shap_values_exp = Explanation(
    sv_df1.values,
    data=X_test,
    feature_names=NUMERIC_FEATURES + CAT_FEATURES
)
#
# # %%

from easy_mpl import bar_chart
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
ax.set_xticklabels(ax.get_xticks().astype(int), size=12, weight='bold')
ax.set_yticklabels(ax.get_yticklabels(), size=12, weight='bold')

labels = df['classes'].unique()
handles = [plt.Rectangle((0,0),1,1, color=colors[l]) for l in labels]
plt.legend(handles, labels, loc='lower right')

plt.tight_layout()
#plt.savefig("results/figures/kernel_ann_bar1.png", dpi=600, bbox_inches='tight')
plt.show()
#
# # %%


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
          fontsize=14, loc=(0.9, 0.9))
plt.tight_layout()
#plt.savefig("results/figures/kernel_ann_pie.png", dpi=600, bbox_inches='tight')
plt.show()
#
# # %%
# # TSNE

from sklearn.manifold import TSNE
from easy_mpl.utils import process_cbar

tsne = TSNE(n_components=2, random_state=313)
sv_2D = tsne.fit_transform(sv_df1.values)

color = 'qe'
s = plt.scatter(
    sv_2D[:, 0],
    sv_2D[:, 1],
    c=y_test,
    cmap="Spectral",
    s=5)

process_cbar(s.axes, s, border=False, title=color,
             title_kws=dict(fontsize=14, weight="bold"))

plt.savefig(f'results/figures/shap_ann_{color}.png', dpi=600, bbox_inches="tight")
plt.show()