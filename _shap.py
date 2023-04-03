
import site
site.addsitedir("D:\\atr\\AI4Water")

# import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()

import os
import numpy as np
import pandas as pd


import shap
from easy_mpl import bar_chart
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

from utils import shap_scatter
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

cat_le_train = np.column_stack([ads_le, fs_le, ino_le, ant_le])

# %%

ads_encoder_test = LabelEncoder()
ads_le_test = ads_encoder_test.fit_transform(test_data["Adsorbent"].values)
fs_encoder_test = LabelEncoder()
fs_le_test = fs_encoder_test.fit_transform(test_data["Feedstock"].values)
ino_encoder_test = LabelEncoder()
ino_le_test = ino_encoder_test.fit_transform(test_data["inorganics"].values)
ant_encoder_test = LabelEncoder()
ant_le_test = ant_encoder_test.fit_transform(test_data["Anion_type"].values)

cat_le_test = np.column_stack([ads_le_test, fs_le_test, ino_le_test, ant_le_test])

ftt_path = r'D:\Papers drafts\HMI adsorption data prof. Cho\modelling\scripts\results\hpo_ftt_20230131_090916\best\20230131_133340'
ftt_model = Model.from_config_file(config_path=os.path.join(ftt_path, "config.json"))
wpath = os.path.join(ftt_path, 'weights', 'weights_080_0.10067.hdf5')
ftt_model.update_weights(wpath)

test_p = ftt_model.predict(x=test_x)

class SingleInputModel:

    def predict(self, X):


        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=NUMERIC_FEATURES + CAT_FEATURES)

        # decode categorical features
        ads = ads_encoder_test.inverse_transform(X.loc[:, 'Adsorbent'].values.astype(int))
        fs = fs_encoder_test.inverse_transform(X.loc[:, 'Feedstock'].values.astype(int))
        ino = ino_encoder_test.inverse_transform(X.loc[:, 'inorganics'].values.astype(int))
        ant = ant_encoder_test.inverse_transform(X.loc[:, 'Anion_type'].values.astype(int))

        cat_x = pd.DataFrame(
            np.column_stack([ads, fs, ino, ant]),
            columns=CAT_FEATURES, dtype=str)

        num_x = X.loc[:, NUMERIC_FEATURES].astype(float)

        X = [num_x.values, cat_x.values]
        return np.exp(ftt_model.predict(x=X).reshape(-1,))



x_train_all = pd.DataFrame(
    np.column_stack([train_data[NUMERIC_FEATURES].values, cat_le_train]),
    columns=NUMERIC_FEATURES + CAT_FEATURES)

# %%

x_test_all = pd.DataFrame(
    np.column_stack([test_data[NUMERIC_FEATURES].values, cat_le_test]),
    columns=NUMERIC_FEATURES + CAT_FEATURES)


#X_train_summary = shap.kmeans(x_train_all, 100)


fname = "kernel_ftt_test1.csv"
fpath = os.path.join(os.getcwd(), "results", "figures", fname)

if os.path.exists(fpath):
    sv_df = pd.read_csv(fpath, index_col="Unnamed: 0")
    sv = sv_df.values
else:
    exp = KernelExplainer(SingleInputModel().predict, x_train_all)

    sv = exp.shap_values(x_test_all, nsamples=200)

    sv_df = pd.DataFrame(sv,
                         columns=NUMERIC_FEATURES+CAT_FEATURES)

    sv_df.to_csv(fpath)

# %%

#
# imshow(sv, xticklabels=NUMERIC_FEATURES + CAT_FEATURES, aspect="auto",
#        colorbar=True)
#
# # %%

shap_values_exp = Explanation(
    sv,
    data=test_x,
    feature_names=NUMERIC_FEATURES + CAT_FEATURES
)

# sv_df = pd.DataFrame(sv, columns=ftt_model.input_features)
# fig, axes = create_subplots(sv.shape[1])
# for ax, col in zip(axes.flat, sv_df.columns):
#     box_violin(ax=ax, data=sv_df[col], palette="Set2")
#     ax.set_xlabel(col)
# plt.tight_layout()
# plt.show()
#
# %%
# beeswarm(shap_values_exp, show=False)
# plt.tight_layout()
# plt.show()

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
        '#E91B23', '#E91B23',
        '#60AB7B',  '#60AB7B'
    ])
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel(xlabel='mean(|SHAP value|)', fontsize=14, weight='bold')
ax.set_xticklabels(ax.get_xticks().astype(int), size=12, weight='bold')
ax.set_yticklabels(ax.get_yticklabels(), size=12, weight='bold')

labels = df['classes'].unique()
handles = [plt.Rectangle((0,0),1,1, color=colors[l]) for l in labels]
plt.legend(handles, labels, loc='lower right')

plt.tight_layout()
#plt.savefig("results/figures/kernel_ftt_bar.png", dpi=600, bbox_inches='tight')
plt.show()

# %%
# pie chart

sv_norm = sv_bar / np.sum(sv_bar)

sv_norm_df = df.copy()
sv_norm_df['frac'] = sv_norm

synthesis = sv_norm_df.loc[sv_norm_df['classes']=="Synthesis Conditions"]['frac'].sum()
physical = sv_norm_df.loc[sv_norm_df['classes']=="Physical Properties"]['frac'].sum()
experimental = sv_norm_df.loc[sv_norm_df['classes']=="Adsorption Experimental Conditions"]['frac'].sum()
composition = sv_norm_df.loc[sv_norm_df['classes']=="Adsorbent Composition"]['frac'].sum()

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
          loc=(0.9, 0.9), fontsize=14)
plt.tight_layout()
#plt.savefig("results/figures/kernel_ftt_pie.png", dpi=600, bbox_inches='tight')
plt.show()


# # %%
# heatmap(shap_values_exp, instance_order=shap_values_exp.sum(1), show=False)
# plt.tight_layout()
# plt.show()
#
# # %%
# im = imshow(sv, aspect="auto", colorbar=True,
#        xticklabels=NUMERIC_FEATURES + CAT_FEATURES, show=False,
#        )
# ax = im.axes
# #ax.set_xticklabels(model.input_features)
# plt.tight_layout()
# plt.show()
#
#
#


# %%
# Feature Importance FT Transformer
imp = ftt_model.get_fttransformer_weights([data[NUMERIC_FEATURES], data[CAT_FEATURES]])
bar_chart(imp.mean(axis=0),
          labels=NUMERIC_FEATURES + CAT_FEATURES,
          sort=True,
          color="salmon",
          ax_kws=dict(xlabel="Feature Importance", xlabel_kws=dict(fontsize=14, weight="bold")),
          show=False)
#plt.savefig("results/figures/ftt_feat_imp.png", dpi=600, bbox_inches='tight')
plt.show()


# %%
index = test_p.argmin()
ax = bar_chart(abs(sv[index]),
          labels=NUMERIC_FEATURES + CAT_FEATURES,
          sort=True,
          color="#ca0020",
          ax_kws=dict(xlabel="SHAP Value", xlabel_kws=dict(fontsize=14, weight="bold")),
          show=False
          )
xticks = ax.get_xticks()
xticklabels = [f"-{val}" for val in xticks]
ax.set_xticklabels(xticklabels)
#plt.savefig(f"results/figures/ftt_shap_lowest_{index}.png", dpi=600, bbox_inches='tight')
plt.show()