

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

sv_df = pd.read_csv(fpath, index_col="Unnamed: 0")
sv = sv_df.values

from shap.plots import colors
cmap = colors.red_blue
#
# # %%
#
# feature_idx = 20  # index of feature
# feature_name = NUMERIC_FEATURES[feature_idx] # Loading
#
# shap_scatter(shap_values=sv[:, feature_idx],
#              data=test_data.loc[:, feature_name].values,
#              feature_name=feature_name,
#               feature_wrt = test_data['Surface area'], cmap = cmap,
#              show=False)
#
# plt.savefig("results/figures/shap_loading_sa.png", dpi=600, bbox_inches='tight')
# plt.show()
#
# # %%
# # %%
#
# shap_scatter(shap_values=sv[:, feature_idx],
#              data=test_data.loc[:, feature_name].values,
#              feature_name=feature_name,
#               feature_wrt = test_data['Ci'], cmap = cmap,
#              show=False)
#
# plt.savefig("results/figures/shap_loading_ci.png", dpi=600, bbox_inches='tight')
# plt.show()
#
# # %%
#
# shap_scatter(shap_values=sv[:, feature_idx],
#              data=test_data.loc[:, feature_name].values,
#              feature_name=feature_name,
#               feature_wrt = test_data['solution pH'], cmap = cmap,
#              show=False)
#
# plt.savefig("results/figures/shap_loading_sph.png", dpi=600, bbox_inches='tight')
# plt.show()
#
# # %%
#
# from utils import Inorganic_TYPES
#
# feature_wrt = test_data['inorganics']
# d = {k:Inorganic_TYPES[k] for k in feature_wrt.unique()}
# shap_scatter(shap_values=sv[:, feature_idx],
#              data=test_data.loc[:, feature_name].values,
#              feature_name=feature_name,
#               feature_wrt = feature_wrt.map(d),
#              cmap = cmap,
#              palette_name="RdBu",
#              is_categorical=True,
#              show=False)
# plt.savefig("results/figures/shap_loading_inorganics.png", dpi=600, bbox_inches='tight')
# plt.show()
#
#
# # %%
#
# feature_idx = 16  # index of feature
# feature_name = NUMERIC_FEATURES[feature_idx] # Ci
#
# shap_scatter(shap_values=sv[:, feature_idx],
#              data=test_data.loc[:, feature_name].values,
#              feature_name=feature_name,
#               feature_wrt = test_data['Surface area'], cmap = cmap,
#              show=False)
#
# plt.savefig("results/figures/shap_ci_sa.png", dpi=600, bbox_inches='tight')
# plt.show()
#
# # %%
#
# shap_scatter(shap_values=sv[:, feature_idx],
#              data=test_data.loc[:, feature_name].values,
#              feature_name=feature_name,
#               feature_wrt = test_data['Pore volume'],
#              cmap = cmap,
#              show=False)
#
# plt.savefig("results/figures/shap_ci_pv.png", dpi=600, bbox_inches='tight')
# plt.show()
#
# # %%
# feature_idx = 26
# feature_name = 'inorganics'
#
# shap_scatter(shap_values=sv[:, feature_idx],
#              data=test_data.loc[:, feature_name].values,
#              feature_name=feature_name,
#               feature_wrt = test_data['Surface area'],
#              cmap = cmap,
#              xticklabels=test_data.loc[:, feature_name].unique(),
#              show=False)
#
# plt.savefig("results/figures/shap_ino_sa.png", dpi=600, bbox_inches='tight')
# plt.show()
#
# # %%
#
# shap_scatter(shap_values=sv[:, feature_idx],
#              data=test_data.loc[:, feature_name].values,
#              feature_name=feature_name,
#               feature_wrt = test_data['Pore volume'],
#              cmap = cmap,
#              xticklabels=test_data.loc[:, feature_name].unique(),
#              show=False)
#
# plt.savefig("results/figures/shap_ino_pv.png", dpi=600, bbox_inches='tight')
# plt.show()
#
# # %%
# feature_idx = 25
# feature_name = 'Feedstock'
#
# shap_scatter(shap_values=sv[:, feature_idx],
#              data=test_data.loc[:, feature_name].values,
#              feature_name=feature_name,
#               feature_wrt = test_data['Surface area'],
#              cmap = cmap,
#              xticklabels=test_data.loc[:, feature_name].unique(),
#              show=False)
#
# plt.savefig("results/figures/shap_fs_sa.png", dpi=600, bbox_inches='tight')
# plt.show()

# %%

feature_idx = 24
feature_name = 'Adsorbent'

ax = shap_scatter(shap_values=sv[:, feature_idx],
             data=test_data.loc[:, feature_name].values,
             feature_name=feature_name,
              feature_wrt = test_data['Surface area'],
             cmap = cmap,
             show=False)

ax.set_xticklabels(test_data.loc[:, feature_name].unique(), size=12, weight='bold',
                   rotation=90)
plt.savefig("results/figures/shap_ads_sa.png", dpi=600, bbox_inches='tight')
plt.show()

# %%

feature_idx = 17  # index of feature
feature_name = NUMERIC_FEATURES[feature_idx] # solution pH

shap_scatter(shap_values=sv[:, feature_idx],
             data=test_data.loc[:, feature_name].values,
             feature_name=feature_name,
              feature_wrt = test_data['Surface area'], cmap = cmap,
             show=False)

plt.savefig("results/figures/shap_sph_sa.png", dpi=600, bbox_inches='tight')
plt.show()

# %%

feature_idx = 22  # index of feature
feature_name = NUMERIC_FEATURES[feature_idx]  # 'Ion Concentration (M)'

shap_scatter(shap_values=sv[:, feature_idx],
             data=test_data.loc[:, feature_name].values,
             feature_name=feature_name,
              feature_wrt = test_data['Surface area'], cmap = cmap,
             show=False)

plt.savefig("results/figures/shap_ic_sa.png", dpi=600, bbox_inches='tight')
plt.show()

# %%

feature_idx = 15  # index of feature
feature_name = NUMERIC_FEATURES[feature_idx]  #  'Adsorption_time (min)'

shap_scatter(shap_values=sv[:, feature_idx],
             data=test_data.loc[:, feature_name].values,
             feature_name=feature_name,
              feature_wrt = test_data['Surface area'], cmap = cmap,
             show=False)

plt.savefig("results/figures/shap_adst_sa.png", dpi=600, bbox_inches='tight')
plt.show()

# %%

feature_idx = 14  # index of feature
feature_name = NUMERIC_FEATURES[feature_idx]  #  'Average pore size

shap_scatter(shap_values=sv[:, feature_idx],
             data=test_data.loc[:, feature_name].values,
             feature_name=feature_name,
              feature_wrt = test_data['Surface area'], cmap = cmap,
             show=False)

plt.savefig("results/figures/shap_aps_sa.png", dpi=600, bbox_inches='tight')
plt.show()

# %%
feature_idx = 18
feature_name = NUMERIC_FEATURES[feature_idx]  # rpm

shap_scatter(shap_values=sv[:, feature_idx],
             data=test_data.loc[:, feature_name].values,
             feature_name=feature_name,
              feature_wrt = test_data['Surface area'], cmap = cmap,
             show=False)

plt.savefig("results/figures/shap_rpm_sa.png", dpi=600, bbox_inches='tight')
plt.show()