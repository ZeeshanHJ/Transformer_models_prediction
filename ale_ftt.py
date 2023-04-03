

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
import matplotlib.pyplot as plt
from easy_mpl.utils import create_subplots
from shap.plots import beeswarm, heatmap
from easy_mpl import imshow
from sklearn.preprocessing import LabelEncoder
from alepython import ale_plot

from ai4water import Model
from ai4water.models.utils import gen_cat_vocab
from ai4water.utils.utils import TrainTestSplit

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

ftt_path = r'D:\Papers drafts\HMI adsorption data prof. Cho\modelling\scripts\results\hpo_ftt_20230131_090916\best\20230131_133340'
ftt_model = Model.from_config_file(config_path=os.path.join(ftt_path, "config.json"))
wpath = os.path.join(ftt_path, 'weights', 'weights_080_0.10067.hdf5')
ftt_model.update_weights(wpath)

# %%
ads_encoder = LabelEncoder()
ads_le = ads_encoder.fit_transform(train_data["Adsorbent"].values)
fs_encoder = LabelEncoder()
fs_le = fs_encoder.fit_transform(train_data["Feedstock"].values)
ino_encoder = LabelEncoder()
ino_le = ino_encoder.fit_transform(train_data["inorganics"].values)
ant_encoder = LabelEncoder()
ant_le = ant_encoder.fit_transform(train_data["Anion_type"].values)

cat_le = np.column_stack([ads_le, fs_le, ino_le, ant_le])

# %%
x_train_all = pd.DataFrame(
    np.column_stack([train_data[NUMERIC_FEATURES].values, cat_le]),
    columns=NUMERIC_FEATURES + CAT_FEATURES)

# %%

class MyModel:

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

        return ftt_model.predict(X).reshape(-1,)

# %%

ale_plot(train_set=x_train_all, model=MyModel(),
         features=["Surface area"],
         bins=100
             )

# %%
ale_plot(train_set=x_train_all, model=MyModel(),
         features=["loading (g)"],
         bins=100
             )

# %%
from easy_mpl import plot
cond = train_data.loc[train_data['Surface area']<100]
plot(cond.loc[:, 'Surface area'].values, cond[LABEL].values, '.')

# %%
from ai4water.postprocessing import PartialDependencePlot

feature_name = 'Surface area'
pdp = PartialDependencePlot(
    MyModel().predict,
    x_train_all,
    num_points=100,
    feature_names=x_train_all.columns.tolist(),
    show=False,
    save=False
)

pdp_sa, ice_vals = pdp.calc_pdp_1dim(x_train_all.values, feature_name)

ax = pdp._plot_pdp_1dim(pdp_sa, ice_vals, x_train_all.values,
                        feature_name, ice=False,
                        pdp_line_kws={'color': 'darkcyan'})
#ax.set_xticklabels(dye_enc.categories_[0])
ax.set_xlabel(feature_name)
plt.show()

# %%
feature_name = 'loading (g)'
pdp_sa, ice_vals = pdp.calc_pdp_1dim(x_train_all.values, feature_name)

ax = pdp._plot_pdp_1dim(pdp_sa, ice_vals, x_train_all.values,
                        feature_name, ice=False,
                        pdp_line_kws={'color': 'darkcyan'})
#ax.set_xticklabels(dye_enc.categories_[0])
ax.set_xlabel(feature_name)
plt.show()