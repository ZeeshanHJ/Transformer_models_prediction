
import site
site.addsitedir("D:\\atr\\AI4Water")
import os

# import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()

import numpy as np
import pandas as pd
from easy_mpl import imshow, bar_chart
from ai4water.functional import Model
from ai4water.models.utils import gen_cat_vocab
from ai4water.utils.utils import TrainTestSplit

from sklearn.preprocessing import LabelEncoder

from utils import make_data


path = r'D:\Papers drafts\HMI adsorption data prof. Cho\modelling\scripts\results\hpo_ftt_20230131_090916\best\20230131_133340'
ftt_model = Model.from_config_file(config_path=os.path.join(path, "config.json"))

wpath = os.path.join(path, 'weights', 'weights_080_0.10067.hdf5')
ftt_model.update_weights(wpath)


NUMERIC_FEATURES = ["Pyrolysis_temp", "Heating rate (oC)",
       "Pyrolysis_time (min)", "C", "H", "O", "N", "Ash", "H/C", "O/C", "N/C",
       "(O+N/C)", "Surface area", "Pore volume", "Average pore size",
        "Adsorption_time (min)", "Ci", "solution pH", "rpm",
       "Volume (L)", "loading (g)", "adsorption_temp",
                    "Ion Concentration (M)", "DOM"]
CAT_FEATURES = ["Adsorbent", "Feedstock", "inorganics", "Anion_type"]
LABEL = "qe"
data, *_ = make_data()

splitter = TrainTestSplit(seed=1225)

data[NUMERIC_FEATURES] = data[NUMERIC_FEATURES].astype(float)
data[CAT_FEATURES] = data[CAT_FEATURES].astype(str)
data['qe'] = data['qe'].astype(float)


train_data, test_data, _, _ = splitter.split_by_random(data)
cat_vocabulary = gen_cat_vocab(data)

X_train = [train_data[NUMERIC_FEATURES].values, train_data[CAT_FEATURES].values]
test_x = [test_data[NUMERIC_FEATURES].values, test_data[CAT_FEATURES].values]

print(ftt_model.evaluate(x=X_train, y=train_data[LABEL].values, metrics=['r2', 'r2_score']))
print(ftt_model.evaluate(x=test_x, y=test_data[LABEL].values, metrics=["r2", "r2_score"]))

imp = ftt_model.get_fttransformer_weights(inputs=X_train)
imshow(imp, xticklabels=NUMERIC_FEATURES + CAT_FEATURES, aspect="auto", colorbar=True)

# %%
bar_chart(imp.mean(axis=0), labels=NUMERIC_FEATURES + CAT_FEATURES,
          sort=True)
#
# # %%
#
# ads_encoder = LabelEncoder()
# ads_le = ads_encoder.fit_transform(train_data["Adsorbent"].values)
# fs_encoder = LabelEncoder()
# fs_le = fs_encoder.fit_transform(train_data["Feedstock"].values)
# ino_encoder = LabelEncoder()
# ino_le = ino_encoder.fit_transform(train_data["inorganics"].values)
# ant_encoder = LabelEncoder()
# ant_le = ant_encoder.fit_transform(train_data["Anion_type"].values)
#
# cat_le = np.column_stack([ads_le, fs_le, ino_le, ant_le])
#
# # %%
# from alepython import ale_plot
#
#
# class SingleInputModel:
#
#     def predict(self, X):
#
#
#         if not isinstance(X, pd.DataFrame):
#             X = pd.DataFrame(X, columns=NUMERIC_FEATURES + CAT_FEATURES)
#
#         # decode categorical features
#         ads = ads_encoder.inverse_transform(X.loc[:, 'Adsorbent'].values.astype(int))
#         fs = fs_encoder.inverse_transform(X.loc[:, 'Feedstock'].values.astype(int))
#         ino = ino_encoder.inverse_transform(X.loc[:, 'inorganics'].values.astype(int))
#         ant = ant_encoder.inverse_transform(X.loc[:, 'Anion_type'].values.astype(int))
#
#         cat_x = pd.DataFrame(
#             np.column_stack([ads, fs, ino, ant]),
#             columns=CAT_FEATURES, dtype=str)
#
#         num_x = X.loc[:, NUMERIC_FEATURES].astype(float)
#
#         X = [num_x.values, cat_x.values]
#         return ftt_model.predict(X).reshape(-1,)
#
#
# x_train_all = pd.DataFrame(
#     np.column_stack([train_data[NUMERIC_FEATURES].values, cat_le]),
#     columns=NUMERIC_FEATURES + CAT_FEATURES)
# #
# # ale_plot(train_set=pd.DataFrame(x_train_all, columns=model.input_features),
# #              model=SingleInputModel(),
# #                   features=["Surface area"]
# #              )
# #
# # # %%
# #
# # ale_plot(train_set=pd.DataFrame(x_train_all, columns=model.input_features),
# #              model=SingleInputModel(),
# #                   features=["Pore volume"]
# #              )
# #
# # # %%
# #
# # ale_plot(train_set=pd.DataFrame(x_train_all, columns=model.input_features),
# #              model=SingleInputModel(),
# #                   features=["Surface area", 'Pore volume']
# #              )
# #
# # # %%
# # ale_plot(train_set=pd.DataFrame(x_train_all, columns=model.input_features),
# #              model=SingleInputModel(),
# #                   features=["loading (g)", 'Ci']
# #              )
# #
# # # %%
# # ale_plot(train_set=pd.DataFrame(x_train_all, columns=model.input_features),
# #              model=SingleInputModel(),
# #                   features=["loading (g)", 'Surface area']
# #              )
# #
# #
# # # %%
# # ale_plot(train_set=pd.DataFrame(x_train_all, columns=model.input_features),
# #              model=SingleInputModel(),
# #                   features=["Ci", 'Surface area']
# #              )
#
#
# # %%
# # Kernel Explainer
# # ======================
# import shap
# from shap import KernelExplainer
# from shap import Explanation
# from utils import box_violin
# import matplotlib.pyplot as plt
# from easy_mpl.utils import create_subplots
# from shap.plots import beeswarm, heatmap
# from easy_mpl import imshow
#
# X_train_summary = shap.kmeans(x_train_all, 10)
#
# exp = KernelExplainer(SingleInputModel().predict, X_train_summary)
#
# sv = exp.shap_values(x_train_all)
#
# shap_values_exp = Explanation(
#     sv,
#     data=X_train_summary,
#     feature_names=ftt_model.input_features
# )
#
# sv_df = pd.DataFrame(sv, columns=ftt_model.input_features)
# fig, axes = create_subplots(sv.shape[1])
# for ax, col in zip(axes.flat, sv_df.columns):
#     box_violin(ax=ax, data=sv_df[col], palette="Set2")
#     ax.set_xlabel(col)
# plt.tight_layout()
# plt.show()
#
# # %%
# beeswarm(shap_values_exp, show=False)
# plt.tight_layout()
# plt.show()
#
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