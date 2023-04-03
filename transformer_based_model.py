import site
site.addsitedir("D:\\atr\\AI4Water")

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

import numpy as np
import pandas as pd
from ai4water import Model
from ai4water.utils.utils import TrainTestSplit
from ai4water.models.utils import gen_cat_vocab
from ai4water.models import TabTransformer, FTTransformer
from utils import make_data

from utils import evaluate_model

data, *_ = make_data()


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

# create vocabulary of unique values of categorical features
cat_vocabulary = gen_cat_vocab(data)

train_y = np.log(train_data[LABEL].values)
test_y = np.log(test_data[LABEL].values)

# model = Model (
#     model=TabTransformer(len(NUMERIC_FEATURES),
#                          cat_vocabulary=cat_vocabulary,
#                          hidden_units=16, depth=3,
#                          final_mlp_units= [84, 42]
#                          ),
#     monitor="r2",
#     patience=50
# )
#
#

train_x = [train_data[NUMERIC_FEATURES].values, train_data[CAT_FEATURES].values]
test_x = [test_data[NUMERIC_FEATURES].values, test_data[CAT_FEATURES].values]

#
# model.fit(x=train_x, y= train_y,
#               validation_data=(test_x, test_y),
#               epochs=500)
#
# train_p = model.predict(x=train_x,)
# evaluate_model(train_y, train_p)
#
#
# test_p = model.predict(x=test_x,)
# evaluate_model(test_y, test_p)


#%%%%
# FT Transformer
#===============

model = Model(model=FTTransformer(len(NUMERIC_FEATURES), cat_vocabulary,
                                  hidden_units=16, num_heads=8))


model.fit(x=train_x, y= train_data[LABEL].values,
              validation_data=(test_x, test_data[LABEL].values),
              epochs=1, verbose=1)


# train_p = model.predict(x=train_x)
# evaluate_model(train_data[LABEL].values, train_p)
#
# test_p = model.predict(x=test_x,)
# evaluate_model(test_data[LABEL].values, test_p)
