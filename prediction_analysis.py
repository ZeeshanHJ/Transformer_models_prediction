
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"

from ai4water import Model
from ai4water.models import MLP
from ai4water.postprocessing import prediction_distribution_plot
from easy_mpl import violin_plot, boxplot
from utils import get_dataset, evaluate_model, get_fitted_model
from utils import prediction_distribution


dataset, _, _, _, _ = get_dataset()

train_x, train_y = dataset.training_data()

#%%%

test_x, test_y = dataset.test_data()


#%%%

model = get_fitted_model()


#%%%
#training
train_p = model.predict(x=train_x)

#%%
evaluate_model(train_y, train_p)

#%%% testing
test_p = model.predict(x=test_x)

#%%
evaluate_model(test_y, test_p)

#%%%

prediction_distribution('calcination_temp', test_p, 0.4)
prediction_distribution('Ci', test_p, 0.4, plot_type="boxplot")


model.prediction_analysis(
    x = pd.DataFrame(test_x, columns=dataset.input_features),
    features = ['calcination (min)', 'calcination_temp'],
    feature_names = ['calcination (min)', 'calcination_temp'],
    grid_types=["percentile", "percentile"],
    num_grid_points=[3,3],
    border=True,
    annotate_kws={'annotate_fontsize': 15,
                  'annotate_colors': np.array(
                      [['black', 'black', 'black'],
                       ['black', 'black', 'black'],
                       ['black', 'black', 'black']])}
)


_ = model.prediction_analysis(
    x = pd.DataFrame(test_x, columns=dataset.input_features),
    features = ['Adsorption Time (min)', 'Ci'],
    feature_names = ['Adsorption Time (min)', 'Ci'],
    grid_types=["percentile", "percentile"],
    num_grid_points=[6, 6],
    border=True,
    annotate_kws={'annotate_fontsize': 15,
                  'annotate_colors': np.array(
                      [['black', 'black', 'black', 'black', 'black'],
                       ['black', 'black', 'black', 'black', 'black'],
                       ['black', 'black', 'black', 'black', 'black'],
                       ['white', 'black', 'black', 'black', 'black']])}
)