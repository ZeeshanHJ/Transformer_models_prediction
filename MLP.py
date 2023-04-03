
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import numpy as np
import pandas as pd
import seaborn as sns

from ai4water import Model
from ai4water.models import MLP
from ai4water.utils import edf_plot
from SeqMetrics import RegressionMetrics
from easy_mpl import regplot

from utils import get_dataset, evaluate_model


dataset, _, _, _, _ = get_dataset(encode = True)

train_x, train_y = dataset.training_data()

#%%%

test_x, test_y = dataset.test_data()



train_y = np.log(train_y)
test_y = np.log(test_y)

#%%%

model=Model(
    model = MLP(units=14, num_layers=2, activation="relu"),
    lr=0.01,
    input_features=dataset.input_features,
    output_features=dataset.output_features,
    epochs=500, batch_size=32,
    verbosity=1
)

h= model.fit(train_x, train_y, validation_data= (test_x, test_y))

#%%%

train_p = model.predict(x=train_x)

#%%
evaluate_model(train_y, train_p)


test_p = model.predict(x=test_x)

#%%%
regplot(pd.DataFrame(train_y), pd.DataFrame(train_p),
        annotation_key='$R^2$',
        annotation_val=RegressionMetrics(train_y, train_p).r2(),
        marker_size=60,
        marker_color='snow',
        line_style='--',
        line_color='indigo',
        line_kws=dict(linewidth= 3.0),
        scatter_kws=dict(linewidth=1.1, edgecolors=np.array([56, 86, 199])/255,
                         marker="8",
                         alpha=0.7
                         )
        )
#%%%%

regplot(pd.DataFrame(test_y), pd.DataFrame(test_p),
        annotation_key='$R^2$',
        annotation_val=RegressionMetrics(test_y, test_p).r2(),
        marker_size=60,
        marker_color='snow',
        line_style='--',
        line_color='indigo',
        line_kws=dict(linewidth= 3.0),
        scatter_kws=dict(linewidth=1.1, edgecolors=np.array([56, 86, 199])/255,
                         marker="8",
                         alpha=0.7
                         )
        )


#%%%%
train_er = pd.DataFrame((train_y - train_p), columns=['Error'])
train_er['prediction'] = train_p
train_er['hue'] = 'Training'
test_er = pd.DataFrame((test_y - test_p), columns=['Error'])
test_er['prediction'] = test_p
test_er['hue'] = 'Test'

df_er = pd.concat([train_er, test_er], axis=0)

legend_properties = {'weight':'bold',
                     'size': 14,}
g = sns.jointplot(data=df_er, x="prediction",
                  y="Error",
                  hue='hue', palette='husl')
ax = g.ax_joint
ax.axhline(0.0)
# ax.set_ylablel(ylabel= 'Residuals', fontsize = 14, weight = 'bold')
# ax.set_xlabel(xlabal='prediction', fontsize=14, weight='bold')
# ax.set_xticklabels(ax.get_xticks().astype(int), size=12, weight='bold')
# ax.set_yticklabels(ax.get_yticks().astype(int), size=12, weight='bold')
# ax.legend(prop=legend_properties)
plt.tight_layout()
plt.show()

#%%%%
legend_properties = {'weight':'bold',
                     'size':14}
_, ax = plt.subplots(#figsize=(5,4)
                     )

edf_plot(np.abs(train_y-train_p), label='Traning',
         c=np.array([200, 49, 40])/255,
         # c=np.array([234, 106, 41])/255,
         linewidth=2.5,
         show=False, ax=ax,)

edf_plot(np.abs(test_y-test_p), label='Test',
         c=np.array([68, 178, 205])/255,
         linewidth=2.5,
         show=False, ax=ax,
         ax_kws=dict(grid=True, xlabel='Absolute error'))
# ax.set_ylabel(ylabel= 'Commulative Probability', fontsize=14, weight='bold')
# ax.set_xlabel(xlabel='Absolute Error', fontsize=14, weight='bold')
# ax.set_xticklabels(ax.get_xtick().astypr(int), size=12, weight='bold')
# ax.set_yticklabels(ax.get_ytick().round(2), size=12, weight='bold')
# plt.title("Empirical Distribution Function Plot", fontweight="bold")
plt.tight_layout()
plt.show()

#%%%
train_df = pd.DataFrame(np.column_stack([train_y, train_p]),
                        columns=['true', 'predicted'])

train_df['hue'] = 'Training'

test_df = pd.DataFrame(np.column_stack([test_y, test_p]),
                        columns=['true', 'predicted'])

test_df['hue'] = 'Test'

df = pd.concat([train_df, test_df], axis=0)

legend_properties = {'weight':'bold',
                     'size':14,}

g = sns.jointplot(data=df, x="true",
                  y="predicted",
                  hue='hue', palette='husl')

ax = g.ax_joint

ax.set_ylabel(ylabel= 'Predicted Adsorption Capacity (mg/g)', fontsize=14, weight='bold')
ax.set_xlabel(xlabel='Experimental Adsorptpion Capacity (mg/g)', fontsize=14, weight='bold')
# ax.set_xticklabels(ax.get_xtick().astypr(int), size=12, weight='bold')
# ax.set_yticklabels(ax.get_ytick().round(2), size=12, weight='bold')
ax.legend(prop=legend_properties)
plt.tight_layout()
plt.show()

# X_train, y_train = dataset.training_data()
# X_test, y_test = dataset.test_data()
print(model.evaluate(train_x, train_y, metrics=['r2', 'nse']))
print(model.evaluate(test_x, test_y, metrics=['r2', 'nse']))
