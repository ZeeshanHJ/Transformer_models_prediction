"""
==================
1.0 EDA
==================
"""
from easy_mpl.utils import create_subplots
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from ai4water.eda import EDA
from easy_mpl import hist
from easy_mpl import boxplot
from utils import make_data
from utils import make_data, box_violin, Inorganic_TYPES
from easy_mpl import plot

data, ae, fe, ie, ate = make_data(encode=False)

print(data.shape)

# %%
# Remove the categoricall features from our data

data.pop("Adsorbent")
data.pop("Feedstock")
data.pop("inorganics")
data.pop("Anion_type")

#%%%
print(data.describe())

#%%%

eda = EDA(data = data, save=False, show=False)

#%%%
# PC matrix
ax = eda.correlation(figsize=(9,9))
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, weight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, weight='bold')
plt.tight_layout()
plt.show()

#%%%
# making a line plot for numerical features

fig, axes = create_subplots(data.shape[1])

for ax, col, label in zip(axes.flat, data, data.columns):

    plot(data[col].values, ax=ax, ax_kws=dict(ylabel=col),
         lw=1.5,
         color='darkcyan', show=False)
    plt.tight_layout()
    plt.show()


#%%%
# Boxplot

fig, axes = create_subplots(data.shape[1])

for ax, col in zip(axes.flat, data.columns):
    boxplot(data[col].values, ax=ax, vert=False, fill_color='lightpink',
            flierprops={"ms": 1.0}, show=False, patch_artist=True,
            widths=0.6, medianprops={"color":"gray"},
            ax_kws=dict(xlabel=col, xlabel_kws={'weight': "bold"}))
plt.tight_layout()
plt.show()


#%%%%%
#Box_violin plot

fig, axes = create_subplots(data.shape[1])
for ax, col in zip(axes.flat, data.columns):
    box_violin(ax=ax, data=data[col], palette="Set2")
    ax.set_xlabel(xlabel=col, weight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), weight='bold')
plt.tight_layout()
plt.show()


#%%%%%
# Hist plot

fig, axes = create_subplots(data.shape[1])

for ax, col, label  in zip(axes.flat, data, data.columns):

    hist(data[col].values, ax=ax, bins=10,  show=False,
         grid=False,linewidth=0.5, edgecolor="k", color="khaki",
         ax_kws=dict(ylabel="Counts", xlabel=col))
plt.tight_layout()
plt.show()

#%%%%
# conditional distribution with respect to the inorganics

data, _, _,_,_ = make_data(encode=False)
# data.pop('inorganics')
feature = data['inorganics']
d = {k:Inorganic_TYPES[k] for k in feature.unique()}
feature = feature.map(d)
data['inorganics'] = feature

df_HMI = data.loc[data['inorganics']=="HMI"]
df_Rad_HMI = data.loc[data['inorganics']=="Rad_HMI"]
df_Fertilizer = data.loc[data['inorganics']=="Fertilizer"]
data.pop('inorganics')
data.pop('Feedstock')
data.pop('Adsorbent')
data.pop('Anion_type')

fig, axes = create_subplots(data.shape[1])

for ax, col in zip(axes.flat, data.columns):

    boxplot([df_HMI[col], df_Rad_HMI[col], df_Fertilizer[col]],
            labels=["HMI", "Rad_HMI", "Fertilizer"],
                ax=ax,
                flierprops={"ms": 0.6},
            medianprops={"color": "gray"},
                fill_color='lightpink',
            patch_artist=True,
                vert=False,
                widths=0.5,
            show=False,
            ax_kws=dict(xlabel=col, xlabel_kws={"weight": "bold"})
                )
    ax.set_yticklabels(ax.get_yticklabels(), weight='bold')
plt.tight_layout()
plt.show()
