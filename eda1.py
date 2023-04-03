
import site
site.addsitedir("D:\\atr\\AI4Water")

from easy_mpl import boxplot, scatter, ridge
from ai4water.eda import EDA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
from utils import make_data

data, *_ = make_data(encode=False)

CATEGORICAL = ["Adsorbent", "Anion_type", "inorganics"]

for col in CATEGORICAL:
    data.pop(col)

# %%
#
# eda = EDA(data = data, save=False, show=False)
# # %%
# ax = eda.correlation(figsize=(9,9), annot=True, fmt=".1f",
#                      annot_kws={"size": 9}, cmap="flare")
# ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, weight='bold')
# ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, weight='bold')
# plt.tight_layout()
# plt.savefig("results/figures/correlation.png", dpi=600, bbox_inches='tight')
# plt.show()

# %%
material_props = ['Pyrolysis_temp', 'Heating rate (oC)', 'Pyrolysis_time (min)',
          'C', 'H', 'O', 'N', 'Ash', 'H/C', 'O/C', 'N/C', '(O+N/C)',
          'Surface area', 'Pore volume', 'Average pore size']


from easy_mpl.utils import create_subplots
from seaborn import violinplot

f, axes = create_subplots(15)
axes = axes.flatten()

for idx, col in enumerate(material_props):
    ax = axes[idx]

    violinplot(data[col], ax=ax, palette="flare")
    ax.set_xticks([])
    ax.set_title(col, fontsize=10, weight="bold")

    ax.set_yticklabels(ax.get_yticks(), size=9, weight="bold")

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig("results/figures/box_material.png", dpi=600, bbox_inches='tight')
plt.show()

# %%


# %%
#
# palette_name = "hot"
# feature_wrt = data['Feedstock']
# feature_wrt_name = "Feedstock"
# rgb_values = sns.color_palette(palette_name, feature_wrt.unique().__len__())
#
# color_map = dict(zip(feature_wrt.unique(), rgb_values))
# c = feature_wrt.map(color_map)
#
# ax, pc = scatter(data['O/C'].values,
#                  data['H/C'].values,
#                  edgecolors='black',
#                  linewidth=0.8,
#                  alpha=0.8,
#                  c=c,
#                  show=False)
#
# # add a legend
# handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v,
#                   label=k, markersize=8) for k, v in color_map.items()]
#
# ax.legend(title=feature_wrt_name,
#           handles=handles, bbox_to_anchor=(0.7, 0.8), loc='upper left',
#           title_fontsize=14
#           )
#
# ticks = np.round(ax.get_xticks(), 2)
# ax.set_xticklabels(ticks, size=12, weight='bold')
# ticks = np.round(ax.get_yticks(), 2)
# ax.set_yticklabels(ticks, size=12, weight='bold')
# ax.set_xlabel("O/C", fontdict={"size": 14, 'weight': "bold"})
# ax.set_ylabel("H/C", fontdict={"size": 14, 'weight': 'bold'})
# plt.savefig("results/figures/van_kreven.png", dpi=600, bbox_inches='tight')
# plt.show()

