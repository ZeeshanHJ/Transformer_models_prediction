import site
site.addsitedir("D:\\atr\\AI4Water")

import os

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import pandas as pd
from sklearn.manifold import TSNE
from easy_mpl.utils import process_cbar
from utils import make_data
from ai4water.utils.utils import TrainTestSplit


data, *_ = make_data()

splitter = TrainTestSplit(seed=1000)

train_data, test_data, _, _ = splitter.split_by_random(data)


df = pd.read_csv(
    os.path.join('results/figures/kernel_ftt_test1.csv'),
    index_col="Unnamed: 0"
)

tsne = TSNE(n_components=2, random_state=313)
sv_2D = tsne.fit_transform(df.values)

import seaborn as sns
import matplotlib as mpl
from matplotlib.lines import Line2D

color = 'Feedstock'

if color in ["Adsorbent", 'inorganics', 'Feedstock']:
    num_vals, codes = pd.factorize(test_data[color])
    rgb_values = sns.color_palette("tab20", len(codes))
    color_map = dict(zip(codes, rgb_values))
    c = test_data[color].map(color_map)
    cmap = None

else:
    c = test_data[color]
    cmap = "Spectral"

fig, axes = plt.subplots()
s = axes.scatter(
    sv_2D[:, 0],
    sv_2D[:, 1],
    c=c,
    cmap=cmap,
    s=5)
plt.gca().set_aspect('equal', 'datalim')

if color in ["Adsorbent", 'inorganics', 'Feedstock']:
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v,
                      label=k, markersize=8) for k, v in color_map.items()]

    axes.legend(title=color,
              handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left',
              title_fontsize=14
              )
else:
    process_cbar(s.axes, c, border=False, title=color,
             title_kws=dict(fontsize=14, weight="bold"))


plt.savefig(f'results/figures/tsne_shap_ftt_{color}.png', dpi=600, bbox_inches="tight")
plt.tight_layout()
plt.show()