import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
from ai4water.preprocessing import DataSet
from ai4water import Model
from SeqMetrics import RegressionMetrics

from ai4water.postprocessing import prediction_distribution_plot
from easy_mpl import violin_plot, boxplot


Inorganic_TYPES = {
    "Fe(III)": "HMI",
    "Mn(VII)": "HMI",
    "Iodine": "Rad_HMI",
    "Ba(II)": "Rad_HMI",
    "Co(II)": "Rad_HMI",
    "Sr(II)": "Rad_HMI",
    "Cr(VI)": "HMI",
}




def _load_data(input_features=None):
    # read excell file
    dirname = os.path.dirname(__file__)
    data = pd.read_excel(os.path.join(dirname, 'HMI_data.xlsx'), sheet_name=0)

    # default inputs
    def_inputs = ['Adsorbent', 'Feedstock', 'Pyrolysis_temp', 'Heating rate (oC)',
           'Pyrolysis_time (min)', 'C', 'H', 'O', 'N', 'Ash', 'H/C', 'O/C', 'N/C',
           '(O+N/C)', 'Surface area', 'Pore volume', 'Average pore size',
           'inorganics',
                    'Adsorption_time (min)', 'Ci', 'solution pH', 'rpm',
           'Volume (L)', 'loading (g)', 'g/L', 'adsorption_temp',
           'Ion Concentration (M)', 'Anion_type', 'DOM', 'Cf']

    data.columns = ['Adsorbent', 'Feedstock', 'Pyrolysis_temp', 'Heating rate (oC)',
           'Pyrolysis_time (min)', 'C', 'H', 'O', 'N', 'Ash', 'H/C', 'O/C', 'N/C',
           '(O+N/C)', 'Surface area', 'Pore volume', 'Average pore size',
           'inorganics',
                    'radius (pm)', 'hydra_radius_pm', 'First_ionic_IE_KJ/mol',
                    'Adsorption_time (min)', 'Ci', 'solution pH', 'rpm',
           'Volume (L)', 'loading (g)', 'g/L', 'adsorption_temp',
           'Ion Concentration (M)', 'Anion_type', 'DOM', 'Cf', 'qe']

    target = ['qe']

    if input_features is None:
        input_features = def_inputs
    else:
        assert isinstance(input_features, list)
        assert all([feature in data.columns for feature in input_features])


    return data[input_features + target]



def _ohe_column(df, col_name:str):


        encoder = OneHotEncoder(sparse=False)
        ohe_cat = encoder.fit_transform(df[col_name].values.reshape(-1,1))
        cols_added = [f"{col_name}_{i}" for i in range (ohe_cat.shape[-1])]

        df[cols_added] = ohe_cat
        df.pop(col_name)
        return df, cols_added, encoder


def make_data(encode:bool = False):
    data = _load_data()

    Adsorbent_encoder, Feedstock_encoder, inorganics_encoder, Anion_type_encoder = None, None, None, None

    if encode:
        data, _, Adsorbent_encoder = _ohe_column(data, 'Adsorbent')
        data, _, Feedstock_encoder = _ohe_column(data, 'Feedstock')
        data, _, inorganics_encoder = _ohe_column(data, 'inorganics')
        data, _, Anion_type_encoder = _ohe_column(data, 'Anion_type')


    # moving target to the last
    target = data.pop('qe')
    data['qe'] = target

    return data, Adsorbent_encoder, Feedstock_encoder, inorganics_encoder, Anion_type_encoder


import matplotlib.pyplot as plt
import seaborn as sns
def box_violin(ax, data, palette=None):
    if palette is None:
        palette = sns.cubehelix_palette(start=.5, rot=-.5, dark=0.3, light=0.7)
    ax = sns.violinplot(orient='h', data=data,
                        palette=palette,
                        scale="width", inner=None,
                        ax=ax)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(plt.Rectangle((x0, y0), width, height / 2, transform=ax.transData))

    sns.boxplot(orient='h', data=data, saturation=1, showfliers=False,
                width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
    old_len_collections = len(ax.collections)

    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0, 0.12]))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


    return

def get_dataset(encode=False):

    data, Adsorbent_encoder, Feedstock_encoder, inorganics_encoder, Anion_type_encoder = make_data(
        encode=encode
    )

    dataset = DataSet(data=data,
                      seed=1000,
                      val_fraction=0.0,
                      train_fraction=0.7,
                      split_random=True,
                      input_features=data.columns.tolist()[0:-1],
                      output_features=data.columns.tolist()[-1:],
                      )

    return dataset, Adsorbent_encoder, Feedstock_encoder, inorganics_encoder, Anion_type_encoder

def evaluate_model(true, predicted):
    metrics = RegressionMetrics(true, predicted)
    for i in ['mse', 'rmse', 'r2_score', 'mape', 'mae']:
        print(i, getattr(metrics, i)())
    return

def get_fitted_model(

):
    path = os.path.join(os.path.dirname(__file__), 'results', '20230125_163726')
    # cpath = os.path.join(path, 'config.jason')
    cpath = r'D:\Papers drafts\HMI adsorption data prof. Cho\modelling\scripts\results\20230125_163726\config.json'
    model = Model.from_config_file(config_path=cpath)

    # %%%
    wpath = os.path.join(path, 'weights', 'weights_370_383.91742.hdf5')
    model.update_weights(wpath)

    return model

def prediction_distribution(
        feature_name,
        test_p,
        cut,
        grid=None,
        plot_type="violin",
        show:bool = True,
):

    dataset, _, _, _, _ = get_dataset()

    X_test, _ = dataset.test_data()

    ax, df = prediction_distribution_plot(
        mode='regression',
        inputs=pd.DataFrame(X_test, columns=dataset.input_features),
        prediction=test_p,
        feature=feature_name,
        feature_name=feature_name,
        show=False,
        cust_grid_points=grid
    )

    if plot_type == "bar":
        if show:
            plt.show()
        return ax

    if feature_name == 'calcination_temp':
        df.drop(3, inplace=True)
        df['display_column'] = ['[500,600)', '[600,700)', '[700,800)']

    elif feature_name == 'calcination (min)':
        df.drop(3, inplace=True)
        df['display_column'] = ['[0,60)', '[60,90)', '[90,120)']

    elif feature_name == 'Ci':
        df.drop(0, inplace=True)
        df['display_column'] = ['[0.4,1.1)', '[1.1,3)', '[3,5.1)', '[5.1,10.1)', '[10.1,30.3)', '[30.3,52)']

    elif feature_name == 'Volume (L)':
        df.drop(1, inplace=True)
        df['display_column'] = ['[0.001,0.02)', '[0.02,0.04)', '[0.04,0.06)', '[0.06,0.1)']

    elif feature_name == 'Adsorbent Loading':
        df.drop(2, inplace=True)
        df['display_column'] = ['[0,0.05)', '[0.05,0.1)', '[0.1,0.2)', '[0.2,0.3)', '[0.3,0.4)', '[0.4,0.5)', '[0.5,0.6)']


    preds = {}
    for interval in df['display_column']:
        st, en = interval.split(',')
        st = float(''.join(e for e in st if e not in ["]", ")", "[", "("]))
        en = float(''.join(e for e in en if e not in ["]", ")", "[", "("]))
        df1 = pd.DataFrame(X_test, columns=dataset.input_features)
        df1['target'] = test_p
        df1 = df1[[feature_name, 'target']]
        df1 = df1[(df1[feature_name] >= st) & (df1[feature_name] < en)]
        preds[interval] = df1['target'].values

    for k, v in preds.items():
        assert len(v) > 0, f"{k} has no values in it"
    plt.close('all')

    if plot_type == "violin":
        ax = violin_plot(list(preds.values()), cut=cut, show=False)
        ax.set_xticks(range(len(preds)))
        ax.set_facecolor("#fbf9f4")

    else:
        ax, _ = boxplot(
                list(preds.values()), show=False,
                fill_color="lightpink", patch_artist=True,
                medianprops={"color": "black"}, flierprops={"ms":1.0}
            )

    ax.set_xticklabels(list(preds.keys()), size=12, weight='bold')
    ax.set_yticklabels(ax.get_yticks().astype(int), size=12, weight='bold')
    ax.set_title(feature_name, size=14, fontweight="bold")
    plt.show()

    return ax


def shap_scatter(
        shap_values:np.ndarray,  # SHAP values for a single feature
        data:np.ndarray,  # data for which shap value has been calculated
        feature_name:str, # name of feature for which shap vlaue has been calculated
        feature_wrt:pd.Series = None,
        show_hist=True,
        show=True,
        is_categorical=False,
        palette_name = "tab10",
        s = 70,
        edgecolors='black',
        linewidth=0.8,
        alpha=0.8,
        ax = None,
        xticklabels:list = None,
        **scatter_kws
):
    from matplotlib.lines import Line2D
    from easy_mpl import scatter
    import seaborn as sns

    assert isinstance(data, np.ndarray)
    assert len(data) == data.size

    assert isinstance(shap_values, np.ndarray)
    assert len(shap_values) == shap_values.size

    if ax is None:
        fig, ax = plt.subplots()

    if feature_wrt is None:
        c = None
    else:
        if is_categorical:
            if isinstance(palette_name, (tuple, list)):
                assert len(palette_name) == len(feature_wrt.unique())
                rgb_values = palette_name
            else:
                rgb_values = sns.color_palette(palette_name, feature_wrt.unique().__len__())
            color_map = dict(zip(feature_wrt.unique(), rgb_values))
            c= feature_wrt.map(color_map)
        else:
            c= feature_wrt.values.reshape(-1,)

    _, pc = scatter(
        data,
        shap_values,
        c=c,
        s=s,
        marker="o",
        edgecolors=edgecolors,
        linewidth=linewidth,
        alpha=alpha,
        ax=ax,
        show=False,
        **scatter_kws
    )

    if feature_wrt is not None:
        feature_wrt_name = ' '.join(feature_wrt.name.split('_'))
        if is_categorical:
            # add a legend
            handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v,
                              label=k, markersize=8) for k, v in color_map.items()]

            ax.legend(title=feature_wrt_name,
                  handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left',
                      title_fontsize=14
                      )
        else:
            cbar = plt.colorbar(pc, aspect=80)
            cbar.ax.set_ylabel(feature_wrt_name, rotation=90, labelpad=14,
                               fontsize=14, weight="bold")

            if 'volume' in feature_wrt_name.lower():
                ticks = np.round(cbar.ax.get_yticks(), 2)
                cbar.ax.set_yticklabels(ticks, size=12, weight='bold')
            else:
                cbar.ax.set_yticklabels(cbar.ax.get_yticks().astype(int), size=12, weight='bold')

            cbar.set_alpha(1)
            cbar.outline.set_visible(False)

    #feature_name = ' '.join(shap_values.feature_names.split('_'))

    ax.set_xlabel(feature_name, fontsize=14, weight="bold")
    ax.set_ylabel(f"SHAP value for {feature_name}", fontsize=14, weight="bold")
    ax.axhline(0, color='grey', linewidth=1.3, alpha=0.3, linestyle='--')

    if xticklabels is None:
        xticks = np.array(ax.get_xticks())
        if 'float' in xticks.dtype.name:
            ticks = np.round(ax.get_xticks(), 2)
            ax.set_xticklabels(ticks, size=12, weight='bold')
        elif 'int' in xticks.dtype.name:
            ax.set_xticklabels(xticks, size=12, weight='bold')
        else:
            xticks = np.array(ax.get_xticks(), dtype=int)
            ax.set_xticklabels(xticks.astype(int), size=12, weight='bold')
    else:
        ax.set_xticklabels(xticklabels, size=12, weight='bold')

    if 'float' in ax.get_yticks().dtype.name:
        ticks = np.round(ax.get_yticks(), 2)
        ax.set_yticklabels(ticks, size=12, weight='bold')
    else:
        yticks = np.array(ax.get_yticks(), dtype=int)
        ax.set_yticklabels(yticks.astype(int), size=12, weight='bold')


    if show_hist:
        x = data

        if len(x) >= 500:
            bin_edges = 50
        elif len(x) >= 200:
            bin_edges = 20
        elif len(x) >= 100:
            bin_edges = 10
        else:
            bin_edges = 5

        ax2 = ax.twinx()

        xlim = ax.get_xlim()

        ax2.hist(x.reshape(-1,), bin_edges,
                 range=(xlim[0], xlim[1]),
                 density=False, facecolor='#000000', alpha=0.1, zorder=-1)
        ax2.set_ylim(0, len(x))
        ax2.set_yticks([])

    if show:
        plt.show()

    return ax


def shap_local_barchart(
        shap_values,
        base_value,
        max_display=10,
        **kwargs
):

    import matplotlib
    from shap.plots._labels import labels
    from shap.utils import safe_isinstance, format_value
    from shap.plots import colors

    base_values = shap_values.base_values
    features = shap_values.display_data if shap_values.display_data is not None else shap_values.data
    feature_names = shap_values.feature_names
    lower_bounds = getattr(shap_values, "lower_bounds", None)
    upper_bounds = getattr(shap_values, "upper_bounds", None)
    values = shap_values.values

    # make sure we only have a single output to explain
    if (type(base_values) == np.ndarray and len(base_values) > 0) or type(base_values) == list:
        raise Exception("waterfall_plot requires a scalar base_values of the model output as the first "
                        "parameter, but you have passed an array as the first parameter! "
                        "Try shap.waterfall_plot(explainer.base_values[0], values[0], X[0]) or "
                        "for multi-output models try "
                        "shap.waterfall_plot(explainer.base_values[0], values[0][0], X[0]).")

    # make sure we only have a single explanation to plot
    if len(values.shape) == 2:
        raise Exception(
            "The waterfall_plot can currently only plot a single explanation but a matrix of explanations was passed!")

    # unwrap pandas series
    if safe_isinstance(features, "pandas.core.series.Series"):
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values

    # fallback feature names
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(values))])

    # init variables we use for tracking the plot locations
    num_features = min(max_display, len(values))
    row_height = 0.5
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(values))
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    loc = base_values + values.sum()
    yticklabels = ["" for i in range(num_features + 1)]

    # size the plot based on how many features we are plotting
    plt.gcf().set_size_inches(8, num_features * row_height + 1.5)

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features == len(values):
        num_individual = num_features
    else:
        num_individual = num_features - 1


    # compute the locations of the individual features and plot the dashed connecting lines
    for i in range(num_individual):
        sval = values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            plt.plot([loc, loc], [rng[i] - 1 - 0.4, rng[i] + 0.4],
                     color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
        if features is None:
            yticklabels[rng[i]] = feature_names[order[i]]
        else:
            if np.issubdtype(type(features[order[i]]), np.number):
                yticklabels[rng[i]] = format_value(float(features[order[i]]), "%0.03f") + " = " + feature_names[order[i]]
            else:
                yticklabels[rng[i]] = features[order[i]] + " = " + feature_names[order[i]]


    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(values):
        yticklabels[0] = "%d other features" % (len(values) - num_features + 1)
        remaining_impact = base_values - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
            c = colors.red_rgb
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)
            c = colors.blue_rgb

    points = pos_lefts + list(np.array(pos_lefts) + np.array(pos_widths)) + neg_lefts + \
        list(np.array(neg_lefts) + np.array(neg_widths))
    dataw = np.max(points) - np.min(points)

    # draw invisible bars just for sizing the axes
    label_padding = np.array([0.1*dataw if w < 1 else 0 for w in pos_widths])
    plt.barh(pos_inds, np.array(pos_widths) + label_padding + 0.02*dataw,
             left=np.array(pos_lefts) - 0.01*dataw, color=colors.red_rgb, alpha=0)
    label_padding = np.array([-0.1*dataw if -w < 1 else 0 for w in neg_widths])
    plt.barh(neg_inds, np.array(neg_widths) + label_padding - 0.02*dataw,
             left=np.array(neg_lefts) + 0.01*dataw, color=colors.blue_rgb, alpha=0)

    # define variable we need for plotting the arrows
    head_length = 0.08
    bar_width = 0.8
    xlen = plt.xlim()[1] - plt.xlim()[0]
    fig = plt.gcf()
    ax = plt.gca()
    xticks = ax.get_xticks()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    bbox_to_xscale = xlen/width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()

    # draw the positive arrows
    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = plt.arrow(
            pos_lefts[i], pos_inds[i], max(dist-hl_scaled, 0.000001), 0,
            head_length=min(dist, hl_scaled),
            color=colors.red_rgb, width=bar_width,
            head_width=bar_width
        )

        if pos_low is not None and i < len(pos_low):
            plt.errorbar(
                pos_lefts[i] + pos_widths[i], pos_inds[i],
                xerr=np.array([[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]),
                ecolor=colors.light_red_rgb
            )

        txt_obj = plt.text(
            pos_lefts[i] + 0.5*dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=12
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                pos_lefts[i] + (5/72)*bbox_to_xscale + dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
                horizontalalignment='left', verticalalignment='center', color=colors.red_rgb,
                fontsize=12
            )

    # draw the negative arrows
    for i in range(len(neg_inds)):
        dist = neg_widths[i]

        arrow_obj = plt.arrow(
            neg_lefts[i], neg_inds[i], -max(-dist-hl_scaled, 0.000001), 0,
            head_length=min(-dist, hl_scaled),
            color=colors.blue_rgb, width=bar_width,
            head_width=bar_width
        )

        if neg_low is not None and i < len(neg_low):
            plt.errorbar(
                neg_lefts[i] + neg_widths[i], neg_inds[i],
                xerr=np.array([[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]),
                ecolor=colors.light_blue_rgb
            )

        txt_obj = plt.text(
            neg_lefts[i] + 0.5*dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=12
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                neg_lefts[i] - (5/72)*bbox_to_xscale + dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
                horizontalalignment='right', verticalalignment='center', color=colors.blue_rgb,
                fontsize=12
            )

    # draw the y-ticks twice, once in gray and then again with just the feature names in black
    # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ytick_pos = list(range(num_features)) + list(np.arange(num_features)+1e-8)
    plt.yticks(ytick_pos, yticklabels[:-1] + [l.split('=')[-1] for l in yticklabels[:-1]], fontsize=13)

    # put horizontal lines for each feature row
    for i in range(num_features):
        plt.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

    # mark the prior expected value and the model prediction
    plt.axvline(base_values, 0, 1/num_features, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
    fx = base_values + values.sum()
    plt.axvline(fx, 0, 1, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)

    # clean up the main axis
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    ax.tick_params(labelsize=13)
    #plt.xlabel("\nModel output", fontsize=12)

    # draw the E[f(X)] tick mark
    xmin, xmax = ax.get_xlim()
    ax2 = ax.twiny()
    ax2.set_xlim(xmin, xmax)
    ax2.set_xticks([base_values, base_values+1e-8])  # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax2.set_xticklabels(["\n$E[f(X)]$", "\n$ = "+format_value(base_values, "%0.03f")+"$"], fontsize=12, ha="left")
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # draw the f(x) tick mark
    ax3 = ax2.twiny()
    ax3.set_xlim(xmin, xmax)
    # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax3.set_xticks([base_values + values.sum(), base_values + values.sum() + 1e-8])
    ax3.set_xticklabels(["$f(x)$", "$ = "+format_value(fx, "%0.03f")+"$"], fontsize=12, ha="left")
    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(-10/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(12/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_color("#999999")
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    # adjust the position of the E[f(X)] = x.xx label
    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(-20/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(22/72., -1/72., fig.dpi_scale_trans))

    tick_labels[1].set_color("#999999")

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color("#999999")

    return