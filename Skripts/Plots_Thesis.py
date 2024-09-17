import plotly.graph_objects as go
from sklearn import metrics
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from matplotlib.patches import Patch

class Plotter:

    def __init__(self, output):
        self.output = output
        self.colorpalette = sns.color_palette("colorblind")
        if output == 'thesis':
            self.width = 412
            # 459 in cau vorlage
        elif output == 'presentation':
            self.width = 800
        else:
            ValueError("Output must be 'thesis' or 'presentation'")

        # TODO change set
        # sns.set_theme()
        # sns.set_palette(self.colorpalette)
        # plt.rcParams['axes.prop_cycle'] = plt.cycler(color=self.colorpalette)

        # plt.rcParams['text.usetex'] = True
        # plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

        sns.set_theme(style='darkgrid', palette=self.colorpalette)
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{mathpazo} \usepackage{xcolor}'
        tex_fonts = {
            # Use LaTeX to write all text
            "text.usetex": True,
            "font.family": "Palatino",
            # "font.family": "serif",
            # "font.serif": "Palatino",
            # Use 10pt font in plots, to match 10pt font in document
            
            "font.size": 11,
            # Make the legend/label fonts a little smaller
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }

        plt.rcParams.update(tex_fonts)




    def plot_CM(self, Y, Y_hat, title, width = 800, height = 800, percentagereference = 'recall'):
        # Generate a confusion matrix
        cm = metrics.confusion_matrix(Y, Y_hat)
        n_classes = len(np.unique(Y))
        # Build normalized confusion matrix
        #classes = np.arange(0, n_classes)
        classes = np.arange(Y.min(), Y.max() + 1)
        if percentagereference == 'recall':
            cm_norm = np.around(cm.astype('float')/cm.sum(axis=1)[:, np.newaxis], decimals=2)
            name = 'Percentage of<br>true label'
        else: # precision is calculated as percentages
            cm_norm = np.around(cm.astype('float')/cm.sum(axis=0)[np.newaxis, :], decimals=2)
            name = 'Percentage of<br>predicted label'
        cm_norm_df = pd.DataFrame(cm_norm, index=classes, columns=classes)

        # Create a go.Image figure
        fig = go.Figure(data=go.Heatmap(
                            z=cm_norm_df.values,
                            x=cm_norm_df.columns,
                            y=cm_norm_df.index,
                            text = cm,
                            texttemplate="%{z:.0%}<br>(%{text:d})",
                            colorscale='Viridis',
                            colorbar=dict(title=name),
                            zmin=0,  # Lower color limit
                            zmax=1))

        fig.update_layout(
            title=title,
            xaxis_title='Predicted label',
            yaxis_title='True Label',
            autosize=False,
            width=width,
            height=height)

        fig.update_yaxes(autorange="reversed")

        fig.show()

        return fig




    def plot_CM_mplt(self, Y, Y_hat, title, fraction, percentagereference='recall'):
        # Generate a confusion matrix
        cm = metrics.confusion_matrix(Y, Y_hat)
        n_classes = len(np.unique(Y))
        # Build normalized confusion matrix
        classes = np.arange(Y.min(), Y.max() + 1)
        if percentagereference == 'recall':
            cm_norm = np.around(cm.astype('float')/cm.sum(axis=1)[:, np.newaxis], decimals=2)
            name = 'Percentage of true label'
        else:  # precision is calculated as percentages
            cm_norm = np.around(cm.astype('float')/cm.sum(axis=0)[np.newaxis, :], decimals=2)
            name = 'Percentage of predicted label'
        
        cm_norm_df = pd.DataFrame(cm_norm, index=classes, columns=classes)
        annotations = np.char.add( np.char.add((cm_norm*100).round(0).astype(int).astype(str), '%\n('), np.char.add((cm/1000).round(0).astype(int).astype(str), ')'))

        width, _ = set_size(width = self.width, fraction=fraction)
        fig, ax = plt.subplots(figsize=(width*1.15, width*1.15))

        cbar_kws=dict(label= name, format= PercentFormatter(xmax=1), use_gridspec=True,location="right",pad=0.01,shrink=0.85)
    
        sns.heatmap(cm_norm_df, ax = ax, annot=annotations, fmt = '', cmap='viridis', square=True, cbar_kws=cbar_kws, vmin = 0, vmax = 1, annot_kws={"size": 7})


        plt.title(title)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.gca().invert_yaxis()
        
        # fig.tight_layout()
        return fig
    

    def plot_timepredictions(self, data, windowsize, mean_smallest_timediff, fraction=1):
        # Scatter plot for Y on the first subplot

        # Create a subplot with 2 rows and 1 column
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=set_size(width=self.width, fraction=fraction, subplots=(2, 1)))

        # Scatter plot for Y_hat on the first subplot
        sns.scatterplot(x='timestamp', y='Y_hat', data=data, s=25, ax=ax1, label='Y_hat', linewidth = 0)

        sns.scatterplot(x='timestamp', y='label', data=data, s=8, ax=ax1, label='Y', linewidth=0)

        # Add window annotation
        window_start = data['timestamp'].iloc[0]
        window_end = window_start + mean_smallest_timediff * (windowsize-1)
        ax1.plot([window_start, window_start, window_end, window_end], [-0.5, -1, -1, -0.5], 'k-', linewidth=2)
        windowannotation = f'Window Size: {windowsize} frames\n ~{round(mean_smallest_timediff * windowsize, 2)} s'
        ax1.text(window_start, -1.5, windowannotation, verticalalignment='top')

        # Line plot for Y_score_true on the second subplot
        sns.lineplot(x='timestamp', y='Y_score_true', data=data, ax=ax2, label='Y_score_true')

        # Set titles and labels
        fig.suptitle('Prediction - label comparison')
        ax1.set_ylabel('label number')
        ax1.set_ylim(-1.5, 15)
        ax1.legend(frameon = True, loc = 'upper right')

        ax2.set_ylabel('score')
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel('time in s')
        ax2.xaxis.set_major_locator(plt.MultipleLocator(0.25))
        ax2.legend(frameon = True, loc = 'lower right')

        #plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        return fig
    
    def plot_timeanalysis2(self, sequenceinfos, frame_duration, fraction = 1):
        bin_width = 0.25

        # Create a figure with 2 subplots
        (length, width) = set_size(width=self.width, fraction=fraction, subplots=(1, 2))
        fig, axes = plt.subplots(1, 2, figsize=(length, width*1.5))

        # Create a histogram for 'sequencespan'
        sns.histplot(sequenceinfos['sequencespan'].dropna(), binwidth=bin_width, ax=axes[0])
        secax1 = axes[0].secondary_xaxis(location = 'top', functions=(lambda x: x/frame_duration, lambda x: x*frame_duration))
        secax1.set_xlabel('Frames')
        # axes[0].set_title('Temporal lenght of sequences')
        axes[0].set_xlabel('Seconds')
        axes[0].set_ylabel('Sequence count')

        # Create a histogram for 'gesturespan'
        sns.histplot(sequenceinfos['gesturespan'].dropna(), binwidth=bin_width, ax=axes[1])
        secax2 = axes[1].secondary_xaxis(location = 'top', functions=(lambda x: x/frame_duration, lambda x: x*frame_duration))
        secax2.set_xlabel('Frames')
        # axes[1].set_title('Temporal lenght of gestures')
        axes[1].set_xlabel('Seconds')
        axes[1].set_ylabel('Gesture count')

        # Get the maximum x and y limits
        x_max = max(axes[0].get_xlim()[1], axes[1].get_xlim()[1])
        y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])

        # Set the same x and y limits for both subplots
        axes[0].set_xlim(0, x_max)
        axes[1].set_xlim(0, x_max)
        axes[0].set_ylim(0, y_max)
        axes[1].set_ylim(0, y_max)

        fig.tight_layout()
        return fig

    def plot_timeanalysis(self, sequenceinfos, frame_duration, fraction = 1):
        bin_width = 0.25

        (length, width) = set_size(width=self.width, fraction=fraction, subplots=(1, 1))
        fig, axes = plt.subplots(1, 1, figsize=(length, width*1.3))

        # Create a histogram for 'gesturespan'
        sns.histplot(sequenceinfos['gesturespan'].dropna().unique(), binwidth=bin_width, ax=axes)
        secax2 = axes.secondary_xaxis(location = 'top', functions=(lambda x: x/frame_duration, lambda x: x*frame_duration))
        secax2.set_xlabel('Frames')
        # axes.set_title('Temporal lenght of gestures')
        axes.set_xlabel('Seconds')
        axes.set_ylabel('Gesture count')
        # axes.set_xscale('log')
        return fig

    def plot_framelabelshistogram(self, data, fraction = 1):
        fig, ax = plt.subplots(1, 1, figsize=set_size(width=self.width, fraction=fraction, subplots=(1, 1)))

        sns.histplot(data['label'].dropna(), binwidth=1, ax=ax)
        # data['label'].hist(bins=15, ax=ax)
        ax.set_xlabel('Label')
        ax.set_ylabel('Frame label count')
        # ax.set_yscale('log')
        # ax.set_title('Histogram frame classes')

        return fig

    def plot_result_boxplot(self, scores,  NLLrange = [0,20], percentrange = [0,100], fraction=1):
        notch = False
        width = 0.25
        n_ticks = 6

        if 'Validation Person' in scores.columns:
            categorie = 'Validation Person'
        elif 'Configuration' in scores.columns:
            categorie = 'Configuration'
        elif 'Classifier' in scores.columns:
            categorie = 'Classifier'
        else:
            ValueError('No matching categorie found')



        # Data preprocessing
        if 'accuracy_valid' in scores.columns:
            acc = 'accuracy_valid'
            f1 = 'f1_score_valid'
            nll = 'NLL_valid'
        else:
            acc = 'accuracy'
            f1 = 'f1_score'
            nll = 'NLL'

        n_classifier = scores.loc[:, categorie].unique().shape[0]

        scores_concat1 = scores.loc[:, [categorie, acc]]
        scores_concat2 = scores.loc[:, [categorie, f1]]
        scores_concat3 = scores.loc[:, [categorie, nll]]
        scores_concat_plot1 = scores_concat1.melt(id_vars=[categorie], var_name='metric', value_name='value')
        scores_concat_plot2 = scores_concat2.melt(id_vars=[categorie], var_name='metric', value_name='value')
        scores_concat_plot3 = scores_concat3.melt(id_vars=[categorie], var_name='metric', value_name='value')
        scores_concat_plot1['value'] = scores_concat_plot1['value'] * 100
        scores_concat_plot2['value'] = scores_concat_plot2['value'] * 100

        # Calculate positions for boxplots
        x_positions1 = np.arange(n_classifier) - 0.3  # Shifted to the left
        x_positions2 = np.arange(n_classifier) # Shifted to the right
        x_positions3 = np.arange(n_classifier) + 0.3 # Shifted to the right

        # Plot
        fig, ax1 = plt.subplots(1, 1, figsize=set_size(width=self.width, fraction=fraction, subplots=(1, 1)))

        # Create boxplot with manually controlled positions
        sns.boxplot(ax=ax1, data=scores_concat_plot1, x=categorie, y="value", color=self.colorpalette[0],
                    dodge=False, width=width, positions=x_positions1, notch = notch)


        ax1.set_ylabel('Accuracy, F1-Score in \%')
        ax1.set_ylim(percentrange[0], percentrange[1])
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}\%'))

        ax1.set_yticks(np.linspace(percentrange[0], percentrange[1], n_ticks*2-1), minor=True)
        ax1.yaxis.grid(True, which='minor')
        ax1.yaxis.labelpad = 0


        ax2 = ax1.twinx()
        sns.boxplot(ax=ax1, data=scores_concat_plot2, x=categorie, y="value", color=self.colorpalette[1],
                    dodge=False, width=width, positions=x_positions2, notch = notch)

        sns.boxplot(ax=ax2, data=scores_concat_plot3, x=categorie, y="value", color=self.colorpalette[2],
                    dodge=False, width=width, positions=x_positions3, notch = notch)

        ax2.set_ylim(NLLrange[1], NLLrange[0])
        ax2.grid(False)

        ax2.set_yticks(np.linspace(NLLrange[1], NLLrange[0], n_ticks))
        ax2.set_yticks(np.linspace(NLLrange[1], NLLrange[0], n_ticks*2-1), minor=True)
        ax2.set_ylabel('Negative Log Likelihood')

        legend_patches = [
            Patch(color=self.colorpalette[0], label='Accuracy'),
            Patch(color=self.colorpalette[1], label='F1-score'),
            Patch(color=self.colorpalette[2], label='NLL'),
        ]
        ax1.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)

        return fig

    def plot_result_boxplot_withref(self, scores, ref_scores, fraction=1):
        notch = False
        width = 0.25
        n_ticks = 6
        # Data preprocessing
        n_classifier = scores.loc[:, 'Classifier'].unique().shape[0]

        scores_concat1 = scores.loc[:, ['Classifier', 'accuracy_valid']]
        scores_concat2 = scores.loc[:, ['Classifier', 'f1_score_valid']]
        scores_concat3 = scores.loc[:, ['Classifier', 'NLL_valid']]
        scores_concat_plot1 = scores_concat1.melt(id_vars=['Classifier'], var_name='metric', value_name='value')
        scores_concat_plot2 = scores_concat2.melt(id_vars=['Classifier'], var_name='metric', value_name='value')
        scores_concat_plot3 = scores_concat3.melt(id_vars=['Classifier'], var_name='metric', value_name='value')
        scores_concat_plot1['value'] = scores_concat_plot1['value'] * 100
        scores_concat_plot2['value'] = scores_concat_plot2['value'] * 100
        ref_scors_pt = ref_scores.copy(deep = True)
        ref_scors_pt['accuracy_valid'] = ref_scors_pt['accuracy_valid'] * 100

        # Calculate positions for boxplots
        x_positions1 = np.arange(n_classifier) - 0.3  # Shifted to the left
        x_positions2 = np.arange(n_classifier) # Shifted to the right
        x_positions3 = np.arange(n_classifier) + 0.3 # Shifted to the right

        # Plotts
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=set_size(width=self.width, fraction=fraction), width_ratios=[3, 1.2])
        ax1_2 = ax1.twinx()

        sns.boxplot(ax=ax1, data=scores_concat_plot1, x="Classifier", y="value", color=self.colorpalette[0],
                    dodge=False, width=width, positions=x_positions1, notch = notch)

        sns.boxplot(ax=ax1, data=scores_concat_plot2, x="Classifier", y="value", color=self.colorpalette[1],
                    dodge=False, width=width, positions=x_positions2, notch = notch)

        sns.boxplot(ax=ax1_2, data=scores_concat_plot3, x="Classifier", y="value", color=self.colorpalette[2],
                    dodge=False, width=width, positions=x_positions3, notch = notch)

        sns.barplot(ax=ax2, data=ref_scors_pt, x="Source", y="accuracy_valid", color=self.colorpalette[0], dodge=False, width=width)

        # Formatitting

        ax1.set_ylabel('Accuracy, F1-Score in \%')
        ax1.set_ylim(0, 100)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}\%'))

        ax1.set_yticks(np.linspace(0, 100, n_ticks*2-1), minor=True)
        ax1.yaxis.grid(True, which='minor')
        
        ax1.yaxis.labelpad = 0

        ax1_2.set_ylim(2, 0)
        ax1_2.grid(False)

        ax1_2.set_yticks(np.linspace(2, 0, n_ticks))
        ax1_2.set_yticks(np.linspace(2, 0, n_ticks*2-1), minor=True)
        ax1_2.set_ylabel('Negative Log Likelihood')

        ax2.set_ylabel('Accuracy in \%')
        ax2.set_ylim(0, 100)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}\%'))

        ax2.set_yticks(np.linspace(0, 100, n_ticks))
        ax2.set_yticks(np.linspace(0, 100, n_ticks*2-1), minor=True)
        ax2.yaxis.grid(True, which='minor')
        ax2.yaxis.labelpad = 0


        legend_patches = [
            Patch(color=self.colorpalette[0], label='Accuracy'),
            Patch(color=self.colorpalette[1], label='F1-score'),
            Patch(color=self.colorpalette[2], label='NLL'),
        ]
        fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)
        fig.tight_layout()
        return fig



    def plot_resultbars_ODHG(self, scores, fraction = 1, NLLrange = [0,18], percentrange = [0,100], n_ticks = 6):
        # data preprocessing
        if 'Person' in scores.columns:
            categorie = 'Person'
        elif 'Configuration' in scores.columns:
            categorie = 'Configuration'
        else:
            ValueError('No matching categorie found')

        n_classifier = scores.loc[:, categorie].unique().shape[0]

        scores_concat1 = scores.loc[:, [categorie, 'accuracy', 'f1_score']]
        scores_concat2 = scores.loc[:, [categorie, 'NLL']]
        scores_concat_plot1 = scores_concat1.melt(id_vars=[categorie], var_name='metric', value_name='value')
        scores_concat_plot2 = scores_concat2.melt(id_vars=[categorie], var_name='metric', value_name='value')
        scores_concat_plot1['value'] = scores_concat_plot1['value'] * 100
        NLL_categoriekeys = scores_concat_plot2.loc[:, categorie].unique()
        NLL_meanvalues = scores_concat_plot2.loc[:, [categorie,'value']].groupby(categorie).mean()
        # plot
        fig, ax1 = plt.subplots(1,1, figsize=set_size(width=self.width, fraction=fraction, subplots=(1, 1)))

        colorpalette2 = self.colorpalette[2:] + self.colorpalette[:2]

        sns.barplot(ax = ax1, data=scores_concat_plot1, x=categorie, y="value", hue="metric", errorbar = lambda x: np.quantile(x, [0.025, 0.975]), dodge = False,width = 0.3, palette=self.colorpalette)

        ax1.set_ylabel('Accuracy, F1-Score in \%')

        ax1.set_ylim(percentrange[0], percentrange[1])
        ax1.set_yticks(np.linspace(percentrange[0], percentrange[1], n_ticks))
        ax1.set_yticks(np.linspace(percentrange[0], percentrange[1], n_ticks*2-1), minor=True)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}\%'))
        ax1.yaxis.grid(True, which='minor')

        ax1_2 = ax1.twinx()
        sns.barplot(ax = ax1_2, data=scores_concat_plot2, x=categorie, y="value", hue="metric", errorbar = lambda x: np.quantile(x, [0.025, 0.975]), dodge = False, width = 0.3, palette=colorpalette2)
        # plt.bar(ax = ax1_2, x=df.x, y=2*np.ones(len(df))-df.y, bottom= df.y )
        for idx in np.arange(n_classifier):
            ax1.patches[idx].set_x(ax1.patches[idx].get_x() - 0.3)
            ax1_2.patches[idx].set_x(ax1_2.patches[idx].get_x() + 0.3)
            ax1_2.patches[idx].set_y(NLL_meanvalues.loc[NLL_categoriekeys[idx], 'value'])
            ax1_2.patches[idx].set_height(NLLrange[1] - NLL_meanvalues.loc[NLL_categoriekeys[idx], 'value']) # next level murks

            ax1.get_lines()[idx].set_xdata(ax1.get_lines()[idx].get_xdata() - 0.3)
            ax1_2.get_lines()[idx].set_xdata(ax1_2.get_lines()[idx].get_xdata() + 0.3)

        ax1_2.grid(False)
        ax1_2.set_ylim(NLLrange[1], NLLrange[0])
        ax1_2.set_yticks(np.linspace(NLLrange[1], NLLrange[0], n_ticks))
        ax1_2.set_yticks(np.linspace(NLLrange[1], NLLrange[0], n_ticks*2-1), minor=True)
        ax1_2.set_ylabel('Negative Log Likelihood')
        ax1_2.get_legend().remove()

        # Create custom legend
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax1_2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = ['Accuracy', 'F1-score', 'NLL']
        ax1.legend(handles, labels)
        sns.move_legend(ax1, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,)
        fig.tight_layout()

        return fig

def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)