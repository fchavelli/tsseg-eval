
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from typing import Tuple, Union, Dict, List, Any

from baselines.patss.embedding import PatternBasedEmbedding

# If you want to zoom in on the figure
ZOOM_IN = False
ZOOM_IN_RANGE: Union[None, Tuple] = None

# Height of the different elements
HEIGHTS = {
    'title': 0.4,
    'attribute': 0.7,
    'distribution': 0.5,
    'segment': 0.5,
    'embedding_per_pattern': 0.1
}


def update_runtime_configuration_parameters(
        zoom_in_range: Tuple = None,
        heights: Dict[str, float] = None,
        line_colors: List = None,
        rc_params: Dict[str, Any] = None) -> None:
    """
    Update the configuration parameters for visualizing the semantic segmentation
    with gradual state transitions and the embedding matrix.

    Parameters
    ----------
    zoom_in_range : Tuple, default=None
        On which range the figure should be zoomed in. The tuple provides the start
        and end point of the interval to show in the figure. If no value is provided,
        the entire time series will be shown.
    heights : Dict[str, float], default=None
        The resulting figure contains various components, each with a dedicated height.
        This parameter states the relative height of each component. The dictionary can
        have the following keys:

            - ``'title'``: The space for plotting the title above each component;

            - ``'attribute'``: The space for plotting each attribute of the time series;

            - ``'distribution'``: The space for the segment probabilities;

            - ``'segment'``: The space for showing each individual segment;

            - ``'embedding_per_pattern'``: The space for a single pattern in the embedding matrix,
              thus the height of the embedding matrix will be this value multiplied by the number
              of patterns in the embedding.

        It is not necessary to provide all keys upon calling this function.
    line_colors : List, default=None
        A list of colors that should be used to colour the trend data.
    rc_params : Dict[str, Any], default=None
        The matplotlib.pyplot runtime configuration parameters.
        https://matplotlib.org/stable/users/explain/customizing.html
    """
    global ZOOM_IN, ZOOM_IN_RANGE, HEIGHTS

    # Update the zoom
    ZOOM_IN = zoom_in_range is not None
    ZOOM_IN_RANGE = zoom_in_range

    # Update the heights
    if heights is not None:
        HEIGHTS.update(heights)

    # Update the default colors
    if line_colors is not None:
        plt.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=line_colors)

    # Update the matplotlib.pyplot rc parameters
    if rc_params is not None:
        plt.rcParams.update(rc_params)


# Default configuration
# update_runtime_configuration_parameters(
# { for SDM 24
#     # https://matplotlib.org/stable/tutorials/introductory/customizing.html
#     'line_colors': [(0.3, 0.3, 0.3)],
#     'rc_params': {
#         'font.family': 'Times New Roman',
#         'figure.titlesize': 8,
#         'axes.titlesize': 8,
#         'lines.linewidth': 0.7,
#         'axes.labelsize': 6,
#         'xtick.labelsize': 6,
#         'ytick.labelsize': 6,
#         'savefig.bbox': 'tight',
#         'savefig.pad_inches': 0.01,
#         'figure.figsize': (100 / 25.4, 50 / 25.4),  # (100mm, 50mm) converted to inches
#         'figure.subplot.hspace': 0.3,
#         'savefig.format': 'pdf',
#         'legend.frameon': False,
#         'legend.fontsize': 4,
#         'legend.labelspacing': 0,
#         'legend.borderpad': 0,
#         'legend.borderaxespad': 0
#     }
# })

update_runtime_configuration_parameters(
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html
    line_colors=[(0.3, 0.3, 0.3)],
    rc_params={
        'font.family': 'Times New Roman',
        'axes.titlesize': 32,
        'lines.linewidth': 0.7,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 6,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.01,
        'figure.figsize': (20, 20),
        'figure.subplot.hspace': 0.3,
        'legend.frameon': False,
        'legend.fontsize': 16,
        'legend.labelspacing': 0,
        'legend.borderpad': 0,
        'legend.borderaxespad': 0
    }
)


def visualize_segmentation(
        pattern_based_embedding: PatternBasedEmbedding,
        segmentation: np.ndarray,
        ground_truth_segmentation: np.ndarray = None,
        path: str = None,
        plot_distribution: bool = True,
        plot_segments: bool = True,
        plot_embedding: bool = True,
        time_steps: np.array = None) -> plt.Figure:
    """
    Visualization the time series data, segmentation with gradual state
    transitions, and the pattern based embedding.

    Parameters
    ----------
    pattern_based_embedding : PatternBasedEmbedding
        The pattern-based embedding to visualize.
    segmentation : np.ndarray of shape (n_segments, n_samples)
        The predicted segmentation with gradual transitions to visualize.
    ground_truth_segmentation : np.ndarray of shape (n_segments, n_samples), default=None
        The ground truth semantic segmentation with gradual state transitions.
        If not provided, no ground truth will not be shown.
    path : str, default=None
        The path to which the resulting figure should be saved. If no path
        is provided, the figure will not be saved.
    plot_distribution : bool, default=True
        Whether to plot the predicted segment probabilities as a distribution.
    plot_segments : bool, default=True
        Whether to plot the semantic segments on top of the time series to better
        inspect in which parts of the time series each semantic segment is active.
    plot_embedding : bool, default=True
        Whether to plot the embedding matrix.
    time_steps : np.array, default=None
        Custom time steps to set on the x-axis. If no time steps are provided, the
        values 0, ..., N will be used, with N the number of samples in the time series.

    Returns
    -------
    fig : plt.Figure
        The figure containing all the plots.
    """
    # Make room for plotting the data
    nb_attributes = pattern_based_embedding.time_series.shape[1]
    height_ratios = [HEIGHTS['title']] + [HEIGHTS['attribute'] for _ in range(nb_attributes)]
    # Make room for plotting the segment distribution
    if plot_distribution:
        height_ratios += [HEIGHTS['title'], HEIGHTS['distribution']]
        if ground_truth_segmentation is not None:
            height_ratios += [HEIGHTS['title'], HEIGHTS['distribution']]
    # Make room for plotting the segments
    if plot_segments:
        height_ratios += [HEIGHTS['title']] + [HEIGHTS['segment'] for _ in range(segmentation.shape[0])]
    # Make room for plotting the embedding
    if plot_embedding:
        nb_patterns = pattern_based_embedding.embedding_matrix.shape[0]
        height_ratios += [HEIGHTS['title']] + [HEIGHTS['embedding_per_pattern']*nb_patterns]

    # Initialize the axis
    fig, axs = plt.subplots(len(height_ratios), 1, height_ratios=height_ratios, sharex='all')
    axs_counter = 0

    # Format the parameters
    if time_steps is None:
        time_steps = np.arange(pattern_based_embedding.time_series.shape[0])
    colors = __get_colours(segmentation.shape[0])
    segment_names = [chr(ord('A') + i) for i in range(segmentation.shape[0])]
    if segmentation.shape[1] < pattern_based_embedding.time_series.shape[0]:
        ratio = pattern_based_embedding.time_series.shape[0] // segmentation.shape[1]
        segmentation = np.repeat(segmentation, ratio, axis=1)

    # Plot the time series data
    __create_title_plot('The time series data', axs[axs_counter])
    axs_counter += 1
    for i in range(pattern_based_embedding.time_series.shape[1]):
        axs[axs_counter].plot(time_steps, pattern_based_embedding.time_series[:, i])
        axs[axs_counter].tick_params(axis='y', which='both')
        axs[axs_counter].axis('off')
        axs_counter += 1

    if plot_distribution:
        if ground_truth_segmentation is not None:
            __create_title_plot('Ground truth segment probability', axs[axs_counter])
            axs_counter += 1
            # Order the colors such that they align more with the prediction
            distance_matrix = np.linalg.norm(segmentation[:, np.newaxis] - ground_truth_segmentation, axis=2)
            row_indices, col_indices = linear_sum_assignment(distance_matrix)
            ordered_colors = colors.copy()
            for i, j in zip(row_indices, col_indices):
                ordered_colors[j] = colors[i]
            __create_distribution_plot(time_steps, ground_truth_segmentation, ordered_colors, segment_names, axs[axs_counter])
            axs_counter += 1

        __create_title_plot('Predicted segment probability', axs[axs_counter])
        axs_counter += 1
        __create_distribution_plot(time_steps, segmentation, colors, segment_names, axs[axs_counter])
        axs[axs_counter].legend(loc='lower left', bbox_to_anchor=(0.0, 1))
        axs_counter += 1

    if plot_segments:
        __create_title_plot('The semantic segments on top of the time series', axs[axs_counter])
        axs_counter += 1
        for i in range(segmentation.shape[0]):
            __create_segment_plot(time_steps, segmentation[i, :], pattern_based_embedding.time_series, colors[i], axs[axs_counter])
            axs[axs_counter].set_ylabel(segment_names[i], rotation='horizontal', ha='right', va='center')
            axs_counter += 1

    if plot_embedding:
        __create_title_plot('The embedding matrix', axs[axs_counter])
        axs_counter += 1
        __create_embedding_plot(time_steps, pattern_based_embedding, axs[axs_counter])
        axs_counter += 1

    axs[-1].set_xlabel('Time')
    if path is not None:
        fig.savefig(fname=path)

    return fig


def __create_title_plot(title: str, ax: plt.Axes) -> None:
    ax.set_title(title, y=0.0)
    ax.axis('off')
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)


def __get_colours(nb_colours):
    return [cm.jet(x) for x in np.linspace(0.0, 1.0, nb_colours)]


def __create_distribution_plot(time_steps: np.array, segmentation: np.ndarray, colors, segment_names, ax: plt.Axes) -> None:
    # Plot the predicted probability distributions
    for i in range(segmentation.shape[0]):
        distribution = segmentation[i, :]
        ax.plot(time_steps, distribution)
        ax.plot(time_steps, distribution, color=colors[i], label='Probability of segment ' + segment_names[i])
    ax.set_yticks([0.0, 1.0])


def __create_segment_plot(time_steps: np.array, segment_values: np.array, time_series: np.ndarray, color, ax: plt.Axes) -> None:
    # Plot the data
    for i in range(time_series.shape[1]):
        ax.plot(time_steps, time_series[:, i])
    # Plot the segment on top of the data
    segmentation_values_expand = np.expand_dims(segment_values, axis=0)
    cmap = LinearSegmentedColormap.from_list('cmap_gradient', ['white', color])
    extent = time_steps.min(), time_steps.max(), np.min(time_series), np.max(time_series)
    ax.imshow(segmentation_values_expand, vmin=0.0, vmax=1.0, aspect='auto', cmap=cmap, alpha=0.5, extent=extent)
    ax.set_yticks([])


def __create_embedding_plot(time_steps: np.array, pattern_based_embedding: PatternBasedEmbedding, ax: plt.Axes) -> None:
    embedding_labels = ['%s (%d, %s)' % (
        pattern_based_embedding.patterns.loc[i, 'pattern'],
        pattern_based_embedding.patterns.loc[i, 'interval'],
        np.round(pattern_based_embedding.patterns.loc[i, 'rsupport'], 2)
    ) for i in pattern_based_embedding.patterns.index]
    ax.set_yticks(ticks=np.arange(len(embedding_labels))+0.5, labels=embedding_labels[::-1])
    extent = time_steps.min(), time_steps.max(), 0, pattern_based_embedding.embedding_matrix.shape[0]
    ax.imshow(pattern_based_embedding.embedding_matrix, aspect='auto', cmap='Greys', extent=extent)

