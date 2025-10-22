'''
Created by Chengyu on 2021/12/12.
Views defined in StateCorr.
'''

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap


from scipy.optimize import linear_sum_assignment

from TSpy.utils import z_normalize,calculate_density_matrix, calculate_velocity_list, find
from TSpy.color import *

def plot_mts(X, groundtruth=None, prediction=None, figsize=(18,2), show=False):
    '''
    X: Time series, whose shape is (T, C) or (T, 1), (T, ) for uts, where T is length, C
        is the number of channels.
    groundtruth: can be of shape (T,) or (T, 1).
    prediction: can be of shape (T,) or (T, 1) or a dict of predictions with algo names as keys.
    '''

    if groundtruth is None and prediction is None:
        plt.plot(X)

    elif groundtruth is not None and prediction is not None:
        if isinstance(prediction, dict):
            num_predictions = len(prediction)
            plt.figure(figsize=(16, 4 + num_predictions))
            grid = plt.GridSpec(5 + num_predictions, 1)
        else:
            plt.figure(figsize=(14, 4))
            grid = plt.GridSpec(5, 1)

        ax1 = plt.subplot(grid[0:3])
        plt.title('Time Series')
        plt.yticks([])
        plt.plot(X)

        plt.subplot(grid[3], sharex=ax1)
        plt.title('State Sequence (Groundtruth)')
        plt.yticks([])
        plt.imshow(groundtruth.reshape(1, -1), aspect='auto', cmap='tab20c',
                   interpolation='nearest')

        if isinstance(prediction, dict):
            for i, (algo, pred) in enumerate(prediction.items()):
                plt.subplot(grid[4 + i], sharex=ax1)
                metrics = pred[1]
                metric_str = ', '.join([f'{name}: {round(value, 4)}' for name, value in zip(metrics.columns, metrics.values[0]) if name != 'dataset'])
                plt.title(f'{algo} ({metric_str})')
                plt.yticks([])
                plt.imshow(pred[0].reshape(1, -1), aspect='auto', cmap='tab20c',
                           interpolation='nearest')
        else:
            plt.subplot(grid[4], sharex=ax1)
            plt.title('State Sequence (Prediction)')
            plt.yticks([])
            plt.imshow(prediction.reshape(1, -1), aspect='auto', cmap='tab20c',
                       interpolation='nearest')

    else:
        if groundtruth is not None:
            plt.figure(figsize=(16, 4))
            grid = plt.GridSpec(4, 1)
            ax1 = plt.subplot(grid[0:3])
            plt.title('Time Series')
            plt.yticks([])
            plt.plot(X)

            plt.subplot(grid[3], sharex=ax1)
            plt.title('State Sequence')
            plt.yticks([])
            plt.imshow(groundtruth.reshape(1, -1), aspect='auto', cmap='tab20c',
                       interpolation='nearest')
        
        if prediction is not None:
            plt.figure(figsize=(16, 4))
            grid = plt.GridSpec(4, 1)
            ax1 = plt.subplot(grid[0:3])
            plt.title('Time Series')
            plt.yticks([])
            plt.plot(X)

            plt.subplot(grid[3], sharex=ax1)
            plt.title('State Sequence (Prediction)')
            plt.yticks([])
            plt.imshow(prediction.reshape(1, -1), aspect='auto', cmap='tab20c',
                       interpolation='nearest')

    plt.tight_layout()
    if show:
        plt.show()

def plot_mulvariate_time_series(series, figsize=(18,2), separate=False, save_path=None, show=False):
    _, num_channel = series.shape
    plt.style.use('ggplot')
    if not separate:
        plt.figure(figsize=figsize)
        for i in range(num_channel):
            plt.plot(series[:,i])
    else:
        _, ax = plt.subplots(nrows=num_channel, sharex=True, figsize=figsize)
        for i, ax_ in enumerate(ax):
            ax_.plot(series[:,i])
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

def plot_mulvariate_time_series_and_label(series, groundtruth=None, label=None, figsize=(18,2)):
    _, num_channel = series.shape
    plt.style.use('ggplot')
    _, ax = plt.subplots(nrows=2, sharex=True, figsize=figsize)

    for i in range(num_channel):
        ax[0].plot(series[:,i])
    
    if groundtruth is not None:
        ax[1].step(np.arange(len(groundtruth)), groundtruth, label='groundtruth')

    if label is not None:
        ax[1].step(np.arange(len(label)), label, label='prediction')

    plt.legend()
    plt.tight_layout()
    plt.show()

def embedding_space(embeddings, label=None, alpha=0.8, s=0.1, color='blue', show=False):
    color_list = ['b', 'r', 'g', 'purple', 'y', 'gray']
    embeddings = np.array(embeddings)
    x = embeddings[:,0]
    y = embeddings[:,1]
    # plt.style.use('ggplot')
    plt.style.use('classic')
    # plt.style.use('bmh')
    plt.figure(figsize=(4,4))
    plt.grid()
    i = 0
    if label is not None:
        for l in set(label):
            idx = np.argwhere(label==l)
            plt.scatter(x[idx],y[idx],alpha=alpha,s=s, color=color_list[i])
            # plt.scatter(x[idx],y[idx],alpha=alpha,s=s)
            i+=1
    else:
        plt.scatter(x,y,alpha=alpha,s=s)
    if show:
        # plt.tight_layout()
        plt.show()

# arrow map.
def arrow_map(feature_list, n=100, t=100):
    feature_list = np.array(feature_list)
    x = feature_list[:,0]
    y = feature_list[:,1]
    velocity_list_x, velocity_list_y = calculate_velocity_list(feature_list,interval=t)
    h = w = n
    h_start = np.min(y)
    h_end = np.max(y)
    h_step = (h_end-h_start)/h
    w_start = np.min(x)
    w_end = np.max(x)
    w_step = (w_end-w_start)/w

    row_partition = []
    for i in range(h):
        row_partition.append(find(y,h_start+i*h_step,h_start+(i+1)*h_step))

    U = []
    V = []
    for col_idx in row_partition:
        col = x[col_idx]
        U_col = []
        V_col = []
        for i in range(w):
            idx = find(col,w_start+i*w_step,w_start+(i+1)*w_step)
            x_list = velocity_list_x[idx]
            x_mean = np.mean(x_list)
            y_list = velocity_list_y[idx]
            y_mean = np.mean(y_list)
            U_col.append(x_mean)
            V_col.append(y_mean)
        U.append(np.array(U_col))
        V.append(np.array(V_col))
    U = np.array(U)
    V = np.array(V)
    # U=U.T
    # V=V.T
    U[np.isnan(U)]=0
    V[np.isnan(V)]=0
    # U = normalize(U)
    # V = normalize(V)

    x_, y_ = np.meshgrid(np.linspace(-3, 3, n),
                   np.linspace(-3, 3, n))
    M = np.hypot(U, V)
    fig1, ax1 = plt.subplots()
    ax1.set_title('Arrows scale with plot width, not view')
    Q = ax1.quiver(x_, y_, U, V, M, units='width')

    plt.show()
    # fig3, ax3 = plt.subplots()
    # ax3.set_title("pivot='tip'; scales with x view")
    # M = np.hypot(U, V)
    # Q = ax3.quiver(x_, y_, U, V, M, units='x', pivot='tip', width=0.022,
    #            scale=1 / 0.15)
    # qk = ax3.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
    #                coordinates='figure')
    # ax3.scatter(x_, y_, color='0.5', s=1)

def flow_map(feature_list, n=50, t=100):
    feature_list = np.array(feature_list)
    x = feature_list[:,0]
    y = feature_list[:,1]
    velocity_list_x, velocity_list_y = calculate_velocity_list(feature_list,interval=t)
    h = w = n
    h_start = np.min(y)
    h_end = np.max(y)
    h_step = (h_end-h_start)/h
    w_start = np.min(x)
    w_end = np.max(x)
    w_step = (w_end-w_start)/w

    row_partition = []
    for i in range(h):
        row_partition.append(find(y,h_start+i*h_step,h_start+(i+1)*h_step))
    
    row_partition = list(reversed(row_partition))

    U = []
    V = []
    for col_idx in row_partition:
        col = x[col_idx]
        U_col = []
        V_col = []
        for i in range(w):
            idx = find(col,w_start+i*w_step,w_start+(i+1)*w_step)
            x_list = velocity_list_x[idx]
            x_mean = np.mean(x_list)
            y_list = velocity_list_y[idx]
            y_mean = np.mean(y_list)
            U_col.append(x_mean)
            V_col.append(y_mean)
            # print(U_col)
        U.append(np.array(U_col))
        V.append(np.array(V_col))
    U = np.array(U)
    V = np.array(V)
    U[np.isnan(U)]=0
    V[np.isnan(V)]=0
    # U = normalize(U)
    # V = normalize(V)

    x_, y_ = np.meshgrid(np.linspace(-3, 3, n),
                   np.linspace(-3, 3, n))

    fig0, ax0 = plt.subplots()
    strm = ax0.streamplot(x_, y_, U, V, color=U, linewidth=2, cmap=plt.cm.autumn)
    fig0.colorbar(strm.lines)
    plt.show()

def density_map_3d(feature_list, n=100, show=False, figsize=(6,6), op=None, t = 1):
    density_matrix,x_s,x_e,y_s,y_e = calculate_density_matrix(feature_list,n)

    density_matrix = z_normalize(density_matrix)

    x, y = np.meshgrid(np.linspace(x_s, x_e, n),
                   np.linspace(y_s, y_e, n))

    if op == 'normalize':
        density_matrix = z_normalize(density_matrix)
    # elif op == 'standardize':
    #     density_matrix = standardize(density_matrix)

    # Plot the surface.
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, density_matrix, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False, vmax=t)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    if show:
        plt.show()

# color = 'viridis','plasma','inferno','cividis','magma'
# pre-process = 'normalize', 'standardize'
def density_map(feature_list, n=101, show=False, figsize=(4,4), fontsize=10, color = 'plasma', t=1):
    density_matrix,_,_,_,_ = calculate_density_matrix(feature_list,n)

    # normalize
    density_matrix = z_normalize(density_matrix)

    max = np.max(density_matrix)
    min = np.min(density_matrix)
    
    plt.figure(figsize=figsize)
    plt.matshow(density_matrix, cmap=color,vmax=max*t,vmin=min,fignum=0)
    plt.tick_params(labelsize=fontsize)
    cb = plt.colorbar(fraction=0.045)
    cb.ax.tick_params(labelsize=fontsize)
    if show:
        plt.show()
    return density_matrix

#------------------------------------------------------
# Added by Felix Chavelli, 2025-04
#------------------------------------------------------

# This function maps predicted labels to match true labels using the Hungarian algorithm.
def map_predicted_labels(labels_true, labels_pred):
    """
    Maps predicted labels to match true labels using the Hungarian algorithm,
    ensuring the number of unique output labels equals the number of unique
    input predicted labels.

    Parameters:
        labels_true: numpy array of ground truth labels.
        labels_pred: numpy array of predicted labels.

    Returns:
        mapped_pred: numpy array of predicted labels after mapping.
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    original_shape = labels_pred.shape

    if labels_pred.size == 0:
        return np.array([])

    all_unique_pred = np.unique(labels_pred)

    # Handle cases where true labels are empty or comparison is not possible
    if labels_true.size == 0:
        mapping = {label: i for i, label in enumerate(all_unique_pred)}
        mapped_pred_flat = np.array([mapping[val] for val in labels_pred.ravel()])
        return mapped_pred_flat.reshape(original_shape)

    # Determine common length for comparison
    compare_len = min(len(labels_pred.ravel()), len(labels_true.ravel()))
    if compare_len == 0: # No overlap to compare
        mapping = {label: i for i, label in enumerate(all_unique_pred)}
        mapped_pred_flat = np.array([mapping[val] for val in labels_pred.ravel()])
        return mapped_pred_flat.reshape(original_shape)

    true_comp = labels_true.ravel()[:compare_len]
    pred_comp = labels_pred.ravel()[:compare_len]

    unique_pred_comp = np.unique(pred_comp)
    unique_true_comp = np.unique(true_comp)

    # Cost matrix: negative overlap
    cost_matrix = np.zeros((len(unique_pred_comp), len(unique_true_comp)))
    for i, p_label in enumerate(unique_pred_comp):
        for j, t_label in enumerate(unique_true_comp):
            overlap = np.sum((pred_comp == p_label) & (true_comp == t_label))
            cost_matrix[i, j] = -overlap

    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    hungarian_map = {unique_pred_comp[i]: unique_true_comp[j] for i, j in zip(row_ind, col_ind)}

    # Build the final mapping ensuring unique outputs for all unique predicted labels
    final_map = {}
    used_targets = set()

    # 1. Apply Hungarian assignments where the target is available
    for p_label, t_label in hungarian_map.items():
        if t_label not in used_targets:
            final_map[p_label] = t_label
            used_targets.add(t_label)
        # else: Target taken, p_label will be handled in step 2

    # 2. Assign remaining unique predicted labels
    next_new_label = 0
    for p_label in all_unique_pred:
        if p_label not in final_map: # If not assigned yet
            # Try assigning p_label to itself if available
            if p_label not in used_targets:
                final_map[p_label] = p_label
                used_targets.add(p_label)
            # Otherwise, find the next available non-negative integer
            else:
                while next_new_label in used_targets:
                    next_new_label += 1
                final_map[p_label] = next_new_label
                used_targets.add(next_new_label)

    # Apply the final map to the original full predicted sequence
    mapped_pred_flat = np.array([final_map[val] for val in labels_pred.ravel()])
    mapped_pred = mapped_pred_flat.reshape(original_shape)

    return mapped_pred

# This function plots multivariate time series with color-coded states.
def plot_mts_map(X, groundtruth=None, prediction=None, figsize=(18,2), show=False):
    '''
    X: Time series, whose shape is (T, C) or (T, 1), (T, ) for uts, where T is length, C
        is the number of channels.
    groundtruth: can be of shape (T,) or (T, 1).
    prediction: can be of shape (T,) or (T, 1) or a dict of predictions with algo names as keys.
    '''

    # If groundtruth is provided, determine common vmin and vmax for consistent colors.
    if groundtruth is not None:
        gt_array = groundtruth.ravel()
        unique_states = np.unique(gt_array)
        vmin = unique_states.min() - 0.5
        vmax = unique_states.max() + 0.5

    if groundtruth is None and prediction is None:
        plt.plot(X)

    elif groundtruth is not None and prediction is not None:
        if isinstance(prediction, dict):
            num_predictions = len(prediction)
            plt.figure(figsize=(16, 4 + num_predictions))
            grid = plt.GridSpec(5 + num_predictions, 1)
        else:
            plt.figure(figsize=(14, 4))
            grid = plt.GridSpec(5, 1)

        ax1 = plt.subplot(grid[0:3])
        #plt.title('Time Series')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        plt.yticks([])
        plt.plot(X)

        plt.subplot(grid[3], sharex=ax1)
        plt.title('State Sequence (Groundtruth)')
        plt.yticks([])
        plt.imshow(groundtruth.reshape(1, -1), aspect='auto', cmap='tab20c',
                   interpolation='nearest', vmin=vmin, vmax=vmax)

        if isinstance(prediction, dict):
            for i, (algo, pred) in enumerate(prediction.items()):
                # pred[0] is the prediction, pred[1] is the metrics dataframe.
                mapped_pred = map_predicted_labels(groundtruth, pred[0])
                
                plt.subplot(grid[4 + i], sharex=ax1)
                metrics_df = pred[1]
                metric_str = ', '.join([f'{name}: {value:.2f}' for name, value in zip(metrics_df.columns, metrics_df.values[0]) if name != 'dataset'])
                plt.title(f'{algo} ({metric_str})')
                plt.yticks([])
                plt.imshow(mapped_pred.reshape(1, -1), aspect='auto', cmap='tab20c',
                           interpolation='nearest', vmin=vmin, vmax=vmax)
        else:
            mapped_prediction = map_predicted_labels(groundtruth, prediction)
            plt.subplot(grid[4], sharex=ax1)
            plt.title('State Sequence (Prediction)')
            plt.yticks([])
            plt.imshow(mapped_prediction.reshape(1, -1), aspect='auto', cmap='tab20c',
                       interpolation='nearest', vmin=vmin, vmax=vmax)

    else:
        if groundtruth is not None:
            plt.figure(figsize=(16, 4))
            grid = plt.GridSpec(4, 1)
            ax1 = plt.subplot(grid[0:3])
            plt.title('Time Series')
            plt.yticks([])
            plt.plot(X)

            plt.subplot(grid[3], sharex=ax1)
            plt.title('State Sequence')
            plt.yticks([])
            plt.imshow(groundtruth.reshape(1, -1), aspect='auto', cmap='tab20c',
                       interpolation='nearest', vmin=vmin, vmax=vmax)
        
        if prediction is not None:
            mapped_prediction = map_predicted_labels(groundtruth, prediction) if groundtruth is not None else prediction
            plt.figure(figsize=(16, 4))
            grid = plt.GridSpec(4, 1)
            ax1 = plt.subplot(grid[0:3])
            plt.title('Time Series')
            plt.yticks([])
            plt.plot(X)

            plt.subplot(grid[3], sharex=ax1)
            plt.title('State Sequence (Prediction)')
            plt.yticks([])
            if groundtruth is not None:
                plt.imshow(mapped_prediction.reshape(1, -1), aspect='auto', cmap='tab20c',
                           interpolation='nearest', vmin=vmin, vmax=vmax)
            else:
                plt.imshow(mapped_prediction.reshape(1, -1), aspect='auto', cmap='tab20c',
                           interpolation='nearest')

    plt.tight_layout()
    if show:
        plt.show()

# This function plots multivariate time series with color-coded states and error bars.
def plot_sms(X, groundtruth, mapped_prediction, errors_list=None, show_error_impact=False, show_error_type=False, figsize=(18, 6), show=False):
    """
    Plots time series, ground truth, mapped prediction, and optional error bars.

    Parameters:
        X: Time series data, shape (T, C).
        groundtruth: Ground truth labels, shape (T,) or (T, 1).
        mapped_prediction: Predicted labels already mapped to ground truth, shape (T,) or (T, 1).
        errors_list: List of error dictionaries from state_matching_score (with return_errors=True).
        show_error_impact (bool): If True, show an error bar with impact intensity (penalty/size).
        show_error_type (bool): If True, show an error bar with error types.
        figsize (tuple): Figure size.
        show (bool): If True, display the plot.
    """
    if errors_list is not None and (groundtruth is None or mapped_prediction is None):
        raise ValueError("groundtruth and mapped_prediction must be provided when errors_list is given.")

    n = len(X)
    if groundtruth is not None:
        gt_array = groundtruth.ravel()
        unique_states = np.unique(gt_array)
        vmin = unique_states.min() - 0.5
        vmax = unique_states.max() + 0.5
    elif mapped_prediction is not None:
        # Fallback if only mapped_prediction is somehow provided without groundtruth
        pred_array = mapped_prediction.ravel()
        unique_states = np.unique(pred_array)
        vmin = unique_states.min() - 0.5
        vmax = unique_states.max() + 0.5
    else: # Should not happen based on initial check, but for safety
        vmin = None
        vmax = None


    num_error_plots = sum([show_error_impact, show_error_type])
    total_plots = 3 + num_error_plots # TS + GT + Pred + Error bars

    # Adjust figsize height based on the number of plots - reduced base height
    base_height_per_plot = 0.8 # Reduced height per plot
    ts_height_ratio = 5 # Time series plot takes more space
    fig_height = base_height_per_plot * (ts_height_ratio + (total_plots - 1))
    figsize = (figsize[0], fig_height)

    plt.figure(figsize=figsize)
    # GridSpec rows: ts_height_ratio for TS, 1 for GT, 1 for Pred, 1 for each error plot
    height_ratios = [ts_height_ratio] + [1] * (total_plots - 1)
    grid = plt.GridSpec(total_plots, 1, height_ratios=height_ratios)

    # 1. Plot Time Series
    ax1 = plt.subplot(grid[0])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    plt.yticks([])
    plt.plot(X)
    #plt.title('Time Series')

    # 2. Plot Ground Truth
    ax2 = plt.subplot(grid[1], sharex=ax1)
    plt.title('State Sequence (Groundtruth)')
    plt.yticks([])
    if groundtruth is not None:
        plt.imshow(groundtruth.reshape(1, -1), aspect='auto', cmap='tab20c',
                    interpolation='nearest', vmin=vmin, vmax=vmax)

    # 3. Plot Mapped Prediction
    ax3 = plt.subplot(grid[2], sharex=ax1)
    plt.title('State Sequence (Mapped Prediction)')
    plt.yticks([])
    if mapped_prediction is not None:
        plt.imshow(mapped_prediction.reshape(1, -1), aspect='auto', cmap='tab20c',
                    interpolation='nearest', vmin=vmin, vmax=vmax)

    current_grid_row = 3

    # 4. Plot Error Impact Bar (Optional)
    if errors_list is not None and show_error_impact:
        ax_impact = plt.subplot(grid[current_grid_row], sharex=ax1)
        impact_array = np.full(n, np.nan) # Initialize with NaN
        max_impact_intensity = 0
        for error in errors_list:
            start, end = error['start'], error['end']
            size = error['size']
            penalty = error['penalty']
            # Ensure end does not exceed array bounds
            actual_end = min(end + 1, n)
            if size > 0:
                intensity = penalty / size
                impact_array[start : actual_end] = intensity
                if intensity > max_impact_intensity:
                    max_impact_intensity = intensity
            else: # Handle zero size errors if they occur
                impact_array[start : actual_end] = 0 # Or some indicator

        cmap_impact = plt.cm.Reds
        norm_impact = mcolors.Normalize(vmin=0, vmax=max_impact_intensity if max_impact_intensity > 0 else 1)
        im_impact = plt.imshow(impact_array.reshape(1, -1), aspect='auto', cmap=cmap_impact, norm=norm_impact,
                                interpolation='nearest')
        plt.title('Error Impact')
        plt.yticks([])
        # Reduced padding for colorbar
        plt.colorbar(im_impact, ax=ax_impact, orientation='horizontal', fraction=0.1, pad=0.2, aspect=40, label='Impact Intensity')
        current_grid_row += 1

    # 5. Plot Error Type Bar (Optional)
    if errors_list is not None and show_error_type:
        ax_type = plt.subplot(grid[current_grid_row], sharex=ax1)
        type_array = np.full(n, np.nan) # Initialize with NaN for no error

        # Define error types, numerical values, and colors
        # Order matters for colormap and ticks
        error_type_order = ['delay', 'transition', 'missing', 'isolation']
        error_type_map = {name: i for i, name in enumerate(error_type_order)}
        cmap = plt.cm.get_cmap('tab10')
        # Select 4 distinct colors from tab20c for the error types
        # Example: using colors 0, 4, 8, 12 for better visual separation
        # error_colors = [cmap.colors[0], cmap.colors[4], cmap.colors[8], cmap.colors[12]]
        # Or simply take the first 4 colors if specific choices aren't critical
        error_colors = list(cmap.colors[:len(error_type_order)])
        num_types = len(error_type_map)

        for error in errors_list:
            start, end = error['start'], error['end']
            err_type = error['type']
            # Ensure end does not exceed array bounds
            actual_end = min(end + 1, n)
            if err_type in error_type_map:
                type_array[start : actual_end] = error_type_map[err_type]

        # Define custom colormap and normalization for categorical data
        cmap_type = mcolors.ListedColormap(error_colors)
        # Define boundaries for discrete colors: [-0.5, 0.5), [0.5, 1.5), ...
        bounds = np.arange(-0.5, num_types, 1)
        norm_type = mcolors.BoundaryNorm(bounds, cmap_type.N)

        im_type = plt.imshow(type_array.reshape(1, -1), aspect='auto', cmap=cmap_type, norm=norm_type,
                                interpolation='nearest')
        plt.title('Error Type')
        plt.yticks([])

        # Add colorbar with labels below the plot
        # Ticks should be centered within the color blocks, i.e., 0, 1, 2, ...
        # Reduced padding, increased fraction, adjusted aspect for horizontal layout
        cbar = plt.colorbar(im_type, ax=ax_type, orientation='horizontal', fraction=0.1, pad=0.3, aspect=40, ticks=np.arange(num_types))
        # Set labels based on the defined order
        cbar.set_ticklabels(error_type_order)
        # Rotate the labels
        cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45, ha='right') # Use ha='right' for better alignment after rotation
        current_grid_row += 1


    # Final adjustments
    #plt.xlabel("Time")
    # Adjust layout with reduced vertical padding (h_pad)
    plt.tight_layout(h_pad=0.5, rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap and reduce space

    # Hide x-axis labels for all but the bottom plot
    all_axes = plt.gcf().axes
    # Identify axes that are part of the main plot grid (exclude colorbar axes)
    main_plot_axes = [ax for ax in all_axes if hasattr(ax, 'get_subplotspec') and ax.get_subplotspec() is not None]
    if len(main_plot_axes) > 1:
        for ax in main_plot_axes[:-1]:
             # Check if the axis has tick labels before trying to set visibility
            if ax.get_xticklabels():
                plt.setp(ax.get_xticklabels(), visible=False)
            # Also hide the x-axis line and ticks for cleaner look
            ax.xaxis.set_ticks_position('none')


    if show:
        plt.show()


# This function plots multivariate time series with color-coded states and error bars.
def plot_sms_multi(X, groundtruth, mapped_prediction, errors_list=None,
                   show_error_impact=False, show_error_type=False,
                   figsize=(18, 6), show=False, groundtruth_state_names=None,
                   custom_colors=False, custom_text=None, save_path=None):
    """
    Plots time series, ground truth, mapped prediction(s), and optional error bars
    with combined legends for errors at the bottom.

    Parameters:
        X: Time series data, shape (T, C). Must be provided.
        groundtruth: Ground truth labels, shape (T,) or (T, 1). Can be None.
        mapped_prediction: Predicted labels.
                           Can be a single numpy array (already mapped if groundtruth provided),
                           or a dictionary {algo_name: prediction_array}.
        errors_list: List of error dictionaries or a dict {algo_name: error_list}.
                     Required if show_error_impact or show_error_type is True for any prediction.
                     Structure should correspond to mapped_prediction.
        show_error_impact (bool): If True, show an error bar with impact intensity.
        show_error_type (bool): If True, show an error bar with error types.
        figsize (tuple): Figure width and base for height calculations.
        show (bool): If True, display the plot.
        groundtruth_state_names (list of str, optional): Names for unique states in groundtruth.
                                                        Order should correspond to sorted unique states.
    """
    if X is None:
        raise ValueError("X (time series data) must be provided.")
    n = len(X)

    # --- Prepare plot items and gather global info ---
    plot_items = []
    global_max_impact_intensity = 0.0
    actual_show_error_impact = False
    actual_show_error_type = False

    _errors_source = errors_list
    if isinstance(mapped_prediction, dict):
        if errors_list is not None and not isinstance(errors_list, dict):
            print("Warning: mapped_prediction is a dict, but errors_list is not. Error info might be misaligned.")
            _errors_source = {}
        for name, pred_array in mapped_prediction.items():
            if pred_array is None:
                print(f"Warning: Prediction array for '{name}' is None. Skipping.")
                continue
            current_errors = _errors_source.get(name) if isinstance(_errors_source, dict) else None
            plot_items.append((name, pred_array, current_errors))
    elif mapped_prediction is not None:
        _current_errors = None
        if errors_list is not None:
            if isinstance(errors_list, dict):
                print("Warning: mapped_prediction is single array, errors_list is dict. No specific errors will be shown for the single prediction.")
                _current_errors = None
            elif isinstance(errors_list, list):
                _current_errors = errors_list
        plot_items.append(("Prediction", mapped_prediction, _current_errors))

    for _, _, err_list_for_pred in plot_items:
        if err_list_for_pred:
            if show_error_impact:
                actual_show_error_impact = True
                max_local_impact = 0
                for error in err_list_for_pred:
                    size = error.get('size', 0)
                    penalty = error.get('penalty', 0)
                    if size > 0:
                        intensity = penalty / size
                        if intensity > max_local_impact:
                            max_local_impact = intensity
                if max_local_impact > global_max_impact_intensity:
                    global_max_impact_intensity = max_local_impact
            if show_error_type:
                actual_show_error_type = True

    if (actual_show_error_impact or actual_show_error_type) and groundtruth is None and errors_list is not None:
        # Check if errors_list actually contains errors for any plot_item
        has_relevant_errors = False
        for _, _, err_list_for_item in plot_items:
            if err_list_for_item:
                has_relevant_errors = True
                break
        if has_relevant_errors:
            raise ValueError("groundtruth must be provided when show_error_impact or show_error_type is True and errors_list contains relevant errors.")


    # --- Determine vmin and vmax for consistent state coloring ---
    all_state_arrays_for_vmin_vmax = []
    if groundtruth is not None:
        all_state_arrays_for_vmin_vmax.append(groundtruth.ravel())
    for _, pred_arr, _ in plot_items:
        if pred_arr is not None:
            all_state_arrays_for_vmin_vmax.append(pred_arr.ravel())

    vmin, vmax = None, None
    if all_state_arrays_for_vmin_vmax:
        combined_states = np.concatenate(all_state_arrays_for_vmin_vmax)
        if combined_states.size > 0:
            unique_overall_states = np.unique(combined_states)
            vmin = unique_overall_states.min() - 0.5
            vmax = unique_overall_states.max() + 0.5

    # --- Calculate number of subplots and their height ratios ---
    num_main_rows = 0
    height_ratios = []
    ts_height_ratio = 5

    num_main_rows += 1 # For X
    height_ratios.append(ts_height_ratio)

    if groundtruth is not None:
        num_main_rows += 1
        height_ratios.append(1)

    for _, pred_arr, err_list_for_pred in plot_items:
        if pred_arr is not None:
            num_main_rows += 1
            height_ratios.append(1)
            if show_error_impact and err_list_for_pred:
                num_main_rows += 1
                height_ratios.append(0.75)
            if show_error_type and err_list_for_pred:
                num_main_rows += 1
                height_ratios.append(0.75)

    legend_row_needed = actual_show_error_impact or actual_show_error_type
    if legend_row_needed:
        num_main_rows += 1
        height_ratios.append(0.75)

    if num_main_rows == 1 and not plot_items and groundtruth is None: # Only X to plot
         plt.figure(figsize=figsize)
         plt.plot(X)
         if show: plt.show()
         else: plt.close()
         return
    elif num_main_rows == 0 : # Should not happen if X is required
        if show: print("Nothing to plot.")
        return


    # --- Adjust figure height ---
    base_height_per_unit = 0.7
    fig_height = base_height_per_unit * sum(height_ratios)

    fig = plt.figure(figsize=(figsize[0], fig_height))
    main_grid = gridspec.GridSpec(num_main_rows, 1, height_ratios=height_ratios, figure=fig)

    current_grid_idx = 0
    axes_for_xlabel_management = []
    ax_ts = None # Initialize ax_ts

    # 1. Plot Time Series
    ax_ts = fig.add_subplot(main_grid[current_grid_idx])
    ax_ts.spines['top'].set_visible(False)
    ax_ts.spines['right'].set_visible(False)
    ax_ts.spines['left'].set_visible(False)
    ax_ts.set_yticks([])
    ax_ts.plot(X)
    axes_for_xlabel_management.append(ax_ts)
    current_grid_idx += 1

    # 2. Plot Ground Truth
    if groundtruth is not None:
        ax_gt = fig.add_subplot(main_grid[current_grid_idx], sharex=ax_ts)
        ax_gt.set_title('Ground truth', fontsize='xx-large')
        ax_gt.set_yticks([])
        if custom_colors:
            cmap = ListedColormap(custom_colors)
            ax_gt.imshow(
            groundtruth.reshape(1, -1),
            aspect='auto',
            cmap=cmap,
            interpolation='nearest'
            )
        else:
            ax_gt.imshow(groundtruth.reshape(1, -1), aspect='auto', cmap='tab20c',
                    interpolation='nearest', vmin=vmin, vmax=vmax)
        axes_for_xlabel_management.append(ax_gt)

        if groundtruth_state_names is not None:
            gt_flat = groundtruth.ravel()
            unique_gt_states = np.unique(gt_flat)
            if len(groundtruth_state_names) != len(unique_gt_states):
                print(f"Warning: Number of groundtruth_state_names ({len(groundtruth_state_names)}) "
                      f"does not match number of unique groundtruth states ({len(unique_gt_states)}). "
                      "Skipping state name display.")
            else:
                name_map = {state_val: name for state_val, name in zip(unique_gt_states, groundtruth_state_names)}
                # Find segments and plot names
                last_state = None
                segment_start = 0
                for i, state in enumerate(gt_flat):
                    if state != last_state and last_state is not None:
                        # Plot text for the previous segment
                        mid_point = segment_start + (i - 1 - segment_start) / 2
                        if last_state in name_map:
                             ax_gt.text(mid_point, 0, name_map[last_state],
                                       ha='center', va='center', color='white', fontsize='x-large',
                                       bbox=dict(facecolor='black', alpha=0.3, pad=0.1, boxstyle='round,pad=0.2'))
                        segment_start = i
                    last_state = state
                # Plot text for the last segment
                if last_state is not None: # Ensure there was at least one state
                    mid_point = segment_start + (len(gt_flat) - 1 - segment_start) / 2
                    if last_state in name_map:
                        ax_gt.text(mid_point, 0, name_map[last_state],
                                   ha='center', va='center', color='white', fontsize='x-large',
                                   bbox=dict(facecolor='black', alpha=0.3, pad=0.1, boxstyle='round,pad=0.2'))
        current_grid_idx += 1

    # 3. Plot Mapped Prediction(s) and their errors
    title_suffix = "Segmentation" if groundtruth is not None else " (Prediction)"
    for name, pred_array, err_list_for_pred in plot_items:
        if pred_array is None: continue
        ax_pred = fig.add_subplot(main_grid[current_grid_idx], sharex=ax_ts)
        if custom_text:
            # use plain text to avoid mathtext parsing errors
            title = f"{name.upper()} | {custom_text.get(name)}"
        else:
            title = f"{title_suffix} ({name.upper()})"
        ax_pred.set_title(title, fontsize='xx-large')
        ax_pred.set_yticks([])
        if custom_colors:
            cmap = ListedColormap(custom_colors)
            ax_pred.imshow(
            pred_array.reshape(1, -1),
            aspect='auto',
            cmap=cmap,
            interpolation='nearest'
            )
        else:
            ax_pred.imshow(pred_array.reshape(1, -1), aspect='auto', cmap='tab20c',
                    interpolation='nearest', vmin=vmin, vmax=vmax)
        axes_for_xlabel_management.append(ax_pred)
        current_grid_idx += 1

        if show_error_impact and err_list_for_pred:
            ax_impact_plot = fig.add_subplot(main_grid[current_grid_idx], sharex=ax_ts)
            impact_array = np.full(n, np.nan)
            for error in err_list_for_pred:
                start, end = error['start'], error['end']
                size = error.get('size',0)
                penalty = error.get('penalty',0)
                actual_end = min(end + 1, n)
                if size > 0:
                    intensity = penalty / size
                    impact_array[start:actual_end] = intensity
            
            cmap_impact = plt.colormaps.get('Reds')
            norm_impact = mcolors.Normalize(vmin=0, vmax=global_max_impact_intensity if global_max_impact_intensity > 0 else 1)
            ax_impact_plot.imshow(impact_array.reshape(1, -1), aspect='auto', cmap=cmap_impact, norm=norm_impact, interpolation='nearest')
            #ax_impact_plot.set_title(f'Error Impact ({name})')
            ax_impact_plot.set_yticks([])
            axes_for_xlabel_management.append(ax_impact_plot)
            current_grid_idx += 1

        if show_error_type and err_list_for_pred:
            ax_type_plot = fig.add_subplot(main_grid[current_grid_idx], sharex=ax_ts)
            type_array = np.full(n, np.nan)
            error_type_order = ['delay', 'transition', 'missing', 'isolation']
            error_type_map = {name_val: i for i, name_val in enumerate(error_type_order)}
            
            error_type_color_config = {
                'delay': 'orange',
                'transition': 'tomato',
                'missing': 'crimson',
                'isolation': 'purple',
            }
            error_colors = [error_type_color_config[err_type] for err_type in error_type_order]
            num_types = len(error_type_map)

            for error in err_list_for_pred:
                start, end = error['start'], error['end']
                err_type = error['type']
                actual_end = min(end + 1, n)
                if err_type in error_type_map:
                    type_array[start:actual_end] = error_type_map[err_type]
            
            cmap_type_plot = mcolors.ListedColormap(error_colors[:num_types])
            bounds_plot = np.arange(-0.5, num_types, 1)
            norm_type_plot = mcolors.BoundaryNorm(bounds_plot, cmap_type_plot.N)
            
            ax_type_plot.imshow(type_array.reshape(1, -1), aspect='auto', cmap=cmap_type_plot, norm=norm_type_plot, interpolation='nearest')
            ax_type_plot.set_title(f'{name.upper()} | Segmentation errors', fontsize='xx-large')
            ax_type_plot.set_yticks([])
            axes_for_xlabel_management.append(ax_type_plot)
            current_grid_idx += 1
    
    # 4. Plot Legends at the bottom
    if legend_row_needed:
        legend_gs_item = main_grid[current_grid_idx]
        ax_legend_type, ax_legend_impact = None, None
        
        if actual_show_error_type and actual_show_error_impact:
            sub_gs = legend_gs_item.subgridspec(1, 2, wspace=0.3)
            ax_legend_type = fig.add_subplot(sub_gs[0, 0])
            ax_legend_impact = fig.add_subplot(sub_gs[0, 1])
        elif actual_show_error_type:
            ax_legend_type = fig.add_subplot(legend_gs_item)
        elif actual_show_error_impact:
            ax_legend_impact = fig.add_subplot(legend_gs_item)

        if ax_legend_type:
            # center the error-type legend within ax_legend_type
            width = 0.3
            inset_ax_type = ax_legend_type.inset_axes(
                [(1.0 - width) / 2.0, 0.0, width, 1.0],
                transform=ax_legend_type.transAxes
            )
            error_type_order = ['delay', 'transition', 'missing', 'isolation']
            error_type_map = {name: i for i, name in enumerate(error_type_order)}
            num_types = len(error_type_order)
            
            legend_type_data = np.arange(num_types).reshape(1, -1)
            cmap_type_legend = mcolors.ListedColormap(error_colors[:num_types]) # error_colors defined above
            bounds_legend = np.arange(-0.5, num_types, 1)
            norm_type_legend = mcolors.BoundaryNorm(bounds_legend, cmap_type_legend.N)
            
            inset_ax_type.imshow(legend_type_data, aspect='auto', cmap=cmap_type_legend, norm=norm_type_legend, interpolation='nearest')
            inset_ax_type.set_xticks(np.arange(num_types))
            inset_ax_type.set_xticklabels(error_type_order, rotation=0, ha='center', fontsize='xx-large')
            inset_ax_type.set_yticks([])
            inset_ax_type.set_title('Error types', fontsize='xx-large')
            ax_legend_type.axis('off')

        if ax_legend_impact:
            inset_cax_impact = ax_legend_impact.inset_axes([0, 0, 0.5, 1.0])
            cmap_impact_legend = plt.colormaps.get('Reds')
            norm_impact_legend = mcolors.Normalize(vmin=0, vmax=global_max_impact_intensity if global_max_impact_intensity > 0 else 1)
            sm_impact = plt.cm.ScalarMappable(cmap=cmap_impact_legend, norm=norm_impact_legend)
            sm_impact.set_array([]) 
            cbar_impact = fig.colorbar(sm_impact, cax=inset_cax_impact, orientation='horizontal', aspect=15, pad=0.25) 
            cbar_impact.ax.tick_params(labelsize='small')
            inset_cax_impact.set_title('Error Impact Intensity', fontsize='xx-large')
            ax_legend_impact.axis('off')
            
        current_grid_idx +=1

    # Final adjustments
    fig.tight_layout(h_pad=0.15, rect=[0, 0.03, 1, 0.97])

    # Manage x-axis ticks and labels:
    # Only ax_ts (the time series plot) should have x-axis ticks and labels.
    # All other main plots (GT, Preds, Errors) should not.
    if ax_ts and axes_for_xlabel_management: # Ensure ax_ts is defined
        # Ensure the primary time series plot (ax_ts) shows x-ticks and labels
        ax_ts.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

        # For all other plots in the main stack, hide x-ticks and labels
        for ax_to_modify in axes_for_xlabel_management:
            if ax_to_modify != ax_ts:
                ax_to_modify.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    # Optional: Set x-label for the time series plot if desired
    # if ax_ts:
    #     ax_ts.set_xlabel("Time")

    if save_path is not None:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

# noise = np.random.rand(30000)
# embedding_space(noise.reshape((5000,2)),show=True)
# density_map(noise.reshape((5000,2)),show=True)
# density_map_3d(noise.reshape((10000,3)),show=True)