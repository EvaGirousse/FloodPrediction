
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

def process_image(image, size_patch):
    """Set image to the good size.

    Args:
        image (np.ndarray): Image in full resolution.

    Returns:
        np.ndarray: Image croped so that it works with patches creation.
    """
    w = image.shape[0]
    h = image.shape[1]
    new_image = image[:w-w % size_patch, :h-h%size_patch]

    return new_image

def patchify(image, size_patch):
    """Transforms an image in multiple patches.

    Args:
        image (np.ndarray): Initial image.
        size_patch (int): Size of the patch.

    Returns:
        Tuple[np.ndarray,np.ndarray]: The first array contais patches, which are themselves array of size size_path*size_patch. The second array is an index of their position in the initial image.
    """
    if len(image.shape)==3:
        h, w, nb_features = image.shape
    else:
        h,w = image.shape

    nb_h = (h // size_patch)
    nb_w = (w // size_patch)

    
    if len(image.shape) ==3:
        patches = np.zeros((nb_h*nb_w, size_patch, size_patch, nb_features))
    else:
        patches = np.zeros((nb_h*nb_w, size_patch, size_patch))
        
    index = []

    for i in range(nb_h):
        for j in range(nb_w):
            patch = image[i * size_patch: (i + 1) * size_patch, j * size_patch: (j + 1) * size_patch]
            patches[i * nb_w + j] = patch
            index.append((i,j))

    index = np.asarray(index)        
    return patches, index

from src.utils import patchify, process_image
from scipy.ndimage import distance_transform_edt

def get_useful_patches_index(labels_image, config):
    size_patch = config['SizePatch']
    dist = 1
    ## Compute sum of labels over time
    labels_image = labels_image['__xarray_dataarray_variable__']
    image_sum = np.nansum(labels_image, axis=0)
    image_sum = process_image(image_sum, size_patch)

    ## Compute patches on the sum's image
    patches, index = patchify(image_sum,size_patch)
    grid_patch = np.zeros((image_sum.shape[0]//size_patch,image_sum.shape[1]//size_patch))
    sum_patches = []
    for p in patches:
        sum_patches.append(p.sum())
    for i, p in enumerate(sum_patches):
        grid_patch[index[i][0], index[i][1]] = p
    distance_map = distance_transform_edt(1 - (grid_patch>0).astype(int))
    flood_mask = (distance_map <= dist).astype(np.uint8)
    flood_indices = np.argwhere(flood_mask == 1)
    flood_index = flood_indices[:, 0]*(image_sum.shape[1]//size_patch) + flood_indices[:, 1]
    return flood_index

def get_static_patches(static_xr, useful_patches_index, size_patch):
    static = static_xr['__xarray_dataarray_variable__'].to_numpy()
    n_bands = static.shape[0]
    useful_patches_static = np.zeros((n_bands, len(useful_patches_index), size_patch,size_patch))
    for b in range(n_bands):
        image = process_image(static[b], size_patch)
        patches, _ = patchify(image, size_patch)
        useful_patches_static[b] = patches[useful_patches_index]
    return useful_patches_static

def get_label_patches(labels,useful_patches_index, size_patch):
    """Gets useful labels patches based on the given index.

    Args:
        labels (xr.Dataset): _description_
        useful_patches_index (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    train_set_size = len(labels.time)
    labels = labels['__xarray_dataarray_variable__'].to_numpy()
    useful_patches_labels = np.zeros((train_set_size, len(useful_patches_index), size_patch,size_patch))
    for t in range(train_set_size):
        image = process_image(labels[t], size_patch)
        patches, _ = patchify(image,size_patch)
        useful_patches_labels[t] = patches[useful_patches_index]

    return useful_patches_labels

def show_AUC_ROC_graph(y,yprob):
    y_true = y.copy()
    y_prob = yprob.copy()
    key_thresholds = [0.001,0.005,0.01,0.05,0.1, 0.2,0.3, 0.5, 0.9]
    # key_thresholds =  [0.01,0.05,0.1,0.15, 0.2,0.3, 0.5, 0.9]
    plt.figure(figsize=(8, 6))
    y_true[y_true == -1] = np.nan
    y_prob[y_prob == -1] = np.nan
    y_true = y_true[~np.isnan(y_true)]
    y_prob = y_prob[~np.isnan(y_prob)]
    fpr, tpr, thresholds = roc_curve(y_true,y_prob,pos_label=1)
    auc = roc_auc_score(y_true,y_prob)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % auc)
    plt.xlabel(f'False Positive Rate')
    plt.ylabel(f'True Positive Rate')
    for thresh in key_thresholds:
        idx = np.where(thresholds >= thresh)[0][-1]
        plt.plot(fpr[idx], tpr[idx], 'o', markersize=10, label=f'Threshold {thresh:.3f}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title(f'Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    print(f"AUC: {auc}")

    return