import numpy as np
import xarray as xr

def apply_mask_permanent_water(full_map, mask):
    mask = mask['__xarray_dataarray_variable__'].to_numpy()
    index_permanent_water = np.where(mask == 1)
    full_map[index_permanent_water] = -1

    return full_map


def recreate_full_map_useful_patches(useful_patches,useful_patches_index, mask, patch_size):

    ## Place the useful patches at the right place and zeros elsewhere
    shape_1 = 3936 // patch_size
    shape_2 = 5953 // patch_size
    grid_date = np.zeros((shape_1,shape_2,patch_size,patch_size))
    for i in range(shape_1):
        for j in range(shape_2):
            if i*shape_2+j in useful_patches_index:
                idx = np.where(useful_patches_index==i*shape_2+j)[0][0]
                grid_date[i,j] = useful_patches[idx]

    ## Concatenate this np.array
    concatenated_rows = []
    for i in range(grid_date.shape[0]):
        concatenated_i = []
        for j in range(grid_date.shape[1]):
            patch = grid_date[i, j]
            concatenated_i.append(patch)
        concatenated_i = np.concatenate(concatenated_i, axis=1)
        concatenated_rows.append(concatenated_i)
    large_image = np.concatenate(concatenated_rows, axis=0)

    ## Set image to the initial size
    left_side = np.zeros((3936-large_image.shape[0],large_image.shape[1]))
    large_image = np.concatenate([left_side,large_image],axis = 0)
    up_side = np.zeros((large_image.shape[0],5953-large_image.shape[1]))
    large_image = np.concatenate([up_side,large_image],axis = 1)

    large_image = apply_mask_permanent_water(large_image, mask)
    return large_image

def from_results_tensor_to_list_full_image(y_prob,useful_index, mask, size_patch):
    list_image = []
    nb_weeks = int(len(y_prob)/len(useful_index))
    for i in range(nb_weeks):
        id_left = i*len(useful_index)
        id_right = (i+1)*len(useful_index)
        y_prob_week = y_prob[id_left:id_right]
        y_prob_week = recreate_full_map_useful_patches(y_prob_week,useful_index, mask, size_patch)
        list_image.append(y_prob_week)
    return np.array(list_image)


def from_full_images_array_list_to_xarray(list_image, test_times, save_path, x=None, y=None):

    if isinstance(list_image, np.ndarray):
        xr_array_score = xr.DataArray(data=np.array(list_image), 
                                        dims=["time", "y", "x"],
                                        coords={"time": test_times, 
                                                "x": x, 
                                                "y": y},
                                        name="Score")
    else:
        xr_array_score = list_image
    
    relevent_data_tag_binary_mask = xr.open_dataset(save_path+"/relevent_data_tag_binary_mask.nc")
    nan_mask = relevent_data_tag_binary_mask["__xarray_dataarray_variable__"] == 1

    for time in test_times:
        time_slice =  xr_array_score.sel(time=time)
        time_slice = time_slice.where(~nan_mask, -1)
        xr_array_score.loc[dict(time=time)] = time_slice

    return xr_array_score


def from_xarray_to_vector(data):
    data = data.sortby("time").sortby("x").sortby("y")
    xry = data.values
    vectors = xry.reshape(xry.shape[0], xry.shape[1]*xry.shape[2])
    vector = vectors.flatten()
    mask = vector == -1
    return vector[~mask]
