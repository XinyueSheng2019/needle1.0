import os
import json
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from config import *
import re 

ztf_id_pattern = re.compile(r'ZTF\d+')


def load_exceptions():
    # load bad samples id to avoid them; light curve & image restoration results
    with open(EXCEPTION_IMG_PATH, 'r') as f:
        failed_img_ids = set([line.strip() for line in f])

    with open(EXCEPTION_LC_PATH, 'r') as f:
        failed_lc_ids = set([line.strip() for line in f])
    
    return failed_img_ids, failed_lc_ids


def load_samples(designated_class='TDE', info_path = OBJ_INFO_PATH, load_2025 = False, hosted = None):

    f = pd.read_csv(info_path)
    if hosted is not None:
        f = f[f.hosted == hosted]

    if designated_class == 'SN':
        class_list = f[(f.label == 0)]['ZTFID'].tolist()
    elif designated_class == 'SLSN-I':
        class_list = f[f.label == 1]['ZTFID'].tolist()
    elif designated_class == 'TDE':
        class_list = f[f.label == 2]['ZTFID'].tolist()
    else:
        print('No samples for {} classes.'.format(designated_class))
        return None

    print('original class_list: ', designated_class, len(set(class_list)))

    if load_2025:
        class_list = list(set(class_list).union(set(load_2025_samples(designated_class))))
        print('2025 class_list: ', designated_class, len(set(class_list)))

    return list(set(class_list))


def load_rare_2024_samples(designated_class='TDE', input_path = None):
    if input_path is None:
        input_path = '../info/20220301_20240225.csv'
    f = pd.read_csv(input_path)
    if designated_class == 'SLSN-I' or designated_class == 'TDE':
        class_list = f[f.type == designated_class]['ZTFID'].tolist()
        print('2024 class list: ', designated_class, len(set(class_list)))

    else:
        print('No samples for SN classes in 2024 samples.\n')
        return None
   
    return list(set(class_list))


def load_2025_samples(designated_class='TDE', input_path = None):# TODO: need to check the overlap, and pick up SN objects with labels

    if input_path is None:
        input_path = '../info/20240225_20250603.csv'
    f = pd.read_csv(input_path)
    if designated_class == 'SLSN-I' or designated_class == 'TDE':
        class_list = f[f.type == designated_class]['ZTFID'].tolist()
    else:
        class_list = f[(f.type != 'TDE') & (f.type != 'SLSN-I')]['ZTFID'].tolist()
   
    return list(set(class_list))

def load_sample_lc(default_path = PHOTO_OUTPUT_PATH):
    '''
    load all available light curve samples
    '''
    if os.path.exists(EXCEPTION_LC_PATH):
        with open(EXCEPTION_LC_PATH, 'r') as f:
            failed_lc_ids = set([line.strip() for line in f])
    else:
        failed_lc_ids = set()

    if os.path.exists(default_path):
        lc_list = os.listdir(default_path)
        lc_list = [obj for obj in lc_list if ztf_id_pattern.match(obj) and os.path.exists(os.path.join(default_path,obj,'photo_dict.npy')) and obj not in failed_lc_ids]
        return lc_list
    else:
        return None
    

def load_sample_imgs(default_path = IMG_OUTPUT_PATH):
    '''
    load all available image samples
    '''
    if os.path.exists(EXCEPTION_IMG_PATH):
        with open(EXCEPTION_IMG_PATH, 'r') as f:
            failed_img_ids = set([line.strip() for line in f])
    else:
        failed_img_ids = set()

    if os.path.exists(default_path):
        img_list = os.listdir(default_path)
        img_list = [obj for obj in img_list if ztf_id_pattern.match(obj) and os.path.exists(os.path.join(default_path,obj,'imgdata.npy')) and obj not in failed_img_ids]
        return img_list
    else:
        return None
  
def show_images(*images, num_cols=2, titles=None, global_scale=False, **imshow_kwargs):
    """
    Display multiple images in subplots with consistent scaling
    
    Parameters:
    -----------
    *images : array-like
        Variable number of images to display
    num_cols : int
        Number of columns in the subplot grid
    figsize : tuple
        Figure size (width, height)
    titles : list
        List of titles for each image
    global_scale : bool
        If True, use the same vmin/vmax for all images
    **imshow_kwargs : dict
        Additional arguments to pass to imshow
    """
    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols  # Ceiling division
        # Set up default parameters
    default_params = {
        'cmap': 'gray',
    }
    
    # If global scaling is requested, compute global min/max
    if global_scale:
        default_params['vmin'] = np.nanmin([np.min(img) for img in images])
        default_params['vmax'] = np.nanmax([np.max(img) for img in images])
    
    # Update with any user-provided parameters
    default_params.update(imshow_kwargs)
    
    fig_size = (num_cols*3, num_rows*3)
    # Create figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
    if num_images == 1:
        axes = np.array([axes])
    axes = axes.ravel()  # Flatten axes array
    
    # Display each image
    for idx, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img, **default_params)
        if titles and idx < len(titles):
            ax.set_title(titles[idx])
     
    # Turn off any unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
        plt.tight_layout()
        
    return fig, axes



    
def load_redshift_database():
    '''
    load the redshift database
    '''
    if not os.path.exists(OBJ_INFO_PATH):
        raise FileNotFoundError(f"Database file not found: {OBJ_INFO_PATH}")
    else:
        data = pd.read_csv(OBJ_INFO_PATH)[['ZTFID', 'redshift']]
        data = data.replace('-', np.nan).dropna()
        data = np.array(data)
        f1 = {k: float(v) for k, v in zip(data[:, 0], data[:, 1])}
        return f1
    
    
def display_image_pair(sci_data, ref_data, titles=None):
    """Helper function to display science and reference image pairs""" 
    if titles is None:
        titles = ['Science Image', 'Reference Image']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    if sci_data is not None:
        ax1.imshow(sci_data)
        ax1.set_title(titles[0])
    if ref_data is not None:
        ax2.imshow(ref_data)
        ax2.set_title(titles[1])
    plt.show()

def get_noise_distribution(image, factor1 = 2, factor2 = 3):
    # remove all the bright stars/hosts including the host and transient pixels.
    mean = np.nanmean(image.flatten())
    std = np.nanstd(image.flatten())
    noise_mask = image <= mean + factor1 * std
    noise_image = np.where(noise_mask, image, 0).flatten()
    noise_image = noise_image[noise_image != 0]
    below_3_sigma = noise_image[noise_image < np.nanmean(noise_image) + factor2 * np.nanstd(noise_image)]
    # Remove any NaN values from the noise distribution
    below_3_sigma = below_3_sigma[~np.isnan(below_3_sigma)]
    return below_3_sigma

# def load_bad_samples():
#     f = open('Bad_Samples.json', 'r')
#     bad_samples = json.load(f)
#     return bad_samples['sci_data_shape']
   

def img_reshape(img):
    """
    Reshape the image to add a single channel.

    Parameters:
    - img: ndarray, input image

    Returns:
    - ndarray, reshaped image
    """
    return img.reshape(img.shape[0], img.shape[1], 1)


def save_to_h5py(imageset, metaset, labels, idx_set, filepath):
    """
    Save the dataset, metaset, labels, and index set to an HDF5 file.
    """

    print('Image shape: ', imageset.shape)
    print('Meta shape: ', metaset.shape)
    print('Label shape: ', labels.shape)

    try:
        with h5py.File(filepath, "w") as f:
            f.create_dataset("imageset", imageset.shape, dtype='f', data=imageset)
            f.create_dataset("metaset", metaset.shape, dtype='f', data=metaset)
            f.create_dataset("labels", labels.shape, dtype='i', data=labels)
            f.create_dataset("idx_set", idx_set.shape, dtype='i', data=idx_set)
        print(f"Saved to {filepath}")
    except Exception as e:
        print(f"Error saving to HDF5 file {filepath}: {e}")



def open_with_h5py(filepath):
    
    imageset = np.array(h5py.File(filepath, mode = 'r')['imageset'])
    try: 
        labels = np.array(h5py.File(filepath, mode = 'r')['labels'])
    except:
        labels = np.array(h5py.File(filepath, mode = 'r')['label'])
    metaset = np.array(h5py.File(filepath, mode = 'r')['metaset'])
    idx_set = np.array(h5py.File(filepath, mode = 'r')['idx_set'])
    return imageset, metaset, labels, idx_set

def open_with_npy(filepath):
    return np.load(filepath)

def most_common(lst):
    return max(set(lst), key=lst.count)


def add_obj_meta(obj, obj_path, filefracday, add_host=False, recent_values=False):
    """
    Add object metadata from a CSV file for a specific object and filefracday.

    Parameters:
    - obj: str, object identifier
    - obj_path: str, path to the object's metadata files
    - filefracday: str, identifier for the file fraction day
    - add_host: bool, whether to add host galaxy information
    - recent_values: bool, whether to include recent values in the metadata

    Returns:
    - list, metadata for the object or None if not found
    """
    meta = pd.read_csv(f"{obj_path}/{obj}/obj_meta4ML.csv")
    d_row = meta.loc[meta.filefracday == int(filefracday)].fillna(0)
    if d_row.empty:
        print(f"{obj_path} {filefracday} NOT FOUND.\n")
        return None

    new_row = [
        d_row['candi_mag'].values[0],
        d_row['disc_mag'].values[0],
        d_row['delta_mag_discovery'].values[0],
        d_row['delta_t_discovery'].values[0]
    ]

    if recent_values:
        new_row += [
            d_row['delta_mag_recent'].values[0],
            d_row['delta_t_recent'].values[0]
        ]

    ratio_recent, ratio_disc = get_ratio(
        d_row['delta_mag_recent'].values[0],
        d_row['delta_t_recent'].values[0],
        d_row['delta_mag_discovery'].values[0],
        d_row['delta_t_discovery'].values[0]
    )

    new_row += [ratio_recent, ratio_disc]

    if add_host:
        new_row.append(d_row['delta_host_mag'].values[0])

    return new_row



def get_ratio(delta_mag_recent, delta_t_recent, delta_mag_disc, delta_t_disc):
    """
    Calculate the ratios of delta magnitude to delta time for recent and discovery values.

    Parameters:
    - delta_mag_recent: float, recent delta magnitude
    - delta_t_recent: float, recent delta time
    - delta_mag_disc: float, discovery delta magnitude
    - delta_t_disc: float, discovery delta time

    Returns:
    - tuple of floats, (ratio_recent, ratio_disc)
    """
    if isinstance(delta_mag_disc, np.ndarray):
        return (
            np.divide(delta_mag_recent, delta_t_recent, out=np.zeros_like(delta_mag_recent), where=delta_t_recent != 0),
            np.divide(delta_mag_disc, delta_t_disc, out=np.zeros_like(delta_mag_disc), where=delta_t_disc != 0)
        )
    else:
        if delta_t_disc == 0.0 or delta_t_recent == 0.0:
            return 0.0, 0.0
        return delta_mag_recent / delta_t_recent, delta_mag_disc / delta_t_disc


def add_host_meta(obj, host_path, only_complete=True):
    """
    Add host galaxy metadata from a CSV file.

    Parameters:
    - obj: str, object identifier
    - host_path: str, path to the host metadata files
    - only_complete: bool, whether to include only complete metadata

    Returns:
    - list, host metadata for the object or None if not found
    """
    def add_mag(line, band):
        return line.get(f'{band}Ap') or line.get(f'{band}PSF') or None

    if os.path.exists(f"{host_path}/{obj}.csv"):
        meta = pd.read_csv(f"{host_path}/{obj}.csv")
        line = meta.iloc[0]
        h_row = [add_mag(line, band) for band in ['g', 'r', 'i', 'z', 'y', 'g-r_', 'r-i_']]
        return h_row if all(h_row) or not only_complete else None

    return None if only_complete else [None] * 7


def add_sherlock_info(mag_records_path, ztf_id, properties, only_complete=True):
    """
    Add Sherlock information from a JSON file.

    Parameters:
    - mag_records_path: str, path to the magnitude records files
    - ztf_id: str, ZTF object identifier
    - properties: list, properties to extract from the Sherlock information
    - only_complete: bool, whether to include only complete information

    Returns:
    - list, Sherlock metadata for the object or None if not found
    """
    try:
        with open(f"{mag_records_path}/{ztf_id}.json") as f:
            obj_mags = json.load(f)

        sherlock_info = obj_mags.get('sherlock', {})
        sherlock_meta = [sherlock_info.get(p, 0 if not only_complete else None) for p in properties]

        return sherlock_meta if all(sherlock_meta) or not only_complete else None

    except (FileNotFoundError, KeyError):
        return None



def data_is_sci(fits_file):
        # check if data is sci
    if fits_file.startswith('sci'):
        data_is_sci = True
    else:
        data_is_sci = False
    return data_is_sci



   
def get_derived_image(image):
    # remove all the bright stars/hosts including the host and transient pixels.
    mean = np.nanmean(image.flatten())
    std = np.nanstd(image.flatten())
    threshold = mean
    noise_mask = image <= threshold + 2 * std
    source_mask = ~noise_mask
    source_image = np.where(source_mask, image, 0).flatten()
    source_image = source_image[source_image != 0]
    noise_image = np.where(noise_mask, image, 0).flatten()
    noise_image = noise_image[noise_image != 0]

    below_3_sigma = noise_image[noise_image < np.mean(noise_image) + 3 * np.std(noise_image)]
    noise_mean = np.nanmean(below_3_sigma)
    # noise_std = np.std(below_3_sigma)
    noise_max  = np.nanmax(below_3_sigma)

    above_3_sigma = noise_image[noise_image >= np.mean(noise_image) + 3 * np.std(noise_image)]
    source_image = np.concatenate((source_image, above_3_sigma), axis = 0)
    source_min = np.nanmin(source_image)
    source_mean = np.nanmean(source_image)
    # source_std = np.std(source_image)


    return noise_mean, source_mean, noise_max, source_min
        

def get_matrix_intersection(matrix1, matrix2, mode='element'):
    """
    Find the intersection of two matrices based on the specified mode.
    
    Parameters
    ----------
    matrix1 : numpy.ndarray
        First input matrix
    matrix2 : numpy.ndarray
        Second input matrix
    mode : str, optional
        Type of intersection to compute:
        - 'element': Element-wise intersection (default)
        - 'binary': Intersection of binary masks
        - 'nonzero': Intersection of non-zero elements
        - 'threshold': Intersection where both matrices exceed their mean values
    
    Returns
    -------
    numpy.ndarray
        The intersection matrix
    float
        Intersection score (ratio of intersection to union)
    
    Raises
    ------
    ValueError
        If matrices have different shapes or if mode is invalid
    """
    if matrix1.shape != matrix2.shape:
        raise ValueError(f"Matrix shapes do not match: {matrix1.shape} vs {matrix2.shape}")
    
    if mode == 'element':
        # Element-wise intersection (exact value matching)
        intersection = np.where(matrix1 == matrix2, matrix1, 0)
        union = np.where((matrix1 != 0) | (matrix2 != 0), 1, 0)
        
    elif mode == 'binary':
        # Binary mask intersection
        mask1 = matrix1.astype(bool)
        mask2 = matrix2.astype(bool)
        intersection = mask1 & mask2
        union = mask1 | mask2
        
    elif mode == 'nonzero':
        # Non-zero elements intersection
        mask1 = matrix1 != 0
        mask2 = matrix2 != 0
        intersection = np.where(mask1 & mask2, matrix1, 0)
        union = mask1 | mask2
        
    elif mode == 'threshold':
        # Threshold-based intersection
        threshold1 = np.mean(matrix1)
        threshold2 = np.mean(matrix2)
        mask1 = matrix1 > threshold1
        mask2 = matrix2 > threshold2
        intersection = np.where(mask1 & mask2, matrix1, 0)
        union = mask1 | mask2
        
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be one of ['element', 'binary', 'nonzero', 'threshold']")
    
    # Calculate intersection score (IoU - Intersection over Union)
    intersection_score = np.sum(intersection != 0) / np.sum(union != 0) if np.sum(union != 0) > 0 else 0.0
    
    return intersection, intersection_score
        

def get_matrix_intersection_fast(matrix1, matrix2):
    """
    Efficiently compute the intersection of two matrices using NumPy operations.
    This is optimized for performance using vectorized operations.
    
    Parameters
    ----------
    matrix1 : numpy.ndarray
        First input matrix
    matrix2 : numpy.ndarray
        Second input matrix
    
    Returns
    -------
    numpy.ndarray
        The intersection matrix where both inputs have non-zero values
    numpy.ndarray
        Boolean mask showing where intersection occurs
    float
        Intersection over Union (IoU) score
    """
    # Convert to boolean masks for efficient operations
    mask1 = matrix1 != 0
    mask2 = matrix2 != 0
    
    # Compute intersection and union masks using fast boolean operations
    intersection_mask = mask1 & mask2
    union_mask = mask1 | mask2
    
    # Create the intersection matrix preserving values from matrix1
    intersection = np.where(intersection_mask, matrix1, 0)
    
    # Calculate IoU score efficiently
    iou_score = np.sum(intersection_mask) / np.sum(union_mask) if np.any(union_mask) else 0.0
    
    return intersection, intersection_mask, iou_score
        

def rotate_image_numpy(image, angle_degrees, center=None, fill_value=0):
    """
    Rotate an image using NumPy operations.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image to rotate
    angle_degrees : float
        Rotation angle in degrees (counterclockwise)
    center : tuple, optional
        Center of rotation (x, y). If None, uses image center
    fill_value : float, optional
        Value to fill areas outside the rotated image
        
    Returns
    -------
    numpy.ndarray
        Rotated image
    """
    # Convert angle to radians
    angle_rad = np.radians(angle_degrees)
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Set rotation center if not provided
    if center is None:
        center = (width / 2, height / 2)
    
    # Create rotation matrix
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    
    # Create coordinate matrices
    y_coords, x_coords = np.mgrid[:height, :width]
    coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))
    
    # Adjust coordinates relative to rotation center
    coords -= np.array(center)
    
    # Apply rotation
    rotated_coords = np.dot(coords, rotation_matrix.T)
    
    # Readjust coordinates back
    rotated_coords += np.array(center)
    
    # Get source coordinates
    src_x = rotated_coords[:, 0].reshape(height, width)
    src_y = rotated_coords[:, 1].reshape(height, width)
    
    # Create mask for valid coordinates
    valid_coords = (
        (src_x >= 0) & (src_x < width) &
        (src_y >= 0) & (src_y < height)
    )
    
    # Initialize output image with fill value
    rotated = np.full_like(image, fill_value, dtype=float)
    
    # Use nearest neighbor interpolation for valid coordinates
    x_indices = np.clip(src_x[valid_coords].astype(int), 0, width - 1)
    y_indices = np.clip(src_y[valid_coords].astype(int), 0, height - 1)
    rotated[valid_coords] = image[y_indices, x_indices]
    
    return rotated

def rotate_image_numpy_interpolated(image, angle_degrees, center=None, fill_value=0):
    """
    Rotate an image using NumPy operations with bilinear interpolation.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image to rotate
    angle_degrees : float
        Rotation angle in degrees (counterclockwise)
    center : tuple, optional
        Center of rotation (x, y). If None, uses image center
    fill_value : float, optional
        Value to fill areas outside the rotated image
        
    Returns
    -------
    numpy.ndarray
        Rotated image with bilinear interpolation
    """
    # Convert angle to radians
    angle_rad = np.radians(angle_degrees)
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Set rotation center if not provided
    if center is None:
        center = (width / 2, height / 2)
    
    # Create rotation matrix
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    
    # Create coordinate matrices
    y_coords, x_coords = np.mgrid[:height, :width]
    coords = np.column_stack((x_coords.ravel(), y_coords.ravel()))
    
    # Adjust coordinates relative to rotation center
    coords -= np.array(center)
    
    # Apply rotation
    rotated_coords = np.dot(coords, rotation_matrix.T)
    
    # Readjust coordinates back
    rotated_coords += np.array(center)
    
    # Get source coordinates
    src_x = rotated_coords[:, 0].reshape(height, width)
    src_y = rotated_coords[:, 1].reshape(height, width)
    
    # Get floor and ceiling coordinates for interpolation
    x0 = np.floor(src_x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(src_y).astype(int)
    y1 = y0 + 1
    
    # Clip coordinates to image boundaries
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)
    
    # Calculate interpolation weights
    wa = (x1 - src_x) * (y1 - src_y)
    wb = (src_x - x0) * (y1 - src_y)
    wc = (x1 - src_x) * (src_y - y0)
    wd = (src_x - x0) * (src_y - y0)
    
    # Create mask for valid coordinates
    valid_coords = (
        (src_x >= 0) & (src_x < width) &
        (src_y >= 0) & (src_y < height)
    )
    
    # Initialize output image with fill value
    rotated = np.full_like(image, fill_value, dtype=float)
    
    # Apply bilinear interpolation for valid coordinates
    rotated[valid_coords] = (
        wa[valid_coords] * image[y0[valid_coords], x0[valid_coords]] +
        wb[valid_coords] * image[y0[valid_coords], x1[valid_coords]] +
        wc[valid_coords] * image[y1[valid_coords], x0[valid_coords]] +
        wd[valid_coords] * image[y1[valid_coords], x1[valid_coords]]
    )
    
    return rotated

def rotate_image_90(image, k=1):
    """
    Rotate an image by 90 degrees k times.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image to rotate
    k : int
        Number of times to rotate 90 degrees:
        k=0: no rotation (0 degrees)
        k=1: rotate 90 degrees counterclockwise
        k=2: rotate 180 degrees
        k=3: rotate 270 degrees counterclockwise (or 90 clockwise)
    
    Returns
    -------
    numpy.ndarray
        Rotated image
    """
    if k not in [0, 1, 2, 3]:
        raise ValueError("k must be 0, 1, 2, or 3 for rotations of 0, 90, 180, or 270 degrees")
    
    if k == 0:
        return image.copy()
    
    return np.rot90(image, k=k)
        