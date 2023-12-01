from transformers import AutoProcessor, CLIPSegForImageSegmentation
from skimage.segmentation import find_boundaries
from PIL import Image
import torch
import matplotlib.pyplot as plt
import time
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import cv2
from scipy import ndimage
from skimage.measure import approximate_polygon
from matplotlib.path import Path
from datetime import datetime

from CLIPSEG_Engine_ImagePrompt import CLIPSEG_Engine_ImagePrompt

#plot result
def cropSquare(image):
    # Get image size
    width, height = image.size

    # Find the smaller dimension
    smaller_dim = min(width, height)

    # Calculate the area to crop
    left = (width - smaller_dim)/2
    top = (height - smaller_dim)/2
    right = (width + smaller_dim)/2
    bottom = (height + smaller_dim)/2

    # Crop the image
    image = image.crop((left, top, right, bottom))

    # Show the cropped image
    return image

def retain_max_cluster_and_find_centroid(data):
    # Erode to remove dilation and "inset" the final boundary
    #kernelErode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
    #eroded_data = cv2.erode(data, kernelErode, iterations=1)

    #data = eroded_data

    # Label connected components (clusters) in the data
    labels, num_labels = ndimage.label(data > 0)

    max_avg_val = -np.inf
    max_label = None

    # For each label, compute average and track the one with the maximum average value
    for label_num in range(1, num_labels + 1):
        label_data = data[labels == label_num]
        avg_val = label_data.mean()

        if avg_val > max_avg_val:
            max_avg_val = avg_val
            max_label = label_num

    # Create a mask for the cluster with maximum average value
    mask = (labels == max_label)

    # Set all other clusters to zero
    filtered_data = data * mask
    
    # Find the centroid of the cluster
    y, x = np.nonzero(mask)  # Note: numpy index is (row, col) so (y, x)
    centroid = (int(x.mean()), int(y.mean()))

    return filtered_data, centroid

def sample_uniform_grid(cluster_data, grid_size=10):
    # Get the bounds of the cluster
    y_indices, x_indices = np.nonzero(cluster_data)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    # Create the grid
    x_vals = np.linspace(x_min, x_max, grid_size)
    y_vals = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x_vals, y_vals)

    # Round the meshgrid to get integer indices
    grid_x_indices = np.round(xx).astype(int)
    grid_y_indices = np.round(yy).astype(int)

    # Create an empty mask
    mask = np.zeros_like(cluster_data)

    # Set the mask values at the grid points to 1
    mask[grid_y_indices, grid_x_indices] = 1

    # Multiply the cluster data mask by the new mask to filter it
    filtered_data = cluster_data * mask

    # Get the coordinates of non-zero points
    non_zero_points = np.nonzero(filtered_data)

    # Transpose the result to get a 2xn array
    non_zero_points = np.transpose(non_zero_points)
    
    return non_zero_points

def downsample_perimeter(data, tolerance=15):
    # Find the perimeter of the cluster
    boundaries = find_boundaries(data > 0, mode='outer').astype(int)
    
    # Get the coordinates of the perimeter points
    perimeter_coords = np.argwhere(boundaries > 0)
    
    # Downsample the perimeter points
    downsampled_coords = approximate_polygon(perimeter_coords, tolerance)
    
    # Create a blank array to hold the downsampled perimeter
    downsampled_perimeter = np.zeros_like(data)
    
    # Add the downsampled points to the array
    for coord in downsampled_coords:
        downsampled_perimeter[tuple(coord)] = 1

    kernelDilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    downsampled_perimeter = cv2.dilate(downsampled_perimeter, kernelDilate, iterations=1)
    
    return downsampled_perimeter

'''
#Preload ML Engine
CLIPSEG_Engine = CLIPSEG_Engine_ImagePrompt()

image = Image.open("photos/PXL_20230705_053948478.jpg")# Load your image
names = ["PXL_20230705_054626285"]# Define your prompts
prompts = [Image.open(f"photos/{i}.jpg") for i in names]
preds = CLIPSEG_Engine.main(image, prompts)
'''
##Assume you will be provided with preds and the images used
def plotData(preds, prompts, image):
    fig, ax = plt.subplots(7, 4, figsize=(10, 20))  # create 5 rows for 5 images
    [a.axis('off') for a in ax.flatten()]

    for i in range(len(prompts)):    
        ax[i, 0].imshow(prompts[i])
        np_array_sigmoid = torch.sigmoid(preds[i][0]).squeeze(0).numpy()
        np_array_sigmoid[np_array_sigmoid < 0.2] = 0
        ax[i, 1].imshow(np_array_sigmoid)

        #overlay results over image
        mask = np_array_sigmoid
        mask_height, mask_width = mask.shape
        color_mask = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
        color_mask[:,:,0] = mask*255  # RGB for red
        resized_image = cv2.resize(np.asarray(image), (mask_width, mask_height))  # PIL Image to numpy array conversion
        alpha = 0.5
        overlay = cv2.addWeighted(resized_image, alpha, color_mask, 1 - alpha, 0)

        ax[i, 2].imshow(overlay)
        ax[i, 2].set_title(f"max res: %3.3f"%np_array_sigmoid.max())

        filtered_data, centroid = retain_max_cluster_and_find_centroid(np_array_sigmoid)
        perimeterPoints = downsample_perimeter(filtered_data)
        color_maskPerimeter = np.zeros((mask_height, mask_width, 3), dtype=np.uint8)
        color_maskPerimeter[:,:,0] = perimeterPoints*255  # RGB for red
        #perimeterOverlay = cv2.addWeighted(resized_image, alpha, color_maskPerimeter, 1 - alpha, 0)
        
        perimeterOverlay = resized_image
        subset = sample_uniform_grid(filtered_data)
        [cv2.circle(perimeterOverlay, (int(point[1]), int(point[0])), 3, (255, 0, 0), -1) for point in subset]
        
        ax[i, 3].imshow(perimeterOverlay)
        ax[i, 3].set_title(f"number points: %d"%np.count_nonzero(perimeterPoints))
    plt.show()

def rescale_to_original(coords, original_width, original_height, resized_dim):
    # Calculate the scale factors
    x = coords[:, 1]
    y = coords[:, 0]
    
    if original_width > original_height:
        offset = (original_width - original_height) / 2
        scale = original_height / resized_dim
        x = x*scale + offset
        y = y*scale
    elif original_height >= original_width: # Modified condition here to cover equal dimensions too
        offset = (original_height - original_width) / 2
        scale = original_width / resized_dim
        y = y*scale + offset
        x = x*scale

    rescaled_coords = np.column_stack([x, y])

    return rescaled_coords

##Assume you will be provided with preds and the images used
##Assume only a single pred
def subsampleCluster(preds, target):
    np_array_sigmoid = torch.sigmoid(preds[0][0]).squeeze(0).numpy()
    np_array_sigmoid[np_array_sigmoid < 0.2] = 0
    cv2.imwrite("results/imgHeatmap" + datetime.utcnow().strftime("%Y%m%d-%H%M%S") + ".png",np_array_sigmoid*255)
    
    filtered_data, centroid = retain_max_cluster_and_find_centroid(np_array_sigmoid)
    subset_xy = sample_uniform_grid(filtered_data)

    imgOverlay = np.copy(target)
    resized_height, resized_width = np_array_sigmoid.shape#numpy array shape
    original_width, original_height = target.size#PIL image size

    subset_XY = rescale_to_original(subset_xy, original_width, original_height, resized_width)


    resized_image = cv2.resize(np.asarray(cropSquare(target)), (resized_width, resized_height))  # PIL Image to numpy array conversion
    [cv2.circle(imgOverlay, (int(point[0]), int(point[1])), 15, (255, 0, 0), -1) for point in subset_XY]
    return subset_XY, imgOverlay
