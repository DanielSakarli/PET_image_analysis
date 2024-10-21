# README: The usage of the script to plot several recovery coefficients of different reconstructions is as follows
# 1. Run the script 
# 2. Load the DICOM images from the folder containing the DICOM files
# 3. Click on the "Isocontour detection" button to detect the ROIs for the predefined centers
# 4. Repeat step 2 and 3 for different reconstructions
# 5. When all the reconstructed images are loaded, click on the "Draw Plot" button to plot the recovery coefficients. Note: this will only save the plot, but not show it yet
# 6. Click on the "Show Plot" button to display the plot

import pydicom
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy import ndimage
from matplotlib.colors import Normalize
import pickle


# Initialize global variables
rgb_image = None
rois = None
roi_pixels = None
roi_masks = []
# Global variables to maintain state across multiple DICOM folder additions
iteration_count = 0
# Initialize an array to store the recovery coefficients
recovery_coefficients = []
# Initialize global variables
dicom_images = []  # List to store DICOM images
current_index = 0  # Current slice index


# Function to load DICOM images from a directory
def load_dicom_images(directory):
    global dicom_images
    dicom_images = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        dicom_image = pydicom.dcmread(filepath)
        dicom_images.append(dicom_image)

    #print(f"Pixel array of dicom images in load_dicom_images: {dicom_images[0].pixel_array}")
    #print(f"Five maximal values of the 62nd DICOM image: {np.sort(dicom_images[61].pixel_array.flatten())[-5:]}")
    return dicom_images

def display_dicom_image(dicom_image, canvas, ax, rois=None, roi_pixels=None):
    global rgb_image, roi_masks

    ax.clear()
    # print("Displaying image with ROIs:", roi_pixels)

    # Check if dicom_image is already a NumPy array
    if isinstance(dicom_image, np.ndarray):
        img_array = dicom_image
    else:
        img_array = dicom_image.pixel_array

    # Normalize the image array for display
    norm = Normalize(vmin=img_array.min(), vmax=img_array.max())
    normalized_img = norm(img_array)
    
    # Convert grayscale to RGB
    rgb_image = plt.cm.gray(normalized_img)[:, :, :3]  # Discard alpha channel from grayscale to RGB conversion


    # Display ROIs from isocontour detection
    roi_color = [0, 1, 1]  # Cyan color for VOI
    if roi_masks:
        for mask in roi_masks:
            rgb_image[mask] = roi_color
    
    if roi_pixels:
        # Ensure roi_pixels is a list of tuples
        if isinstance(roi_pixels, np.ndarray):
            roi_pixels = list(map(tuple, roi_pixels))
        # Set ROI pixels to red
        for (x, y) in roi_pixels:
            if 0 <= x < img_array.shape[0] and 0 <= y < img_array.shape[1]:
                rgb_image[x, y] = [1, 0, 0]  # Red color for ROI pixels
    if False:
        if roi_pixels is not None:
            # Set ROI pixels to red
            for roi in roi_pixels:
                for (x, y) in roi:
                    if 0 <= x < img_array.shape[0] and 0 <= y < img_array.shape[1]:
                        rgb_image[x, y] = [1, 0, 0]  # Red color for ROI pixels

            if False:
                # Set ROI pixels to red
                for (x, y) in roi_pixels:
                    if 0 <= x < img_array.shape[0] and 0 <= y < img_array.shape[1]:
                        rgb_image[x, y] = [1, 0, 0]  # Red color in RGB

    # Define a list of colors for the ROIs
    colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
        [1, 0, 1]   # Magenta
    ]
    # If circular ROIs are provided, set their pixels to different colors
    if rois:
        for i, roi_set in enumerate(rois):
            for j, roi in enumerate(roi_set):
                if roi is not None:
                    x_center, y_center, radius = roi['x'], roi['y'], roi['radius']
                    color = colors[j % len(colors)]  # Cycle through the colors
                    for x in range(int(x_center - radius), int(x_center + radius)):
                        for y in range(int(y_center - radius), int(y_center + radius)):
                            if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                                if 0 <= x < img_array.shape[0] and 0 <= y < img_array.shape[1]:
                                    rgb_image[x, y] = color  # Set the color for the ROI


    # Display the RGB image
    ax.imshow(rgb_image)
    
    # This function can be modified to display the maximum pixel value elsewhere or removed if not needed
    max_pixel_value = np.max(img_array)
    max_pixel_label.config(text=f"Max Pixel Value: {max_pixel_value}")

    canvas.draw()

# Function to draw ROIs on the DICOM slice
def draw_rois():
    global current_index, dicom_images, roi_entries, rois
    
    dicom_image = dicom_images[current_index]
    
    # Read in the pixel size of the DICOM image
    pixel_spacing = dicom_image[0x0028, 0x0030].value

    # Initialize a 12x5 matrix to store multiple sets of ROIs
    rois = [[None for _ in range(6)] for _ in range(12)]

    # Get ROI coordinates from the input fields and append to the matrix
    for i in range(12):
        try:
            x = int(roi_entries[i]['x'].get())
            y = int(roi_entries[i]['y'].get())

            # Divide the radius by the pixel spacing to convert from mm to pixels
            rois[i][0] = {'x': x, 'y': y, 'radius': round(37/2/pixel_spacing[0])}  # Circular 37 mm diameter ROI
            rois[i][1] = {'x': x, 'y': y, 'radius': round(28/2/pixel_spacing[0])}  # Inner ROIs with decreasing diameters
            rois[i][2] = {'x': x, 'y': y, 'radius': round(22/2/pixel_spacing[0])}
            rois[i][3] = {'x': x, 'y': y, 'radius': round(17/2/pixel_spacing[0])}
            rois[i][4] = {'x': x, 'y': y, 'radius': round(13/2/pixel_spacing[0])}
            rois[i][5] = {'x': x, 'y': y, 'radius': round(10/2/pixel_spacing[0])}

        except ValueError:
            continue  # If invalid input, skip this ROI
    
    # Display the slice with the updated ROIs
    display_dicom_image(dicom_image, canvas, ax, rois, roi_pixels)

# Function to handle "Next" button click
def next_slice():
    global current_index
    if current_index < len(dicom_images) - 1:
        current_index += 1
        draw_rois()
        slice_slider.set(current_index)

# Function to handle "Previous" button click
def previous_slice():
    global current_index
    if current_index > 0:
        current_index -= 1
        draw_rois()
        slice_slider.set(current_index)

# Calculate background variability according to NEMA NU 2-2007
def background_variability(current_index):
    global roi_pixels, rois, dicom_images

    # Extract slice_thickness using the DICOM tag's hexadecimal code
    slice_thickness = dicom_images[0][0x0018, 0x0050].value
    print(f"slice_thickness: {slice_thickness}")

    # Calculate the number of slices for 10 mm and 20 mm
    slices_in_10mm = round(10.0 / slice_thickness)
    slices_in_20mm = round(20.0 / slice_thickness)
    # Calculate the specific slices based on the distance
    slice_numbers = [
        max(0, current_index - slices_in_20mm), 
        max(0, current_index - slices_in_10mm),
        current_index, 
        min(len(dicom_images) - 1, current_index + slices_in_10mm), 
        min(len(dicom_images) - 1, current_index + slices_in_20mm)
    ]

    # Store mean values for each slice
    # mean_values = []
    
    # Initialize a list to store the mean values for each column
    column_mean_values = [[] for _ in range(6)]
    # Initialize a list to store the mean values for each of the 60 ROIs of the same size
    mean_values = [[[] for _ in range(6)] for _ in range(60)]

    for slice_idx in slice_numbers:
        #pixel_values = []  # Initialize here to reset for each slice

        img_array = dicom_images[slice_idx].pixel_array    

        # Extract pixel values from circular rois
        # There are 5 slices which are being analyzed and in each of these 5 slices we have a rois matrix of 12x6 rois.
        # I want to save for each column of the rois matrix the average pixel value for the 12 rois per slice, i.e. 60 average pixel
        # values of the 12 pixels per slice with 5 slices. So in the end I want a 60x6 matrix with the average pixel values of each ROI.


        if rois:
            for col in range(6):  # Iterate over each column of ROIs
                column_pixel_values = []  # Collect pixel values for the current column
                for row_idx, row in enumerate(rois):  # Iterate over each row of ROIs
                    roi = row[col]
                    if roi is not None:
                        x_center, y_center, radius = roi['x'], roi['y'], roi['radius']
                        for x in range(int(x_center - radius), int(x_center + radius) + 1):
                            for y in range(int(y_center - radius), int(y_center + radius) + 1):
                                if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                                    if 0 <= x < img_array.shape[0] and 0 <= y < img_array.shape[1]:
                                        column_pixel_values.append(img_array[x, y])
                                        # Calculate the average pixel value for the current column and row and slice

                        if column_pixel_values:
                            average_pixel_value = np.mean(column_pixel_values)
                        else:
                            average_pixel_value = None  

                        # Calculate the index in the 60x6 matrix
                        matrix_row_idx = slice_numbers.index(slice_idx) * 12 + row_idx   
                        mean_values[matrix_row_idx][col].append(average_pixel_value)  
                        #if average_pixel_value is not None:
                        #    print(f"Row {matrix_row_idx}, Column {col}: {average_pixel_value:.2f}")
                        #else:
                        #    print(f"Row {matrix_row_idx}, Column {col}: None")
                        
                          
                                    

                # Calculate the average pixel value for the current column and slice
                if column_pixel_values:
                    average_pixel_value = np.mean(column_pixel_values)
                    column_mean_values[col].append(average_pixel_value)
                else:
                    column_mean_values[col].append(None)


            
    
    # Print the 60x6 matrix with the average pixel values
    for row_idx, row in enumerate(mean_values):
        for col_idx, col in enumerate(row):
            if col:
                print(f"Row {row_idx}, Column {col_idx}: {np.mean(col):.2f}")
            else:
                print(f"Row {row_idx}, Column {col_idx}: None")

    # Calculate the final average for each column over the 5 slices
    final_mean_values = []
    for col_values in column_mean_values:
        # Filter out None values before averaging
        valid_values = [val for val in col_values if val is not None]
        if valid_values:
            final_mean_values.append(np.mean(valid_values))
        else:
            final_mean_values.append(None)
            #final_mean_values.append(None)

    roi_sizes = [37, 28, 22, 17, 13, 10]
    # Print the average values with the corresponding ROI sizes
    for size, mean_value in zip(roi_sizes, final_mean_values):
        if mean_value is not None:
            print(f"Average value for ROI size {size} mm: {mean_value:.2f}")
        else:
            print(f"Average value for ROI size {size} mm: None")

    # Calculate standard deviation according to NEMA NU 2-2007
    K = 60 # Number of background ROIs
    
   # Calculate the sum of squared differences for each column (so for each sphere size, see NEMA NU 2-2007)
    sum_squared_differences = [0] * 6
    for row_idx, row in enumerate(mean_values):
        for col_idx, col in enumerate(row):
            if col is not None and final_mean_values[col_idx] is not None:
                sum_squared_differences[col_idx] += (np.mean(col) - final_mean_values[col_idx]) ** 2

    # Print the sum of squared differences for each column
    for col in range(6):
        print(f"Sum of squared differences for column {col}: {float(sum_squared_differences[col]):.2f}")
        
    # Divide the sum of squared differences by K - 1 to get the variance
    variances = [ssd / (K - 1) for ssd in sum_squared_differences]
    print("Variances:", variances)

    # Calculate the standard deviation by taking the square root of the variance
    standard_deviations = [np.sqrt(var) for var in variances]
    print("Standard Deviations:", standard_deviations)

    # Initialize an array to store the background variability for each column
    background_variability_array = []

    # Calculate the percent background variability N_j for sphere size j (i.e. for each column which represent the different spheres)
    for col, (standard_deviation, mean_value) in enumerate(zip(standard_deviations, final_mean_values)):
        if standard_deviation is not None and mean_value is not None:
            background_variability = standard_deviation / mean_value * 100
            background_variability_array.append(background_variability)
            print(f"Percent background variability for ROI size {roi_sizes[col]} mm: {background_variability:.2f}%")
        else:
            background_variability_array.append(None)
            print(f"Percent background variability for ROI size {roi_sizes[col]} mm: None")

    return background_variability_array

# Function to handle "Select" button click (saves the current slice)
def select_slice():
    selected_slice = dicom_images[current_index]
    save_selected_slice(selected_slice)
    background_variability(current_index)
    process_rois_for_predefined_centers()
    suv_peak_with_spherical_voi()

# Function to save the selected DICOM slice
def save_selected_slice(dicom_image):
    save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if save_path:
        plt.imsave(save_path, dicom_image.pixel_array, cmap='gray')
        messagebox.showinfo("Save Successful", f"Slice saved as {save_path}")

# Function to handle scrollbar movement
def on_slider_change(val):
    global current_index
    current_index = int(val)
    draw_rois()

# Function to allow the user to select a folder and load DICOM images
def load_folder():
    global dicom_images, current_index
    directory = filedialog.askdirectory()  # Open folder dialog for the user to select a directory
    if directory:
        dicom_images = load_dicom_images(directory)  # Load DICOM images from the selected folder
        if dicom_images:
            current_index = 0
            slice_slider.config(to=len(dicom_images) - 1)  # Update scrollbar range
            draw_rois()  # Display the first slice with ROIs
        else:
            messagebox.showerror("Error", "No DICOM images found in the selected folder.")

# Function to apply threshold and expand ROI pixels using nearest neighbor
def apply_threshold():
    global current_index, dicom_images, threshold_entry, canvas, ax, rois, roi_pixels
    threshold = float(threshold_entry.get())
    image = dicom_images[current_index].pixel_array
    # Binary image where pixels are above the threshold
    binary_image = image > threshold
    # Label connected components
    labeled, num_features = ndimage.label(binary_image)
    roi_pixels = []

    for region_index in range(1, num_features + 1):
        coords = np.column_stack(np.where(labeled == region_index))
        for coord in coords:
            if image[coord[0], coord[1]] >= threshold * 0.95:
                roi_pixels.append((coord[0], coord[1]))

    # Display the updated image with ROI pixels highlighted
    display_dicom_image(dicom_images[current_index], canvas, ax, rois, roi_pixels)


def get_max_value(img_array, center, radius):
    """ Extract the maximum value within a spherical ROI """
    x_center, y_center = center
    max_value = 0
    for y in range(max(0, int(y_center - radius)), min(img_array.shape[0], int(y_center + radius) + 1)):
        for x in range(max(0, int(x_center - radius)), min(img_array.shape[1], int(x_center + radius) + 1)):
            if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                max_value = max(max_value, img_array[x, y])
    return max_value

def get_mean_value(image_stack, mask):
    """
    Calculate the mean value of the pixel values within the mask.
    image_stack: 3D image stack.
    mask: 3D boolean mask.
    """
    global current_index
    # Extract the z, y, and x coordinates from the mask
    z_coords, y_coords, x_coords = np.where(mask)
    
    # Adjust the z coordinates by adding current_index
    adjusted_z_coords = z_coords + current_index
    
    # Ensure the adjusted z coordinates are within bounds
    adjusted_z_coords = np.clip(adjusted_z_coords, 0, image_stack.shape[0] - 1)
    
    # Extract the pixel values using the adjusted z, y, and x coordinates
    pixel_values = image_stack[adjusted_z_coords, y_coords, x_coords]
    
    # Calculate and return the mean value of the pixel values
    return np.mean(pixel_values)

def suv_peak_with_spherical_voi():
    global current_index, dicom_images, roi_masks

    radius_mm = 6.204  # Sphere radius in mm for a 1 mL 3D sphere
    dicom_image = dicom_images[current_index]
    
    # Read in the pixel size of the DICOM image
    pixel_spacing = dicom_image[0x0028, 0x0030].value
    # Extract slice_thickness using the DICOM tag's hexadecimal code
    slice_thickness = dicom_image[0x0018, 0x0050].value
    # Calculate the radius in pixels (assuming isotropic pixels)
    radius_pixels = radius_mm / pixel_spacing[0]   
    # Determine the number of slices the sphere covers
    num_slices = int(np.ceil(2 * radius_mm / slice_thickness))
    
    print(f"Number of pixels equalling radius of 6.204 mm sphere: {radius_pixels:.2f}, Number of slices the sphere covers: {num_slices}")

    mean_values = []
    positions = []  # To save the sphere centers that give valid results
    image_stack = build_image_stack()
    
    # Convert roi_masks to a NumPy array
    roi_masks_array = np.array(roi_masks)

    print(f"Shape of image stack: {image_stack.shape}")
    print(f"roi_masks shape: {roi_masks_array.shape}")
    print(f"roi_masks content: {roi_masks_array}")
    # Assume roi_mask is a boolean 3D array

    # Initialize a dictionary to store the maximum mean value for each z-coordinate
    max_values_per_slice = {z: {'max_mean': 0, 'position': None} for z in range(image_stack.shape[0])}

    # Loop through each index where roi_masks is True
    for z, y, x in np.argwhere(roi_masks_array):
        index = (z, y, x)
        if can_place_sphere(index, radius_pixels):
            mask = create_3d_spherical_mask(index, radius_pixels, image_stack.shape)
            mean_value = get_mean_value(image_stack, mask)
            if mean_value > max_values_per_slice[z]['max_mean']:
                max_values_per_slice[z]['max_mean'] = mean_value
                max_values_per_slice[z]['position'] = (y, x)  # Store the y, x position for the max value

    # Print results for each slice
    for z, details in max_values_per_slice.items():
        print(f"Max mean value in slice {z}: {details['max_mean']} at position {details['position']}")

    return max_values_per_slice
    if False:
        for index in np.argwhere(roi_masks):
            print("I am here")
            if can_place_sphere(index, radius_pixels):
                print(f"Valid sphere center: {index}")
                mask = create_3d_spherical_mask(index, radius_pixels, image_stack.shape)
                #print(f"Mask created by create_3d_spherical_mask: {mask}")
                mean_value = get_mean_value(image_stack, mask)
                print(f"SUV_peak: Mean value: {mean_value} at position {index}")
                mean_values.append(mean_value)
                positions.append(index)
            
        # Identify the maximum mean value
        max_index = np.argmax(mean_values)
        max_mean_value = mean_values[max_index]
        max_position = positions[max_index]
            
        print(f"SUV_peak: Max mean value: {max_mean_value} at position {max_position}")
        return max_mean_value, max_position    

def can_place_sphere(center, radius_pixels):
    """
    Check if a sphere with given radius can be placed within the 2D ROI.
    center: (x_center, y_center) - Center of the sphere in the 2D image.
    radius_pixels: Radius of the sphere in pixels.
    roi_mask: 2D boolean array where True values indicate the ROI.
    """
    global roi_masks

    z_center, y_center, x_center = center
    
    # Convert roi_masks to a NumPy array
    roi_masks_array = np.array(roi_masks)
    depth, height, width = roi_masks_array.shape
    if False:
        # Check if all points within the sphere's radius are within the ROI and image boundaries
        for y in range(max(0, int(y_center - radius_pixels)), min(height, int(y_center + radius_pixels) + 1)):
            for x in range(max(0, int(x_center - radius_pixels)), min(width, int(x_center + radius_pixels) + 1)):
                if (x - x_center)**2 + (y - y_center)**2 <= radius_pixels**2:
                    if not roi_masks[y, x]:  # Check if the point is outside the ROI
                        return False
    print(f"depth: {depth}, height: {height}, width: {width} z_center: {z_center}, y_center: {y_center}, x_center: {x_center}, radius_pixels: {radius_pixels}")
    # Check if the sphere fits within the bounds of the 3D ROI
    if (x_center - radius_pixels >= 0 and x_center + radius_pixels < height and
        y_center - radius_pixels >= 0 and y_center + radius_pixels < width):
        return True
    else:
        return False

def create_3d_spherical_mask(center, radius_pixels, shape):
    """
    Create a 3D spherical mask.
    center: (z_center, y_center, x_center) - Center of the sphere in the 3D image.
    radius_pixels: Radius of the sphere in pixels.
    num_slices: Number of slices the sphere covers.
    shape: Shape of the 3D image stack.
    """
    z_center, y_center, x_center = center
    depth, height, width = shape
    
    z, y, x = np.ogrid[:depth, :height, :width]
    distance = np.sqrt((z - z_center)**2 + (y - y_center)**2 + (x - x_center)**2)
    mask = distance <= radius_pixels
    return mask


def build_image_stack():
    global dicom_images
    #print(f"Content of dicom_images: {dicom_images}")
    #print(f"Pixel array of the first DICOM image: {dicom_images[0].pixel_array}")
    #print(f"Pixel array of the 50th DICOM image: {dicom_images[49].pixel_array}")
    # Ensure dicom_images is not None and contains elements
    if dicom_images is None or len(dicom_images) == 0:
        raise ValueError("dicom_images is not initialized or empty")

    # Sort DICOM files by their acquisition number or another relevant attribute for correct sequence
    try:
        temp_data = sorted(dicom_images, key=lambda x: int(x.InstanceNumber))
    except AttributeError as e:
        raise ValueError("DICOM images do not have the expected attributes") from e

    # Create a 3D numpy array from the DICOM pixel arrays
    try:
        image_stack = np.stack([ds.pixel_array for ds in temp_data])
    except AttributeError as e:
        raise ValueError("DICOM objects do not have pixel_array attribute") from e

    return image_stack

def create_isocontour_voi(img_array, center, radius, threshold):
    """ Creates a binary mask for the isocontour ROI based on the threshold. """
    y_center, x_center = center
    mask = np.zeros_like(img_array, dtype=bool)
    # Rows (y coord.)
    for y in range(max(0, int(y_center - radius)), min(img_array.shape[0], int(y_center + radius) + 1)):
        # Columns (x coord.)
        for x in range(max(0, int(x_center - radius)), min(img_array.shape[1], int(x_center + radius) + 1)):
            if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                if img_array[y, x] >= threshold:
                    mask[y, x] = True
    return mask

def process_rois_for_predefined_centers():
    global roi_masks
    selected_slice = dicom_images[current_index].pixel_array
    print(f"Selected slice: {selected_slice}")
    # Centers of the 6 spheres with a 344x344 image size, increasing sphere sizes
    # centers = [(200, 165), (189, 190), (160, 194), (144, 171), (154, 146), (183, 142)] 
    # Centers of the 6 spheres with a 512x512 image size, increasing sphere sizes
    centers = [(210, 271), (217, 228), (257, 214), (287, 242), (280, 282), (242, 298)]
    
    radius = 15  # Covers even the biggest sphere with a radius of 18.5 pixels (times approx. 2 mm pixel_spacing = 37 mm sphere)
    roi_masks = []
    # roi_pixels = []  # Initialize roi_pixels as an empty list


    for center in centers:
        # Assuming a threshold of 40% of the max value within each sphere's bounding box
        #local_max = np.max(selected_slice[
        #    max(0, center[0] - radius):min(selected_slice.shape[0], center[0] + radius),
        #    max(0, center[1] - radius):min(selected_slice.shape[1], center[1] + radius)
        #])
        true_activity = 28136.08 #Calculated the theoretical activity at scan start (Daniel, 10. Oct. 2024 12:22 pm)
        threshold = 0.4 * true_activity #local_max
        roi_mask_temp = create_isocontour_voi(selected_slice, center, radius, threshold)
        print(f"VOI {len(roi_masks) + 1} - Threshold: {threshold:.2f}, Max Value: {true_activity:.2f}, Number of Pixels: {np.sum(roi_mask_temp)}")
        roi_masks.append(roi_mask_temp)
    print(f"roi_masks: {roi_masks}")
        # Create circular ROI and extract coordinate pairs to see if the radius of the max value search and the ROIs in which the max value is searched is correct
        #rr, cc = np.ogrid[:selected_slice.shape[0], :selected_slice.shape[1]]
        #circle_mask = (rr - center[0])**2 + (cc - center[1])**2 <= radius**2
        #roi_coords = np.column_stack(np.where(circle_mask))
        #roi_pixels.append(roi_coords)
    display_dicom_image(selected_slice, canvas, ax)
    #display_dicom_image(selected_slice, canvas, ax, roi_pixels=roi_pixels)
  
    if False:
        for center in centers:
            # Assuming a threshold of 40% of the max value within each sphere's bounding box
            local_max = np.max(selected_slice[
                max(0, center[0] - radius):min(selected_slice.shape[0], center[0] + radius),
                max(0, center[1] - radius):min(selected_slice.shape[1], center[1] + radius)
            ])
            threshold = 0.4 * local_max
            voi_mask = create_isocontour_voi(selected_slice, center, radius, threshold)
            print(f"VOI {len(vois) + 1} - Threshold: {threshold:.2f}, Max Value: {local_max:.2f}, Number of Pixels: {np.sum(voi_mask)}")
            vois.append(voi_mask)
    
        display_dicom_image(selected_slice, canvas, ax, voi_masks=vois)
    # Initialize an array to store the mean values of the different VOIs
    mean_values = []

    # Calculate the mean values of the different ROIs
    for i, roi_in_roi_masks in enumerate(roi_masks):
        mean_value = np.mean(selected_slice[roi_in_roi_masks])
        num_pixels = np.sum(roi_in_roi_masks)
        mean_values.append(mean_value)
        print(f"Mean value for VOI {i + 1}: {mean_value:.2f}")
        print(f"Number of pixels in VOI {i + 1}: {num_pixels}")

    # Calculate the max value from the very first VOI (i.e. the 37 mm sphere)
    max_value = true_activity #np.mean(selected_slice[roi_masks[5]])
    print(f"Max value from the biggest VOI: {max_value:.2f}")

    

    # Calculate the recovery coefficient of the different ROIs using the stored mean values
    for i, mean_value in enumerate(mean_values):
        recovery_coefficient = mean_value / max_value
        recovery_coefficients.append(recovery_coefficient)
        print(f"Recovery coefficient for VOI {i + 1}: {recovery_coefficient:.2f}")


    # Ensure the length of voi_sizes matches the length of recovery_coefficients
    #if len(voi_sizes) != len(recovery_coefficients):
    #    raise ValueError("The length of VOI numbers does not match the length of recovery coefficients.")
    
    # Convert roi_masks to a NumPy array
    roi_masks_array = np.array(roi_masks)
    print(f"Roi masks shape: {roi_masks_array.shape}")
    return roi_masks


def draw_plot():
    global iteration_count, recovery_coefficients
    
    voi_sizes = [10, 13, 17, 22, 28, 37]
    
    # Reshape recovery_coefficients to a 2D array where each row is a set of 6 coefficients
    recovery_coefficients_reshaped = np.reshape(recovery_coefficients, (-1, 6))
    
    # Create figure and axis locally
    fig, ax = plt.subplots()
    
    # Plot each set of recovery coefficients
    for i, recovery_coeffs in enumerate(recovery_coefficients_reshaped):
        ax.plot(voi_sizes, recovery_coeffs, marker='o', label=f'Iteration {i + 1}')
    
    ax.set_xlabel('Sphere Size [mm]')
    ax.set_ylabel('Recovery Coefficient [1]')
    ax.set_title('Recovery Coefficients vs Sphere Size')
    ax.grid(True)
    ax.legend()

    # Save the plot to a pickle file
    pickle_path = f'C:/Users/DANIE/OneDrive/FAU/Master Thesis/Project/Python/Recovery coefficients/Phantom-01.pickle'
    with open(pickle_path, 'wb') as f:
        pickle.dump(fig, f)

    #plt.show()  # Show the plot and block interaction until closed
    #plt.close(fig)  # Ensure the figure is closed after displaying
    
    plt.draw()  # Update the plot without blocking

def show_plot():
    # Path to the pickle file
    pickle_path = 'C:/Users/DANIE/OneDrive/FAU/Master Thesis/Project/Python/Recovery coefficients/Phantom-01.pickle'

    # Load the figure object from the pickle file
    with open(pickle_path, 'rb') as f:
        fig = pickle.load(f)

    # Display the loaded figure
    fig.show()

def update_coords(event):
    global coords_label
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        coords_label.config(text=f"X: {x}, Y: {y}")
    else:
        coords_label.config(text="")

def zoom_in():
    global ax
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim([xlim[0]*0.9, xlim[1]*0.9])  # Reducing limits by 10%
    ax.set_ylim([ylim[0]*0.9, ylim[1]*0.9])
    canvas.draw()

def zoom_out():
    global ax
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim([xlim[0]*1.1, xlim[1]*1.1])  # Increasing limits by 10%
    ax.set_ylim([ylim[0]*1.1, ylim[1]*1.1])
    canvas.draw()

def on_click(event):
    # Check if the click was on the canvas and the left button was used
    if event.inaxes is not None and event.button == 1:
        # Calculate the new limits around the click point
        ax = event.inaxes
        xdata, ydata = event.xdata, event.ydata
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0]) * 0.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0]) * 0.5
        new_xlim = [xdata - cur_xrange * 0.5, xdata + cur_xrange * 0.5]
        new_ylim = [ydata - cur_yrange * 0.5, ydata + cur_yrange * 0.5]
        
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        canvas.draw()

def create_gui():
    global root, canvas, ax, slice_slider, roi_entries, threshold_entry, max_pixel_label, coords_label

    # Create main window
    root = tk.Tk()
    root.title("DICOM Viewer with Adjustable ROIs and Threshold")

    # Create a figure and axis for displaying DICOM images
    fig, ax = plt.subplots(figsize=(6, 6))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    canvas.mpl_connect('button_press_event', on_click)  # Connect the click event

    # Add a button to load the DICOM folder
    load_button = tk.Button(root, text="Load Folder", command=load_folder)
    load_button.pack(side=tk.TOP, padx=10, pady=10)
    
    zoom_in_button = tk.Button(root, text="Zoom In", command=zoom_in)
    zoom_in_button.pack(side=tk.LEFT, padx=5, pady=10)

    zoom_out_button = tk.Button(root, text="Zoom Out", command=zoom_out)
    zoom_out_button.pack(side=tk.LEFT, padx=5, pady=10)

    # VOI Processing Button
    process_voi_button = tk.Button(root, text="Isocontour detection", command=process_rois_for_predefined_centers)
    process_voi_button.pack(side=tk.LEFT, padx=20, pady=10)

    # Draw Plot Button
    draw_plot_button = tk.Button(root, text="Draw Plot", command=draw_plot)
    draw_plot_button.pack(side=tk.LEFT, padx=25, pady=10)

    # Show Plot Button
    show_plot_button = tk.Button(root, text="Show Plot", command=show_plot)
    show_plot_button.pack(side=tk.LEFT, padx=25, pady=10)

    # Label for displaying the maximum pixel value
    max_pixel_label = tk.Label(root, text="Max Pixel Value: N/A")
    max_pixel_label.pack(side=tk.TOP, pady=10)

    # Add a horizontal scrollbar (slider) to scroll through the slices
    slice_slider = tk.Scale(root, from_=0, to=0, orient=tk.HORIZONTAL,
                            label="Slice", length=400, command=on_slider_change)
    slice_slider.pack(side=tk.TOP, padx=10, pady=10)

    # Add "Previous" and "Next" buttons to scroll through slices
    prev_button = tk.Button(root, text="Previous", command=previous_slice)
    prev_button.pack(side=tk.LEFT, padx=10, pady=10)

    next_button = tk.Button(root, text="Next", command=next_slice)
    next_button.pack(side=tk.LEFT, padx=10, pady=10)

    # Add "Select" button to save the current slice
    select_button = tk.Button(root, text="Select Slice", command=select_slice)
    select_button.pack(side=tk.LEFT, padx=10, pady=10)

    # ROI Input panel for 12 ROIs arranged in a 3x4 grid
    roi_panel = tk.Frame(root)
    roi_panel.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
    
    # Initialize the list to store the ROI entries
    roi_entries = []

    # Define default values for the ROI coordinates
    default_values = [
        {'x': 145, 'y': 137}, {'x': 161, 'y': 127}, {'x': 179, 'y': 127}, {'x': 197, 'y': 129},
        {'x': 205, 'y': 145}, {'x': 207, 'y': 176}, {'x': 205, 'y': 204}, {'x': 194, 'y': 219},
        {'x': 177, 'y': 222}, {'x': 141, 'y': 202}, {'x': 160, 'y': 222}, {'x': 132, 'y': 169}
    ]

    
    coords_label = tk.Label(root, text="Move cursor over image")
    coords_label.pack(side=tk.BOTTOM)

    canvas.mpl_connect('motion_notify_event', update_coords)

    # Create the ROI entry fields with default values
    for row in range(3):  # 3 rows
        for col in range(4):  # 4 columns
            roi_frame = tk.Frame(roi_panel)
            roi_frame.grid(row=row, column=col, padx=5, pady=5)
            
            tk.Label(roi_frame, text=f"ROI {row * 4 + col + 1} - X:").pack(side=tk.LEFT)
            x_entry = tk.Entry(roi_frame, width=5)
            x_entry.pack(side=tk.LEFT, padx=5)
            x_entry.insert(0, default_values[row * 4 + col]['x'])  # Insert default x value
            
            tk.Label(roi_frame, text="Y:").pack(side=tk.LEFT)
            y_entry = tk.Entry(roi_frame, width=5)
            y_entry.pack(side=tk.LEFT, padx=5)
            y_entry.insert(0, default_values[row * 4 + col]['y'])  # Insert default y value
            
            roi_entries.append({'x': x_entry, 'y': y_entry})

    # Threshold input field and button
    threshold_label = tk.Label(root, text="Threshold Value:")
    threshold_label.pack(side=tk.TOP, pady=5)
    threshold_entry = tk.Entry(root, width=10)
    threshold_entry.pack(side=tk.TOP, pady=5)
    
    threshold_button = tk.Button(root, text="Apply Threshold", command=apply_threshold)
    threshold_button.pack(side=tk.TOP, padx=10, pady=10)

    # Button to update the ROIs based on input values
    update_rois_button = tk.Button(root, text="Update ROIs", command=draw_rois)
    update_rois_button.pack(side=tk.BOTTOM, padx=10, pady=10)

    # Start the GUI main loop
    root.mainloop()

if __name__ == "__main__":

    # Create and launch the GUI
    create_gui()
