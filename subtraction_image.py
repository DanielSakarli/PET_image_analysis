# README:
# The usage of the script to plot several recovery coefficients of different reconstructions is as follows
# 1. Run the script 
# 2. Load the DICOM images from the folder containing the DICOM files
# 3. Click on the "Isocontour detection" button to detect the ROIs for the predefined centers
# 4. Repeat step 2 and 3 for different reconstructions
# 5. When all the reconstructed images are loaded, click on the "Draw Plot" button to plot the recovery coefficients. Note: this will only save the plot, but not show it yet
# 6. Click on the "Show Plot" button to display the plot
# Plot the absolute errors of the different SUV modes as follows:
# 1. Run the script 
# 2. Load the DICOM images from the folder containing the DICOM files
# 3. Click on the "Calculate SUV_N" button to calculate the SUV_N values for the predefined centers
# 4. Close the 2 figures for the code to continue (1: the figure of the selected slice, 2: the delta SUV vs Mode of SUV plot)
# 5. Click on "Yes" when it asks you if you want to load more data. It will show you for the first iteration the summed absolute delta SUV vs Mode of SUV plot
# 6. Do not close this plot but just repeat now step 2-5 (the first figure of step 4 will not appear again, just close the second figure of step 4)
# Get Image Roughness in the background as follows:
# 1. Run the script
# 2. Load the DICOM images from the folder containing the DICOM files and choose the slice with the biggest sphere sizes
# 3. Click on the "Select Slice" button to select the slice for which you want to calculate the image roughness
# 4. It will ask you if you want to save the image, you can either click yes or no
# 5. The image roughness will be calculated and displayed in a plot
# 6. It will ask you if you want to save the plot, you can either click yes or no
# 7. To calculate the image roughness for another recon, repeat step 2-6. It will automatically add the image roughness of the new recon to the same plot

import concurrent.futures
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
from matplotlib.colors import TwoSlopeNorm

# Initialize global variables
rgb_image = None
rois = None
roi_pixels = None
roi_masks = []
colorbar = None
# Global variables to maintain state across multiple DICOM folder additions
iteration_count = 0
legend_entries = []
# Initialize an array to store the recovery coefficients
SUV_max_values = []
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
        try:
            dicom_image = pydicom.dcmread(filepath)
            dicom_images.append(dicom_image)
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    #print(f"Pixel array of dicom images in load_dicom_images: {dicom_images[0].pixel_array}")
    #print(f"Five maximal values of the 62nd DICOM image: {np.sort(dicom_images[61].pixel_array.flatten())[-5:]}")
    return dicom_images

def display_dicom_image(ax, canvas, dicom_matrix=None):
    """
    Display a subtraction DICOM image with a color scale.
    Args:
    subtraction_img (np.array): The 2D numpy array of the subtraction image.
    ax (matplotlib.axes.Axes): The matplotlib axes to plot on.
    canvas (FigureCanvasTkAgg): The tkinter canvas for matplotlib.
    max_pixel_label (tk.Label): Label to display the maximum pixel value.
    """
    global current_index, dicom_images, colorbar
    # test comment
    ax.clear()  # Clear the axes to refresh the image display
    if dicom_matrix is not None:
        img_slice = dicom_matrix[current_index]
        dicom_images = dicom_matrix
    else:
        img_slice = dicom_images[current_index].pixel_array
        # Convert the image data to a numerical array (e.g., np.float32)
        img_slice = np.frombuffer(img_slice, dtype=np.int16).reshape(dicom_images[current_index].Rows, dicom_images[current_index].Columns)

    
    # Normalize the image array to the range [-1, 1]
    img_slice_normalized = 2 * (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice)) - 1

    # Convert the normalized data to RGB using a blue-white-red colormap
    rgb_image = plt.cm.bwr(img_slice_normalized)

    # Display the RGB image
    im = ax.imshow(rgb_image, interpolation='nearest')
    im.set_clim(-1, 1)  # Set the color limits to [-1, 1]
    #ax.axis('off')  # Optionally turn off the axes

    # Add a colorbar if needed
    if colorbar is None:
        colorbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    # Show plot within matplotlib's viewer
    #plt.show()
    # Draw the canvas
    canvas.draw()
    

def ddisplay_dicom_image(dicom_image, canvas, ax, rois=None, roi_pixels=None):
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
    roi_color = np.array([0, 1, 1])  # Cyan color for VOI
    if roi_masks:
        alpha = 0.5  # Transparency level for the ROI masks
        for mask in roi_masks:
            if mask.shape == (512, 512):
                # Blend the ROI color with the existing color
                rgb_image[mask] = (1 - alpha) * rgb_image[mask] + alpha * roi_color
            elif mask.shape == (127, 512, 512):
                # Extract the relevant slice from the 3D mask
                slice_index = current_index  # Assuming current_index is the relevant slice index
                if 0 <= slice_index < mask.shape[0]:
                    rgb_image[mask[slice_index]] = (1 - alpha) * rgb_image[mask[slice_index]] + alpha * roi_color
                else:
                    print(f"Slice index {slice_index} is out of bounds for mask with shape {mask.shape}")
            else:
                print(f"Mask shape {mask.shape} does not match image shape {img_array.shape}")

    
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
            for j, roi in enumerate(reversed(roi_set)): # Reverse the order to start with the largest ROI, so the colour of the larger roi don't cover the smaller ones (otherwise the small rois won't appear in the image)
                if roi is not None:
                    x_center, y_center, radius = roi['x'], roi['y'], roi['radius']
                    color = colors[j % len(colors)]  # Cycle through the colors
                    for x in range(int(x_center - radius), int(x_center + radius)):
                        for y in range(int(y_center - radius), int(y_center + radius)):
                            if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
                                if 0 <= x < img_array.shape[0] and 0 <= y < img_array.shape[1]:
                                    rgb_image[y, x] = color  # Set the color for the ROI


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
            rois[i][0] = {'x': x, 'y': y, 'radius': 10/2/pixel_spacing[0]}  # Circular 10 mm diameter ROI
            rois[i][1] = {'x': x, 'y': y, 'radius': 13/2/pixel_spacing[0]}  # Inner ROIs with increasing diameters
            rois[i][2] = {'x': x, 'y': y, 'radius': 17/2/pixel_spacing[0]}
            rois[i][3] = {'x': x, 'y': y, 'radius': 22/2/pixel_spacing[0]}
            rois[i][4] = {'x': x, 'y': y, 'radius': 28/2/pixel_spacing[0]}
            rois[i][5] = {'x': x, 'y': y, 'radius': 37/2/pixel_spacing[0]}

        except ValueError:
            continue  # If invalid input, skip this ROI
    
    # Display the slice with the updated ROIs
    display_dicom_image(canvas, ax)

# Function to handle "Next" button click
def next_slice():
    global current_index
    if current_index < len(dicom_images) - 1:
        current_index += 1
        display_dicom_image(ax, canvas)
        slice_slider.set(current_index)

# Function to handle "Previous" button click
def previous_slice():
    global current_index
    if current_index > 0:
        current_index -= 1
        display_dicom_image(ax, canvas)
        slice_slider.set(current_index)

# Calculate background variability according to NEMA NU 2-2007
def background_variability(current_index):
    global roi_pixels, rois, dicom_images

    sphere_sizes = [10, 13, 17, 22, 28, 37]  # Sphere sizes in mm
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

    image_stack = build_image_stack()
    shape = image_stack.shape
    roi_masks = [[[] for _ in range(6)] for _ in range(60)] # Initialize a 60x6 matrix to store the ROI masks
    # Print the 60x6 matrix with the average pixel values
    for row_idx, row in enumerate(mean_values):
        for col_idx, col in enumerate(row):
            if col:
                # Extract the center and radius_pixels from rois
                roi = rois[row_idx % 12][col_idx]
                if roi is not None:
                    center = (roi['y'], roi['x'])  # Note: center is (y, x)
                    radius_pixels = roi['radius']
                    mask = create_2d_spherical_mask(center, radius_pixels, shape)
                    print(f"Sum of pixels in mask in background_variablity(): {np.sum(mask)}")
                    roi_masks[row_idx][col_idx] = mask
                print(f"ROI {row_idx}, Sphere Size {sphere_sizes[col_idx]} mm, Mean Value Within ROI: {np.mean(col):.2f}")
            else:
                print(f"ROI {row_idx}, Sphere Size {sphere_sizes[col_idx]} mm: No mean value")


    
    ir_values = get_ir_value(roi_masks, slice_numbers)

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

    roi_sizes = [10, 13, 17, 22, 28, 37]  # Sphere sizes in mm
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


def get_ir_value(masks, slice_numbers):
    """
    Calculate the image roughness in the background for all recons
    Followed fromula (3) of http://dx.doi.org/10.1088/0031-9155/55/5/013
    Takes the mean value of the sphere instead of SUV_N=40.
    image_stack: 3D image stack.
    mask: 3D boolean mask.
    sphere_index: Index of the sphere to access the correct SUV_N values.
    """
    global current_index, rois
    image_stack = build_image_stack()
															   
    # Convert masks to a NumPy array
    masks_array = np.array(masks)
    print(f"Shape of masks in get_ir_value: {masks_array.shape}")
												  
																					
												   
    '''
    # Earlier calculated SUV_N values for N = 40 for the different sphere sizes at recon NEMA_IQ_01/_02/_03/_......
    # Do not delete or change these values. If you want to update the values, comment the old values out.
    SUV_N = [
        # These are the SUV_N values for my NEMA IQ scan with background activity (ratio 1:4) from the 05.11.2024
        # Used a spherical VOI of the true size of the spheres, no isocontour detection becuase it was delineating pixels that were not part of the spheres
        [11217.52, 18049.38, 24637.45, 27948.10, 29819.10, 32157.55], #NEMA_IQ_01
        [13341.70, 23084.22, 29678.75, 30543.72, 31378.25, 31764.33], #NEMA_IQ_02
        [15063.55, 25432.20, 31010.53, 30502.62, 31531.20, 31496.33], #NEMA_IQ_03
        [16082.25, 26268.30, 30999.67, 30034.17, 31217.08, 31088.40], #NEMA_IQ_04
        [16950.70, 26895.42, 31082.22, 30094.42, 31345.62, 31257.05], #NEMA_IQ_05
        [17579.17, 27225.28, 31013.72, 30144.60, 31482.20, 31500.88], #NEMA_IQ_06
        [17914.97, 27191.67, 30674.42, 29999.25, 31397.58, 31506.95], #NEMA_IQ_07
        [17977.90, 26831.17, 30076.20, 29603.53, 31029.97, 31203.60]  #NEMA_IQ_08
    ]
    
    These are the SUV_N values for N = 15 and my first NEMA IQ scan without background activity from the 10.10.2024
    SUV_N = [
        [24112.53, 27057.20, 28927.80, 31733.00, 31394.60, 31100.07], #SUV_N_01
        [23197.80, 26127.93, 28382.27, 31224.20, 31661.93, 31961.00], #SUV_N_02
        [22330.27, 25897.73, 27985.33, 30909.07, 31821.67, 32160.80], #SUV_N_03
        [21833.73, 25974.00, 27838.13, 30907.33, 31895.27, 32314.60], #SUV_N_04
        [21341.13, 25854.53, 27627.93, 30744.87, 31737.33, 32199.73], #SUV_N_05
        [20835.33, 25594.93, 27327.27, 30440.73, 31461.13, 31928.67], #SUV_N_06
        [20438.00, 25366.53, 27110.00, 30185.53, 31237.40, 31706.07], #SUV_N_07
        [20131.80, 25170.27, 26933.20, 26933.20, 29963.80, 31056.60] #SUV_N_08
    ]
    '''
    sphere_sizes = [10, 13, 17, 22, 28, 37]  # Sphere sizes in mm
    ir_values = [[[] for _ in range(6)] for _ in range(60)]
    # Iterate through each mask (each corresponding to a different sphere size)
    for row_idx, row in enumerate(masks):
        z_coords = slice_numbers[row_idx // 12]  # Use the floor division operator of 12 to assign the z_coord
        for col_idx, mask in enumerate(row):
            if mask is not None:
                # Extract pixel values from the image stack where mask is True
                # Create a 3D mask with the z_coords dimension
                mask_3d = np.zeros_like(image_stack, dtype=bool)
                #y_coords, x_coords = np.where(mask)
                mask_3d[z_coords, :, :] = mask
                pixel_values = image_stack[mask_3d]

                print(f"Mask size in pixels: {np.sum(mask_3d)}")
                mean_value = get_mean_value(image_stack, mask_3d)  # SUV_N value for this sphere in this recon
                temp_value = np.sum((pixel_values - mean_value) ** 2)
                ir = (np.sqrt(temp_value / (np.sum(mask_3d) - 1)) / mean_value)*100
                ir_values[row_idx][col_idx] = ir
                
                print(f"IR values for ROI {row_idx + 1} and sphere size {sphere_sizes[col_idx]} mm: {ir:.2f}")

    # Convert ir_values to a NumPy array for easier manipulation
    ir_values_array = np.array(ir_values, dtype=np.float64)
    
    # Calculate the mean values for each column (sphere size)
    mean_ir_values = np.nanmean(ir_values_array, axis=0)

    plot_ir_values(mean_ir_values)

    return ir_values


def plot_ir_values(ir_values):
    global iteration_count

    # Increment the iteration counter for the legend of the plot
    iteration_count += 1
    # Add the current iteration count to the legend entries
    #if iteration_count == 1:
    #    legend_entries.append(f'{iteration_count} iteration')
    #else:
    #    legend_entries.append(f'{iteration_count} iterations')
    sphere_sizes = [10, 13, 17, 22, 28, 37]
    #legend_entries = ['2 iterations, Gauss 3x3', '2 iterations, Gauss 5x5', '2 iterations, Gauss 7x7', '3 iterations, Gauss 3x3', '3 iterations, Gauss 5x5', '3 iterations, Gauss 7x7', '4 iterations, Gauss 3x3', '4 iterations, Gauss 5x5', '4 iterations, Gauss 7x7']
    legend_entries = ['Absolute Scattering, 2i', 'Relative Scattering, 2i', 'Absolute Scattering, 3i', 'Relative Scattering, 3i', 'Absolute Scattering, 4i', 'Relative Scattering, 4i']
    # Define line styles
    #line_styles = ['-', '--', '-.', '-', '--', '-.', '-', '--', '-.']
    line_styles = ['-', '--', '-', '--', '-', '--']
    # Define colors
    #colors = ['orange', 'orange', 'orange', 'green', 'green', 'green', 'red', 'red', 'red']
    colors = ['orange', 'orange', 'green', 'green', 'red', 'red']

    plt.figure(f'Image Roughness vs Sphere Size')
    plt.plot(sphere_sizes, ir_values, marker='o', linestyle=line_styles[iteration_count - 1], color=colors[iteration_count - 1])
    plt.xlabel('Sphere Sizes [mm]')
    plt.ylabel('Image Roughness [%]')
    plt.title('Image Roughness vs Sphere Size')
    plt.legend(legend_entries[0:iteration_count]) #, title=f'Number of iterations: ')
    plt.grid(True)
    plt.xticks(sphere_sizes)
    plt.ylim(0, 10)
    #plt.legend(recon_names, title=f'Number of iterations: ')
    plt.draw()

    # Show the plot to the user
    plt.show(block=False)

    save_path = "C://Users//DANIE//OneDrive//FAU//Master Thesis//Project//Data//Image Roughness"
    answer = messagebox.askyesno("Plot Saving", f"Do you want to save the plot here: {save_path}?")
    if answer: 
        # Save the plot as PNG, PDF, and pickle files
        png_path = os.path.join(save_path, 'NEMA_IQ_02_04-c_image_roughness_in_background_vs_sphere_size.png')
        pdf_path = os.path.join(save_path, 'NEMA_IQ_02_04-c_image_roughness_in_background_vs_sphere_size.pdf')
        pickle_path = os.path.join(save_path, 'NEMA_IQ_02_04-c_image_roughness_in_background_vs_sphere_size.pickle')
        
        plt.savefig(png_path)
        plt.savefig(pdf_path)
        with open(pickle_path, 'wb') as f:
            pickle.dump(plt.gcf(), f)
    # Ask user to load more data or not
    #answer = messagebox.askyesno("Load More Data", "Do you want to load more data?")
    #if answer:
    #    load_folder()
    # Show the plot again to ensure it remains visible
    plt.show() 
    
def plot_snr_values():
    # SNR calculation adapted from Tong et al. 2010 https://doi.org/10.1109/NSSMIC.2009.5401574 but with SUV_N=40 instead of SUV_mean
    global iteration_count

    
    sphere_sizes = [10, 13, 17, 22, 28, 37]
    # Do not delete or change these values. If you want to update the values, comment the old values out.
    # SUV_N values for N = 40 for NEMA IQ scan with background activity from the 05.11.2024
    SUV_N = [
        [13341.70, 23084.22, 29678.75, 30543.72, 31378.25, 31764.33], # NEMA_IQ_02
        [13207.15, 22949.40, 29626.25, 30591.67, 31388.78, 31683.80], # NEMA_IQ_02_c
        [15063.55, 25432.20, 31010.53, 30502.62, 31531.20, 31496.33], # NEMA_IQ_03
        [14951.60, 25349.90, 31000.88, 30566.65, 31563.42, 31438.22], # NEMA_IQ_03_c
        [16082.25, 26268.30, 30999.67, 30034.17, 31217.08, 31088.40], # NEMA_IQ_04
        [15977.73, 26192.25, 30982.05, 30077.90, 31234.95, 31012.97], # NEMA_IQ_04_c
    ]
    '''
    SUV_N = [
    # Scan from the 05.11.2024 with a 1:4 background activity ratio
            [13341.70, 23084.22, 29678.75, 30543.72, 31378.25, 31764.33], # NEMA_IQ_02
            [12482.75, 21252.53, 28507.85, 31075.72, 31578.72, 32145.90], # NEMA_IQ_02_a
            [11556.73, 18945.58, 26116.03, 30529.75, 31494.72, 32348.22], # NEMA_IQ_02_b
            [15063.55, 25432.20, 31010.53, 30502.62, 31531.20, 31496.33], # NEMA_IQ_03
            [13918.33, 23452.40, 30370.70, 31493.33, 31815.47, 32053.58], # NEMA_IQ_03_a
            [12649.10, 20726.67, 27998.78, 31479.95, 31848.30, 32322.97], # NEMA_IQ_03_b
            [16082.25, 26268.30, 30999.67, 30034.17, 31217.08, 31088.40], # NEMA_IQ_04
            [14750.90, 24351.83, 30816.50, 31237.28, 31641.05, 31745.53], # NEMA_IQ_04_a
            [13325.77, 21627.67, 28845.83, 31810.90, 32000.35, 32332.00]  # NEMA_IQ_04_b
    ]
    '''
    
    '''
    # Earlier calculated SUV_N values for N = 40 for the different sphere sizes at recon NEMA_IQ_01/_02/_03/_......
    SUV_N = [
        # These are the SUV_N values for my NEMA IQ scan with background activity (ratio 1:4) from the 05.11.2024
        # Used a spherical VOI of the true size of the spheres, no isocontour detection becuase it was delineating pixels that were not part of the spheres
        [11217.52, 18049.38, 24637.45, 27948.10, 29819.10, 32157.55], #NEMA_IQ_01
        [13341.70, 23084.22, 29678.75, 30543.72, 31378.25, 31764.33], #NEMA_IQ_02
        [15063.55, 25432.20, 31010.53, 30502.62, 31531.20, 31496.33], #NEMA_IQ_03
        [16082.25, 26268.30, 30999.67, 30034.17, 31217.08, 31088.40], #NEMA_IQ_04
        [16950.70, 26895.42, 31082.22, 30094.42, 31345.62, 31257.05], #NEMA_IQ_05
        [17579.17, 27225.28, 31013.72, 30144.60, 31482.20, 31500.88], #NEMA_IQ_06
        [17914.97, 27191.67, 30674.42, 29999.25, 31397.58, 31506.95], #NEMA_IQ_07
        [17977.90, 26831.17, 30076.20, 29603.53, 31029.97, 31203.60]  #NEMA_IQ_08
    ]
    '''
    true_activity_concentration = 26166.28 #Calculated the theoretical activity at scan start [Bq/mL] (Daniel, 05. Nov. 2024 11:36 am)

    # Calculate the SNR for each sphere size
    SUV_N_array = np.array(SUV_N)
    # Noise to Signal ratio
    nsr = np.sqrt((SUV_N_array - true_activity_concentration)**2) / true_activity_concentration
    # Signal to Noise ratio, normalized to the true activity concentration
    snr = 1 - nsr
    print(f"SNR values: {snr}")
    #legend_entries = ['1 iteration', '2 iterations', '3 iterations', '4 iterations', '5 iterations', '6 iterations', '7 iterations', '8 iterations']
    #legend_entries = ['2 iterations, Gauss 3x3', '2 iterations, Gauss 5x5', '2 iterations, Gauss 7x7', '3 iterations, Gauss 3x3', '3 iterations, Gauss 5x5', '3 iterations, Gauss 7x7', '4 iterations, Gauss 3x3', '4 iterations, Gauss 5x5', '4 iterations, Gauss 7x7']
    legend_entries = ['Absolute Scattering, 2i', 'Relative Scattering, 2i', 'Absolute Scattering, 3i', 'Relative Scattering, 3i', 'Absolute Scattering, 4i', 'Relative Scattering, 4i']
   
    # Define line styles
    line_styles = ['-', '--', '-', '--', '-', '--']
    #line_styles = ['-', '--', '-.', '-', '--', '-.', '-', '--', '-.']
    
    # Define colors
    colors = ['orange', 'orange', 'green', 'green', 'red', 'red']
    #colors = ['orange', 'orange', 'orange', 'green', 'green', 'green', 'red', 'red', 'red']
    
    # Plot the SNRs for each sphere size
    plt.figure('Signal-to-Noise Ratio vs Sphere Size')
    for i, snr_row in enumerate(snr):
        plt.plot(sphere_sizes, snr_row, marker='o', linestyle=line_styles[i], color=colors[i], label=legend_entries[i])
    
    #plt.figure('Signal-to-Noise Ratio vs Sphere Size')
    #for i, snr_row in enumerate(snr):
    #    plt.plot(sphere_sizes, snr_row, marker='o', zorder=3) #, label=f'{i + 1} iteration{"s" if i > 0 else ""}')
        #legend_entries.append(f'{i + 1} iteration{"s" if i > 0 else ""}')
    plt.xlabel('Sphere Sizes [mm]')
    plt.ylabel('SNR [1]')
    plt.title('Signal-to-Noise Ratio vs Sphere Size')
    plt.legend(legend_entries, title=f'Number of iterations: ')
    plt.grid(True)
    plt.xticks(sphere_sizes)
    plt.ylim(0.4, 1)
    #plt.legend(recon_names, title=f'Number of iterations: ')
    plt.draw()

    # Show the plot to the user
    plt.show(block=False)

    save_path = "C://Users//DANIE//OneDrive//FAU//Master Thesis//Project//Data//SNR"
    png_path = os.path.join(save_path, 'NEMA_IQ_02-04-c_SNR_vs_sphere_size_calculated_with_SUV_40.png')
    pdf_path = os.path.join(save_path, 'NEMA_IQ_02-04-c_SNR_vs_sphere_size_calculated_with_SUV_40.pdf')
    pickle_path = os.path.join(save_path, 'NEMA_IQ_02-04-c_SNR_vs_sphere_size_calculated_with_SUV_40.pickle')
    answer = messagebox.askyesno("Plot Saving", f"Do you want to save the plot here:\n{save_path}\nas:\n{png_path}?")
    if answer: 
        # Save the plot as PNG, PDF, and pickle files        
        plt.savefig(png_path)
        plt.savefig(pdf_path)
        with open(pickle_path, 'wb') as f:
            pickle.dump(plt.gcf(), f)
    # Ask user to load more data or not
    #answer = messagebox.askyesno("Load More Data", "Do you want to load more data?")
    #if answer:
    #    load_folder()
    # Show the plot again to ensure it remains visible
    plt.show() 



# Function to handle "Select" button click (saves the current slice)
def select_slice():
    selected_slice = dicom_images[current_index]
    save_selected_slice(selected_slice)
    background_variability(current_index)
    #process_rois_for_predefined_centers()
    #suv_peak_with_spherical_voi()

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
    display_dicom_image(ax, canvas)


# Function to allow the user to select a folder and load DICOM images
def load_folder():
    global dicom_images, current_index, loaded_folder_path
    directory = filedialog.askdirectory()  # Open folder dialog for the user to select a directory
    if directory:
        dicom_images = load_dicom_images(directory)  # Load DICOM images from the selected folder
        #print(f"dicom images: {dicom_images}")
        if dicom_images:
            current_index = 0
            loaded_folder_path = directory  # Save the path of the loaded folder
            slice_slider.config(to=len(dicom_images) - 1)  # Update scrollbar range
            display_dicom_image(ax, canvas)  # Display the first image
        else:
            messagebox.showerror("Error", "No DICOM images found in the selected folder.")

def subtract_images():
    global dicom_images
    # Load the two image sets
    load_folder()
    image_set_1 = dicom_images
    load_folder()
    image_set_2 = dicom_images
    subtraction_images = []

    for dicom1, dicom2 in zip(image_set_1, image_set_2):
        image1 = dicom1.pixel_array.astype(np.float32)
        image2 = dicom2.pixel_array.astype(np.float32)
        subtraction_image = image1 - image2
        subtraction_images.append(subtraction_image)

    # Convert list to a 3D NumPy array
    subtraction_matrix = np.stack(subtraction_images)
    print(f"Shape of subtraction matrix: {subtraction_matrix.shape}")

    display_dicom_image(ax, canvas, subtraction_matrix)

    return subtraction_matrix
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
    display_dicom_image(ax, canvas)


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
    #adjusted_z_coords = z_coords + current_index
    
    # Ensure the adjusted z coordinates are within bounds
    #adjusted_z_coords = np.clip(adjusted_z_coords, 0, image_stack.shape[0] - 1)
    
    # Extract the pixel values using the adjusted z, y, and x coordinates
    pixel_values = image_stack[z_coords, y_coords, x_coords]
    
																																			 
    # Calculate and return the mean value of the pixel values
    return np.mean(pixel_values)

def calculate_SUV_N():
    #process_rois_for_predefined_centers('roi') # initialize the 2D ROI mask
    #suv_peak_values = suv_peak_with_spherical_voi() # Get the SUV_peak with 2D ROI mask (3D is computationally too expensive)
    process_rois_for_predefined_centers('voi') # update the 2D ROI mask to be a 3D VOI mask for SUV_N calculation
    global dicom_images, current_index, roi_masks, iteration_count, loaded_folder_path
    sphere_sizes = [10, 13, 17, 22, 28, 37]  # Sphere sizes
    results = {size: [] for size in sphere_sizes}  # Dictionary to store results for each sphere size
    image_stack = build_image_stack()
    roi_masks_array = np.array(roi_masks)
    print(f"Shape of roi_masks_array: {roi_masks_array.shape}")
    while True:    
        # Extract the relevant slice from the image stack
        #current_slice = image_stack[current_index]
        # Extract the top N pixel values where roi_masks is True
        # Plot for SUV_N vs N for different spheres
        # Loop over each sphere in roi_masks_array
        for i, sphere_size in enumerate(sphere_sizes):
            masked_values = image_stack[roi_masks_array[i]]
            if masked_values.size == 0:
                print(f"No masked values found for sphere size {sphere_size} mm.")
                continue
            for N in range(5, 45, 5):  # Loop over N values from 5 to 40 in increments of 5
                if masked_values.size < N:
                    print(f"Not enough values for N={N} for sphere size {sphere_size} mm.")
                    results[sphere_size].append(np.nan)
                    continue
                top_N_values = np.partition(masked_values, -N)[-N:]
                mean_top_N = np.mean(top_N_values)
                results[sphere_size].append(mean_top_N)
                print(f"SUV_{N} for sphere size {sphere_size} mm: {mean_top_N:.2f} Bq/mL")

        # Update plot
        load_more_data = plot_SUV_N(sphere_sizes, results) # , suv_peak_values)
        if not load_more_data:
            break
        # More data to plot
        iteration_count += 1
        
        if False:
            # Plot for RC vs sphere size for different recons
            # Extract the top N pixel values where roi_masks is True
            masked_values = image_stack[roi_masks_array]
            top_N_values = np.partition(masked_values, -N)[-N:]
            mean_top_N = np.mean(top_N_values)
            results.append(mean_top_N)
        
            # Update plot
            plt.figure('SUV${_N}$ Plot')
            plt.plot(sphere_sizes[:len(results)], results, marker='o', label=f'Iteration: {iteration_count + 1}')
            plt.xlabel('Sphere Size [mm]')
            plt.ylabel('SUV${_N}$ [Bq/mL]')
            plt.title('SUV${_N}$ vs Sphere Size')
            plt.legend()
            plt.grid(True)
        
        # Ask user to load more data or not
        if False:
            answer = messagebox.askyesno("Load More Data", "Do you want to load more data?")
            if not answer:
                parent_directory = os.path.dirname(loaded_folder_path)
                png_path = os.path.join(parent_directory, 'SUV_N_plot.png')
                pickle_path = os.path.join(parent_directory, 'SUV_N_plot.pickle')
                plt.savefig(png_path)
                with open(pickle_path, 'wb') as f:
                    pickle.dump(plt.gcf(), f)
                plt.show()
                break
        
        

def plot_SUV_N(sphere_sizes, results, suv_peak_values):
    global SUV_max_values, loaded_folder_path, iteration_count

    # Takes the first 6 values (i.e. the SUV_peak of the 6 spheres)
    suv_peak_values = [details['max_mean'] for details in suv_peak_values.values()][:6] 
    
    # Add SUV_max_values to the beginning of the results for each sphere size
    for i, sphere_size in enumerate(sphere_sizes):
        results[sphere_size].insert(0, SUV_max_values[i])

    # Add suv_peak_values to the results for each sphere size
    for i, sphere_size in enumerate(sphere_sizes):
        results[sphere_size].append(suv_peak_values[i])
    
    # Normalize the SUV_max/SUV_N/SUV_peak values with formula (1) provided in https://doi.org/10.1007/s11604-021-01112-w
    phantom_weight =  12.6 # measured the water-filled NEMQ IQ phantom in kg (+-0.1 kg) (NEMA NU 2-2007)
    #injected_activity = 2729200 # measured the injected activity in Bq
    injected_activity = 1267756 # true activity conc at scan start * 48.45 mL (total volume of all spheres)
    activty_conc_at_scan_start = 28136.08 # calculated the activity with the measured injected_activity and the decay constant of F-18 (in Bq)
    activity_conc_at_scan_end = 25593.21
    # take the true activtiy concentration as the average of the activity concentration at the start and end of the scan
    # reason: can't decay-correct as usual since it is a static image and not a dynamic one
    #true_activity_conc = ((activty_conc_at_scan_start - activity_conc_at_scan_end) / 2) + activity_conc_at_scan_end
    true_activity_concentration = 26166.28 #Calculated the theoretical activity at scan start [kBq/mL] (Daniel, 05. Nov. 2024 11:36 am)
        
    # measured activity concentration in Bq/mL / (injected activity in Bq / phantom weight in kg)    
    #for sphere_size in sphere_sizes:
    #    results[sphere_size] = [value / (injected_activity * phantom_weight) for value in results[sphere_size]]
    
    # Normalize the SUV_max/SUV_N/SUV_peak values with formula (1) and (2) provided in https://doi.org/10.1007/s11604-021-01112-w
    SUV_ref = true_activity_concentration #Try not normalizing SUV_ref because SUV_N has also the same normalization and they just cancel each other out / (injected_activity * phantom_weight) # true activity concentration in Bq/mL / (injected activity in Bq / phantom weight in kg)
    for sphere_size in sphere_sizes:
        results[sphere_size] = [((value - SUV_ref) / SUV_ref) * 100 for value in results[sphere_size]]

    # Calculate and print the absolute sum of the results for each sphere size
    for sphere_size in sphere_sizes:
        abs_sum = sum(abs(value) for value in results[sphere_size])
        print(f"Absolute sum of results for sphere size {sphere_size} mm: {abs_sum:.2f}")
    
    # Calculate and print the absolute sum of the results for each index across all sphere sizes
    num_values = len(results[sphere_sizes[0]])  # Assuming all sphere sizes have the same number of values
    abs_sums = []
    for idx in range(num_values):
        abs_sum = sum(abs(results[sphere_size][idx]) for sphere_size in sphere_sizes)
        abs_sums.append(abs_sum)
        print(f"Absolute sum of results for index {idx + 1}: {abs_sum:.2f}")



    # Define x-axis labels
    x_labels = [r'SUV$_{max}$'] + [f'SUV$_{{{N}}}$' for N in range(5, 45, 5)] + [r'SUV$_{peak}$']
   
    # Plot the SUV_peak against the sphere size
    plt.figure('SUV$_{N}$ Plot')
    for sphere_size in sphere_sizes:
        plt.plot(range(0, 50, 5), results[sphere_size], marker='o', label=f'Sphere size: {sphere_size} mm')
    plt.xlabel('Mode of SUV')
    plt.ylabel(r'$\Delta$SUV [%]')
    plt.title('Different Modes of SUV')
    plt.xticks(range(0, 50, 5), x_labels)  # Set x-ticks to the defined labels
    plt.legend()
    plt.grid(True)
    parent_directory = os.path.dirname(loaded_folder_path)
    png_path = os.path.join(parent_directory, 'SUV_N_plot.png')
    pickle_path = os.path.join(parent_directory, 'SUV_N_plot.pickle')
    plt.savefig(png_path)
    with open(pickle_path, 'wb') as f:
        pickle.dump(plt.gcf(), f)
    plt.show()
    #plt.show()
    # Ask user to load more data or not
    answer = messagebox.askyesno("Load More Data", "Do you want to load more data?")
    if answer:
        # Increment the iteration counter for the legend of the plot
        iteration_count += 1
        # Add the current iteration count to the legend entries
        legend_entries.append(f'{iteration_count} iterations')

        # Plot the abs_sum values against the x_labels
        plt.figure('Summed Absolute Error Plot')
        plt.plot(range(num_values), abs_sums, marker='o')
        plt.xlabel('Mode of SUV')
        plt.ylabel(r'Summed Absolute $\Delta$SUV [%]')
        plt.title('SUV Mode Dependent Error')
        plt.xticks(range(num_values), x_labels)  # Set x-ticks to the defined labels
        plt.ylim(90, 180)
        plt.grid(True)

        # Add legend with the iteration counter in the title and entries
        plt.legend(legend_entries, title=f'Number of iterations: ')


        png_path = os.path.join(parent_directory, 'SUV_mode_dependent_error.png')
        pickle_path = os.path.join(parent_directory, 'SUV_mode_dependent_error.pickle')
        plt.savefig(png_path)
        with open(pickle_path, 'wb') as f:
            pickle.dump(plt.gcf(), f)
        plt.show()

        return False
    else:

        return True
            

def suv_peak_with_spherical_voi():
    global current_index, dicom_images, roi_masks, loaded_folder_path

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
    def process_index(index):
        z, y, x = index
        if can_place_sphere(index, radius_pixels, roi_masks_array):
            mask = create_3d_spherical_mask(index, radius_pixels, image_stack.shape)
            mean_value = get_mean_value(image_stack, mask)
            return z, mean_value, (y, x)
        return None

    # Use ThreadPoolExecutor to parallelize the processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_index, index) for index in np.argwhere(roi_masks_array)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                z, mean_value, position = result
                if mean_value > max_values_per_slice[z]['max_mean']:
                    max_values_per_slice[z]['max_mean'] = mean_value
                    max_values_per_slice[z]['position'] = position

    if False:
        # Loop through each index where roi_masks is True
        for z, y, x in np.argwhere(roi_masks_array):
            #if 1 <= z <= 6:  # Only consider z values from 1 to 6 (the number of spheres)
                index = (z, y, x)
                if can_place_sphere(index, radius_pixels):
                    mask = create_3d_spherical_mask(index, radius_pixels, image_stack.shape)
                    mean_value = get_mean_value(image_stack, mask)
                    if mean_value > max_values_per_slice[z]['max_mean']:
                        max_values_per_slice[z]['max_mean'] = mean_value
                        max_values_per_slice[z]['position'] = (y, x)  # Store the y, x position for the max value

    # Print results for each slice
    for z, details in max_values_per_slice.items():
        print(f"SUV_peak in sphere {z}: {details['max_mean']} at position {details['position']}")

    # Plot the SUV_peak against the sphere size
    suv_peak_values = [details['max_mean'] for details in max_values_per_slice.values()][:6] # takes the first 6 values (i.e. the SUV_peak of the 6 spheres)
    sphere_sizes = [10, 13, 17, 22, 28, 37]
    
    plot_suv_peak_against_sphere_size(suv_peak_values, sphere_sizes)
    return max_values_per_slice

def plot_suv_peak_against_sphere_size(suv_peak_values, sphere_sizes):
    global current_index, loaded_folder_path
    plt.figure(figsize=(8, 6))  # Adjust plot size to make it more readable
    plt.plot(sphere_sizes, suv_peak_values, marker='o')
    plt.xlabel('Sphere Size [mm]')
    plt.ylabel('SUV$_{peak}$ [Bq/mL]')
    plt.title('SUV$_{peak}$ vs Sphere Size')
    plt.xlim(7, 40)  # Set a reasonable range for x-axis
    plt.ylim(0, 30000) #max(suv_peak_values) * 1.1)  # Set a reasonable range for y-axis
    plt.xticks(sphere_sizes)  # Set x-ticks to the exact sphere sizes
    plt.grid(True)  # Add grid to the plot
    plt.tight_layout()


    # Annotate each marker with its y-value
    for x, y in zip(sphere_sizes, suv_peak_values):
        plt.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    # Get the parent directory of loaded_folder_path
    parent_directory = os.path.dirname(loaded_folder_path)

    # Save the plot as PNG and pickle
    png_path = os.path.join(parent_directory, f'SUV_peak_against_sphere_size_algorithm_started_at_slice_{current_index}.png')
    pickle_path = os.path.join(parent_directory, f'SUV_peak_against_sphere_size_algorithm_started_at_slice_{current_index}.pickle')

    plt.savefig(png_path)
    with open(pickle_path, 'wb') as f:
        pickle.dump(plt.gcf(), f)

    plt.show()
    

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


def create_2d_spherical_mask(center, radius_pixels, shape):
    """
    Create a 3D spherical mask.
    center: (z_center, y_center, x_center) - Center of the sphere in the 3D image.
    radius_pixels: Radius of the sphere in pixels.
    num_slices: Number of slices the sphere covers.
    shape: Shape of the 3D image stack.
    """
    global current_index
    if len(center) == 2:
        y_center, x_center = center
    elif len(center) == 3:
        z_center, y_center, x_center = center
    else:
        raise ValueError("Center should be a 2D or 3D coordinate")
    
    #z_center += current_index  # Add current_index to z_center
    depth, height, width = shape
    print(f"Shape of the spherical mask: {shape}")
    print(f"Center of the sphere: z: {current_index}, y: {y_center}, x: {x_center}")
    print(f"Radius of the sphere: {radius_pixels}")
    
    y, x = np.ogrid[:height, :width]
    distance = np.sqrt((y - y_center)**2 + (x - x_center)**2)
    #print(f"Distance of the sphere: {distance}")
    mask = distance <= radius_pixels
    return mask

def create_3d_spherical_mask(center, radius_pixels, shape):
    """
    Create a 3D spherical mask.
    center: (z_center, y_center, x_center) - Center of the sphere in the 3D image.
    radius_pixels: Radius of the sphere in pixels.
    num_slices: Number of slices the sphere covers.
    shape: Shape of the 3D image stack.
    """
    global current_index
    z_center, y_center, x_center = center
    #z_center += current_index  # Add current_index to z_center
    depth, height, width = shape
    print(f"Shape of the spherical mask: {shape}")
    print(f"Center of the sphere: z: {z_center}, y: {y_center}, x: {x_center}")
    print(f"Radius of the sphere: {radius_pixels}")
    
    z, y, x = np.ogrid[:depth, :height, :width]
    distance = np.sqrt((z - z_center)**2 + (y - y_center)**2 + (x - x_center)**2)
    #print(f"Distance of the sphere: {distance}")
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

def create_isocontour_roi(img_array, center, radius, threshold):
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

def create_isocontour_voi_3d(img_array, center, radius, threshold):
    """ Creates a binary mask for the isocontour ROI based on the threshold in a 3D array. """
    z_center, y_center, x_center = center
    mask = np.zeros_like(img_array, dtype=bool)
    # Slices (z coord.)
    for z in range(max(0, int(z_center - radius)), min(img_array.shape[0], int(z_center + radius) + 1)):
        # Rows (y coord.)
        for y in range(max(0, int(y_center - radius)), min(img_array.shape[1], int(y_center + radius) + 1)):
            # Columns (x coord.)
            for x in range(max(0, int(x_center - radius)), min(img_array.shape[2], int(x_center + radius) + 1)):
                if (z - z_center) ** 2 + (y - y_center) ** 2 + (x - x_center) ** 2 <= radius ** 2:
                    if img_array[z, y, x] >= threshold:
                        mask[z, y, x] = True
    return mask


def process_rois_for_predefined_centers(roi_or_voi = 'roi'):
    global roi_masks, current_index, SUV_max_values, dicom_images
    image_stack = build_image_stack()
    shape = image_stack.shape
    selected_slice = image_stack[current_index]
    print(f"Selected slice: {selected_slice}")
    print(f"Maximum of selected slice: {np.max(selected_slice)}")
    print(f"Shape of selected slice: {selected_slice.shape}")
    # Centers of 6 2D spheres with a 344x344 image size, increasing sphere sizes
    # centers = [(200, 165), (189, 190), (160, 194), (144, 171), (154, 146), (183, 142)] 
    if roi_or_voi == 'roi':
        # Centers of 6 2D spheres with a 512x512 image size, increasing sphere sizes
        centers = [(current_index, 212, 273), (current_index, 218, 230), (current_index, 257, 214), (current_index, 290, 240), (current_index, 284, 281), (current_index, 245, 298)]
    else:
        # Centers of 6 3D spheres with a 512x512 image size, increasing sphere sizes
        centers = [(current_index, 212, 273), (current_index, 218, 230), (current_index, 257, 214), (current_index, 290, 240), (current_index, 284, 281), (current_index, 245, 298)]
    radius = 15  # Covers even the biggest sphere with a diameter of 18.5 pixels (times approx. 2 mm pixel_spacing = 37 mm sphere)
    roi_masks = []
    # roi_pixels = []  # Initialize roi_pixels as an empty list
    sphere_sizes = [10, 13, 17, 22, 28, 37] # Sphere diameters in mm

    for i, center in enumerate(centers):
        # Assuming a threshold of 40% of the max value within each sphere's bounding box
        #local_max = np.max(selected_slice[
        #    max(0, center[0] - radius):min(selected_slice.shape[0], center[0] + radius),
        #    max(0, center[1] - radius):min(selected_slice.shape[1], center[1] + radius)
        #])
        #true_activity_concentration = 28136.08 #Calculated the theoretical activity at scan start (Daniel, 10. Oct. 2024 12:22 pm)
        true_activity_concentration = 26166.28 #Calculated the theoretical activity at scan start (Daniel, 05. Nov. 2024 11:36 am)
        # Calculate the radius in pixels (assuming isotropic pixels)
        

        threshold = 0.4 * true_activity_concentration #local_max
        if roi_or_voi == 'roi':
            #Following line commented out because isocontour threshold didn't perfectly delineate the sphere
            #roi_mask_temp = create_isocontour_roi(selected_slice, center, radius, threshold)
            radius_mm = sphere_sizes[i] / 2
            
            # Read in the pixel size of the DICOM image
            pixel_spacing = dicom_images[0][0x0028, 0x0030].value
            radius_pixels = radius_mm / pixel_spacing[0]
            roi_mask_temp = create_2d_spherical_mask(center, radius_pixels, shape)
        else:
            #Following line commented out because isocontour threshold didn't perfectly delineate the sphere
            #roi_mask_temp = create_isocontour_voi_3d(image_stack, center, radius, threshold)
            
            radius_mm = sphere_sizes[i] / 2
            
            # Read in the pixel size of the DICOM image
            pixel_spacing = dicom_images[0][0x0028, 0x0030].value
            radius_pixels = radius_mm / pixel_spacing[0]
            roi_mask_temp = create_3d_spherical_mask(center, radius_pixels, shape)
        print(f"VOI {len(roi_masks) + 1} - Threshold: {threshold:.2f}, Max Value: {true_activity_concentration:.2f}, Number of Pixels: {np.sum(roi_mask_temp)}")
        roi_masks.append(roi_mask_temp)
    print(f"roi_masks: {roi_masks}")
        # Create circular ROI and extract coordinate pairs to see if the radius of the max value search and the ROIs in which the max value is searched is correct
        #rr, cc = np.ogrid[:selected_slice.shape[0], :selected_slice.shape[1]]
        #circle_mask = (rr - center[0])**2 + (cc - center[1])**2 <= radius**2
        #roi_coords = np.column_stack(np.where(circle_mask))
        #roi_pixels.append(roi_coords)
    display_dicom_image(canvas, ax)
    #display_dicom_image(selected_slice, canvas, ax, roi_pixels=roi_pixels)
  
    if False:
        for center in centers:
            # Assuming a threshold of 40% of the max value within each sphere's bounding box
            local_max = np.max(selected_slice[
                max(0, center[0] - radius):min(selected_slice.shape[0], center[0] + radius),
                max(0, center[1] - radius):min(selected_slice.shape[1], center[1] + radius)
            ])
            threshold = 0.4 * local_max
            voi_mask = create_isocontour_roi(selected_slice, center, radius, threshold)
            print(f"VOI {len(vois) + 1} - Threshold: {threshold:.2f}, Max Value: {local_max:.2f}, Number of Pixels: {np.sum(voi_mask)}")
            vois.append(voi_mask)
    
        display_dicom_image(selected_slice, canvas, ax, voi_masks=vois)
    # Initialize an array to store the mean values of the different VOIs
    mean_values = []
    if roi_or_voi == 'voi':
        SUV_max_values = [] # Reset the global SUV_max_values array if called by the VOI function (to avoid appending the SUV_max to the ones from the ROIs)
    # Calculate the mean values of the different ROIs/VOIs
    for i, roi_in_roi_masks in enumerate(roi_masks):
        if roi_in_roi_masks.ndim == 3:
            mean_value = np.mean(image_stack[roi_in_roi_masks])
            max_value = np.max(image_stack[roi_in_roi_masks])
        else:
            mean_value = np.mean(selected_slice[roi_in_roi_masks])
            max_value = np.max(selected_slice[roi_in_roi_masks])
        num_pixels = np.sum(roi_in_roi_masks)
        mean_values.append(mean_value)
        SUV_max_values.append(max_value)
        print(f"Mean value for VOI {i + 1}: {mean_value:.2f}")
        print(f"SUV_max value for VOI {i + 1}: {max_value:.2f}")
        print(f"Number of pixels in VOI {i + 1}: {num_pixels}")
 
    # Calculate the recovery coefficient of the different ROIs using the stored mean values
    print(f"True activity: {true_activity_concentration:.2f}")
    for i, mean_value in enumerate(mean_values):
        recovery_coefficient = mean_value / true_activity_concentration
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
    global iteration_count, recovery_coefficients, loaded_folder_path
    
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

    # Get the parent directory of loaded_folder_path
    parent_directory = os.path.dirname(loaded_folder_path)

    # Save the plot as PNG and pickle
    png_path = os.path.join(parent_directory, f'Recovery_coefficients_vs_sphere_size.png')
    pickle_path = os.path.join(parent_directory, f'Recovery_coefficients_vs_sphere_size.pickle')

    plt.savefig(png_path)
    with open(pickle_path, 'wb') as f:
        pickle.dump(plt.gcf(), f)

    plt.show()

    #plt.show()  # Show the plot and block interaction until closed
    #plt.close(fig)  # Ensure the figure is closed after displaying
    
    #plt.draw()  # Update the plot without blocking

def show_plot():
    global loaded_folder_path
    # Path to the pickle file
    parent_directory = os.path.dirname(loaded_folder_path)
    png_path = os.path.join(parent_directory, 'plot.png')
    pickle_path = os.path.join(parent_directory, 'plot.pickle')
    plt.savefig(png_path)
    with open(pickle_path, 'wb') as f:
        pickle.dump(plt.gcf(), f)
    plt.show()

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

    # Get Subtraction Image Button
    get_subtraction_image_button = tk.Button(root, text="Get Subtraction Image", command=subtract_images)
    get_subtraction_image_button.pack(side=tk.LEFT, padx=20, pady=10)

    # Draw Plot Button
    draw_plot_button = tk.Button(root, text="Draw Plot", command=draw_plot)
    draw_plot_button.pack(side=tk.LEFT, padx=25, pady=10)

    # Show Plot Button
    show_plot_button = tk.Button(root, text="Show Plot", command=show_plot)
    show_plot_button.pack(side=tk.LEFT, padx=25, pady=10)

    # Plot SNR Button
    plot_snr_button = tk.Button(root, text="Plot SNR", command=plot_snr_values)
    plot_snr_button.pack(side=tk.LEFT, padx=25, pady=10)

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

    # Add "Calculate SUV_N" button
    suv_n_button = tk.Button(root, text="Calculate SUV_N", command=lambda: calculate_SUV_N())
    suv_n_button.pack(side=tk.TOP, padx=10, pady=10)

    # ROI Input panel for 12 ROIs arranged in a 3x4 grid
    roi_panel = tk.Frame(root)
    roi_panel.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
    
    # Initialize the list to store the ROI entries
    roi_entries = []

    # Default values for 512x512 image size
    default_values = [
        {'x': 258, 'y': 191}, {'x': 285, 'y': 194}, {'x': 308, 'y': 210}, {'x': 338, 'y': 255},
        {'x': 336, 'y': 285}, {'x': 315, 'y': 302}, {'x': 265, 'y': 313}, {'x': 205, 'y': 308},
        {'x': 195, 'y': 228}, {'x': 187, 'y': 282}, {'x': 178, 'y': 255}, {'x': 209, 'y': 205}
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
