import pickle
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def open_pickle_plot(pickle_path):
    with open(pickle_path, 'rb') as f:
        old_fig = pickle.load(f)

    old_ax = old_fig.axes[0]  # Grab the old Axes

    # Extract data from the old lines
    #line_data = []
    for i, line in enumerate(old_ax.lines):
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        
        y_max = max(y_data)
        print(f"Line {i+1}: y_max = {y_max}")
        #line_data.append((x_data, y_data))

    # Display the plot
    plt.show()

def alter_pickle_plot(pickle_path, output_pickle_path):
    # Load the old figure
    with open(pickle_path, 'rb') as f:
        old_fig = pickle.load(f)

    old_ax = old_fig.axes[0]  # Grab the old Axes
    if False:
        legend_entries = ["(1i, 3mm)", "(2i, 3mm)", "(3i, 3mm)", "(4i, 3mm)", "(5i, 3mm)", "(6i, 3mm)", "(7i, 3mm)", "(8i, 3mm)"]
        #legend_entries = ["(4i, 3mm)", "(4i, 5mm)", "(4i, 7mm)"]
        
        # Extract data from all lines on the old Axes
        lines_data = []
        for line in old_ax.lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            lines_data.append((x_data, y_data))

        # Create a new figure and Axes
        new_fig, old_ax = plt.subplots()
        old_ax.set_xlabel("Spherical VOI Diameter [mm]", fontsize=12)
        old_ax.set_ylabel("Standard Deviation [Bq/mL]", fontsize=12)
        
        
        old_ax.set_ylim(0, 1000)
        old_ax.set_xlim(0, 40)
        old_ax.grid(True)

        # Re-plot the data on the new Axes
        for i, (x_data, y_data) in enumerate(lines_data):
            old_ax.plot(x_data, y_data, marker='o', label=legend_entries[i])
        old_ax.legend(title="Recon-ID:", fontsize=10)
    old_ax.set_title(r"Quantification Type Dependent Error", fontsize=14)
    old_ax.set_xlabel("Type of Quantification", fontsize=12)
    old_ax.set_ylabel(r"Summed Absolute Error [%]", fontsize=12)
    legend_entries = ["(1i, 3mm)", "(2i, 3mm)", "(3i, 3mm)", "(4i, 3mm)", "(5i, 3mm)", "(6i, 3mm)", "(7i, 3mm)", "(8i, 3mm)"]
    old_ax.legend(legend_entries, title="Recon-ID:", fontsize=10)
    if False:  
        # Anzahl der vorhandenen xticks
        xticks = old_ax.get_xticks()
        new_xticklabels = [r'$c_{max}$', r'$c_{5}$', r'$c_{10}$', r'$c_{15}$', r'$c_{20}$', r'$c_{25}$', r'$c_{30}$', r'$c_{35}$', r'$c_{40}$', r'$c_{peak}$', r'$c_{mean}$']
        
        # Prüfen, ob die Anzahl der neuen Labels mit den vorhandenen xticks übereinstimmt
        if len(xticks) != len(new_xticklabels):
            print("Fehler: Anzahl der neuen xticklabels stimmt nicht mit den vorhandenen xticks überein.")
            return

        # Setze die neuen xticks und Labels
        old_ax.set_xticklabels(new_xticklabels)

        # Extract data from the old lines
        line_data = []
        # Swap x locations of c_mean and c_peak
        for line in old_ax.lines:
            # Get current x and y data as lists
            x_data = list(line.get_xdata())
            y_data = list(line.get_ydata())

            # Ensure we have at least two points to swap (assumed to be c_mean and c_peak)
            if len(x_data) < 2:
                continue

            # Original assumption:
            #   c_mean is the second last point and c_peak is the last point.
            #
            # To "switch the x location" means that:
            #   - c_mean should take the x coordinate of c_peak but keep its original y value.
            #   - c_peak should take the x coordinate of c_mean but keep its original y value.
            #
            # Thus, define new pairs:
            new_c_mean = (x_data[-1], y_data[-2])  # Use x from c_peak, y from c_mean
            new_c_peak = (x_data[-2], y_data[-1])  # Use x from c_mean, y from c_peak

            # Replace the last two points with the new values:
            x_data[-2] = new_c_mean[0]
            y_data[-2] = new_c_mean[1]
            x_data[-1] = new_c_peak[0]
            y_data[-1] = new_c_peak[1]

            # Now, although the dots (markers) will appear at the correct x positions,
            # the line connecting them is drawn in the order stored.
            # To have the connecting line go from left to right, we reorder the last two points
            # if the first of the two ends up to the right of the second.
            if x_data[-2] > x_data[-1]:
                # Swap the entire points (both x and y) so that the left-most point comes first.
                x_data[-2], x_data[-1] = x_data[-1], x_data[-2]
                y_data[-2], y_data[-1] = y_data[-1], y_data[-2]

            # Set the modified data back to the line
            line.set_xdata(x_data)
            line.set_ydata(y_data)
    # Get all axes in the figure
    #axes = fig.get_axes()
    if False:
        # Determine number of rows and columns
        num_rows = 3  # X, Y, Z profiles
        num_cols = 6  # Sphere sizes (10 mm to 37 mm)

        for i, ax in enumerate(old_ax):
            row = i // num_cols
            col = i % num_cols
            
            # Remove x-labels for all rows except the last row
            if row < num_rows - 1:  
                ax.set_xlabel('')
            if row == 2:
                ax.set_xlabel('Distance [mm]', fontsize=12)
            # Remove y-labels for all columns except the first column
            if col > 0:  
                ax.set_ylabel('')
            if col == 0:
                ax.set_ylabel('Signal Intensity [Bq/mL]', fontsize=12)
    
    if False:
        for i, line in enumerate(old_ax.lines):
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            print(f"Line {i+1}: x = {x_data}, y = {y_data}")
            line_data.append((x_data, y_data))

        # Create a new figure/axes
        new_fig, new_ax = plt.subplots()
        labels=['IDIF', 'AIF', 'RC Corrected IDIF']

        # Re-plot the lines onto the main axes
        for i, (x_data, y_data) in enumerate(line_data):
            new_ax.plot(x_data, y_data, marker='o', label=labels[i])

        # Main plot formatting
        new_ax.set_title("PSMA007 Recovery Corrected IDIF", fontsize=20)
        new_ax.set_xlabel("Time [minutes]", fontsize=16)
        new_ax.set_ylabel(r"SUV [(kg*mL)$^{-1}$]", fontsize=16)
        new_ax.legend(loc='lower right', bbox_to_anchor=(0.95, 0.08), fontsize=14) #title="Number of\niterations i:", 
        new_ax.set_ylim(0, 80)
        new_ax.set_xlim(-5, 65)
        new_ax.grid(True)
        new_ax.tick_params(axis='both', which='major', labelsize=14)

        # Create an inset Axes for the zoomed region
        ax_inset = inset_axes(
            new_ax,                # parent axes
            width="65%",           # inset width (as a percentage of parent)
            height="65%",          # inset height
            loc='upper right'      # choose a corner of the parent Axes
        )

        # Re-plot the same data on the inset
        for i, (x_data, y_data) in enumerate(line_data):
            ax_inset.plot(x_data, y_data, marker='o')

        # Set the x- and y-range
        ax_inset.set_xlim(0, 4)
        ax_inset.set_ylim(0, 75)  # Adjust to suit your data’s scale
        ax_inset.set_xticks([0, 1, 2, 3, 4])
        ax_inset.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
        ax_inset.grid(True)
        ax_inset.tick_params(axis='both', which='major', labelsize=14)

    # Save the new figure to a pickle, plus PDF/PNG
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(old_fig, f)

    base_filename = output_pickle_path.rsplit('.', 1)[0]
    pdf_path = f"{base_filename}.pdf"
    png_path = f"{base_filename}.png"
    plt.savefig(pdf_path)
    plt.savefig(png_path)
    plt.show()
    # Extract existing plot data
    #line = ax.lines[0]
    #ax.set_ylim(1.5, 8.5)

    
    # Include a legend with a title
    #ax.legend(title="Number of\niterations i:", loc='lower right', labels=["1i", "2i", "3i", "4i", "5i", "6i", "7i", "8i"])

     
     
    if False:
        x_data, y_data = line.get_xdata(), line.get_ydata()

        # Interpolate the data to create a smooth curve
        interp_func = interp1d(x_data, y_data, kind='cubic', fill_value="extrapolate")
        x_interp = np.linspace(3, max(x_data), 500)  # Extend interpolation to start at 3 mm
        y_interp = interp_func(x_interp)

        # Save the interpolated x and y values to a CSV file
        csv_output_path = "C://Users//DANIE//OneDrive//FAU//Master Thesis//Project//Data//RC Correction//interpolated_rc_curve_calculated_with_c_4_hottest_pixels_4i_with_background.csv"
        interpolated_data = pd.DataFrame({'x': x_interp, 'y': y_interp})
        interpolated_data.to_csv(csv_output_path, index=False)

        # Remove the original line and re-plot only the interpolated curve
        for line in ax.lines[:]:  # Clear all lines
            line.remove()
        ax.plot(x_interp, y_interp, label='Interpolated Curve', color='red')

        # Plot the original data points
        ax.scatter(x_data, y_data, color='blue', label='Original Data', zorder=5)

        # Define x' and y' and add vertical lines
        x_prime = 9
        y_prime = 12

        # Vertical line for x' = 8 mm
        if min(x_interp) <= x_prime <= max(x_interp):
            y_at_x_prime = interp_func(x_prime)
            ax.axvline(x=x_prime, color='red', linestyle='--')  # Red vertical line
            ax.scatter([x_prime], [y_at_x_prime], color='red')  # Mark the intersection
            ax.text(x_prime + 0.5, y_at_x_prime - 1.5, f'{y_at_x_prime:.1f}', color='red', fontsize=10)  # Annotate y-value

        # Vertical line for y' = 12 mm
        if min(x_interp) <= y_prime <= max(x_interp):
            y_at_y_prime = interp_func(y_prime)
            ax.axvline(x=y_prime, color='red', linestyle='--')  # Red vertical line
            ax.scatter([y_prime], [y_at_y_prime], color='red')  # Mark the intersection
            ax.text(y_prime + 0.5, y_at_y_prime - 1.5, f'{y_at_y_prime:.1f}', color='red', fontsize=10)  # Annotate y-value

        # Add x' and y' ticks on the x-axis
        original_xticks = list(ax.get_xticks())  # Get the original ticks
        new_xticks = sorted(original_xticks + [x_prime, y_prime])  # Add x' and y' to ticks
        ax.set_xticks(new_xticks)  # Set the updated ticks

        # Create tick labels, making x' and y' red
        xtick_labels = []
        for tick in new_xticks:
            if tick == x_prime:
                xtick_labels.append(f"x'")  # Label for x'
            elif tick == y_prime:
                xtick_labels.append(f"y'")  # Label for y'
            else:
                xtick_labels.append(f"{int(tick)}")  # Keep other ticks as integers

        # Set the tick labels with red color for x' and y'
        for label, tick in zip(ax.set_xticklabels(xtick_labels), new_xticks):
            if tick == x_prime or tick == y_prime:
                label.set_color('red')  # Make x' and y' labels red
            else:
                label.set_color('black')  # Keep other labels black

    # Limit the x-axis range
    #ax.set_xlim(5, 40)


    # Update legend and title
    #ax.legend()

    # Get the current axes
    #ax = fig.gca()
    if False:
        # Get all the axes in the figure
        axes = fig.axes
        # Print the number of axes and their positions
        print(f"Number of axes: {len(axes)}")
        for i, ax in enumerate(axes):
            pos = ax.get_position()
            print(f"Axes {i}: position = {pos}")
        
        # Define the indices of the subplots to be altered
        subplots_to_alter = [0, 6]  # (1st row, 1st column) and (2nd row, 1st column)
        
        # Alter the x-values to center the maximum y-value at x = 0 for the specified subplots
        for i, ax in enumerate(axes):
            if i in subplots_to_alter:
                for line in ax.get_lines():
                    x_data = line.get_xdata()
                    y_data = line.get_ydata()
                    
                    # Find the x-coordinate of the maximum y-value
                    max_y_index = y_data.argmax()
                    max_x_value = x_data[max_y_index]
                    
                    # Shift the x-data to center the maximum y-value at x = 0
                    new_x_data = x_data - max_x_value
                    line.set_xdata(new_x_data)
    
    


    # Alter all y-values of the plot
    if False:
        for line in ax.get_lines():
            y_data = line.get_ydata()
            new_y_data = y_data * 26166.28 / 6300.04

            line.set_ydata(new_y_data)
        
    # Change the ylabel to "Image Roughness [%]"
    #ax.set_ylabel("Image Roughness [%]")
    #ax.set_ylim(0, 7000) # Limit the y-axis
    # Save the modified plot back to a pickle file
    # Edit title
    #ax.set_title("Recovery Coefficients within a 10 mm Hot Sphere calculated with c$_{mean}$")
    if False:
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(fig, f)
        
        
        # Also save the figure as PDF and PNG
        base_filename = output_pickle_path.rsplit('.', 1)[0]  # Remove .pickle
        pdf_path = f"{base_filename}.pdf"
        png_path = f"{base_filename}.png"
        plt.savefig(pdf_path)
        plt.savefig(png_path)
        # Optionally, display the modified plot
        plt.show()

if __name__ == "__main__":
    # Hide the root window
    root = Tk()
    root.withdraw()
    
    # Open file dialog to select the pickle file
    pickle_path = filedialog.askopenfilename(title="Select Pickle Plot", filetypes=[("Pickle files", "*.pickle")])
    
    if pickle_path:
        # Ask the user if they want to open or edit the file
        choice = messagebox.askyesno("Open or Edit", "Do you want to edit the pickle plot? (Yes: edit the plot, No: only open the plot)")
        
        if choice:
            # User chose to edit the file
            output_pickle_path = filedialog.asksaveasfilename(title="Save Modified Plot As", defaultextension=".pickle", filetypes=[("Pickle files", "*.pickle")])
            if output_pickle_path:
                alter_pickle_plot(pickle_path, output_pickle_path)
        else:
            # User chose to open the file without editing
            open_pickle_plot(pickle_path)