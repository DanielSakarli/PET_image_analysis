import pickle
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

def open_pickle_plot(pickle_path):
    with open(pickle_path, 'rb') as f:
        old_fig = pickle.load(f)

    old_ax = old_fig.axes[0]  # Grab the old Axes

    # Extract data from the old lines
    #line_data = []
    for i, line in enumerate(old_ax.lines):
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        print(f"Line {i+1}: x = {x_data}, y = {y_data}")
        #line_data.append((x_data, y_data))

    # Display the plot
    plt.show()

def alter_pickle_plot(pickle_path, output_pickle_path):
    # Load the old figure
    with open(pickle_path, 'rb') as f:
        old_fig = pickle.load(f)

    old_ax = old_fig.axes[0]  # Grab the old Axes

    # Extract data from the old lines
    line_data = []
    for i, line in enumerate(old_ax.lines):
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        print(f"Line {i+1}: x = {x_data}, y = {y_data}")
        line_data.append((x_data, y_data))

    # Create a new figure/axes
    new_fig, new_ax = plt.subplots()

    # Re-plot the lines onto the new axes
    for i, (x_data, y_data) in enumerate(line_data):
        new_ax.plot(x_data, y_data, marker='o', label=f"{i+1}i")

    # Now you can safely change the title, labels, etc.
    new_ax.set_title("Recovery Coefficients within a 10 mm Hot Sphere calculated with c$_{mean}$", fontsize=20)
    new_ax.set_xlabel("Cylindrical VOI Diameter [mm]", fontsize=16)
    new_ax.set_ylabel("Recovery Coefficient [%]", fontsize=16)
    new_ax.legend(title="Number of\niterations i:")
    new_ax.set_ylim(0, 100)
    new_ax.set_xlim(0, 40)
    new_ax.grid(True)
    # Increase fontsize of xticks and yticks
    new_ax.tick_params(axis='both', which='major', labelsize=14)

    # Save the new figure
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(new_fig, f)

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