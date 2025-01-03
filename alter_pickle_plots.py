import pickle
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox

def open_pickle_plot(pickle_path):
    # Load the plot from the pickle file
    with open(pickle_path, 'rb') as f:
        fig = pickle.load(f)
    
    # Display the plot
    plt.show()

def alter_pickle_plot(pickle_path, output_pickle_path):
    # Load the plot from the pickle file
    with open(pickle_path, 'rb') as f:
        fig = pickle.load(f)
    
    # Get the current axes
    #ax = fig.gca()

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
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(fig, f)
    
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