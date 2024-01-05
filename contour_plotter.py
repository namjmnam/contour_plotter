import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tkinter as tk
from tkinter.simpledialog import askstring
from matplotlib import cm

# Function to create multiple flat circles
def create_flat_circles(num_circles=5, radius_step=1, num_points=100):
    circles = []
    for i in range(num_circles):
        radius = radius_step * (i + 1)
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = 0  # Initial z value (height) is zero
        circles.append([x, y, z])  # x, y, z for each circle
    return circles

# # Function to parse CSV file and extract points
# def load_csv_file(file_path):
#     with open(file_path, newline='') as csvfile:
#         csv_reader = csv.reader(csvfile, delimiter=',')
#         # Extract x and y coordinates, assuming they are stored in the first two columns
#         x_coords, y_coords = zip(*((float(row[0]), float(row[1])) for row in csv_reader))
#     return list(x_coords), list(y_coords), [0.0] * len(x_coords)  # z-coordinates initialized to 0

# # Ask the user to upload a CSV file and load it
# def upload_and_load_csv():
#     file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")])
#     if file_path:
#         return load_csv_file(file_path)
#     else:
#         return None, None, None

def on_pick(event):
    circle_index = event.artist.get_gid()
    height = askstring("Input", f"Enter height for circle {circle_index + 1}:", parent=main_window)
    if height:
        try:
            circles[circle_index][2] = float(height)  # Update the circle's height
            # Update height text
            height_texts[circle_index].set_text(f'{float(height):.2f}')
            fig.canvas.draw_idle()
        except ValueError:
            print("Please enter a valid number for height.")

def show_3d_plot():
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # Create a grid for the surface plot
    max_radius = max([radius_step * (i + 1) for i in range(len(circles))])
    grid_x, grid_y = np.meshgrid(np.linspace(-max_radius, max_radius, 300), np.linspace(-max_radius, max_radius, 300))
    grid_z = np.zeros_like(grid_x)

    # Calculate the z values on the grid
    for x, y, z in circles:
        grid_z += np.where(np.sqrt(grid_x**2 + grid_y**2) <= x[0], z, 0)

    # Create a surface plot
    surf = ax_3d.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', alpha=0.8)
    fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=5, label='Height')

    plt.show()

circles = create_flat_circles()

# Create a main window for Tkinter
main_window = tk.Tk()
main_window.title("Circle Height Adjustment")

# Set the radius step as a variable accessible in the scope of the show_3d_plot function
radius_step = 1

# Interactive 2D plot
fig, ax = plt.subplots()
height_texts = []

for i, (x, y, z) in enumerate(circles):
    # Plot each circle
    circle_line, = ax.plot(x, y, '-', label=f'Circle {i+1}')
    # Add a pickable point with a unique gid for each circle
    pickable_point, = ax.plot(x[0], y[0], 'ro', markersize=10, picker=5, gid=i)
    # Add height text
    height_text = ax.text(x[0], y[0], f'{z:.2f}', fontsize=12, color='blue')
    height_texts.append(height_text)

ax.legend()

# Connect the pick event to the on_pick function
fig.canvas.mpl_connect('pick_event', on_pick)

# Button to show 3D plot
show_3d_button = tk.Button(main_window, text="Show 3D Plot", command=show_3d_plot)
show_3d_button.pack()

# Start the Tkinter main loop in a non-blocking way
main_window.after(0, main_window.mainloop)

# Display the 2D plot
plt.show()


