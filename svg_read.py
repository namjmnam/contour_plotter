import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from svg.path import parse_path
from xml.dom import minidom
import tkinter as tk
from tkinter import simpledialog
import numpy as np
from scipy.interpolate import griddata
import random

# Initialize Tkinter root - needed for dialog
tk_root = tk.Tk()
tk_root.withdraw()  # We don't need a full GUI, so keep the root window from appearing

# Function to extract paths from the SVG file
def extract_svg_paths(svg_file):
    doc = minidom.parse(svg_file)  # Parse the SVG file
    path_strings = [path.getAttribute('d') for path in doc.getElementsByTagName('path')]
    doc.unlink()
    return path_strings

# Function to plot paths and add interactive points
def plot_interactive_paths(paths):
    fig, ax = plt.subplots()

    path_data = []
    text_labels = []  # List to store text labels

    def on_pick(event):
        artist = event.artist
        path_index = artist.get_gid()
        z = simpledialog.askfloat("Input", f"Enter Z value for path {path_index}:", parent=tk_root)
        if z is not None:
            path_data[path_index]['z'] = z
            artist.set_color('blue')  # Change color to blue
            x, y = artist.get_data()
            # Create or update the text label
            if len(text_labels) > path_index:
                text_labels[path_index].set_text(f'{z:.2f}')
                text_labels[path_index].set_visible(True)
            else:
                label = ax.text(x[0], y[0], f'{z:.2f}', color='blue')
                text_labels.append(label)
                fig.canvas.draw_idle()  # Update the figure

    for index, path_string in enumerate(paths):
        path = parse_path(path_string)
        path_points = []

        for segment in path:
            # Sampling points from each segment
            segment_points = np.array([segment.point(t) for t in np.linspace(0, 1, 50)])
            path_points.extend(segment_points)

        # Assign a random Z value to each path
        random_z = random.uniform(1, 20)  # You can adjust the range as needed

        x_values = [p.real for p in path_points]
        y_values = [p.imag for p in path_points]

        ax.plot(x_values, y_values, 'b')

        clickable_point, = ax.plot(x_values[0], y_values[0], 'ro', picker=5, markersize=8, gid=index)
        clickable_point.set_picker(5)

        path_data.append({'x': x_values, 'y': y_values, 'z': random_z})

    fig.canvas.mpl_connect('pick_event', on_pick)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    return path_data

def show_3d_plot_and_save_obj(path_data, max_z_value=100, scale_factor=1.0, grid_density=100, figsize=(12, 9), cmap='viridis', alpha=0.6, filename='output.obj'):
    fig_3d = plt.figure(figsize=figsize)
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    # Apply scaling factor to all x and y coordinates from the paths
    all_x = np.hstack([np.array(path['x']) * scale_factor for path in path_data])
    all_y = np.hstack([np.array(path['y']) * scale_factor for path in path_data])

    # Determine the range for the grid
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    # Testing a small portion
    # x_min, x_max = 700, 900
    # y_min, y_max = 700, 900

    # Create a grid for the surface plot
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, grid_density),
        np.linspace(y_min, y_max, grid_density)
    )

    # Flatten the grid for griddata input
    grid_x_flat = grid_x.flatten()
    grid_y_flat = grid_y.flatten()

    # Prepare scaled contour data for interpolation
    points = np.vstack([all_x, all_y]).T
    values = np.hstack([np.full(len(path['x']), path['z']) for path in path_data])

    # Interpolate z values on the grid using 'nearest' to avoid NaNs
    grid_z = griddata(points, values, (grid_x_flat, grid_y_flat), method='nearest')

    # Reshape the z values back into a grid
    grid_z = grid_z.reshape(grid_x.shape)

    # Clip z values to enforce the maximum height constraint
    grid_z = np.clip(grid_z, None, max_z_value)

    # Create a surface plot
    surf = ax_3d.plot_surface(
        grid_x, grid_y, grid_z, 
        cmap=cmap, alpha=alpha, 
        linewidth=0, antialiased=True
    )

    # Add a color bar which maps values to colors.
    fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=5, label='Height')

    # Set axes limits to cover the full range of x and y values
    ax_3d.set_xlim(x_min, x_max)
    ax_3d.set_ylim(y_min, y_max)
    ax_3d.set_zlim(0, max_z_value)

    plt.show()
    # After creating the surface plot, save the grid to an OBJ file
    save_grid_to_obj(grid_x, grid_y, grid_z, filename)

def save_grid_to_obj(grid_x, grid_y, grid_z, filename):
    with open(filename, 'w') as file:
        # Write vertices
        for i in range(len(grid_x)):
            for j in range(len(grid_x[0])):
                file.write(f"v {grid_x[i][j]} {grid_y[i][j]} {grid_z[i][j]}\n")

        # Write faces (as quads)
        for i in range(len(grid_x) - 1):
            for j in range(len(grid_x[0]) - 1):
                # OBJ files are 1-indexed
                v1 = i * len(grid_x[0]) + j + 1
                v2 = v1 + 1
                v3 = v1 + len(grid_x[0]) + 1
                v4 = v3 - 1
                file.write(f"f {v1} {v2} {v3} {v4}\n")

# SVG file path
svg_file = './p5oil.svg'

# Extract and plot paths interactively
path_data = plot_interactive_paths(extract_svg_paths(svg_file))
show_3d_plot_and_save_obj(path_data, grid_density=2000)


# dummy_path_data = [
#     {'x': [0, 1, 2], 'y': [0, 1, 2], 'z': 50},
#     {'x': [2, 3, 4], 'y': [2, 3, 4], 'z': 75}
# ]
# show_3d_plot(dummy_path_data, grid_density=10000)
