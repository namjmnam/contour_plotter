import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from svg.path import parse_path
from xml.dom import minidom
import tkinter as tk
from tkinter import simpledialog

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

    path_data = []  # List to store path data and their Z values

    def on_pick(event):
        # This function will be called when a path point is clicked
        path_index = event.artist.get_gid()
        z = simpledialog.askfloat("Input", f"Enter Z value for path {path_index}:", parent=tk_root)
        if z is not None:  # Check if the user entered a value
            path_data[path_index]['z'] = z

    for index, path_string in enumerate(paths):
        path = parse_path(path_string)
        x_values, y_values = [], []

        for segment in path:
            if hasattr(segment, 'start') and hasattr(segment, 'end'):
                x_values.extend([segment.start.real, segment.end.real])
                y_values.extend([segment.start.imag, segment.end.imag])

        # Plot the path
        ax.plot(x_values, y_values, 'b')

        # Add a clickable point for the path
        clickable_point, = ax.plot(x_values[0], y_values[0], 'ro', picker=5, markersize=8, gid=index)
        clickable_point.set_picker(5)  # 5 points tolerance

        # Store the path data
        path_data.append({'x': x_values, 'y': y_values, 'z': 0})

    # Connect the pick event to the on_pick function
    fig.canvas.mpl_connect('pick_event', on_pick)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    return path_data

# Function to plot paths in 3D
def plot_3d_paths(path_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for path in path_data:
        # Each path's Z value is constant across all its points
        z_values = [path['z']] * len(path['x'])
        ax.plot(path['x'], path['y'], z_values)

    plt.show()

# SVG file path
svg_file = './p5oil.svg'

# Extract and plot paths interactively
path_data = plot_interactive_paths(extract_svg_paths(svg_file))

# Plot the paths in 3D
plot_3d_paths(path_data)
