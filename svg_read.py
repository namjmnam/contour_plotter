import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from svg.path import parse_path
from svg.path.path import CubicBezier
from svg.path.path import Line
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
    paths = doc.getElementsByTagName('path')
    path_data = [{'d': path.getAttribute('d'), 'class': path.getAttribute('class')} for path in paths]
    doc.unlink()
    return path_data

# Function to plot paths and add interactive points
def plot_interactive_paths(paths):
    fig, ax = plt.subplots()

    path_data = []
    outer_contours = []
    inner_contours = []
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
        path = parse_path(path_string['d'])
        path_points = []
        path_class = path_string['class']

        if path_class=='outer':
            outer_contours.append(path)
            continue

        if path_class=='inner':
            inner_contours.append(path)
            continue

        for segment in path:
            # Sampling points from each segment
            segment_points = np.array([segment.point(t) for t in np.linspace(0, 1, 50)])
            path_points.extend(segment_points)

        # Assign a random Z value to each path
        # random_z = random.uniform(1, 20)  # You can adjust the range as needed
        z_value = parse_z_from_class(path_class)  # Get z-value based on class name

        x_values = [p.real for p in path_points]
        y_values = [p.imag for p in path_points]

        ax.plot(x_values, y_values, 'b')

        clickable_point, = ax.plot(x_values[0], y_values[0], 'ro', picker=5, markersize=8, gid=index)
        clickable_point.set_picker(5)

        path_data.append({'x': x_values, 'y': y_values, 'z': z_value})

    fig.canvas.mpl_connect('pick_event', on_pick)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    return path_data, outer_contours, inner_contours

def show_3d_plot_and_save_obj(path_data, outer_contour, inner_contour, max_z_value=100, scale_factor=1.0, grid_density=100, figsize=(12, 9), cmap='viridis', alpha=0.6, filename='output.obj'):
    fig_3d = plt.figure(figsize=figsize)
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    floor = -350

    # Apply scaling factor to all x and y coordinates from the paths
    all_x = np.hstack([np.array(path['x']) * scale_factor for path in path_data])
    all_y = np.hstack([np.array(path['y']) * scale_factor for path in path_data])

    # Determine the range for the grid
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

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
    grid_z = grid_z.astype(float)  # Convert grid_z to float type

    # Reshape the z values back into a grid
    grid_z = grid_z.reshape(grid_x.shape)

    # Clip z values to enforce the maximum height constraint
    grid_z = np.clip(grid_z, None, max_z_value)

    # Set z values to 0 for points outside the "outer" contour or inside the "inner" contour
    for i in range(grid_z.shape[0]):
        for j in range(grid_z.shape[1]):
            if not is_point_inside_path(grid_x[i, j], grid_y[i, j], outer_contour) or is_point_inside_path(grid_x[i, j], grid_y[i, j], inner_contour):
                # grid_z[i, j] = floor
                grid_z[i, j] = np.nan

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
    ax_3d.set_zlim(floor, max_z_value)

    plt.show()
    # After creating the surface plot, save the grid to an OBJ file
    # save_grid_to_obj(grid_x, grid_y, grid_z, filename)

# Function to parse z-value from class name
def parse_z_from_class(class_name):
    if class_name.startswith("contour-"):
        try:
            return int(class_name.split('-')[1]) * (-1)
        except ValueError:
            return 0  # Default to 0 if parsing fails
    return 0

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

def is_point_inside_path(x, y, path_d, num_segments=10):
    """
    Determine if a point (x, y) is inside the given SVG path.

    :param x: X-coordinate of the point
    :param y: Y-coordinate of the point
    :param path_d: 'd' attribute of the SVG path
    :param num_segments: Number of segments for approximating the Bezier curves
    :return: True if the point is inside the path, False otherwise
    """
    def cubic_bezier_to_points(start, control1, control2, end):
        points = []
        for t in np.linspace(0, 1, num_segments):
            point = (1-t)**3 * start + 3*(1-t)**2 * t * control1 + 3*(1-t) * t**2 * control2 + t**3 * end
            points.append((point.real, point.imag))
        return points

    # path = parse_path(path_d)
    path = path_d
    polygon = []
    for segment in path:
        if isinstance(segment, CubicBezier):
            polygon.extend(cubic_bezier_to_points(segment.start, segment.control1, segment.control2, segment.end))
        elif isinstance(segment, Line):
            polygon.append((segment.end.real, segment.end.imag))
        # Add other segment types if needed

    # Point-in-polygon algorithm
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n+1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

# SVG file path
svg_file = './p5fulldata1-modified.svg'

# Extract and plot paths interactively
path_data, outer_contour, inner_contour = plot_interactive_paths(extract_svg_paths(svg_file))
show_3d_plot_and_save_obj(path_data, outer_contour[0], inner_contour[0], grid_density=50)
