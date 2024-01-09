import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from svg.path import parse_path
from xml.dom import minidom
import tkinter as tk
from tkinter import simpledialog
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
import random
from scipy.interpolate import Rbf

# Function to plot paths in 3D
def plot_3d_paths(path_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for path in path_data:
        # Each path's Z value is constant across all its points
        z_values = [path['z']] * len(path['x'])
        ax.plot(path['x'], path['y'], z_values)

    plt.show()

# Function to plot 3D surfaces based on contours
def plot_3d_surfaces(path_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for path in path_data:
        x = path['x']
        y = path['y']
        z = path['z']

        # Create vertices for each polygon slice
        verts = [list(zip(x, y, [z] * len(x)))]
        
        # Create a Poly3DCollection object
        poly = Poly3DCollection(verts, alpha=0.5)
        
        # Add the polygon to the axes
        ax.add_collection3d(poly)

    # Set the limits of the axes
    ax.set_xlim([min(path['x']), max(path['x'])])
    ax.set_ylim([min(path['y']), max(path['y'])])
    ax.set_zlim([0, max(path['z'] for path in path_data)])

    plt.show()

# Function to show 3D plot with contours projecting upwards
def show_wall_plot(path_data):
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    for path in path_data:
        x = np.array(path['x'])
        y = np.array(path['y'])
        z = np.array([path['z']] * len(x))

        # Duplicate x, y, and z for the base of the vertical lines
        x_base = np.repeat(x, 2)
        y_base = np.repeat(y, 2)
        z_base = np.repeat(z, 2)
        z_base[::2] = 0  # Set every other z value to 0 for the base

        # Create vertical lines for each contour
        ax_3d.plot(x_base, y_base, z_base, color='b')

    # Set the limits of the axes
    ax_3d.set_xlim([min(path['x']), max(path['x'])])
    ax_3d.set_ylim([min(path['y']), max(path['y'])])
    ax_3d.set_zlim([0, max(path['z'] for path in path_data)])

    plt.show()

def show_3d_contour_plot(path_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for path in path_data:
        # Combine x, y, and z values for triangulation
        xyz = np.vstack((np.array(path['x']), np.array(path['y']), np.array([path['z']] * len(path['x'])))).T
        if len(xyz) > 3:  # Need at least 4 points to create a triangulation
            tri = Delaunay(xyz[:, :2])  # 2D Delaunay triangulation, ignore Z
            ax.plot_trisurf(xyz[:, 0], xyz[:, 1], xyz[:, 2], triangles=tri.simplices, cmap='viridis', alpha=0.6)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()

def interpolate_z_values(points, values, grid_size, method='linear', plot=False):
    """
    Interpolate z values on a grid.

    Parameters:
    points : ndarray of floats, shape (n, D)
        Data point coordinates.
    values : ndarray of float or complex, shape (n,)
        Data values.
    grid_size : tuple of int
        Size of the grid (nx, ny).
    method : str, optional
        Method of interpolation. One of ['linear', 'nearest', 'cubic'].
    plot : bool, optional
        If True, plot the interpolated grid.

    Returns:
    grid_z : ndarray
        Z values on the interpolated grid.
    """

    # Create grid
    grid_x, grid_y = np.mgrid[0:1:complex(grid_size[0]), 0:1:complex(grid_size[1])]
    grid_x_flat = grid_x.ravel()
    grid_y_flat = grid_y.ravel()

    # Interpolation
    try:
        grid_z = griddata(points, values, (grid_x_flat, grid_y_flat), method=method)
        grid_z = grid_z.reshape(grid_x.shape)
    except Exception as e:
        print(f"An error occurred during interpolation: {e}")
        return None

    # Plotting
    if plot:
        plt.figure()
        plt.imshow(grid_z.T, extent=(0,1,0,1), origin='lower')
        plt.title(f'Interpolated Grid (method: {method})')
        plt.scatter(points[:,0], points[:,1], c=values)
        plt.colorbar()
        plt.show()

    return grid_z

# Function to show 3D plot using RBF interpolation
def show_3d_plot_with_rbf(path_data):
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    # Prepare data for interpolation
    all_x = np.hstack([path['x'] for path in path_data])
    all_y = np.hstack([path['y'] for path in path_data])
    all_z = np.hstack([np.full(len(path['x']), path['z']) for path in path_data])

    # Create an RBF interpolator
    rbf_interpolator = Rbf(all_x, all_y, all_z, function='linear')

    # Create a grid for the surface plot
    grid_x, grid_y = np.meshgrid(np.linspace(np.min(all_x), np.max(all_x), 1200), 
                                 np.linspace(np.min(all_y), np.max(all_y), 1200))

    # Interpolate z values on the grid using RBF
    grid_z = rbf_interpolator(grid_x, grid_y)

    # Create a surface plot
    surf = ax_3d.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', alpha=0.8)
    fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=5, label='Height')

    plt.show()

def show_3d_plot_linear_interpolation(path_data, max_z_value=100, scale_factor=1.0, grid_density=100, figsize=(12, 9), cmap='viridis', alpha=0.6, dpi=300):
    fig_3d = plt.figure(figsize=figsize, dpi=dpi)
    ax_3d = fig_3d.add_subplot(111, projection='3d')

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

    # Interpolate z values on the grid using linear interpolation
    grid_z = griddata(points, values, (grid_x_flat, grid_y_flat), method='linear')

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

    # Add grid lines for better visualization
    ax_3d.xaxis._axinfo["grid"].update({"color": "k", "linewidth": 0.5})
    ax_3d.yaxis._axinfo["grid"].update({"color": "k", "linewidth": 0.5})
    ax_3d.zaxis._axinfo["grid"].update({"color": "k", "linewidth": 0.5})

    # Add a color bar which maps values to colors.
    fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=5, label='Height')

    # Set axes limits to cover the full range of x and y values
    ax_3d.set_xlim(x_min, x_max)
    ax_3d.set_ylim(y_min, y_max)
    ax_3d.set_zlim(0, max_z_value)

    plt.show()

    
def show_3d_plot(path_data, max_z_value=100, scale_factor=1.0, grid_density=100, figsize=(12, 9), cmap='viridis', alpha=0.6):
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

def is_point_inside_polygon(point, polygon):
    """
    Check if a point (x, y) is inside a given polygon using the ray casting method.
    polygon is a list of (x, y) pairs.
    """
    x, y = point
    inside = False
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if y > min(y1, y2) and y <= max(y1, y2) and x <= max(x1, x2):
            if y1 != y2:
                xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
            if x1 == x2 or x <= xinters:
                inside = not inside
    return inside