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