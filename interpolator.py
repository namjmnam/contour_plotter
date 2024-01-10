from svg.path.path import CubicBezier
from svg.path.path import Line
from svg.path import parse_path
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from xml.dom import minidom
import pandas as pd
import numpy as np

def is_point_inside_path(x, y, path, num_segments=10):
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

# Function to plot paths and add interactive points
def get_outer_inner(paths):
    paths_data = []
    outer_contours = []
    inner_contours = []

    for index, path_string in enumerate(paths):
        path = parse_path(path_string['d'])
        path_points = []
        path_class = path_string['class']

        if path_class=='outer':
            outer_contours.append(path)

        elif path_class=='inner':
            inner_contours.append(path)

        else:
            continue

        for segment in path:
            # Sampling points from each segment
            segment_points = np.array([segment.point(t) for t in np.linspace(0, 1, 50)])
            path_points.extend(segment_points)

        x_values = [p.real for p in path_points]
        y_values = [p.imag for p in path_points]

        paths_data.append({'x': x_values, 'y': y_values, 'z': 0})
    return paths_data, outer_contours, inner_contours

# Function to extract paths from the SVG file
def extract_svg_paths(svg_file):
    doc = minidom.parse(svg_file)  # Parse the SVG file
    paths = doc.getElementsByTagName('path')
    path_data = [{'d': path.getAttribute('d'), 'class': path.getAttribute('class')} for path in paths]
    doc.unlink()
    return path_data

# Read the CSV file
df = pd.read_csv('./coordinates.csv')
svg_file = './p5fulldata1-modified.svg'

# Extract X, Y, and Z values
x = df['X'].values
y = df['Y'].values
z = df['Z'].values

paths_data, outer_contour, inner_contour = get_outer_inner(extract_svg_paths(svg_file))
outer = outer_contour[0]
inner = inner_contour[0]

# Apply scaling factor to all x and y coordinates from the paths
path_x = np.hstack([np.array(path['x']) for path in paths_data])
path_y = np.hstack([np.array(path['y']) for path in paths_data])

# Define the grid (change the range and density as needed)
# grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
grid_density = 50
grid_x, grid_y = np.meshgrid(
    np.linspace(min(x), max(x), grid_density),
    np.linspace(min(y), max(y), grid_density)
)

# Perform interpolation
grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

# Set z values to 0 for points outside the "outer" contour or inside the "inner" contour
for i in range(grid_z.shape[0]):
    for j in range(grid_z.shape[1]):
        if not is_point_inside_path(grid_x[i, j], grid_y[i, j], outer) or is_point_inside_path(grid_x[i, j], grid_y[i, j], inner):
            grid_z[i, j] = np.nan

# Plotting 1
plt.figure(figsize=(10, 8))
plt.contourf(grid_x, grid_y, grid_z, levels=15, cmap='viridis')  # Change levels and colormap as needed
plt.colorbar(label='Height (Z)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('3D Contour Map')
plt.scatter(x, y, c='black', marker='o')  # Add original data points to the plot
plt.xlim(95.47, 95.47+1705.53)
plt.ylim(73.74, 73.74+1399.15)
plt.show()

# Plotting 2
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Create a surface plot
surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')
# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Height (Z)')
# Labels and Title
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Height')
ax.set_title('3D Surface Plot')
ax.set_xlim(95.47, 95.47+1705.53)
ax.set_ylim(73.74, 73.74+1399.15)
plt.show()

# print(len(paths_data))