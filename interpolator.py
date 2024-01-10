import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('./coordinates.csv')

# Extract X, Y, and Z values
x = df['X'].values
y = df['Y'].values
z = df['Z'].values

# Define the grid (change the range and density as needed)
grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]

# Perform interpolation
grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

# Plotting 1
plt.figure(figsize=(10, 8))
plt.contourf(grid_x, grid_y, grid_z, levels=15, cmap='viridis')  # Change levels and colormap as needed
plt.colorbar(label='Height (Z)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('3D Contour Map')
plt.scatter(x, y, c='black', marker='o')  # Add original data points to the plot
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
plt.show()