import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from xml.dom import minidom
import csv

# Function to extract paths and their classes from the SVG file
def extract_svg_paths(svg_file):
    doc = minidom.parse(svg_file)  # Parse the SVG file
    points = doc.getElementsByTagName('line')
    point_data = [{'x1': point.getAttribute('x1'), 'x2': point.getAttribute('x2'), 'y1': point.getAttribute('y1'), 'y2': point.getAttribute('y2'), 'class': point.getAttribute('class')} for point in points]
    doc.unlink()
    return point_data

# Replace 'path_to_svg_file.svg' with the path to your SVG file
svg_file_path = './p5fulldata1-points.svg'
point_data = extract_svg_paths(svg_file_path)

# Prepare lists for coordinates
x_coords = []
y_coords = []
z_coords = []

# Process data points
for data in point_data:
    x = (float(data['x1']) + float(data['x2'])) / 2
    y = (float(data['y1']) + float(data['y2'])) / 2
    z = float(data['class'].split('-')[1])
    x_coords.append(x)
    y_coords.append(y)
    z_coords.append(z)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_coords, y_coords, z_coords)
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
plt.show()

# Exporting to CSV
with open('coordinates.csv', 'w', newline='') as csvfile:
    fieldnames = ['X', 'Y', 'Z']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for x, y, z in zip(x_coords, y_coords, z_coords):
        writer.writerow({'X': x, 'Y': y, 'Z': z})

print("CSV file 'coordinates.csv' has been created.")