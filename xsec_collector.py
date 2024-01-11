import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd

def get_polyline_points(svg_filename):
    tree = ET.parse(svg_filename)
    root = tree.getroot()
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}
    all_polylines_points = []

    for polyline in root.findall('.//svg:polyline', namespaces):
        points = polyline.get('points').strip()
        points_values = list(map(float, points.split()))
        
        # Pairing up the points as (x, y)
        points_list = [(points_values[i], points_values[i + 1]) for i in range(0, len(points_values), 2)]
        all_polylines_points.append(points_list)

    return all_polylines_points

def plot_polyline_points(polyline_points):
    plt.figure(figsize=(10, 6))

    for points in polyline_points:
        # Unpacking the list of tuples into separate lists for x and y coordinates
        x, y = zip(*points)
        plt.scatter(x, y)

    plt.title("Scatter Plot of Polyline Points")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

# Generate points separating sections into segments
def interpolate_points(points, num_of_segments):
    all_sections = []

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]

        # Calculate the step size for each coordinate
        dx = (p2[0] - p1[0]) / (num_of_segments + 1)
        dy = (p2[1] - p1[1]) / (num_of_segments + 1)

        # Create the list of interpolated points for this segment
        segment_points = [p1]
        for j in range(1, num_of_segments + 1):
            new_point = (p1[0] + j * dx, p1[1] + j * dy)
            segment_points.append(new_point)

        # Add the next point only if it's the last segment
        if i == len(points) - 2:
            segment_points.append(p2)

        all_sections.append(segment_points)

    return all_sections

horizontal_vertical = []

# Example usage
svg_filename = './p5fulldata1-points.svg'
polyline_points = get_polyline_points(svg_filename)
for i in polyline_points:
    # print(i)
    # print(len(i))
    horizontal_vertical.append(interpolate_points(i, 9))
# First one is x (horizontal)
# Second one is y (vertical)
# print(horizontal_vertical)
# for i in horizontal_vertical[0]:
#     print(i)
# for i in horizontal_vertical[1]:
#     print(i)

plot_polyline_points(polyline_points)
plot_polyline_points(horizontal_vertical[0]+horizontal_vertical[1])

flat_list = [point for polyline in horizontal_vertical for section in polyline for point in section]

# Create a DataFrame from the flat list
df = pd.DataFrame(flat_list, columns=['X', 'Y'])
print(df)