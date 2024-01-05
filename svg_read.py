import matplotlib.pyplot as plt
from svg.path import parse_path
from xml.dom import minidom
import numpy as np

# Function to extract paths from the SVG file
def extract_svg_paths(svg_file):
    doc = minidom.parse(svg_file)  # Parse the SVG file
    path_strings = [path.getAttribute('d') for path in doc.getElementsByTagName('path')]
    doc.unlink()
    return path_strings

# Function to plot paths and add interactive points
def plot_interactive_paths(paths):
    fig, ax = plt.subplots()

    points = []  # List to store the points

    def on_pick(event):
        # This function will be called when a point is clicked
        point = event.artist
        x, y = point.get_data()
        z = float(input(f"Enter Z value for point ({x[0]}, {y[0]}): "))  # Ask user for Z value
        point.set_color('green')  # Change color to indicate the point has a Z value set
        points.append((x[0], y[0], z))  # Add the point and Z value to the list
        plt.draw()

    for path_string in paths:
        path = parse_path(path_string)

        for segment in path:
            if hasattr(segment, 'start') and hasattr(segment, 'end'):
                x_values = [segment.start.real, segment.end.real]
                y_values = [segment.start.imag, segment.end.imag]
                line, = ax.plot(x_values, y_values, 'b')

                # Add clickable red dot at the start of the segment
                red_dot, = ax.plot(segment.start.real, segment.start.imag, 'ro', picker=5, markersize=8)
                red_dot.set_picker(5)  # 5 points tolerance

    # Connect the pick event to the on_pick function
    fig.canvas.mpl_connect('pick_event', on_pick)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    return points

# SVG file path
svg_file = './JtossSVG1.svg'

# Extract and plot paths interactively
paths = extract_svg_paths(svg_file)
points_with_z = plot_interactive_paths(paths)

# You can now use points_with_z for further processing
