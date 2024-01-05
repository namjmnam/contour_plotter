import matplotlib.pyplot as plt
from svg.path import parse_path
from xml.dom import minidom

# Function to extract paths from the SVG file
def extract_svg_paths(svg_file):
    doc = minidom.parse(svg_file)  # Parse the SVG file
    path_strings = [path.getAttribute('d') for path in doc.getElementsByTagName('path')]
    doc.unlink()
    return path_strings

# Function to plot paths
def plot_paths(paths):
    for path_string in paths:
        path = parse_path(path_string)

        for segment in path:
            # Assuming segment is a Line or CubicBezier, more types can be added
            if hasattr(segment, 'start') and hasattr(segment, 'end'):
                x_values = [segment.start.real, segment.end.real]
                y_values = [segment.start.imag, segment.end.imag]
                plt.plot(x_values, y_values, 'b')
            # Add more handling here for other segment types if needed

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# SVG file path
svg_file = './JtossSVG1.svg'

# Extract and plot paths
paths = extract_svg_paths(svg_file)
plot_paths(paths)
