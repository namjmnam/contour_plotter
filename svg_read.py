import svgpathtools
import numpy as np

def parse_svg(file_path):
    paths, attributes = svgpathtools.svg2paths(file_path)
    circles = []
    lines = []

    for path, attribute in zip(paths, attributes):
        if 'circle' in attribute['id']:
            # Handle circle
            center = np.array(path[0].start)
            radius = np.abs(path[0].end - path[0].start)
            theta = np.linspace(0, 2 * np.pi, 100)
            x = center.real + radius * np.cos(theta)
            y = center.imag + radius * np.sin(theta)
            circles.append([x, y, np.zeros_like(x)])  # Add zero z-coordinate
        elif 'line' in attribute['id']:
            # Handle line
            start_point = np.array(path[0].start)
            end_point = np.array(path[0].end)
            x = [start_point.real, end_point.real]
            y = [start_point.imag, end_point.imag]
            lines.append([x, y, [0, 0]])  # Add zero z-coordinate for line start and end

    return circles, lines

# Example usage
file_path = "./JtossSVG1.svg"
circles, lines = parse_svg(file_path)

print(circles)
print(lines)