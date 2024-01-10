import xml.etree.ElementTree as ET
from xml.dom import minidom

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

# Output the path data
for data in point_data:
    x = (float(data['x1']) + float(data['x2']))/2
    y = (float(data['y1']) + float(data['y2']))/2
    z = float(data['class'].split('-')[1])
    print(x, y, z)
