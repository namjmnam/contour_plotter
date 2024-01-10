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

# Create and write to a CSV file
with open('coordinates.csv', 'w', newline='') as csvfile:
    fieldnames = ['X', 'Y', 'Z']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Iterate over each data point and write to the CSV
    for data in point_data:
        x = (float(data['x1']) + float(data['x2'])) / 2
        y = (float(data['y1']) + float(data['y2'])) / 2
        z = float(data['class'].split('-')[1])
        writer.writerow({'X': x, 'Y': y, 'Z': z})
