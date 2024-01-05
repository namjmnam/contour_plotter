from xml.etree import ElementTree as ET

def remove_open_paths(svg_file_path, output_file_path):
    # Read the SVG file
    with open(svg_file_path, 'r', encoding='utf-8') as file:
        svg_content = file.read()

    # Parse the SVG content
    tree = ET.ElementTree(ET.fromstring(svg_content))
    root = tree.getroot()

    # Namespace handling (if required)
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}

    # Find all path elements
    paths = root.findall('.//svg:path', namespaces)

    # Iterate and remove open paths
    for path in paths:
        d_attr = path.get('d')
        if 'Z' not in d_attr and 'z' not in d_attr:
            root.remove(path)

    # Write the modified SVG content to a new file
    tree.write(output_file_path, encoding='unicode')

# Specify the file paths
input_svg_file = './p5oil.svg'  # Change to your input file path
output_svg_file = './p5oil-modified.svg'  # Change to your desired output file path

# Process the SVG and save the result
remove_open_paths(input_svg_file, output_svg_file)
