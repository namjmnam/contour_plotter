from xml.etree import ElementTree as ET

def remove_open_paths(svg_content):
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

    # Return the modified SVG content
    return ET.tostring(root, encoding='unicode')

# Your SVG content
svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg id="Layer_14" data-name="Layer 14" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 3315.21 2892.83">
    <!-- SVG paths here -->
</svg>
'''

# Process the SVG to remove open paths
processed_svg = remove_open_paths(svg_content)

# Output the processed SVG content
print(processed_svg)
