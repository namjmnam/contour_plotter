import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from xml.dom import minidom
import svgpathtools

def plot_svg(svg_file):
    # Parse the SVG file
    doc = minidom.parse(svg_file)
    paths = [path for path in doc.getElementsByTagName('path')]
    doc.unlink()

    # Create a plot
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_axis_off()

    # Read each path and add it to the plot
    for path in paths:
        path_data = path.getAttribute('d')
        patch = mpatches.PathPatch(plt.Path.make_compound_path(plt.Path(*svg_path_to_polygons(path_data))),
                                   facecolor='none', edgecolor='black', lw=1)
        ax.add_patch(patch)

    # Show the plot
    plt.show()

# Helper function to convert SVG path to polygons
def svg_path_to_polygons(svg_path):
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from svgpathtools import parse_path

    path = parse_path(svg_path)
    polygons = []
    codes = []
    last_pos = None

    for segment in path:
        if isinstance(segment, svgpathtools.Line):
            codes += [Path.MOVETO if last_pos is None else Path.LINETO, Path.LINETO]
            polygons += [segment.start, segment.end]
            last_pos = segment.end
        elif isinstance(segment, svgpathtools.CubicBezier):
            curve_points = [segment.start, segment.control1, segment.control2, segment.end]
            codes += [Path.MOVETO if last_pos is None else Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            polygons += curve_points
            last_pos = segment.end
        elif isinstance(segment, svgpathtools.QuadraticBezier):
            curve_points = [segment.start, segment.control, segment.end]
            codes += [Path.MOVETO if last_pos is None else Path.LINETO, Path.CURVE3, Path.CURVE3]
            polygons += curve_points
            last_pos = segment.end

    return polygons, codes

plot_svg("flat_circles.svg")

