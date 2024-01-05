import numpy as np
from svgwrite import Drawing

def create_flat_circles(num_circles=5, radius_step=1, num_points=100):
    circles = []
    for i in range(num_circles):
        radius = radius_step * (i + 1)
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = 0  # Initial z value (height) is zero
        circles.append([x, y, z])  # x, y, z for each circle
    return circles

def save_circles_to_svg(circles, filename="circles.svg"):
    dwg = Drawing(filename, profile='tiny')

    for circle in circles:
        # We only need the radius for the SVG circle element
        radius = np.max(circle[0])  # Assuming the first index is x and it's a full circle
        dwg.add(dwg.circle(center=(0, 0), r=radius, stroke="black", fill="none"))

    dwg.save()

# Example usage
circles = create_flat_circles()
save_circles_to_svg(circles, "flat_circles.svg")
