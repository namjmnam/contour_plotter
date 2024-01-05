# import numpy as np
# from svgwrite import Drawing

# def create_flat_circles(num_circles=5, radius_step=1, num_points=100):
#     circles = []
#     for i in range(num_circles):
#         radius = radius_step * (i + 1)
#         theta = np.linspace(0, 2 * np.pi, num_points)
#         x = radius * np.cos(theta)
#         y = radius * np.sin(theta)
#         z = 0  # Initial z value (height) is zero
#         circles.append([x, y, z])  # x, y, z for each circle
#     return circles

# def save_circles_to_svg(circles, filename="circles.svg"):
#     dwg = Drawing(filename, profile='tiny')

#     for circle in circles:
#         x, y, z = circle
#         # Connect each point in the circle with a line
#         for i in range(1, len(x)):
#             start_point = (x[i-1], y[i-1])
#             end_point = (x[i], y[i])
#             dwg.add(dwg.line(start=start_point, end=end_point, stroke="black"))
#         # Connect the last point with the first to close the circle
#         dwg.add(dwg.line(start=(x[-1], y[-1]), end=(x[0], y[0]), stroke="black"))

#     dwg.save()

# # Example usage
# circles = create_flat_circles()
# save_circles_to_svg(circles, "flat_circles.svg")


import numpy as np
from svgwrite import Drawing

def create_svg_with_polyline_circles(filename="polyline_circles.svg", num_circles=5, radius_step=1, num_points=100):
    dwg = Drawing(filename, profile='tiny')

    for i in range(num_circles):
        radius = radius_step * (i + 1)
        theta = np.linspace(0, 2 * np.pi, num_points)
        points = [(radius * np.cos(t), radius * np.sin(t)) for t in theta]
        # Close the circle by adding the first point at the end
        points.append(points[0])
        # Adding a polyline that represents the circle
        dwg.add(dwg.polyline(points, stroke='black', fill='none', stroke_width=0.2))

    dwg.save()

# Example usage
create_svg_with_polyline_circles("polyline_circles.svg", num_circles=5, radius_step=10)
