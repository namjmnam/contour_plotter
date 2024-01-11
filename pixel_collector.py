from PIL import Image
import matplotlib.pyplot as plt
import os

def pixel_counter(image_path):
    # Open the image
    with Image.open(image_path) as img:
        width, height = img.size

        # Initialize a list to store pixel counts
        black_pixel_counts = [0] * width

        # Iterate over each pixel
        for x in range(width):
            for y in range(height):
                pixel = img.getpixel((x, y))
                # Check for black color and non-transparent (alpha not 0)
                if pixel[:3] == (0, 0, 0) and (len(pixel) < 4 or pixel[3] != 0):
                    black_pixel_counts[x] += 1

        return black_pixel_counts

def get_all_pics(folder_path):
    pics = []

    # Loop through all files in the folder
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        # Check if it's a file and not a directory
        if os.path.isfile(file_path):
            pics.append(file_path)

    return pics

def plot_data(y_values, plot_type='scatter'):
    # Create x values as indices of the y values
    x_values = list(range(len(y_values)))

    # Plot based on the specified type
    if plot_type == 'scatter':
        plt.scatter(x_values, y_values)
    elif plot_type == 'bar':
        plt.bar(x_values, y_values)
    elif plot_type == 'line':
        plt.plot(x_values, y_values)
    else:
        raise ValueError("Invalid plot type. Choose 'scatter', 'bar', or 'line'.")

    # Labeling the axes
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Title of the plot
    plt.title(f'{plot_type.capitalize()} Plot of Values vs. Indices')

    # Show the plot
    plt.show()

full_pixels = []
for i in get_all_pics('./height1'):
    full_pixels+=pixel_counter(i)

plot_data(full_pixels, plot_type='line')