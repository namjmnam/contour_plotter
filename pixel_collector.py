from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import pandas as pd

def pixel_counter(image_path):
    # Open the image
    with Image.open(image_path) as img:
        # Convert 'P' mode images to 'RGBA' for consistent processing
        if img.mode == 'P':
            img = img.convert('RGBA')

        width, height = img.size
        mode = img.mode

        # Initialize a list to store pixel counts
        black_pixel_counts = [0] * width

        # Iterate over each pixel
        for x in range(width):
            for y in range(height):
                pixel = img.getpixel((x, y))

                # Check for black color based on the mode
                if mode == 'RGBA':
                    # Check if pixel is black and non-transparent
                    if pixel[:3] == (0, 0, 0) and pixel[3] != 0:
                        black_pixel_counts[x] += 1
                elif mode in ['RGB', 'L']:
                    # Check if pixel is black (for grayscale, 0 is black)
                    if pixel == 0 or pixel[:3] == (0, 0, 0):
                        black_pixel_counts[x] += 1
                else:
                    raise ValueError(f"Unsupported image mode: {mode}")

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

def interpolate_and_sample(y_values, num_samples=50, plot_type='scatter'):
    # Create x values as indices of the y values
    x_values = np.arange(len(y_values))

    # Interpolate the data
    interpolation_function = interp1d(x_values, y_values, kind='linear')

    # Generate equally spaced sample points
    x_samples = np.linspace(x_values.min(), x_values.max(), num_samples)
    y_samples = interpolation_function(x_samples)

    # # Plot based on the specified type
    # if plot_type == 'scatter':
    #     plt.scatter(x_samples, y_samples)
    # elif plot_type == 'bar':
    #     plt.bar(x_samples, y_samples)
    # elif plot_type == 'line':
    #     plt.plot(x_samples, y_samples)
    # else:
    #     raise ValueError("Invalid plot type. Choose 'scatter', 'bar', or 'line'.")

    # # Labeling the axes
    # plt.xlabel('Sampled Index')
    # plt.ylabel('Sampled Value')

    # # Title of the plot
    # plt.title(f'{plot_type.capitalize()} Plot of Interpolated and Sampled Data')

    # # Show the plot
    # plt.show()
    return y_samples

def main():
    # For the first one (vertical)
    full_pixels = []
    full_interpolated = []
    for i in get_all_pics('./height1'):
        segment = pixel_counter(i)
        full_pixels+=segment
        interpolated_segment = interpolate_and_sample(segment, 11, plot_type='line')
        full_interpolated+=list(interpolated_segment)[:-1] # Remove overlapping element from the last of the list
    full_interpolated+=[0.0] # Re-add the last element of the last segment, which will always be 0

    # plot_data(full_pixels, plot_type='line')
    # plot_data(full_interpolated, plot_type='line')
    # print(len(full_interpolated))
    df1 = pd.DataFrame(full_interpolated, columns=['Z'])
    # print('vertical')
    # print(df1)

    # For the second one (horizontal)
    full_pixels = []
    full_interpolated = []
    for i in get_all_pics('./height2'):
        segment = pixel_counter(i)
        full_pixels+=segment
        interpolated_segment = interpolate_and_sample(segment, 11, plot_type='line')
        full_interpolated+=list(interpolated_segment)[:-1] # Remove overlapping element from the last of the list
    full_interpolated+=[51.0] # Re-add the last element of the last segment, which will always be 51

    # plot_data(full_pixels, plot_type='line')
    # plot_data(full_interpolated, plot_type='line')
    # print(len(full_interpolated))
    df2 = pd.DataFrame(full_interpolated, columns=['Z'])
    # print('horizontal')
    # print(df2)

    concatenated_df = pd.concat([df2, df1], axis=0, ignore_index=True) # horizontal first
    # print(concatenated_df)
    return concatenated_df

if __name__ == "__main__":
    returned_value = main()
    # print(returned_value)