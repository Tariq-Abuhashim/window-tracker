import os
import json
import numpy as np

def read_normals_and_centers(filename):
    """
    Read normals and centers from a JSON file.
    :param filename: Name of the JSON file containing the window data.
    :return: A tuple containing two lists: one for normals and another for centers, both represented as NumPy arrays.
    """
    with open(filename, 'r') as file:
        data = json.load(file)

    normals = []
    centers = []
    for item in data:
        center_array = np.array(item["normal_location"])
        normal_array = np.array(item["normal_direction"])

        centers.append(center_array)
        normals.append(normal_array)

    return normals, centers

def read_normals_from_input(input_file):
    """
    General reader for normals and centers based on file type.
    :param input_file: Path to the input file.
    :return: A tuple of lists (normals, centers) depending on the file type.
    """
    if not os.path.exists(input_file):
        raise ValueError(f"Error! Input file/directory {input_file} not found.")

    if input_file.endswith('.json'):
        return read_normals_and_centers(input_file)

    # Placeholder for potential future file type support
    # elif input_file.endswith('.txt'):
    #     return read_normals_from_txt(input_file)

    raise ValueError(f"Error! File {input_file} not supported. Should be .json for the normals and centers.")

# Usage example
try:
    normals, centers = read_normals_from_input('/media/mrt/Whale/data/mission-systems/2024-06-28-03-47-19-uotf-orbit-16/colmap/normals_results.json')
    print("Normals:", normals)
    print("Centers:", centers)
except ValueError as e:
    print(e)
