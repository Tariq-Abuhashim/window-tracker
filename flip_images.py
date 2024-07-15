from PIL import Image
import os

def flip_images(directory):
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check for image files (e.g., .jpg, .png)
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Construct the full file path
            filepath = os.path.join(directory, filename)
            # Open the image
            with Image.open(filepath) as img:
                # Flip the image horizontally
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                # Construct new filename for the flipped image
                new_filename = f'{filename}'
                # Save the flipped image in the same directory
                flipped_img.save(os.path.join(directory, new_filename))
                print(f"Flipped image saved as {new_filename}")

# Replace 'path_to_your_directory' with the path to your directory containing images
flip_images('/media/mrt/Whale/data/mission-systems/2024-06-28-03-47-19-uotf-orbit-16/colmap/images')

