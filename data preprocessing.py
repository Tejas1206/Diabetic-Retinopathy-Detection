'''
================================================================================
Circular Image Cropping and Resizing Script
================================================================================

Description:
    This script processes images by applying a circular crop around the image 
    center, applying a Gaussian blur, and resizing them to a specified output 
    size. The processed images are saved to a specified output directory.


Functions:
    crop_image_from_gray(img, tol=7):
        Crops an image by removing gray or nearly black borders.

    circle_crop(img_path, sigmaX=10, output_size=(256, 256)):
        Creates a circular crop around the image center, applies Gaussian blur, 
        and resizes the image.

    process_images(input_dir, output_dir, sigmaX=10, output_size=(224, 224)):
        Processes each image in the input directory by applying circular cropping, 
        resizing, and saving them to the output directory.

Usage:
    1. Set Input and Output Directories:
        input_dir = "path/to/input/directory"
        output_dir = "path/to/output/directory"
    
    2. Define Gaussian Blur and Output Size:
        sigmaX = 10
        output_size = (256, 256)
    
    3. Run the Script:
        process_images(input_dir, output_dir, sigmaX, output_size)


Error Handling:
    The script includes error handling for processing each image. If an image 
    fails to process, an error message is printed, and the script continues with 
    the next image.

Notes:
    Ensure that the input directory contains valid image files with extensions 
    .jpg, .jpeg, or .png.
    The script creates the output directory if it does not already exist.
'''

'''Importing Dependencies'''
import cv2
import os
import numpy as np

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:  # Grayscale image
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:  # Color image
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # Image is too dark, return original image
            return img
        else:
            # Apply mask to each channel
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            # Stack channels back into an image
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def circle_crop(img_path, sigmaX=10, output_size=(256, 256)):
    """Create circular crop around image centre and resize to output_size."""
    img = cv2.imread(img_path)
    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    
    # Resize the image to the desired output size
    img = cv2.resize(img, output_size)
    
    return img

def process_images(input_dir, output_dir, sigmaX=10, output_size=(256, 256)):
    """Process each image in the input directory, apply circular cropping, resize, and save to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_dir, filename)
            try:
                cropped_image = circle_crop(image_path, sigmaX, output_size)
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
                print(f"Cropped image saved to {output_path}")
            except Exception as e:
                print(f"Failed to process image {filename}. Error: {e}")

# Define input and output directories
input_dir = "D:/Downloads/resized_train/resized_train"
output_dir = "D:/Diabetic Retinopathy Detection/img_resized"

# Define sigmaX for Gaussian blur and desired output size
sigmaX = 10
output_size = (256, 256)

# Process and save the images
process_images(input_dir, output_dir, sigmaX, output_size)
