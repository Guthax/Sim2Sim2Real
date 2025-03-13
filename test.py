# Generalized function for lane detection across different environments (CARLA & Duckietown)
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_yellow_lanes(image_path):
    """
    Detects yellow lane markings in an image, applicable to both CARLA and Duckietown environments.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define yellow color range in HSV (tuned for both CARLA and Duckietown)
    yellow_lower = np.array([15, 60, 60])   # Adjusted for lighting variations
    yellow_upper = np.array([35, 255, 255])

    # Create a binary mask for yellow lanes
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Apply morphological operations to close gaps in dashed lines
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask_closed = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

    # Detect contours
    contours, _ = cv2.findContours(yellow_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (remove noise)
    min_contour_area = 50  # Adjust based on environment
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Draw contours on the original image
    lane_detected = image.copy()
    cv2.drawContours(lane_detected, filtered_contours, -1, (0, 255, 0), 2)  # Green contours

    # Display results
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Image")

    ax[1].imshow(yellow_mask_closed, cmap="gray")
    ax[1].set_title("Processed Yellow Lane Mask")

    ax[2].imshow(cv2.cvtColor(lane_detected, cv2.COLOR_BGR2RGB))
    ax[2].set_title("Lane Detection Output")

    plt.show()

# Run on both images
detect_yellow_lanes("C:\\Users\Jurriaan\Pictures\image.PNG")  # CARLA
detect_yellow_lanes("C:\\Users\Jurriaan\Pictures\duckie.jpg") # Duckietown
