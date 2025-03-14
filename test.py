# Generalized function for lane detection across different environments (CARLA & Duckietown)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def show_seg_result(img, result,save_dir=None, is_ll=True, palette=None, is_demo=False, is_gt=False):
    # img = mmcv.imread(img)
    # img = img.copy()
    # seg = result[0]
    if palette is None:
        palette = np.random.randint(
            0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3  # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2

    if not is_demo:
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
    else:
        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)

        # for label, color in enumerate(palette):
        #     color_area[result[0] == label, :] = color

        color_area[result[0] == 1] = [0, 255, 0]
        color_area[result[1] == 1] = [255, 0, 0]
        color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    # print(color_seg.shape)
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    # img = img * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)

    return img



def feature_torch(image_path):
    model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
    model.eval()


    image = cv2.imread(image_path)
    cv2.imshow("cropped", image)
    cv2.waitKey(0)
    image = cv2.resize(image, (640, 640))
    shapes = image.shape
    img = transform(image).to(torch.device("cpu"))
    #img = img.permute(2, 0, 1)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    print(img.shape)
    det_out, da_seg_out, ll_seg_out = model(img)

    _, _, height, width = img.shape
    b, h, w, _ = img.shape
    pad_w, pad_h = 0,0
    pad_w = int(pad_w)
    pad_h = int(pad_h)
    ratio = 1

    da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
    da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1 / ratio), mode='bilinear')
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
    # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)

    ll_predict = ll_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
    ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1 / ratio), mode='bilinear')
    _, ll_seg_mask = torch.max(ll_seg_mask, 1)
    ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

    img_det = show_seg_result(image, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)
    cv2.imshow("window", img_det)
    cv2.waitKey(0)


def enhance_yellow_features(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define yellow color range in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Create a mask for yellow regions
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Extract yellow areas
    yellow_parts = cv2.bitwise_and(hsv, hsv, mask=mask)

    # Increase brightness and saturation of yellow areas
    yellow_parts = yellow_parts.astype(np.float32)
    yellow_parts[..., 1] = np.clip(yellow_parts[..., 1] * 1.5, 0, 255)  # Increase saturation
    yellow_parts[..., 2] = np.clip(yellow_parts[..., 2] * 1.5, 0, 255)  # Increase brightness
    yellow_parts = yellow_parts.astype(np.uint8)

    # Convert back to BGR
    enhanced_yellow = cv2.cvtColor(yellow_parts, cv2.COLOR_HSV2BGR)

    # Combine with original image
    enhanced_image = cv2.addWeighted(image, 0.7, enhanced_yellow, 0.3, 0)


    # Display the images (optional)
    cv2.imshow("Original Image", image)
    cv2.imshow("Enhanced Yellow Features", enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
feature_torch("C:\\Users\Jurriaan\Pictures\\test_duckie.jpg")
#detect_yellow_lanes("C:\\Users\Jurriaan\Pictures\image.PNG")  # CARLA
#detect_yellow_lanes("C:\\Users\Jurriaan\Pictures\duckie.jpg") # Duckietown
