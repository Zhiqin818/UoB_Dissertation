import torch
import clip
from PIL import Image
import cv2
import numpy as np
from camera import Camera

# Load the CLIP model and the preprocessing method
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def get_color_and_shape(text_prompt):
    colors = ["red", "green", "blue", "yellow", "black", "purple", "orange", "pink", "gray", "brown"]
    shapes = ["cube", "cuboid", "sphere", "pyramid", "cylinder", "cone", "torus", "prism", "circle", "square", "triangle"]

    words = text_prompt.lower().split()
    target_color = None
    target_shape = None

    for word in words:
        if word in colors:
            target_color = word
        if word in shapes:
            target_shape = word

    return target_color, target_shape

# Function to convert color name to HSV range
def get_hsv_range(color_name):
    color_ranges = {
        "red": ([0, 50, 50], [10, 255, 255]),
        "green": ([35, 50, 50], [85, 255, 255]),
        "blue": ([90, 50, 50], [130, 255, 255]),
        "yellow": ([20, 100, 100], [30, 255, 255]),
        "orange": ([10, 100, 100], [20, 255, 255]),
        "purple": ([130, 50, 50], [160, 255, 255]),
        "pink": ([160, 50, 50], [180, 255, 255]),
    }
    return color_ranges.get(color_name.lower(), ([0, 0, 0], [180, 255, 255]))


def detect_shape(contour):
    # Approximate the contour to reduce the number of vertices
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    num_vertices = len(approx)

    if num_vertices == 3:
        return "triangle prism"
    elif num_vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.6 <= aspect_ratio <= 1.4:
            return "cube"
        else:
            return "cuboid"
    elif num_vertices > 10:
        return "sphere"

    return "unknown"

# Define the target color and shape
text_prompt = input("Enter a target description: ")
target_color, target_shape = get_color_and_shape(text_prompt)

# Tokenize the text prompt
text = clip.tokenize([text_prompt]).to(device)

camera = Camera()

while True:
    # get the color frame
    color_image, depth_frame, depth_colormap = camera.getting_frame()

    # Convert the frame to a PIL image
    image = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

    # Preprocess the image
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Perform inference with CLIP
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text)

        # Calculate similarity between image and text
        logits_per_image, logits_per_text = model(image_input, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    if (probs[0][0] > 0.5):

        # Convert frame to HSV color space
        hsv_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        lower_hsv, upper_hsv = get_hsv_range(target_color)
        mask = cv2.inRange(hsv_frame, np.array(lower_hsv), np.array(upper_hsv))

        # Find contours of the masked objects
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize a new mask to draw filtered contours
        filtered_mask = np.zeros_like(mask)

        # Define the minimum area threshold
        min_area = 200

        # Loop over contours and filter by area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                # Draw the contour on the filtered mask
                cv2.drawContours(filtered_mask, [contour], -1, (255), thickness=cv2.FILLED)

        mask = filtered_mask

        # Define the structuring element
        ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Apply morphological opening to remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse_kernel, iterations=3)

        # Find contours of the masked objects
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # If the model detects a high probability for the color description
        if probs[0][0] > 0.5:  # You can adjust this threshold
            for contour in contours:
                if cv2.contourArea(contour) > 50:  # Filter small contours
                    shape = detect_shape(contour)
                    if shape in target_shape:
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(color_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(color_image, text_prompt, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', color_image)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.pipeline.stop()
cv2.destroyAllWindows()