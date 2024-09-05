import torch
import cv2
from PIL import Image
import clip
from camera import Camera

# Load YOLOv5 model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load CLIP model
clip_model, preprocess = clip.load('ViT-B/32', device=device)

# Open video capture
camera = Camera()

# Define text prompt for CLIP
text_prompt = input("Enter a target description: ")
text_features = clip_model.encode_text(clip.tokenize(text_prompt).to(device))

while True:
    frame, depth_frame, depth_colormap = camera.getting_frame()

    # Convert frame to PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform object detection with YOLOv5
    results = model(pil_img)

    # Parse results
    detections = results.pandas().xyxy[0] 

    # Process detections
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        detected_object_name = row['name']

        # Extract the object region from the frame
        object_img = frame[y1:y2, x1:x2]
        object_pil_img = Image.fromarray(cv2.cvtColor(object_img, cv2.COLOR_BGR2RGB))
        object_pil_img = preprocess(object_pil_img).unsqueeze(0).to(device)

        # Get the object features from CLIP
        with torch.no_grad():
            object_features = clip_model.encode_image(object_pil_img)

        # Compute similarity between detected object and text prompt
        similarity = torch.cosine_similarity(object_features, text_features)
        if similarity.item() > 0.25:  # Adjust threshold as needed
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{detected_object_name}: {similarity.item():.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display results
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.pipeline.stop()
cv2.destroyAllWindows()
