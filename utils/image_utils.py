from PIL import Image, ImageEnhance
import os
import uuid
import io
import cv2
import numpy as np

def validate_and_process_image(uploaded_file):
    ext = uploaded_file.filename.split('.')[-1].lower()
    if ext not in ["jpg", "jpeg", "png"]:
        raise ValueError("Unsupported file type")

    filename = f"temp_{uuid.uuid4()}.jpg"
    path = os.path.join("static", filename)
    os.makedirs("static", exist_ok=True)

    # Load and convert to RGB
    with Image.open(uploaded_file.file) as img:
        img = img.convert("RGB")

        # Convert PIL to OpenCV for processing
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Denoise the image using OpenCV fastNlMeansDenoisingColored
        img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)

        # Enhance sharpness and brightness
        pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.3)
        pil_img = ImageEnhance.Brightness(pil_img).enhance(1.05)

        # Auto crop based on center + max size fit (future hook for face/body detection)
        pil_img.thumbnail((512, 512))

        pil_img.save(path, "JPEG")
        size = pil_img.size

    return path, size, os.path.getsize(path)