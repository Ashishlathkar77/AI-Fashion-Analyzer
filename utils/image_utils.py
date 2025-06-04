from PIL import Image
import os
import uuid

def validate_and_process_image(uploaded_file):
    ext = uploaded_file.filename.split('.')[-1].lower()
    if ext not in ["jpg", "jpeg", "png"]:
        raise ValueError("Unsupported file type")

    filename = f"temp_{uuid.uuid4()}.jpg"
    path = os.path.join("static", filename)
    os.makedirs("static", exist_ok=True)

    with Image.open(uploaded_file.file) as img:
        img = img.convert("RGB")
        img.thumbnail((512, 512))
        img.save(path, "JPEG")
        size = img.size

    return path, size, os.path.getsize(path)