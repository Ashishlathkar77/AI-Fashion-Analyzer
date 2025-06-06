from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import cv2
from skimage import color

def closest_color(requested_rgb):
    fashion_colors = {
        'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 128, 0),
        'black': (0, 0, 0), 'white': (255, 255, 255), 'gray': (128, 128, 128),
        'navy': (0, 0, 128), 'beige': (245, 245, 220), 'khaki': (240, 230, 140),
        'brown': (165, 42, 42), 'pink': (255, 192, 203), 'purple': (128, 0, 128),
        'orange': (255, 165, 0), 'yellow': (255, 255, 0), 'maroon': (128, 0, 0),
        'olive': (128, 128, 0), 'teal': (0, 128, 128), 'cream': (255, 253, 208),
        'burgundy': (128, 0, 32), 'coral': (255, 127, 80), 'mint': (189, 252, 201)
    }

    def euclidean(rgb1, rgb2):
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))

    closest = min(fashion_colors.items(), key=lambda c: euclidean(requested_rgb, c[1]))
    return closest[0]

def get_dominant_colors(image_path, num_colors=3):
    try:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Smart foreground extraction using GrabCut
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        result = image_rgb * mask2[:, :, np.newaxis]
        pixels = result.reshape(-1, 3)
        pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]  # Remove black

        if len(pixels) == 0:
            pixels = image_rgb.reshape(-1, 3)

        # Convert to LAB for perceptual clustering
        lab_pixels = color.rgb2lab(pixels.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)

        kmeans = KMeans(n_clusters=min(num_colors, len(np.unique(lab_pixels, axis=0))),
                        random_state=42, n_init=10)
        kmeans.fit(lab_pixels)

        labels = kmeans.labels_
        label_counts = Counter(labels)
        total_pixels = len(lab_pixels)

        result_colors = []
        for i, center_lab in enumerate(kmeans.cluster_centers_):
            # Convert back to RGB
            center_rgb = color.lab2rgb(center_lab.reshape(1, 1, 3)).reshape(3,) * 255
            center_rgb = tuple(map(int, np.clip(center_rgb, 0, 255)))
            hex_code = f"#{center_rgb[0]:02x}{center_rgb[1]:02x}{center_rgb[2]:02x}"
            color_name = closest_color(center_rgb)
            percentage = round((label_counts[i] / total_pixels) * 100, 2)

            result_colors.append({
                "rgb": center_rgb,
                "hex": hex_code,
                "name": color_name,
                "percentage": percentage
            })

        return sorted(result_colors, key=lambda x: x["percentage"], reverse=True)

    except Exception as e:
        print(f"AI-enhanced color analysis error: {e}")
        return []