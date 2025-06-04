from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import webcolors
import cv2

def closest_color(requested_rgb):
    """Find the closest named color using improved color matching."""
    min_distance = float('inf')
    closest_name = None
    
    # Enhanced color dictionary with fashion-specific colors
    fashion_colors = {
        'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 128, 0),
        'black': (0, 0, 0), 'white': (255, 255, 255), 'gray': (128, 128, 128),
        'navy': (0, 0, 128), 'beige': (245, 245, 220), 'khaki': (240, 230, 140),
        'brown': (165, 42, 42), 'pink': (255, 192, 203), 'purple': (128, 0, 128),
        'orange': (255, 165, 0), 'yellow': (255, 255, 0), 'maroon': (128, 0, 0),
        'olive': (128, 128, 0), 'teal': (0, 128, 128), 'cream': (255, 253, 208),
        'burgundy': (128, 0, 32), 'coral': (255, 127, 80), 'mint': (189, 252, 201)
    }
    
    for name, rgb in fashion_colors.items():
        distance = sum((c1 - c2) ** 2 for c1, c2 in zip(requested_rgb, rgb)) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    
    return closest_name or 'unknown'

def get_dominant_colors(image_path, num_colors=3):
    """Extract dominant colors using K-means clustering for better accuracy."""
    try:
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Remove background (simple method - can be enhanced)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Focus on clothing areas (exclude extreme whites/backgrounds)
        pixels = image.reshape(-1, 3)
        
        # Filter out background pixels
        mask_flat = mask.reshape(-1)
        clothing_pixels = pixels[mask_flat > 0]
        
        if len(clothing_pixels) == 0:
            clothing_pixels = pixels
        
        # Use K-means for better color clustering
        kmeans = KMeans(n_clusters=min(num_colors, len(np.unique(clothing_pixels, axis=0))), 
                       random_state=42, n_init=10)
        kmeans.fit(clothing_pixels)
        
        colors = []
        total_pixels = len(clothing_pixels)
        
        # Get cluster labels and count frequencies
        labels = kmeans.labels_
        label_counts = Counter(labels)
        
        for i, center in enumerate(kmeans.cluster_centers_):
            rgb = tuple(map(int, center))
            count = label_counts[i]
            
            hex_code = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            color_name = closest_color(rgb)
            percentage = round((count / total_pixels) * 100, 2)
            
            colors.append({
                "rgb": rgb,
                "hex": hex_code,
                "name": color_name,
                "percentage": percentage
            })
        
        return sorted(colors, key=lambda x: x['percentage'], reverse=True)
        
    except Exception as e:
        print(f"Enhanced color analysis error: {e}")
        return []