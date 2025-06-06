import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def detect_gender(image_path, detected_items):
    """Detect gender based on clothing items and visual cues."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {"gender": "unisex", "confidence": 0.5, "reasoning": "Could not analyze image"}

        male_score = 0
        female_score = 0
        total_indicators = 0

        clothing_gender_score = analyze_clothing_gender(detected_items)
        male_score += clothing_gender_score['male']
        female_score += clothing_gender_score['female']
        total_indicators += clothing_gender_score['total']

        color_gender_score = analyze_color_patterns(image)
        male_score += color_gender_score['male']
        female_score += color_gender_score['female']
        total_indicators += color_gender_score['total']

        shape_gender_score = analyze_garment_shapes(image)
        male_score += shape_gender_score['male']
        female_score += shape_gender_score['female']
        total_indicators += shape_gender_score['total']

        if total_indicators == 0:
            return {"gender": "unisex", "confidence": 0.5, "reasoning": "Insufficient data for gender detection"}

        male_percentage = (male_score / total_indicators) * 100
        female_percentage = (female_score / total_indicators) * 100

        if abs(male_percentage - female_percentage) < 15:
            gender = "unisex"
            confidence = 0.6
            reasoning = "Mixed gender indicators detected"
        elif male_percentage > female_percentage:
            gender = "male"
            confidence = round(min(0.95, 0.6 + (male_percentage - female_percentage) / 100), 2)
            reasoning = f"More masculine features detected ({male_percentage:.1f}%)"
        else:
            gender = "female"
            confidence = round(min(0.95, 0.6 + (female_percentage - male_percentage) / 100), 2)
            reasoning = f"More feminine features detected ({female_percentage:.1f}%)"

        return {
            "gender": gender,
            "confidence": confidence,
            "reasoning": reasoning,
            "scores": {
                "male_percentage": round(male_percentage, 1),
                "female_percentage": round(female_percentage, 1)
            }
        }

    except Exception as e:
        print(f"Gender detection error: {e}")
        return {"gender": "unisex", "confidence": 0.5, "reasoning": "Error in gender detection"}

def analyze_clothing_gender(detected_items):
    """Analyze clothing items for gender indicators."""
    male_items = {'shirt', 'tie', 'suit', 'blazer', 'polo', 'cargo pants', 'shorts'}
    female_items = {'dress', 'skirt', 'blouse', 'crop top', 'heels', 'handbag', 'purse'}
    unisex_items = {'jeans', 't-shirt', 'jacket', 'sneakers', 'pants', 'hoodie'}

    male_score = 0
    female_score = 0
    total = 0

    for item in detected_items:
        item_name = item.get('item', '').lower()
        confidence = item.get('confidence', 0.5)

        if any(male_item in item_name for male_item in male_items):
            male_score += confidence
            total += 1
        elif any(female_item in item_name for female_item in female_items):
            female_score += confidence
            total += 1
        elif any(unisex_item in item_name for unisex_item in unisex_items):
            male_score += confidence * 0.4
            female_score += confidence * 0.4
            total += 1

    return {'male': male_score, 'female': female_score, 'total': max(1, total)}

def analyze_color_patterns(image):
    """Analyze dominant colors using clustering for gender cues."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_small = cv2.resize(image_rgb, (64, 64)).reshape((-1, 3))

    try:
        kmeans = KMeans(n_clusters=3, random_state=42).fit(image_small)
        centers = kmeans.cluster_centers_

        male_score = 0
        female_score = 0
        for r, g, b in centers:
            brightness = np.mean([r, g, b])

            if brightness < 80:
                male_score += 0.3
            if brightness > 200:
                female_score += 0.2

            if r > g and r > b and r - g > 30:
                female_score += 0.4
            if b > r and b > g and b - r > 20:
                male_score += 0.3

        return {'male': male_score, 'female': female_score, 'total': 1}
    except Exception as e:
        print(f"Color analysis error: {e}")
        return {'male': 0, 'female': 0, 'total': 0}

def analyze_garment_shapes(image):
    """Detect garment shape silhouettes."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    male_score = 0
    female_score = 0
    count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 800:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            count += 1

            if 0.8 < aspect_ratio < 1.2:
                male_score += 0.2
            elif aspect_ratio < 0.6:
                female_score += 0.3

    return {'male': male_score, 'female': female_score, 'total': max(1, count // 5)}