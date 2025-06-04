import cv2
import numpy as np
from PIL import Image

def detect_gender(image_path, detected_items):
    """Detect gender based on clothing items and visual cues."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {"gender": "unisex", "confidence": 0.5, "reasoning": "Could not analyze image"}
        
        # Initialize scoring system
        male_score = 0
        female_score = 0
        total_indicators = 0
        
        # Analyze detected clothing items
        clothing_gender_score = analyze_clothing_gender(detected_items)
        male_score += clothing_gender_score['male']
        female_score += clothing_gender_score['female']
        total_indicators += clothing_gender_score['total']
        
        # Analyze colors for gender tendencies
        color_gender_score = analyze_color_patterns(image)
        male_score += color_gender_score['male']
        female_score += color_gender_score['female']
        total_indicators += color_gender_score['total']
        
        # Analyze garment shapes and cuts
        shape_gender_score = analyze_garment_shapes(image)
        male_score += shape_gender_score['male']
        female_score += shape_gender_score['female']
        total_indicators += shape_gender_score['total']
        
        # Determine final gender prediction
        if total_indicators == 0:
            return {"gender": "unisex", "confidence": 0.5, "reasoning": "Insufficient data for gender detection"}
        
        male_percentage = (male_score / total_indicators) * 100
        female_percentage = (female_score / total_indicators) * 100
        
        # Decision logic
        if abs(male_percentage - female_percentage) < 20:  # Close scores
            gender = "unisex"
            confidence = 0.6
            reasoning = "Mixed gender indicators detected"
        elif male_percentage > female_percentage:
            gender = "male"
            confidence = min(0.9, 0.5 + (male_percentage - female_percentage) / 100)
            reasoning = f"Male indicators: {male_percentage:.1f}%"
        else:
            gender = "female"
            confidence = min(0.9, 0.5 + (female_percentage - male_percentage) / 100)
            reasoning = f"Female indicators: {female_percentage:.1f}%"
        
        return {
            "gender": gender,
            "confidence": round(confidence, 2),
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
            # Neutral items get half score for both
            male_score += confidence * 0.3
            female_score += confidence * 0.3
            total += 1
    
    return {'male': male_score, 'female': female_score, 'total': max(1, total)}

def analyze_color_patterns(image):
    """Analyze color patterns for gender tendencies."""
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get average colors
    avg_color = np.mean(image_rgb.reshape(-1, 3), axis=0)
    
    male_score = 0
    female_score = 0
    
    # Color analysis based on traditional patterns
    r, g, b = avg_color
    
    # Dark colors tend to be more common in men's fashion
    if r < 100 and g < 100 and b < 100:  # Dark colors
        male_score += 0.6
    
    # Bright colors and pastels more common in women's fashion
    if r > 180 or g > 180 or b > 180:  # Bright/light colors
        female_score += 0.4
    
    # Pink/purple range
    if r > g and r > b and r - g > 30:  # Reddish tones
        female_score += 0.5
    
    # Blue tones (traditionally male)
    if b > r and b > g and b - r > 20:
        male_score += 0.3
    
    return {'male': male_score, 'female': female_score, 'total': 1}

def analyze_garment_shapes(image):
    """Analyze garment shapes and silhouettes."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    male_score = 0
    female_score = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            # Analyze shape characteristics
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Rectangular/boxy shapes (more common in menswear)
            if 0.8 < aspect_ratio < 1.2:
                male_score += 0.3
            
            # Fitted/curved shapes (more common in womenswear)
            elif aspect_ratio < 0.6:
                female_score += 0.4
    
    return {'male': male_score, 'female': female_score, 'total': 1}