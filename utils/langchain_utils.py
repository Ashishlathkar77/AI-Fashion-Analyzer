from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4", temperature=0.7)

def get_fashion_advice(colors, size_info, detected_items, gender_info):
    """Generate advanced fashion advice using dynamic input conditioning."""

    gender = gender_info.get('gender', 'unisex')
    confidence = gender_info.get('confidence', 0.5)

    # Seasonal + palette inference
    season = detect_season()
    palette_type = detect_palette_type(colors)

    # Select appropriate prompt template
    if gender == 'male':
        prompt_template = create_male_fashion_prompt()
    elif gender == 'female':
        prompt_template = create_female_fashion_prompt()
    else:
        prompt_template = create_unisex_fashion_prompt()

    chain = LLMChain(llm=llm, prompt=prompt_template)

    colors_text = ", ".join([f"{c['name']} ({c['percentage']}%)" for c in colors[:3]])
    items_text = ", ".join([f"{item['item']} ({item['confidence']*100:.0f}% confidence)" for item in detected_items])

    # Run LLM chain
    response = chain.run({
        "colors": colors_text,
        "size_info": size_info,
        "detected_items": items_text,
        "gender": gender,
        "gender_confidence": confidence,
        "season": season,
        "palette_type": palette_type
    })

    return response

# === Advanced Inference Helpers ===

def detect_season():
    """Infer season based on current month."""
    month = datetime.now().month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    return "Fall"

def detect_palette_type(colors):
    """Infer color palette type (bold vs. neutral) based on saturation heuristics."""
    bold_colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow']
    neutral_colors = ['black', 'white', 'gray', 'brown', 'beige', 'navy']
    score = sum(1 for c in colors if c['name'].lower() in bold_colors)
    return "Bold" if score >= 2 else "Neutral"

# === Prompt Templates ===

def create_male_fashion_prompt():
    return PromptTemplate(
        input_variables=["colors", "size_info", "detected_items", "gender", "gender_confidence", "season", "palette_type"],
        template="""
You are a professional men's fashion consultant. Based on the details below, provide a detailed analysis:

- Detected Items: {detected_items}
- Dominant Colors: {colors} (Palette Type: {palette_type})
- Size Info: {size_info}
- Season: {season}
- Gender: {gender} (Confidence: {gender_confidence})

Please deliver:
1. **Stylized Product Description**: Describe the detected items with rich masculine styling cues.
2. **Style Fit & Seasonality**: Rate the ensemble and explain how well it fits {season} trends.
3. **Masculine Accessories**: Suggest men’s accessories (e.g. watches, bags, belts).
4. **Style Alternatives**: Suggest 2 outfit combinations with similar color palette ({palette_type}) for {season}.
5. **Instagram Caption**: Write a bold, masculine, engaging caption using fashion tone.

IMPORTANT: Avoid any feminine style suggestions. Stick to modern or timeless men's styling.
"""
    )

def create_female_fashion_prompt():
    return PromptTemplate(
        input_variables=["colors", "size_info", "detected_items", "gender", "gender_confidence", "season", "palette_type"],
        template="""
You are a leading women's fashion stylist. Review this outfit:

- Detected Items: {detected_items}
- Colors: {colors} (Palette: {palette_type})
- Size Info: {size_info}
- Season: {season}
- Gender: {gender} (Confidence: {gender_confidence})

Deliver:
1. **Feminine Product Breakdown**: Describe each item with stylish, feminine detail.
2. **Trend Check**: Evaluate the fit and suggest updates based on {season} fashion.
3. **Accessories for Her**: Recommend perfect accessories (jewelry, bags, shoes).
4. **Outfit Suggestions**: Share 2 fashionable women's outfit ideas for {season}, using {palette_type} colors.
5. **Instagram Caption**: Craft a chic, elegant, empowering caption.

Avoid any masculine references. Focus on elevated, trendy women's wear.
"""
    )

def create_unisex_fashion_prompt():
    return PromptTemplate(
        input_variables=["colors", "size_info", "detected_items", "gender", "gender_confidence", "season", "palette_type"],
        template="""
You are a unisex fashion stylist. The gender was inconclusive, so remain neutral:

- Items: {detected_items}
- Colors: {colors} (Palette: {palette_type})
- Size: {size_info}
- Season: {season}
- Gender Detection: {gender} (Confidence: {gender_confidence})

Please deliver:
1. **Gender-Neutral Product Summary**: Describe the look without gender bias.
2. **Style Evaluation**: Assess overall style fit for the current {season}.
3. **Unisex Accessories**: Recommend universally wearable items (hats, shoes, watches).
4. **Neutral Outfit Suggestions**: Suggest 2 versatile outfits suitable for anyone using the {palette_type} palette.
5. **Inclusive Caption**: Write a creative, inclusive social caption.

Avoid gender-specific language or styling. Prioritize versatility and universal appeal.
"""
    )