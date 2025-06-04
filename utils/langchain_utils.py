from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)

def get_fashion_advice(colors, size_info, detected_items, gender_info):
    """Generate gender-appropriate fashion advice."""
    
    # Create gender-specific prompt
    gender = gender_info.get('gender', 'unisex')
    confidence = gender_info.get('confidence', 0.5)
    
    if gender == 'male':
        prompt_template = create_male_fashion_prompt()
    elif gender == 'female':
        prompt_template = create_female_fashion_prompt()
    else:
        prompt_template = create_unisex_fashion_prompt()
    
    # Create the chain
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    # Prepare input data
    colors_text = ", ".join([f"{c['name']} ({c['percentage']}%)" for c in colors[:3]])
    items_text = ", ".join([f"{item['item']} ({item['confidence']*100:.0f}% confidence)" for item in detected_items])
    
    response = chain.run({
        "colors": colors_text,
        "size_info": size_info,
        "detected_items": items_text,
        "gender": gender,
        "gender_confidence": confidence
    })
    
    return response

def create_male_fashion_prompt():
    """Create male-specific fashion advice prompt."""
    return PromptTemplate(
        input_variables=["colors", "size_info", "detected_items", "gender", "gender_confidence"],
        template="""
        You are a men's fashion consultant. Analyze this outfit for a male customer:
        
        Detected Items: {detected_items}
        Colors: {colors}
        Size: {size_info}
        Gender Detection: {gender} (confidence: {gender_confidence})
        
        Provide MALE-SPECIFIC advice including:
        
        1. **Product Description**: Describe the detected clothing items professionally
        
        2. **Style Assessment**: Rate the outfit and suggest improvements for men's fashion
        
        3. **Accessory Recommendations**: Suggest MALE accessories only:
        - Watches (not jewelry)
        - Belts
        - Ties/bow ties if formal
        - Cufflinks for dress shirts
        - Men's bags (briefcase, messenger bag)
        - Shoes that match
        - Hats/caps if appropriate
        
        4. **Outfit Suggestions**: Recommend complete men's outfits using similar colors
        
        5. **Social Media Caption**: Write a masculine, confident caption
        
        IMPORTANT: Do NOT suggest jewelry, handbags, heels, or feminine accessories.
        Focus on classic menswear and masculine styling.
        """
    )

def create_female_fashion_prompt():
    """Create female-specific fashion advice prompt."""
    return PromptTemplate(
        input_variables=["colors", "size_info", "detected_items", "gender", "gender_confidence"],
        template="""
        You are a women's fashion consultant. Analyze this outfit for a female customer:
        
        Detected Items: {detected_items}
        Colors: {colors}
        Size: {size_info}
        Gender Detection: {gender} (confidence: {gender_confidence})
        
        Provide FEMALE-SPECIFIC advice including:
        
        1. **Product Description**: Describe the detected clothing items with feminine styling focus
        
        2. **Style Assessment**: Rate the outfit and suggest improvements for women's fashion
        
        3. **Accessory Recommendations**: Suggest FEMALE accessories:
        - Jewelry (necklaces, earrings, bracelets, rings)
        - Handbags and purses
        - Scarves
        - Hair accessories
        - Belts
        - Shoes (heels, flats, boots)
        - Makeup suggestions to complement colors
        
        4. **Outfit Suggestions**: Recommend complete women's outfits with styling tips
        
        5. **Social Media Caption**: Write a fashionable, empowering caption
        
        Focus on feminine styling, elegant combinations, and women's fashion trends.
        """
    )

def create_unisex_fashion_prompt():
    """Create unisex fashion advice prompt."""
    return PromptTemplate(
        input_variables=["colors", "size_info", "detected_items", "gender", "gender_confidence"],
        template="""
        You are a fashion consultant. The gender detection was inconclusive, so provide UNISEX advice:
        
        Detected Items: {detected_items}
        Colors: {colors}
        Size: {size_info}
        Gender Detection: {gender} (confidence: {gender_confidence})
        
        Provide GENDER-NEUTRAL advice including:
        
        1. **Product Description**: Describe the outfit in neutral terms
        
        2. **Style Assessment**: Rate the outfit with unisex styling principles
        
        3. **Accessory Recommendations**: Suggest UNISEX accessories only:
        - Watches
        - Sunglasses
        - Bags (backpacks, crossbody bags)
        - Belts
        - Sneakers/casual shoes
        - Hats/caps
        - Minimal jewelry (if any)
        
        4. **Outfit Suggestions**: Recommend versatile, gender-neutral combinations
        
        5. **Social Media Caption**: Write an inclusive, style-focused caption
        
        AVOID: Gender-specific suggestions like makeup, high heels, or masculine/feminine language.
        Focus on universal style principles and versatile pieces.
        """
    )