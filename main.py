from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from utils.image_utils import validate_and_process_image
from utils.color_analysis import get_dominant_colors
from utils.object_detection import detect_clothing, estimate_size
from utils.langchain_utils import get_fashion_advice
import json
from datetime import datetime
import os

app = FastAPI(title="Fashion Analyzer API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        # Process image
        path, size, file_size = validate_and_process_image(file)
        
        # Enhanced analysis
        colors = get_dominant_colors(path)
        size_info = estimate_size(path)
        detected = detect_clothing(path)
        gpt_insights = get_fashion_advice(colors, size_info, detected)
        
        # Generate comprehensive report
        report = {
            "analysis_id": datetime.now().strftime("%Y%m%d%H%M%S"),
            "image_info": {
                "name": file.filename,
                "size_bytes": file_size,
                "dimensions": size,
                "processed_path": path
            },
            "color_analysis": {
                "dominant_colors": colors,
                "color_count": len(colors),
                "primary_color": colors[0]['name'] if colors else "unknown"
            },
            "garment_detection": {
                "detected_items": detected,
                "item_count": len(detected),
                "confidence_avg": round(sum(item['confidence'] for item in detected) / len(detected), 2) if detected else 0
            },
            "size_analysis": size_info,
            "ai_recommendations": gpt_insights,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save enhanced report
        timestamp = report["analysis_id"]
        output_path = f"exports/analysis_{timestamp}.json"  # Fixed variable name
        os.makedirs("exports", exist_ok=True)
        
        with open(output_path, "w") as f:  # Fixed variable name
            json.dump(report, f, indent=2)
        
        return {
            "success": True,
            "report": report,
            "download_link": f"/download/{timestamp}",
            "summary": {
                "colors": len(colors),
                "items": len(detected),
                "size": size_info["recommended_size"],
                "confidence": report["garment_detection"]["confidence_avg"]
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Analysis failed. Please try with a different image."
        }

@app.get("/download/{timestamp}")
async def download_report(timestamp: str):
    file_path = f"exports/analysis_{timestamp}.json"  # Fixed variable name
    
    if not os.path.exists(file_path):
        return {"error": "Report not found"}
    
    return FileResponse(
        path=file_path, 
        filename=f"fashion_analysis_{timestamp}.json", 
        media_type='application/json'
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0", "features": ["color_analysis", "object_detection", "size_estimation", "ai_insights"]}