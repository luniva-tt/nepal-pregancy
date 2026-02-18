from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from route_engine import SafeRouter
import uvicorn
import os

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Pregnancy Safe Route API")

# Global router instance
router = None

@app.on_event("startup")
async def startup_event():
    global router
    road_file = os.path.join(BASE_DIR, 'dataset', 'nepal_roads_full.gpkg')
    clinic_file = os.path.join(BASE_DIR, 'dataset', 'nepal_hospitals_full.geojson')
    
    if os.path.exists(road_file) and os.path.exists(clinic_file):
        print("Initializing Routing Engine... (This may take a minute)")
        router = SafeRouter(road_file, clinic_file)
        print("Routing Engine Ready!")
    else:
        print(f"Warning: Data files not found at {road_file} or {clinic_file}. Routing will fail.")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(BASE_DIR, 'static', 'index.html'))

@app.get("/api/route")
async def get_route(
    lat: float = Query(..., description="Start Latitude"),
    lon: float = Query(..., description="Start Longitude"),
    week: int = Query(None, description="Pregnancy Week"),
    mode: str = Query("routine", description="Routing Mode (routine/high_risk/emergency)")
):
    global router
    if not router:
        raise HTTPException(status_code=503, detail="Routing engine not initialized")
    
    try:
        results = router.get_safest_route(lat, lon, week, mode)
        if not results:
            print("No route found.")
            raise HTTPException(status_code=404, detail="No route found")
        
        # Log a snippet of the result
        print(f"Routes found: {len(results)}. Mode: {mode}")
        return results
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error calculating route: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/hospitals")
async def get_hospitals():
    global router
    if not router or router.clinics_gdf is None:
        raise HTTPException(status_code=503, detail="Hospitals not loaded")
    
    return router.clinics_gdf.__geo_interface__

# Mount static files
static_dir = os.path.join(BASE_DIR, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
