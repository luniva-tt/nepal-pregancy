from route_engine import SafeRouter
import time

def build_cache():
    print("Starting cache build process...")
    start_time = time.time()
    
    # Files
    road_file = 'dataset/nepal_roads_full.gpkg'
    clinic_file = 'dataset/nepal_hospitals_full.geojson'
 # Dummy for init
    
    # Initialize router (triggers build_graph and pickle dump)
    router = SafeRouter(road_file, clinic_file)
    
    end_time = time.time()
    print(f"Done! Cache built in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    build_cache()
