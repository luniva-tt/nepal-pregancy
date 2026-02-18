import os
import geopandas as gpd
from route_engine import SafeRouter

def test_filtering():
    road_file = 'dataset/roads_subset.geojson'
    clinic_file = 'dataset/nepal_hospitals_full.geojson'
    
    print("Initializing SafeRouter and filtering hospitals...")
    router = SafeRouter(road_file, clinic_file)
    
    if router.clinics_gdf is None or router.clinics_gdf.empty:
        print("Error: No clinics loaded or filtered.")
        return

    print(f"Total pregnancy-related clinics found: {len(router.clinics_gdf)}")
    
    # Check for any unnamed clinics
    unnamed = router.clinics_gdf[router.clinics_gdf['name'].str.contains('unnamed', case=False, na=True)]
    if not unnamed.empty:
        print(f"Error: Found {len(unnamed)} unnamed clinics!")
        print(unnamed[['name']].head())
    else:
        print("Success: No unnamed clinics found.")
        
    # Check for any pediatric-only clinics
    pediatric = router.clinics_gdf[
        (router.clinics_gdf['name'].str.contains('child', case=False, na=False) | 
         router.clinics_gdf['name'].str.contains('pediatric', case=False, na=False) |
         router.clinics_gdf['name'].str.contains('paediatric', case=False, na=False)) &
        ~(router.clinics_gdf['speciality'].str.contains('gynaec', case=False, na=False) |
          router.clinics_gdf['speciality'].str.contains('obstet', case=False, na=False) |
          router.clinics_gdf['speciality'].str.contains('matern', case=False, na=False))
    ]
    if not pediatric.empty:
        print(f"Error: Found {len(pediatric)} pediatric facilities that should be excluded!")
        print(pediatric[['name', 'speciality']].head())
    else:
        print("Success: No pediatric-only facilities found.")

if __name__ == "__main__":
    test_filtering()
