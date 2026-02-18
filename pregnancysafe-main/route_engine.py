import geopandas as gpd
import networkx as nx
import math
import logging
from shapely.geometry import Point, LineString

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafeRouter:
    def __init__(self, road_file, clinic_file):
        """
        Initialize the SafeRouter with road network and clinic data.
        """
        self.road_file = road_file
        self.clinic_file = clinic_file
        self.G = None 
        self.current_bbox = None # (minx, miny, maxx, maxy)
        
        logger.info(f"Nationwide Router initialized with {road_file}")
        try:
            full_clinics = gpd.read_file(clinic_file)
            
            # Keywords strictly for pregnancy-related facilities
            pregnancy_keywords = [
                'gynaecology', 'obstetrics', 'delivery', 'maternity', 
                'pregnancy', 'birth', 'matern', 'abortion', 
                'family_planning', 'fertility'
            ]
            
            # Specialties to explicitly exclude unless pregnancy keywords present
            exclude_keywords = [
                'dental', 'dentel', 'eye', 'vision', 'optical', 'skin', 'dermat', 
                'ent', 'ortho', 'physio', 'cardio', 'heart', 'urology', 'ayurved'
            ]
            
            def is_pregnancy_related(row):
                name = str(row.get('name', '')).lower()
                speciality = str(row.get('speciality', '')).lower()
                amenity = str(row.get('amenity', '')).lower()
                
                # Check for valid name
                if not name or name == 'none' or 'unnamed' in name or 'anonymous' in name:
                    return False
                
                # Exclude pediatric-only facilities unless pregnancy keywords present
                if ('child' in name or 'pediatric' in name or 'paediatric' in name) and not any(kw in speciality for kw in pregnancy_keywords):
                    return False
                
                # Main check: keywords OR hospital tag
                has_keywords = any(kw in name for kw in pregnancy_keywords) or \
                               any(kw in speciality for kw in pregnancy_keywords)
                
                # Exclusion logic
                has_exclude = any(kw in name for kw in exclude_keywords) or \
                              any(kw in speciality for kw in exclude_keywords)
                
                if has_exclude and not has_keywords:
                    return False

                # Major hospitals are pregnancy-safe fallbacks
                is_major_hosp = amenity == 'hospital'
                
                return has_keywords or is_major_hosp

            # Filter the GeoDataFrame
            self.clinics_gdf = full_clinics[full_clinics.apply(is_pregnancy_related, axis=1)].copy()
            logger.info(f"Loaded {len(self.clinics_gdf)} total medical facilities")
            
            # Debug: Check if Dhulikhel Hospital is in
            dh_name = "Dhulikhel"
            dh_check = self.clinics_gdf[self.clinics_gdf['name'].str.contains(dh_name, case=False, na=False)]
            if not dh_check.empty:
                logger.info(f"Confirmed {dh_name} matches in filtered list: {dh_check['name'].tolist()}")
            else:
                logger.warning(f"{dh_name} Hospital MISSING from filtered list!")
                
        except Exception as e:
            logger.error(f"Failed to load clinics: {e}. Routing won't work.")
            self.clinics_gdf = None

    # --- Build regional road network ---
    def _build_regional_graph(self, bbox):
        import time
        start_t = time.time()
        try:
            print(f"Loading regional roads for bbox: {bbox}")
            # Buffer the bbox slightly (approx 5km)
            buffered_bbox = (bbox[0]-0.05, bbox[1]-0.05, bbox[2]+0.05, bbox[3]+0.05)
            gdf = gpd.read_file(self.road_file, bbox=buffered_bbox, engine="pyogrio")
            
            G = nx.Graph()
            geoms = gdf.geometry
            surfaces = gdf.get('surface', ['unknown'] * len(gdf))
            highways = gdf.get('highway', ['unknown'] * len(gdf))
            
            edge_count = 0
            for geom, surface, highway in zip(geoms, surfaces, highways):
                if not geom: continue
                lines = [geom] if geom.geom_type == 'LineString' else list(geom.geoms)
                for line in lines:
                    safety_factor = self._get_safety_factor(surface, highway)
                    coords = [(round(p[0], 6), round(p[1], 6)) for p in line.coords]
                    for i in range(len(coords) - 1):
                        G.add_edge(coords[i], coords[i+1], weight=self._haversine(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1]), safety_factor=safety_factor)
                        edge_count += 1
            
            print(f"Regional graph ready in {time.time()-start_t:.2f}s: {len(G.nodes)} nodes, {edge_count} edges")
            return G
        except Exception as e:
            print(f"Error building regional graph: {e}")
            return nx.Graph()

    # --- Surface-based safety factor ---
    def _get_safety_factor(self, surface, highway):
        surface = str(surface).lower()
        highway = str(highway).lower()
        if any(s in surface for s in ['paved', 'asphalt', 'concrete', 'metal', 'black topped', 'rcc', 'cement', 'paving_stones']):
            return 1.0
        if any(s in surface for s in ['gravel', 'unpaved', 'compacted', 'fine_gravel', 'brick', 'bricks', 'sett', 'paving_stones']):
            return 1.15
        if any(s in surface for s in ['dirt', 'earth', 'ground', 'sand', 'shingle', 'pebblestone']):
            return 1.4
        if any(s in surface for s in ['mud', 'rock', 'clay', 'moraine', 'stairs', 'steps']):
            return 3.0
        if highway in ['primary', 'secondary', 'trunk', 'motorway']:
            return 1.0
        elif highway in ['residential', 'tertiary']:
            return 1.1
        elif highway in ['track', 'path']:
            return 1.8
        return 1.2

    # --- Haversine distance ---
    def _haversine(self, lon1, lat1, lon2, lat2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    # --- Safest route with nearby hospital shortcut ---
    def get_safest_route(self, start_lat, start_lon, week=None, mode=None):
        """
        Finds the top 3 safest hospitals. Auto-includes hospitals within 100m.
        """
        if self.clinics_gdf is None or self.clinics_gdf.empty:
            return None

        # Step 0: immediate nearby hospital check
        nearby_results = []
        for idx, row in self.clinics_gdf.iterrows():
            c = row.geometry if row.geometry.geom_type == 'Point' else row.geometry.centroid
            d_s = self._haversine(start_lat, start_lon, c.y, c.x)
            if d_s <= 100:
                nearby_results.append({
                    "score": 100,
                    "distance_meters": d_s,
                    "time_minutes": 0,
                    "avg_safety_factor": 1.0,
                    "route_segments": [],
                    "destination": {
                        "name": row['name'],
                        "lat": c.y,
                        "lon": c.x,
                        "opening_hours": str(row.get('opening_hours', 'None'))
                    }
                })

        if nearby_results:
            nearby_results.sort(key=lambda x: x['distance_meters'])
            return nearby_results[:3]

        # Step 1: Mode logic
        mode = mode.lower() if mode else "routine"
        if mode == "emergency":
            penalties = {"smooth": 1.0, "moderate": 1.1, "rough": 1.3, "avoid": 2.5}
            max_dist = 10000 
        elif mode == "high_risk" or (week and int(week) >= 28):
            penalties = {"smooth": 1.0, "moderate": 1.2, "rough": 1.6, "avoid": 3.5}
            max_dist = 20000 
        else:
            penalties = {"smooth": 1.0, "moderate": 1.15, "rough": 1.4, "avoid": 3.0}
            max_dist = 50000 

        def weight_func(u, v, d):
            safety = d.get('safety_factor', 1.2)
            p = penalties["smooth"] if safety <= 1.0 else (penalties["moderate"] if safety <= 1.15 else (penalties["rough"] if safety <= 1.5 else penalties["avoid"]))
            return d.get('weight', 1.0) * p

        # Step 2: Candidate selection
        candidates = []
        for idx, row in self.clinics_gdf.iterrows():
            c = row.geometry if row.geometry.geom_type == 'Point' else row.geometry.centroid
            d_s = self._haversine(start_lat, start_lon, c.y, c.x)
            if d_s <= max_dist:
                is_hosp = 1 if row.get('amenity') == 'hospital' else 0
                candidates.append((is_hosp, d_s, idx, c.y, c.x))
        candidates.sort(key=lambda x: (-x[0], x[1]))
        candidates = candidates[:8]
        if not candidates: 
            return None

        # Step 3: Regional graph
        all_lats = [start_lat] + [c[3] for c in candidates]
        all_lons = [start_lon] + [c[4] for c in candidates]
        bbox = (min(all_lons)-0.01, min(all_lats)-0.01, max(all_lons)+0.01, max(all_lats)+0.01)

        if self.G and self.current_bbox and \
           bbox[0] >= self.current_bbox[0] and bbox[1] >= self.current_bbox[1] and \
           bbox[2] <= self.current_bbox[2] and bbox[3] <= self.current_bbox[3]:
            logger.info("Reusing cached regional graph.")
            G_reg = self.G
        else:
            G_reg = self._build_regional_graph(bbox)
            self.G = G_reg
            self.current_bbox = bbox

        if not G_reg or len(G_reg.nodes) == 0: 
            logger.warning("No roads found in regional search area.")
            return None

        node_list = list(G_reg.nodes)
        def find_fast(lat, lon):
            return min(node_list, key=lambda n: (n[1]-lat)**2 + (n[0]-lon)**2)

        start_node = find_fast(start_lat, start_lon)

        # Step 4: A* pathfinding and scoring
        results = []
        for is_hosp, dist_s, clinic_idx, t_lat, t_lon in candidates:
            end_node = find_fast(t_lat, t_lon)
            try:
                path = nx.astar_path(G_reg, start_node, end_node,
                                     lambda u,v: self._haversine(u[1], u[0], v[1], v[0]),
                                     weight_func)
                total_dist = 0; roughness_sum = 0; travel_time = 0; path_segments = []
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    d = G_reg.get_edge_data(u, v)
                    dist = d.get('weight', 0); s = d.get('safety_factor', 1.0)
                    total_dist += dist; roughness_sum += (s-1.0)*dist
                    speed = 40 if s <=1.0 else (25 if s<=1.15 else (15 if s<=1.5 else 5))
                    travel_time += (dist/(speed*1000/60))
                    path_segments.append({"coords":[u,v], "safety":s})
                avg_roughness = (roughness_sum/total_dist) if total_dist>0 else 0
                dist_penalty = (total_dist/1000)*1.5
                rough_penalty = avg_roughness*30.0
                time_penalty = travel_time*0.8
                score = max(5, 100-(dist_penalty+rough_penalty+time_penalty))
                results.append({
                    "score": round(score,1),
                    "distance_meters": total_dist,
                    "time_minutes": travel_time,
                    "avg_safety_factor": round(1.0+avg_roughness,2),
                    "route_segments": path_segments,
                    "clinic_idx": clinic_idx,
                    "lat": t_lat,
                    "lon": t_lon
                })
            except Exception as e:
                print(f"Routing error for candidate: {e}")
                continue

        # Deduplicate and pick top 3
        results.sort(key=lambda x:x['score'], reverse=True)
        dedup_results = {}
        top_results = []
        for res in results:
            clinic_row = self.clinics_gdf.loc[res['clinic_idx']]
            name = str(clinic_row.get('name','Unknown')).strip()
            if name not in dedup_results:
                dedup_results[name] = res
                if len(dedup_results) >=3:
                    break
        for name,res in dedup_results.items():
            clinic_row = self.clinics_gdf.loc[res['clinic_idx']]
            segments_geojson = []
            for seg in res['route_segments']:
                segments_geojson.append({
                    "geometry": LineString(seg['coords']).__geo_interface__,
                    "properties":{"safety":round(seg['safety'],2)}
                })
            top_results.append({
                "score": res['score'],
                "distance_meters": round(res['distance_meters'],2),
                "time_minutes": res['time_minutes'],
                "avg_safety_factor": res['avg_safety_factor'],
                "route_segments": segments_geojson,
                "destination": {
                    "name": str(clinic_row.get('name','Unknown')),
                    "opening_hours": str(clinic_row.get('opening_hours','None')),
                    "lat": res['lat'],
                    "lon": res['lon']
                }
            })
        return top_results
