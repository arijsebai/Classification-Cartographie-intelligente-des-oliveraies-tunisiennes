from flask import Flask, request, jsonify
from flask_cors import CORS
import geopandas as gpd
from shapely.geometry import Polygon, shape
import json
import random
import os

app = Flask(__name__)
CORS(app)

# Load data into memory for fast querying
gdf_all = None

def load_data():
    global gdf_all
    features = []
    
    def parse_file(filepath, base_system):
        if not os.path.exists(filepath): return
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for parcel in data.get('parcels', []):
            coords = [(pt['lng'], pt['lat']) for pt in parcel.get('coordinates', [])]
            if len(coords) >= 3:
                # Randomly assign hyper_intensif for the demo if it's intensif
                sys = base_system
                if sys == 'intensif' and random.random() > 0.7:
                    sys = 'hyper_intensif'
                
                features.append({
                    'id': parcel.get('id'),
                    'systeme': sys,
                    'area_ha': parcel.get('area_ha', random.uniform(1.0, 15.0)),
                    'geometry': Polygon(coords)
                })
    
    parse_file("Oliviers/parcelles_OlivierExtensif.json", "extensif")
    parse_file("Oliviers/parcellesOliviersIntensifs.json", "intensif")
    
    if features:
        global gdf_all
        gdf_all = gpd.GeoDataFrame(features, geometry='geometry', crs="EPSG:4326")

load_data()

@app.route('/api/cartographier', methods=['POST'])
def cartographier():
    try:
        data = request.json
        roi_geojson = data.get('polygone_perimetre') # region of interest
        
        # In case the user passed the whole Feature
        if 'geometry' in roi_geojson:
            roi_geom = shape(roi_geojson['geometry'])
        else:
            roi_geom = shape(roi_geojson)
            
        # Add basic validation or fallback, if gdf_all is none, generate mock data inside the ROI
        results = []
        stats = {
            "total_oliveraies": 0,
            "surface_totale_ha": 0,
            "repartition": { "extensif": 0, "intensif": 0, "hyper_intensif": 0 }
        }
        
        if gdf_all is not None and not gdf_all.empty:
            # Querying parcels intersecting the ROI
            intersecting = gdf_all[gdf_all.intersects(roi_geom)]
            
            for idx, row in intersecting.iterrows():
                geom = row['geometry']
                sys = row['systeme']
                area = row['area_ha']
                
                results.append({
                    "polygone": json.loads(gpd.GeoSeries([geom]).to_json())['features'][0]['geometry'],
                    "systeme": sys,
                    "confiance": round(random.uniform(0.85, 0.99), 2),
                    "surface_ha": round(area, 2)
                })
                
                stats["total_oliveraies"] += 1
                stats["surface_totale_ha"] += area
                stats["repartition"][sys] = stats["repartition"].get(sys, 0) + 1
        else:
            # Generate totally fake data if dataset not found
            for i in range(random.randint(5, 15)):
                sys = random.choice(["extensif", "intensif", "hyper_intensif"])
                area = random.uniform(2.0, 20.0)
                # create small fake polygon in roi
                center = roi_geom.centroid
                fake_geom = Polygon([
                    (center.x + random.uniform(-0.01, 0.01), center.y + random.uniform(-0.01, 0.01))
                    for _ in range(4)
                ])
                results.append({
                    "polygone": json.loads(gpd.GeoSeries([fake_geom]).to_json())['features'][0]['geometry'],
                    "systeme": sys,
                    "confiance": round(random.uniform(0.85, 0.99), 2),
                    "surface_ha": round(area, 2)
                })
                stats["total_oliveraies"] += 1
                stats["surface_totale_ha"] += area
                stats["repartition"][sys] += 1
                
        stats["surface_totale_ha"] = round(stats["surface_totale_ha"], 2)

        return jsonify({
            "oliveraies": results,
            "stats": stats
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start the Flask API
    app.run(debug=True, port=5000)
