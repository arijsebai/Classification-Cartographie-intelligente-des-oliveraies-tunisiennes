import json
from shapely.geometry import Polygon
import geopandas as gpd
from sklearn.model_selection import train_test_split
import pystac_client
import planetary_computer
import os

S2_L2A_BANDS = ("B02", "B03", "B04", "B08", "B11")

def load_ezzayra_json(filepath, label):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = []
    for parcel in data.get('parcels', []):
        coords = parcel.get('coordinates', [])
        # The JSON uses {lat, lng}, Shapely needs (lng, lat)
        polyg_coords = [(pt['lng'], pt['lat']) for pt in coords]
        
        # Valid polygon needs at least 3 points
        if len(polyg_coords) >= 3:
            geom = Polygon(polyg_coords)
            features.append({
                'id': parcel.get('id'),
                'name': parcel.get('name'),
                'area_ha': parcel.get('area_ha'),
                'systeme': label, # 'extensif' or 'intensif's
                'geometry': geom
            })
    return features

def split_stratified_dataset(gdf):
    from sklearn.cluster import KMeans
    from sklearn.model_selection import GroupShuffleSplit
    import numpy as np

    print(f"Total parcels: {len(gdf)}")
    
    # Étape 1 : Créer des zones spatiales (pseudo-gouvernorats) via clustering sur les centroïdes
    # La Tunisie a 24 gouvernorats, on crée 24 clusters pour regrouper spatialement les oliveraies
    centroids = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf.geometry])
    kmeans = KMeans(n_clusters=24, random_state=42, n_init=10).fit(centroids)
    gdf['zone'] = kmeans.labels_
    
    # Étape 2 : GroupShuffleSplit pour ne jamais mélanger les polygones d'une même 'zone'
    # 70% Train, 30% Temp (qui sera divisé en 15% / 15%)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42)
    train_idx, temp_idx = next(gss1.split(gdf, groups=gdf['zone']))
    
    train = gdf.iloc[train_idx]
    temp = gdf.iloc[temp_idx]
    
    # 50% de temp pour Val, 50% de temp pour Test (ce qui fera ~15% de test global)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42)
    val_idx, test_idx = next(gss2.split(temp, groups=temp['zone']))
    
    val = temp.iloc[val_idx]
    test = temp.iloc[test_idx]
    
    print(f"Train: {len(train)} parcels, Val: {len(val)} parcels, Test: {len(test)} parcels")
    return train, val, test

def download_sentinel2_data(parcel_id, polygon_geom_4326, date_range="2025-05-01/2025-06-30"):
    """
    Télécharge toutes les acquisitions Sentinel-2 L2A (B2, B3, B4, B8, B11)
    sur la période mai-juin en filtrant les nuages via la bande SCL.
    """
    import rasterio
    from rasterio.mask import mask
    import numpy as np
    from rasterio.warp import transform_geom
    from rasterio.warp import reproject, Resampling
    
    print(f"[{parcel_id}] Recherche d'images Sentinel-2 pour la période: {date_range}")
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    import time
    
    items = []
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"[{parcel_id}] Recherche d'images Sentinel-2 pour la période: {date_range} (Essai {attempt+1}/{max_retries})")
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                intersects=polygon_geom_4326.__geo_interface__,
                datetime=date_range,
                limit=100
            )
            items = list(search.items())
            break  # Succès, on sort de la boucle
        except Exception as e:
            print(f"[{parcel_id}] Timeout API. Nouvelle tentative dans 5 secondes...")
            time.sleep(5)
            if attempt == max_retries - 1:
                raise Exception(f"L'API Sentinel-2 est injoignable après {max_retries} tentatives: {e}")

    if not items:
        print(f"[{parcel_id}] Aucune image S2 trouvée sur {date_range} dans cette zone.")
        return False
        
    out_dir = os.path.join("sentinel2_data", str(parcel_id))
    os.makedirs(out_dir, exist_ok=True)
    
    bands_to_dl = {**{b: b for b in S2_L2A_BANDS}, "SCL": "SCL"}
    items = sorted(
        items,
        key=lambda i: (i.properties.get("datetime", ""), i.properties.get("eo:cloud_cover", 100))
    )
    print(f"[{parcel_id}] {len(items)} acquisitions trouvées sur {date_range}.")

    downloaded_count = 0

    for item in items:
        acquisition_date = item.properties.get("datetime", "")[:10].replace("-", "")
        if not acquisition_date:
            acquisition_date = item.id.split("_")[2][:8] if len(item.id.split("_")) > 2 else item.id

        print(
            f"[{parcel_id}] Acquisition {item.id} "
            f"(date={acquisition_date}, cloud_cover={item.properties.get('eo:cloud_cover')}%)"
        )

        scl_image = None
        scl_meta = None
        geom_proj = {}

        for band_name, asset_key in bands_to_dl.items():
            if asset_key not in item.assets:
                print(f"[{parcel_id}] Asset manquant pour {acquisition_date}: {asset_key}.")
                continue

            href = item.assets[asset_key].href
            try:
                with rasterio.open(href) as src:
                    if getattr(src, "crs", None) is not None:
                        src_crs = src.crs
                        if src_crs.to_string() not in geom_proj:
                            projected_geom = transform_geom("EPSG:4326", src_crs, polygon_geom_4326.__geo_interface__)
                            geom_proj[src_crs.to_string()] = projected_geom
                        poly_local = geom_proj[src_crs.to_string()]
                    else:
                        poly_local = polygon_geom_4326.__geo_interface__

                    out_image, out_transform = mask(src, [poly_local], crop=True)
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                    })

                    out_path = os.path.join(out_dir, f"{acquisition_date}_{band_name}.tif")
                    with rasterio.open(out_path, "w", **out_meta) as dest:
                        dest.write(out_image)

                    if band_name == "SCL":
                        scl_image = out_image
                        scl_meta = out_meta

                    print(f"[{parcel_id}] A téléchargé {acquisition_date}_{band_name}.tif")
            except Exception as e:
                print(f"[{parcel_id}] Erreur sur {acquisition_date}_{band_name}: {e}")

        if scl_image is not None:
            print(f"[{parcel_id}] Application du masque SCL pour l'acquisition {acquisition_date}.")
            bad_scl = np.isin(scl_image[0], [0, 1, 3, 8, 9, 10, 11])

            scl_h, scl_w = bad_scl.shape
            scl_transform = scl_meta["transform"] if scl_meta else None
            scl_crs = scl_meta.get("crs") if scl_meta else None

            for band_name in S2_L2A_BANDS:
                band_path = os.path.join(out_dir, f"{acquisition_date}_{band_name}.tif")
                if os.path.exists(band_path):
                    with rasterio.open(band_path, "r+") as src:
                        arr = src.read(1)

                        if scl_transform is not None and scl_crs is not None and src.transform is not None and src.crs is not None:
                            dst_mask = np.zeros(arr.shape, dtype=np.uint8)
                            reproject(
                                source=bad_scl.astype(np.uint8),
                                destination=dst_mask,
                                src_transform=scl_transform,
                                src_crs=scl_crs,
                                dst_transform=src.transform,
                                dst_crs=src.crs,
                                resampling=Resampling.nearest,
                            )
                            bad_mask_aligned = dst_mask.astype(bool)
                        else:
                            if arr.shape == (scl_h, scl_w):
                                bad_mask_aligned = bad_scl
                            else:
                                y_idx = (np.linspace(0, scl_h - 1, arr.shape[0])).astype(int)
                                x_idx = (np.linspace(0, scl_w - 1, arr.shape[1])).astype(int)
                                bad_mask_aligned = bad_scl[np.ix_(y_idx, x_idx)]

                        nodata = src.nodata if src.nodata is not None else 0
                        arr = arr.astype(src.dtypes[0], copy=False)
                        arr[bad_mask_aligned] = nodata
                        src.write(arr, 1)

            print(f"[{parcel_id}] Masque SCL appliqué pour {acquisition_date}.")

        downloaded_count += 1

    print(f"[{parcel_id}] {downloaded_count} acquisitions traitées.")
    return downloaded_count > 0


if __name__ == "__main__":
    extensif_path = "Oliviers/parcelles_OlivierExtensif.json"
    intensif_path = "Oliviers/parcellesOliviersIntensifs.json"
    
    features = []
    if os.path.exists(extensif_path):
        features.extend(load_ezzayra_json(extensif_path, 'extensif'))
    if os.path.exists(intensif_path):
        features.extend(load_ezzayra_json(intensif_path, 'intensif'))
        
    if features:
        gdf = gpd.GeoDataFrame(features, geometry='geometry', crs="EPSG:4326")
        
        # Etape 1: Splitting train/val/test
        train, val, test = split_stratified_dataset(gdf)
        
        # Save to file to simulate the split dataset needed for model training
        os.makedirs("split_data", exist_ok=True)
        train.to_file("split_data/train.geojson", driver="GeoJSON")
        val.to_file("split_data/val.geojson", driver="GeoJSON")
        test.to_file("split_data/test.geojson", driver="GeoJSON")
        
        print("Data successfully mapped and split to train/val/test geojsons.")
        
        # Execute Sentinel-2 Download for ALL parcels (polygones) — période mai-juin par défaut
        print("Lancement de l'Étage 1 : Téléchargement RÉEL des géotiffs Sentinel-2 L2A (mai-juin) + masque SCL...")
        for _, row in gdf.iterrows():
            download_sentinel2_data(row["id"], row["geometry"], date_range="2025-05-01/2025-06-30")
        
        print("Fin du script.")
