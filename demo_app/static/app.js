let maxPolygonKm2 = 100;
let drawnLayer = null;

const areaValue = document.getElementById("areaValue");
const limitValue = document.getElementById("limitValue");
const statusBox = document.getElementById("status");

const map = L.map("map", {
  zoomControl: true,
  preferCanvas: true,
});

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap",
}).addTo(map);

const drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

const drawControl = new L.Control.Draw({
  draw: {
    marker: false,
    circle: false,
    circlemarker: false,
    polyline: false,
    rectangle: false,
    polygon: {
      allowIntersection: false,
      showArea: true,
      shapeOptions: {
        color: "#f97316",
        fillColor: "#f97316",
        fillOpacity: 0.22,
        weight: 3,
      },
    },
  },
  edit: {
    featureGroup: drawnItems,
    remove: true,
  },
});
map.addControl(drawControl);

function setStatus(text, state = "neutral") {
  statusBox.textContent = text;
  statusBox.className = `status ${state}`;
}

function formatArea(areaKm2, areaHa) {
  if (areaKm2 >= 1) return `${areaKm2.toFixed(2)} km²`;
  return `${areaHa.toFixed(1)} ha`;
}

function polygonAreaM2(latlngs) {
  const ring = latlngs[0] || latlngs;
  return Math.abs(L.GeometryUtil.geodesicArea(ring));
}

function styleParcel(feature) {
  const system = feature.properties?.cultivation_system;
  const color = system === "intensif" ? "#0f8b8d" : "#7a8f22";
  return {
    color,
    fillColor: color,
    fillOpacity: 0.18,
    weight: 1.5,
  };
}

function parcelPopup(feature) {
  const props = feature.properties || {};
  return `
    <strong>${props.name || props.id}</strong><br>
    ${props.cultivation_system || "système inconnu"}<br>
    ${props.governorate || ""}<br>
    ${(props.area_ha || 0).toFixed ? props.area_ha.toFixed(1) : props.area_ha} ha
  `;
}

async function loadConfig() {
  const response = await fetch("/api/config");
  const config = await response.json();
  maxPolygonKm2 = config.max_polygon_km2;
  limitValue.textContent = `${maxPolygonKm2} km²`;
  map.setView([config.default_center.lat, config.default_center.lng], config.default_zoom);
}

async function loadParcels() {
  const response = await fetch("/api/parcels");
  const geojson = await response.json();
  const layer = L.geoJSON(geojson, {
    style: styleParcel,
    onEachFeature: (feature, layer) => layer.bindPopup(parcelPopup(feature)),
  }).addTo(map);

  if (geojson.features?.length) {
    map.fitBounds(layer.getBounds(), { padding: [24, 24] });
  }
}

async function analyzeLayer(layer) {
  const latlngs = layer.getLatLngs();
  const areaM2 = polygonAreaM2(latlngs);
  const areaKm2 = areaM2 / 1_000_000;
  const areaHa = areaM2 / 10_000;
  areaValue.textContent = formatArea(areaKm2, areaHa);

  if (areaKm2 > maxPolygonKm2) {
    layer.setStyle({ color: "#b42318", fillColor: "#b42318" });
    setStatus(`Polygone refusé: ${areaKm2.toFixed(2)} km² dépasse la limite de ${maxPolygonKm2} km².`, "bad");
    return;
  }

  layer.setStyle({ color: "#f97316", fillColor: "#f97316" });
  setStatus("Validation serveur en cours...", "neutral");

  const geometry = layer.toGeoJSON().geometry;
  const response = await fetch("/api/analyze-polygon", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ geometry }),
  });

  const payload = await response.json();
  if (!response.ok) {
    const detail = payload.detail || {};
    setStatus(`${detail.message || "Polygone refusé"} Surface: ${detail.area_km2 || areaKm2.toFixed(2)} km².`, "bad");
    return;
  }

  setStatus(`Polygone accepté: ${payload.area_ha} ha. Prêt pour inférence démo.`, "ok");
}

map.on(L.Draw.Event.CREATED, async (event) => {
  if (drawnLayer) drawnItems.removeLayer(drawnLayer);
  drawnLayer = event.layer;
  drawnItems.addLayer(drawnLayer);
  await analyzeLayer(drawnLayer);
});

map.on(L.Draw.Event.EDITED, async (event) => {
  event.layers.eachLayer(async (layer) => {
    drawnLayer = layer;
    await analyzeLayer(layer);
  });
});

map.on(L.Draw.Event.DELETED, () => {
  drawnLayer = null;
  areaValue.textContent = "0 ha";
  setStatus("Dessine un polygone sur la carte.", "neutral");
});

loadConfig()
  .then(loadParcels)
  .catch((error) => {
    console.error(error);
    setStatus("Impossible de charger la démo.", "bad");
  });
