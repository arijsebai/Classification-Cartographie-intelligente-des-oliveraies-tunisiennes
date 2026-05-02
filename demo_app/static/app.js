let maxPolygonKm2 = 100;
let drawnLayer = null;
let currentGeometry = null;
let currentOpenEOJobId = null;
let openEOPollTimer = null;
let autoAnalysisRunning = false;

const areaValue = document.getElementById("areaValue");
const limitValue = document.getElementById("limitValue");
const statusBox = document.getElementById("status");
const modelStatusBox = document.getElementById("modelStatus");
const sentinelButton = document.getElementById("sentinelButton");
const classifyButton = document.getElementById("classifyButton");
const openeoPanel = document.getElementById("openeoPanel");
const openeoJob = document.getElementById("openeoJob");
const openeoLog = document.getElementById("openeoLog");
const runOpenEOButton = document.getElementById("runOpenEOButton");

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

function setSentinelReady(isReady) {
  if (!isReady) {
    currentGeometry = null;
    currentOpenEOJobId = null;
    openeoPanel.classList.add("hidden");
    openeoLog.textContent = "";
  }
  sentinelButton.disabled = !isReady;
  classifyButton.disabled = !isReady;
}

function setOpenEOJob(jobId) {
  currentOpenEOJobId = jobId;
  openeoPanel.classList.remove("hidden");
  openeoJob.textContent = jobId;
  openeoLog.textContent = "Demande prete. Lance les etapes openEO depuis ces boutons.";
}

async function refreshOpenEOStatus(action) {
  if (!currentOpenEOJobId) return null;
  const response = await fetch(`/api/openeo/${currentOpenEOJobId}/${action}`);
  const payload = await response.json();
  if (!response.ok) {
    openeoLog.textContent = payload.detail || "Impossible de lire le statut openEO.";
    return null;
  }

  const logLines = [
    `Action: ${payload.action}`,
    `Statut: ${payload.status}`,
    `GeoJSON: ${payload.geojson}`,
    `Manifest: ${payload.manifest}`,
    `Log: ${payload.log}`,
  ];

  if (payload.analysis_result) {
    const resultSummary = classificationSummary(payload.analysis_result, "Resultat final");
    logLines.push("", resultSummary);
    if (payload.result_source) logLines.push(`Source: ${payload.result_source}`);
    if (payload.result_file) logLines.push(`Fichier: ${payload.result_file}`);
    if (payload.analysis_completed_at) logLines.push(`Termine le: ${payload.analysis_completed_at}`);
    setStatus(resultSummary, "ok");
  } else if (payload.analysis_error) {
    logLines.push("", `Erreur: ${payload.analysis_error}`);
    setStatus(payload.analysis_error, "warn");
  } else {
    logLines.push("", payload.log_tail || "En attente de sortie...");
  }

  openeoLog.textContent = logLines.join("\n");
  return payload;
}

function pollOpenEO(action) {
  if (openEOPollTimer) clearInterval(openEOPollTimer);
  openEOPollTimer = setInterval(async () => {
    const payload = await refreshOpenEOStatus(action);
    if (payload && payload.status !== "running") {
      clearInterval(openEOPollTimer);
      openEOPollTimer = null;
    }
  }, 3000);
}

async function runOpenEOWorkflow() {
  if (!currentOpenEOJobId) return;
  runOpenEOButton.disabled = true;
  openeoLog.textContent = "Workflow openEO lance en arriere-plan...";
  try {
    const action = "full-workflow";
    const response = await fetch(`/api/openeo/${currentOpenEOJobId}/${action}`, { method: "POST" });
    const payload = await response.json();
    if (!response.ok) {
      openeoLog.textContent = payload.detail || "Action openEO impossible.";
      return;
    }
    await refreshOpenEOStatus(action);
    pollOpenEO(action);
  } finally {
    runOpenEOButton.disabled = false;
  }
}

function formatArea(areaKm2, areaHa) {
  if (areaKm2 >= 1) return `${areaKm2.toFixed(2)} km2`;
  return `${areaHa.toFixed(1)} ha`;
}

function polygonAreaM2(latlngs) {
  const ring = latlngs[0] || latlngs;
  return Math.abs(L.GeometryUtil.geodesicArea(ring));
}

function classificationSummary(cls, prefix = "Classification") {
  const confidence = Math.round((cls.confidence || 0) * 100);
  const prob = cls.prob_intensif === undefined || cls.prob_intensif === null
    ? ""
    : ` P(intensif): ${Math.round(cls.prob_intensif * 100)}%.`;
  return `${prefix}: ${cls.label} (${confidence}%).${prob}`;
}

function styleParcel(feature) {
  const system = feature.properties?.model_prediction || feature.properties?.cultivation_system;
  const color = system === "intensif" ? "#0f8b8d" : "#7a8f22";
  return {
    color,
    fillColor: color,
    fillOpacity: 0.18,
    weight: 1.5,
  };
}

function styleSentinelFootprint() {
  return {
    color: "#2563eb",
    fillColor: "#2563eb",
    fillOpacity: 0.06,
    weight: 1,
    dashArray: "5 5",
  };
}

function parcelPopup(feature) {
  const props = feature.properties || {};
  const predicted = props.model_prediction || "non disponible";
  const truth = props.cultivation_system || "inconnu";
  const prob = props.prob_intensif === null || props.prob_intensif === undefined
    ? ""
    : `<br>P(intensif): ${Math.round(props.prob_intensif * 100)}%`;
  const area = Number(props.area_ha || 0);

  return `
    <strong>${props.name || props.id}</strong><br>
    Verite terrain: ${truth}<br>
    Prediction modele: <strong>${predicted}</strong>${prob}<br>
    Source: ${props.model_source || "label terrain"}<br>
    ${props.governorate || ""}<br>
    ${area.toFixed(1)} ha
  `;
}

async function loadConfig() {
  const response = await fetch("/api/config");
  const config = await response.json();
  maxPolygonKm2 = config.max_polygon_km2;
  limitValue.textContent = `${maxPolygonKm2} km2`;
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

async function loadSentinelFootprints() {
  const response = await fetch("/api/sentinel-footprints");
  const geojson = await response.json();
  L.geoJSON(geojson, {
    style: styleSentinelFootprint,
    interactive: false,
  }).addTo(map);
}

async function loadModelSummary() {
  const response = await fetch("/api/model-summary");
  const summary = await response.json();
  if (!summary.available) {
    modelStatusBox.textContent = "Modele offline non disponible.";
    modelStatusBox.className = "status warn";
    return;
  }

  const f1 = summary.spatial_cv_macro_f1 === null || summary.spatial_cv_macro_f1 === undefined
    ? "n/a"
    : summary.spatial_cv_macro_f1.toFixed(3);
  modelStatusBox.textContent = `Modele reel: ${summary.model} CV spatiale ${summary.group_column}, macro-F1 ${f1}.`;
  modelStatusBox.className = "status ok";
}

async function analyzeLayer(layer) {
  const latlngs = layer.getLatLngs();
  const areaM2 = polygonAreaM2(latlngs);
  const areaKm2 = areaM2 / 1_000_000;
  const areaHa = areaM2 / 10_000;
  areaValue.textContent = formatArea(areaKm2, areaHa);

  if (areaKm2 > maxPolygonKm2) {
    layer.setStyle({ color: "#b42318", fillColor: "#b42318" });
    setStatus(`Polygone refuse: ${areaKm2.toFixed(2)} km2 depasse la limite de ${maxPolygonKm2} km2.`, "bad");
    setSentinelReady(false);
    return;
  }

  layer.setStyle({ color: "#f97316", fillColor: "#f97316" });
  setStatus("Validation serveur en cours...", "neutral");
  setSentinelReady(false);

  const geometry = layer.toGeoJSON().geometry;
  const response = await fetch("/api/analyze-polygon", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ geometry }),
  });

  const payload = await response.json();
  if (!response.ok) {
    const detail = payload.detail || {};
    setStatus(`${detail.message || "Polygone refuse"} Surface: ${detail.area_km2 || areaKm2.toFixed(2)} km2.`, "bad");
    setSentinelReady(false);
    return;
  }

  currentGeometry = geometry;
  sentinelButton.disabled = false;
  classifyButton.disabled = false;

  setStatus(
    `Polygone accepte: ${payload.area_ha} ha. Recherche d'image Sentinel-2 locale...`,
    "ok"
  );

  await runAutomaticAnalysis();
}

async function submitSentinelAnalysis({ silent = false } = {}) {
  if (!currentGeometry) return null;

  sentinelButton.disabled = true;
  sentinelButton.textContent = "Soumission...";

  try {
    const response = await fetch("/api/submit-sentinel-analysis", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ geometry: currentGeometry }),
    });

    const payload = await response.json();
    if (!response.ok) {
      const detail = payload.detail || {};
      if (!silent) {
        setStatus(`${detail.message || "Soumission Sentinel-2 refusee"} Surface: ${detail.area_km2 || "n/a"} km2.`, "bad");
      }
      return null;
    }

    if (!silent) {
      setOpenEOJob(payload.job_id);
      setStatus(
        `Demande openEO creee: ${payload.job_id}. Surface: ${payload.area_ha} ha. Clique sur Lancer openEO en arriere-plan.`,
        "warn"
      );
    }
    return payload;
  } catch (error) {
    console.error(error);
    if (!silent) {
      setStatus("Impossible de soumettre le polygone a l'analyse Sentinel-2.", "bad");
    }
    return null;
  } finally {
    sentinelButton.textContent = "Soumettre Sentinel-2";
    sentinelButton.disabled = !currentGeometry;
  }
}

async function classifyWithLocalSentinel({ fallbackToQueue = false } = {}) {
  if (!currentGeometry) return;

  classifyButton.disabled = true;
  classifyButton.textContent = "Analyse...";
  setStatus("Extraction des pixels Sentinel-2 locaux et classification en cours...", "neutral");

  try {
    const response = await fetch("/api/classify-sentinel-local", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ geometry: currentGeometry }),
    });

    const payload = await response.json();
    if (!response.ok) {
      const detail = typeof payload.detail === "string"
        ? payload.detail
        : payload.detail?.message || "Sentinel-2 local indisponible pour ce polygone.";
      if (fallbackToQueue) {
        const queued = await submitSentinelAnalysis({ silent: true });
        if (queued) {
          setOpenEOJob(queued.job_id);
          setStatus(
            `Aucune image Sentinel-2 locale ne couvre ce polygone. Demande openEO creee: ${queued.job_id}. Lancement du workflow en arriere-plan...`,
            "warn"
          );
          await runOpenEOWorkflow();
        } else {
          setStatus("Aucune image Sentinel-2 locale ne couvre ce polygone. La demande openEO n'a pas pu etre creee.", "bad");
        }
      } else {
        setStatus(`${detail} Lance l'extraction openEO pour obtenir la vraie classification.`, "warn");
      }
      return;
    }

    const cls = payload.classification;
    const summary = classificationSummary(cls, "Classification Sentinel-2");
    setStatus(
      `${summary} Pixels valides: ${cls.valid_pixel_count}. Image: ${cls.image_path}.`,
      "ok"
    );
  } catch (error) {
    console.error(error);
    setStatus("Impossible de classifier avec Sentinel-2 local.", "bad");
  } finally {
    classifyButton.textContent = "Analyse automatique";
    classifyButton.disabled = !currentGeometry;
  }
}

async function runAutomaticAnalysis() {
  if (autoAnalysisRunning || !currentGeometry) return;

  autoAnalysisRunning = true;
  try {
    await classifyWithLocalSentinel({ fallbackToQueue: true });
  } finally {
    autoAnalysisRunning = false;
  }
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
  currentGeometry = null;
  areaValue.textContent = "0 ha";
  setSentinelReady(false);
  setStatus("Dessine un polygone sur la carte.", "neutral");
});

sentinelButton.addEventListener("click", submitSentinelAnalysis);
classifyButton.addEventListener("click", runAutomaticAnalysis);
runOpenEOButton.addEventListener("click", runOpenEOWorkflow);

loadConfig()
  .then(loadModelSummary)
  .then(loadParcels)
  .then(loadSentinelFootprints)
  .catch((error) => {
    console.error(error);
    setStatus("Impossible de charger la demo.", "bad");
  });
