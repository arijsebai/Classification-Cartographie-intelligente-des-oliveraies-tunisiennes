/**
 * Cache Indicators for Demo Front-end
 * Shows which zones are pre-calculated and cached
 */

// Enhance parcel styling to show cache status
function styleParcelWithCacheIndicator(feature) {
  const system = feature.properties?.model_prediction || feature.properties?.cultivation_system;
  const color = system === "intensif" ? "#0f8b8d" : "#7a8f22";
  
  // Parcels with model_prediction are CACHED (pre-calculated)
  const inCache = feature.properties?.model_prediction !== undefined;
  
  return {
    // Always use the system color to keep parcels identifiable,
    // but visually differentiate cached vs non-cached via weight and dash.
    color: color,
    fillColor: color,
    fillOpacity: inCache ? 0.35 : 0.14,
    weight: inCache ? 3.5 : 2,
    dashArray: inCache ? "" : "4 2",
  };
}

// Enhanced popup showing cache status
function parcelPopupWithCacheStatus(feature) {
  const props = feature.properties || {};
  const predicted = props.model_prediction || "—";
  const truth = props.cultivation_system || "inconnu";
  const prob = props.prob_intensif === null || props.prob_intensif === undefined
    ? ""
    : `<br>Probabilité intensif: ${Math.round(props.prob_intensif * 100)}%`;
  const area = Number(props.area_ha || 0);
  
  // Cache status badge
  const inCache = props.model_prediction !== undefined;
  const cacheBadge = inCache 
    ? '<span style="background:#4ade80;color:white;padding:2px 6px;border-radius:3px;font-size:11px;font-weight:bold;">✓ En cache</span>'
    : '<span style="background:#ef4444;color:white;padding:2px 6px;border-radius:3px;font-size:11px;">? Non calculée</span>';

  return `
    <div style="font-size:12px;line-height:1.4;">
      <strong>${props.name || props.id}</strong><br>
      <div style="margin-top:4px;">${cacheBadge}</div>
      <br>
      Vérité terrain: <strong>${truth}</strong><br>
      Prédiction modèle: <strong>${predicted}</strong>${prob}<br>
      Source: ${props.model_source || "label terrain"}<br>
      ${props.governorate || ""}<br>
      <strong>${area.toFixed(1)} ha</strong>
    </div>
  `;
}

// Status message for drawn polygon
function getCacheStatusMessage(cacheHit) {
  if (cacheHit === undefined) return "";
  if (cacheHit) {
    return ' <span style="color:#4ade80;font-weight:bold;">✓ En cache</span>';
  } else {
    return ' <span style="color:#ef4444;">? Zone nouvelle</span>';
  }
}

// Create a cache indicator layer group
function createCacheIndicatorLayers(parcelGeoJSON) {
  const cachedParcels = [];
  const nonCachedParcels = [];
  
  for (const feature of parcelGeoJSON.features || []) {
    const hasModel = feature.properties?.model_prediction !== undefined;
    if (hasModel) {
      cachedParcels.push(feature);
    } else {
      nonCachedParcels.push(feature);
    }
  }
  
  return { cachedParcels, nonCachedParcels };
}

// Add cache-aware legend to map
function addCacheLegendToMap(map) {
  const legend = L.control({ position: "topleft" });
  
  legend.onAdd = () => {
    const div = L.DomUtil.create("div", "cache-legend");
    div.style.background = "white";
    div.style.padding = "10px";
    div.style.borderRadius = "5px";
    div.style.boxShadow = "0 0 15px rgba(0,0,0,0.2)";
    div.style.fontSize = "12px";
    div.style.lineHeight = "1.6";
    
    div.innerHTML = `
      <div><strong>Zones Précalculées</strong></div>
      <div style="margin:8px 0;">
        <span style="display:inline-block;width:20px;height:12px;background:#0f8b8d;border:2.5px solid #0f8b8d;margin-right:4px;"></span>
        <span>Intensif <strong>✓</strong></span>
      </div>
      <div>
        <span style="display:inline-block;width:20px;height:12px;background:#7a8f22;border:2.5px solid #7a8f22;margin-right:4px;"></span>
        <span>Extensif <strong>✓</strong></span>
      </div>
      <hr style="margin:8px 0;border:none;border-top:1px solid #ddd;">
      <div style="color:#888;font-size:11px;">
        <strong>Bordure:</strong> Épaisse = cache<br>
        <strong>Tiretée:</strong> Non calculée
      </div>
    `;
    
    return div;
  };
  
  legend.addTo(map);
}

// Tooltip showing cache status on hover
function addCacheStatusTooltip(map, parcelLayer) {
  parcelLayer.on("mouseover", (e) => {
    const props = e.target.feature.properties;
    const inCache = props.model_prediction !== undefined;
    const text = inCache ? "✓ Pré-calculée" : "? À calculer";
    
    e.target.bindTooltip(text, {
      permanent: false,
      direction: "top",
      offset: [0, -5],
      className: "cache-tooltip"
    }).openTooltip();
  });
}
