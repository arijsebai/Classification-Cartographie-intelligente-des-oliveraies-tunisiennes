const map = L.map('map').setView([33.8869, 9.5375], 6); // Centered on Tunisia

L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap contributors & CARTO',
    subdomains: 'abcd',
    maxZoom: 20
}).addTo(map);

// FeatureGroup to store editable layers
const drawnItems = new L.FeatureGroup();
map.addLayer(drawnItems);

// Results feature group
const resultsGroup = new L.FeatureGroup();
map.addLayer(resultsGroup);

const drawControl = new L.Control.Draw({
    draw: {
        polyline: false,
        marker: false,
        circlemarker: false,
        circle: false,
        polygon: {
            allowIntersection: false,
            drawError: { color: '#e1e100', message: '<strong>Oh snap!<strong> you can\'t draw that!' },
            shapeOptions: { color: '#3b82f6' }
        },
        rectangle: { shapeOptions: { color: '#3b82f6' } }
    },
    edit: { featureGroup: drawnItems }
});
map.addControl(drawControl);

let currentPolygon = null;
let currentGeoJsonBlob = null;

map.on(L.Draw.Event.CREATED, function (e) {
    drawnItems.clearLayers();
    resultsGroup.clearLayers();
    const layer = e.layer;
    drawnItems.addLayer(layer);
    currentPolygon = layer.toGeoJSON();
    
    document.getElementById('btn-analyze').disabled = false;
    document.getElementById('loading-state').style.display = 'block';
    document.getElementById('results-state').style.display = 'none';
});

const getColor = (systeme) => {
    switch(systeme) {
        case 'extensif': return '#22c55e';
        case 'intensif': return '#eab308';
        case 'hyper_intensif': return '#ef4444';
        default: return '#3b82f6';
    }
};

document.getElementById('btn-analyze').addEventListener('click', async () => {
    if (!currentPolygon) return;

    const btn = document.getElementById('btn-analyze');
    const btnText = btn.querySelector('.btn-text');
    const spinner = btn.querySelector('.loader-spinner');
    
    btn.disabled = true;
    btnText.textContent = 'Analyse en cours...';
    spinner.style.display = 'block';

    try {
        const response = await fetch('http://127.0.0.1:5000/api/cartographier', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                polygone_perimetre: currentPolygon,
                date: new Date().toISOString().split('T')[0]
            })
        });

        if (!response.ok) throw new Error('API Error');
        
        const data = await response.json();
        const oliveraies = data.oliveraies;
        const stats = data.stats;

        // Clear previous
        resultsGroup.clearLayers();
        const geojsonFeatures = [];

        oliveraies.forEach(ol => {
            const feature = {
                type: "Feature",
                properties: {
                    systeme: ol.systeme,
                    confiance: ol.confiance,
                    surface_ha: ol.surface_ha
                },
                geometry: ol.polygone
            };
            geojsonFeatures.push(feature);

            L.geoJSON(feature, {
                style: {
                    color: getColor(ol.systeme),
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.5
                },
                onEachFeature: function (f, layer) {
                    layer.bindPopup(`
                        <strong>Système:</strong> ${f.properties.systeme}<br>
                        <strong>Surface:</strong> ${f.properties.surface_ha} ha<br>
                        <strong>Confiance:</strong> ${f.properties.confiance * 100}%
                    `);
                }
            }).addTo(resultsGroup);
        });

        // Fit map
        if (oliveraies.length > 0) {
            map.fitBounds(resultsGroup.getBounds(), { padding: [50, 50] });
        }

        // Update Stats
        document.getElementById('stat-total').textContent = stats.total_oliveraies;
        document.getElementById('stat-surface').textContent = stats.surface_totale_ha + ' ha';
        document.getElementById('stat-ext').textContent = stats.repartition.extensif;
        document.getElementById('stat-int').textContent = stats.repartition.intensif;
        document.getElementById('stat-hyper').textContent = stats.repartition.hyper_intensif;

        document.getElementById('loading-state').style.display = 'none';
        document.getElementById('results-state').style.display = 'block';

        // Prepare blob for download GeoJSON
        const featureCollection = { type: "FeatureCollection", features: geojsonFeatures };
        const blob = new Blob([JSON.stringify(featureCollection, null, 2)], { type: 'application/json' });
        currentGeoJsonBlob = window.URL.createObjectURL(blob);

    } catch (error) {
        console.error(error);
        alert("Erreur lors de l'analyse.");
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Analyser cette zone';
        spinner.style.display = 'none';
    }
});

document.getElementById('btn-download').addEventListener('click', () => {
    if (!currentGeoJsonBlob) return;
    const a = document.createElement('a');
    a.href = currentGeoJsonBlob;
    a.download = 'oliveraies_detectees.geojson';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
});
