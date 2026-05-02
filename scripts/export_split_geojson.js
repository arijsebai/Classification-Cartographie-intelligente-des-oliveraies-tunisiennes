const fs = require("fs");
const path = require("path");

const SPLITS = ["train", "val", "test"];

function parseArgs(argv) {
  const args = {
    input: path.resolve(__dirname, "..", "data_splits", "ezzayra_oliviers"),
    output: path.resolve(__dirname, "..", "data_splits", "ezzayra_oliviers_geojson"),
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--input") args.input = path.resolve(argv[++i]);
    else if (arg === "--output") args.output = path.resolve(argv[++i]);
    else if (arg === "--help") {
      console.log("Usage: node scripts/export_split_geojson.js [--input DIR] [--output DIR]");
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function assertCoordinate(point, parcelId) {
  const lat = Number(point.lat);
  const lng = Number(point.lng);

  if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
    throw new Error(`Invalid coordinate in parcel ${parcelId}`);
  }

  if (lat < -90 || lat > 90 || lng < -180 || lng > 180) {
    throw new Error(`Coordinate out of range in parcel ${parcelId}: ${lat}, ${lng}`);
  }

  return [lng, lat];
}

function closeRing(ring) {
  const first = ring[0];
  const last = ring[ring.length - 1];
  if (first[0] === last[0] && first[1] === last[1]) return ring;
  return [...ring, first];
}

function bboxFromRing(ring) {
  return ring.reduce(
    (bbox, coordinate) => [
      Math.min(bbox[0], coordinate[0]),
      Math.min(bbox[1], coordinate[1]),
      Math.max(bbox[2], coordinate[0]),
      Math.max(bbox[3], coordinate[1]),
    ],
    [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY],
  );
}

function parcelToFeature(parcel, split) {
  if (!Array.isArray(parcel.coordinates) || parcel.coordinates.length < 3) {
    throw new Error(`Parcel ${parcel.id} must contain at least 3 coordinates`);
  }

  const ring = closeRing(parcel.coordinates.map((point) => assertCoordinate(point, parcel.id)));
  const bbox = bboxFromRing(ring);

  return {
    type: "Feature",
    id: parcel.id,
    bbox,
    properties: {
      id: parcel.id,
      name: parcel.name,
      split,
      crop: parcel.crop,
      cultivation_system: parcel.cultivation_system,
      source_file: parcel.source_file,
      area_ha: parcel.area_ha,
      governorate: parcel.governorate,
      zone_id: parcel.zone_id,
      centroid_lat: parcel.centroid?.lat,
      centroid_lng: parcel.centroid?.lng,
      created_at: parcel.created_at,
      modified_at: parcel.modified_at,
    },
    geometry: {
      type: "Polygon",
      coordinates: [ring],
    },
  };
}

function featureCollection(features, metadata = {}) {
  return {
    type: "FeatureCollection",
    ...metadata,
    feature_count: features.length,
    features,
  };
}

function main() {
  const args = parseArgs(process.argv);
  const allFeatures = [];
  const manifestRows = [["split", "id", "cultivation_system", "governorate", "zone_id", "area_ha", "bbox_min_lng", "bbox_min_lat", "bbox_max_lng", "bbox_max_lat"]];

  for (const split of SPLITS) {
    const splitPath = path.join(args.input, `${split}.json`);
    const data = readJson(splitPath);
    const features = data.parcels.map((parcel) => parcelToFeature(parcel, split));

    writeJson(
      path.join(args.output, `${split}.geojson`),
      featureCollection(features, {
        generated_at: new Date().toISOString(),
        source_split_file: splitPath,
        split,
      }),
    );

    for (const feature of features) {
      allFeatures.push(feature);
      manifestRows.push([
        split,
        feature.properties.id,
        feature.properties.cultivation_system,
        feature.properties.governorate,
        feature.properties.zone_id,
        feature.properties.area_ha,
        ...feature.bbox,
      ]);
    }
  }

  writeJson(
    path.join(args.output, "all_splits.geojson"),
    featureCollection(allFeatures, {
      generated_at: new Date().toISOString(),
      splits: SPLITS,
    }),
  );

  fs.writeFileSync(
    path.join(args.output, "geojson_manifest.csv"),
    `${manifestRows.map((row) => row.join(",")).join("\n")}\n`,
    "utf8",
  );

  console.log(
    JSON.stringify(
      {
        output: args.output,
        feature_count: allFeatures.length,
        files: SPLITS.map((split) => `${split}.geojson`).concat(["all_splits.geojson", "geojson_manifest.csv"]),
      },
      null,
      2,
    ),
  );
}

main();
