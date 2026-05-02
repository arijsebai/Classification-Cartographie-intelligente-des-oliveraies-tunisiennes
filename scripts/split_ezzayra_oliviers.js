const fs = require("fs");
const path = require("path");

const GOVERNORATE_CENTERS = {
  Tunis: [36.8065, 10.1815],
  Ariana: [36.8625, 10.1956],
  Ben_Arous: [36.7531, 10.2189],
  Manouba: [36.8093, 10.0863],
  Nabeul: [36.4513, 10.7352],
  Zaghouan: [36.4029, 10.1429],
  Bizerte: [37.2744, 9.8739],
  Beja: [36.7256, 9.1817],
  Jendouba: [36.5011, 8.7802],
  Kef: [36.1822, 8.7148],
  Siliana: [36.0833, 9.3667],
  Sousse: [35.8256, 10.6084],
  Monastir: [35.7643, 10.8113],
  Mahdia: [35.5047, 11.0622],
  Sfax: [34.7406, 10.7603],
  Kairouan: [35.6781, 10.0963],
  Kasserine: [35.1676, 8.8365],
  Sidi_Bouzid: [35.0382, 9.4858],
  Gabes: [33.8815, 10.0982],
  Medenine: [33.3549, 10.5055],
  Tataouine: [32.9297, 10.4518],
  Gafsa: [34.425, 8.7842],
  Tozeur: [33.9197, 8.1335],
  Kebili: [33.7044, 8.969],
};

const RATIOS = { train: 0.7, val: 0.15, test: 0.15 };
const SPLITS = Object.keys(RATIOS);

function parseArgs(argv) {
  const args = {
    input: path.resolve(__dirname, ".."),
    output: path.resolve(__dirname, "..", "data_splits", "ezzayra_oliviers"),
    seed: "ezzayra-oliviers-v1",
    gridDeg: 0.2,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--input") args.input = path.resolve(argv[++i]);
    else if (arg === "--output") args.output = path.resolve(argv[++i]);
    else if (arg === "--seed") args.seed = argv[++i];
    else if (arg === "--grid-deg") args.gridDeg = Number(argv[++i]);
    else if (arg === "--help") {
      console.log("Usage: node scripts/split_ezzayra_oliviers.js [--input DIR] [--output DIR] [--seed TEXT] [--grid-deg N]");
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!Number.isFinite(args.gridDeg) || args.gridDeg <= 0) {
    throw new Error("--grid-deg must be a positive number");
  }

  return args;
}

function hashString(value) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function mulberry32(seed) {
  return function random() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function centroid(coordinates) {
  if (!Array.isArray(coordinates) || coordinates.length === 0) {
    throw new Error("Parcel has no coordinates");
  }

  const sums = coordinates.reduce(
    (acc, point) => {
      acc.lat += Number(point.lat);
      acc.lng += Number(point.lng);
      return acc;
    },
    { lat: 0, lng: 0 },
  );

  return {
    lat: sums.lat / coordinates.length,
    lng: sums.lng / coordinates.length,
  };
}

function nearestGovernorate(point) {
  let bestName = null;
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const [name, center] of Object.entries(GOVERNORATE_CENTERS)) {
    const distance = (point.lat - center[0]) ** 2 + (point.lng - center[1]) ** 2;
    if (distance < bestDistance) {
      bestName = name;
      bestDistance = distance;
    }
  }

  return bestName;
}

function inferCultivationSystem(fileName) {
  const lower = fileName.toLowerCase();
  if (lower.includes("extensif")) return "extensif";
  if (lower.includes("intensif")) return "intensif";
  if (lower.includes("hyper")) return "hyper_intensif";
  return "unknown";
}

function zoneId(governorate, point, gridDeg) {
  const latCell = Math.floor(point.lat / gridDeg);
  const lngCell = Math.floor(point.lng / gridDeg);
  return `${governorate}__lat${latCell}_lng${lngCell}`;
}

function loadParcels(inputDir, gridDeg) {
  const files = fs
    .readdirSync(inputDir)
    .filter((file) => file.toLowerCase().endsWith(".json"))
    .sort();

  return files.flatMap((file) => {
    const sourcePath = path.join(inputDir, file);
    const data = JSON.parse(fs.readFileSync(sourcePath, "utf8"));
    const cultivationSystem = inferCultivationSystem(file);

    if (!Array.isArray(data.parcels)) {
      throw new Error(`${file} does not contain a parcels array`);
    }

    return data.parcels.map((parcel) => {
      const center = centroid(parcel.coordinates);
      const governorate = nearestGovernorate(center);
      const zone = zoneId(governorate, center, gridDeg);

      return {
        ...parcel,
        cultivation_system: cultivationSystem,
        source_file: file,
        centroid: center,
        governorate,
        zone_id: zone,
      };
    });
  });
}

function countBy(items, keyFn) {
  return items.reduce((counts, item) => {
    const key = keyFn(item);
    counts[key] = (counts[key] || 0) + 1;
    return counts;
  }, {});
}

function makeEmptyStats() {
  return Object.fromEntries(SPLITS.map((split) => [split, { parcels: [] }]));
}

function scoreAssignment(stats, totals, splitName, group) {
  const simulated = SPLITS.reduce((acc, split) => {
    acc[split] = stats[split].parcels.length + (split === splitName ? group.length : 0);
    return acc;
  }, {});

  let score = 0;
  for (const split of SPLITS) {
    const expected = totals.all * RATIOS[split];
    score += Math.abs(simulated[split] - expected) * 3;
  }

  const dimensions = [
    ["system", (parcel) => parcel.cultivation_system],
    ["governorate", (parcel) => parcel.governorate],
  ];

  for (const [, keyFn] of dimensions) {
    const totalCounts = countBy(totals.parcels, keyFn);
    for (const key of Object.keys(totalCounts)) {
      for (const split of SPLITS) {
        const splitItems = stats[split].parcels.concat(split === splitName ? group : []);
        const actual = splitItems.filter((parcel) => keyFn(parcel) === key).length;
        const expected = totalCounts[key] * RATIOS[split];
        score += Math.abs(actual - expected);
      }
    }
  }

  return score;
}

function splitGroups(parcels, seed) {
  const groups = new Map();
  for (const parcel of parcels) {
    if (!groups.has(parcel.zone_id)) groups.set(parcel.zone_id, []);
    groups.get(parcel.zone_id).push(parcel);
  }

  const random = mulberry32(hashString(seed));
  const orderedGroups = [...groups.values()]
    .map((group) => ({ group, tieBreaker: random() }))
    .sort((a, b) => b.group.length - a.group.length || a.tieBreaker - b.tieBreaker)
    .map((entry) => entry.group);

  const stats = makeEmptyStats();
  const totals = { all: parcels.length, parcels };

  for (const group of orderedGroups) {
    const bestSplit = SPLITS.map((split) => ({
      split,
      score: scoreAssignment(stats, totals, split, group),
    })).sort((a, b) => a.score - b.score || SPLITS.indexOf(a.split) - SPLITS.indexOf(b.split))[0].split;

    stats[bestSplit].parcels.push(...group);
  }

  return stats;
}

function summarize(parcelsBySplit, args) {
  const allParcels = SPLITS.flatMap((split) => parcelsBySplit[split].parcels);
  const zoneSplits = {};
  for (const split of SPLITS) {
    for (const parcel of parcelsBySplit[split].parcels) {
      zoneSplits[parcel.zone_id] = zoneSplits[parcel.zone_id] || new Set();
      zoneSplits[parcel.zone_id].add(split);
    }
  }

  const leakingZones = Object.entries(zoneSplits)
    .filter(([, splitSet]) => splitSet.size > 1)
    .map(([zone]) => zone);

  return {
    generated_at: new Date().toISOString(),
    seed: args.seed,
    ratios: RATIOS,
    zone_grid_degrees: args.gridDeg,
    governorate_method: "nearest Tunisian governorate center from parcel centroid",
    total_parcels: allParcels.length,
    total_zones: Object.keys(zoneSplits).length,
    leaking_zones: leakingZones,
    splits: Object.fromEntries(
      SPLITS.map((split) => {
        const parcels = parcelsBySplit[split].parcels;
        return [
          split,
          {
            parcel_count: parcels.length,
            zone_count: new Set(parcels.map((parcel) => parcel.zone_id)).size,
            by_cultivation_system: countBy(parcels, (parcel) => parcel.cultivation_system),
            by_governorate: countBy(parcels, (parcel) => parcel.governorate),
          },
        ];
      }),
    ),
  };
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function csvEscape(value) {
  const text = String(value ?? "");
  if (!/[",\n]/.test(text)) return text;
  return `"${text.replace(/"/g, '""')}"`;
}

function writeOutputs(parcelsBySplit, args, summary) {
  fs.mkdirSync(args.output, { recursive: true });

  for (const split of SPLITS) {
    const parcels = parcelsBySplit[split].parcels.sort((a, b) => a.id.localeCompare(b.id));
    writeJson(path.join(args.output, `${split}.json`), {
      generated_at: summary.generated_at,
      split,
      parcel_count: parcels.length,
      parcels,
    });

    const bySource = parcels.reduce((acc, parcel) => {
      acc[parcel.source_file] = acc[parcel.source_file] || [];
      acc[parcel.source_file].push(parcel);
      return acc;
    }, {});

    for (const [sourceFile, sourceParcels] of Object.entries(bySource)) {
      writeJson(path.join(args.output, "by_source", split, sourceFile), {
        generated_at: summary.generated_at,
        split,
        source_file: sourceFile,
        parcel_count: sourceParcels.length,
        parcels: sourceParcels,
      });
    }
  }

  writeJson(path.join(args.output, "split_summary.json"), summary);

  const rows = [
    ["split", "id", "source_file", "cultivation_system", "governorate", "zone_id", "centroid_lat", "centroid_lng", "area_ha"],
  ];
  for (const split of SPLITS) {
    for (const parcel of parcelsBySplit[split].parcels) {
      rows.push([
        split,
        parcel.id,
        parcel.source_file,
        parcel.cultivation_system,
        parcel.governorate,
        parcel.zone_id,
        parcel.centroid.lat,
        parcel.centroid.lng,
        parcel.area_ha,
      ]);
    }
  }

  fs.writeFileSync(
    path.join(args.output, "manifest.csv"),
    `${rows.map((row) => row.map(csvEscape).join(",")).join("\n")}\n`,
    "utf8",
  );
}

function main() {
  const args = parseArgs(process.argv);
  const parcels = loadParcels(args.input, args.gridDeg);
  const parcelsBySplit = splitGroups(parcels, args.seed);
  const summary = summarize(parcelsBySplit, args);

  if (summary.leaking_zones.length > 0) {
    throw new Error(`Spatial leakage detected in zones: ${summary.leaking_zones.join(", ")}`);
  }

  writeOutputs(parcelsBySplit, args, summary);
  console.log(JSON.stringify(summary, null, 2));
}

main();
