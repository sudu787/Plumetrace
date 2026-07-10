"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Map, {
  Layer,
  Marker,
  NavigationControl,
  Popup,
  Source,
  MapRef
} from "react-map-gl/maplibre";
import { Factory, Radio, Wind } from "lucide-react";
import type { Feature, FeatureCollection, LineString, Point, Polygon } from "geojson";
import type { StyleSpecification } from "maplibre-gl";

const MAPBOX_DARK_STYLE = "mapbox://styles/mapbox/dark-v11";
const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN?.trim() ?? "";
const HAS_MAPBOX_TOKEN = MAPBOX_TOKEN.length > 0;
const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  process.env.NEXT_PUBLIC_BACKEND_URL ||
  "http://localhost:8000"
).replace(/\/$/, "");
const SSE_URL = `${API_BASE_URL}/api/stream`;
const TOKENLESS_DARK_STYLE: StyleSpecification = {
  version: 8,
  sources: {
    "carto-dark": {
      type: "raster",
      tiles: [
        "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "https://d.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
      ],
      tileSize: 256,
      attribution: "Map tiles by CARTO, under CC BY 3.0. Data by OpenStreetMap, under ODbL."
    }
  },
  layers: [
    {
      id: "carto-dark-base",
      type: "raster",
      source: "carto-dark",
      minzoom: 0,
      maxzoom: 22
    }
  ]
};
const SO2_SAFE_LIMIT = 75;
const PM25_SAFE_LIMIT = 35;

interface FactoryFacility {
  factory_name: string;
  corporate_owner: string;
  zoning_permit_id: string;
  latitude: number;
  longitude: number;
  plot_radius_km: number;
}

const REGISTRY_FACTORIES: FactoryFacility[] = [
  {
    factory_name: "Apex Petrochemical Complex",
    corporate_owner: "Apex Industrial Holdings LLC",
    zoning_permit_id: "MUNI-IZ-2044-APX-17",
    latitude: 40.71345,
    longitude: -74.00765,
    plot_radius_km: 0.42
  },
  {
    factory_name: "Global Logistics Foundry",
    corporate_owner: "Harborline Materials Group",
    zoning_permit_id: "MUNI-IZ-2041-GLF-09",
    latitude: 40.71610,
    longitude: -74.00280,
    plot_radius_km: 0.36
  },
  {
    factory_name: "Northbank Solvents Terminal",
    corporate_owner: "CivicChem Transport Partners",
    zoning_permit_id: "MUNI-IZ-2039-NST-22",
    latitude: 40.71020,
    longitude: -74.01160,
    plot_radius_km: 0.31
  }
];

export interface TelemetryReading {
  id?: string;
  sensor_id: string;
  timestamp: string;
  latitude: number;
  longitude: number;
  pm25: number;
  so2: number;
  wind_speed: number;
  wind_direction: number;
}

interface SensorNode {
  sensor_id: string;
  label: string;
  latitude: number;
  longitude: number;
  pm25: number;
  so2: number;
  wind_speed: number;
  wind_direction: number;
  timestamp: string;
}

export interface HazardEvent {
  reading: TelemetryReading;
  suspectedSource: {
    latitude: number;
    longitude: number;
    confidence: number;
  };
  plume: Feature<Polygon>;
}

export interface AttributionSource {
  latitude: number;
  longitude: number;
  confidence: number;
}

interface MapEngineProps {
  attributedSource?: AttributionSource | null;
  onTelemetry?: (reading: TelemetryReading, severity: "normal" | "critical") => void;
  onHazard?: (event: HazardEvent) => void;
  onConnectionStateChange?: (state: "connecting" | "online" | "offline") => void;
  onEnforcementReport?: (reportData: any) => void;
}

const SENSOR_REGISTRY: SensorNode[] = [
  {
    sensor_id: "industrial_north",
    label: "Industrial North",
    latitude: 40.718,
    longitude: -74.006,
    pm25: 0,
    so2: 0,
    wind_speed: 0,
    wind_direction: 0,
    timestamp: ""
  },
  {
    sensor_id: "residential_east",
    label: "Residential East",
    latitude: 40.714,
    longitude: -73.998,
    pm25: 0,
    so2: 0,
    wind_speed: 0,
    wind_direction: 0,
    timestamp: ""
  },
  {
    sensor_id: "park_south",
    label: "Park South",
    latitude: 40.708,
    longitude: -74.004,
    pm25: 0,
    so2: 0,
    wind_speed: 0,
    wind_direction: 0,
    timestamp: ""
  },
  {
    sensor_id: "river_west",
    label: "River West",
    latitude: 40.712,
    longitude: -74.012,
    pm25: 0,
    so2: 0,
    wind_speed: 0,
    wind_direction: 0,
    timestamp: ""
  },
  {
    sensor_id: "downtown_center",
    label: "Downtown Center",
    latitude: 40.713,
    longitude: -74.008,
    pm25: 0,
    so2: 0,
    wind_speed: 0,
    wind_direction: 0,
    timestamp: ""
  },
  {
    sensor_id: "commercial_northeast",
    label: "Commercial Northeast",
    latitude: 40.716,
    longitude: -74.002,
    pm25: 0,
    so2: 0,
    wind_speed: 0,
    wind_direction: 0,
    timestamp: ""
  },
  {
    sensor_id: "highway_southeast",
    label: "Highway Southeast",
    latitude: 40.710,
    longitude: -74.000,
    pm25: 0,
    so2: 0,
    wind_speed: 0,
    wind_direction: 0,
    timestamp: ""
  },
  {
    sensor_id: "suburban_southwest",
    label: "Suburban Southwest",
    latitude: 40.709,
    longitude: -74.010,
    pm25: 0,
    so2: 0,
    wind_speed: 0,
    wind_direction: 0,
    timestamp: ""
  }
];

function isCritical(reading: Pick<TelemetryReading, "so2" | "pm25">): boolean {
  return reading.so2 >= SO2_SAFE_LIMIT || reading.pm25 >= PM25_SAFE_LIMIT;
}

function toFiniteNumber(value: unknown, fallback = 0): number {
  const numeric = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function normalizeTelemetry(payload: unknown): TelemetryReading | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const candidate = payload as Partial<TelemetryReading>;
  if (!candidate.sensor_id) {
    return null;
  }

  return {
    id: candidate.id,
    sensor_id: String(candidate.sensor_id),
    timestamp: candidate.timestamp ? String(candidate.timestamp) : new Date().toISOString(),
    latitude: toFiniteNumber(candidate.latitude),
    longitude: toFiniteNumber(candidate.longitude),
    pm25: toFiniteNumber(candidate.pm25),
    so2: toFiniteNumber(candidate.so2),
    wind_speed: toFiniteNumber(candidate.wind_speed),
    wind_direction: toFiniteNumber(candidate.wind_direction)
  };
}

function offsetCoordinate(latitude: number, longitude: number, bearingDegrees: number, distanceKm: number) {
  const bearing = (bearingDegrees * Math.PI) / 180;
  const deltaNorthKm = Math.cos(bearing) * distanceKm;
  const deltaEastKm = Math.sin(bearing) * distanceKm;
  const nextLatitude = latitude + deltaNorthKm / 111;
  const nextLongitude = longitude + deltaEastKm / (111 * Math.cos((latitude * Math.PI) / 180));

  return {
    latitude: nextLatitude,
    longitude: nextLongitude
  };
}

function getBearing(lat1: number, lon1: number, lat2: number, lon2: number) {
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const lat1Rad = (lat1 * Math.PI) / 180;
  const lat2Rad = (lat2 * Math.PI) / 180;
  const y = Math.sin(dLon) * Math.cos(lat2Rad);
  const x = Math.cos(lat1Rad) * Math.sin(lat2Rad) - Math.sin(lat1Rad) * Math.cos(lat2Rad) * Math.cos(dLon);
  const brng = (Math.atan2(y, x) * 180) / Math.PI;
  return (brng + 360) % 360;
}

function snapToNearestFactory(
  latitude: number,
  longitude: number,
  windDirection?: number,
  sensorLat?: number,
  sensorLon?: number
) {
  let bestFac = REGISTRY_FACTORIES[0];

  if (windDirection !== undefined && sensorLat !== undefined && sensorLon !== undefined) {
    let bestScore = -Infinity;
    const upstreamBearing = windDirection % 360;
    const severityFactor = 1.0;
    const plumeLengthKm = 0.48 + severityFactor * 0.22;
    const rawSource = offsetCoordinate(sensorLat, sensorLon, upstreamBearing, plumeLengthKm);

    REGISTRY_FACTORIES.forEach((fac) => {
      // Physical distance from sensor to factory
      const distToSensor = Math.sqrt(Math.pow(fac.latitude - sensorLat, 2) + Math.pow(fac.longitude - sensorLon, 2)) * 111;
      
      // If the factory is within 150 meters, it is the direct candidate!
      if (distToSensor < 0.15) {
        bestFac = fac;
        bestScore = 99999;
        return;
      }

      // Distance from raw offset to factory
      const distToOffset = Math.sqrt(Math.pow(fac.latitude - rawSource.latitude, 2) + Math.pow(fac.longitude - rawSource.longitude, 2)) * 111;
      
      // Bearing alignment (factory to sensor should match downwind direction)
      const facToSensorBearing = getBearing(fac.latitude, fac.longitude, sensorLat, sensorLon);
      const downwindBearing = (windDirection + 180) % 360;
      let angleDiff = Math.abs(facToSensorBearing - downwindBearing);
      if (angleDiff > 180) angleDiff = 360 - angleDiff;

      // Penalize angle mismatch and offset distance
      const score = -(angleDiff * 8) - (distToOffset * 50);
      if (score > bestScore) {
        bestScore = score;
        bestFac = fac;
      }
    });
  } else {
    // Normal haversine/Euclidean snap (for post-attribution PINN output)
    let minDistance = Infinity;
    REGISTRY_FACTORIES.forEach((fac) => {
      const d = Math.pow(fac.latitude - latitude, 2) + Math.pow(fac.longitude - longitude, 2);
      if (d < minDistance) {
        minDistance = d;
        bestFac = fac;
      }
    });
  }

  return {
    latitude: bestFac.latitude,
    longitude: bestFac.longitude
  };
}

function buildPlumeFeature(
  reading: TelemetryReading,
  pulseScale: number,
  customSource?: { latitude: number; longitude: number; confidence?: number } | null
): {
  polygon: Feature<Polygon>;
  centerline: Feature<LineString>;
  origin: Feature<Point>;
  suspectedSource: HazardEvent["suspectedSource"];
} {
  const severityFactor = Math.min(2.4, Math.max(reading.so2 / SO2_SAFE_LIMIT, reading.pm25 / PM25_SAFE_LIMIT, 1));
  const plumeLengthKm = 0.48 + severityFactor * 0.22;
  const plumeWidthKm = (0.12 + severityFactor * 0.04) * pulseScale;
  const upstreamBearing = reading.wind_direction % 360;

  let sourceCoord;
  if (customSource) {
    sourceCoord = snapToNearestFactory(customSource.latitude, customSource.longitude);
  } else {
    sourceCoord = snapToNearestFactory(
      reading.latitude,
      reading.longitude,
      reading.wind_direction,
      reading.latitude,
      reading.longitude
    );
  }

  const left = offsetCoordinate(sourceCoord.latitude, sourceCoord.longitude, upstreamBearing - 90, plumeWidthKm);
  const right = offsetCoordinate(sourceCoord.latitude, sourceCoord.longitude, upstreamBearing + 90, plumeWidthKm);
  const sensorLeft = offsetCoordinate(reading.latitude, reading.longitude, upstreamBearing - 90, plumeWidthKm * 0.18);
  const sensorRight = offsetCoordinate(reading.latitude, reading.longitude, upstreamBearing + 90, plumeWidthKm * 0.18);

  const polygon: Feature<Polygon> = {
    type: "Feature",
    properties: {
      sensor_id: reading.sensor_id,
      so2: reading.so2,
      pm25: reading.pm25,
      pulseScale
    },
    geometry: {
      type: "Polygon",
      coordinates: [
        [
          [sensorLeft.longitude, sensorLeft.latitude],
          [left.longitude, left.latitude],
          [sourceCoord.longitude, sourceCoord.latitude],
          [right.longitude, right.latitude],
          [sensorRight.longitude, sensorRight.latitude],
          [sensorLeft.longitude, sensorLeft.latitude]
        ]
      ]
    }
  };

  const centerline: Feature<LineString> = {
    type: "Feature",
    properties: {
      sensor_id: reading.sensor_id
    },
    geometry: {
      type: "LineString",
      coordinates: [
        [reading.longitude, reading.latitude],
        [sourceCoord.longitude, sourceCoord.latitude]
      ]
    }
  };

  const origin: Feature<Point> = {
    type: "Feature",
    properties: {
      sensor_id: reading.sensor_id
    },
    geometry: {
      type: "Point",
      coordinates: [sourceCoord.longitude, sourceCoord.latitude]
    }
  };

  return {
    polygon,
    centerline,
    origin,
    suspectedSource: {
      latitude: sourceCoord.latitude,
      longitude: sourceCoord.longitude,
      confidence: customSource && customSource.confidence !== undefined ? customSource.confidence : Math.min(0.97, 0.61 + severityFactor * 0.12)
    }
  };
}

function SensorMarker({ node, selected }: { node: SensorNode; selected: boolean }) {
  const critical = isCritical(node);
  const fill = critical ? "#f43f5e" : "#10b981";
  const halo = critical ? "#fecdd3" : "#a7f3d0";

  return (
    <div className="relative h-12 w-12 cursor-pointer flex items-center justify-center">
      <span
        className={`sensor-marker-pulse absolute left-2 top-2 h-8 w-8 rounded-full ${critical ? "bg-rose-500/45" : "bg-emerald-400/35"}`}
      />
      <svg
        viewBox="0 0 48 48"
        className={`relative h-12 w-12 drop-shadow-2xl ${critical ? "sensor-marker-warning" : ""}`}
        role="img"
        aria-label={`${node.label} sensor marker`}
      >
        <circle cx="24" cy="24" r={selected ? "15" : "13"} fill="rgba(2,6,23,0.92)" stroke={halo} strokeWidth="2" />
        <path
          d="M24 13.5c-5.25 0-9.5 4.25-9.5 9.5 0 7.2 9.5 13.5 9.5 13.5s9.5-6.3 9.5-13.5c0-5.25-4.25-9.5-9.5-9.5Zm0 13.4a3.9 3.9 0 1 1 0-7.8 3.9 3.9 0 0 1 0 7.8Z"
          fill={fill}
        />
      </svg>
      
      {/* Wind direction indicator arrow pointing downwind (flow direction) */}
      {node.wind_speed > 0 && (
        <div
          className="absolute pointer-events-none text-cyan-300"
          style={{
            transform: `rotate(${node.wind_direction + 180}deg) translateY(-22px)`,
            transition: "transform 0.5s ease-in-out"
          }}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="3.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-3.5 w-3.5 filter drop-shadow-[0_0_3px_rgba(34,211,238,0.7)]"
          >
            <line x1="12" y1="19" x2="12" y2="5"></line>
            <polyline points="5 12 12 5 19 12"></polyline>
          </svg>
        </div>
      )}
    </div>
  );
}

function SourceMarker() {
  return (
    <div className="source-marker-shell relative flex h-16 w-16 items-center justify-center">
      <span className="source-marker-wave absolute h-16 w-16 rounded-full border border-rose-200/45" />
      <span className="source-marker-wave source-marker-wave-delay absolute h-16 w-16 rounded-full border border-rose-300/35" />
      <div className="source-marker-core flex h-12 w-12 items-center justify-center rounded-full border border-rose-300/70 bg-rose-950/85 text-rose-100 shadow-plume-red backdrop-blur">
        <Factory className="h-5 w-5" />
      </div>
    </div>
  );
}

export default function MapEngine({ attributedSource, onTelemetry, onHazard, onConnectionStateChange, onEnforcementReport }: MapEngineProps) {
  const mapRef = useRef<MapRef>(null);
  const [sensorMap, setSensorMap] = useState<Record<string, SensorNode>>(() =>
    Object.fromEntries(SENSOR_REGISTRY.map((node) => [node.sensor_id, node]))
  );
  const [selectedSensorId, setSelectedSensorId] = useState<string | null>(SENSOR_REGISTRY[0]?.sensor_id ?? null);
  const [latestHazardReading, setLatestHazardReading] = useState<TelemetryReading | null>(null);

  const fitToPlume = useCallback((sensorLat: number, sensorLon: number, sourceLat: number, sourceLon: number) => {
    if (!mapRef.current) return;
    const minLat = Math.min(sensorLat, sourceLat);
    const maxLat = Math.max(sensorLat, sourceLat);
    const minLon = Math.min(sensorLon, sourceLon);
    const maxLon = Math.max(sensorLon, sourceLon);

    mapRef.current.fitBounds(
      [minLon, minLat, maxLon, maxLat],
      {
        padding: { top: 120, bottom: 120, left: 120, right: 120 },
        duration: 1200,
        maxZoom: 15.5
      }
    );
  }, []);

  useEffect(() => {
    if (latestHazardReading) {
      const source = attributedSource || buildPlumeFeature(latestHazardReading, 1, null).suspectedSource;
      fitToPlume(latestHazardReading.latitude, latestHazardReading.longitude, source.latitude, source.longitude);
    }
  }, [latestHazardReading, attributedSource, fitToPlume]);

  useEffect(() => {
    onConnectionStateChange?.("connecting");
    const apiKey = process.env.NEXT_PUBLIC_API_KEY || "dev-insecure-key";
    const eventSource = new EventSource(`${SSE_URL}?api_key=${apiKey}`);

    eventSource.onopen = () => {
      onConnectionStateChange?.("online");
    };

    eventSource.onmessage = (event: MessageEvent<string>) => {
      try {
        const parsed = JSON.parse(event.data) as any;
        const records = Array.isArray(parsed) ? parsed : [parsed];
        records.forEach((record) => {
          if (record.type === "enforcement_report") {
            onEnforcementReport?.(record);
            return;
          }
          const reading = normalizeTelemetry(record);
          if (!reading) {
            return;
          }

          const severity = isCritical(reading) ? "critical" : "normal";
          setSensorMap((current) => {
            const previous = current[reading.sensor_id];
            return {
              ...current,
              [reading.sensor_id]: {
                sensor_id: reading.sensor_id,
                label: previous?.label ?? reading.sensor_id.replaceAll("_", " "),
                latitude: reading.latitude,
                longitude: reading.longitude,
                pm25: reading.pm25,
                so2: reading.so2,
                wind_speed: reading.wind_speed,
                wind_direction: reading.wind_direction,
                timestamp: reading.timestamp
              }
            };
          });
          onTelemetry?.(reading, severity);

          if (severity === "critical") {
            const plume = buildPlumeFeature(reading, 1);
            setLatestHazardReading(reading);
            onHazard?.({
              reading,
              suspectedSource: plume.suspectedSource,
              plume: plume.polygon
            });
          }
        });
      } catch (error) {
        console.error("Unable to parse PlumeTrace telemetry payload", error);
      }
    };

    eventSource.onerror = () => {
      onConnectionStateChange?.("offline");
    };

    return () => {
      eventSource.close();
    };
  }, [onConnectionStateChange, onHazard, onTelemetry]);

  const sensors = useMemo(() => Object.values(sensorMap), [sensorMap]);
  const selectedSensor = selectedSensorId ? sensorMap[selectedSensorId] : null;

  const plumeData = useMemo<FeatureCollection>(() => {
    if (!latestHazardReading) {
      return {
        type: "FeatureCollection",
        features: []
      };
    }

    const plume = buildPlumeFeature(latestHazardReading, 1.1, attributedSource);
    return {
      type: "FeatureCollection",
      features: [plume.polygon, plume.centerline, plume.origin]
    };
  }, [latestHazardReading, attributedSource]);

  const heatmapData = useMemo<FeatureCollection<Point>>(
    () => ({
      type: "FeatureCollection",
      features: sensors
        .filter((sensor) => sensor.so2 > 0 || sensor.pm25 > 0)
        .map((sensor) => ({
          type: "Feature",
          properties: {
            intensity: Math.max(sensor.so2 / SO2_SAFE_LIMIT, sensor.pm25 / PM25_SAFE_LIMIT)
          },
          geometry: {
            type: "Point",
            coordinates: [sensor.longitude, sensor.latitude]
          }
        }))
    }),
    [sensors]
  );

  const plumeOrigin = useMemo(() => {
    if (!latestHazardReading) {
      return null;
    }
    return buildPlumeFeature(latestHazardReading, 1, attributedSource).suspectedSource;
  }, [attributedSource, latestHazardReading]);

  const handleMarkerClick = useCallback((sensorId: string) => {
    setSelectedSensorId(sensorId);
  }, []);

  return (
    <div className="relative h-full w-full">
      <Map
        ref={mapRef}
        mapStyle={HAS_MAPBOX_TOKEN ? MAPBOX_DARK_STYLE : TOKENLESS_DARK_STYLE}
        initialViewState={{
          latitude: 40.7138,
          longitude: -74.006,
          zoom: 13.4,
          pitch: 48,
          bearing: -18
        }}
        minZoom={11}
        maxZoom={18}
        attributionControl={false}
        style={{ height: "100%", width: "100%" }}
      >
        <NavigationControl position="top-right" visualizePitch />

        <Source id="plumetrace-hazard-heatmap" type="geojson" data={heatmapData}>
          <Layer
            id="hazard-heatmap-layer"
            type="heatmap"
            paint={{
              "heatmap-weight": ["interpolate", ["linear"], ["get", "intensity"], 0, 0, 1, 0.55, 3, 1],
              "heatmap-intensity": 1.2,
              "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 11, 24, 15, 62],
              "heatmap-opacity": 0.48,
              "heatmap-color": [
                "interpolate",
                ["linear"],
                ["heatmap-density"],
                0,
                "rgba(16,185,129,0)",
                0.25,
                "rgba(16,185,129,0.28)",
                0.55,
                "rgba(251,191,36,0.48)",
                0.82,
                "rgba(244,63,94,0.62)",
                1,
                "rgba(127,29,29,0.82)"
              ]
            }}
          />
        </Source>

        <Source id="plumetrace-plume-source" type="geojson" data={plumeData}>
          <Layer
            id="plume-fill-layer"
            type="fill"
            filter={["==", ["geometry-type"], "Polygon"]}
            paint={{
              "fill-color": "rgba(248,113,113,0.46)",
              "fill-opacity": ["interpolate", ["linear"], ["get", "pulseScale"], 1, 0.28, 1.42, 0.12]
            }}
          />
          <Layer
            id="plume-centerline-layer"
            type="line"
            filter={["==", ["geometry-type"], "LineString"]}
            paint={{
              "line-color": "rgba(254,202,202,0.95)",
              "line-width": 2,
              "line-dasharray": [2, 1.2]
            }}
          />
        </Source>

        {sensors.map((sensor) => (
          <Marker
            key={sensor.sensor_id}
            latitude={sensor.latitude}
            longitude={sensor.longitude}
            anchor="center"
            onClick={(event) => {
              event.originalEvent.stopPropagation();
              handleMarkerClick(sensor.sensor_id);
            }}
          >
            <SensorMarker node={sensor} selected={selectedSensorId === sensor.sensor_id} />
          </Marker>
        ))}

        {REGISTRY_FACTORIES.map((fac) => {
          const isActive = plumeOrigin && 
            Math.abs(fac.latitude - plumeOrigin.latitude) < 0.0001 && 
            Math.abs(fac.longitude - plumeOrigin.longitude) < 0.0001;
          
          if (isActive) return null;

          return (
            <Marker key={fac.factory_name} latitude={fac.latitude} longitude={fac.longitude} anchor="center">
              <div
                title={`${fac.factory_name} (${fac.corporate_owner})`}
                className="flex h-8 w-8 items-center justify-center rounded-full border border-slate-800 bg-slate-950/80 text-slate-400 hover:text-cyan-300 hover:border-cyan-500/50 shadow-lg backdrop-blur opacity-65 hover:opacity-100 hover:scale-105 transition-all cursor-pointer"
              >
                <Factory className="h-4 w-4" />
              </div>
            </Marker>
          );
        })}

        {plumeOrigin ? (
          <Marker latitude={plumeOrigin.latitude} longitude={plumeOrigin.longitude} anchor="center">
            <SourceMarker />
          </Marker>
        ) : null}

        {selectedSensor ? (
          <Popup
            latitude={selectedSensor.latitude}
            longitude={selectedSensor.longitude}
            anchor="bottom"
            closeButton={false}
            offset={28}
            onClose={() => setSelectedSensorId(null)}
          >
            <div className="w-72 p-4">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-300">Sensor Node</p>
                  <h3 className="mt-1 text-base font-semibold text-white">{selectedSensor.label}</h3>
                </div>
                <div className={`rounded-md px-2 py-1 text-xs font-semibold ${isCritical(selectedSensor) ? "bg-rose-500/20 text-rose-200" : "bg-emerald-500/15 text-emerald-200"}`}>
                  {isCritical(selectedSensor) ? "hazard" : "nominal"}
                </div>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-2">
                <Metric label="PM2.5" value={selectedSensor.pm25.toFixed(1)} unit="ug/m3" critical={selectedSensor.pm25 >= PM25_SAFE_LIMIT} />
                <Metric label="SO2" value={selectedSensor.so2.toFixed(1)} unit="ppb" critical={selectedSensor.so2 >= SO2_SAFE_LIMIT} />
                <Metric label="Wind" value={selectedSensor.wind_speed.toFixed(1)} unit="m/s" />
                <Metric label="Vector" value={selectedSensor.wind_direction.toFixed(0)} unit="deg" rotation={selectedSensor.wind_direction + 180} />
              </div>

              <div className="mt-4 flex items-center gap-2 border-t border-slate-800 pt-3 text-xs text-slate-400">
                <Wind className="h-4 w-4 text-cyan-300" />
                <span>{selectedSensor.timestamp || "awaiting live packet"}</span>
              </div>
            </div>
          </Popup>
        ) : null}
      </Map>

      <div className="pointer-events-none absolute inset-x-5 top-5 flex items-center justify-between">
        <div className="rounded-lg border border-slate-800/70 bg-slate-950/75 px-4 py-3 shadow-2xl backdrop-blur-xl">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-emerald-300/30 bg-emerald-400/10 text-emerald-200">
              <Radio className="h-4 w-4" />
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Live Sector</p>
              <p className="text-sm font-medium text-white">Industrial waterfront grid</p>
            </div>
          </div>
        </div>
        {!HAS_MAPBOX_TOKEN ? (
          <div className="rounded-lg border border-amber-400/30 bg-amber-950/40 px-4 py-3 text-xs font-medium text-amber-100 shadow-2xl backdrop-blur-xl">
            Tokenless basemap fallback active
          </div>
        ) : null}
      </div>
    </div>
  );
}

function Metric({ label, value, unit, critical = false, rotation }: { label: string; value: string; unit: string; critical?: boolean; rotation?: number }) {
  return (
    <div className={`rounded-md border px-3 py-2 ${critical ? "border-rose-400/30 bg-rose-500/10" : "border-slate-800 bg-slate-900/60"}`}>
      <p className="text-[11px] uppercase tracking-wide text-slate-400">{label}</p>
      <div className="mt-1 flex items-center justify-between">
        <p className={`text-sm font-semibold ${critical ? "text-rose-200" : "text-slate-100"}`}>
          {value} <span className="text-xs font-normal text-slate-500">{unit}</span>
        </p>
        {rotation !== undefined && (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-4 w-4 text-cyan-300 transition-transform duration-500"
            style={{ transform: `rotate(${rotation}deg)` }}
          >
            <line x1="12" y1="19" x2="12" y2="5"></line>
            <polyline points="5 12 12 5 19 12"></polyline>
          </svg>
        )}
      </div>
    </div>
  );
}
