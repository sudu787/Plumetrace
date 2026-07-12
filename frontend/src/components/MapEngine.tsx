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
import { Factory, Radio, Wind, Play, Pause, RotateCcw, History } from "lucide-react";
import type { Feature, FeatureCollection, LineString, Point, Polygon } from "geojson";
import type { StyleSpecification } from "maplibre-gl";
import { SENSOR_REGISTRY, type SensorNode } from "../constants/sensorRegistry";
import {
  SO2_SAFE_LIMIT,
  PM25_SAFE_LIMIT,
  isCritical,
  normalizeTelemetry,
  buildPlumeFeature,
  offsetCoordinate,
  type FactoryFacility,
  type TelemetryReading,
  type HazardEvent
} from "../utils/snappingUtils";

const MAPBOX_DARK_STYLE = "mapbox://styles/mapbox/dark-v11";
const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN?.trim() ?? "";
const HAS_MAPBOX_TOKEN = MAPBOX_TOKEN.length > 0;
const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  process.env.NEXT_PUBLIC_BACKEND_URL ||
  "http://127.0.0.1:8000"
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
  const [registryFactories, setRegistryFactories] = useState<FactoryFacility[]>([]);
  const registryFactoriesRef = useRef<FactoryFacility[]>([]);

  // Simulator States
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulatedPlume, setSimulatedPlume] = useState<FeatureCollection | null>(null);
  const [simOrigin, setSimOrigin] = useState<{ latitude: number; longitude: number } | null>(null);
  
  const [simFactoryName, setSimFactoryName] = useState("");
  const [simWindDirection, setSimWindDirection] = useState(270);
  const [simWindSpeed, setSimWindSpeed] = useState(5.0);
  const [simPollutant, setSimPollutant] = useState<"SO2" | "PM25">("SO2");
  const [simIntensity, setSimIntensity] = useState(150);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/v1/factories`)
      .then((res) => res.json())
      .then((data) => {
        setRegistryFactories(data);
        registryFactoriesRef.current = data;
        if (data.length > 0) {
          setSimFactoryName(data[0].factory_name);
        }
      })
      .catch((err) => console.error("Failed to load factory registry", err));
  }, []);

  const [sensorMap, setSensorMap] = useState<Record<string, SensorNode>>(() =>
    Object.fromEntries(SENSOR_REGISTRY.map((node) => [node.sensor_id, node]))
  );
  const [selectedSensorId, setSelectedSensorId] = useState<string | null>(SENSOR_REGISTRY[0]?.sensor_id ?? null);
  const [latestHazardReading, setLatestHazardReading] = useState<TelemetryReading | null>(null);

  const [isReplayMode, setIsReplayMode] = useState(false);
  const [replayHistory, setReplayHistory] = useState<TelemetryReading[]>([]);
  const [replayIndex, setReplayIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  // Playback loop
  useEffect(() => {
    let timer: NodeJS.Timeout | null = null;
    if (isPlaying && isReplayMode && replayHistory.length > 0) {
      timer = setInterval(() => {
        setReplayIndex((prev) => {
          if (prev >= replayHistory.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 1000);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isPlaying, isReplayMode, replayHistory]);

  const enterReplayMode = useCallback(async () => {
    if (replayHistory.length > 0) {
      setIsReplayMode(true);
      return;
    }
    setIsLoadingHistory(true);
    try {
      const apiKey = process.env.NEXT_PUBLIC_API_KEY || "dev-insecure-key";
      const response = await fetch(`${API_BASE_URL}/api/v1/sensors/history?api_key=${apiKey}`);
      if (!response.ok) throw new Error("History API failed");
      const rawData = await response.json();
      const history = rawData
        .map((r: any) => normalizeTelemetry(r))
        .filter(Boolean)
        .reverse() as TelemetryReading[];
      setReplayHistory(history);
      if (history.length > 0) {
        setReplayIndex(history.length - 1);
      }
      setIsReplayMode(true);
    } catch (err) {
      console.error("Failed to load sensor history", err);
    } finally {
      setIsLoadingHistory(false);
    }
  }, [replayHistory]);

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
    if (latestHazardReading && !isReplayMode) {
      const source = attributedSource || buildPlumeFeature(registryFactories, latestHazardReading, 1, null).suspectedSource;
      fitToPlume(latestHazardReading.latitude, latestHazardReading.longitude, source.latitude, source.longitude);
    }
  }, [latestHazardReading, attributedSource, fitToPlume, isReplayMode, registryFactories]);

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
            const plume = buildPlumeFeature(registryFactoriesRef.current, reading, 1);
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

  // Reconstruct sensor states up to replayIndex in replay mode
  const currentSensorMap = useMemo(() => {
    if (!isReplayMode || replayHistory.length === 0) return sensorMap;

    const baseMap = Object.fromEntries(
      SENSOR_REGISTRY.map((node) => [
        node.sensor_id,
        { ...node, pm25: 0, so2: 0, wind_speed: 0, wind_direction: 0, timestamp: "" }
      ])
    );

    for (let i = 0; i <= replayIndex; i++) {
      const reading = replayHistory[i];
      if (baseMap[reading.sensor_id]) {
        baseMap[reading.sensor_id] = {
          ...baseMap[reading.sensor_id],
          pm25: reading.pm25,
          so2: reading.so2,
          wind_speed: reading.wind_speed,
          wind_direction: reading.wind_direction,
          timestamp: reading.timestamp
        };
      }
    }
    return baseMap;
  }, [isReplayMode, replayHistory, replayIndex, sensorMap]);

  const sensors = useMemo(() => Object.values(currentSensorMap), [currentSensorMap]);
  const selectedSensor = selectedSensorId ? currentSensorMap[selectedSensorId] : null;

  // Resolve the active hazard reading for drawing plumes
  const activeHazardReading = useMemo(() => {
    if (!isReplayMode) return latestHazardReading;

    // Find the most recent critical reading up to replayIndex
    for (let i = replayIndex; i >= 0; i--) {
      const r = replayHistory[i];
      if (r.so2 >= SO2_SAFE_LIMIT || r.pm25 >= PM25_SAFE_LIMIT) {
        return r;
      }
    }
    return null;
  }, [isReplayMode, replayHistory, replayIndex, latestHazardReading]);

  const plumeData = useMemo<FeatureCollection>(() => {
    if (!activeHazardReading) {
      return {
        type: "FeatureCollection",
        features: []
      };
    }

    const plume = buildPlumeFeature(registryFactories, activeHazardReading, 1.1, attributedSource);
    return {
      type: "FeatureCollection",
      features: [plume.polygon, plume.centerline, plume.origin]
    };
  }, [activeHazardReading, attributedSource, registryFactories]);

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
    if (!activeHazardReading) {
      return null;
    }
    return buildPlumeFeature(registryFactories, activeHazardReading, 1, attributedSource).suspectedSource;
  }, [attributedSource, activeHazardReading, registryFactories]);

  const runSimulation = useCallback(() => {
    const selectedFac = registryFactories.find((f) => f.factory_name === simFactoryName);
    if (!selectedFac) return;

    const downwindBearing = (simWindDirection + 180) % 360;
    const plumeLengthKm = 0.45 + (simIntensity / 100) * 0.15;
    const mockSensorCoord = offsetCoordinate(
      selectedFac.latitude,
      selectedFac.longitude,
      downwindBearing,
      plumeLengthKm
    );

    const simReading: TelemetryReading = {
      sensor_id: "SIM_DOWNWIND",
      timestamp: new Date().toISOString(),
      latitude: mockSensorCoord.latitude,
      longitude: mockSensorCoord.longitude,
      pm25: simPollutant === "PM25" ? simIntensity : 0,
      so2: simPollutant === "SO2" ? simIntensity : 0,
      wind_speed: simWindSpeed,
      wind_direction: simWindDirection,
    };

    const pulseScale = 1.0;
    const plumeData = buildPlumeFeature(registryFactories, simReading, pulseScale, {
      latitude: selectedFac.latitude,
      longitude: selectedFac.longitude,
      confidence: 1.0,
    });

    const featureCol: FeatureCollection = {
      type: "FeatureCollection",
      features: [plumeData.polygon],
    };

    setSimulatedPlume(featureCol);
    setSimOrigin({ latitude: selectedFac.latitude, longitude: selectedFac.longitude });
  }, [registryFactories, simFactoryName, simWindDirection, simWindSpeed, simPollutant, simIntensity]);

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

        {!isSimulating && (
          <>
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
          </>
        )}

        {isSimulating && simulatedPlume && (
          <Source id="simulated-plume-source" type="geojson" data={simulatedPlume}>
            <Layer
              id="simulated-plume-layer"
              type="fill"
              filter={["==", ["geometry-type"], "Polygon"]}
              paint={{
                "fill-color": "rgba(6,182,212,0.48)",
                "fill-outline-color": "rgba(34,211,238,0.9)"
              }}
            />
          </Source>
        )}

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

        {registryFactories.map((fac) => {
          const isActive = plumeOrigin && 
            Math.abs(fac.latitude - plumeOrigin.latitude) < 0.0001 && 
            Math.abs(fac.longitude - plumeOrigin.longitude) < 0.0001;
          
          if (isActive) return null;

          return (
            <Marker key={fac.factory_name} latitude={fac.latitude} longitude={fac.longitude} anchor="center">
              <div className="flex flex-col items-center justify-center pointer-events-none group">
                <div
                  title={`${fac.factory_name} (${fac.corporate_owner})`}
                  className="flex h-8 w-8 items-center justify-center rounded-full border border-slate-800 bg-slate-950/80 text-slate-400 hover:text-cyan-300 hover:border-cyan-500/50 shadow-lg backdrop-blur opacity-65 group-hover:opacity-100 group-hover:scale-105 transition-all cursor-pointer pointer-events-auto"
                >
                  <Factory className="h-4 w-4" />
                </div>
                <span className="mt-1 px-1.5 py-0.5 rounded bg-slate-950/90 border border-slate-800/40 text-[9px] font-medium text-slate-400 shadow-md backdrop-blur select-none group-hover:text-cyan-300 group-hover:border-cyan-500/30 transition-all">
                  {fac.factory_name.replace(" Petrochemical Complex", "").replace(" Solvents Terminal", "").replace(" Logistics Foundry", "")}
                </span>
              </div>
            </Marker>
          );
        })}

        {!isSimulating && plumeOrigin ? (
          <Marker latitude={plumeOrigin.latitude} longitude={plumeOrigin.longitude} anchor="center">
            <div className="flex flex-col items-center justify-center">
              <SourceMarker />
              {(() => {
                const fac = registryFactories.find(f => 
                  Math.abs(f.latitude - plumeOrigin.latitude) < 0.0001 && 
                  Math.abs(f.longitude - plumeOrigin.longitude) < 0.0001
                );
                return fac ? (
                  <span className="mt-1.5 px-2 py-0.5 rounded bg-rose-950/95 border border-rose-500/30 text-[9px] font-bold text-rose-300 shadow-lg shadow-rose-950/50 backdrop-blur select-none animate-pulse">
                    {fac.factory_name}
                  </span>
                ) : null;
              })()}
            </div>
          </Marker>
        ) : null}

        {isSimulating && simOrigin && (
          <Marker latitude={simOrigin.latitude} longitude={simOrigin.longitude} anchor="center">
            <div className="flex flex-col items-center justify-center">
              <div className="source-marker-shell relative flex h-16 w-16 items-center justify-center">
                <span className="source-marker-wave absolute h-16 w-16 rounded-full border border-cyan-200/45 animate-ping" />
                <div className="source-marker-core flex h-12 w-12 items-center justify-center rounded-full border border-cyan-300/70 bg-cyan-950/85 text-cyan-100 shadow-[0_0_15px_rgba(6,182,212,0.7)] backdrop-blur">
                  <Factory className="h-5 w-5" />
                </div>
              </div>
              {(() => {
                const fac = registryFactories.find(f => 
                  Math.abs(f.latitude - simOrigin.latitude) < 0.0001 && 
                  Math.abs(f.longitude - simOrigin.longitude) < 0.0001
                );
                return fac ? (
                  <span className="mt-1.5 px-2 py-0.5 rounded bg-cyan-950/95 border border-cyan-500/30 text-[9px] font-bold text-cyan-300 shadow-lg shadow-cyan-950/50 backdrop-blur select-none animate-pulse">
                    {fac.factory_name}
                  </span>
                ) : null;
              })()}
            </div>
          </Marker>
        )}

        {selectedSensor ? (
          <Popup
            latitude={selectedSensor.latitude}
            longitude={selectedSensor.longitude}
            anchor="bottom"
            closeButton={false}
            offset={28}
            onClose={() => setSelectedSensorId(null)}
            maxWidth="none"
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

      {/* Replay Controls Panel */}
      <div className="absolute inset-x-5 bottom-5 z-20 flex flex-col gap-3 rounded-xl border border-slate-800/80 bg-slate-950/85 p-4 shadow-2xl backdrop-blur-xl transition-all duration-300">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="flex items-center gap-3">
            <button
              onClick={() => {
                if (isReplayMode) {
                  setIsReplayMode(false);
                  setIsPlaying(false);
                } else {
                  enterReplayMode();
                }
              }}
              className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-semibold uppercase tracking-wider transition-all duration-300 border pointer-events-auto cursor-pointer ${
                isReplayMode
                  ? "bg-cyan-500/20 border-cyan-400/50 text-cyan-300 shadow-md shadow-cyan-950/20"
                  : "bg-slate-900 border-slate-800 text-slate-400 hover:text-white hover:border-slate-700"
              }`}
            >
              <History className="h-3.5 w-3.5" />
              {isReplayMode ? "Replay Mode" : "Go to Replay"}
            </button>
            <button
              onClick={() => {
                setIsSimulating((prev) => !prev);
                if (!isSimulating) {
                  setIsReplayMode(false);
                  setIsPlaying(false);
                } else {
                  setSimulatedPlume(null);
                  setSimOrigin(null);
                }
              }}
              className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-semibold uppercase tracking-wider transition-all duration-300 border pointer-events-auto cursor-pointer ${
                isSimulating
                  ? "bg-cyan-500/20 border-cyan-400/50 text-cyan-300 shadow-md shadow-cyan-950/20"
                  : "bg-slate-900 border-slate-800 text-slate-400 hover:text-white hover:border-slate-700"
              }`}
            >
              <Radio className="h-3.5 w-3.5" />
              {isSimulating ? "Simulator Mode" : "Scenario Simulator"}
            </button>
            <div className="h-4 w-[1px] bg-slate-800" />
            <span className="text-[11px] font-medium uppercase tracking-widest text-slate-500">
              {isReplayMode ? "Historical playback" : "Monitoring Live Stream"}
            </span>
          </div>

          {isReplayMode && replayHistory.length > 0 && (
            <div className="flex items-center gap-2 pointer-events-auto">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="flex h-8 w-8 items-center justify-center rounded-lg border border-slate-800 bg-slate-900 hover:bg-slate-800 text-white transition-all cursor-pointer"
              >
                {isPlaying ? <Pause className="h-4 w-4 text-cyan-300" /> : <Play className="h-4 w-4" />}
              </button>
              <button
                onClick={() => {
                  setReplayIndex(0);
                  setIsPlaying(false);
                }}
                className="flex h-8 w-8 items-center justify-center rounded-lg border border-slate-800 bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white transition-all cursor-pointer"
                title="Reset Replay"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </button>
              <span className="text-xs font-mono text-slate-400 px-2 py-1 bg-slate-900/50 border border-slate-800/40 rounded">
                {replayHistory[replayIndex]?.timestamp ? new Date(replayHistory[replayIndex].timestamp).toLocaleTimeString() : "--"}
              </span>
            </div>
          )}
        </div>

        {isReplayMode && (
          <div className="mt-2 flex flex-col gap-2 pointer-events-auto">
            {isLoadingHistory ? (
              <div className="text-center py-2 text-xs text-slate-500 animate-pulse">
                Loading compliance history database...
              </div>
            ) : replayHistory.length === 0 ? (
              <div className="text-center py-2 text-xs text-slate-500">
                No telemetry history found on backend.
              </div>
            ) : (
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min="0"
                  max={replayHistory.length - 1}
                  value={replayIndex}
                  onChange={(e) => {
                    setReplayIndex(Number(e.target.value));
                    setIsPlaying(false);
                  }}
                  className="h-1.5 w-full cursor-pointer rounded-lg bg-slate-800 accent-cyan-400 outline-none transition-all hover:bg-slate-700"
                />
                <span className="text-[10px] font-mono text-slate-500 whitespace-nowrap">
                  {replayIndex + 1} / {replayHistory.length}
                </span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Scenario Simulator panel overlay */}
      {isSimulating && (
        <div className="absolute right-5 top-20 z-20 w-80 rounded-xl border border-slate-800/80 bg-slate-950/85 p-5 shadow-2xl backdrop-blur-xl pointer-events-auto flex flex-col gap-4 text-white">
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-300">Scenario Simulator</h3>
            <p className="mt-1 text-[11px] text-slate-400">Mock a leak at any industrial facility to preview plume dispersion.</p>
          </div>

          <div className="flex flex-col gap-3">
            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Select Facility</label>
              <select
                value={simFactoryName}
                onChange={(e) => setSimFactoryName(e.target.value)}
                className="w-full rounded border border-slate-800 bg-slate-900 px-3 py-2 text-xs text-slate-100 focus:border-cyan-500/50 focus:outline-none cursor-pointer"
              >
                {registryFactories.map((fac) => (
                  <option key={fac.factory_name} value={fac.factory_name}>
                    {fac.factory_name}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex flex-col gap-1.5">
              <div className="flex justify-between">
                <label className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Wind Direction</label>
                <span className="text-[11px] font-bold text-cyan-300">{simWindDirection}°</span>
              </div>
              <input
                type="range"
                min="0"
                max="359"
                value={simWindDirection}
                onChange={(e) => setSimWindDirection(Number(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer bg-slate-800 h-1.5 rounded-lg appearance-none"
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <div className="flex justify-between">
                <label className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Wind Speed</label>
                <span className="text-[11px] font-bold text-cyan-300">{simWindSpeed.toFixed(1)} m/s</span>
              </div>
              <input
                type="range"
                min="1"
                max="15"
                step="0.5"
                value={simWindSpeed}
                onChange={(e) => setSimWindSpeed(Number(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer bg-slate-800 h-1.5 rounded-lg appearance-none"
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Pollutant Type</label>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setSimPollutant("SO2")}
                  className={`flex-1 rounded border py-1.5 text-xs font-semibold cursor-pointer transition ${
                    simPollutant === "SO2"
                      ? "bg-cyan-500/20 border-cyan-400/50 text-cyan-300"
                      : "bg-slate-900 border-slate-800 text-slate-400 hover:text-white"
                  }`}
                >
                  SO2
                </button>
                <button
                  type="button"
                  onClick={() => setSimPollutant("PM25")}
                  className={`flex-1 rounded border py-1.5 text-xs font-semibold cursor-pointer transition ${
                    simPollutant === "PM25"
                      ? "bg-cyan-500/20 border-cyan-400/50 text-cyan-300"
                      : "bg-slate-900 border-slate-800 text-slate-400 hover:text-white"
                  }`}
                >
                  PM2.5
                </button>
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <div className="flex justify-between">
                <label className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Leak Intensity</label>
                <span className="text-[11px] font-bold text-cyan-300">{simIntensity} {simPollutant === "SO2" ? "ppb" : "ug/m³"}</span>
              </div>
              <input
                type="range"
                min="50"
                max="500"
                value={simIntensity}
                onChange={(e) => setSimIntensity(Number(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer bg-slate-800 h-1.5 rounded-lg appearance-none"
              />
            </div>
          </div>

          <div className="mt-2 flex gap-3">
            <button
              type="button"
              onClick={runSimulation}
              className="flex-1 rounded-lg bg-cyan-500 px-4 py-2 text-xs font-bold text-slate-950 hover:bg-cyan-400 transition cursor-pointer"
            >
              Simulate Leak
            </button>
            <button
              type="button"
              onClick={() => {
                setSimulatedPlume(null);
                setSimOrigin(null);
              }}
              className="rounded-lg bg-slate-900 border border-slate-800 px-4 py-2 text-xs font-semibold text-slate-300 hover:text-white hover:border-slate-700 transition cursor-pointer"
            >
              Reset
            </button>
          </div>
        </div>
      )}
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
