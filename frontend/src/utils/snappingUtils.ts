import type { Feature, LineString, Point, Polygon } from "geojson";

export const SO2_SAFE_LIMIT = 75;
export const PM25_SAFE_LIMIT = 35;

export interface FactoryFacility {
  factory_name: string;
  corporate_owner: string;
  zoning_permit_id: string;
  latitude: number;
  longitude: number;
  plot_radius_km: number;
}

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

export interface HazardEvent {
  reading: TelemetryReading;
  suspectedSource: {
    latitude: number;
    longitude: number;
    confidence: number;
  };
  plume: Feature<Polygon>;
}

export function isCritical(reading: Pick<TelemetryReading, "so2" | "pm25">): boolean {
  return reading.so2 >= SO2_SAFE_LIMIT || reading.pm25 >= PM25_SAFE_LIMIT;
}

export function toFiniteNumber(value: unknown, fallback = 0): number {
  const numeric = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

export function normalizeTelemetry(payload: unknown): TelemetryReading | null {
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

export function offsetCoordinate(latitude: number, longitude: number, bearingDegrees: number, distanceKm: number) {
  const bearing = (bearingDegrees * Math.PI) / 180;
  const deltaNorthKm = Math.cos(bearing) * distanceKm;
  const deltaEastKm = Math.sin(bearing) * distanceKm;
  const nextLatitude = latitude + deltaNorthKm / 111;
  const nextLongitude = longitude + deltaEastKm / (111 * Math.cos((latitude * Math.PI) / 180));
  return {
    latitude: toFiniteNumber(nextLatitude),
    longitude: toFiniteNumber(nextLongitude)
  };
}

export function haversine_km(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371; // Earth radius in km
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

export function getBearing(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const lat1Rad = (lat1 * Math.PI) / 180;
  const lat2Rad = (lat2 * Math.PI) / 180;
  const y = Math.sin(dLon) * Math.cos(lat2Rad);
  const x = Math.cos(lat1Rad) * Math.sin(lat2Rad) - Math.sin(lat1Rad) * Math.cos(lat2Rad) * Math.cos(dLon);
  const brng = (Math.atan2(y, x) * 180) / Math.PI;
  return (brng + 360) % 360;
}

export function snapToNearestFactory(
  factories: FactoryFacility[],
  latitude: number,
  longitude: number,
  windDirection?: number,
  sensorLat?: number,
  sensorLon?: number
) {
  if (factories.length === 0) return { latitude, longitude };
  let bestFac = factories[0];

  if (windDirection !== undefined && sensorLat !== undefined && sensorLon !== undefined) {
    let bestScore = -Infinity;
    const upstreamBearing = windDirection % 360;
    const severityFactor = 1.0;
    const plumeLengthKm = 0.48 + severityFactor * 0.22;
    const rawSource = offsetCoordinate(sensorLat, sensorLon, upstreamBearing, plumeLengthKm);

    factories.forEach((fac) => {
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
    // Normal haversine/Euclidean snap
    let minDistance = Infinity;
    factories.forEach((fac) => {
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

export function buildPlumeFeature(
  factories: FactoryFacility[],
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
  const plumeWidthKm = (0.12 + severityFactor * 0.04) * pulseScale;
  const upstreamBearing = reading.wind_direction % 360;

  let sourceCoord;
  if (customSource) {
    sourceCoord = snapToNearestFactory(factories, customSource.latitude, customSource.longitude);
  } else {
    sourceCoord = snapToNearestFactory(
      factories,
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
