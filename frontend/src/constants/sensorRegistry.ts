export interface SensorNode {
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

export const SENSOR_REGISTRY: SensorNode[] = [
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
