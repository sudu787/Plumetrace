"use client";

import { useCallback, useMemo, useState } from "react";
import { Activity, Gauge, RadioTower, ShieldAlert } from "lucide-react";

import MapEngine, {
  type AttributionSource,
  type HazardEvent,
  type TelemetryReading
} from "@/components/MapEngine";
import SidebarLogs, { type TelemetryLogEntry } from "@/components/SidebarLogs";

const MAX_LOG_ENTRIES = 120;

function buildLogEntry(reading: TelemetryReading, severity: TelemetryLogEntry["severity"]): TelemetryLogEntry {
  const timestamp = reading.timestamp || new Date().toISOString();
  return {
    id: `${reading.sensor_id}-${timestamp}-${Math.random().toString(36).slice(2, 8)}`,
    timestamp,
    sensorId: reading.sensor_id,
    severity,
    message:
      severity === "critical"
        ? `${reading.sensor_id} crossed hazard threshold: SO2 ${reading.so2.toFixed(1)} ppb, PM2.5 ${reading.pm25.toFixed(1)} ug/m3`
        : `${reading.sensor_id} nominal telemetry: SO2 ${reading.so2.toFixed(1)} ppb, PM2.5 ${reading.pm25.toFixed(1)} ug/m3`,
    reading
  };
}

export default function DashboardPage() {
  const [logs, setLogs] = useState<TelemetryLogEntry[]>([]);
  const [latestHazard, setLatestHazard] = useState<HazardEvent | null>(null);
  const [attributedSource, setAttributedSource] = useState<AttributionSource | null>(null);
  const [autonomousReport, setAutonomousReport] = useState<string | null>(null);
  const [draftId, setDraftId] = useState<string | null>(null);
  const [reviewStatus, setReviewStatus] = useState<string | null>(null);
  const [connectionState, setConnectionState] = useState<"connecting" | "online" | "offline">("connecting");

  const handleTelemetry = useCallback((reading: TelemetryReading, severity: TelemetryLogEntry["severity"]) => {
    setLogs((current) => [buildLogEntry(reading, severity), ...current].slice(0, MAX_LOG_ENTRIES));
  }, []);

  const handleHazard = useCallback((event: HazardEvent) => {
    setLatestHazard(event);
    setAttributedSource(null);
    setAutonomousReport(null);
  }, []);

  const handleEnforcementReport = useCallback((data: any) => {
    const coords = data.source_coordinates || {};
    if (coords.source_latitude && coords.source_longitude) {
      setAttributedSource({
        latitude: Number(coords.source_latitude),
        longitude: Number(coords.source_longitude),
        confidence: Number(coords.confidence_score) || 0.95
      });
    }
    if (data.report) {
      setAutonomousReport(data.report);
    }
    if (data.draft_id) {
      setDraftId(data.draft_id);
    }
    if (data.review_status) {
      setReviewStatus(data.review_status);
    }
  }, []);

  const stats = useMemo(() => {
    const criticalCount = logs.filter((entry) => entry.severity === "critical").length;
    const latest = logs[0]?.reading;
    return {
      criticalCount,
      latestSo2: latest ? latest.so2.toFixed(1) : "--",
      latestPm25: latest ? latest.pm25.toFixed(1) : "--"
    };
  }, [logs]);

  return (
    <main className="h-screen w-screen overflow-hidden bg-slate-950 text-slate-100">
      <div className="grid h-full w-full grid-rows-[45vh_minmax(0,1fr)] lg:grid-cols-[400px_minmax(0,1fr)] lg:grid-rows-none">
        <aside className="relative z-20 h-full min-h-0 border-b border-slate-800/50 bg-slate-950/80 shadow-2xl backdrop-blur-xl lg:border-b-0 lg:border-r">
          <div className="flex h-full flex-col">
            <header className="border-b border-slate-800/60 px-6 py-5">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.22em] text-cyan-300">PlumeTrace</p>
                  <h1 className="mt-1 text-2xl font-semibold tracking-tight text-white">Command Center</h1>
                </div>
                <div className="flex h-11 w-11 items-center justify-center rounded-lg border border-cyan-400/30 bg-cyan-400/10 text-cyan-200 shadow-plume-green">
                  <RadioTower className="h-5 w-5" />
                </div>
              </div>

              <div className="mt-5 grid grid-cols-3 gap-2">
                <div className="rounded-lg border border-slate-800/70 bg-slate-900/70 px-3 py-2">
                  <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wide text-slate-400">
                    <Activity className="h-3.5 w-3.5" />
                    Link
                  </div>
                  <p className={`mt-1 text-sm font-semibold ${connectionState === "online" ? "text-emerald-300" : connectionState === "connecting" ? "text-amber-300" : "text-rose-300"}`}>
                    {connectionState}
                  </p>
                </div>
                <div className="rounded-lg border border-slate-800/70 bg-slate-900/70 px-3 py-2">
                  <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wide text-slate-400">
                    <ShieldAlert className="h-3.5 w-3.5" />
                    Alerts
                  </div>
                  <p className="mt-1 text-sm font-semibold text-rose-300">{stats.criticalCount}</p>
                </div>
                <div className="rounded-lg border border-slate-800/70 bg-slate-900/70 px-3 py-2">
                  <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wide text-slate-400">
                    <Gauge className="h-3.5 w-3.5" />
                    SO2
                  </div>
                  <p className="mt-1 text-sm font-semibold text-slate-100">{stats.latestSo2}</p>
                </div>
              </div>
            </header>

            <SidebarLogs
              logs={logs}
              latestHazard={latestHazard}
              onAttributionComplete={setAttributedSource}
              externalReport={autonomousReport}
              draftId={draftId}
              reviewStatus={reviewStatus}
            />
          </div>
        </aside>

        <section className="relative min-w-0 overflow-hidden">
          <MapEngine
            attributedSource={attributedSource}
            onTelemetry={handleTelemetry}
            onHazard={handleHazard}
            onConnectionStateChange={setConnectionState}
            onEnforcementReport={handleEnforcementReport}
          />
        </section>
      </div>
    </main>
  );
}
