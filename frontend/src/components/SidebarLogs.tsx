"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Download, FileWarning, Loader2, Play, TerminalSquare } from "lucide-react";

import type { AttributionSource, HazardEvent, TelemetryReading } from "@/components/MapEngine";

export interface TelemetryLogEntry {
  id: string;
  timestamp: string;
  sensorId: string;
  severity: "normal" | "critical";
  message: string;
  reading: TelemetryReading;
}

interface SidebarLogsProps {
  logs: TelemetryLogEntry[];
  latestHazard: HazardEvent | null;
  onAttributionComplete?: (source: AttributionSource | null) => void;
  externalReport?: string | null;
  draftId?: string | null;
  reviewStatus?: string | null;
}

interface AgentResponse {
  enforcement_report?: string;
  core_ai_inversion_coordinates?: {
    source_latitude?: number;
    source_longitude?: number;
    confidence_score?: number;
  };
  report?: string;
  text?: string;
}

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  process.env.NEXT_PUBLIC_BACKEND_URL ||
  "http://127.0.0.1:8000"
).replace(/\/$/, "");
const ORCHESTRATOR_ENDPOINT =
  process.env.NEXT_PUBLIC_AGENT_ORCHESTRATOR_URL || `${API_BASE_URL}/api/v1/agent/orchestrate`;

function formatClock(timestamp: string): string {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return "--:--:--";
  }
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
}

function buildAgentPayload(latestHazard: HazardEvent | null) {
  if (!latestHazard) {
    return null;
  }

  const reading = latestHazard.reading;
  return {
    sensor_alert_payload: {
      sensor_id: reading.sensor_id,
      latitude: reading.latitude,
      longitude: reading.longitude,
      gas_type: reading.so2 >= 75 ? "SO2" : "PM2.5",
      value: reading.so2 >= 75 ? reading.so2 : reading.pm25,
      timestamp: reading.timestamp
    },
    suspected_source_hint: latestHazard.suspectedSource
  };
}

export default function SidebarLogs({ logs, latestHazard, onAttributionComplete, externalReport, draftId, reviewStatus }: SidebarLogsProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [report, setReport] = useState("");
  const [error, setError] = useState("");
  const [localDraftId, setLocalDraftId] = useState<string | null>(null);
  const [localReviewStatus, setLocalReviewStatus] = useState<string | null>(null);
  const [isReviewing, setIsReviewing] = useState(false);
  const [feedback, setFeedback] = useState("");

  useEffect(() => {
    if (externalReport) {
      setReport(externalReport);
      setIsRunning(false);
      setError("");
    }
  }, [externalReport]);

  useEffect(() => {
    if (draftId) setLocalDraftId(draftId);
    if (reviewStatus) setLocalReviewStatus(reviewStatus);
  }, [draftId, reviewStatus]);

  const lastCritical = useMemo(() => logs.find((entry) => entry.severity === "critical"), [logs]);

  const runAttributionEngine = useCallback(async () => {
    const activeHazard = latestHazard;
    const payload = buildAgentPayload(activeHazard);
    if (!payload || !activeHazard) {
      setError("No critical hazard packet is available for attribution.");
      return;
    }

    setIsRunning(true);
    setError("");

    try {
      const apiKey = process.env.NEXT_PUBLIC_API_KEY || "dev-insecure-key";
      const response = await fetch(ORCHESTRATOR_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": apiKey
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`Orchestrator returned HTTP ${response.status}`);
      }

      const data = (await response.json()) as AgentResponse;
      const generatedReport = data.enforcement_report || data.report || data.text;
      if (!generatedReport) {
        throw new Error("Orchestrator response did not include report text.");
      }
      const coordinates = data.core_ai_inversion_coordinates;
      const sourceLatitude = Number(coordinates?.source_latitude);
      const sourceLongitude = Number(coordinates?.source_longitude);
      const confidence = Number(coordinates?.confidence_score);
      if (Number.isFinite(sourceLatitude) && Number.isFinite(sourceLongitude)) {
        onAttributionComplete?.({
          latitude: sourceLatitude,
          longitude: sourceLongitude,
          confidence: Number.isFinite(confidence) ? confidence : activeHazard.suspectedSource.confidence
        });
      }
      setReport(generatedReport);
    } catch (requestError) {
      const message = requestError instanceof Error ? requestError.message : "Unknown orchestration error";
      setError(message);
    } finally {
      setIsRunning(false);
    }
  }, [latestHazard, onAttributionComplete]);

  const submitReview = useCallback(async (status: "approved" | "rejected") => {
    if (!localDraftId) return;
    setIsReviewing(true);
    setError("");
    try {
      const apiKey = process.env.NEXT_PUBLIC_API_KEY || "dev-insecure-key";
      const response = await fetch(`${API_BASE_URL}/api/v1/drafts/${localDraftId}/review`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "X-API-Key": apiKey 
        },
        body: JSON.stringify({ status, reviewer_id: "operator_01", report_text: report })
      });
      if (!response.ok) throw new Error("Failed to submit review");
      const data = await response.json();
      setLocalReviewStatus(data.status);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Review submission failed");
    } finally {
      setIsReviewing(false);
    }
  }, [localDraftId, report]);

  const refineReview = useCallback(async () => {
    if (!localDraftId || !feedback.trim()) return;
    setIsReviewing(true);
    setError("");
    try {
      const apiKey = process.env.NEXT_PUBLIC_API_KEY || "dev-insecure-key";
      const response = await fetch(`${API_BASE_URL}/api/v1/drafts/${localDraftId}/review`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "X-API-Key": apiKey 
        },
        body: JSON.stringify({ status: "refine", reviewer_id: "operator_01", feedback: feedback })
      });
      if (!response.ok) throw new Error("Failed to submit refinement request");
      const data = await response.json();
      setReport(data.report_text);
      setFeedback("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Refinement request failed");
    } finally {
      setIsReviewing(false);
    }
  }, [localDraftId, feedback]);

  const exportReport = useCallback(() => {
    if (!report) {
      return;
    }

    const blob = new Blob([report], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `plumetrace-enforcement-report-${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }, [report]);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-5">
      <section className="rounded-lg border border-slate-800/70 bg-slate-950/60 shadow-2xl backdrop-blur-xl">
        <div className="flex items-center justify-between border-b border-slate-800/70 px-4 py-3">
          <div className="flex items-center gap-2">
            <TerminalSquare className="h-4 w-4 text-emerald-300" />
            <h2 className="text-sm font-semibold text-white">Telemetry Terminal</h2>
          </div>
          <span className="rounded-md border border-slate-700/80 px-2 py-1 text-[11px] uppercase tracking-wide text-slate-400">
            {logs.length} packets
          </span>
        </div>

        <div className="h-[140px] overflow-y-auto px-3 py-3 lg:h-[260px]">
          {logs.length === 0 ? (
            <div className="flex h-full items-center justify-center rounded-md border border-dashed border-slate-800 text-sm text-slate-500">
              Awaiting live sensor telemetry
            </div>
          ) : (
            <div className="space-y-2 font-mono text-xs">
              {logs.map((entry) => (
                <div
                  key={entry.id}
                  className={`rounded-md border px-3 py-2 ${
                    entry.severity === "critical"
                      ? "border-rose-400/30 bg-rose-950/30 text-rose-100"
                      : "border-slate-800/80 bg-slate-900/50 text-slate-300"
                  }`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-slate-500">{formatClock(entry.timestamp)}</span>
                    <span className={entry.severity === "critical" ? "text-rose-300" : "text-emerald-300"}>
                      {entry.severity}
                    </span>
                  </div>
                  <p className="mt-1 leading-5">{entry.message}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      <section className="rounded-lg border border-slate-800/70 bg-slate-950/60 shadow-2xl backdrop-blur-xl">
        <div className="border-b border-slate-800/70 px-4 py-3">
          <div className="flex items-center gap-2">
            <FileWarning className="h-4 w-4 text-rose-300" />
            <h2 className="text-sm font-semibold text-white">Forensic Attribution</h2>
          </div>
          <p className="mt-2 text-xs leading-5 text-slate-400">
            {lastCritical
              ? `${lastCritical.sensorId} has an active critical packet ready for escalation.`
              : "Critical hazard packets will appear here for escalation."}
          </p>
        </div>

        <div className="space-y-3 p-4">
          <button
            type="button"
            onClick={runAttributionEngine}
            disabled={isRunning || !latestHazard}
            className="flex h-11 w-full items-center justify-center gap-2 rounded-lg border border-rose-300/30 bg-rose-500/15 px-4 text-sm font-semibold text-rose-100 shadow-plume-red transition hover:bg-rose-500/25 disabled:cursor-not-allowed disabled:border-slate-800 disabled:bg-slate-900 disabled:text-slate-500 disabled:shadow-none"
          >
            {isRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Run Forensic Attribution Engine
          </button>

          {error ? (
            <div className="rounded-md border border-amber-400/30 bg-amber-500/10 px-3 py-2 text-xs leading-5 text-amber-100">
              {error}
            </div>
          ) : null}

          {isRunning ? (
            <div className="space-y-3 rounded-lg border border-slate-800/70 bg-slate-900/40 p-4">
              <div className="h-3 w-5/6 animate-pulse rounded bg-slate-700" />
              <div className="h-3 w-full animate-pulse rounded bg-slate-800" />
              <div className="h-3 w-4/5 animate-pulse rounded bg-slate-800" />
              <div className="h-3 w-11/12 animate-pulse rounded bg-slate-800" />
            </div>
          ) : null}

          {report ? (
            <div className="rounded-lg border border-slate-800/70 bg-slate-900/60">
              <div className="flex items-center justify-between border-b border-slate-800/70 px-4 py-3">
                <h3 className="text-sm font-semibold text-white">Compliance Report</h3>
                <button
                  type="button"
                  onClick={exportReport}
                  className="flex h-8 items-center gap-2 rounded-md border border-slate-700 bg-slate-950 px-2.5 text-xs font-medium text-slate-200 transition hover:border-cyan-300/50 hover:text-cyan-100"
                >
                  <Download className="h-3.5 w-3.5" />
                  Export
                </button>
              </div>
              <div className="max-h-[190px] overflow-y-auto px-4 py-4 text-sm leading-6 text-slate-200 lg:max-h-[270px]">
                {localReviewStatus === "pending_human_review" ? (
                  <textarea
                    value={report}
                    onChange={(e) => setReport(e.target.value)}
                    className="w-full h-full min-h-[150px] bg-slate-950/50 border border-slate-700/50 rounded-md p-3 text-slate-200 focus:outline-none focus:ring-1 focus:ring-cyan-500/50 resize-none"
                  />
                ) : (
                  <div className="whitespace-pre-line">{report}</div>
                )}
              </div>
              
              {localReviewStatus === "pending_human_review" && (
                <div className="border-t border-slate-800/70 p-4 bg-slate-900/40">
                  <div className="flex gap-3">
                    <button
                      onClick={() => submitReview("approved")}
                      disabled={isReviewing}
                      className="flex-1 rounded-md bg-emerald-500/20 px-3 py-2 text-xs font-semibold text-emerald-300 border border-emerald-500/30 hover:bg-emerald-500/30 transition disabled:opacity-50 cursor-pointer"
                    >
                      {isReviewing ? "Submitting..." : "Approve & Dispatch"}
                    </button>
                    <button
                      onClick={() => submitReview("rejected")}
                      disabled={isReviewing}
                      className="flex-1 rounded-md bg-rose-500/20 px-3 py-2 text-xs font-semibold text-rose-300 border border-rose-500/30 hover:bg-rose-500/30 transition disabled:opacity-50 cursor-pointer"
                    >
                      Reject Draft
                    </button>
                  </div>
                  
                  {/* Refinement input controls */}
                  <div className="mt-4 flex flex-col gap-2 border-t border-slate-800/60 pt-3">
                    <input
                      type="text"
                      placeholder="Revision instructions (e.g. Cite Clean Air Act)"
                      value={feedback}
                      onChange={(e) => setFeedback(e.target.value)}
                      className="w-full bg-slate-950/65 border border-slate-800 rounded px-2.5 py-1.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
                      disabled={isReviewing}
                    />
                    <button
                      onClick={refineReview}
                      disabled={isReviewing || !feedback.trim()}
                      className="w-full rounded-md bg-cyan-500/20 px-3 py-1.5 text-xs font-semibold text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/30 transition disabled:opacity-50 cursor-pointer"
                    >
                      {isReviewing ? "Refining..." : "Request Refinements"}
                    </button>
                  </div>
                </div>
              )}
              {localReviewStatus && localReviewStatus !== "pending_human_review" && (
                <div className="border-t border-slate-800/70 p-3 bg-slate-900/80">
                  <p className={`text-xs text-center font-medium ${localReviewStatus === 'approved' ? 'text-emerald-400' : 'text-rose-400'}`}>
                    Draft {localReviewStatus.toUpperCase()}
                  </p>
                </div>
              )}
            </div>
          ) : null}
        </div>
      </section>
    </div>
  );
}
