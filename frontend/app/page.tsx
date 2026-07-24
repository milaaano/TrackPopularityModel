"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import AnalyzeForm from "../components/AnalyzeForm";
import Results from "../components/Results";
import { SAMPLE_RESULT } from "../lib/sample";
import type { AnalyzeResult, BackendState, SubmitPayload } from "../lib/types";

const API = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");
const WAKE_RETRY_MS = 10_000; // HF free Spaces cold-start: poll /health while user reads
const WAKE_GIVE_UP_MS = 120_000;
const ANALYZE_TIMEOUT_MS = 150_000;

export default function Page() {
  const [backendState, setBackendState] = useState<BackendState>("checking");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const wakeStart = useRef<number>(Date.now());

  // PLAN.md: ping /health on page load so a sleeping backend wakes while the
  // visitor is still reading the header.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function ping(): Promise<void> {
      try {
        const res = await fetch(`${API}/health`, { signal: AbortSignal.timeout(8000) });
        if (!cancelled && res.ok) return setBackendState("ready");
        throw new Error();
      } catch {
        if (cancelled) return;
        if (Date.now() - wakeStart.current > WAKE_GIVE_UP_MS) {
          setBackendState("offline");
        } else {
          setBackendState("waking");
          timer = setTimeout(ping, WAKE_RETRY_MS);
        }
      }
    }

    ping();
    return () => { cancelled = true; clearTimeout(timer); };
  }, []);

  const analyze = useCallback(async ({ file, artist, genre, explicit }: SubmitPayload) => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const body = new FormData();
      body.append("audio_file", file);
      body.append("artist_name", artist);
      body.append("genre", genre);
      body.append("explicit", String(explicit));

      const res = await fetch(`${API}/analyze`, {
        method: "POST",
        body,
        signal: AbortSignal.timeout(ANALYZE_TIMEOUT_MS),
      });
      if (!res.ok) {
        const detail = await res.text().catch(() => "");
        throw new Error(`The analysis service returned ${res.status}. ${detail.slice(0, 200)}`);
      }
      setResult((await res.json()) as AnalyzeResult);
    } catch (err) {
      const isTimeout = err instanceof DOMException && err.name === "TimeoutError";
      const message = err instanceof Error ? err.message : String(err);
      setError(
        isTimeout
          ? "The analysis timed out. A cold backend can take up to 2 minutes on its first run — please try once more."
          : `Analysis failed: ${message}. If the backend was asleep, a retry usually works.`
      );
    } finally {
      setBusy(false);
    }
  }, []);

  return (
    <main className="shell">
      <header className="hero">
        <span className="wordmark"><span className="dot" />SoundSignal</span>
        <h1>How popular could your song be — and why?</h1>
        <p className="sub">
          Upload a track. We predict its Spotify-style popularity and split the
          number into what artist fame buys and what the song itself earns —
          with the drivers behind each part.
        </p>

        {backendState === "waking" && (
          <div className="banner warn">
            <span className="ic">⏳</span>
            Waking the analysis service, this can
            take a minute or two. The form works meanwhile.
          </div>
        )}
        {backendState === "offline" && (
          <div className="banner warn">
            <span className="ic">⚠</span>
            The analysis service is offline. You can explore the interface with a
            clearly-labeled sample result below.
          </div>
        )}
      </header>

      <AnalyzeForm onSubmit={analyze} busy={busy} backendState={backendState} />

      {error && (
        <div className="banner err" style={{ marginTop: 18 }}>
          <span className="ic">⚠</span> {error}
        </div>
      )}

      {!result && (backendState === "offline" || backendState === "waking") && (
        <div style={{ marginTop: 18 }}>
          <button className="btn ghost" onClick={() => setResult(SAMPLE_RESULT)}>
            Show a sample result while you wait
          </button>
        </div>
      )}

      <Results result={result} />
    </main>
  );
}
