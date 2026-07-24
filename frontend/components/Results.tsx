"use client";

import type { AnalyzeResult, ShapMap } from "../lib/types";

// The backend already returns human-readable driver names (the 58 librosa
// descriptors are grouped into 12 concepts server-side in model/explain.py), so
// this map only needs to cover the two context keys. Anything else passes through.
const FEATURE_LABELS: Record<string, string> = {
  fame: "artist fame",
  genre: "genre",
};

const fmt = (x: number | null | undefined, digits = 1): string =>
  x == null ? "—" : Number(x).toFixed(digits).replace(/\.0$/, "");
const signed = (x: number | null | undefined): string =>
  x == null ? "—" : x >= 0 ? `+${fmt(x)}` : fmt(x);
const ordinal = (n: number): string => {
  const s = ["th", "st", "nd", "rd"], v = Math.round(n) % 100;
  return `${Math.round(n)}${s[(v - 20) % 10] || s[v] || s[0]}`;
};

interface ShapBarsProps {
  title: string;
  sub: string;
  shap?: ShapMap;
}

function ShapBars({ title, sub, shap }: ShapBarsProps) {
  const entries = Object.entries(shap || {})
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 6);
  if (!entries.length) return null;
  const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v)), 1e-9);

  return (
    <div>
      <h3>{title}</h3>
      <p className="sub">{sub}</p>
      {entries.map(([feature, value]) => (
        <div className="shaprow" key={feature}>
          <span className="flab">{FEATURE_LABELS[feature] || feature}</span>
          <span className="barwrap">
            <span className="axis" />
            <span
              className={`bar ${value >= 0 ? "pos" : "neg"}`}
              style={{ width: `${(Math.abs(value) / maxAbs) * 46}%` }}
            />
          </span>
          <span className="val">{signed(value)}</span>
        </div>
      ))}
    </div>
  );
}

export default function Results({ result }: { result: AnalyzeResult | null }) {
  if (!result) return null;
  const {
    sample,
    baseline,
    fame_contribution,
    genre_contribution,
    audio_contribution,
    predicted_popularity,
    audio_percentile,
    audio_percentile_scope,
    fame_estimated,
    genre_imputed,
    shap_context,
    shap_audio,
    warnings = [],
    explanation,
    audio_features,
  } = result;

  const scope = audio_percentile_scope || "all genres";

  return (
    <section className="card" aria-label="Analysis result">
      <h2>
        Prediction
        {sample && <span className="samplebadge">SAMPLE — backend not connected</span>}
      </h2>
      <p className="hint">
        Spotify-style popularity, 0–100, split into what the artist brings and what
        the song itself does.
      </p>

      {warnings.map((w, i) => (
        <div className="banner warn" key={i} style={{ margin: "0 0 14px" }}>
          <span className="ic">⚠</span> {w}
        </div>
      ))}

      <div className="heroNum">
        <span className="value">{fmt(predicted_popularity)}</span>
        <span className="of">/ 100 predicted popularity</span>
      </div>
      <div
        className="meter"
        role="img"
        aria-label={`Predicted popularity ${fmt(predicted_popularity)} out of 100`}
      >
        <div
          className="fill"
          style={{ width: `${Math.max(0, Math.min(100, predicted_popularity))}%` }}
        />
      </div>

      {/* The four parts sum exactly to the total — that identity is asserted
          server-side and in tests, so the strip can be read as real arithmetic. */}
      <div className="equation">
        <div className="eqtile">
          <div className="lab">Typical track</div>
          <div className="num">{fmt(baseline)}</div>
        </div>
        <span className="eqop">+</span>
        <div className="eqtile">
          <div className="lab">Artist fame</div>
          <div className="num">{signed(fame_contribution)}</div>
          {fame_estimated && <span className="chip est">fame estimated</span>}
        </div>
        <span className="eqop">+</span>
        <div className="eqtile">
          <div className="lab">Style (genre)</div>
          <div className="num">{signed(genre_contribution)}</div>
          {genre_imputed && <span className="chip est">averaged over genres</span>}
        </div>
        <span className="eqop">+</span>
        <div className="eqtile">
          <div className="lab">This recording</div>
          <div className="num">{signed(audio_contribution)}</div>
          {audio_percentile != null && (
            <span className="chip">
              {ordinal(audio_percentile)} percentile of {scope} tracks
            </span>
          )}
        </div>
        <span className="eqop">=</span>
        <div className="eqtile final">
          <div className="lab">Predicted popularity</div>
          <div className="num">{fmt(predicted_popularity)}</div>
        </div>
      </div>

      <hr style={{ border: "none", borderTop: "1px solid var(--line)", margin: "24px 0 18px" }} />

      <div className="drivers">
        <ShapBars
          title="What the artist brought"
          sub="Fame and style, measured at this artist's reach"
          shap={shap_context}
        />
        <ShapBars
          title="What the recording added"
          sub={`Audio traits behind the ${signed(audio_contribution)}, vs other ${scope} tracks`}
          shap={shap_audio}
        />
      </div>
      <div className="legendrow" aria-hidden="true">
        <span className="key">
          <span className="sw" style={{ background: "var(--mark-pos)" }} /> pushes the score up
        </span>
        <span className="key">
          <span className="sw" style={{ background: "var(--mark-neg)" }} /> pulls it down
        </span>
      </div>

      {explanation && (
        <>
          <hr style={{ border: "none", borderTop: "1px solid var(--line)", margin: "22px 0 16px" }} />
          <p className="explain">{explanation}</p>
        </>
      )}

      {audio_features && (
        <details className="raw">
          <summary>Feature details</summary>
          <p style={{ margin: "10px 0 4px" }}>
            librosa descriptors extracted from your audio:
          </p>
          <pre>{JSON.stringify(audio_features, null, 2)}</pre>
        </details>
      )}
    </section>
  );
}
