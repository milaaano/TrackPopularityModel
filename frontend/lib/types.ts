// Shared types for the SoundSignal frontend.
//
// AnalyzeResult mirrors the FastAPI /analyze contract. Keep it in sync with
// backend/app/main.py — it is the single source of truth the UI renders against.

export type BackendState = "checking" | "ready" | "waking" | "offline";

export interface SubmitPayload {
  file: File;
  artist: string;
  genre: string;
  explicit: boolean;
}

/** A per-driver attribution map, e.g. { brightness: 0.6, "overall timbre": 0.8 }. */
export type ShapMap = Record<string, number>;

export interface AnalyzeResult {
  /** Set only by the canned client-side sample; real responses omit it. */
  sample?: boolean;

  // --- the 4-part breakdown, in popularity points ---
  // baseline + fame + genre + audio === predicted_popularity (exactly).
  baseline: number; // a typical track: median fame, average genre
  fame_contribution: number; // what the artist's reach adds
  genre_contribution: number; // what this style is worth, at this fame
  audio_contribution: number; // what THIS recording adds within its genre
  predicted_popularity: number;

  // --- is that audio contribution any good? ---
  // Ranked against other songs of the SAME genre where we have enough of them;
  // audio_percentile_scope names the field it was ranked in ("pop" / "all genres").
  audio_percentile?: number | null;
  audio_percentile_scope?: string | null;

  // --- honesty flags: both must be surfaced in the UI when true ---
  fame_estimated?: boolean; // artist not found -> low prior used
  genre_imputed?: boolean; // genre unknown -> averaged over all genres
  artist_fame?: number;

  // --- Stage 9 grounding ---
  shap_context?: ShapMap; // { fame, genre } — mirrors the tiles exactly
  shap_audio?: ShapMap; // the 12 audio concepts, top drivers first

  /** The librosa descriptors that were actually scored. */
  audio_features?: Record<string, number>;

  warnings?: string[];
  explanation?: string;
  explanation_source?: "llm" | "template";
}
