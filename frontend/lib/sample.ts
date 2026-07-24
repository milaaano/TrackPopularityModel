import type { AnalyzeResult } from "./types";

// A canned /analyze response for exploring the UI when the backend is offline.
// Values are realistic for our models (fame and genre dominate; the recording
// itself moves the score only a few points) but they are FAKE — every surface
// showing them must carry the SAMPLE badge.
export const SAMPLE_RESULT: AnalyzeResult = {
  sample: true,
  baseline: 38.7,
  fame_contribution: 3.4,
  genre_contribution: 30.3,
  audio_contribution: 2.1,
  predicted_popularity: 74.5,
  audio_percentile: 73,
  audio_percentile_scope: "pop",
  fame_estimated: false,
  genre_imputed: false,
  artist_fame: 500000,
  shap_context: {
    fame: 3.4,
    genre: 30.3,
  },
  shap_audio: {
    "overall timbre": 0.8,
    "dynamic range": 0.7,
    "note density": 0.4,
    brightness: -0.3,
    tempo: -0.2,
    percussiveness: 0.1,
  },
  audio_features: {
    lb_tempo: 121.9,
    lb_onset_rate: 3.42,
    lb_rms_mean: 0.181,
    lb_centroid_mean: 2431.7,
    lb_flatness_mean: 0.032,
    lb_mfcc1_mean: -102.4,
  },
  warnings: [],
  explanation_source: "template",
  explanation:
    "This track scores 74.5 out of 100. Starting from a typical track at 38.7, " +
    "artist fame adds 3.4 and the style adds 30.3 points. The recording itself " +
    "adds 2.1 points, which ranks it in the 73rd percentile among pop tracks. " +
    "The main audio drivers are overall timbre (+0.8) and dynamic range (+0.7). " +
    "Audio typically moves a score by only a few points; fame and genre dominate " +
    "popularity, which is why this breakdown keeps them apart.",
};
