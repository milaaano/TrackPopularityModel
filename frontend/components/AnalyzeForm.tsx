"use client";

import { useRef, useState } from "react";
import type { DragEvent, KeyboardEvent } from "react";
import { GENRES } from "../lib/genres";
import type { BackendState, SubmitPayload } from "../lib/types";

const MAX_MB = 25;
const AUDIO_TYPES = /\.(mp3|wav|m4a|ogg|flac)$/i;

interface AnalyzeFormProps {
  onSubmit: (payload: SubmitPayload) => void;
  busy: boolean;
  backendState: BackendState;
}

export default function AnalyzeForm({ onSubmit, busy, backendState }: AnalyzeFormProps) {
  const [file, setFile] = useState<File | null>(null);
  const [artist, setArtist] = useState("");
  const [genre, setGenre] = useState("");
  const [explicit, setExplicit] = useState(false);
  const [fieldError, setFieldError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function takeFile(f?: File | null) {
    if (!f) return;
    if (!AUDIO_TYPES.test(f.name)) {
      setFieldError("That doesn't look like an audio file (mp3 / wav / m4a / ogg / flac).");
      return;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setFieldError(`File is ${(f.size / 1048576).toFixed(1)} MB — the limit is ${MAX_MB} MB.`);
      return;
    }
    setFieldError(null);
    setFile(f);
  }

  function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!file) return setFieldError("Add an audio file first.");
    if (!artist.trim()) return setFieldError("Artist name is required — fame is half the breakdown.");
    if (!genre) return setFieldError("Pick the closest genre.");
    setFieldError(null);
    onSubmit({ file, artist: artist.trim(), genre, explicit });
  }

  return (
    <form className="card" onSubmit={submit} noValidate>
      <h2>Analyze a song</h2>
      <p className="hint">
        The audio drives the song&rsquo;s own score; the artist name is used to look up
        fame (Last.fm), which sets the context baseline.
      </p>

      <div
        className={`drop ${dragOver ? "over" : ""} ${file ? "hasfile" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e: DragEvent<HTMLDivElement>) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e: DragEvent<HTMLDivElement>) => {
          e.preventDefault();
          setDragOver(false);
          takeFile(e.dataTransfer.files?.[0]);
        }}
        role="button"
        tabIndex={0}
        onKeyDown={(e: KeyboardEvent<HTMLDivElement>) =>
          (e.key === "Enter" || e.key === " ") && inputRef.current?.click()
        }
        aria-label="Upload audio file"
      >
        {file ? (
          <>
            <div className="big">{file.name}</div>
            <div className="small">
              {(file.size / 1048576).toFixed(1)} MB — click to replace
            </div>
          </>
        ) : (
          <>
            <div className="big">Drop your track here, or click to browse</div>
            <div className="small">mp3 · wav · m4a · ogg · flac — up to {MAX_MB} MB</div>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="audio/*,.mp3,.wav,.m4a,.ogg,.flac"
          hidden
          onChange={(e) => takeFile(e.target.files?.[0])}
        />
      </div>

      <div className="grid2" style={{ marginTop: 18 }}>
        <div>
          <label className="f" htmlFor="artist">Artist name</label>
          <input
            id="artist"
            type="text"
            placeholder="e.g. Mitski"
            value={artist}
            onChange={(e) => setArtist(e.target.value)}
            autoComplete="off"
          />
        </div>
        <div>
          <label className="f" htmlFor="genre">Genre</label>
          <select id="genre" value={genre} onChange={(e) => setGenre(e.target.value)}>
            <option value="" disabled>Choose the closest…</option>
            {GENRES.map((g) => (
              <option key={g} value={g}>{g}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="switchrow" style={{ marginTop: 16 }}>
        <button
          type="button"
          className="switch"
          role="switch"
          aria-checked={explicit}
          aria-label="Explicit lyrics"
          onClick={() => setExplicit(!explicit)}
        >
          <span className="knob" />
        </button>
        <span className="lab">Explicit lyrics</span>
      </div>

      {fieldError && (
        <div className="banner err" style={{ marginTop: 16 }}>
          <span className="ic">⚠</span> {fieldError}
        </div>
      )}

      <div className="formfoot">
        <button className="btn" type="submit" disabled={busy}>
          {busy ? (<><span className="spin" />Analyzing…</>) : "Analyze"}
        </button>
        <span className="note">
          {backendState === "offline"
            ? "Backend offline — you can still explore a sample result below."
            : "Warm analysis takes ~10–30 s; first run after a cold start can take up to 2 min."}
        </span>
      </div>
    </form>
  );
}
