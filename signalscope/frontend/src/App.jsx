import React, { useCallback, useRef, useState } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

function useDropZone(onFile) {
  const [dragging, setDragging] = useState(false);
  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) onFile(file);
  }, [onFile]);
  const onDragOver = useCallback((e) => { e.preventDefault(); setDragging(true); }, []);
  const onDragLeave = useCallback(() => setDragging(false), []);
  return { dragging, onDrop, onDragOver, onDragLeave };
}

function Waveform() {
  // decorative signal motif; one orchestrated animation, not scattered per-element
  return (
    <svg className="waveform" viewBox="0 0 400 120" preserveAspectRatio="none" aria-hidden="true">
      <path
        className="waveform-path"
        d="M0,60 C20,20 40,100 60,60 C80,20 100,100 120,60 C140,10 160,110 180,60
           C200,25 220,95 240,60 C260,15 280,105 300,60 C320,25 340,95 360,60 C380,30 400,90 400,60"
        fill="none"
      />
    </svg>
  );
}

function ConfidenceDial({ label, pFake }) {
  const isFake = pFake >= 0.5;
  const pct = Math.round((isFake ? pFake : 1 - pFake) * 100);
  const color = isFake ? "var(--flag)" : "var(--real)";
  return (
    <div className="dial">
      <div className="dial-number" style={{ color }}>{pct}<span className="dial-pct">%</span></div>
      <div className="dial-caption">{label}</div>
    </div>
  );
}

function VerdictPanel({ prediction }) {
  if (!prediction) return null;
  const isFake = prediction.label === "ai-generated";
  return (
    <section className="panel panel--verdict" style={{ borderColor: isFake ? "var(--flag)" : "var(--real)" }}>
      <div className="panel-eyebrow">core verdict</div>
      <h2 className="verdict-headline">
        {prediction.presentation}
      </h2>
      <ConfidenceDial label="calibrated confidence" pFake={prediction.p_fake} />
      <p className="verdict-note">
        Presented as a likelihood, not an accusation. Confidence is
        temperature-calibrated on validation data, not raw softmax output.
      </p>
    </section>
  );
}

function ExplanationPanel({ explanation }) {
  if (!explanation) return null;
  if (explanation.error) {
    return (
      <section className="panel panel--muted">
        <div className="panel-eyebrow">explanation (module A)</div>
        <p className="panel-error">Explanation unavailable: {explanation.error}</p>
      </section>
    );
  }
  return (
    <section className="panel panel--wide">
      <div className="panel-eyebrow">explanation (module A)</div>
      {explanation.heatmap_base64 && (
        <img
          className="heatmap"
          src={`data:image/jpeg;base64,${explanation.heatmap_base64}`}
          alt="Grad-CAM heat-map highlighting the image region driving the verdict"
        />
      )}
      <p className="explanation-text">{explanation.explanation}</p>
      <div className="localisation-stat">
        <span className="mono">{Math.round((explanation.localisation?.area_frac || 0) * 100)}%</span>
        <span className="text-muted"> of frame highlighted · region: {explanation.localisation?.region}</span>
      </div>
    </section>
  );
}

function AttributionPanel({ attribution }) {
  if (!attribution || attribution.error) return null;
  return (
    <section className="panel">
      <div className="panel-eyebrow">generator family (module B)</div>
      <div className="attribution-family">{attribution.family}</div>
      <div className="text-muted small">confidence {Math.round(attribution.confidence * 100)}% · {attribution.method}</div>
    </section>
  );
}

function ProvenancePanel({ provenance }) {
  if (!provenance || provenance.error) return null;
  return (
    <section className="panel">
      <div className="panel-eyebrow">provenance signals (module D)</div>
      <ul className="reasoning-list">
        {provenance.reasoning?.map((r, i) => <li key={i}>{r}</li>)}
      </ul>
      <div className="text-muted small mono">
        visual p(fake) {provenance.visual_p_fake} → combined {provenance.combined_p_fake}
      </div>
    </section>
  );
}

function MultimodalPanel({ multimodal }) {
  if (!multimodal || multimodal.error) return null;
  return (
    <section className="panel">
      <div className="panel-eyebrow">caption consistency (module E)</div>
      <div className="attribution-family">{multimodal.verdict}</div>
      <div className="text-muted small mono">similarity {multimodal.similarity}</div>
    </section>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [caption, setCaption] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const handleFile = useCallback((f) => {
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const { dragging, onDrop, onDragOver, onDragLeave } = useDropZone(handleFile);

  const runScan = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      if (caption.trim()) form.append("caption", caption.trim());
      const res = await fetch(`${API_BASE}/full-scan`, { method: "POST", body: form });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Request failed (${res.status})`);
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="hero">
        <div className="hero-left">
          <div className="wordmark">SignalScope</div>
          <p className="tagline">
            Telling real from synthetic. Drop an image to get a calibrated
            likelihood, a localised explanation, and — where present — its
            provenance signals. Nothing here is an accusation.
          </p>
        </div>
        <div className="hero-right"><Waveform /></div>
      </header>

      <main className="content">
        <div
          className={`dropzone ${dragging ? "dropzone--active" : ""} ${loading ? "dropzone--scanning" : ""}`}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
          {previewUrl ? (
            <img className="preview" src={previewUrl} alt="Selected upload preview" />
          ) : (
            <div className="dropzone-copy">
              <div className="dropzone-title">Drop an image, or click to choose one</div>
              <div className="dropzone-sub text-muted">JPEG or PNG · analysed locally by your own model</div>
            </div>
          )}
          {loading && <div className="scanline" />}
        </div>

        <div className="controls">
          <input
            className="caption-input"
            type="text"
            placeholder="Optional caption to check for consistency (module E)"
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
          />
          <button className="scan-button" onClick={runScan} disabled={!file || loading}>
            {loading ? "Analysing…" : "Run scan"}
          </button>
        </div>

        {error && <div className="error-banner">{error}</div>}

        {result && (
          <div className="results-grid">
            <VerdictPanel prediction={result.prediction} />
            <ExplanationPanel explanation={result.explanation} />
            <AttributionPanel attribution={result.attribution} />
            <ProvenancePanel provenance={result.provenance} />
            <MultimodalPanel multimodal={result.multimodal} />
          </div>
        )}
      </main>

      <footer className="footer">
        Built for the SIH-2026 SignalScope challenge · classifies general
        synthetic imagery only — not face-swap deepfakes of real people, and
        not political or real-event claims.
      </footer>
    </div>
  );
}
