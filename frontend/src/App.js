import { useState } from "react";
import "./App.css";

const SAMPLE_ARTICLES = [
  {
    name: "Real-style sample",
    title: "City Council Approves Downtown Transit Expansion",
    text: "The city council voted 9-2 to approve the next phase of a public transit expansion project after a six-hour hearing. The proposal includes new bus lanes, accessibility upgrades, and a construction timeline that begins this fall.",
  },
  {
    name: "Fake-style sample",
    title: "Scientists Furious About One Secret Trick",
    text: "A shocking secret has been hidden from the public for years. Experts do not want you to know this one simple trick that changes everything overnight. Industry insiders are panicking and trying to bury the truth.",
  },
];

function App() {
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeSample, setActiveSample] = useState("");
  const API_URL = "http://localhost:5001/predict";

  const titleLength = title.trim().length;
  const textLength = text.trim().length;
  const textWordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const canSubmit = titleLength >= 8 && textLength >= 40 && !loading;

  const confidencePct = result && typeof result.score === "number"
    ? Math.max(0, Math.min(100, result.score * 100))
    : 0;

  const fakePct = result && typeof result.fake_score === "number"
    ? Math.max(0, Math.min(100, result.fake_score * 100))
    : null;

  const realPct = result && typeof result.real_score === "number"
    ? Math.max(0, Math.min(100, result.real_score * 100))
    : null;

  const setSample = (sample) => {
    setTitle(sample.title);
    setText(sample.text);
    setResult(null);
    setActiveSample(sample.name);
  };

  const clearInputs = () => {
    setTitle("");
    setText("");
    setResult(null);
    setActiveSample("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const resp = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, text }),
      });

      if (!resp.ok) {
        throw new Error(`Backend returned ${resp.status}`);
      }

      const data = await resp.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setResult({ error: "Request failed" });
    }

    setLoading(false);
  };

  return (
    <div className="page-shell">
      <div className="background-orb orb-a" />
      <div className="background-orb orb-b" />

      <main className="app-card">
        <header className="hero">
          <p className="eyebrow">Neural Content Scanner</p>
          <h1>Fake News Detector</h1>
          <p className="subhead">
            Paste a headline and article text to get a model prediction.
          </p>
        </header>

        <section className="sample-row" aria-label="Quick samples">
          {SAMPLE_ARTICLES.map((sample) => (
            <button
              key={sample.name}
              type="button"
              className={`sample-chip ${activeSample === sample.name ? "active" : ""}`}
              onClick={() => setSample(sample)}
            >
              {sample.name}
            </button>
          ))}
          <button type="button" className="sample-chip ghost" onClick={clearInputs}>
            Clear
          </button>
        </section>

        <form onSubmit={handleSubmit} className="analyze-form">
          <label>
            <span>Title</span>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Enter article headline"
              required
            />
          </label>

          <label>
            <span>Text</span>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter full article text"
              rows={7}
              required
            />
          </label>

          <div className="live-stats">
            <div>
              <strong>{titleLength}</strong>
              <span>Title chars</span>
            </div>
            <div>
              <strong>{textLength}</strong>
              <span>Body chars</span>
            </div>
            <div>
              <strong>{textWordCount}</strong>
              <span>Words</span>
            </div>
          </div>

          <button type="submit" disabled={!canSubmit} className="submit-btn">
            {loading ? "Analyzing..." : "Check News"}
          </button>

          {!canSubmit && (
            <p className="hint">Add at least 8 title chars and 40 body chars to analyze.</p>
          )}
        </form>

        {result && (
          <section className="result-panel">
            {"error" in result ? (
              <p className="error">{result.error}</p>
            ) : (
              <>
                <p className="result-label">
                  Model Label:
                  <span className={result.label === "fake" ? "fake" : "real"}>
                    {result.label.toUpperCase()}
                  </span>
                </p>

                <div className="confidence-wrap">
                  <div className="confidence-head">
                    <span>Confidence</span>
                    <strong>{confidencePct.toFixed(1)}%</strong>
                  </div>
                  <div className="confidence-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow={confidencePct}>
                    <div
                      className={`confidence-fill ${result.label === "fake" ? "fake" : "real"}`}
                      style={{ width: `${confidencePct}%` }}
                    />
                  </div>
                </div>

                {fakePct !== null && realPct !== null && (
                  <div className="probability-grid">
                    <p>Real score: {realPct.toFixed(1)}%</p>
                    <p>Fake score: {fakePct.toFixed(1)}%</p>
                  </div>
                )}

                {typeof result.threshold === "number" && (
                  <p className="threshold-note">
                    Decision threshold for fake: {(result.threshold * 100).toFixed(0)}%
                  </p>
                )}
              </>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
