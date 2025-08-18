"use client";
import { useEffect, useRef, useState } from "react";

export default function Page() {
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<{ status: string; progress: number; message: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const backend = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

  const startPolling = (jid: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch(`${backend}/status/${jid}`);
        const s = await r.json();
        setStatus(s);
        if (s.status === "done" || s.status === "error") {
          if (pollRef.current) clearInterval(pollRef.current);
        }
      } catch (e) {
        // ignore transient errors
      }
    }, 1000);
  };

  const onUpload = async () => {
    setError(null);
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch(`${backend}/upload`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setJobId(data.job_id);
      setStatus({ status: "processing", progress: 0, message: "queued" });
      startPolling(data.job_id);
    } catch (e: any) {
      setError(e?.message || "Upload failed");
    }
  };

  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  const videoUrl = jobId ? `${backend}/result/${jobId}/video` : null;
  const metricsUrl = jobId ? `${backend}/result/${jobId}/metrics` : null;

  return (
    <div>
      <div style={{ background: "#0f2235", border: "1px solid #24425f", borderRadius: 8, padding: 16 }}>
        <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        <button onClick={onUpload} style={{ marginLeft: 12, padding: "8px 14px", background: "#2a8cff", color: "white", border: 0, borderRadius: 6, cursor: "pointer" }} disabled={!file}>
          Upload & Process
        </button>
        {error && <div style={{ color: "#ff7b7b", marginTop: 8 }}>Error: {error}</div>}
        {status && (
          <div style={{ marginTop: 12 }}>
            <div style={{ height: 10, width: "100%", background: "#1a2e44", borderRadius: 6, overflow: "hidden" }}>
              <div style={{ height: "100%", width: `${Math.round((status.progress || 0) * 100)}%`, background: "#31d07f" }} />
            </div>
            <div style={{ fontSize: 12, opacity: 0.9, marginTop: 6 }}>
              {status.status} — {status.message}
            </div>
          </div>
        )}
      </div>

      {status?.status === "done" && (
        <div style={{ marginTop: 24 }}>
          <h3 style={{ marginTop: 0 }}>Results</h3>
          <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
            <a href={videoUrl!} target="_blank" style={{ color: "#9ad1ff" }}>Download MP4</a>
            <a href={metricsUrl!} target="_blank" style={{ color: "#9ad1ff" }}>Download metrics JSON</a>
          </div>
          <div style={{ marginTop: 12 }}>
            <video src={videoUrl!} controls style={{ maxWidth: "100%", border: "1px solid #24425f", borderRadius: 8 }} />
          </div>
        </div>
      )}
    </div>
  );
}
