export const metadata = { title: 'ApexTracer-Lite', description: 'Golf shot tracer' };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, Arial', background: '#0b1724', color: 'white' }}>
        <div style={{ maxWidth: 980, margin: '0 auto', padding: 24 }}>
          <h1 style={{ marginBottom: 8 }}>ApexTracer-Lite</h1>
          <p style={{ opacity: 0.8, marginTop: 0 }}>Upload a golf shot video to get a traced overlay and metrics.</p>
          {children}
        </div>
      </body>
    </html>
  );
}
