import { useState, useEffect, useRef } from "react";
import {
  LineChart, Line, ComposedChart, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Scatter
} from "recharts";

// ─── Config ───────────────────────────────────────────────────────────────────
const WORKER_URL = "https://vdgr-signal-worker.b2mucoach.workers.dev";

// ─── Design Tokens ────────────────────────────────────────────────────────────
const C = {
  signal: {
    NONE:    "#4B5563",
    LOW:     "#EAB308",
    MEDIUM:  "#F97316",
    HIGH:    "#EF4444",
    EXTREME: "#8B5CF6",
    WATCH:   "#EAB308",
  },
  bg: { primary: "#0A0A0A", secondary: "#141414", card: "#1C1C1C", border: "#252525" },
  text: { primary: "#F0F0F0", secondary: "#D1D5DB", muted: "#9CA3AF" },
  accent: "#3B82F6",
  green:  "#10B981",
  red:    "#EF4444",
};

function normaliseTier(tier) {
  if (!tier) return "NONE";
  return tier === "WATCH" ? "LOW" : tier;
}

async function api(path, options = {}) {
  const res = await fetch(`${WORKER_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

const fmt = {
  aud:   v => `$${Number(v).toLocaleString("en-AU", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
  date:  d => new Date(d).toLocaleDateString("en-AU", { weekday: "short", day: "numeric", month: "short" }),
  short: d => new Date(d).toLocaleDateString("en-AU", { day: "2-digit", month: "short" }),
  mmm:   d => new Date(d).toLocaleDateString("en-AU", { month: "short", year: "2-digit" }),
};

function buildXTicks(data, maxTicks = 6) {
  if (!data.length) return [];
  const step = Math.max(1, Math.floor(data.length / maxTicks));
  return data.filter((_, i) => i % step === 0).map(d => d.date);
}

// ─── Global Styles ────────────────────────────────────────────────────────────
const globalStyle = `
  @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0A0A0A; color: #F0F0F0; font-family: 'DM Sans', sans-serif; }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: #141414; }
  ::-webkit-scrollbar-thumb { background: #2A2A2A; border-radius: 2px; }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.7;transform:scale(1.03)} }
  @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
  @keyframes shimmer { 0%{background-position:-400px 0} 100%{background-position:400px 0} }
  @keyframes slideUp { from{transform:translateY(100%);opacity:0} to{transform:translateY(0);opacity:1} }
  @keyframes spin { to{transform:rotate(360deg)} }
  .fade-in { animation: fadeIn .35s ease both; }
  .slide-up { animation: slideUp .3s cubic-bezier(.16,1,.3,1) both; }
  .skeleton { background: linear-gradient(90deg,#1C1C1C 25%,#252525 50%,#1C1C1C 75%); background-size:400px 100%; animation:shimmer 1.4s infinite; border-radius:6px; }
  .tab-content { padding: 72px 16px 100px; max-width: 520px; margin: 0 auto; min-height: calc(100vh - 56px); }
  .card { background:#1C1C1C; border:1px solid #252525; border-radius:12px; padding:16px; margin-bottom:12px; }
  .btn { display:inline-flex;align-items:center;justify-content:center;gap:6px;padding:8px 16px;border-radius:8px;border:none;cursor:pointer;font-family:'DM Sans',sans-serif;font-size:14px;font-weight:500;transition:all .15s; }
  .btn:active { transform:scale(.97); }
  .btn-primary { background:#3B82F6;color:#fff; }
  .btn-primary:hover { background:#2563EB; }
  .btn-ghost { background:#252525;color:#9CA3AF; }
  .btn-ghost:hover { background:#2E2E2E;color:#F0F0F0; }
  .pill-btn { padding:5px 12px;border-radius:20px;border:1px solid #252525;background:transparent;color:#9CA3AF;font-family:'DM Sans',sans-serif;font-size:13px;cursor:pointer;transition:all .15s; }
  .pill-btn.active { background:#3B82F6;color:#fff;border-color:#3B82F6; }
  .pill-btn:hover:not(.active) { border-color:#3B82F6;color:#F0F0F0; }
  .mono { font-family:'JetBrains Mono',monospace; }
  .chat-bubble-user { background:#1E3A5F;border-radius:12px 12px 2px 12px;padding:10px 14px;margin-left:40px; }
  .chat-bubble-ai { background:#1C1C1C;border:1px solid #252525;border-radius:12px 12px 12px 2px;padding:10px 14px;margin-right:40px; }
  input,textarea { background:#141414;border:1px solid #252525;border-radius:8px;color:#F0F0F0;font-family:'DM Sans',sans-serif;font-size:14px;padding:10px 12px;width:100%;outline:none;transition:border-color .15s; }
  input:focus,textarea:focus { border-color:#3B82F6; }
  input::placeholder,textarea::placeholder { color:#9CA3AF; }
  .overlay { position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:50;display:flex;align-items:flex-end; }
  .sheet { background:#1C1C1C;border:1px solid #252525;border-radius:16px 16px 0 0;padding:20px;width:100%;max-width:520px;margin:0 auto; }
`;

// ─── Shared Components ────────────────────────────────────────────────────────
function Spinner({ size = 18 }) {
  return <div style={{ width:size, height:size, border:"2px solid #252525", borderTop:"2px solid #3B82F6", borderRadius:"50%", animation:"spin .7s linear infinite", flexShrink:0 }} />;
}

function SignalBadge({ signal, size = "lg" }) {
  const tier = normaliseTier(signal);
  const color = C.signal[tier] || C.signal.NONE;
  const pulse = ["HIGH","EXTREME"].includes(tier);
  return (
    <div style={{ display:"inline-flex", alignItems:"center", justifyContent:"center", background:`${color}18`, border:`2px solid ${color}`, borderRadius:999, padding: size==="lg" ? "10px 28px" : "4px 12px", animation: pulse ? "pulse 2s ease-in-out infinite" : "none" }}>
      <span style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize: size==="lg" ? 36 : 13, fontWeight:800, color, letterSpacing:2 }}>{tier}</span>
    </div>
  );
}

function SectionLabel({ children }) {
  return <div style={{ fontSize:12, color:"#9CA3AF", textTransform:"uppercase", letterSpacing:.8, marginBottom:10 }}>{children}</div>;
}

function ErrorState({ message, onRetry }) {
  return (
    <div style={{ textAlign:"center", padding:"40px 20px" }}>
      <div style={{ fontSize:13, color:C.text.muted, marginBottom:12 }}>{message}</div>
      {onRetry && <button className="btn btn-ghost" onClick={onRetry}>Retry</button>}
    </div>
  );
}

function SkeletonCard({ height = 80 }) {
  return <div className="card"><div className="skeleton" style={{ height }} /></div>;
}

// ─── Drawdown Bands Component ─────────────────────────────────────────────────
function DrawdownBands({ price, high52w, noCard }) {
  if (!price || !high52w) return null;
  const currentDd = ((price - high52w) / high52w) * 100;
  const bands = [5, 10, 15, 20];
  const content = (
    <>
      <SectionLabel>Price vs 52-Week High</SectionLabel>
      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:12 }}>
        <div>
          <div style={{ fontSize:11, color:C.text.muted }}>52-Week High</div>
          <div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:16, color:C.text.primary }}>${Number(high52w).toFixed(2)}</div>
        </div>
        <div style={{ textAlign:"right" }}>
          <div style={{ fontSize:11, color:C.text.muted }}>Current Drawdown</div>
          <div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:16, color: currentDd < -10 ? C.red : currentDd < -5 ? C.signal.MEDIUM : C.green }}>
            {currentDd.toFixed(1)}%
          </div>
        </div>
      </div>
      <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
        {bands.map(pct => {
          const threshold = high52w * (1 - pct / 100);
          const reached = price <= threshold;
          const diff = price - threshold;
          return (
            <div key={pct} style={{ display:"flex", alignItems:"center", justifyContent:"space-between", padding:"7px 10px", borderRadius:7, background: reached ? `${C.red}15` : "#141414", border:`1px solid ${reached ? C.red+"40" : "#252525"}` }}>
              <div style={{ display:"flex", alignItems:"center", gap:8 }}>
                <div style={{ width:8, height:8, borderRadius:"50%", background: reached ? C.red : C.text.muted }} />
                <span style={{ fontSize:13, color: reached ? C.text.primary : C.text.muted }}>{pct}% below high</span>
                <span style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:11, color:C.text.muted }}>${threshold.toFixed(2)}</span>
              </div>
              <span style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:12, color: reached ? C.red : C.green }}>
                {reached ? `✓ ${Math.abs(diff).toFixed(2)} below` : `+${diff.toFixed(2)} away`}
              </span>
            </div>
          );
        })}
      </div>
    </>
  );
  return noCard ? content : <div className="card">{content}</div>;
}

// ─── SIGNAL JOURNEY ───────────────────────────────────────────────────────────
function SignalJourney({ rsi, vix, drawdown }) {
  if (!rsi || !vix || drawdown == null) return null;

  const thresholds = {
    WATCH:   { rsi:50, vix:18, dd:-5 },
    MEDIUM:  { rsi:45, vix:20, dd:-10 },
    HIGH:    { rsi:35, vix:25, dd:-15 },
    EXTREME: { rsi:30, vix:30, dd:-20 },
  };

  // Helper to calculate distance text for an indicator
  function getDistanceText(value, metThresholds, type) {
    const levels = ["WATCH", "MEDIUM", "HIGH", "EXTREME"];
    const metLevels = [];
    const unmetLevels = [];

    for (const level of levels) {
      if (metThresholds.includes(level)) {
        metLevels.push(level);
      } else {
        unmetLevels.push(level);
      }
    }

    if (unmetLevels.length === 0) {
      // All met - show the highest met level
      const highestMet = metLevels[metLevels.length - 1];
      return { text: `✓ ${highestMet} met`, color: C.signal[highestMet] };
    }

    // Show distance to next unmet level
    const nextLevel = unmetLevels[0];
    const nextThreshold = thresholds[nextLevel];
    let dist;
    if (type === "rsi") dist = (value - nextThreshold.rsi).toFixed(1);
    else if (type === "vix") dist = (nextThreshold.vix - value).toFixed(1);
    else if (type === "dd") dist = (Math.abs(value) - Math.abs(nextThreshold.dd)).toFixed(1);

    return { text: `${dist} to ${nextLevel}`, color: C.signal[nextLevel] };
  }

  // RSI segments (70-50 NONE, 50-45 LOW, 45-35 MEDIUM, 35-30 HIGH, 30-25 EXTREME)
  const rsiMin = 25, rsiMax = 70;
  const rsiSegments = [
    { start:70, end:50, color:C.signal.NONE, tier:"NONE" },
    { start:50, end:45, color:C.signal.LOW, tier:"WATCH" },
    { start:45, end:35, color:C.signal.MEDIUM, tier:"MEDIUM" },
    { start:35, end:30, color:C.signal.HIGH, tier:"HIGH" },
    { start:30, end:25, color:C.signal.EXTREME, tier:"EXTREME" },
  ];
  const rsiPos = ((rsiMax - rsi) / (rsiMax - rsiMin)) * 100;
  const rsiMetThresholds = [];
  if (rsi < 50) rsiMetThresholds.push("WATCH");
  if (rsi < 45) rsiMetThresholds.push("MEDIUM");
  if (rsi < 35) rsiMetThresholds.push("HIGH");
  if (rsi < 30) rsiMetThresholds.push("EXTREME");
  const rsiDistance = getDistanceText(rsi, rsiMetThresholds, "rsi");

  // VIX segments (10-18 NONE, 18-20 LOW, 20-25 MEDIUM, 25-30 HIGH, 30-35 EXTREME)
  const vixMin = 10, vixMax = 35;
  const vixSegments = [
    { start:10, end:18, color:C.signal.NONE, tier:"NONE" },
    { start:18, end:20, color:C.signal.LOW, tier:"WATCH" },
    { start:20, end:25, color:C.signal.MEDIUM, tier:"MEDIUM" },
    { start:25, end:30, color:C.signal.HIGH, tier:"HIGH" },
    { start:30, end:35, color:C.signal.EXTREME, tier:"EXTREME" },
  ];
  const vixPos = ((vix - vixMin) / (vixMax - vixMin)) * 100;
  const vixMetThresholds = [];
  if (vix > 18) vixMetThresholds.push("WATCH");
  if (vix > 20) vixMetThresholds.push("MEDIUM");
  if (vix > 25) vixMetThresholds.push("HIGH");
  if (vix > 30) vixMetThresholds.push("EXTREME");
  const vixDistance = getDistanceText(vix, vixMetThresholds, "vix");

  // Drawdown segments (0 to -5 NONE, -5 to -10 LOW, -10 to -15 MEDIUM, -15 to -20 HIGH)
  const ddMin = 0, ddMax = -20;
  const ddSegments = [
    { start:0, end:-5, color:C.signal.NONE, tier:"NONE" },
    { start:-5, end:-10, color:C.signal.LOW, tier:"WATCH" },
    { start:-10, end:-15, color:C.signal.MEDIUM, tier:"MEDIUM" },
    { start:-15, end:-20, color:C.signal.HIGH, tier:"HIGH" },
  ];
  const ddPos = ((drawdown - ddMin) / (ddMax - ddMin)) * 100;
  const ddMetThresholds = [];
  if (drawdown < -5) ddMetThresholds.push("WATCH");
  if (drawdown < -10) ddMetThresholds.push("MEDIUM");
  if (drawdown < -15) ddMetThresholds.push("HIGH");
  const ddDistance = getDistanceText(drawdown, ddMetThresholds, "dd");

  // Value colors based on current tier
  let rsiValueColor = C.text.primary;
  if (rsi < 30) rsiValueColor = C.signal.EXTREME;
  else if (rsi < 35) rsiValueColor = C.signal.HIGH;
  else if (rsi < 45) rsiValueColor = C.signal.MEDIUM;
  else if (rsi < 50) rsiValueColor = C.signal.LOW;

  let vixValueColor = C.text.primary;
  if (vix > 30) vixValueColor = C.signal.EXTREME;
  else if (vix > 25) vixValueColor = C.signal.HIGH;
  else if (vix > 20) vixValueColor = C.signal.MEDIUM;
  else if (vix > 18) vixValueColor = C.signal.LOW;

  let ddValueColor = C.text.primary;
  if (drawdown <= -15) ddValueColor = C.signal.HIGH;
  else if (drawdown <= -10) ddValueColor = C.signal.MEDIUM;
  else if (drawdown <= -5) ddValueColor = C.signal.LOW;

  // Render segmented bar
  function SegmentedBar({ segments, position, min, max }) {
    const totalRange = Math.abs(max - min);
    return (
      <div style={{ position:"relative", height:6 }}>
        <div style={{ display:"flex", width:"100%", height:"100%", overflow:"hidden" }}>
          {segments.map((seg, i) => {
            const segWidth = (Math.abs(seg.end - seg.start) / totalRange) * 100;
            const isFirst = i === 0;
            const isLast = i === segments.length - 1;
            return (
              <div key={i} style={{
                width:`${segWidth}%`,
                height:"100%",
                background:seg.color,
                borderTopLeftRadius: isFirst ? 999 : 0,
                borderBottomLeftRadius: isFirst ? 999 : 0,
                borderTopRightRadius: isLast ? 999 : 0,
                borderBottomRightRadius: isLast ? 999 : 0,
              }} />
            );
          })}
        </div>
        {/* White indicator line */}
        <div style={{
          position:"absolute",
          left:`${Math.min(100, Math.max(0, position))}%`,
          top:-4,
          width:2,
          height:14,
          background:"#F0F0F0",
          borderRadius:1,
          transform:"translateX(-50%)",
          pointerEvents:"none"
        }} />
      </div>
    );
  }

  return (
    <div className="card" style={{ marginBottom:12 }}>
      <SectionLabel>Signal Indicators</SectionLabel>

      {/* RSI Row */}
      <div style={{ marginBottom:16 }}>
        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          <div style={{ minWidth:70 }}>
            <div style={{ fontSize:12, color:C.text.muted }}>RSI</div>
            <div style={{ fontSize:18, fontWeight:600, color:rsiValueColor, fontFamily:"'JetBrains Mono',monospace" }}>{rsi.toFixed(1)}</div>
          </div>
          <div style={{ flex:1 }}>
            <SegmentedBar segments={rsiSegments} position={rsiPos} min={rsiMin} max={rsiMax} />
            <div style={{ position:"relative", height:14, marginTop:2 }}>
              {[50, 45, 35, 30].map(threshold => {
                const pos = ((rsiMax - threshold) / (rsiMax - rsiMin)) * 100;
                return (
                  <div key={threshold} style={{
                    position:"absolute",
                    left:`${pos}%`,
                    transform:"translateX(-50%)",
                    fontSize:9,
                    color:C.text.muted
                  }}>{threshold}</div>
                );
              })}
            </div>
          </div>
          <div style={{ minWidth:90, textAlign:"right", fontSize:11, color:rsiDistance.color }}>{rsiDistance.text}</div>
        </div>
      </div>

      {/* VIX Row */}
      <div style={{ marginBottom:16 }}>
        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          <div style={{ minWidth:70 }}>
            <div style={{ fontSize:12, color:C.text.muted }}>VIX</div>
            <div style={{ fontSize:18, fontWeight:600, color:vixValueColor, fontFamily:"'JetBrains Mono',monospace" }}>{vix.toFixed(1)}</div>
          </div>
          <div style={{ flex:1 }}>
            <SegmentedBar segments={vixSegments} position={vixPos} min={vixMin} max={vixMax} />
            <div style={{ position:"relative", height:14, marginTop:2 }}>
              {[18, 20, 25, 30].map(threshold => {
                const pos = ((threshold - vixMin) / (vixMax - vixMin)) * 100;
                return (
                  <div key={threshold} style={{
                    position:"absolute",
                    left:`${pos}%`,
                    transform:"translateX(-50%)",
                    fontSize:9,
                    color:C.text.muted
                  }}>{threshold}</div>
                );
              })}
            </div>
          </div>
          <div style={{ minWidth:90, textAlign:"right", fontSize:11, color:vixDistance.color }}>{vixDistance.text}</div>
        </div>
      </div>

      {/* Drawdown Row */}
      <div>
        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          <div style={{ minWidth:70 }}>
            <div style={{ fontSize:12, color:C.text.muted }}>DRAWDOWN</div>
            <div style={{ fontSize:18, fontWeight:600, color:ddValueColor, fontFamily:"'JetBrains Mono',monospace" }}>{drawdown.toFixed(1)}%</div>
          </div>
          <div style={{ flex:1 }}>
            <SegmentedBar segments={ddSegments} position={ddPos} min={ddMin} max={ddMax} />
            <div style={{ position:"relative", height:14, marginTop:2, marginBottom:6 }}>
              {[-5, -10, -15].map(threshold => {
                const pos = ((threshold - ddMin) / (ddMax - ddMin)) * 100;
                return (
                  <div key={threshold} style={{
                    position:"absolute",
                    left:`${pos}%`,
                    transform:"translateX(-50%)",
                    fontSize:9,
                    color:C.text.muted
                  }}>{threshold}%</div>
                );
              })}
            </div>
          </div>
          <div style={{ minWidth:90, textAlign:"right", fontSize:11, color:ddDistance.color }}>{ddDistance.text}</div>
        </div>
      </div>
    </div>
  );
}

// ─── Signal Streak ────────────────────────────────────────────────────────────
function SignalStreak({ history }) {
  if (!history?.length) return null;
  const sorted = [...history].sort((a, b) => new Date(b.date) - new Date(a.date));
  const latest = sorted[0];
  const latestTier = normaliseTier(latest.signal_tier);
  let daysSince = 0;
  let streakCount = 0;
  if (latestTier === "NONE") {
    for (const s of sorted) { if (normaliseTier(s.signal_tier) !== "NONE") break; daysSince++; }
  } else {
    for (const s of sorted) { if (normaliseTier(s.signal_tier) === "NONE") break; streakCount++; }
  }
  if (latestTier === "NONE") {
    return (
      <div style={{ display:"flex", alignItems:"center", gap:10, padding:"10px 14px", background:"#141414", borderRadius:8, marginBottom:12 }}>
        <div style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:28, fontWeight:800, color:C.text.muted }}>{daysSince}</div>
        <div>
          <div style={{ fontSize:13, color:C.text.secondary }}>days since last signal</div>
          <div style={{ fontSize:11, color:C.text.muted }}>Markets are calm</div>
        </div>
      </div>
    );
  }
  const color = C.signal[latestTier];
  return (
    <div style={{ display:"flex", alignItems:"center", gap:10, padding:"10px 14px", background:`${color}10`, border:`1px solid ${color}30`, borderRadius:8, marginBottom:12 }}>
      <div style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:28, fontWeight:800, color }}>{streakCount}</div>
      <div>
        <div style={{ fontSize:13, color:C.text.secondary }}>consecutive signal day{streakCount !== 1 ? "s" : ""}</div>
        <div style={{ fontSize:11, color:C.text.muted }}>Active {latestTier} signal streak</div>
      </div>
    </div>
  );
}


// ─── TODAY TAB ────────────────────────────────────────────────────────────────
function TodayTab() {
  const [signal, setSignal] = useState(null);
  const [history, setHistory] = useState([]);
  const [perf, setPerf] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatRef = useRef(null);
  const [sparklineData, setSparklineData] = useState([]);
  const [sparklineLoading, setSparklineLoading] = useState(true);
  const [marketTab, setMarketTab] = useState("snapshot");

  async function loadData() {
    setLoading(true); setError(null);
    try {
      const [s, h, p] = await Promise.all([
        api("/signal/today"),
        api("/signal/history?days=30"),
        api("/performance").catch(() => null),
      ]);
      setSignal(s.signal);
      setHistory(h.history || []);
      setPerf(p?.summary || null);
    } catch { setError("Unable to load today's signal."); }
    setLoading(false);
  }

  async function loadSparkline() {
    setSparklineLoading(true);
    try {
      const data = await api("/signal/chart?range=1mo");
      setSparklineData((data.chartData || []).map(d => ({ date: d.date, price: d.price })));
    } catch { setSparklineData([]); }
    setSparklineLoading(false);
  }

  useEffect(() => { loadData(); loadSparkline(); }, []);
  useEffect(() => { if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight; }, [messages]);

  async function sendChat() {
    if (!input.trim() || chatLoading) return;
    const userMsg = input.trim();
    setInput("");
    const newMessages = [...messages, { role:"user", content:userMsg }];
    setMessages(newMessages);
    setChatLoading(true);
    try {
      const data = await api("/analyst/chat", { method:"POST", body: JSON.stringify({ messages: newMessages, date: signal?.date }) });
      setMessages(m => [...m, { role:"assistant", content: data.response }]);
    } catch { setMessages(m => [...m, { role:"assistant", content:"Connection error. Please try again." }]); }
    setChatLoading(false);
  }

  // Human-readable one-line briefing
  function briefingLine(tier, rsi, vix, history) {
    if (["MEDIUM","HIGH","EXTREME"].includes(tier)) {
      const amt = signal?.recommended_amount;
      return amt > 0 ? `Buy signal active · $${amt} suggested` : `${tier} signal active`;
    }
    if (tier === "LOW") return `Early weakness · RSI ${rsi?.toFixed(1)}, VIX ${vix?.toFixed(1)}`;
    // NONE — count days since last signal
    const sorted = [...(history || [])].sort((a,b) => new Date(b.date)-new Date(a.date));
    let days = 0;
    for (const s of sorted) { if (normaliseTier(s.signal_tier) !== "NONE") break; days++; }
    return days > 0 ? `Markets quiet · ${days} days since last signal` : "Markets quiet";
  }

  // Calculate next update time (8am AEDT weekday)
  function getNextUpdate(signalDate) {
    if (!signalDate) return null;
    const now = new Date();
    // Convert current time to AEDT (UTC+10)
    const aedtNow = new Date(now.toLocaleString("en-US", { timeZone: "Australia/Sydney" }));
    const currentHour = aedtNow.getHours();

    // Start from tomorrow (since update runs at 8am and by the time users see it, it's already happened)
    let next = new Date(aedtNow);
    next.setDate(next.getDate() + 1);
    next.setHours(8, 0, 0, 0);

    // Skip weekends
    while (next.getDay() === 0 || next.getDay() === 6) {
      next.setDate(next.getDate() + 1);
    }

    // Calculate days difference
    const today = new Date(aedtNow.getFullYear(), aedtNow.getMonth(), aedtNow.getDate());
    const nextDay = new Date(next.getFullYear(), next.getMonth(), next.getDate());
    const diffDays = Math.round((nextDay - today) / 86400000);

    if (diffDays === 1) return "Tomorrow 8am AEDT";
    return next.toLocaleDateString("en-AU", { weekday:"short", day:"numeric", month:"short" }) + " 8am AEDT";
  }

  const dateStr = new Date().toLocaleDateString("en-AU", { weekday:"short", day:"numeric", month:"short" });

  if (loading) return (
    <div className="tab-content">
      <SkeletonCard height={180} />
      <SkeletonCard height={56} />
      <SkeletonCard height={80} />
      <SkeletonCard height={80} />
    </div>
  );

  if (error) return <div className="tab-content"><ErrorState message={error} onRetry={loadData} /></div>;

  const tier = normaliseTier(signal?.signal_tier);
  const signalColor = C.signal[tier];
  const rsi = signal ? Number(signal.rsi) : null;
  const vix = signal ? Number(signal.vix) : null;
  const price = signal ? Number(signal.vdgr_price) : null;
  const drawdown = signal ? Number(signal.drawdown_pct) : null;
  const high52w = signal ? Number(signal.high_52w) : null;
  const isActionable = ["MEDIUM","HIGH","EXTREME"].includes(tier);
  const profitColor = perf ? (perf.returnDollar >= 0 ? C.green : C.red) : C.text.muted;

  // Calculate days since last signal
  const sorted = [...(history || [])].sort((a,b) => new Date(b.date)-new Date(a.date));
  let daysSinceSignal = 0;
  for (const s of sorted) { if (normaliseTier(s.signal_tier) !== "NONE") break; daysSinceSignal++; }

  return (
    <div className="tab-content fade-in">

      {/* ── COMBINED SIGNAL HERO ── */}
      <div style={{ marginBottom:4 }}>
        <div className="card" style={{ padding:"20px 16px", marginBottom:12, border:`1px solid ${isActionable ? signalColor+"40" : "#252525"}`, maxHeight:160 }}>
          <div style={{ display:"flex", gap:16, alignItems:"center", minHeight:120 }}>
            {/* Left: Days counter (NONE state only) */}
            {tier === "NONE" && daysSinceSignal > 0 ? (
              <div style={{ flex:1, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center" }}>
                <div style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:56, fontWeight:800, color:C.text.primary, lineHeight:1 }}>
                  {daysSinceSignal}
                </div>
                <div style={{ fontSize:11, color:C.text.muted, marginTop:6, textAlign:"center", whiteSpace:"pre-line" }}>
                  {"days since\nlast signal"}
                </div>
              </div>
            ) : (
              <div style={{ flex:1, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center" }}>
                <div style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:56, fontWeight:800, color:C.text.primary, lineHeight:1 }}>
                  —
                </div>
                <div style={{ fontSize:11, color:C.text.muted, marginTop:6, textAlign:"center" }}>
                  Signal active
                </div>
              </div>
            )}

            {/* Vertical divider */}
            <div style={{ width:1, height:100, background:"#252525" }} />

            {/* Right: Signal badge and info */}
            <div style={{ flex:1, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:8 }}>
              <SignalBadge signal={tier} />
              {isActionable ? (
                <div style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:18, fontWeight:700, color:signalColor }}>
                  {fmt.aud(signal.recommended_amount)}
                </div>
              ) : (
                <div style={{ fontSize:13, color:C.text.muted }}>No investment</div>
              )}
              <div style={{ fontSize:11, color:C.text.muted, textAlign:"center" }}>
                {signal ? fmt.date(signal.date) : "—"}
              </div>
              {signal?.date && (
                <div style={{ fontSize:11, color:C.text.muted, textAlign:"center" }}>
                  {getNextUpdate(signal.date)}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Analyst summary excerpt */}
        {signal?.analyst_summary && (
          <div style={{ fontSize:13, color:C.text.secondary, lineHeight:1.6, marginBottom:12, padding:"0 4px" }}>
            {signal.analyst_summary}
          </div>
        )}

        {/* P&L strip — always visible */}
        {perf ? (
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:1, background:"#252525", borderRadius:10, overflow:"hidden", marginBottom:12 }}>
            {[
              { label:"Invested", value: fmt.aud(perf.totalInvested), color: C.text.primary },
              { label:"Value", value: perf.currentValue != null ? fmt.aud(perf.currentValue) : "—", color: profitColor },
              { label:"Return", value: perf.returnPct != null ? `${perf.returnPct>=0?"+":""}${Number(perf.returnPct).toFixed(1)}%` : "—", color: profitColor },
            ].map(s => (
              <div key={s.label} style={{ background:"#1C1C1C", padding:"10px 10px" }}>
                <div style={{ fontSize:10, color:C.text.muted, marginBottom:3, textTransform:"uppercase", letterSpacing:.6 }}>{s.label}</div>
                <div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:13, fontWeight:500, color:s.color }}>{s.value}</div>
              </div>
            ))}
          </div>
        ) : (
          <div style={{ padding:"10px 14px", background:"#141414", borderRadius:10, marginBottom:12, fontSize:12, color:C.text.muted }}>
            Log your first investment in Ledger to see P&L here
          </div>
        )}
      </div>

      {/* ── SIGNAL JOURNEY ── */}
      {signal && <SignalJourney rsi={rsi} vix={vix} drawdown={drawdown} />}

      {/* ── PRICE & DRAWDOWN SUMMARY ── */}
      {signal && (
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8, marginBottom:12 }}>
          <div style={{ background:"#141414", border:"1px solid #252525", borderRadius:8, padding:"10px 12px" }}>
            <div style={{ fontSize:10, color:C.text.muted, marginBottom:4, textTransform:"uppercase", letterSpacing:.8, fontWeight:700 }}>CURRENT PRICE</div>
            <div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:20, color:C.text.primary, fontWeight:500 }}>${price?.toFixed(2)}</div>
          </div>
          <div style={{ background:"#141414", border:`1px solid ${drawdown < -10 ? C.red+"30" : "#252525"}`, borderRadius:8, padding:"10px 12px" }}>
            <div style={{ fontSize:10, color:C.text.muted, marginBottom:4, textTransform:"uppercase", letterSpacing:.8 }}>DRAWDOWN (52W HIGH)</div>
            <div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:20, color: drawdown < -10 ? C.red : drawdown < -5 ? C.signal.MEDIUM : C.text.secondary, fontWeight:500 }}>{drawdown?.toFixed(1)}%</div>
          </div>
        </div>
      )}


      {/* ── TABBED MARKET INSIGHTS ── */}
      {tier !== "NONE" && signal && (
        <div className="card">
          <div style={{ display:"flex", gap:6, marginBottom:12 }}>
            <button className={`pill-btn ${marketTab==="snapshot"?"active":""}`} onClick={() => setMarketTab("snapshot")}>Market Snapshot</button>
            <button className={`pill-btn ${marketTab==="analysis"?"active":""}`} onClick={() => setMarketTab("analysis")}>Analysis</button>
          </div>

          {marketTab === "snapshot" ? (
            <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
              {/* DrawdownBands content */}
              <DrawdownBands price={price} high52w={high52w} noCard />
            </div>
          ) : (
            <div>
              <SectionLabel>Analyst Commentary</SectionLabel>
              <div style={{ fontSize:14, color:C.text.secondary, lineHeight:1.6 }}>{signal.analyst_summary || "No commentary available."}</div>
            </div>
          )}
        </div>
      )}

      {/* ── CHAT ── */}
      <div className="card">
        <SectionLabel>Ask the Analyst</SectionLabel>
        {messages.length > 0 && (
          <div ref={chatRef} style={{ maxHeight:260, overflowY:"auto", marginBottom:10, display:"flex", flexDirection:"column", gap:8 }}>
            {messages.map((m, i) => (
              <div key={i} className={m.role==="user" ? "chat-bubble-user" : "chat-bubble-ai"}>
                <div style={{ fontSize:13, color:C.text.secondary, lineHeight:1.55 }}>{m.content}</div>
              </div>
            ))}
            {chatLoading && <div className="chat-bubble-ai"><div style={{ display:"flex", gap:6, alignItems:"center" }}><Spinner /><span style={{ fontSize:13, color:C.text.muted }}>Thinking…</span></div></div>}
          </div>
        )}
        <div style={{ display:"flex", gap:8 }}>
          <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key==="Enter" && sendChat()} placeholder="Ask about today's signal…" disabled={chatLoading || !signal} />
          <button className="btn btn-primary" onClick={sendChat} disabled={chatLoading || !input.trim()} style={{ flexShrink:0, padding:"10px 14px" }}>→</button>
        </div>
      </div>

    </div>
  );
}

// ─── CHARTS TAB ───────────────────────────────────────────────────────────────
function ChartsTab() {
  const [range, setRange] = useState("3mo");
  const [chartData, setChartData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  async function loadChart(r) {
    setLoading(true); setError(null);
    try {
      const data = await api(`/signal/chart?range=${r}`);
      setChartData((data.chartData || []).map(d => ({ ...d, signal_tier: normaliseTier(d.signal_tier) })));
    } catch { setError("Unable to load chart data."); }
    setLoading(false);
  }

  useEffect(() => { loadChart(range); }, [range]);

  const xTicks = buildXTicks(chartData, 6);
  const xTickFormatter = (val) => { try { return new Date(val).toLocaleDateString("en-AU", { day:"2-digit", month:"short" }); } catch { return val; } };
  const tooltipStyle = { background:"#1C1C1C", border:"1px solid #252525", borderRadius:8, fontSize:12 };

  const CustomPriceDot = (props) => {
    const { cx, cy, payload } = props;
    if (!payload.signal_tier || payload.signal_tier === "NONE") return null;
    const color = C.signal[payload.signal_tier] || "transparent";
    return <circle cx={cx} cy={cy} r={6} fill={color} stroke="#0A0A0A" strokeWidth={1.5} />;
  };

  return (
    <div className="tab-content fade-in">
      <div style={{ display:"flex", gap:6, marginBottom:16, flexWrap:"wrap" }}>
        {[{l:"1M",v:"1mo"},{l:"3M",v:"3mo"},{l:"6M",v:"6mo"},{l:"1Y",v:"1y"}].map(o => (
          <button key={o.v} className={`pill-btn ${range===o.v?"active":""}`} onClick={() => setRange(o.v)}>{o.l}</button>
        ))}
      </div>
      {loading ? (<><SkeletonCard height={220} /><SkeletonCard height={180} /><SkeletonCard height={180} /><SkeletonCard height={180} /></>) :
       error ? <ErrorState message={error} onRetry={() => loadChart(range)} /> : (
        <>
          <div className="card">
            <SectionLabel>VDGR Price + Signal Markers</SectionLabel>
            <ResponsiveContainer width="100%" height={220}>
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#252525" />
                <XAxis dataKey="date" tick={{ fill:C.text.muted, fontSize:10 }} ticks={xTicks} tickFormatter={xTickFormatter} interval={0} />
                <YAxis tick={{ fill:C.text.muted, fontSize:11 }} domain={["auto","auto"]} width={52} tickFormatter={v => `$${Number(v).toFixed(0)}`} />
                <Tooltip contentStyle={tooltipStyle} labelStyle={{ color:C.text.secondary }} formatter={(v, n) => n==="price" ? [`$${Number(v).toFixed(2)}`, "Price"] : [v, n]} labelFormatter={xTickFormatter} />
                <Line type="monotone" dataKey="price" stroke={C.accent} dot={<CustomPriceDot />} strokeWidth={2} connectNulls name="price" />
              </ComposedChart>
            </ResponsiveContainer>
            <div style={{ display:"flex", gap:12, marginTop:10, flexWrap:"wrap" }}>
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <div style={{ width:16, height:2, background:C.accent, borderRadius:1 }} />
                <span style={{ fontSize:11, color:C.text.muted }}>Price</span>
              </div>
              {["LOW","MEDIUM","HIGH","EXTREME"].map(s => (
                <div key={s} style={{ display:"flex", alignItems:"center", gap:5 }}>
                  <div style={{ width:10, height:10, borderRadius:"50%", background:C.signal[s] }} />
                  <span style={{ fontSize:11, color:C.text.muted }}>{s}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="card">
            <SectionLabel>RSI (14-day)</SectionLabel>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={chartData} margin={{ top:4, right:8, bottom:0, left:0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E1E1E" />
                <XAxis dataKey="date" tick={{ fill:C.text.muted, fontSize:10 }} ticks={xTicks} tickFormatter={xTickFormatter} interval={0} />
                <YAxis tick={{ fill:C.text.muted, fontSize:11 }} domain={[20, 80]} width={28} />
                <Tooltip contentStyle={tooltipStyle} labelStyle={{ color:C.text.secondary }} labelFormatter={xTickFormatter} formatter={v => [Number(v).toFixed(1), "RSI"]} />
                <ReferenceLine y={50} stroke={C.signal.LOW}     strokeDasharray="4 4" label={{ value:"50", position:"insideTopRight", fill:C.signal.LOW,     fontSize:10 }} />
                <ReferenceLine y={45} stroke={C.signal.MEDIUM}  strokeDasharray="4 4" label={{ value:"45", position:"insideTopRight", fill:C.signal.MEDIUM,  fontSize:10 }} />
                <ReferenceLine y={35} stroke={C.signal.HIGH}    strokeDasharray="4 4" label={{ value:"35", position:"insideTopRight", fill:C.signal.HIGH,    fontSize:10 }} />
                <ReferenceLine y={30} stroke={C.signal.EXTREME} strokeDasharray="4 4" label={{ value:"30", position:"insideTopRight", fill:C.signal.EXTREME, fontSize:10 }} />
                <Line type="monotone" dataKey="rsi" stroke={C.green} dot={false} strokeWidth={2.5} connectNulls />
              </LineChart>
            </ResponsiveContainer>
            <div style={{ display:"flex", gap:12, marginTop:10, flexWrap:"wrap" }}>
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <div style={{ width:16, height:2, background:C.green, borderRadius:1 }} />
                <span style={{ fontSize:11, color:C.text.muted }}>RSI</span>
              </div>
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <div style={{ width:10, height:10, borderRadius:"50%", background:C.signal.LOW }} />
                <span style={{ fontSize:11, color:C.text.muted }}>50 Watch</span>
              </div>
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <div style={{ width:10, height:10, borderRadius:"50%", background:C.signal.MEDIUM }} />
                <span style={{ fontSize:11, color:C.text.muted }}>45 Medium</span>
              </div>
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <div style={{ width:10, height:10, borderRadius:"50%", background:C.signal.HIGH }} />
                <span style={{ fontSize:11, color:C.text.muted }}>35 High</span>
              </div>
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <div style={{ width:10, height:10, borderRadius:"50%", background:C.signal.EXTREME }} />
                <span style={{ fontSize:11, color:C.text.muted }}>30 Extreme</span>
              </div>
            </div>
          </div>
          <div className="card">
            <SectionLabel>VIX (Fear Index)</SectionLabel>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={chartData} margin={{ top:4, right:8, bottom:0, left:0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E1E1E" />
                <XAxis dataKey="date" tick={{ fill:C.text.muted, fontSize:10 }} ticks={xTicks} tickFormatter={xTickFormatter} interval={0} />
                <YAxis tick={{ fill:C.text.muted, fontSize:11 }} domain={["auto","auto"]} width={28} />
                <Tooltip contentStyle={tooltipStyle} labelStyle={{ color:C.text.secondary }} labelFormatter={xTickFormatter} formatter={v => [Number(v).toFixed(1), "VIX"]} />
                <ReferenceLine y={18} stroke={C.signal.LOW}     strokeDasharray="4 4" label={{ value:"18", position:"insideTopRight", fill:C.signal.LOW,     fontSize:10 }} />
                <ReferenceLine y={20} stroke={C.signal.MEDIUM}  strokeDasharray="4 4" label={{ value:"20", position:"insideTopRight", fill:C.signal.MEDIUM,  fontSize:10 }} />
                <ReferenceLine y={25} stroke={C.signal.HIGH}    strokeDasharray="4 4" label={{ value:"25", position:"insideTopRight", fill:C.signal.HIGH,    fontSize:10 }} />
                <ReferenceLine y={30} stroke={C.signal.EXTREME} strokeDasharray="4 4" label={{ value:"30", position:"insideTopRight", fill:C.signal.EXTREME, fontSize:10 }} />
                <Line type="monotone" dataKey="vix" stroke="#A78BFA" dot={false} strokeWidth={2.5} connectNulls />
              </LineChart>
            </ResponsiveContainer>
            <div style={{ display:"flex", gap:12, marginTop:10, flexWrap:"wrap" }}>
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <div style={{ width:16, height:2, background:"#A78BFA", borderRadius:1 }} />
                <span style={{ fontSize:11, color:C.text.muted }}>VIX</span>
              </div>
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <div style={{ width:10, height:10, borderRadius:"50%", background:C.signal.LOW }} />
                <span style={{ fontSize:11, color:C.text.muted }}>18 Watch</span>
              </div>
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <div style={{ width:10, height:10, borderRadius:"50%", background:C.signal.MEDIUM }} />
                <span style={{ fontSize:11, color:C.text.muted }}>20 Medium</span>
              </div>
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <div style={{ width:10, height:10, borderRadius:"50%", background:C.signal.HIGH }} />
                <span style={{ fontSize:11, color:C.text.muted }}>25 High</span>
              </div>
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <div style={{ width:10, height:10, borderRadius:"50%", background:C.signal.EXTREME }} />
                <span style={{ fontSize:11, color:C.text.muted }}>30 Extreme</span>
              </div>
            </div>
          </div>
          <div className="card">
            <SectionLabel>Drawdown from 52-Week High</SectionLabel>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={chartData} margin={{ top:4, right:8, bottom:0, left:0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E1E1E" />
                <XAxis dataKey="date" tick={{ fill:C.text.muted, fontSize:10 }} ticks={xTicks} tickFormatter={xTickFormatter} interval={0} />
                <YAxis tick={{ fill:C.text.muted, fontSize:11 }} domain={["auto", 2]} width={36} tickFormatter={v => `${v}%`} />
                <Tooltip contentStyle={tooltipStyle} labelStyle={{ color:C.text.secondary }} labelFormatter={xTickFormatter} formatter={v => [`${Number(v).toFixed(1)}%`, "Drawdown"]} />
                <ReferenceLine y={-5}  stroke={C.signal.LOW}     strokeDasharray="4 4" label={{ value:"-5%",  position:"insideTopRight", fill:C.signal.LOW,     fontSize:10 }} />
                <ReferenceLine y={-10} stroke={C.signal.MEDIUM}  strokeDasharray="4 4" label={{ value:"-10%", position:"insideTopRight", fill:C.signal.MEDIUM,  fontSize:10 }} />
                <ReferenceLine y={-15} stroke={C.signal.HIGH}    strokeDasharray="4 4" label={{ value:"-15%", position:"insideTopRight", fill:C.signal.HIGH,    fontSize:10 }} />
                <ReferenceLine y={-20} stroke={C.signal.EXTREME} strokeDasharray="4 4" label={{ value:"-20%", position:"insideTopRight", fill:C.signal.EXTREME, fontSize:10 }} />
                <Line type="monotone" dataKey="drawdown" stroke={C.signal.HIGH} dot={false} strokeWidth={2.5} connectNulls />
              </LineChart>
            </ResponsiveContainer>
            <div style={{ display:"flex", gap:12, marginTop:10, flexWrap:"wrap" }}>
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <div style={{ width:16, height:2, background:C.signal.HIGH, borderRadius:1 }} />
                <span style={{ fontSize:11, color:C.text.muted }}>Drawdown</span>
              </div>
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <div style={{ width:10, height:10, borderRadius:"50%", background:C.signal.LOW }} />
                <span style={{ fontSize:11, color:C.text.muted }}>-5%</span>
              </div>
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <div style={{ width:10, height:10, borderRadius:"50%", background:C.signal.MEDIUM }} />
                <span style={{ fontSize:11, color:C.text.muted }}>-10%</span>
              </div>
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <div style={{ width:10, height:10, borderRadius:"50%", background:C.signal.HIGH }} />
                <span style={{ fontSize:11, color:C.text.muted }}>-15%</span>
              </div>
              <div style={{ display:"flex", alignItems:"center", gap:5 }}>
                <div style={{ width:10, height:10, borderRadius:"50%", background:C.signal.EXTREME }} />
                <span style={{ fontSize:11, color:C.text.muted }}>-20%</span>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ─── LEDGER TAB ───────────────────────────────────────────────────────────────
// FIX: Screenshot upload redesigned to avoid white flash on iOS.
// The native <input type="file"> triggers a white system sheet on iOS — this is unavoidable.
// Improvements:
// 1. The overlay and sheet are explicitly dark (#0A0A0A background) so the flash is brief
// 2. A visible "Processing..." state replaces the blank period after selection
// 3. The upload zone is a styled dark box that clearly communicates state
// 4. Added a small delay after file selection so the dark sheet re-renders before processing
function LedgerTab() {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showForm, setShowForm] = useState(false);
  const [saving, setSaving] = useState(false);
  const [todaySignal, setTodaySignal] = useState(null);
  const [form, setForm] = useState({ date:"", signal_tier:"MEDIUM", vdgr_price:"", actual_amount:200, notes:"" });

  const [scanMode, setScanMode] = useState(false);
  const [scanState, setScanState] = useState("idle"); // idle | selected | scanning | done | error
  const [scanError, setScanError] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [previewFile, setPreviewFile] = useState(null);
  const fileInputRef = useRef(null);

  async function loadData() {
    setLoading(true);
    try {
      const [ledger, signal] = await Promise.all([api("/ledger"), api("/signal/today")]);
      setEntries(ledger.entries || []);
      setTodaySignal(signal.signal);
    } catch { setError("Unable to load ledger."); }
    setLoading(false);
  }

  useEffect(() => { loadData(); }, []);

  function openForm(mode = "manual") {
    const today = new Date().toISOString().split("T")[0];
    setForm({
      date: today,
      signal_tier: normaliseTier(todaySignal?.signal_tier) || "MEDIUM",
      vdgr_price: todaySignal?.vdgr_price || "",
      actual_amount: todaySignal?.recommended_amount || 200,
      notes: "",
    });
    setScanMode(mode === "screenshot");
    setScanState("idle");
    setScanError(null);
    setPreviewUrl(null);
    setPreviewFile(null);
    setShowForm(true);
  }

  function fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result.split(",")[1]);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  // Called when user picks a file — just show preview, don't scan yet
  function handleFileSelected(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    // Reset input so same file can be re-selected
    e.target.value = "";
    setPreviewUrl(URL.createObjectURL(file));
    setPreviewFile(file);
    setScanState("selected");
    setScanError(null);
  }

  // Called when user taps "Scan this image"
  async function runScan() {
    if (!previewFile) return;
    setScanState("scanning");
    setScanError(null);
    try {
      const base64 = await fileToBase64(previewFile);
      const mediaType = previewFile.type || "image/jpeg";
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 400,
          system: `You are extracting investment transaction details from a Vanguard or broker screenshot.
Extract ONLY these fields and respond with ONLY valid JSON, no other text:
{
  "date": "YYYY-MM-DD",
  "vdgr_price": <number - unit price paid per ETF unit>,
  "actual_amount": <number - total dollar amount invested>,
  "units": <number - number of units purchased>,
  "notes": "<string - fund name or any useful note, or empty string>"
}
If you cannot find a field with confidence, use null for that field.
The ETF may be labelled as VDGR, VanEck Gold Royalties, or similar.
Date formats in Australian screenshots are typically DD/MM/YYYY — convert to YYYY-MM-DD.`,
          messages: [{
            role: "user",
            content: [
              { type: "image", source: { type: "base64", media_type: mediaType, data: base64 } },
              { type: "text", text: "Extract the transaction details from this investment screenshot." }
            ]
          }]
        })
      });
      const data = await res.json();
      const text = data.content?.[0]?.text || "";
      const clean = text.replace(/```json|```/g, "").trim();
      const parsed = JSON.parse(clean);
      setForm(f => ({
        ...f,
        date:          parsed.date          || f.date,
        vdgr_price:    parsed.vdgr_price    || f.vdgr_price,
        actual_amount: parsed.actual_amount || f.actual_amount,
        notes:         parsed.notes         || f.notes,
      }));
      setScanState("done");
      setScanMode(false); // Switch to confirm form
    } catch {
      setScanState("error");
      setScanError("Couldn't read the screenshot. Try a clearer image or enter details manually.");
    }
  }

  async function saveEntry() {
    setSaving(true);
    try {
      await api("/ledger", {
        method: "POST",
        body: JSON.stringify({
          date: form.date,
          signal_tier: form.signal_tier === "LOW" ? "WATCH" : form.signal_tier,
          recommended_amount: todaySignal?.recommended_amount || null,
          actual_amount: parseFloat(form.actual_amount),
          vdgr_price: parseFloat(form.vdgr_price),
          notes: form.notes || null,
        }),
      });
      setShowForm(false);
      setPreviewUrl(null);
      loadData();
    } catch { alert("Failed to save entry. Please try again."); }
    setSaving(false);
  }

  function closeForm() {
    setShowForm(false);
    setPreviewUrl(null);
    setPreviewFile(null);
    setScanError(null);
    setScanState("idle");
  }

  const total = entries.reduce((s, e) => s + parseFloat(e.actual_amount), 0);
  const totalUnits = entries.reduce((s, e) => s + parseFloat(e.units_acquired), 0);
  const avgBuyPrice = totalUnits > 0 ? total / totalUnits : 0;
  const currentPrice = todaySignal ? parseFloat(todaySignal.vdgr_price) : null;
  const currentValue = currentPrice ? totalUnits * currentPrice : null;
  const profitLoss = currentValue !== null ? currentValue - total : null;

  return (
    <div className="tab-content fade-in">
      {/* Summary */}
      <div className="card">
        <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:12 }}>
          <SectionLabel>Investment Ledger</SectionLabel>
          <div style={{ display:"flex", gap:8 }}>
            <button className="btn btn-ghost" style={{ padding:"6px 12px", fontSize:13 }} onClick={() => openForm("screenshot")}>📷 Scan</button>
            <button className="btn btn-primary" style={{ padding:"6px 14px", fontSize:13 }} onClick={() => openForm("manual")}>+ Log</button>
          </div>
        </div>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8, marginBottom:8 }}>
          {[
            { label:"Total invested", value: fmt.aud(total) },
            { label:"Total units",    value: totalUnits.toFixed(4) },
            { label:"Avg buy price",  value: avgBuyPrice > 0 ? `$${avgBuyPrice.toFixed(2)}` : "—" },
            { label:"Entries",        value: entries.length },
          ].map(s => (
            <div key={s.label} style={{ background:"#141414", borderRadius:8, padding:"8px 12px" }}>
              <div style={{ fontSize:11, color:C.text.muted, marginBottom:2 }}>{s.label}</div>
              <div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:15, color:C.text.primary, fontWeight:500 }}>{s.value}</div>
            </div>
          ))}
        </div>
        {currentValue !== null && (
          <div style={{ padding:"8px 12px", background: profitLoss >= 0 ? `${C.green}10` : `${C.red}10`, borderRadius:8, border:`1px solid ${profitLoss >= 0 ? C.green+"30" : C.red+"30"}`, display:"flex", justifyContent:"space-between" }}>
            <span style={{ fontSize:12, color:C.text.muted }}>Current value @ ${currentPrice?.toFixed(2)}</span>
            <span style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:13, color: profitLoss >= 0 ? C.green : C.red }}>
              {fmt.aud(currentValue)} ({profitLoss >= 0 ? "+" : ""}{fmt.aud(profitLoss)})
            </span>
          </div>
        )}
      </div>

      {/* Entries */}
      {loading ? [1,2,3].map(i => <SkeletonCard key={i} height={64} />) :
       error ? <ErrorState message={error} onRetry={loadData} /> :
       entries.length === 0 ? (
         <div style={{ textAlign:"center", padding:"40px 20px", color:C.text.muted, fontSize:14 }}>
           No investments logged yet.<br />Tap 📷 Scan or + Log to add your first entry.
         </div>
       ) :
       entries.map(e => (
        <div key={e.id} className="card">
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start" }}>
            <div>
              <div style={{ fontSize:14, color:C.text.primary, marginBottom:4 }}>{fmt.date(e.date)}</div>
              <div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:11, color:C.text.muted }}>
                ${Number(e.vdgr_price).toFixed(2)} · {Number(e.units_acquired).toFixed(4)} units
              </div>
              {e.notes && <div style={{ fontSize:12, color:C.text.muted, marginTop:4 }}>{e.notes}</div>}
            </div>
            <div style={{ textAlign:"right" }}>
              <div style={{ color:C.signal[normaliseTier(e.signal_tier)], fontWeight:600, fontSize:12, marginBottom:4 }}>{normaliseTier(e.signal_tier)}</div>
              <div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:16, color:C.text.primary }}>{fmt.aud(e.actual_amount)}</div>
            </div>
          </div>
        </div>
      ))}

      {/* Form sheet — always dark background to prevent white flash */}
      {showForm && (
        <div className="overlay" onClick={closeForm} style={{ background:"rgba(0,0,0,.9)" }}>
          <div
            className="sheet slide-up"
            onClick={e => e.stopPropagation()}
            style={{
              maxHeight:"90vh",
              overflowY:"auto",
              background:"#0F0F0F",  // Darker than default card to contrast with overlay
              borderTop:"1px solid #2A2A2A",
              borderRadius:"20px 20px 0 0",
              paddingBottom:"env(safe-area-inset-bottom, 20px)",
            }}
          >
            {/* Hidden file input — always present so iOS doesn't re-init */}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              style={{ display:"none" }}
              onChange={handleFileSelected}
            />

            {/* ── SCAN MODE: choose image ── */}
            {scanMode && (
              <>
                <div style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:22, fontWeight:700, marginBottom:4, color:C.text.primary }}>Scan Screenshot</div>
                <div style={{ fontSize:13, color:C.text.muted, marginBottom:20, lineHeight:1.5 }}>
                  Upload a screenshot from your Vanguard app. Claude will read the transaction details.
                </div>

                {/* State: idle — show upload button */}
                {scanState === "idle" && (
                  <button
                    className="btn btn-ghost"
                    style={{ width:"100%", padding:"20px", fontSize:15, borderRadius:12, border:"2px dashed #2A2A2A", flexDirection:"column", gap:8, height:"auto" }}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <span style={{ fontSize:32 }}>📷</span>
                    <span style={{ color:C.text.secondary }}>Choose screenshot from library</span>
                    <span style={{ fontSize:11, color:C.text.muted }}>JPG or PNG from your camera roll</span>
                  </button>
                )}

                {/* State: selected — show preview + scan button */}
                {scanState === "selected" && previewUrl && (
                  <>
                    <div style={{ background:"#141414", borderRadius:12, overflow:"hidden", marginBottom:12 }}>
                      <img src={previewUrl} alt="preview" style={{ width:"100%", maxHeight:220, objectFit:"contain", display:"block" }} />
                    </div>
                    <div style={{ display:"flex", gap:8, marginBottom:8 }}>
                      <button className="btn btn-ghost" style={{ flex:1 }} onClick={() => { setScanState("idle"); setPreviewUrl(null); setPreviewFile(null); }}>
                        ← Different image
                      </button>
                      <button className="btn btn-primary" style={{ flex:1 }} onClick={runScan}>
                        Scan this image →
                      </button>
                    </div>
                  </>
                )}

                {/* State: scanning — spinner */}
                {scanState === "scanning" && (
                  <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:16, padding:"40px 20px" }}>
                    {previewUrl && (
                      <div style={{ background:"#141414", borderRadius:12, overflow:"hidden", width:"100%", opacity:0.5 }}>
                        <img src={previewUrl} alt="preview" style={{ width:"100%", maxHeight:160, objectFit:"contain", display:"block" }} />
                      </div>
                    )}
                    <div style={{ display:"flex", alignItems:"center", gap:12 }}>
                      <Spinner size={20} />
                      <span style={{ fontSize:14, color:C.text.secondary }}>Reading your screenshot…</span>
                    </div>
                  </div>
                )}

                {/* State: error */}
                {scanState === "error" && (
                  <div style={{ padding:"14px", background:`${C.red}10`, border:`1px solid ${C.red}30`, borderRadius:10, marginBottom:12, fontSize:13, color:C.red, lineHeight:1.5 }}>
                    {scanError}
                  </div>
                )}

                <div style={{ display:"flex", gap:8, marginTop:8 }}>
                  <button className="btn btn-ghost" style={{ flex:1 }} onClick={closeForm}>Cancel</button>
                  {(scanState === "idle" || scanState === "error") && (
                    <button className="btn btn-ghost" style={{ flex:1 }} onClick={() => { setScanMode(false); setScanState("idle"); }}>
                      Enter manually
                    </button>
                  )}
                </div>
              </>
            )}

            {/* ── MANUAL / CONFIRM MODE ── */}
            {!scanMode && (
              <>
                <div style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:22, fontWeight:700, marginBottom:4, color:C.text.primary }}>
                  {scanState === "done" ? "Confirm Details" : "Log Investment"}
                </div>
                <div style={{ fontSize:12, color:C.text.muted, marginBottom:16 }}>
                  {scanState === "done" ? "Review the extracted details and adjust if needed." : "Enter any past date to backfill historical trades."}
                </div>

                {/* Show thumbnail if scanned */}
                {previewUrl && scanState === "done" && (
                  <div style={{ marginBottom:12, background:"#141414", borderRadius:10, overflow:"hidden" }}>
                    <img src={previewUrl} alt="preview" style={{ width:"100%", maxHeight:100, objectFit:"contain", display:"block" }} />
                  </div>
                )}

                <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
                  {/* Signal tier */}
                  <div>
                    <label style={{ fontSize:12, color:C.text.muted, display:"block", marginBottom:4 }}>Signal tier</label>
                    <div style={{ display:"flex", gap:6 }}>
                      {["LOW","MEDIUM","HIGH","EXTREME"].map(s => (
                        <button key={s}
                          onClick={() => setForm(f => ({ ...f, signal_tier:s, actual_amount: f.actual_amount || { LOW:0, MEDIUM:200, HIGH:400, EXTREME:800 }[s] }))}
                          style={{ flex:1, padding:"8px 2px", borderRadius:8, border:`2px solid ${form.signal_tier===s ? C.signal[s] : "#252525"}`, background: form.signal_tier===s ? `${C.signal[s]}18` : "transparent", color: form.signal_tier===s ? C.signal[s] : C.text.muted, cursor:"pointer", fontFamily:"'Barlow Condensed',sans-serif", fontSize:13, fontWeight:700 }}>
                          {s}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Date */}
                  <div>
                    <label style={{ fontSize:12, color:C.text.muted, display:"block", marginBottom:4 }}>Date</label>
                    <input type="date" value={form.date} max={new Date().toISOString().split("T")[0]} onChange={e => setForm(f => ({ ...f, date:e.target.value }))} />
                  </div>

                  {/* Price + Amount */}
                  <div style={{ display:"flex", gap:10 }}>
                    <div style={{ flex:1 }}>
                      <label style={{ fontSize:12, color:C.text.muted, display:"block", marginBottom:4 }}>VDGR Price ($)</label>
                      <input type="number" value={form.vdgr_price} onChange={e => setForm(f => ({ ...f, vdgr_price:e.target.value }))} step="0.01" placeholder="e.g. 66.76" />
                    </div>
                    <div style={{ flex:1 }}>
                      <label style={{ fontSize:12, color:C.text.muted, display:"block", marginBottom:4 }}>Amount ($)</label>
                      <input type="number" value={form.actual_amount} onChange={e => setForm(f => ({ ...f, actual_amount:e.target.value }))} />
                    </div>
                  </div>

                  {/* Units preview */}
                  <div style={{ padding:"8px 12px", background:"#141414", borderRadius:8, fontFamily:"'JetBrains Mono',monospace", fontSize:13, color:C.text.muted }}>
                    Units: {form.vdgr_price > 0 ? (form.actual_amount / form.vdgr_price).toFixed(4) : "—"}
                  </div>

                  {/* Notes */}
                  <div>
                    <label style={{ fontSize:12, color:C.text.muted, display:"block", marginBottom:4 }}>Notes (optional)</label>
                    <input type="text" value={form.notes} onChange={e => setForm(f => ({ ...f, notes:e.target.value }))} placeholder="e.g. First purchase, market dip" />
                  </div>

                  {/* Actions */}
                  <div style={{ display:"flex", gap:8, marginTop:4 }}>
                    <button className="btn btn-ghost" style={{ flex:1 }} onClick={closeForm}>Cancel</button>
                    <button className="btn btn-primary" style={{ flex:1 }} onClick={saveEntry} disabled={saving || !form.vdgr_price || !form.actual_amount}>
                      {saving ? <Spinner size={16} /> : "Save entry"}
                    </button>
                  </div>

                  {/* Switch to scan */}
                  {scanState !== "done" && (
                    <button className="btn btn-ghost" style={{ width:"100%", fontSize:13 }} onClick={() => { setScanMode(true); setScanState("idle"); setScanError(null); }}>
                      📷 Upload screenshot instead
                    </button>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── PERFORMANCE TAB ──────────────────────────────────────────────────────────
function PerformanceTab() {
  const [perf, setPerf] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  async function loadPerf() {
    setLoading(true); setError(null);
    try {
      const perfData = await api("/performance");
      setPerf(perfData);
    }
    catch { setError("Unable to load performance data."); }
    setLoading(false);
  }

  useEffect(() => { loadPerf(); }, []);

  if (loading) return <div className="tab-content"><SkeletonCard height={140} /><SkeletonCard height={180} /><SkeletonCard height={200} /></div>;
  if (error) return <div className="tab-content"><ErrorState message={error} onRetry={loadPerf} /></div>;
  if (!perf?.summary) return (
    <div className="tab-content fade-in">
      <div style={{ textAlign:"center", padding:"60px 20px", color:C.text.muted, fontSize:14, lineHeight:1.6 }}>
        Performance tracking begins after your first ledger entry.<br />Log an investment in the Ledger tab to get started.
      </div>
    </div>
  );

  const { summary, snapshots, forwardReturns, ledger } = perf;
  const profitColor = summary.returnDollar >= 0 ? C.green : C.red;

  // Sort ledger by date ascending for per-trade display
  const sortedLedger = ledger ? [...ledger].sort((a, b) => new Date(a.date) - new Date(b.date)) : [];

  const tierBreakdown = {};
  (ledger || []).forEach(e => {
    const tier = normaliseTier(e.signal_tier);
    if (!tierBreakdown[tier]) tierBreakdown[tier] = { count:0, invested:0, units:0 };
    tierBreakdown[tier].count++;
    tierBreakdown[tier].invested += parseFloat(e.actual_amount);
    tierBreakdown[tier].units += parseFloat(e.units_acquired);
  });

  const currentPrice = summary.currentPrice;
  const tierOrder = ["EXTREME","HIGH","MEDIUM","LOW"];
  const tierRows = tierOrder.filter(t => tierBreakdown[t]).map(t => {
    const b = tierBreakdown[t];
    const cv = currentPrice ? b.units * currentPrice : null;
    const ret = cv !== null ? ((cv - b.invested) / b.invested) * 100 : null;
    const avgPrice = b.units > 0 ? b.invested / b.units : 0;
    return { tier:t, ...b, currentValue:cv, returnPct:ret, avgPrice };
  });

  return (
    <div className="tab-content fade-in">
      {/* Hero P&L Display */}
      <div className="card" style={{ textAlign:"center", padding:"32px 20px 24px" }}>
        <div style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:52, fontWeight:800, color:profitColor, lineHeight:1 }}>
          {summary.returnDollar !== null ? `${summary.returnDollar >= 0 ? "+" : ""}${fmt.aud(summary.returnDollar)}` : "—"}
        </div>
        <div style={{ fontSize:18, color:profitColor, marginTop:8 }}>
          {summary.returnPct !== null ? `${summary.returnPct >= 0 ? "+" : ""}${Number(summary.returnPct).toFixed(2)}%` : "—"}
        </div>
        <div style={{ fontSize:13, color:C.text.muted, marginTop:6 }}>
          on {fmt.aud(summary.totalInvested)} invested
        </div>
      </div>

      {/* By Trade - Horizontal bars */}
      <div className="card">
        <SectionLabel>BY TRADE</SectionLabel>
        <div style={{ display:"flex", flexDirection:"column", gap:16, marginTop:12 }}>
          {sortedLedger.map((trade, i) => {
            const tradeValue = parseFloat(trade.units_acquired) * (summary.currentPrice || 0);
            const tradeReturn = parseFloat(trade.actual_amount) > 0
              ? ((tradeValue - parseFloat(trade.actual_amount)) / parseFloat(trade.actual_amount)) * 100
              : 0;
            const tradeColor = tradeReturn >= 0 ? C.green : C.red;
            const barWidth = Math.min(Math.max(Math.abs(tradeReturn) / 10 * 100, 2), 100);

            return (
              <div key={i}>
                <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:6 }}>
                  <div style={{ fontSize:12, color:C.text.muted, fontFamily:"'JetBrains Mono',monospace", minWidth:60 }}>
                    {fmt.short(trade.date)}
                  </div>
                  <div style={{ flex:1, background:"#252525", height:8, borderRadius:999, overflow:"hidden" }}>
                    <div style={{ width:`${barWidth}%`, height:"100%", background:tradeColor, borderRadius:999, transition:"width 0.4s ease" }} />
                  </div>
                  <div style={{ fontSize:13, fontFamily:"'JetBrains Mono',monospace", color:tradeColor, minWidth:60, textAlign:"right" }}>
                    {tradeReturn >= 0 ? "+" : ""}{tradeReturn.toFixed(1)}%
                  </div>
                </div>
                <div style={{ display:"flex", alignItems:"center", gap:8, paddingLeft:72 }}>
                  <SignalBadge signal={normaliseTier(trade.signal_tier)} size="sm" />
                  <span style={{ fontSize:11, color:C.text.muted }}>·</span>
                  <span style={{ fontSize:11, color:C.text.muted }}>{fmt.aud(parseFloat(trade.actual_amount))}</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Summary footer */}
        <div style={{ marginTop:16, paddingTop:16, borderTop:"1px solid #252525", display:"flex", justifyContent:"space-between" }}>
          <span style={{ fontSize:12, color:C.text.muted, fontFamily:"'JetBrains Mono',monospace" }}>
            Total invested: {fmt.aud(summary.totalInvested)}
          </span>
          <span style={{ fontSize:12, color:C.text.muted, fontFamily:"'JetBrains Mono',monospace" }}>
            Current value: {summary.currentValue !== null ? fmt.aud(summary.currentValue) : "—"}
          </span>
        </div>
      </div>
      {tierRows.length > 0 && (
        <div className="card">
          <SectionLabel>Breakdown by Signal Tier</SectionLabel>
          {tierRows.map(r => (
            <div key={r.tier} style={{ marginBottom:12, paddingBottom:12, borderBottom:"1px solid #252525" }}>
              <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:8 }}>
                <SignalBadge signal={r.tier} size="sm" />
                <span style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:12, color:C.text.muted }}>{r.count} trade{r.count!==1?"s":""}</span>
              </div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:6 }}>
                {[
                  ["Invested", fmt.aud(r.invested)],
                  ["Avg price", `$${r.avgPrice.toFixed(2)}`],
                  ["Return", r.returnPct !== null ? `${r.returnPct>=0?"+":""}${r.returnPct.toFixed(1)}%` : "—"],
                ].map(([l, v], vi) => (
                  <div key={l} style={{ background:"#141414", borderRadius:6, padding:"6px 8px" }}>
                    <div style={{ fontSize:10, color:C.text.muted }}>{l}</div>
                    <div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:12, color: vi===2 && r.returnPct !== null ? (r.returnPct>=0 ? C.green : C.red) : C.text.primary }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
      {forwardReturns?.length > 0 && (
        <div className="card">
          <SectionLabel>Historical Return by Tier</SectionLabel>
          {forwardReturns.map(r => (
            <div key={r.signal_tier} style={{ marginBottom:12, paddingBottom:12, borderBottom:"1px solid #252525" }}>
              <div style={{ display:"flex", justifyContent:"space-between", marginBottom:8 }}>
                <SignalBadge signal={normaliseTier(r.signal_tier)} size="sm" />
                <span style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:12, color:C.text.muted }}>{r.count} signals</span>
              </div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:6 }}>
                {[
                  ["Avg return", `${r.avg_return_pct>=0?"+":""}${Number(r.avg_return_pct).toFixed(2)}%`],
                  ["Win rate", `${Number(r.win_rate_pct).toFixed(0)}%`],
                ].map(([l, v]) => (
                  <div key={l} style={{ background:"#141414", borderRadius:6, padding:"6px 8px" }}>
                    <div style={{ fontSize:10, color:C.text.muted }}>{l}</div>
                    <div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:13, color: r.avg_return_pct>=0 ? C.green : C.red }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── SIGNAL CALENDAR ──────────────────────────────────────────────────────────
// FIX: Weeks are now rendered in reverse order (latest week first, oldest last)
// within each month card, so the most recent dates appear at the top.
function SignalCalendar({ history }) {
  console.log("SignalCalendar received history:", (history || []).length, "days");

  const byMonth = {};
  (history || []).forEach(d => {
    const key = d.date.slice(0, 7);
    if (!byMonth[key]) byMonth[key] = {};
    byMonth[key][d.date.slice(8, 10)] = d.signal_tier;
  });

  console.log("SignalCalendar byMonth keys:", Object.keys(byMonth).length, "months");

  const months = Object.keys(byMonth).sort().reverse().slice(0, 24);
  const DOW = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];
  const sigColors = { LOW:"#EAB308", MEDIUM:"#F97316", HIGH:"#EF4444", EXTREME:"#8B5CF6" };

  // Get today's date for highlighting
  const today = new Date();
  const todayYear = today.getFullYear();
  const todayMonth = today.getMonth() + 1;
  const todayDay = today.getDate();

  if (!months.length) return <div style={{ textAlign:"center", padding:"40px 20px", color:C.text.muted, fontSize:14 }}>No signal history yet.</div>;

  return (
    <div>
      {months.map(monthKey => {
        const [year, month] = monthKey.split("-").map(Number);
        const firstDay = new Date(year, month - 1, 1);
        const daysInMonth = new Date(year, month, 0).getDate();

        // Build offset: Mon=0 ... Sun=6
        let startOffset = firstDay.getDay() - 1;
        if (startOffset < 0) startOffset = 6;

        // Build flat cell array: nulls for padding, then 1..daysInMonth
        const cells = [];
        for (let i = 0; i < startOffset; i++) cells.push(null);
        for (let d = 1; d <= daysInMonth; d++) cells.push(d);

        // Pad to complete last week
        while (cells.length % 7 !== 0) cells.push(null);

        // Split into weeks then REVERSE so latest week is first
        const weeks = [];
        for (let i = 0; i < cells.length; i += 7) weeks.push(cells.slice(i, i + 7));
        const weeksReversed = [...weeks].reverse();

        const monthName = new Date(year, month - 1, 1).toLocaleDateString("en-AU", { month:"long", year:"numeric" });

        return (
          <div key={monthKey} style={{ background:"#1C1C1C", border:"1px solid #252525", borderRadius:12, padding:16, marginBottom:12 }}>
            <div style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:16, fontWeight:700, marginBottom:10 }}>{monthName}</div>

            {/* Day-of-week headers */}
            <div style={{ display:"grid", gridTemplateColumns:"repeat(7,1fr)", gap:2, marginBottom:4 }}>
              {DOW.map(d => <div key={d} style={{ textAlign:"center", fontSize:9, color:C.text.muted, fontWeight:600 }}>{d}</div>)}
            </div>

            {/* Weeks — reversed so most recent is at top */}
            <div style={{ display:"flex", flexDirection:"column", gap:2 }}>
              {weeksReversed.map((week, wi) => (
                <div key={wi} style={{ display:"grid", gridTemplateColumns:"repeat(7,1fr)", gap:2 }}>
                  {week.map((day, di) => {
                    if (!day) return <div key={di} />;
                    const dayStr = String(day).padStart(2, "0");
                    const rawTier = byMonth[monthKey]?.[dayStr];
                    const tier = rawTier === "WATCH" ? "LOW" : rawTier;
                    const color = sigColors[tier] || null;

                    // Check if this is today
                    const isToday = (day === todayDay && month === todayMonth && year === todayYear);

                    return (
                      <div key={di} style={{
                        aspectRatio:"1",
                        borderRadius:4,
                        background: color ? color+"25" : "#141414",
                        border: isToday ? "2px solid #F0F0F0" : "1px solid "+(color ? color+"50" : "#1E1E1E"),
                        display:"flex",
                        alignItems:"center",
                        justifyContent:"center",
                        fontSize:10,
                        color: isToday ? "#F0F0F0" : (color || C.text.muted),
                        fontWeight: isToday ? 700 : (color ? 700 : 400),
                        fontFamily:"'JetBrains Mono',monospace",
                      }}>
                        {day}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>

            {/* Legend for this month */}
            <div style={{ display:"flex", gap:8, marginTop:8, flexWrap:"wrap" }}>
              {["LOW","MEDIUM","HIGH","EXTREME"].filter(t => {
                const vals = Object.values(byMonth[monthKey]);
                return vals.includes(t) || (t === "LOW" && vals.includes("WATCH"));
              }).map(t => (
                <div key={t} style={{ display:"flex", alignItems:"center", gap:4 }}>
                  <div style={{ width:8, height:8, borderRadius:2, background:sigColors[t] }} />
                  <span style={{ fontSize:10, color:C.text.muted }}>{t}</span>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── HISTORY LIST ─────────────────────────────────────────────────────────────
function HistoryList({ history }) {
  const [filter, setFilter] = useState("signals");
  const sigColors = { LOW:"#EAB308", MEDIUM:"#F97316", HIGH:"#EF4444", EXTREME:"#8B5CF6" };
  const filtered = (filter === "signals" ? (history || []).filter(d => d.signal_tier !== "NONE") : (history || [])).sort((a, b) => new Date(b.date) - new Date(a.date));
  return (
    <>
      <div style={{ display:"flex", gap:6, marginBottom:12 }}>
        <button className={"pill-btn "+(filter==="signals"?"active":"")} onClick={() => setFilter("signals")}>Signals only</button>
        <button className={"pill-btn "+(filter==="all"?"active":"")} onClick={() => setFilter("all")}>All days</button>
      </div>
      {filtered.length === 0
        ? <div style={{ textAlign:"center", padding:"40px 20px", color:C.text.muted, fontSize:14 }}>No signal days in this range.</div>
        : filtered.map((d, i) => {
            const isSig = d.signal_tier !== "NONE";
            const color = sigColors[d.signal_tier];
            return (
              <div key={i} className={isSig ? "card" : ""} style={!isSig ? { display:"flex", justifyContent:"space-between", padding:"8px 2px", borderBottom:"1px solid #1E1E1E" } : {}}>
                {isSig ? (
                  <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                    <div>
                      <div style={{ fontSize:14, color:"#F0F0F0" }}>{fmt.date(d.date)}</div>
                      <div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:11, color:C.text.secondary, marginTop:2 }}>RSI {Number(d.rsi).toFixed(1)} · VIX {Number(d.vix).toFixed(1)} · ${Number(d.vdgr_price).toFixed(2)}</div>
                    </div>
                    <div style={{ textAlign:"right" }}>
                      <div style={{ display:"inline-flex", alignItems:"center", justifyContent:"center", background:color+"18", border:"2px solid "+color, borderRadius:999, padding:"4px 12px" }}>
                        <span style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:13, fontWeight:800, color, letterSpacing:2 }}>{d.signal_tier}</span>
                      </div>
                      {d.recommended_amount > 0 && <div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:11, color:C.text.secondary, marginTop:4 }}>${d.recommended_amount}</div>}
                    </div>
                  </div>
                ) : (
                  <><div style={{ fontSize:13, color:C.text.secondary }}>{fmt.date(d.date)}</div><div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:11, color:C.text.secondary }}>— NONE</div></>
                )}
              </div>
            );
          })
      }
    </>
  );
}

// ─── BACKTEST TAB ─────────────────────────────────────────────────────────────
function BacktestTab() {
  const [range, setRange] = useState("all");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const sigColors = { LOW:"#EAB308", MEDIUM:"#F97316", HIGH:"#EF4444", EXTREME:"#8B5CF6" };

  async function load(r) {
    setLoading(true); setError(null);
    try { const res = await api("/backtest?range="+r); setData(res.backtest); }
    catch { setError("Unable to load backtest data."); }
    setLoading(false);
  }

  useEffect(() => { load(range); }, [range]);

  const ranges = [{ l:"1M",v:"1m" },{ l:"3M",v:"3m" },{ l:"6M",v:"6m" },{ l:"1Y",v:"1y" },{ l:"All",v:"all" }];
  const profitColor = data?.return_dollar >= 0 ? "#10B981" : "#EF4444";
  const chartData = (data?.cumulative || []).map(d => ({ date: fmt.short(d.date), invested: d.total_invested, value: d.current_value }));

  return (
    <div className="tab-content fade-in">
      <div style={{ marginBottom:16 }}>
        <div style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:22, fontWeight:700, marginBottom:4 }}>Backtest Simulator</div>
        <div style={{ fontSize:13, color:C.text.secondary, lineHeight:1.5 }}>Simulates buying the recommended amount on every MEDIUM+ signal day, holding forever.</div>
      </div>
      <div style={{ display:"flex", gap:6, marginBottom:16 }}>
        {ranges.map(r => (
          <button key={r.v} className={"pill-btn "+(range===r.v?"active":"")} onClick={() => { setRange(r.v); load(r.v); }}>{r.l}</button>
        ))}
      </div>
      {loading ? (<><div className="card"><div className="skeleton" style={{ height:120 }} /></div><div className="card"><div className="skeleton" style={{ height:180 }} /></div></>) :
       error ? <div style={{ textAlign:"center", padding:"40px 20px" }}><div style={{ fontSize:13, color:C.text.secondary, marginBottom:12 }}>{error}</div></div> :
       !data ? (
        <div style={{ textAlign:"center", padding:"40px 20px", color:C.text.secondary, fontSize:14, lineHeight:1.6 }}>
          Not enough signal history yet.<br />Come back once the system has been running for a few weeks.
        </div>
       ) : (
        <>
          <div className="card">
            <div style={{ fontSize:12, color:C.text.muted, textTransform:"uppercase", letterSpacing:.8, marginBottom:10 }}>Simulated Result · {ranges.find(r=>r.v===range)?.l}</div>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10, marginBottom:10 }}>
              {[
                { label:"Simulated Invested", value: fmt.aud(data.total_invested) },
                { label:"Current Value",      value: fmt.aud(data.current_value), color: profitColor },
                { label:"Profit / Loss",      value: (data.return_dollar>=0?"+":"")+fmt.aud(data.return_dollar), color: profitColor },
                { label:"Return",             value: (data.return_pct>=0?"+":"")+Number(data.return_pct).toFixed(2)+"%", color: profitColor },
              ].map(s => (
                <div key={s.label} style={{ background:"#141414", borderRadius:8, padding:"10px 12px" }}>
                  <div style={{ fontSize:11, color:C.text.muted, marginBottom:4 }}>{s.label}</div>
                  <div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:15, fontWeight:500, color: s.color || "#F0F0F0" }}>{s.value}</div>
                </div>
              ))}
            </div>
            <div style={{ display:"flex", justifyContent:"space-between", paddingTop:8, borderTop:"1px solid #252525" }}>
              <span style={{ fontSize:12, color:C.text.secondary }}>Buys: <span style={{ fontFamily:"'JetBrains Mono',monospace" }}>{data.trade_count}</span></span>
              <span style={{ fontSize:12, color:C.text.secondary }}>Units: <span style={{ fontFamily:"'JetBrains Mono',monospace" }}>{Number(data.total_units).toFixed(4)}</span></span>
              <span style={{ fontSize:12, color:C.text.secondary }}>@ <span style={{ fontFamily:"'JetBrains Mono',monospace", color:"#3B82F6" }}>${Number(data.current_price).toFixed(2)}</span></span>
            </div>
          </div>
          {chartData.length > 1 && (
            <div className="card">
              <div style={{ fontSize:12, color:C.text.muted, textTransform:"uppercase", letterSpacing:.8, marginBottom:10 }}>Cumulative Invested vs Value</div>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="btV" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#10B981" stopOpacity={0.3}/><stop offset="95%" stopColor="#10B981" stopOpacity={0}/></linearGradient>
                    <linearGradient id="btI" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#3B82F6" stopOpacity={0.2}/><stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/></linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#252525" />
                  <XAxis dataKey="date" tick={{ fill:C.text.muted, fontSize:10 }} interval={Math.max(0,Math.floor(chartData.length/5)-1)} />
                  <YAxis tick={{ fill:C.text.muted, fontSize:11 }} width={55} tickFormatter={v=>"$"+Number(v).toFixed(0)} />
                  <Tooltip contentStyle={{ background:"#1C1C1C", border:"1px solid #252525", borderRadius:8, fontSize:12 }} formatter={v=>[fmt.aud(v)]} />
                  <Area type="stepAfter" dataKey="invested" stroke="#3B82F6" fill="url(#btI)" strokeWidth={2} name="Invested" />
                  <Area type="monotone"  dataKey="value"    stroke="#10B981" fill="url(#btV)" strokeWidth={2} name="Value" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
          {data.tier_summary?.length > 0 && (
            <div className="card">
              <div style={{ fontSize:12, color:C.text.muted, textTransform:"uppercase", letterSpacing:.8, marginBottom:10 }}>By Signal Tier</div>
              {["EXTREME","HIGH","MEDIUM"].map(tier => {
                const r = data.tier_summary.find(t => t.tier === tier);
                if (!r) return null;
                return (
                  <div key={tier} style={{ display:"flex", justifyContent:"space-between", alignItems:"center", padding:"10px 0", borderBottom:"1px solid #252525" }}>
                    <div style={{ display:"flex", alignItems:"center", gap:10 }}>
                      <div style={{ display:"inline-flex", alignItems:"center", background:sigColors[tier]+"18", border:"2px solid "+sigColors[tier], borderRadius:999, padding:"4px 12px" }}>
                        <span style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:13, fontWeight:800, color:sigColors[tier], letterSpacing:2 }}>{tier}</span>
                      </div>
                      <span style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:12, color:C.text.secondary }}>{r.count} buys</span>
                    </div>
                    <div style={{ textAlign:"right" }}>
                      <div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:13, color: r.return_pct >= 0 ? "#10B981" : "#EF4444" }}>{r.return_pct >= 0 ? "+" : ""}{Number(r.return_pct).toFixed(1)}%</div>
                      <div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:11, color:C.text.secondary }}>{fmt.aud(r.invested)}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          <div className="card">
            <div style={{ fontSize:12, color:C.text.muted, textTransform:"uppercase", letterSpacing:.8, marginBottom:10 }}>Signal Trade Log</div>
            <div style={{ maxHeight:300, overflowY:"auto" }}>
              {[...(data.trades||[])].reverse().map((t, i) => (
                <div key={i} style={{ display:"flex", justifyContent:"space-between", alignItems:"center", padding:"8px 0", borderBottom:"1px solid #1E1E1E" }}>
                  <div>
                    <div style={{ fontSize:13, color:"#F0F0F0" }}>{fmt.date(t.date)}</div>
                    <div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:11, color:C.text.secondary }}>@ ${Number(t.buy_price).toFixed(2)} · {Number(t.units).toFixed(4)} units</div>
                  </div>
                  <div style={{ textAlign:"right" }}>
                    <div style={{ color:sigColors[t.signal_tier]||C.text.secondary, fontSize:12, fontWeight:600 }}>{t.signal_tier}</div>
                    <div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:13, color:"#F0F0F0" }}>{fmt.aud(t.amount)}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
       )}
    </div>
  );
}

// ─── SETTINGS TAB ─────────────────────────────────────────────────────────────
function SettingsTab() {
  const [settings, setSettings] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState(null);
  const sigColors = { LOW:"#EAB308", MEDIUM:"#F97316", HIGH:"#EF4444", EXTREME:"#8B5CF6" };

  async function loadSettings() {
    setLoading(true); setError(null);
    try { const res = await api("/settings"); setSettings(res.settings); }
    catch { setError("Unable to load settings."); }
    setLoading(false);
  }

  useEffect(() => { loadSettings(); }, []);

  async function saveSettings() {
    setSaving(true); setSaved(false);
    try {
      await api("/settings", { method:"PATCH", body: JSON.stringify({
        low_rsi_below: parseFloat(settings.low_rsi_below), low_vix_above: parseFloat(settings.low_vix_above), low_amount: parseFloat(settings.low_amount),
        medium_rsi_below: parseFloat(settings.medium_rsi_below), medium_vix_above: parseFloat(settings.medium_vix_above), medium_amount: parseFloat(settings.medium_amount),
        high_rsi_below: parseFloat(settings.high_rsi_below), high_vix_above: parseFloat(settings.high_vix_above), high_amount: parseFloat(settings.high_amount),
        extreme_rsi_below: parseFloat(settings.extreme_rsi_below), extreme_vix_above: parseFloat(settings.extreme_vix_above),
        extreme_drawdown_below: parseFloat(settings.extreme_drawdown_below), extreme_amount: parseFloat(settings.extreme_amount),
      })});
      setSaved(true); setTimeout(() => setSaved(false), 3000);
    } catch { setError("Failed to save settings."); }
    setSaving(false);
  }

  const tiers = [
    { name:"LOW",     rsi:"low_rsi_below",     vix:"low_vix_above",     amt:"low_amount",     dd:null },
    { name:"MEDIUM",  rsi:"medium_rsi_below",  vix:"medium_vix_above",  amt:"medium_amount",  dd:null },
    { name:"HIGH",    rsi:"high_rsi_below",    vix:"high_vix_above",    amt:"high_amount",    dd:null },
    { name:"EXTREME", rsi:"extreme_rsi_below", vix:"extreme_vix_above", amt:"extreme_amount", dd:"extreme_drawdown_below" },
  ];

  function Fld({ label, k, step=0.5 }) {
    return (
      <div style={{ flex:1, minWidth:70 }}>
        <label style={{ fontSize:11, color:C.text.muted, display:"block", marginBottom:4 }}>{label}</label>
        <input type="number" step={step} value={settings?.[k]??""} onChange={e => setSettings(s=>({...s,[k]:e.target.value}))} style={{ fontSize:13, padding:"7px 10px" }} />
      </div>
    );
  }

  return (
    <div className="tab-content fade-in">
      <div style={{ marginBottom:16 }}>
        <div style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:22, fontWeight:700, marginBottom:4 }}>Signal Settings</div>
        <div style={{ fontSize:13, color:C.text.secondary, lineHeight:1.5 }}>Adjust thresholds and investment amounts. Changes apply at next 8am AEDT cron run.</div>
      </div>
      {loading ? <><div className="card"><div className="skeleton" style={{ height:120 }} /></div><div className="card"><div className="skeleton" style={{ height:120 }} /></div></> :
       error ? <div style={{ textAlign:"center", padding:"40px 20px", color:C.text.secondary }}>{error}</div> : (
        <>
          {tiers.map(t => (
            <div key={t.name} className="card" style={{ borderColor:sigColors[t.name]+"30" }}>
              <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:12 }}>
                <div style={{ display:"inline-flex", alignItems:"center", background:sigColors[t.name]+"18", border:"2px solid "+sigColors[t.name], borderRadius:999, padding:"4px 12px" }}>
                  <span style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:13, fontWeight:800, color:sigColors[t.name], letterSpacing:2 }}>{t.name}</span>
                </div>
                <span style={{ fontSize:12, color:C.text.muted }}>RSI AND VIX{t.dd?" AND Drawdown":""} must be met</span>
              </div>
              <div style={{ display:"flex", gap:8, marginBottom: t.dd?8:0 }}>
                <Fld label="RSI below" k={t.rsi} />
                <Fld label="VIX above" k={t.vix} />
                <Fld label="Invest ($)" k={t.amt} step={50} />
              </div>
              {t.dd && (
                <div style={{ display:"flex", gap:8 }}>
                  <Fld label="Drawdown below (%)" k={t.dd} />
                  <div style={{ flex:2, fontSize:12, color:C.text.muted, padding:"8px 10px", background:"#141414", borderRadius:8, alignSelf:"flex-end" }}>e.g. -10 = 10% below 52w high</div>
                </div>
              )}
            </div>
          ))}
          <div style={{ padding:"12px 14px", background:"#141414", borderRadius:8, marginBottom:12, fontSize:12, color:C.text.muted, lineHeight:1.8 }}>
            <strong style={{ color:"#9CA3AF" }}>Defaults:</strong> LOW: RSI&lt;50, VIX&gt;18 · MEDIUM: RSI&lt;45, VIX&gt;20, $200 · HIGH: RSI&lt;35, VIX&gt;25, $400 · EXTREME: RSI&lt;30, VIX&gt;30, DD&lt;-10%, $800
          </div>
          <button className="btn btn-primary" style={{ width:"100%", padding:"12px", fontSize:15 }} onClick={saveSettings} disabled={saving}>
            {saving ? "Saving…" : saved ? "✓ Saved" : "Save Settings"}
          </button>
          {saved && <div style={{ marginTop:10, padding:"10px 14px", background:"#10B98115", border:"1px solid #10B98130", borderRadius:8, fontSize:13, color:"#10B981", textAlign:"center" }}>Settings saved — applies at next 8am signal run</div>}
        </>
       )}
    </div>
  );
}

// ─── NAV ICONS ────────────────────────────────────────────────────────────────
const icons = {
  today:   <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M2 12h2M20 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>,
  charts:  <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>,
  history: <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>,
  perf:    <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/><line x1="2" y1="20" x2="22" y2="20"/></svg>,
  ledger:  <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>,
  more:    <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24"><circle cx="5" cy="12" r="1.5"/><circle cx="12" cy="12" r="1.5"/><circle cx="19" cy="12" r="1.5"/></svg>,
};

// ─── MORE DRAWER ──────────────────────────────────────────────────────────────
function MoreDrawer({ onSelect, onClose }) {
  const items = [
    { id:"ledger",   label:"Ledger",          icon:<svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>, desc:"Log your investment entries" },
    { id:"backtest", label:"Backtest",        icon:<svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>, desc:"Simulate signal-based investing" },
    { id:"settings", label:"Signal Settings", icon:<svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>, desc:"Configure thresholds and amounts" },
  ];
  return (
    <div className="overlay" onClick={onClose}>
      <div className="sheet slide-up" style={{ paddingBottom:100 }} onClick={e => e.stopPropagation()}>
        <div style={{ width:36, height:4, background:"#252525", borderRadius:2, margin:"0 auto 20px" }} />
        <div style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:18, fontWeight:700, color:"#9CA3AF", marginBottom:16, letterSpacing:1 }}>MORE</div>
        {items.map(item => (
          <button key={item.id} onClick={() => { onSelect(item.id); onClose(); }}
            style={{ width:"100%", display:"flex", alignItems:"center", gap:14, padding:"14px 12px", background:"transparent", border:"none", borderRadius:10, cursor:"pointer", marginBottom:4, textAlign:"left", transition:"background .15s" }}
            onMouseEnter={e => e.currentTarget.style.background="#252525"}
            onMouseLeave={e => e.currentTarget.style.background="transparent"}
          >
            <div style={{ width:40, height:40, borderRadius:10, background:"#252525", display:"flex", alignItems:"center", justifyContent:"center", color:"#9CA3AF", flexShrink:0 }}>
              {item.icon}
            </div>
            <div>
              <div style={{ fontSize:15, fontWeight:600, color:"#F0F0F0" }}>{item.label}</div>
              <div style={{ fontSize:12, color:C.text.muted, marginTop:2 }}>{item.desc}</div>
            </div>
            <div style={{ marginLeft:"auto", color:C.text.muted, fontSize:18 }}>›</div>
          </button>
        ))}
      </div>
    </div>
  );
}

// ─── PERSISTENT HEADER ────────────────────────────────────────────────────────
function PersistentHeader({ signal }) {
  const dateStr = new Date().toLocaleDateString("en-AU", { weekday:"short", day:"numeric", month:"short" });
  const tier = signal ? normaliseTier(signal.signal_tier) : "NONE";
  const price = signal ? Number(signal.vdgr_price) : null;

  return (
    <div style={{
      position:"fixed",
      top:0,
      left:"50%",
      transform:"translateX(-50%)",
      width:"100%",
      maxWidth:520,
      height:56,
      background:"rgba(10,10,10,0.97)",
      borderBottom:"1px solid #252525",
      backdropFilter:"blur(16px)",
      WebkitBackdropFilter:"blur(16px)",
      display:"flex",
      alignItems:"center",
      justifyContent:"space-between",
      padding:"0 16px",
      zIndex:100,
    }}>
      <div>
        <div style={{ fontFamily:"'Barlow Condensed',sans-serif", fontSize:20, fontWeight:800, color:"#F0F0F0" }}>VDGR</div>
        <div style={{ fontSize:11, color:C.text.muted }}>{dateStr}</div>
      </div>
      <div style={{ display:"flex", alignItems:"center", gap:12 }}>
        <SignalBadge signal={tier} size="sm" />
        {price && (
          <div style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:13, color:C.accent }}>${price.toFixed(2)}</div>
        )}
      </div>
    </div>
  );
}

// ─── APP ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("today");
  const [showMore, setShowMore] = useState(false);
  const [historyData, setHistoryData] = useState([]);
  const [globalSignal, setGlobalSignal] = useState(null);

  useEffect(() => {
    api("/signal/chart?range=1y")
      .then(d => {
        console.log('Chart data (Yahoo Finance):', d.chartData?.length, d.chartData?.[0], d.chartData?.[d.chartData?.length-1]);
        const mapped = (d.chartData || []).map(item => ({
          date: item.date,
          signal_tier: item.signal_tier === 'WATCH' ? 'LOW' : (item.signal_tier || 'NONE'),
          rsi: item.rsi,
          vix: item.vix,
          vdgr_price: item.price,
          drawdown_pct: item.drawdown,
          recommended_amount: 0,
        }));
        console.log("History data loaded from Yahoo Finance:", mapped.length, "days");
        setHistoryData(mapped);
      })
      .catch(() => {});
    // Fetch global signal for header
    api("/signal/today")
      .then(d => setGlobalSignal(d.signal))
      .catch(() => {});
  }, []);

  function CalendarHistoryTab() {
    const [view, setView] = useState("calendar");
    return (
      <div className="tab-content fade-in">
        <div style={{ display:"flex", gap:6, marginBottom:16 }}>
          <button className={"pill-btn "+(view==="calendar"?"active":"")} onClick={() => setView("calendar")}>Calendar</button>
          <button className={"pill-btn "+(view==="list"?"active":"")} onClick={() => setView("list")}>Signal List</button>
        </div>
        {view === "calendar" ? <SignalCalendar history={historyData} /> : <HistoryList history={historyData} />}
      </div>
    );
  }

  const tabs = [
    { id:"today",       label:"Today",       icon:icons.today },
    { id:"charts",      label:"Charts",      icon:icons.charts },
    { id:"history",     label:"History",     icon:icons.history },
    { id:"perf",        label:"Performance", icon:icons.perf },
    { id:"more",        label:"More",        icon:icons.more },
  ];

  const morePages = ["ledger","backtest","settings"];
  const activePrimary = morePages.includes(tab) ? "more" : tab;

  return (
    <>
      <style>{globalStyle}</style>
      <PersistentHeader signal={globalSignal} />
      <div style={{ background:"#0A0A0A", minHeight:"100vh", color:"#F0F0F0" }}>
        {tab==="today"    && <TodayTab />}
        {tab==="charts"   && <ChartsTab />}
        {tab==="history"  && <CalendarHistoryTab />}
        {tab==="ledger"   && <LedgerTab />}
        {tab==="perf"     && <PerformanceTab />}
        {tab==="backtest" && <BacktestTab />}
        {tab==="settings" && <SettingsTab />}

        {/* Fixed bottom nav */}
        <div style={{
          position:"fixed",
          bottom:0,
          left:"50%",
          transform:"translateX(-50%)",
          width:"100%",
          maxWidth:520,
          background:"rgba(10,10,10,.97)",
          borderTop:"1px solid #252525",
          backdropFilter:"blur(16px)",
          WebkitBackdropFilter:"blur(16px)",
          display:"flex",
          zIndex:100,
        }}>
          {tabs.map(t => {
            const active = activePrimary === t.id;
            return (
              <button key={t.id}
                onClick={() => {
                  if (t.id === "more") { setShowMore(true); }
                  else { setTab(t.id); setShowMore(false); }
                }}
                style={{
                  flex:1, padding:"10px 4px 14px", background:"transparent", border:"none",
                  cursor:"pointer", display:"flex", flexDirection:"column", alignItems:"center",
                  gap:4, color: active ? "#3B82F6" : "#9CA3AF", transition:"color .15s",
                  position:"relative", WebkitTapHighlightColor:"transparent",
                }}
              >
                {active && (
                  <div style={{ position:"absolute", top:0, left:"50%", transform:"translateX(-50%)", width:28, height:2, background:"#3B82F6", borderRadius:"0 0 2px 2px" }} />
                )}
                {t.icon}
                <span style={{ fontSize:10, fontWeight: active ? 600 : 400, letterSpacing:.3 }}>{t.label}</span>
              </button>
            );
          })}
        </div>

        {showMore && (
          <MoreDrawer onSelect={(id) => setTab(id)} onClose={() => setShowMore(false)} />
        )}
      </div>
    </>
  );
}
