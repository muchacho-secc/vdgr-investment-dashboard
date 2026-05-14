// routes/signalRoutes.js

import { supabaseGet } from '../supabase.js';

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

// ─── Yahoo Finance helpers ────────────────────────────────────────────────────
async function fetchYahoo(ticker, range) {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?range=${range}&interval=1d`;
  const res = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
  if (!res.ok) throw new Error(`Yahoo Finance ${ticker} fetch failed: ${res.status}`);
  const data = await res.json();
  const result = data.chart?.result?.[0];
  if (!result) throw new Error(`No data returned for ${ticker}`);
  const timestamps = result.timestamp;
  const closes = result.indicators.quote[0].close;
  return timestamps
    .map((t, i) => ({ date: new Date(t * 1000).toISOString().split('T')[0], close: closes[i] }))
    .filter(p => p.close !== null && !isNaN(p.close));
}

// ─── RSI series (14-day Wilder's smoothing) ───────────────────────────────────
function computeRSISeries(prices, period = 14) {
  const result = new Array(prices.length).fill(null);
  if (prices.length < period + 1) return result;

  const changes = prices.slice(1).map((p, i) => p.close - prices[i].close);
  const gains   = changes.map(c => c > 0 ? c : 0);
  const losses  = changes.map(c => c < 0 ? Math.abs(c) : 0);

  let avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
  let avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;

  const rsiVal = (ag, al) => al === 0 ? 100 : Math.round((100 - 100 / (1 + ag / al)) * 100) / 100;
  result[period] = rsiVal(avgGain, avgLoss);

  for (let i = period; i < changes.length; i++) {
    avgGain = (avgGain * (period - 1) + gains[i]) / period;
    avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
    result[i + 1] = rsiVal(avgGain, avgLoss);
  }

  return result;
}

// ─── Rolling 52-week drawdown series ─────────────────────────────────────────
function computeDrawdownSeries(prices) {
  const windowSize = 252;
  return prices.map((p, i) => {
    const window = prices.slice(Math.max(0, i - windowSize + 1), i + 1);
    const high = Math.max(...window.map(w => w.close));
    return high > 0 ? Math.round(((p.close - high) / high) * 10000) / 100 : null;
  });
}

// ─── Signal classification (mirrors signal.js) ────────────────────────────────
function classifyTier(rsi, vix, drawdown, t) {
  if (!rsi || !vix) return 'NONE';
  if (rsi < t.extreme_rsi && vix > t.extreme_vix && drawdown !== null && drawdown < t.extreme_dd) return 'EXTREME';
  if (rsi < t.high_rsi   && vix > t.high_vix)   return 'HIGH';
  if (rsi < t.medium_rsi && vix > t.medium_vix) return 'MEDIUM';
  if (rsi < t.low_rsi    && vix > t.low_vix)    return 'WATCH';
  return 'NONE';
}

// ─── Load user thresholds ─────────────────────────────────────────────────────
async function loadThresholdMap(env) {
  const defaults = { extreme_rsi:30, extreme_vix:30, extreme_dd:-10, high_rsi:35, high_vix:25, medium_rsi:45, medium_vix:20, low_rsi:50, low_vix:18 };
  try {
    const rows = await supabaseGet(env, 'user_settings', `?user_id=eq.${env.USER_ID}&limit=1`);
    if (!rows.length) return defaults;
    const s = rows[0];
    return {
      extreme_rsi: +s.extreme_rsi_below, extreme_vix: +s.extreme_vix_above, extreme_dd: +s.extreme_drawdown_below,
      high_rsi: +s.high_rsi_below, high_vix: +s.high_vix_above,
      medium_rsi: +s.medium_rsi_below, medium_vix: +s.medium_vix_above,
      low_rsi: +s.low_rsi_below, low_vix: +s.low_vix_above,
    };
  } catch { return defaults; }
}

// ─── Chart data builder ───────────────────────────────────────────────────────
async function getChartData(env, range = '3mo') {
  // Always fetch 2y of VDGR so RSI warmup period is accurate
  const vdgrFetchRange = '2y';

  const [vdgrPrices, vixPrices, thresholds] = await Promise.all([
    fetchYahoo('VDGR.AX', vdgrFetchRange),
    fetchYahoo('%5EVIX', vdgrFetchRange),
    loadThresholdMap(env),
  ]);

  // Build VIX lookup by date
  const vixMap = Object.fromEntries(vixPrices.map(v => [v.date, Math.round(v.close * 100) / 100]));

  // Calculate indicator series over full history
  const rsiSeries      = computeRSISeries(vdgrPrices);
  const drawdownSeries = computeDrawdownSeries(vdgrPrices);

  // Cutoff date for the requested display range
  const now = new Date();
  const cutoff = {
    '1mo': new Date(now.getFullYear(), now.getMonth() - 1,  now.getDate()),
    '3mo': new Date(now.getFullYear(), now.getMonth() - 3,  now.getDate()),
    '6mo': new Date(now.getFullYear(), now.getMonth() - 6,  now.getDate()),
    '1y':  new Date(now.getFullYear() - 1, now.getMonth(), now.getDate()),
  }[range] || new Date(now.getFullYear(), now.getMonth() - 3, now.getDate());

  // Fetch stored signals (only used to overlay confirmed tiers)
  let storedMap = {};
  try {
    const stored = await supabaseGet(env, 'signals',
      `?user_id=eq.${env.USER_ID}&order=date.desc&limit=500&select=date,signal_tier`
    );
    storedMap = Object.fromEntries(stored.map(s => [s.date, s.signal_tier]));
  } catch { /* non-fatal */ }

  // Build output — only return days within the display range
  return vdgrPrices
    .map((p, i) => {
      if (new Date(p.date) < cutoff) return null;

      const rsi      = rsiSeries[i];
      const drawdown = drawdownSeries[i];
      const vix      = vixMap[p.date] ?? null;

      // Prefer stored confirmed tier, fall back to calculated
      const signal_tier = storedMap[p.date] || (rsi && vix ? classifyTier(rsi, vix, drawdown, thresholds) : 'NONE');

      return {
        date:        p.date,
        price:       Math.round(p.close * 10000) / 10000,
        rsi:         rsi      !== null ? Math.round(rsi      * 100) / 100 : null,
        vix:         vix      !== null ? vix                              : null,
        drawdown:    drawdown !== null ? drawdown                         : null,
        signal_tier,
      };
    })
    .filter(Boolean);
}

// ─── Route handlers ───────────────────────────────────────────────────────────
async function getTodaySignal(env) {
  const rows = await supabaseGet(env, 'signals',
    `?user_id=eq.${env.USER_ID}&order=date.desc&limit=1&select=*`
  );
  return rows[0] || null;
}

async function getSignalHistory(env, days = 90) {
  return supabaseGet(env, 'signals',
    `?user_id=eq.${env.USER_ID}&order=date.desc&limit=${days}&select=date,signal_tier,rsi,vix,vdgr_price,drawdown_pct,recommended_amount,analyst_summary`
  );
}

export async function handleSignalRoutes(request, env, url) {
  // GET /signal/today
  if (url.pathname === '/signal/today' && request.method === 'GET') {
    const signal = await getTodaySignal(env);
    if (!signal) return json({ signal: null, message: 'No signal generated yet. Check back after 8am AEDT.' });
    return json({ signal });
  }

  // GET /signal/history?days=90
  if (url.pathname === '/signal/history' && request.method === 'GET') {
    const days = parseInt(url.searchParams.get('days') || '90', 10);
    const history = await getSignalHistory(env, Math.min(days, 365));
    return json({ history });
  }

  // GET /signal/chart?range=3mo
  if (url.pathname === '/signal/chart' && request.method === 'GET') {
    const range = url.searchParams.get('range') || '3mo';
    const validRanges = ['1mo', '3mo', '6mo', '1y'];
    const safeRange = validRanges.includes(range) ? range : '3mo';
    const chartData = await getChartData(env, safeRange);
    return json({ chartData });
  }

  return json({ error: 'Not found' }, 404);
}
