// routes/signalRoutes.js

import { supabaseGet } from '../supabase.js';
import { computeSignalSeries } from '../signalComputation.js';

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
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
  // Load thresholds
  const thresholdMap = await loadThresholdMap(env);
  const thresholds = {
    extreme_rsi: thresholdMap.extreme_rsi,
    extreme_vix: thresholdMap.extreme_vix,
    extreme_dd: thresholdMap.extreme_dd,
    high_rsi: thresholdMap.high_rsi,
    high_vix: thresholdMap.high_vix,
    medium_rsi: thresholdMap.medium_rsi,
    medium_vix: thresholdMap.medium_vix,
    low_rsi: thresholdMap.low_rsi,
    low_vix: thresholdMap.low_vix,
  };

  // Get full signal series using shared computation
  const fullSeries = await computeSignalSeries({ thresholds });

  // Fetch stored signals for cross-check
  let storedMap = {};
  try {
    const stored = await supabaseGet(env, 'signals',
      `?user_id=eq.${env.USER_ID}&order=date.desc&limit=500&select=date,signal_tier,rsi,vix`
    );
    storedMap = Object.fromEntries(stored.map(s => [s.date, s]));
  } catch { /* non-fatal */ }

  // Cutoff date for the requested display range
  const now = new Date();
  const cutoff = {
    '1mo': new Date(now.getFullYear(), now.getMonth() - 1,  now.getDate()),
    '3mo': new Date(now.getFullYear(), now.getMonth() - 3,  now.getDate()),
    '6mo': new Date(now.getFullYear(), now.getMonth() - 6,  now.getDate()),
    '1y':  new Date(now.getFullYear() - 1, now.getMonth(), now.getDate()),
  }[range] || new Date(now.getFullYear(), now.getMonth() - 3, now.getDate());

  // Filter to display range and cross-check with stored tiers
  return fullSeries
    .filter(p => new Date(p.date) >= cutoff)
    .map(p => {
      const stored = storedMap[p.date];

      // Cross-check: log warning if stored tier differs from computed
      if (stored && stored.signal_tier !== p.tier) {
        console.warn(`[chart] Tier mismatch for ${p.date}: stored=${stored.signal_tier}, computed=${p.tier} (RSI: stored=${stored.rsi}, computed=${p.rsi}, VIX: stored=${stored.vix}, computed=${p.vix})`);
      }

      return {
        date:        p.date,
        price:       p.price,
        rsi:         p.rsi,
        vix:         p.vix,
        drawdown:    p.drawdown,
        signal_tier: p.tier, // Use computed tier, not stored
      };
    });
}

// ─── Route handlers ───────────────────────────────────────────────────────────
async function getTodaySignal(env) {
  const rows = await supabaseGet(env, 'signals',
    `?user_id=eq.${env.USER_ID}&order=date.desc&limit=1&select=*`
  );
  return rows[0] || null;
}

async function getSignalHistory(env, days = 90) {
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - days);
  const cutoffStr = cutoff.toISOString().split('T')[0];
  return supabaseGet(env, 'signals',
    `?user_id=eq.${env.USER_ID}&date=gte.${cutoffStr}&order=date.desc&limit=500&select=date,signal_tier,rsi,vix,vdgr_price,drawdown_pct,recommended_amount,analyst_summary`
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
    const history = await getSignalHistory(env, Math.min(days, 730));
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
