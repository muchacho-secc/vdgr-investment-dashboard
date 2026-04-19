// signal.js — daily signal engine

import { supabaseGet, supabasePostMinimal, supabasePatch } from './supabase.js';
import { generateAnalystSummary } from './analyst.js';
import { sendTelegramNotification, sendTelegramError } from './telegram.js';

// ─── Default Thresholds ───────────────────────────────────────────────────────
const DEFAULT_THRESHOLDS = [
  { tier: 'EXTREME', rsiBelow: 30, vixAbove: 30, drawdownBelow: -10, amount: 800 },
  { tier: 'HIGH',    rsiBelow: 35, vixAbove: 25, drawdownBelow: null, amount: 400 },
  { tier: 'MEDIUM',  rsiBelow: 45, vixAbove: 20, drawdownBelow: null, amount: 200 },
  { tier: 'WATCH',   rsiBelow: 50, vixAbove: 18, drawdownBelow: null, amount: 0   },
];

async function loadThresholds(env) {
  try {
    const rows = await supabaseGet(env, 'user_settings', `?user_id=eq.${env.USER_ID}&limit=1`);
    if (!rows.length) return DEFAULT_THRESHOLDS;
    const s = rows[0];
    return [
      { tier: 'EXTREME', rsiBelow: +s.extreme_rsi_below, vixAbove: +s.extreme_vix_above, drawdownBelow: +s.extreme_drawdown_below, amount: +s.extreme_amount },
      { tier: 'HIGH',    rsiBelow: +s.high_rsi_below,    vixAbove: +s.high_vix_above,    drawdownBelow: null, amount: +s.high_amount },
      { tier: 'MEDIUM',  rsiBelow: +s.medium_rsi_below,  vixAbove: +s.medium_vix_above,  drawdownBelow: null, amount: +s.medium_amount },
      { tier: 'WATCH',   rsiBelow: +s.low_rsi_below,     vixAbove: +s.low_vix_above,     drawdownBelow: null, amount: +s.low_amount },
    ];
  } catch (err) {
    console.error('[signal] Failed to load thresholds, using defaults:', err.message);
    return DEFAULT_THRESHOLDS;
  }
}

// ─── Yahoo Finance ────────────────────────────────────────────────────────────
async function fetchVDGRHistory(range = '3mo') {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/VDGR.AX?range=${range}&interval=1d`;
  const res = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
  if (!res.ok) throw new Error(`Yahoo Finance VDGR fetch failed: ${res.status}`);
  const data = await res.json();
  const result = data.chart?.result?.[0];
  if (!result) throw new Error('Yahoo Finance VDGR: no result');
  return result.timestamp
    .map((t, i) => ({ date: new Date(t * 1000), close: result.indicators.quote[0].close[i] }))
    .filter(p => p.close !== null && !isNaN(p.close));
}

async function fetchVIX() {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?range=5d&interval=1d`;
  const res = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
  if (!res.ok) throw new Error(`Yahoo Finance VIX fetch failed: ${res.status}`);
  const data = await res.json();
  const closes = data.chart?.result?.[0]?.indicators.quote[0].close.filter(v => v !== null && !isNaN(v));
  if (!closes?.length) throw new Error('VIX: no valid prices');
  return Math.round(closes[closes.length - 1] * 100) / 100;
}

// ─── Indicators ───────────────────────────────────────────────────────────────
function calculateRSI(prices, period = 14) {
  if (prices.length < period + 1) throw new Error(`Not enough price data for RSI`);
  const changes = prices.slice(1).map((p, i) => p.close - prices[i].close);
  const gains = changes.map(c => c > 0 ? c : 0);
  const losses = changes.map(c => c < 0 ? Math.abs(c) : 0);
  let avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
  let avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = period; i < changes.length; i++) {
    avgGain = (avgGain * (period - 1) + gains[i]) / period;
    avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
  }
  if (avgLoss === 0) return 100;
  return Math.round((100 - (100 / (1 + avgGain / avgLoss))) * 100) / 100;
}

function calculateDrawdown(prices) {
  const closes = prices.map(p => p.close);
  const high52w = Math.max(...closes);
  const currentPrice = closes[closes.length - 1];
  return {
    high52w: Math.round(high52w * 10000) / 10000,
    drawdownPct: Math.round(((currentPrice - high52w) / high52w) * 10000) / 100,
  };
}

// ─── Signal Classification ────────────────────────────────────────────────────
export function calculateSignal(rsi, vix, drawdownPct, thresholds) {
  for (const t of thresholds) {
    const rsiOk = rsi < t.rsiBelow;
    const vixOk = vix > t.vixAbove;
    const ddOk  = t.drawdownBelow === null || drawdownPct < t.drawdownBelow;
    if (rsiOk && vixOk && ddOk) {
      return { tier: t.tier, recommendedAmount: t.amount };
    }
  }
  return { tier: 'NONE', recommendedAmount: 0 };
}

// ─── Supabase Helpers ─────────────────────────────────────────────────────────
async function checkExistingSignal(env, date) {
  const rows = await supabaseGet(env, 'signals', `?date=eq.${date}&select=id`);
  return rows.length > 0;
}

async function storeSignal(env, signal) {
  await supabasePostMinimal(env, 'signals', { ...signal, user_id: env.USER_ID });
}

async function markNotificationSent(env, date) {
  await supabasePatch(env, 'signals', `?date=eq.${date}`, { notification_sent: true });
}

async function updatePerformanceSnapshot(env, date, vdgrPrice) {
  try {
    const entries = await supabaseGet(env, 'ledger',
      `?user_id=eq.${env.USER_ID}&select=actual_amount,units_acquired&order=date.asc`
    );
    if (!entries.length) return;
    const totalInvested = entries.reduce((s, e) => s + parseFloat(e.actual_amount), 0);
    const totalUnits    = entries.reduce((s, e) => s + parseFloat(e.units_acquired), 0);
    const currentValue  = totalUnits * vdgrPrice;
    const returnDollar  = currentValue - totalInvested;
    const returnPct     = totalInvested > 0 ? (returnDollar / totalInvested) * 100 : 0;
    await supabasePostMinimal(env, 'performance_snapshots', {
      user_id: env.USER_ID, date, vdgr_price: vdgrPrice,
      total_units: Math.round(totalUnits * 1e6) / 1e6,
      total_invested: Math.round(totalInvested * 100) / 100,
      current_value:  Math.round(currentValue  * 100) / 100,
      return_dollar:  Math.round(returnDollar  * 100) / 100,
      return_pct:     Math.round(returnPct     * 100) / 100,
    });
  } catch (err) {
    console.error('Performance snapshot failed:', err.message);
  }
}

// ─── Main Daily Signal Runner ─────────────────────────────────────────────────
export async function runDailySignal(env) {
  const today = new Date().toISOString().split('T')[0];
  console.log(`[signal] Starting daily run for ${today}`);

  if (await checkExistingSignal(env, today)) {
    console.log(`[signal] Already exists for ${today}, skipping`);
    return;
  }

  const thresholds = await loadThresholds(env);

  const [vdgrPrices3m, vdgrPrices1y, vix] = await Promise.all([
    fetchVDGRHistory('3mo'),
    fetchVDGRHistory('1y'),
    fetchVIX(),
  ]);

  const rsi          = calculateRSI(vdgrPrices3m);
  const currentPrice = vdgrPrices3m[vdgrPrices3m.length - 1].close;
  const { high52w, drawdownPct } = calculateDrawdown(vdgrPrices1y);
  const { tier, recommendedAmount } = calculateSignal(rsi, vix, drawdownPct, thresholds);

  console.log(`[signal] RSI=${rsi}, VIX=${vix}, DD=${drawdownPct}%, Price=${currentPrice}, Tier=${tier}`);

  let analystSummary = null;
  try {
    analystSummary = await generateAnalystSummary(env, {
      date: today, signal_tier: tier, vdgr_price: currentPrice,
      rsi, vix, drawdown_pct: drawdownPct, recommended_amount: recommendedAmount,
    });
  } catch (err) {
    console.error('[signal] Analyst summary failed:', err.message);
  }

  await storeSignal(env, {
    date: today, signal_tier: tier, rsi, vix,
    vdgr_price:         Math.round(currentPrice * 10000) / 10000,
    drawdown_pct:       drawdownPct,
    high_52w:           high52w,
    recommended_amount: recommendedAmount,
    analyst_summary:    analystSummary,
    notification_sent:  false,
  });

  console.log(`[signal] Stored: ${tier} for ${today}`);
  await updatePerformanceSnapshot(env, today, currentPrice);

  if (['MEDIUM', 'HIGH', 'EXTREME'].includes(tier)) {
    try {
      await sendTelegramNotification(env, { tier, recommendedAmount, analystSummary: analystSummary || 'Summary unavailable.', rsi, vix, currentPrice, drawdownPct });
      await markNotificationSent(env, today);
    } catch (err) {
      console.error('[signal] Telegram failed:', err.message);
    }
  }

  console.log(`[signal] Complete: ${tier} for ${today}`);
}
