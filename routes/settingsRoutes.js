// routes/settingsRoutes.js

import { supabaseGet, supabasePatch, supabasePostMinimal } from '../supabase.js';

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

const DEFAULT_SETTINGS = {
  low_rsi_below: 50, low_vix_above: 18, low_amount: 0,
  medium_rsi_below: 45, medium_vix_above: 20, medium_amount: 200,
  high_rsi_below: 35, high_vix_above: 25, high_amount: 400,
  extreme_rsi_below: 30, extreme_vix_above: 30, extreme_drawdown_below: -10, extreme_amount: 800,
};

export async function handleSettingsRoutes(request, env, url) {

  // GET /settings
  if (url.pathname === '/settings' && request.method === 'GET') {
    const rows = await supabaseGet(env, 'user_settings', `?user_id=eq.${env.USER_ID}&limit=1`);
    if (!rows.length) {
      // Auto-create default settings row
      await supabasePostMinimal(env, 'user_settings', { user_id: env.USER_ID, ...DEFAULT_SETTINGS });
      return json({ settings: DEFAULT_SETTINGS });
    }
    return json({ settings: rows[0] });
  }

  // PATCH /settings
  if (url.pathname === '/settings' && request.method === 'PATCH') {
    const body = await request.json().catch(() => null);
    if (!body) return json({ error: 'Invalid request body' }, 400);

    const allowed = [
      'low_rsi_below','low_vix_above','low_amount',
      'medium_rsi_below','medium_vix_above','medium_amount',
      'high_rsi_below','high_vix_above','high_amount',
      'extreme_rsi_below','extreme_vix_above','extreme_drawdown_below','extreme_amount',
    ];

    const updates = Object.fromEntries(
      Object.entries(body).filter(([k]) => allowed.includes(k))
    );

    if (!Object.keys(updates).length) return json({ error: 'No valid fields to update' }, 400);

    // Ensure row exists first
    const existing = await supabaseGet(env, 'user_settings', `?user_id=eq.${env.USER_ID}&limit=1`);
    if (!existing.length) {
      await supabasePostMinimal(env, 'user_settings', { user_id: env.USER_ID, ...DEFAULT_SETTINGS, ...updates });
      return json({ settings: { ...DEFAULT_SETTINGS, ...updates } });
    }

    const updated = await supabasePatch(env, 'user_settings', `?user_id=eq.${env.USER_ID}`, updates);
    return json({ settings: updated[0] });
  }

  // GET /backtest?range=1y
  if (url.pathname === '/backtest' && request.method === 'GET') {
    const range = url.searchParams.get('range') || 'all';
    return handleBacktest(env, range);
  }

  return json({ error: 'Not found' }, 404);
}

async function handleBacktest(env, range) {
  // Fetch all historical signals
  const allSignals = await supabaseGet(env, 'signals',
    `?user_id=eq.${env.USER_ID}&order=date.asc&select=date,signal_tier,vdgr_price,recommended_amount`
  );

  if (!allSignals.length) return json({ backtest: null, message: 'No signal history available yet.' });

  // Get current VDGR price from latest signal
  const latestSignal = allSignals[allSignals.length - 1];
  const currentPrice = parseFloat(latestSignal.vdgr_price);

  // Filter by range
  const now = new Date();
  const cutoff = {
    '1m':  new Date(now.getFullYear(), now.getMonth() - 1,  now.getDate()),
    '3m':  new Date(now.getFullYear(), now.getMonth() - 3,  now.getDate()),
    '6m':  new Date(now.getFullYear(), now.getMonth() - 6,  now.getDate()),
    '1y':  new Date(now.getFullYear() - 1, now.getMonth(),  now.getDate()),
    'all': new Date('2000-01-01'),
  }[range] || new Date('2000-01-01');

  const filtered = allSignals.filter(s =>
    ['MEDIUM', 'HIGH', 'EXTREME'].includes(s.signal_tier) &&
    new Date(s.date) >= cutoff
  );

  if (!filtered.length) {
    return json({ backtest: null, message: `No MEDIUM+ signals found in the selected range.` });
  }

  // Simulate buying recommended amount on every signal day, never selling
  let totalInvested = 0;
  let totalUnits    = 0;
  const trades = [];
  const cumulativeData = [];

  for (const s of filtered) {
    const buyPrice = parseFloat(s.vdgr_price);
    const amount   = parseFloat(s.recommended_amount) || 0;
    if (amount <= 0 || buyPrice <= 0) continue;

    const units = amount / buyPrice;
    totalInvested += amount;
    totalUnits    += units;

    const currentValue  = totalUnits * currentPrice;
    const returnDollar  = currentValue - totalInvested;
    const returnPct     = (returnDollar / totalInvested) * 100;

    trades.push({
      date:          s.date,
      signal_tier:   s.signal_tier,
      buy_price:     Math.round(buyPrice * 100) / 100,
      amount,
      units:         Math.round(units * 1e6) / 1e6,
    });

    cumulativeData.push({
      date:           s.date,
      total_invested: Math.round(totalInvested * 100) / 100,
      total_units:    Math.round(totalUnits * 1e6) / 1e6,
      current_value:  Math.round(currentValue  * 100) / 100,
      return_dollar:  Math.round(returnDollar  * 100) / 100,
      return_pct:     Math.round(returnPct     * 100) / 100,
    });
  }

  if (!trades.length) return json({ backtest: null, message: 'No actionable trades in range.' });

  const finalValue   = totalUnits * currentPrice;
  const returnDollar = finalValue - totalInvested;
  const returnPct    = (returnDollar / totalInvested) * 100;

  // Per-tier breakdown
  const tierBreakdown = {};
  for (const t of trades) {
    if (!tierBreakdown[t.signal_tier]) tierBreakdown[t.signal_tier] = { count: 0, invested: 0, units: 0 };
    tierBreakdown[t.signal_tier].count++;
    tierBreakdown[t.signal_tier].invested += t.amount;
    tierBreakdown[t.signal_tier].units    += t.units;
  }

  const tierSummary = Object.entries(tierBreakdown).map(([tier, b]) => {
    const cv  = b.units * currentPrice;
    const ret = ((cv - b.invested) / b.invested) * 100;
    return {
      tier,
      count:         b.count,
      invested:      Math.round(b.invested * 100) / 100,
      current_value: Math.round(cv   * 100) / 100,
      return_pct:    Math.round(ret  * 100) / 100,
    };
  });

  return json({
    backtest: {
      range,
      current_price:  currentPrice,
      total_invested: Math.round(totalInvested * 100) / 100,
      total_units:    Math.round(totalUnits * 1e6) / 1e6,
      current_value:  Math.round(finalValue  * 100) / 100,
      return_dollar:  Math.round(returnDollar * 100) / 100,
      return_pct:     Math.round(returnPct    * 100) / 100,
      trade_count:    trades.length,
      tier_summary:   tierSummary,
      cumulative:     cumulativeData,
      trades,
    }
  });
}
