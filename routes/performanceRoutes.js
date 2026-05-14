// routes/performanceRoutes.js

import { supabaseGet } from '../supabase.js';

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

export async function handlePerformanceRoutes(request, env, url) {
  // GET /performance
  if (url.pathname === '/performance' && request.method === 'GET') {
    const [ledgerRows, snapshots, latestSignal] = await Promise.all([
      supabaseGet(env, 'ledger',
        `?user_id=eq.${env.USER_ID}&order=date.asc&select=date,signal_tier,actual_amount,vdgr_price,units_acquired`
      ),
      supabaseGet(env, 'performance_snapshots',
        `?user_id=eq.${env.USER_ID}&order=date.asc&select=date,vdgr_price,total_units,total_invested,current_value,return_dollar,return_pct`
      ),
      supabaseGet(env, 'signals',
        `?user_id=eq.${env.USER_ID}&order=date.desc&limit=1&select=vdgr_price,date`
      ),
    ]);

    if (ledgerRows.length === 0) {
      return json({
        summary: null,
        snapshots: [],
        ledger: [],
        message: 'No ledger entries yet. Performance tracking begins after your first investment.',
      });
    }

    const currentPrice = latestSignal[0]?.vdgr_price || null;
    const currentDate = latestSignal[0]?.date || null;

    // Live summary computed from ledger + current price
    const totalInvested = ledgerRows.reduce((s, e) => s + parseFloat(e.actual_amount), 0);
    const totalUnits = ledgerRows.reduce((s, e) => s + parseFloat(e.units_acquired), 0);
    const currentValue = currentPrice ? totalUnits * currentPrice : null;
    const returnDollar = currentValue !== null ? currentValue - totalInvested : null;
    const returnPct = currentValue !== null && totalInvested > 0
      ? ((currentValue - totalInvested) / totalInvested) * 100
      : null;

    // Forward return summary by tier (from historical signals)
    const allSignals = await supabaseGet(env, 'signals',
      `?user_id=eq.${env.USER_ID}&signal_tier=neq.NONE&order=date.asc&select=date,signal_tier,vdgr_price`
    );

    const fwdSummary = buildForwardReturnSummary(allSignals, currentPrice);

    return json({
      summary: {
        totalInvested: Math.round(totalInvested * 100) / 100,
        totalUnits: Math.round(totalUnits * 1000000) / 1000000,
        currentValue: currentValue !== null ? Math.round(currentValue * 100) / 100 : null,
        returnDollar: returnDollar !== null ? Math.round(returnDollar * 100) / 100 : null,
        returnPct: returnPct !== null ? Math.round(returnPct * 100) / 100 : null,
        currentPrice,
        currentDate,
        entryCount: ledgerRows.length,
      },
      snapshots,
      ledger: ledgerRows,
      forwardReturns: fwdSummary,
    });
  }

  return json({ error: 'Not found' }, 404);
}

function buildForwardReturnSummary(signals, currentPrice) {
  if (!signals.length || !currentPrice) return [];

  const tiers = ['MEDIUM', 'HIGH', 'EXTREME'];
  return tiers.map(tier => {
    const subset = signals.filter(s => s.signal_tier === tier);
    if (!subset.length) return null;

    // Estimate forward returns using current price (simplified — no future data stored)
    const returns = subset.map(s => {
      const buyPrice = parseFloat(s.vdgr_price);
      return buyPrice > 0 ? ((currentPrice - buyPrice) / buyPrice) * 100 : null;
    }).filter(r => r !== null);

    if (!returns.length) return null;

    const avg = returns.reduce((a, b) => a + b, 0) / returns.length;
    const winRate = (returns.filter(r => r > 0).length / returns.length) * 100;

    return {
      signal_tier: tier,
      count: subset.length,
      avg_return_pct: Math.round(avg * 100) / 100,
      win_rate_pct: Math.round(winRate * 100) / 100,
    };
  }).filter(Boolean);
}
