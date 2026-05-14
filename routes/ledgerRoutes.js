// routes/ledgerRoutes.js

import { supabaseGet, supabasePost, supabasePatch } from '../supabase.js';

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

export async function handleLedgerRoutes(request, env, url) {
  // GET /ledger — return all entries, newest first
  if (url.pathname === '/ledger' && request.method === 'GET') {
    const entries = await supabaseGet(env, 'ledger',
      `?user_id=eq.${env.USER_ID}&order=date.desc&select=*`
    );
    return json({ entries });
  }

  // POST /ledger — create new entry
  if (url.pathname === '/ledger' && request.method === 'POST') {
    const body = await request.json().catch(() => null);
    if (!body) return json({ error: 'Invalid request body' }, 400);

    const { date, signal_id, signal_tier, recommended_amount, actual_amount, vdgr_price, notes } = body;

    // Validate required fields
    if (!date || !signal_tier || !actual_amount || !vdgr_price) {
      return json({ error: 'date, signal_tier, actual_amount, and vdgr_price are required' }, 400);
    }

    const units_acquired = parseFloat(actual_amount) / parseFloat(vdgr_price);

    const [entry] = await supabasePost(env, 'ledger', {
      user_id: env.USER_ID,
      date,
      signal_id: signal_id || null,
      signal_tier,
      recommended_amount: recommended_amount || null,
      actual_amount: Math.round(parseFloat(actual_amount) * 100) / 100,
      vdgr_price: Math.round(parseFloat(vdgr_price) * 10000) / 10000,
      units_acquired: Math.round(units_acquired * 1000000) / 1000000,
      notes: notes || null,
    });

    return json({ entry }, 201);
  }

  // PATCH /ledger/:id — update existing entry (user override)
  const patchMatch = url.pathname.match(/^\/ledger\/([a-f0-9-]+)$/);
  if (patchMatch && request.method === 'PATCH') {
    const id = patchMatch[1];
    const body = await request.json().catch(() => null);
    if (!body) return json({ error: 'Invalid request body' }, 400);

    const allowed = ['actual_amount', 'vdgr_price', 'notes'];
    const updates = Object.fromEntries(
      Object.entries(body).filter(([k]) => allowed.includes(k))
    );

    // Recalculate units if amount or price changed
    if (updates.actual_amount || updates.vdgr_price) {
      // Fetch existing row to fill gaps
      const existing = await supabaseGet(env, 'ledger',
        `?id=eq.${id}&user_id=eq.${env.USER_ID}&select=actual_amount,vdgr_price`
      );
      if (!existing.length) return json({ error: 'Ledger entry not found' }, 404);

      const amount = parseFloat(updates.actual_amount || existing[0].actual_amount);
      const price = parseFloat(updates.vdgr_price || existing[0].vdgr_price);
      updates.units_acquired = Math.round((amount / price) * 1000000) / 1000000;
    }

    const [updated] = await supabasePatch(env, 'ledger',
      `?id=eq.${id}&user_id=eq.${env.USER_ID}`,
      updates
    );

    if (!updated) return json({ error: 'Ledger entry not found' }, 404);
    return json({ entry: updated });
  }

  return json({ error: 'Not found' }, 404);
}
