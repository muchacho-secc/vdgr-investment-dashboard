// routes/analystRoutes.js

import { handleAnalystChat, handleAnalystSummary } from '../analyst.js';

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

// POST /scan — extract transaction details from a Vanguard screenshot
async function handleScan(request, env) {
  const body = await request.json().catch(() => null);
  if (!body?.image || !body?.mediaType) {
    return json({ error: 'image and mediaType are required' }, 400);
  }

  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': env.ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: 'claude-sonnet-4-5',
      max_tokens: 400,
      system: `You extract investment transaction details from broker or investment app screenshots.
Respond with ONLY a valid JSON object, no other text, no markdown fences:
{
  "date": "YYYY-MM-DD",
  "vdgr_price": <number - unit price per ETF unit>,
  "actual_amount": <number - total dollar amount of the transaction>,
  "units": <number - number of units purchased>,
  "notes": "<string - fund name or any useful note, empty string if nothing useful>"
}
Rules:
- actual_amount is the LARGE dollar figure shown prominently at the top centre of the screen (e.g. $391.14) — this is the total dollars spent
- vdgr_price is the "Unit price" field (e.g. $65.19) — the price per single unit
- units is the "Order amount" field shown as a number of units (e.g. 6 units)
- Australian dates like "Fri 27 Mar, 2026" or DD/MM/YYYY — convert to YYYY-MM-DD
- If a field cannot be found with confidence, use null
- The ETF may be labelled VDGR, VanEck Gold Royalties, Vanguard Diversified Growth, or similar
- Do NOT calculate actual_amount from units × price — read it directly from the screen`,
      messages: [{
        role: 'user',
        content: [
          {
            type: 'image',
            source: {
              type: 'base64',
              media_type: body.mediaType,
              data: body.image,
            },
          },
          { type: 'text', text: 'Extract the transaction details from this screenshot.' },
        ],
      }],
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    return json({ error: `Anthropic API error: ${err}` }, 500);
  }

  const data = await res.json();
  const text = data.content?.[0]?.text || '';

  try {
    const clean = text.replace(/```json|```/g, '').trim();
    const parsed = JSON.parse(clean);
    return json({ extracted: parsed });
  } catch {
    return json({ error: 'Could not parse extracted data', raw: text }, 422);
  }
}

export async function handleAnalystRoutes(request, env, url) {
  // POST /analyst/chat — multi-turn chat from the Today tab
  if (url.pathname === '/analyst/chat' && request.method === 'POST') {
    return handleAnalystChat(request, env);
  }

  // POST /analyst/summary — on-demand regeneration of today's summary
  if (url.pathname === '/analyst/summary' && request.method === 'POST') {
    return handleAnalystSummary(request, env);
  }

  // POST /scan — extract transaction from screenshot
  if (url.pathname === '/scan' && request.method === 'POST') {
    return handleScan(request, env);
  }

  return json({ error: 'Not found' }, 404);
}
