// analyst.js — Anthropic API proxy. API key never leaves the Worker.

import { supabaseGet } from './supabase.js';

const SYSTEM_PROMPT = `You are a calm, clear investment analyst helping a single user invest consistently into VDGR.AX (VanEck Gold Royalties ETF Australia).

Your role:
- Explain today's investment signal in plain, jargon-free English
- Help the user understand WHY the signal is what it is
- Build confidence in a rules-based, disciplined investment approach
- Answer questions about the current market conditions

Your signal system uses two indicators:
- RSI (Relative Strength Index): measures recent price momentum. Lower RSI = more selling pressure recently.
  - < 50 = WATCH, < 45 = MEDIUM, < 35 = HIGH, < 30 = EXTREME
- VIX (Volatility Index): measures US market fear. Higher VIX = more uncertainty.
  - > 18 = WATCH, > 20 = MEDIUM, > 25 = HIGH, > 30 = EXTREME
- Both conditions must be met for a signal to trigger.

Signal tiers and recommended investment amounts:
- WATCH:   RSI < 50 AND VIX > 18 → monitor, no additional investment
- MEDIUM:  RSI < 45 AND VIX > 20 → invest $200
- HIGH:    RSI < 35 AND VIX > 25 → invest $400
- EXTREME: RSI < 30 AND VIX > 30 → invest $800 (may suggest more)

Rules:
- Never predict future prices or market direction
- Never recommend specific amounts beyond the preset tiers (except EXTREME scaling)
- Never use trading jargon the user hasn't introduced
- Keep responses concise — 2-4 sentences for explanations, longer only if asked
- Tone: calm, informative, confident. Not alarmist, not speculative.
- If you don't have enough data to answer, say so clearly
- Always refer to the investment as "additional investment into VDGR" — not "buying" or "trading"
- This is a disciplined, rules-based system — reinforce that consistently
- Use Australian English`;

function buildContext(signal, recentSignals, ledgerSummary) {
  const recent = recentSignals.length > 0
    ? recentSignals.map(s => `  ${s.date}: ${s.signal_tier} (RSI: ${s.rsi}, VIX: ${s.vix})`).join('\n')
    : '  No recent signal history available';

  return `--- TODAY'S MARKET CONTEXT ---
Date: ${signal.date}
Signal Tier: ${signal.signal_tier}
VDGR.AX Price: $${signal.vdgr_price}
RSI (14-day): ${signal.rsi}
VIX: ${signal.vix}
52-Week Drawdown: ${signal.drawdown_pct}%
Recommended Amount: $${signal.recommended_amount}

--- RECENT SIGNAL HISTORY (last 7 days) ---
${recent}

--- INVESTMENT SUMMARY ---
Total invested via signals: $${ledgerSummary.totalInvested.toFixed(2)}
Number of signal investments: ${ledgerSummary.entryCount}
Most recent investment: ${ledgerSummary.lastEntry || 'None yet'}`;
}

async function callAnthropic(env, systemPrompt, messages, maxTokens = 400) {
  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': env.ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: 'claude-sonnet-4-5',
      max_tokens: maxTokens,
      system: systemPrompt,
      messages,
    }),
  });

  if (!res.ok) throw new Error(`Anthropic API error: ${res.status} ${await res.text()}`);
  const data = await res.json();
  return data.content[0].text;
}

// Called during the daily cron to pre-generate and store the analyst summary
export async function generateAnalystSummary(env, signal) {
  const extremeAddendum = signal.signal_tier === 'EXTREME'
    ? `\n\nThe signal today is EXTREME — the most severe tier. The base recommendation is $800. If RSI is significantly below 30 or VIX is significantly above 30, you may suggest the user consider investing more than $800, framed as an option — not a directive. Mention the specific values that support this. Keep the suggestion brief and calm.`
    : '';

  const prompt = `${buildContext(signal, [], { totalInvested: 0, entryCount: 0, lastEntry: null })}

Please provide a brief explanation of today's signal for the user.
Cover: what the RSI and VIX values mean today, why this triggered (or didn't trigger) a signal, and what this suggests about current market conditions.
Keep it to 3 sentences. Do not suggest action — just explain the conditions.${extremeAddendum}`;

  return callAnthropic(env, SYSTEM_PROMPT, [{ role: 'user', content: prompt }], 300);
}

// Called from POST /analyst/summary — on-demand regeneration
export async function handleAnalystSummary(request, env) {
  const { date } = await request.json().catch(() => ({}));
  const targetDate = date || new Date().toISOString().split('T')[0];

  const [signals, recentRows, ledgerRows] = await Promise.all([
    supabaseGet(env, 'signals', `?date=eq.${targetDate}&user_id=eq.${env.USER_ID}&select=*`),
    supabaseGet(env, 'signals', `?user_id=eq.${env.USER_ID}&order=date.desc&limit=7&select=date,signal_tier,rsi,vix`),
    supabaseGet(env, 'ledger', `?user_id=eq.${env.USER_ID}&order=date.desc&select=actual_amount,date`),
  ]);

  if (!signals.length) {
    return json({ error: 'No signal found for this date' }, 404);
  }

  const signal = signals[0];
  const totalInvested = ledgerRows.reduce((s, e) => s + parseFloat(e.actual_amount), 0);
  const lastEntry = ledgerRows[0]?.date || null;

  const summary = await generateAnalystSummary(env, {
    ...signal,
    date: targetDate,
  });

  // Persist updated summary
  await supabasePatch(env, 'signals', `?date=eq.${targetDate}`, { analyst_summary: summary });

  return json({ summary });
}

// Called from POST /analyst/chat — multi-turn chat from Today tab
export async function handleAnalystChat(request, env) {
  const body = await request.json().catch(() => null);
  if (!body?.messages?.length) {
    return json({ error: 'messages array is required' }, 400);
  }

  const { messages, date } = body;
  const targetDate = date || new Date().toISOString().split('T')[0];

  const [signals, recentRows, ledgerRows] = await Promise.all([
    supabaseGet(env, 'signals', `?date=eq.${targetDate}&user_id=eq.${env.USER_ID}&select=*`),
    supabaseGet(env, 'signals', `?user_id=eq.${env.USER_ID}&order=date.desc&limit=7&select=date,signal_tier,rsi,vix`),
    supabaseGet(env, 'ledger', `?user_id=eq.${env.USER_ID}&order=date.desc&select=actual_amount,date`),
  ]);

  const signal = signals[0] || {
    date: targetDate,
    signal_tier: 'NONE',
    vdgr_price: 0,
    rsi: 0,
    vix: 0,
    drawdown_pct: 0,
    recommended_amount: 0,
  };

  const totalInvested = ledgerRows.reduce((s, e) => s + parseFloat(e.actual_amount), 0);
  const ledgerSummary = {
    totalInvested,
    entryCount: ledgerRows.length,
    lastEntry: ledgerRows[0]?.date || null,
  };

  const contextBlock = buildContext(signal, recentRows, ledgerSummary);

  // Inject context only into the first user message
  const enrichedMessages = messages.map((msg, i) => {
    if (i === 0 && msg.role === 'user') {
      return { ...msg, content: `${contextBlock}\n\n--- USER QUESTION ---\n${msg.content}` };
    }
    return msg;
  });

  const response = await callAnthropic(env, SYSTEM_PROMPT, enrichedMessages, 500);
  return json({ response });
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}
