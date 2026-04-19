// index.js — Cloudflare Worker entry point

import { handleSignalRoutes }      from './routes/signalRoutes.js';
import { handleLedgerRoutes }      from './routes/ledgerRoutes.js';
import { handlePerformanceRoutes } from './routes/performanceRoutes.js';
import { handleAnalystRoutes }     from './routes/analystRoutes.js';
import { handleSettingsRoutes }    from './routes/settingsRoutes.js';
import { runDailySignal }          from './signal.js';
import { sendTelegramError }       from './telegram.js';

export default {
  // ── Cron trigger — runs daily at 8am AEDT (see wrangler.toml) ─────────────
  async scheduled(event, env, ctx) {
    ctx.waitUntil(
      runDailySignal(env).catch(err => sendTelegramError(env, err.message))
    );
  },

  // ── HTTP handler — serves all frontend API routes ──────────────────────────
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    const corsHeaders = {
      'Access-Control-Allow-Origin':  env.FRONTEND_URL || '*',
      'Access-Control-Allow-Methods': 'GET, POST, PATCH, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    // Preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders });
    }

    let response;

    try {
      if (url.pathname.startsWith('/signal')) {
        response = await handleSignalRoutes(request, env, url);

      } else if (url.pathname.startsWith('/ledger')) {
        response = await handleLedgerRoutes(request, env, url);

      } else if (url.pathname.startsWith('/performance')) {
        response = await handlePerformanceRoutes(request, env, url);

      } else if (url.pathname.startsWith('/settings') || url.pathname.startsWith('/backtest')) {
        response = await handleSettingsRoutes(request, env, url);

      } else if (url.pathname.startsWith('/analyst')) {
        response = await handleAnalystRoutes(request, env, url);

      // ── Dev-only routes (disabled in production) ──────────────────────────
      } else if (url.pathname === '/trigger-signal' && env.ENVIRONMENT !== 'production') {
        ctx.waitUntil(runDailySignal(env));
        response = new Response(JSON.stringify({ message: 'Signal generation triggered' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });

      } else if (url.pathname === '/test-telegram' && env.ENVIRONMENT !== 'production') {
        const { sendTelegramNotification } = await import('./telegram.js');
        await sendTelegramNotification(env, {
          tier: 'HIGH',
          recommendedAmount: 400,
          analystSummary: 'This is a test notification from the VDGR signal system.',
          rsi: 33.5,
          vix: 26.2,
          currentPrice: 17.45,
        });
        response = new Response(JSON.stringify({ message: 'Test notification sent' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });

      } else {
        response = new Response(
          JSON.stringify({ service: 'VDGR Signal Worker', status: 'ok' }),
          { status: 200, headers: { 'Content-Type': 'application/json' } }
        );
      }

    } catch (err) {
      console.error('[worker] Unhandled error:', err);
      response = new Response(
        JSON.stringify({ error: 'Internal server error' }),
        { status: 500, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Attach CORS headers to every response
    Object.entries(corsHeaders).forEach(([k, v]) => response.headers.set(k, v));
    return response;
  },
};
