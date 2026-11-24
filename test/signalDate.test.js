// Test: the daily run stores the market date of the data, not the run date (VDGR-5 guard)
// Offline: every fetch is mocked and an unexpected URL fails the test.
// Run by name: node --test test/signalDate.test.js
//
// Each timezone runs in its own child process, because TZ must be set before Node starts.

import { test, mock } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const DAY_MS = 24 * 60 * 60 * 1000;
const BAR_COUNT = 100;
const SUPABASE_URL = 'https://supabase.invalid';

const ENV = {
  SUPABASE_URL,
  SUPABASE_ANON_KEY: 'dummy',
  SUPABASE_SERVICE_KEY: 'dummy',
  ANTHROPIC_API_KEY: 'dummy',
  TELEGRAM_BOT_TOKEN: 'dummy',
  TELEGRAM_CHAT_ID: 'dummy',
  USER_ID: 'test-user',
};

// `lastBar` is the newest bar Yahoo returns. `stored` is the date the run must store (null: no write).
const SCENARIOS = [
  { name: 'Sunday 22:00 UTC (Monday 09:00 AEDT) stores Friday',  now: '2026-11-22T22:00:00Z', lastBar: '2026-11-20', stored: '2026-11-20' },
  { name: 'Monday 22:00 UTC (Tuesday 09:00 AEDT) stores Monday', now: '2026-11-23T22:00:00Z', lastBar: '2026-11-23', stored: '2026-11-23' },
  { name: 'Sunday 22:00 UTC (Monday 08:00 AEST) stores Friday',  now: '2026-07-12T22:00:00Z', lastBar: '2026-07-10', stored: '2026-07-10' },
  { name: 'in-progress Sydney bar (Tuesday 12:00 AEDT) is dropped', now: '2026-11-24T01:00:00Z', lastBar: '2026-11-24', stored: '2026-11-23' },
  // Monday 26 January 2026 is an ASX holiday, so Tuesday's run sees Friday's bar again.
  { name: 'holiday run whose row exists writes nothing', now: '2026-01-26T22:00:00Z', lastBar: '2026-01-23', existing: '2026-01-23', stored: null },
];

// Seconds since the epoch for a wall-clock time in a timezone.
function zonedTimestamp(date, hour, minute, timeZone) {
  const wall = Date.parse(`${date}T00:00:00Z`) + (hour * 60 + minute) * 60000;
  const offsetAt = (ms) => {
    const parts = Object.fromEntries(new Intl.DateTimeFormat('en-US', {
      timeZone, hourCycle: 'h23', year: 'numeric', month: 'numeric', day: 'numeric',
      hour: 'numeric', minute: 'numeric',
    }).formatToParts(new Date(ms)).map(p => [p.type, Number(p.value)]));
    return Date.UTC(parts.year, parts.month - 1, parts.day, parts.hour, parts.minute) - ms;
  };
  const guess = wall - offsetAt(wall);
  return (wall - offsetAt(guess)) / 1000;
}

// The last BAR_COUNT weekdays up to and including lastBar.
function weekdaysUpTo(lastBar) {
  const dates = [];
  for (let ms = Date.parse(`${lastBar}T00:00:00Z`); dates.length < BAR_COUNT; ms -= DAY_MS) {
    const d = new Date(ms);
    if (d.getUTCDay() !== 0 && d.getUTCDay() !== 6) dates.unshift(d.toISOString().slice(0, 10));
  }
  return dates;
}

// Every bar has its own close, so the stored price shows which bar was used.
function closeAt(index) {
  return 50 + index * 0.1 + (index % 2 ? 0.3 : 0);
}

function chart(dates, hour, minute, timeZone, close) {
  return { chart: { result: [{
    timestamp: dates.map(d => zonedTimestamp(d, hour, minute, timeZone)),
    indicators: { quote: [{ close: dates.map((_, i) => close(i)) }] },
  }] } };
}

function makeFetch(scenario, log) {
  const dates = weekdaysUpTo(scenario.lastBar);
  const reply = (body, status = 200) =>
    new Response(body === null ? null : JSON.stringify(body), { status });

  return async (input, options = {}) => {
    const url = new URL(String(input));
    const method = options.method || 'GET';

    if (method === 'GET' && url.hostname === 'query1.finance.yahoo.com') {
      if (url.pathname === '/v8/finance/chart/VDGR.AX') return reply(chart(dates, 10, 0, 'Australia/Sydney', closeAt));
      // VIX 15 is below every threshold, so the tier is NONE and no Telegram message is due.
      if (url.pathname === '/v8/finance/chart/%5EVIX') return reply(chart(dates, 9, 30, 'America/New_York', () => 15));
    }
    if (url.origin === SUPABASE_URL) {
      const table = url.pathname.replace('/rest/v1/', '');
      if (method === 'GET' && table === 'user_settings') return reply([]);
      if (method === 'GET' && table === 'ledger') return reply([{ actual_amount: '100', units_acquired: '2' }]);
      if (method === 'GET' && table === 'signals') {
        const date = (url.searchParams.get('date') || '').replace('eq.', '');
        log.checked.push(date);
        return reply(date === scenario.existing ? [{ id: 1 }] : []);
      }
      if (method === 'POST' && (table === 'signals' || table === 'performance_snapshots')) {
        log[table].push(JSON.parse(options.body));
        return reply(null, 201);
      }
    }
    if (method === 'POST' && url.href === 'https://api.anthropic.com/v1/messages') {
      log.anthropic++;
      return reply({ content: [{ text: 'Test summary.' }], usage: {} });
    }

    // runDailySignal catches some errors, so the parent checks this list as well.
    log.unexpected.push(`${method} ${url.href}`);
    throw new Error(`blocked unexpected request: ${method} ${url.href}`);
  };
}

async function runChild() {
  console.log = console.warn = console.error = (...args) => process.stderr.write(`${args.join(' ')}\n`);
  globalThis.fetch = async () => { throw new Error('blocked request outside a scenario'); };
  const { runDailySignal } = await import('../signal.js');

  const results = [];
  for (const scenario of SCENARIOS) {
    const log = { checked: [], signals: [], performance_snapshots: [], anthropic: 0, unexpected: [], error: null };
    globalThis.fetch = makeFetch(scenario, log);
    mock.timers.enable({ apis: ['Date'], now: Date.parse(scenario.now) });
    try {
      await runDailySignal(ENV);
    } catch (err) {
      log.error = err.stack;
    }
    mock.timers.reset();
    results.push(log);
  }

  process.stdout.write(JSON.stringify({
    timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    results,
  }));
}

function runInTimezone(timeZone) {
  const env = { ...process.env, TZ: timeZone, SIGNAL_DATE_CHILD: '1' };
  delete env.NODE_TEST_CONTEXT;
  const child = spawnSync(process.execPath, [fileURLToPath(import.meta.url)], { env, encoding: 'utf8', timeout: 60000 });
  assert.equal(child.status, 0, `child process failed: ${child.stderr || child.error}`);
  return JSON.parse(child.stdout);
}

if (process.env.SIGNAL_DATE_CHILD) {
  await runChild();
} else {
  for (const timeZone of ['UTC', 'Australia/Sydney']) {
    test(`stored signal date in ${timeZone}`, async (t) => {
      const out = runInTimezone(timeZone);
      assert.equal(out.timeZone, timeZone, 'child process did not use the requested timezone');
      assert.equal(out.results.length, SCENARIOS.length);

      for (const [i, s] of SCENARIOS.entries()) {
        await t.test(s.name, () => {
          const log = out.results[i];
          assert.equal(log.error, null);
          assert.deepEqual(log.unexpected, [], 'unexpected request');

          const signalDates = log.signals.map(r => r.date);
          const snapshotDates = log.performance_snapshots.map(r => r.date);

          if (s.stored === null) {
            assert.deepEqual(signalDates, []);
            assert.deepEqual(snapshotDates, []);
            assert.deepEqual(log.checked, [s.existing]);
            assert.equal(log.anthropic, 0);
            return;
          }

          assert.deepEqual(
            { signals: signalDates, performance_snapshots: snapshotDates },
            { signals: [s.stored], performance_snapshots: [s.stored] },
          );
          const price = Math.round(closeAt(weekdaysUpTo(s.lastBar).indexOf(s.stored)) * 10000) / 10000;
          assert.equal(log.signals[0].vdgr_price, price);
          assert.equal(log.signals[0].signal_tier, 'NONE');
        });
      }
    });
  }
}
