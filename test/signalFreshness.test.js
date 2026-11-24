// Test: isSignalFreshToday gives the same answer in every browser timezone (VDGR-6)
// Offline. Run by name: node --test test/signalFreshness.test.js
// (a bare `node --test` would also run signalStability.test.js, which calls Yahoo)
//
// Each timezone runs in its own child process, because TZ must be set before Node starts.

import { test, mock } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const MODULE_URL = new URL('../signalFreshness.js', import.meta.url);
const APP_URL = new URL('../App.jsx', import.meta.url);

// `now` carries the Sydney offset (+11:00 AEDT, +10:00 AEST).
const CASES = [
  { name: 'Mon 23 Nov 09:30, Friday signal',      now: '2026-11-23T09:30:00+11:00', date: '2026-11-20', fresh: true },
  { name: 'Mon 23 Nov 09:30, Thursday signal',    now: '2026-11-23T09:30:00+11:00', date: '2026-11-19', fresh: false },
  { name: 'Mon 23 Nov 12:00, Friday signal',      now: '2026-11-23T12:00:00+11:00', date: '2026-11-20', fresh: true },
  { name: 'Tue 24 Nov 09:30, Monday signal',      now: '2026-11-24T09:30:00+11:00', date: '2026-11-23', fresh: true },
  { name: 'Tue 24 Nov 09:30, Sunday signal',      now: '2026-11-24T09:30:00+11:00', date: '2026-11-22', fresh: false },
  { name: 'Tue 14 Jul 09:30 AEST, Monday signal', now: '2026-07-14T09:30:00+10:00', date: '2026-07-13', fresh: true },
  { name: 'Tue 24 Nov 20:00, Monday signal',      now: '2026-11-24T20:00:00+11:00', date: '2026-11-23', fresh: true },
  { name: 'Mon 23 Nov 09:30, Sunday signal',      now: '2026-11-23T09:30:00+11:00', date: '2026-11-22', fresh: false },
  { name: 'Tue 24 Nov 09:30, Friday signal',      now: '2026-11-24T09:30:00+11:00', date: '2026-11-20', fresh: false },
  { name: 'Sat 21 Nov 09:30, Friday signal',      now: '2026-11-21T09:30:00+11:00', date: '2026-11-20', fresh: false },
  { name: 'Mon 23 Nov 09:30, null signal date',   now: '2026-11-23T09:30:00+11:00', date: null,         fresh: false },
];

function readApp() {
  return readFileSync(APP_URL, 'utf8').replace(/\r\n/g, '\n');
}

// Before the fix the function is nested in App.jsx, so take its text and evaluate it.
function extractFromApp() {
  const source = readApp();
  const start = source.indexOf('function isSignalFreshToday(');
  if (start === -1) throw new Error('isSignalFreshToday not found in App.jsx');
  let depth = 0;
  for (let i = source.indexOf('{', start); i < source.length; i++) {
    if (source[i] === '{') depth++;
    if (source[i] === '}' && --depth === 0) {
      return new Function(`${source.slice(start, i + 1)}\nreturn isSignalFreshToday;`)();
    }
  }
  throw new Error('isSignalFreshToday in App.jsx has no closing brace');
}

async function runChild() {
  const isModule = existsSync(MODULE_URL);
  const fn = isModule ? (await import(MODULE_URL.href)).isSignalFreshToday : extractFromApp();

  const results = CASES.map(({ now, date }) => {
    mock.timers.enable({ apis: ['Date'], now: Date.parse(now) });
    const byClock = fn(date);
    mock.timers.reset();
    // The module also takes `now`. The real clock is in force here.
    const byArgument = isModule ? fn(date, new Date(now)) : undefined;
    return { byClock, byArgument };
  });

  process.stdout.write(JSON.stringify({
    timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    source: isModule ? 'signalFreshness.js' : 'App.jsx',
    results,
  }));
}

function runInTimezone(timeZone) {
  const env = { ...process.env, TZ: timeZone, FRESHNESS_CHILD: '1' };
  delete env.NODE_TEST_CONTEXT;
  const child = spawnSync(process.execPath, [fileURLToPath(import.meta.url)], { env, encoding: 'utf8', timeout: 30000 });
  assert.equal(child.status, 0, `child process failed: ${child.stderr || child.error}`);
  return JSON.parse(child.stdout);
}

if (process.env.FRESHNESS_CHILD) {
  await runChild();
} else {
  for (const timeZone of ['UTC', 'Australia/Sydney', 'America/New_York']) {
    test(`isSignalFreshToday in ${timeZone}`, async (t) => {
      const out = runInTimezone(timeZone);
      assert.equal(out.timeZone, timeZone, 'child process did not use the requested timezone');
      assert.equal(out.results.length, CASES.length);
      t.diagnostic(`tested the function in ${out.source}`);

      for (const [i, c] of CASES.entries()) {
        await t.test(c.name, () => {
          const { byClock, byArgument } = out.results[i];
          assert.equal(byClock, c.fresh, `expected fresh=${c.fresh}, got ${byClock}`);
          if (out.source !== 'App.jsx') {
            assert.equal(byArgument, c.fresh, `with an explicit now: expected fresh=${c.fresh}, got ${byArgument}`);
          }
        });
      }
    });
  }

  test('App.jsx uses the shared module', { skip: !existsSync(MODULE_URL) && 'signalFreshness.js does not exist yet' }, () => {
    const source = readApp();
    assert.match(source, /^import\s*\{[^}]*\bisSignalFreshToday\b[^}]*\}\s*from\s*["']\.\/signalFreshness(\.js)?["']/m);
    assert.doesNotMatch(source, /function\s+isSignalFreshToday\b|(const|let|var)\s+isSignalFreshToday\b/);
  });
}
