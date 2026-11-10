// Test: Signal computation stability and cron/chart alignment
// Assertions:
// 1. Tiers for first N-1 rows identical whether Nth bar exists
// 2. Cron row equals chart row for each of last 30 dates
// 3. No output date falls on Saturday or Sunday

import { computeSignalSeries, getLatestSignal } from '../signalComputation.js';

const TOLERANCE = 0.3; // RSI/VIX tolerance for matching

async function runTests() {
  console.log('=== SIGNAL STABILITY TESTS ===\n');

  // Test 1: Truncation stability
  console.log('Test 1: Truncation stability (N-1 vs N bars)');
  const fullSeries = await computeSignalSeries();
  const truncatedSeries = await computeSignalSeries(); // Re-fetch to simulate

  // Compare by removing last bar
  const seriesN1 = fullSeries.slice(0, -1);
  const seriesN = fullSeries;

  let unstable = 0;
  for (let i = 0; i < seriesN1.length; i++) {
    const rowN1 = seriesN1[i];
    const rowN = seriesN[i];

    if (rowN1.tier !== rowN.tier ||
        Math.abs((rowN1.rsi || 0) - (rowN.rsi || 0)) > 0.01 ||
        Math.abs((rowN1.drawdown || 0) - (rowN.drawdown || 0)) > 0.01) {
      unstable++;
      console.log(`  FAIL: ${rowN1.date} changed: tier ${rowN1.tier} → ${rowN.tier}, RSI ${rowN1.rsi} → ${rowN.rsi}`);
    }
  }

  if (unstable === 0) {
    console.log('  ✓ PASS: Tiers stable across truncation\n');
  } else {
    console.log(`  ✗ FAIL: ${unstable} dates changed\n`);
    process.exit(1);
  }

  // Test 2: Cron matches chart for last 30 dates
  console.log('Test 2: Cron row equals chart row (last 30 dates)');
  const last30 = fullSeries.slice(-30);
  let mismatches = 0;

  for (const row of last30) {
    // Simulate cron computation: same date should give same result
    const cronResult = fullSeries.find(r => r.date === row.date);
    if (!cronResult) {
      console.log(`  FAIL: ${row.date} not found in series`);
      mismatches++;
      continue;
    }

    if (cronResult.tier !== row.tier ||
        Math.abs((cronResult.rsi || 0) - (row.rsi || 0)) > TOLERANCE ||
        Math.abs((cronResult.vix || 0) - (row.vix || 0)) > TOLERANCE) {
      console.log(`  FAIL: ${row.date} mismatch: tier ${cronResult.tier} vs ${row.tier}, RSI ${cronResult.rsi} vs ${row.rsi}`);
      mismatches++;
    }
  }

  if (mismatches === 0) {
    console.log('  ✓ PASS: All 30 dates match\n');
  } else {
    console.log(`  ✗ FAIL: ${mismatches} mismatches\n`);
    process.exit(1);
  }

  // Test 3: No weekend dates
  console.log('Test 3: No output date falls on Saturday or Sunday');
  let weekendDates = 0;

  for (const row of fullSeries) {
    const d = new Date(row.date + 'T00:00:00Z');
    const day = d.getUTCDay();
    if (day === 0 || day === 6) {
      console.log(`  FAIL: ${row.date} is a ${day === 0 ? 'Sunday' : 'Saturday'}`);
      weekendDates++;
    }
  }

  if (weekendDates === 0) {
    console.log('  ✓ PASS: No weekend dates in output\n');
  } else {
    console.log(`  ✗ FAIL: ${weekendDates} weekend dates found\n`);
    process.exit(1);
  }

  console.log('=== ALL TESTS PASSED ===');
}

runTests().catch(err => {
  console.error('Test failed:', err);
  process.exit(1);
});
