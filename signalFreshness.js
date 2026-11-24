// signalFreshness.js — Is the stored signal from the last trading day?
// Uses the Sydney calendar date, so the answer is the same in every browser timezone.
// Does not know ASX holidays.

const DAY_MS = 24 * 60 * 60 * 1000;

// Sydney calendar date (YYYY-MM-DD) for a moment in time
function getSydneyDate(now) {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'Australia/Sydney',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(now);
  const get = type => parts.find(p => p.type === type).value;
  return `${get('year')}-${get('month')}-${get('day')}`;
}

// Fresh = from today's cron run = previous weekday's close
export function isSignalFreshToday(signalDate, now = new Date()) {
  if (!signalDate) return false;

  // From here on the date is UTC midnight and only read with UTC methods
  const today = new Date(`${getSydneyDate(now)}T00:00:00Z`);
  const todayDay = today.getUTCDay(); // 0=Sun, 6=Sat

  // Weekend: market closed, not fresh for buying
  if (todayDay === 0 || todayDay === 6) return false;

  // Monday: last trading day was Friday. Tue-Fri: yesterday.
  const daysBack = todayDay === 1 ? 3 : 1;
  const lastTradingDay = new Date(today.getTime() - daysBack * DAY_MS).toISOString().slice(0, 10);

  return String(signalDate).slice(0, 10) === lastTradingDay;
}
