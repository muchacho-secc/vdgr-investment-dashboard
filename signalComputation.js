// signalComputation.js — Shared signal computation logic
// Used by cron (signal.js), chart endpoint (/signal/chart), and other routes

// ─── Yahoo Finance ────────────────────────────────────────────────────────────
async function fetchYahooSeries(ticker, range = '2y') {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?range=${range}&interval=1d`;
  const res = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
  if (!res.ok) throw new Error(`Yahoo Finance ${ticker} fetch failed: ${res.status}`);
  const data = await res.json();
  const result = data.chart?.result?.[0];
  if (!result) throw new Error(`No data returned for ${ticker}`);
  return result;
}

// ─── Exchange Timezone Conversion ─────────────────────────────────────────────
function getExchangeDate(timestamp, timezone) {
  const date = new Date(timestamp * 1000);
  const formatter = new Intl.DateTimeFormat('en-CA', {
    timeZone: timezone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit'
  });
  return formatter.format(date); // Returns YYYY-MM-DD
}

function isMarketOpen(timestamp, timezone) {
  // Check if this bar's date equals today in the exchange timezone
  // and if the current time is before market close (~16:30 local)
  const now = new Date();
  const barDate = getExchangeDate(timestamp, timezone);
  const todayDate = getExchangeDate(Math.floor(now.getTime() / 1000), timezone);

  if (barDate !== todayDate) return false; // Not today's bar

  // Get current hour in exchange timezone
  const formatter = new Intl.DateTimeFormat('en-US', {
    timeZone: timezone,
    hour: 'numeric',
    hour12: false
  });
  const currentHour = parseInt(formatter.format(now));

  // Assume market closes at 16:30 local time
  return currentHour < 17; // Still open if before 5pm
}

// ─── RSI Series (14-day Wilder's smoothing) ───────────────────────────────────
function computeRSISeries(prices, period = 14) {
  const result = new Array(prices.length).fill(null);
  if (prices.length < period + 1) return result;

  const changes = prices.slice(1).map((p, i) => p.close - prices[i].close);
  const gains = changes.map(c => c > 0 ? c : 0);
  const losses = changes.map(c => c < 0 ? Math.abs(c) : 0);

  let avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
  let avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;

  const rsiVal = (ag, al) => al === 0 ? 100 : Math.round((100 - 100 / (1 + ag / al)) * 100) / 100;
  result[period] = rsiVal(avgGain, avgLoss);

  for (let i = period; i < changes.length; i++) {
    avgGain = (avgGain * (period - 1) + gains[i]) / period;
    avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
    result[i + 1] = rsiVal(avgGain, avgLoss);
  }

  return result;
}

// ─── Rolling 52-week drawdown series ─────────────────────────────────────────
function computeDrawdownSeries(prices) {
  const windowSize = 252; // ~1 trading year
  return prices.map((p, i) => {
    const window = prices.slice(Math.max(0, i - windowSize + 1), i + 1);
    const high = Math.max(...window.map(w => w.close));
    return high > 0 ? Math.round(((p.close - high) / high) * 10000) / 100 : null;
  });
}

// ─── Signal Classification ────────────────────────────────────────────────────
function classifyTier(rsi, vix, drawdown, thresholds) {
  if (!rsi || !vix) return 'NONE';
  const t = thresholds;
  if (rsi < t.extreme_rsi && vix > t.extreme_vix && drawdown !== null && drawdown < t.extreme_dd) return 'EXTREME';
  if (rsi < t.high_rsi && vix > t.high_vix) return 'HIGH';
  if (rsi < t.medium_rsi && vix > t.medium_vix) return 'MEDIUM';
  if (rsi < t.low_rsi && vix > t.low_vix) return 'WATCH';
  return 'NONE';
}

// ─── Main Computation Function ───────────────────────────────────────────────
/**
 * Compute signal series for VDGR with VIX overlay
 * @param {Object} options - Configuration
 * @param {string} options.vdgrRange - Yahoo range for VDGR (default '2y')
 * @param {string} options.vixRange - Yahoo range for VIX (default '2y')
 * @param {Object} options.thresholds - Signal thresholds
 * @returns {Array} Array of { date, price, rsi, vix, drawdown, tier }
 */
export async function computeSignalSeries(options = {}) {
  const {
    vdgrRange = '2y',
    vixRange = '2y',
    thresholds = {
      extreme_rsi: 30, extreme_vix: 30, extreme_dd: -10,
      high_rsi: 35, high_vix: 25,
      medium_rsi: 45, medium_vix: 20,
      low_rsi: 50, low_vix: 18
    }
  } = options;

  // Fetch raw data
  const [vdgrResult, vixResult] = await Promise.all([
    fetchYahooSeries('VDGR.AX', vdgrRange),
    fetchYahooSeries('%5EVIX', vixRange)
  ]);

  // Parse VDGR prices with exchange timezone dates
  const vdgrPrices = vdgrResult.timestamp
    .map((t, i) => ({
      timestamp: t,
      date: getExchangeDate(t, 'Australia/Sydney'),
      close: vdgrResult.indicators.quote[0].close[i]
    }))
    .filter(p => p.close !== null && !isNaN(p.close));

  // Drop in-progress bar (today's bar if market is still open)
  const lastBar = vdgrPrices[vdgrPrices.length - 1];
  if (lastBar && isMarketOpen(lastBar.timestamp, 'Australia/Sydney')) {
    vdgrPrices.pop();
    console.log(`[signalComputation] Dropped in-progress bar: ${lastBar.date}`);
  }

  // Parse VIX prices with exchange timezone dates
  const vixPrices = vixResult.timestamp
    .map((t, i) => ({
      timestamp: t,
      date: getExchangeDate(t, 'America/New_York'),
      close: vixResult.indicators.quote[0].close[i]
    }))
    .filter(p => p.close !== null && !isNaN(p.close));

  // Drop in-progress VIX bar
  const lastVixBar = vixPrices[vixPrices.length - 1];
  if (lastVixBar && isMarketOpen(lastVixBar.timestamp, 'America/New_York')) {
    vixPrices.pop();
    console.log(`[signalComputation] Dropped in-progress VIX bar: ${lastVixBar.date}`);
  }

  // Build VIX lookup by date
  const vixByDate = {};
  vixPrices.forEach(v => {
    vixByDate[v.date] = Math.round(v.close * 100) / 100;
  });

  // Compute indicator series over full history
  const rsiSeries = computeRSISeries(vdgrPrices);
  const drawdownSeries = computeDrawdownSeries(vdgrPrices);

  // Build output with VIX joined by date
  const series = vdgrPrices.map((p, i) => {
    const rsi = rsiSeries[i];
    const drawdown = drawdownSeries[i];

    // Join VIX by date, falling back to latest earlier VIX
    let vix = vixByDate[p.date];
    if (!vix) {
      // Find latest earlier VIX
      const vixDates = Object.keys(vixByDate).sort();
      for (let j = vixDates.length - 1; j >= 0; j--) {
        if (vixDates[j] < p.date) {
          vix = vixByDate[vixDates[j]];
          break;
        }
      }
    }

    const tier = classifyTier(rsi, vix, drawdown, thresholds);

    return {
      date: p.date,
      price: Math.round(p.close * 10000) / 10000,
      rsi: rsi !== null ? Math.round(rsi * 100) / 100 : null,
      vix: vix !== null ? vix : null,
      drawdown: drawdown !== null ? drawdown : null,
      tier
    };
  });

  return series;
}

/**
 * Get the last completed bar from the signal series
 * @param {Object} options - Same as computeSignalSeries
 * @returns {Object} { date, price, rsi, vix, drawdown, tier, high52w }
 */
export async function getLatestSignal(options = {}) {
  const series = await computeSignalSeries(options);
  if (!series.length) throw new Error('No data available');

  const latest = series[series.length - 1];

  // Compute 52-week high from the series
  const last252 = series.slice(-252);
  const high52w = Math.max(...last252.map(s => s.price));

  return {
    ...latest,
    high52w: Math.round(high52w * 10000) / 10000
  };
}
