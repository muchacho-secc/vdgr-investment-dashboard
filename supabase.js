// supabase.js — Supabase REST helpers for the VDGR Signal Worker
// All VDGR tables live in the vdgr schema — schema headers are added to every request.

const SCHEMA_READ  = { 'Accept-Profile':  'vdgr' };
const SCHEMA_WRITE = { 'Content-Profile': 'vdgr' };

const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 1000; // Base delay, will be multiplied by attempt number

function headers(env, extra = {}) {
  return {
    'apikey':         env.SUPABASE_ANON_KEY,
    'Authorization':  `Bearer ${env.SUPABASE_SERVICE_KEY}`,
    'Content-Type':   'application/json',
    ...extra,
  };
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function isRetryableError(status, body) {
  // Network errors: no status
  if (!status) return true;

  // HTTP errors: 429 (rate limit), 502/503/504 (server errors)
  if ([429, 502, 503, 504].includes(status)) return true;

  // PostgREST errors: PGRST002 (connection failed), PGRST000 (internal error)
  if (body && typeof body === 'string') {
    if (body.includes('PGRST002') || body.includes('PGRST000')) return true;
  }

  return false;
}

async function fetchWithRetry(url, options, operation) {
  let lastError;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const res = await fetch(url, options);
      const body = await res.text();

      if (!res.ok) {
        if (isRetryableError(res.status, body) && attempt < MAX_RETRIES) {
          const delay = RETRY_DELAY_MS * attempt;
          console.warn(`[supabase] ${operation} failed (${res.status}), retrying in ${delay}ms (attempt ${attempt}/${MAX_RETRIES})`);
          await sleep(delay);
          continue;
        }
        throw new Error(`Supabase ${operation} failed: ${body}`);
      }

      return body ? JSON.parse(body) : null;
    } catch (err) {
      lastError = err;

      // Network error (fetch failed)
      if (err.message.includes('fetch') && attempt < MAX_RETRIES) {
        const delay = RETRY_DELAY_MS * attempt;
        console.warn(`[supabase] ${operation} network error, retrying in ${delay}ms (attempt ${attempt}/${MAX_RETRIES}): ${err.message}`);
        await sleep(delay);
        continue;
      }

      throw err;
    }
  }

  throw lastError;
}

export async function supabaseGet(env, table, query = '') {
  return fetchWithRetry(
    `${env.SUPABASE_URL}/rest/v1/${table}${query}`,
    { headers: headers(env, SCHEMA_READ) },
    `GET ${table}`
  );
}

export async function supabasePost(env, table, data) {
  return fetchWithRetry(
    `${env.SUPABASE_URL}/rest/v1/${table}`,
    {
      method:  'POST',
      headers: headers(env, { 'Prefer': 'return=representation', ...SCHEMA_WRITE }),
      body:    JSON.stringify(data),
    },
    `POST ${table}`
  );
}

export async function supabasePostMinimal(env, table, data) {
  await fetchWithRetry(
    `${env.SUPABASE_URL}/rest/v1/${table}`,
    {
      method:  'POST',
      headers: headers(env, { 'Prefer': 'return=minimal', ...SCHEMA_WRITE }),
      body:    JSON.stringify(data),
    },
    `POST ${table}`
  );
}

export async function supabasePatch(env, table, query, data) {
  return fetchWithRetry(
    `${env.SUPABASE_URL}/rest/v1/${table}${query}`,
    {
      method:  'PATCH',
      headers: headers(env, { 'Prefer': 'return=representation', ...SCHEMA_WRITE }),
      body:    JSON.stringify(data),
    },
    `PATCH ${table}`
  );
}
