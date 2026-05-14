// supabase.js — Supabase REST helpers for the VDGR Signal Worker
// All VDGR tables live in the vdgr schema — schema headers are added to every request.

const SCHEMA_READ  = { 'Accept-Profile':  'vdgr' };
const SCHEMA_WRITE = { 'Content-Profile': 'vdgr' };

function headers(env, extra = {}) {
  return {
    'apikey':         env.SUPABASE_ANON_KEY,
    'Authorization':  `Bearer ${env.SUPABASE_SERVICE_KEY}`,
    'Content-Type':   'application/json',
    ...extra,
  };
}

export async function supabaseGet(env, table, query = '') {
  const res = await fetch(`${env.SUPABASE_URL}/rest/v1/${table}${query}`, {
    headers: headers(env, SCHEMA_READ),
  });
  if (!res.ok) throw new Error(`Supabase GET ${table} failed: ${await res.text()}`);
  return res.json();
}

export async function supabasePost(env, table, data) {
  const res = await fetch(`${env.SUPABASE_URL}/rest/v1/${table}`, {
    method:  'POST',
    headers: headers(env, { 'Prefer': 'return=representation', ...SCHEMA_WRITE }),
    body:    JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Supabase POST ${table} failed: ${await res.text()}`);
  return res.json();
}

export async function supabasePostMinimal(env, table, data) {
  const res = await fetch(`${env.SUPABASE_URL}/rest/v1/${table}`, {
    method:  'POST',
    headers: headers(env, { 'Prefer': 'return=minimal', ...SCHEMA_WRITE }),
    body:    JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Supabase POST ${table} failed: ${await res.text()}`);
}

export async function supabasePatch(env, table, query, data) {
  const res = await fetch(`${env.SUPABASE_URL}/rest/v1/${table}${query}`, {
    method:  'PATCH',
    headers: headers(env, { 'Prefer': 'return=representation', ...SCHEMA_WRITE }),
    body:    JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Supabase PATCH ${table} failed: ${await res.text()}`);
  return res.json();
}
