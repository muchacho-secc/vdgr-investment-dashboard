// langsmith.js — Workers-compatible LangSmith tracing using REST API
// Silently skips tracing if LANGSMITH_API_KEY is not set

const LANGSMITH_PROJECT = 'vdgr-dashboard';

function getLangSmithAPI(env) {
  const endpoint = env.LANGSMITH_ENDPOINT || 'https://api.smith.langchain.com';
  console.log('[langsmith] API key present:', !!env.LANGSMITH_API_KEY);
  console.log('[langsmith] Using endpoint:', endpoint);
  return endpoint;
}

function generateUUID() {
  return crypto.randomUUID();
}

async function createRun(env, runData) {
  if (!env.LANGSMITH_API_KEY) return null;

  const apiEndpoint = getLangSmithAPI(env);
  console.log('[langsmith] Creating run with session_name:', runData.session_name);

  try {
    const response = await fetch(`${apiEndpoint}/runs`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': env.LANGSMITH_API_KEY,
      },
      body: JSON.stringify(runData),
    });

    if (!response.ok) {
      console.error('LangSmith create run failed:', response.status);
      return null;
    }

    return runData.id;
  } catch (error) {
    console.error('LangSmith create run error:', error.message);
    return null;
  }
}

async function updateRun(env, runId, updates) {
  if (!env.LANGSMITH_API_KEY || !runId) return;

  const apiEndpoint = getLangSmithAPI(env);

  try {
    await fetch(`${apiEndpoint}/runs/${runId}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': env.LANGSMITH_API_KEY,
      },
      body: JSON.stringify(updates),
    });
  } catch (error) {
    console.error('LangSmith update run error:', error.message);
  }
}

/**
 * Generic traced function wrapper
 * @param {Object} env - Cloudflare env with LANGSMITH_API_KEY and LANGSMITH_PROJECT
 * @param {string} name - Run name for LangSmith
 * @param {Function} fn - Async function to trace
 * @param {Object} metadata - Additional metadata to attach to the run
 * @returns {Promise<any>} - Result from fn
 */
export async function traced(env, name, fn, metadata = {}) {
  const runId = generateUUID();
  const startTime = Date.now();

  console.log('[langsmith] traced() - Project name:', LANGSMITH_PROJECT);

  const runData = {
    id: runId,
    name,
    run_type: 'chain',
    start_time: new Date(startTime).toISOString(),
    session_name: LANGSMITH_PROJECT,
    extra: metadata,
  };

  await createRun(env, runData);

  try {
    const result = await fn();
    const endTime = Date.now();

    await updateRun(env, runId, {
      end_time: new Date(endTime).toISOString(),
      outputs: { result },
      error: null,
    });

    return result;
  } catch (error) {
    const endTime = Date.now();

    await updateRun(env, runId, {
      end_time: new Date(endTime).toISOString(),
      error: error.message,
    });

    throw error;
  }
}

/**
 * Convenience wrapper for Anthropic API calls with LangSmith tracing
 * @param {Object} env - Cloudflare env
 * @param {string} name - Run name (e.g., "daily-summary", "analyst-chat")
 * @param {string} systemPrompt - System prompt
 * @param {Array} messages - Messages array
 * @param {number} maxTokens - Max tokens
 * @returns {Promise<string>} - Response text from Claude
 */
export async function traceAnthropicCall(env, name, systemPrompt, messages, maxTokens = 400) {
  const runId = generateUUID();
  const startTime = Date.now();

  console.log('[langsmith] traceAnthropicCall() - Project name:', LANGSMITH_PROJECT);

  const runData = {
    id: runId,
    name,
    run_type: 'llm',
    start_time: new Date(startTime).toISOString(),
    session_name: LANGSMITH_PROJECT,
    inputs: {
      system: systemPrompt,
      messages,
    },
    extra: {
      model: 'claude-sonnet-4-5',
      max_tokens: maxTokens,
    },
  };

  await createRun(env, runData);

  try {
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

    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`Anthropic API error: ${res.status} ${errorText}`);
    }

    const data = await res.json();
    const responseText = data.content[0].text;
    const endTime = Date.now();

    await updateRun(env, runId, {
      end_time: new Date(endTime).toISOString(),
      outputs: {
        response: responseText,
      },
      extra: {
        ...runData.extra,
        usage: data.usage,
      },
    });

    return responseText;
  } catch (error) {
    const endTime = Date.now();

    await updateRun(env, runId, {
      end_time: new Date(endTime).toISOString(),
      error: error.message,
    });

    throw error;
  }
}
