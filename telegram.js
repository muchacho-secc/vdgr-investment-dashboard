// telegram.js — Telegram Bot API notifications

const SIGNAL_EMOJI = {
  WATCH:   '🟡',
  MEDIUM:  '🟠',
  HIGH:    '🔴',
  EXTREME: '🟣',
};

const SIGNAL_CTA = {
  MEDIUM:  'Open the app to log your investment.',
  HIGH:    'Open the app to log your investment.',
  EXTREME: 'Open the app to review and log your investment.',
};

// Strip markdown formatting characters from text
function stripMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/[*_~`\[\]()]/g, '') // Remove markdown characters
    .replace(/#{1,6}\s/g, '')     // Remove header markers
    .replace(/>\s/g, '')          // Remove blockquote markers
    .trim();
}

function buildTelegramMessage({ tier, recommendedAmount, analystSummary, rsi, vix, currentPrice, drawdownPct }) {
  const emoji = SIGNAL_EMOJI[tier] || '⚪';
  const amountDisplay = tier === 'EXTREME' ? `$${recommendedAmount}+` : `$${recommendedAmount}`;
  const cta = SIGNAL_CTA[tier] || '';
  const priceFormatted = typeof currentPrice === 'number' ? currentPrice.toFixed(2) : currentPrice;
  const rsiFormatted = typeof rsi === 'number' ? rsi.toFixed(1) : rsi;
  const vixFormatted = typeof vix === 'number' ? vix.toFixed(1) : vix;
  const ddFormatted  = typeof drawdownPct === 'number' ? drawdownPct.toFixed(1) : '—';

  // Strip markdown from analyst summary
  const summary = stripMarkdown(analystSummary);

  return [
    `${emoji} VDGR Signal: ${tier}`,
    ``,
    `💰 Recommended: ${amountDisplay}`,
    `📊 RSI: ${rsiFormatted} | VIX: ${vixFormatted} | Price: $${priceFormatted}`,
    `📉 Drawdown: ${ddFormatted}%`,
    ``,
    summary,
    ``,
    cta,
  ].join('\n');
}

async function sendMessage(env, text) {
  const res = await fetch(
    `https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chat_id: env.TELEGRAM_CHAT_ID,
        text,
        // Send as plain text (not markdown) to avoid formatting issues
        disable_web_page_preview: true,
      }),
    }
  );

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Telegram sendMessage failed: ${err}`);
  }

  return res.json();
}

// Send a signal notification — only called for MEDIUM, HIGH, EXTREME
export async function sendTelegramNotification(env, messageData) {
  const text = buildTelegramMessage(messageData);

  try {
    return await sendMessage(env, text);
  } catch (err) {
    console.error('[telegram] Failed to send signal notification:', err.message);

    // Fallback: send minimal notification without analyst summary
    const { tier, recommendedAmount } = messageData;
    const emoji = SIGNAL_EMOJI[tier] || '⚪';
    const fallbackText = `${emoji} VDGR Signal: ${tier}\n\n💰 Recommended: $${recommendedAmount}\n\nOpen the app for details.`;

    try {
      await sendMessage(env, fallbackText);
      console.log('[telegram] Sent fallback notification');
    } catch (fallbackErr) {
      console.error('[telegram] Fallback notification also failed:', fallbackErr.message);
      throw err; // Throw original error
    }
  }
}

// Send an error alert when signal generation fails
export async function sendTelegramError(env, errorMessage) {
  // Sanitise — never send raw stack traces
  const sanitised = String(errorMessage).slice(0, 200);
  const text = `⚠️ *VDGR Signal Error*\n\nSignal generation failed today.\n\n\`${sanitised}\``;
  try {
    await sendMessage(env, text);
  } catch {
    // If error notification itself fails, just log — don't throw
    console.error('[telegram] Failed to send error notification');
  }
}
