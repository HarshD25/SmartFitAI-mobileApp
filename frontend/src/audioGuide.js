// src/audioGuide.js
const COOLDOWN_MS = 1200;
let lastTime = 0;
let lastMsg = '';

function canSpeak(msg) {
  if (!msg) return false;
  const now = Date.now();
  if (msg === lastMsg && (now - lastTime) < 6000) return false;
  if ((now - lastTime) < COOLDOWN_MS) return false;
  return true;
}

export const Voice = {
  speak(msg) {
    try {
      if (!canSpeak(msg)) return;
      lastTime = Date.now();
      lastMsg = msg;
      if ('speechSynthesis' in window) {
        try {
          const u = new SpeechSynthesisUtterance(msg);
          u.lang = 'en-US';
          u.rate = 1.0;
          window.speechSynthesis.cancel();
          window.speechSynthesis.speak(u);
          return;
        } catch (e) {
          console.warn('TTS error', e);
        }
      }
    } catch (e) {
      console.warn('Voice failure', e);
    }
  }
};
