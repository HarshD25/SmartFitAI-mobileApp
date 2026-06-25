// src/ui.js
export function showStep(n) {
  const step1 = document.getElementById('step1');
  const step2 = document.getElementById('step2');
  const step3 = document.getElementById('step3');
  const dots = [document.getElementById('dot1'), document.getElementById('dot2'), document.getElementById('dot3')];
  [step1, step2, step3].forEach((el,i) => el.classList.toggle('active', i === n-1));
  dots.forEach((d,i) => { if (d) d.classList.toggle('on', i === n-1); });
  window.scrollTo({ top:0, behavior:'smooth' });
}

export function setStatus(text, subtitle = '') {
  const st = document.getElementById('status');
  const sub = document.getElementById('subtitle');
  if (st) st.textContent = text || '';
  if (sub) sub.textContent = subtitle || '';
}
