/* Pro volume engine. Loads after nifty.js and reuses its pricer (bs, gk, ivK), legs() notation, plot() and mini().
   Adds: Greek fingerprints, stress grid, P&L explain, delta-hedge Monte Carlo, risk-neutral density, 0DTE clock,
   and the calculators (forward vol, implied correlation, HAR-RV, Kelly with tails, deflated Sharpe, participant-OI gamma, SPRT). */
calc = function () {};                                   // the base manual's calculator is not on this page
const $ = id => document.getElementById(id), V = id => +$(id).value;
const pf = (v, n = 1) => { const t = v.toFixed(n); return (/^-0\.?0*$/.test(t) ? t.slice(1) : t).replace(/^(?!-)/, '+') };
const rs = v => (v < 0 ? '−₹' : '₹') + Math.abs(Math.round(v)).toLocaleString('en-IN');
const rk = v => (v < 0 ? '−' : '') + '₹' + (Math.abs(v) >= 1e5 ? (Math.abs(v) / 1e5).toFixed(2) + 'L' : (Math.abs(v) / 1e3).toFixed(1) + 'k');
const sc = v => v > 1e-9 ? 'pos' : v < -1e-9 ? 'neg' : '';

/* ---------- calibration (guide/pro_calib.js, written by pro_calibrate.py) ---------- */
const CAL = window.PRO_CALIB || null, SURF = CAL && CAL.surface_latest && CAL.surface_latest.svi ? CAL.surface_latest : null;
let MODEL = SURF ? 'svi' : 'toy';
const sviW = (p, k) => p[0] + p[1] * (p[2] * (k - p[3]) + Math.sqrt((k - p[3]) ** 2 + p[4] ** 2));
/* your fitted smile, carried to other expiries in normalised moneyness k·√(T_fit/T); ATM level = the IV input */
function ivSurf(K, T) {
  const Tu = Math.max(T || 7 / 365, 1 / 8760), w0 = sviW(SURF.svi, 0);
  let k = Math.log(K / NF.S) * Math.sqrt(SURF.T / Tu); k = Math.max(SURF.k_range[0] * 1.6, Math.min(SURF.k_range[1] * 1.6, k));
  return NF.IV * Math.sqrt(Math.max(sviW(SURF.svi, k), 1e-12) / w0);
}
const _ivToy = ivK;
ivK = (K, T) => MODEL == 'svi' && SURF ? ivSurf(K, T) : _ivToy(K);
const _legs = legs;
legs = s => _legs(s).map(l => { if (l.cp != 'f') { l.v = ivK(l.K, l.d / 365); l.p = bs(l.cp, NF.S, l.K, l.d / 365, l.v) } return l });
function mTerms(T, m) {
  const kb = Math.exp(m.muJ + m.deltaJ ** 2 / 2) - 1, out = []; let p = Math.exp(-m.lam * T), tot = 0;
  for (let n = 0; n <= 40; n++) { if (n) p *= m.lam * T / n; out.push([p, -m.lam * kb * T - m.sigma ** 2 * T / 2 + n * m.muJ, m.sigma ** 2 * T + n * m.deltaJ ** 2]); tot += p; if (tot > 1 - 1e-10) break }
  return out;
}
const mCdf = (x, T, m) => mTerms(T, m).reduce((a, [p, mu, v]) => a + p * Nc((x - mu) / Math.sqrt(v)), 0);
const mPdf = (x, T, m) => mTerms(T, m).reduce((a, [p, mu, v]) => a + p * np((x - mu) / Math.sqrt(v)) / Math.sqrt(v), 0);
const lnCdf = (x, T, v) => Nc((x + v * v * T / 2) / (v * Math.sqrt(T)));

/* ---------- position valuation with shocks ---------- */
/* skew shock: +1 means 25Δ-style risk reversal steepens by 1 vol pt (strike 3% below spot +0.5, 3% above −0.5), linear, capped */
const skw = K => Math.max(-3, Math.min(3, .5 * (NF.S - K) / (.03 * NF.S))) / 100;
function pv(L, S, el = 0, dv = 0, sk = 0, fdv = null) {
  return L.reduce((a, l) => {
    if (l.cp == 'f') return a + l.q * S;
    const v = Math.max(.01, l.v + (fdv !== null && l.d <= 7 ? fdv : dv) + sk * skw(l.K));
    return a + l.q * bs(l.cp, S, l.K, Math.max(l.d - el, 0) / 365, v);
  }, 0);
}
function gx(L, S = NF.S) {
  const h = S * .002, P = (s, e = 0, dv = 0, sk = 0) => pv(L, s, e, dv, sk);
  const D = dv => (P(S + h, 0, dv) - P(S - h, 0, dv)) / (2 * h), Vg = dv => P(S, 0, dv + .005) - P(S, 0, dv - .005), v0 = P(S);
  const minD = Math.min(...L.filter(l => l.cp != 'f').map(l => l.d), 99);
  return {
    v: v0, d: D(0), g: (P(S + h) - 2 * v0 + P(S - h)) / (h * h), t: P(S, Math.min(1, minD)) - v0,
    ve: Vg(0), va: D(.005) - D(-.005), vo: (Vg(.01) - Vg(-.01)) / 2, sk: P(S, 0, 0, 1) - v0,
    fr: pv(L, S, 0, 0, 0, .01) - v0                       // front-week (≤7d) vol +1, back unchanged
  };
}

/* ---------- presets shared by the tools ---------- */
const PRE = [
  ['Short straddle', 'c0:-1 p0:-1'], ['Short strangle ±400', 'c400:-1 p-400:-1'],
  ['Iron fly, ±400 wings', 'c0:-1 p0:-1 c400:1 p-400:1'], ['Iron condor 300/500', 'c300:-1 c500:1 p-300:-1 p-500:1'],
  ['Long straddle', 'c0:1 p0:1'], ['Put ratio 1×2 (−200/−500)', 'p-200:1 p-500:-2'],
  ['Call backspread 1×2 (0/+300)', 'c0:-1 c300:2'], ['Calendar ATM call (7d/35d)', 'c0:-1 c0:1@35'],
  ['Double diagonal (7d short ±300, 35d long ±400)', 'c300:-1 p-300:-1 c400:1@35 p-400:1@35'],
  ['Jade lizard', 'p-200:-1 c300:-1 c400:1'], ['Broken-wing put fly 0/−200/−500', 'p0:1 p-200:-2 p-500:1'],
  ['Risk reversal (long +300C, short −300P)', 'c300:1 p-300:-1'], ['Long OTM call fly +300/+500/+700', 'c300:1 c500:-2 c700:1'],
  ['Zebra: 2× ITM −300C, short ATM C', 'c-300:2 c0:-1'], ['Collar on long future', 'f:1 p-200:1 c300:-1'],
  ['Short weekly put vs long monthly put', 'p-200:-1 p-400:1@35'], ['Long future', 'f:1']
];
function fillSel(id, def = 0) { const s = $(id); if (!s || s.options.length) return; PRE.forEach((p, i) => s.add(new Option(p[0], p[1], i == def, i == def))); }
function linkSel(sel, inp, fn) { const s = $(sel), t = $(inp); if (!s) return; if (!t.value) t.value = s.value; s.onchange = () => { t.value = s.value; fn() }; t.oninput = fn; }

/* ---------- Greek fingerprint table ---------- */
function fingerprint(el) {
  const rows = PRE.filter(p => p[1] != 'f:1').map(([n, s]) => { const L = legs(s), g = gx(L); return { n, s, g } });
  const c = (v, k = 1) => `<td class="${sc(v)} r">${pf(v, k)}</td>`;
  el.innerHTML = '<tr><th>Structure</th><th>Expiry payoff</th><th>Δ</th><th>Γ ×10⁴</th><th>Θ /day</th><th>Vega</th><th>Front-week vega</th><th>Vanna</th><th>Volga</th><th>Skew +1</th></tr>' +
    rows.map(r => `<tr><td>${r.n}<div class="lg2">${r.s}</div></td><td data-m="${r.s}"></td>${c(r.g.d, 2)}${c(r.g.g * 1e4, 1)}${c(r.g.t, 1)}${c(r.g.ve, 1)}${c(r.g.fr, 1)}${c(r.g.va, 3)}${c(r.g.vo, 2)}${c(r.g.sk, 1)}</tr>`).join('');
  el.querySelectorAll('[data-m]').forEach(mini);
}

/* ---------- stress grid ---------- */
const MV = [-10, -8, -5, -3, -2, -1, 0, 1, 2, 3, 5], DV = [-4, -2, 0, 3, 6, 10, 15];
const coupled = m => m < 0 ? Math.min(15, -2 * m) : -.6 * m;   // illustrative spot-vol coupling (vol pts per 1%)
function stress() {
  const o = $('sgO'); if (!o) return; let L; try { L = legs($('sgL').value) } catch (e) { o.innerHTML = '<p class=neg>Cannot parse legs.</p>'; return }
  const el = V('sgE'), base = pv(L, NF.S), cells = MV.map(m => DV.map(d => (pv(L, NF.S * (1 + m / 100), el, d / 100, 0) - base) * NF.LOT));
  const mx = Math.max(...cells.flat().map(Math.abs)) || 1, worst = Math.min(...cells.flat());
  const mer = SURF && SURF.merton, Tm = SURF ? SURF.T : 0;
  const pr = m => { if (!mer || !m) return ['', '']; const x = Math.log(1 + m / 100), j = mCdf(x, Tm, mer), l = lnCdf(x, Tm, NF.IV); return (m < 0 ? [j, l] : [1 - j, 1 - l]).map(v => (v * 100).toFixed(v < .001 ? 3 : 2) + '%') };
  let h = '<table class="sg2"><tr><th>Spot \\ IV</th>' + DV.map(d => `<th>${pf(d, 0)} vol</th>`).join('') + (mer ? '<th>P by expiry: jump fit</th><th>lognormal</th>' : '') + '</tr>';
  MV.forEach((m, i) => {
    const cv = coupled(m), ci = DV.reduce((b, d, j) => Math.abs(d - cv) < Math.abs(DV[b] - cv) ? j : b, 0);
    h += `<tr><th>${pf(m, 0)}% <span class=mu>${Math.round(NF.S * (1 + m / 100))}</span></th>` + cells[i].map((v, j) =>
      `<td class="r${j == ci ? ' cp' : ''}" style="background:color-mix(in srgb,var(${v >= 0 ? '--pos' : '--neg'}) ${Math.round(Math.min(1, Math.abs(v) / mx) * 42)}%,var(--card))">${rk(v)}</td>`).join('') + (mer ? pr(m).map(x => `<td class="r mu">${x}</td>`).join('') : '') + '</tr>';
  });
  o.innerHTML = h + '</table>' + histTail(L) + `<p class="cap">Per 1 lot (${NF.LOT}) of each leg, ${el ? el + ' day(s) later' : 'instant gap'}, sticky-strike skew. Outlined cells: the IV change that has historically come with that spot move (about +2 vol per 1% down, −0.6 per 1% up; illustrative). Worst cell ${rk(worst)}. Size so the outlined −8% to −10% cell is a loss you can take and keep trading.${mer ? ` Right-hand columns: probability of finishing at or beyond that move by the fitted weekly's expiry (${SURF.expiry}), from the Merton jump model fitted to your chain, vs a lognormal at the ATM IV. Their ratio is how much fatter the market prices the tail.` : ''}</p>`;
}

/* historical replay: every archived session's Nifty move and VIX ratio applied to these legs over one day */
function histTail(L, alpha = .975) {
  const H = CAL && CAL.scenarios; if (!H || !H.n || H.n < 40) return '';
  const base = pv(L, NF.S), out = [];
  for (let i = 0; i < H.n; i++) {
    const vr = H.v[i], Ls = L.map(l => l.cp == 'f' ? l : Object.assign({}, l, { v: Math.min(3, Math.max(.01, l.v * vr)) }));
    out.push([(pv(Ls, NF.S * Math.exp(H.r[i]), 1) - base) * NF.LOT, H.d[i]]);
  }
  out.sort((a, b) => a[0] - b[0]);
  const k = Math.max(1, Math.floor(H.n * (1 - alpha))), es = -out.slice(0, k).reduce((a, x) => a + x[0], 0) / k, vr = -out[k - 1][0];
  const pl = out.filter(x => x[0] < 0).length / H.n;
  return `<p class="cap"><b>Historical replay</b>, ${H.n} sessions ${H.from} to ${H.to} on your archive: each day's Nifty move and VIX ratio applied to these legs (each leg's IV scaled by the VIX ratio), one day later. ` +
    `1-day VaR97.5 <b>${rs(Math.max(vr, 0))}</b>, ES97.5 <b class="neg">${rs(Math.max(es, 0))}</b> (the average of the worst ${k} days), loss on ${(pl * 100).toFixed(0)}% of days. Worst: ` +
    out.slice(0, 3).map(x => `${x[1]} ${rs(x[0])}`).join(', ') + '. Unlike the grid, these moves and IV changes came together; unlike a model, the sample has only the crashes it has seen.</p>';
}

/* ---------- P&L explain ---------- */
function explain() {
  const o = $('peO'); if (!o) return; let L; try { L = legs($('peL').value) } catch (e) { o.innerHTML = '<p class=neg>Cannot parse legs.</p>'; return }
  const dS = V('peS'), dv = V('peV'), dt = V('peT'), g = gx(L), dvp = dv;   // dv in vol points; Greeks are per vol point
  const T = [['Delta  Δ·dS', g.d * dS], ['Gamma  ½Γ·dS²', .5 * g.g * dS * dS], ['Theta  Θ·dt', g.t * dt], ['Vega  ν·dσ', g.ve * dvp],
    ['Vanna  dΔ/dσ·dS·dσ', g.va * dS * dvp], ['Volga  ½·dν/dσ·dσ²', .5 * g.vo * dvp * dvp]];
  const sum = T.reduce((a, r) => a + r[1], 0), act = pv(L, NF.S + dS, dt, dv / 100) - g.v;
  o.innerHTML = '<div class=tw><table><tr><th>Term</th><th class=r>Points</th><th class=r>₹ per lot</th></tr>' + T.map(r => `<tr><td>${r[0]}</td><td class="r ${sc(r[1])}">${pf(r[1])}</td><td class="r ${sc(r[1])}">${rs(r[1] * NF.LOT)}</td></tr>`).join('') +
    `<tr><th>Sum of terms</th><th class=r>${pf(sum)}</th><th class=r>${rs(sum * NF.LOT)}</th></tr><tr><th>Full re-price</th><th class=r>${pf(act)}</th><th class=r>${rs(act * NF.LOT)}</th></tr>` +
    `<tr><td>Residual (higher order: speed, colour, charm)</td><td class=r>${pf(act - sum)}</td><td class=r>${rs((act - sum) * NF.LOT)}</td></tr></table></div>`;
}

/* ---------- delta-hedge Monte Carlo ---------- */
function rng32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296 } }
const b0 = (cp, S, K, T, v) => { if (T <= 1e-9) return cp == 'c' ? Math.max(S - K, 0) : Math.max(K - S, 0); const s = v * Math.sqrt(T), a = (Math.log(S / K) + s * s / 2) / s; return cp == 'c' ? S * Nc(a) - K * Nc(a - s) : K * Nc(s - a) - S * Nc(-a) };
const d0 = (S, K, T, v) => { if (T <= 1e-9) return S > K ? 1 : S < K ? -1 : 0; const a = (Math.log(S / K) + v * v * T / 2) / (v * Math.sqrt(T)); return 2 * Nc(a) - 1 };   // straddle delta
function hedgeSim() {
  const o = $('hsO'); if (!o) return;
  const side = V('hsSide'), D = Math.max(1, V('hsD')), si = V('hsI') / 100, sr = V('hsR') / 100, n = Math.max(1, V('hsN')), lots = Math.max(1, V('hsL')),
    cost = V('hsC'), rnd = $('hsRd').checked, M = 600, S0 = NF.S, K = ATM(), T = D / 365, N = Math.round(n * D), dt = T / N, U = lots * NF.LOT, r = rng32(7);
  const prem = b0('c', S0, K, T, si) + b0('p', S0, K, T, si), res = [];
  let hedges = 0;
  for (let p = 0; p < M; p++) {
    let S = S0, h = 0, pl = 0;
    for (let k = 0; k < N; k++) {
      let want = -side * d0(S, K, T - k * dt, si) * U; if (rnd) want = Math.round(want / NF.LOT) * NF.LOT;
      if (want != h) { pl -= Math.abs(want - h) * cost; h = want; hedges++ }
      const z = Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(6.283185307 * r()), S1 = S * Math.exp(-sr * sr * dt / 2 + sr * Math.sqrt(dt) * z);
      pl += h * (S1 - S); S = S1;
    }
    pl += side * (Math.abs(S - K) - prem) * U; res.push(pl);
  }
  const mean = res.reduce((a, b) => a + b, 0) / M, sd = Math.sqrt(res.reduce((a, b) => a + (b - mean) ** 2, 0) / M), win = res.filter(x => x > 0).length / M;
  const vega = 2 * S0 * np(si * Math.sqrt(T) / 2) * Math.sqrt(T), theo = side * vega * (sr - si) * U, dk = Math.sqrt(Math.PI / 4) * vega * sr / Math.sqrt(N) * U;
  const lo = Math.min(...res), hi = Math.max(...res), B = 32, w = (hi - lo) / B || 1, bins = Array(B).fill(0); res.forEach(x => bins[Math.min(B - 1, Math.floor((x - lo) / w))]++);
  const mb = Math.max(...bins), W = 600, H = 170, sx = x => 40 + (x - lo) / (hi - lo || 1) * (W - 52);
  let g = `<svg viewBox="0 0 ${W} ${H + 22}" role=img aria-label="hedged P&L histogram">`;
  bins.forEach((c, i) => { const x0 = lo + i * w, hgt = c / mb * (H - 10); g += `<rect class="${x0 + w / 2 >= 0 ? 'bp' : 'bn'}" x="${sx(x0) + .5}" y="${H - hgt}" width="${Math.max(1, sx(x0 + w) - sx(x0) - 1)}" height="${hgt}"/>` });
  if (lo < 0 && hi > 0) g += `<line class=zl x1=${sx(0)} x2=${sx(0)} y1=0 y2=${H} />`;
  g += `<line class=vx x1=${sx(mean)} x2=${sx(mean)} y1=0 y2=${H} />` + ticks(lo, hi, 6).map(t => `<text class=tx x=${sx(t)} y=${H + 15} text-anchor=middle>${rk(t)}</text>`).join('') + '</svg>';
  o.innerHTML = g + `<div class="ps">${side > 0 ? 'Long' : 'Short'} ${lots} lot ATM ${K} straddle, ${D}d, bought/sold at IV ${(si * 100).toFixed(1)}%, path realises ${(sr * 100).toFixed(1)}%, hedged ${n}×/day${rnd ? ' in whole futures lots' : ' continuously sized'}, ${M} paths.<br>
  <b>Mean</b> ${rk(mean)} (theory vega×(RV−IV) ≈ ${rk(theo)}) · <b>σ of P&amp;L</b> ${rk(sd)} (Derman–Kamal √(π/4)·vega·σ/√N ≈ ${rk(dk)}) · <b>P(profit)</b> ${(win * 100).toFixed(0)}% · hedge trades/path ${(hedges / M).toFixed(0)}</div>`;
}

/* ---------- charts added to the base CH registry ---------- */
const hm = h => { const H = Math.floor(h + 1e-9), m = Math.round((h - H) * 60); return `${H}:${String(m).padStart(2, '0')}` };
function clockRem(m, kind) {      // remaining settlement variance as a fraction of a full session's variance, minute m of 375
  const cu = CAL && CAL.clock && CAL.clock.u5, u = kind == 'U' ? (cu ? s => cu[Math.min(74, Math.floor(s / 5))] : s => 1 + 2.2 * Math.exp(-s / 30) + .6 * Math.exp(-(375 - s) / 40)) : () => 1;
  let tot = 0; for (let s = 0; s < 375; s++) tot += u(s + .5);
  let r = 0; for (let s = Math.floor(m); s < 375; s++) { const w = kind == 'raw' || s < 345 ? 1 : (375 - s - .5) / 30; r += w * w * u(s + .5) }
  return r / tot;
}
Object.assign(CH, {
  odte: () => {
    const sd = NF.S * NF.IV / Math.sqrt(252), xs = rng(0, 374, 120), K = .7979 * sd;
    const ln = (k, lab, kind, dash) => ({ x: xs.map(m => 9.25 + m / 60), y: xs.map(m => K * Math.sqrt(clockRem(m, kind))), k, lab, dash });
    return plot([ln(3, 'flat clock, 15:30 tick (what a naive model assumes)', 'raw', 1), ln(1, 'flat clock + 30-min settlement average', 'flat'), ln(2, (CAL && CAL.clock ? `your clock (${CAL.clock.n_days} days)` : 'U-shaped clock') + ' + settlement average', 'U')],
      { x0: 9.25, x1: 15.5, fx: hm, yl: 'fair ATM straddle on expiry day (pts)', xl: 'time (IST)', vx: [15], al: '0DTE straddle by clock' });
  },
  rnd: () => {
    const lines = [], A = NF.S, h = A * .0025, xs = rng(A * .9, A * 1.1, 161);
    [[7, 1], [30, 2]].forEach(([d, k]) => {
      const T = d / 365, C = (K, f) => bs('c', A, K, T, f ? NF.IV : ivK(K, T)), q = (K, f) => Math.exp(NF.R * T) * (C(K - h, f) - 2 * C(K, f) + C(K + h, f)) / (h * h) * 100;
      lines.push({ x: xs, y: xs.map(K => q(K, 0)), k, lab: `${d}d, ${MODEL == 'svi' ? 'your surface' : 'model skew'}` }, { x: xs, y: xs.map(K => q(K, 1)), k, dash: 1, lab: `${d}d, flat IV (lognormal)` });
    });
    if (SURF && SURF.merton) lines.push({ x: xs, y: xs.map(K => mPdf(Math.log(K / A), 7 / 365, SURF.merton) / K * 100), k: 3, lab: '7d, Merton jump fit' });
    return plot(lines, { fx: v => v.toFixed(0), yl: 'probability per 100 pts', xl: 'Nifty at expiry', vx: [A], al: 'risk-neutral density' });
  },
  spotvol: () => {
    const xs = rng(-.05, .05, 81), sl = -2.2 * NF.IV;                  // ∂σ/∂(ln K) of the model skew
    return plot([{ x: xs.map(x => x * 100), y: xs.map(() => NF.IV * 100), k: 3, lab: 'sticky delta (ATM IV unchanged)' },
      { x: xs.map(x => x * 100), y: xs.map(x => (NF.IV + sl * x) * 100), k: 1, lab: 'sticky strike (slides down the skew)' },
      { x: xs.map(x => x * 100), y: xs.map(x => (NF.IV + 2 * sl * x) * 100), k: 2, lab: 'local vol (≈2× skew slope)' }],
      { zero: 0, fx: v => v.toFixed(0) + '%', yl: 'ATM implied vol %', xl: 'spot move', al: 'spot-vol dynamics' });
  },
  gweight: () => {
    const K = ATM(), xs = rng(NF.S * .96, NF.S * 1.04, 121);
    return plot([5, 2, .5].map((d, i) => ({ x: xs, y: xs.map(S => gk('c', S, K, d / 365, NF.IV).g * S * S * NF.IV ** 2 / 365), k: i + 1, lab: `${d} DTE` })),
      { vx: [K], fx: v => v.toFixed(0), yl: 'straddle theta earned per day at that spot (pts)', xl: 'where spot sits', al: 'dollar gamma' });
  }
});
function rndTable(el) {
  const A = NF.S, h = A * .002, out = [];
  [7, 30].forEach(d => {
    const T = d / 365, P = (K, f) => bs('p', A, K, T, f ? NF.IV : ivK(K, T)), cdf = (K, f) => Math.exp(NF.R * T) * (P(K + h, f) - P(K - h, f)) / (2 * h);
    [-6, -4, -2, 2, 4].forEach(m => { const K = A * (1 + m / 100), a = cdf(K, 0), b = cdf(K, 1); out.push([d, m, K, m < 0 ? a : 1 - a, m < 0 ? b : 1 - b]) });
  });
  el.innerHTML = '<tr><th>Expiry</th><th>Event</th><th class=r>Market (skew)</th><th class=r>Flat-IV lognormal</th><th class=r>Ratio</th></tr>' +
    out.map(r => `<tr><td>${r[0]}d</td><td>Nifty ${r[1] < 0 ? 'below' : 'above'} ${Math.round(r[2])} (${pf(r[1], 0)}%)</td><td class=r>${(r[3] * 100).toFixed(2)}%</td><td class=r>${(r[4] * 100).toFixed(2)}%</td><td class=r>${(r[3] / Math.max(r[4], 1e-9)).toFixed(2)}×</td></tr>`).join('');
}
function odteTable(el) {
  const sd = NF.S * NF.IV / Math.sqrt(252), K = .7979 * sd, T = [[0, '09:15'], [45, '10:00'], [165, '12:00'], [255, '13:30'], [315, '14:30'], [345, '15:00'], [360, '15:15'], [370, '15:25']];
  el.innerHTML = `<tr><th>Time</th><th class=r>Flat, 15:30 tick</th><th class=r>Flat + averaging</th><th class=r>${CAL && CAL.clock ? 'Your clock' : 'U-clock'} + averaging</th></tr>` +
    T.map(([m, t]) => `<tr><td>${t}</td>${['raw', 'flat', 'U'].map(k => `<td class=r>${(K * Math.sqrt(clockRem(m, k))).toFixed(1)}</td>`).join('')}</tr>`).join('');
}

function clockTable(el) {     // same fair price, IV read on a calendar clock vs a trading clock that weights nights and weekends
  const c = CAL && CAL.clock, ow = c && c.overnight_w != null ? c.overnight_w : .25, ww = c && c.weekend_w != null ? c.weekend_w : .3, uy = 252 + 200 * ow + 52 * ww, ex = clockRem(0, 'U');
  const R = [['Wed close', [ow, 1, ow, 1, ww, 1, ow], 6], ['Thu close', [ow, 1, ww, 1, ow], 5], ['Fri close', [ww, 1, ow], 4], ['Mon close', [ow], 1]];
  el.innerHTML = '<tr><th>Snapshot (weekly expires Tue 15:30)</th><th class=r>Calendar days</th><th class=r>Variance units left</th><th class=r>True vol (trading clock)</th><th class=r>IV a calendar-clock model shows</th><th class=r>Error</th></tr>' +
    R.map(([n, L, D]) => { const u = L.reduce((a, b) => a + b, 0) + ex, f = Math.sqrt((u / uy) / (D / 365)); return `<tr><td>${n}</td><td class=r>${D}</td><td class=r>${u.toFixed(2)}</td><td class=r>${(NF.IV * 100).toFixed(1)}%</td><td class=r>${(NF.IV * f * 100).toFixed(1)}%</td><td class="r ${sc(f - 1)}">${pf((f - 1) * 100, 0)}%</td></tr>` }).join('') +
    `<tr><td colspan=6 class=mu style="font-size:.8rem">A session = 1 unit; a weekday night = ${ow.toFixed(2)}, a weekend = ${ww.toFixed(2)}${c ? ` (measured on your ${c.n_overnight} nights and ${c.n_weekend} weekends)` : ' (illustrative)'}; ${uy.toFixed(0)} units a year; expiry day counts ${ex.toFixed(2)} after the settlement average.</td></tr>`;
}

/* ---------- calculators ---------- */
const AB = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239],
  BB = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1],
  CC = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783],
  DD = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
function ninv(p) {                                     // Acklam's inverse normal
  const q = Math.min(Math.max(p, 1e-12), 1 - 1e-12);
  if (q < .02425) { const t = Math.sqrt(-2 * Math.log(q)); return (((((CC[0] * t + CC[1]) * t + CC[2]) * t + CC[3]) * t + CC[4]) * t + CC[5]) / ((((DD[0] * t + DD[1]) * t + DD[2]) * t + DD[3]) * t + 1) }
  if (q > 1 - .02425) return -ninv(1 - q);
  const t = q - .5, r = t * t; return (((((AB[0] * r + AB[1]) * r + AB[2]) * r + AB[3]) * r + AB[4]) * r + AB[5]) * t / (((((BB[0] * r + BB[1]) * r + BB[2]) * r + BB[3]) * r + BB[4]) * r + 1);
}
const CALC = {
  fwd() {
    const a = V('fv1') / 100, d1 = V('fd1'), b = V('fv2') / 100, d2 = V('fd2'), fv = (b * b * d2 - a * a * d1) / (d2 - d1);
    $('fvO').innerHTML = fv <= 0 ? '<b class=neg>Negative forward variance: calendar arbitrage (or a data error). The back expiry is too cheap relative to the front.</b>'
      : `Forward vol between day ${d1} and day ${d2} = <b>${(Math.sqrt(fv) * 100).toFixed(2)}%</b>. If that is far below your view of normal-day vol for those days, the back month is cheap: buy the calendar; if far above, the back is rich.`;
  },
  corr() {
    const si = V('icI') / 100, ss = V('icS') / 100, ws = V('icW') / 100, rho = (si * si - ws * ss * ss) / (ss * ss - ws * ss * ss);
    $('icO').innerHTML = `Implied correlation ≈ <b>${(Math.max(-1, Math.min(1, rho)) * 100).toFixed(0)}%</b> (simple proxy (σ<sub>I</sub>/σ̄)² = ${((si / ss) ** 2 * 100).toFixed(0)}%). Index vol = ${(si * 100).toFixed(1)}% vs ${(ss * 100).toFixed(1)}% weighted stock vol: the index trades at ${(si / ss * 100).toFixed(0)}% of the vol it would have at correlation 1.`;
  },
  har() {
    const d = V('hrD'), w = V('hrW'), m = V('hrM'), iv = V('hrI'), c = V('hrC'), bd = V('hrBd'), bw = V('hrBw'), bm = V('hrBm'), f = c + bd * d + bw * w + bm * m;
    $('hrO').innerHTML = `Forecast RV (next 1–5 days) ≈ <b>${f.toFixed(1)}%</b>. IV ${iv.toFixed(1)}% → expected VRP <b class="${iv - f >= 0 ? 'pos' : 'neg'}">${pf(iv - f)} vol pts</b> for sellers. ${iv - f > 2 ? 'Premium is rich against the forecast.' : iv - f < 0 ? 'Implied is below the forecast: do not be short gamma here.' : 'Thin: sell only with an extra reason (event crush, pin).'}`;
  },
  kelly() {
    const p = V('kP') / 100, W = V('kW'), Lz = V('kL'), pt = V('kT') / 100, T = V('kTs'), pw = p, pl = Math.max(0, 1 - p - pt);
    const G = (f, tail) => { const q = tail ? pt : 0, pL = tail ? pl : 1 - p; if (1 - f * (tail ? T : Lz) <= 0) return -Infinity; return pw * Math.log(1 + f * W) + pL * Math.log(1 - f * Lz) + q * Math.log(1 - f * T) };
    const best = tail => { let b = 0, g = 0; for (let f = 0; f <= 1; f += .001) { const v = G(f, tail); if (v > g) { g = v; b = f } } return [b, g] };
    const [fn, gn] = best(0), [ft, gt] = best(1), gx2 = G(fn, 1);
    $('kO').innerHTML = `Kelly ignoring the tail: <b>${(fn * 100).toFixed(1)}%</b> of capital at risk per trade. With the tail: <b>${(ft * 100).toFixed(1)}%</b> (half-Kelly ${(ft * 50).toFixed(1)}%). ` +
      `Expected log-growth per trade at the tail-aware Kelly ${(gt * 100).toFixed(3)}%; betting the naive Kelly when the tail is real gives ${isFinite(gx2) ? (gx2 * 100).toFixed(3) + '%' : 'ruin'}${isFinite(gx2) && gx2 < 0 ? ' — a shrinking account despite a positive edge' : ''}. Edge per trade ${pf((pw * W - pl * Lz - pt * T) * 100, 2)}% of the stake.`;
  },
  dsr() {
    const sr = V('dsS'), yrs = V('dsY'), N = Math.max(1, V('dsN')), sk = V('dsK'), ku = V('dsU'), n = Math.round(252 * yrs), s = sr / Math.sqrt(252);
    const se = Math.sqrt(Math.max(1e-9, (1 - sk * s + (ku - 1) / 4 * s * s) / (n - 1))), g = .5772156649;
    const e0 = N < 2 ? 0 : se * ((1 - g) * ninv(1 - 1 / N) + g * ninv(1 - 1 / (N * Math.E))), dsr = Nc((s - e0) / se), psr = Nc(s / se);
    $('dsO').innerHTML = `Sharpe ${sr.toFixed(2)} over ${yrs} yr (${n} days). Probability it beats zero, ignoring selection: <b>${(psr * 100).toFixed(1)}%</b>. Best Sharpe expected from ${N} worthless trials: <b>${(e0 * Math.sqrt(252)).toFixed(2)}</b>. Deflated: <b class="${dsr > .95 ? 'pos' : 'neg'}">${(dsr * 100).toFixed(1)}%</b> ${dsr > .95 ? '— survives the selection haircut.' : '— not distinguishable from the luckiest of your trials. Collect more data or test fewer variants.'}`;
  },
  poi() {
    const C = ['Client', 'Pro', 'FII', 'DII'], r = C.map((c, i) => { const g = k => V(`po${i}${k}`); return { c, cl: g(0), cs: g(1), pl: g(2), ps: g(3) } });
    const tot = r.reduce((a, x) => a + x.cs + x.ps, 0) || 1;
    $('poO').innerHTML = '<div class=tw><table><tr><th>Category</th><th class=r>Net calls</th><th class=r>Net puts</th><th class=r>Net options (gamma sign)</th><th class=r>Delta proxy (calls − puts)</th><th class=r>Share of all shorts</th></tr>' +
      r.map(x => { const nc = x.cl - x.cs, np2 = x.pl - x.ps, no = nc + np2; return `<tr><td>${x.c}</td><td class="r ${sc(nc)}">${nc.toLocaleString('en-IN')}</td><td class="r ${sc(np2)}">${np2.toLocaleString('en-IN')}</td><td class="r ${sc(no)}">${no.toLocaleString('en-IN')} ${no > 0 ? '(long Γ)' : no < 0 ? '(short Γ)' : ''}</td><td class="r ${sc(nc - np2)}">${(nc - np2).toLocaleString('en-IN')}</td><td class=r>${((x.cs + x.ps) / tot * 100).toFixed(0)}%</td></tr>` }).join('') + '</table></div>';
  },
  sprt() {
    const mu = V('spM'), sd = V('spS'), a = V('spA') / 100, b = V('spB') / 100, A = Math.log((1 - b) / a), B = Math.log(b / (1 - a)), k = mu / (sd * sd);
    const xs = $('spX').value.split(/[\s,]+/).filter(Boolean).map(Number).filter(x => !isNaN(x));
    let llr = 0, at = null, verdict = 'undecided: keep trading at normal size';
    for (let i = 0; i < xs.length; i++) { llr += k * (xs[i] - mu / 2); if (llr <= B) { at = i + 1; verdict = '<b class=neg>edge gone (accept H0: mean 0)</b>: stop and re-test'; break } if (llr >= A) { at = i + 1; verdict = '<b class=pos>edge confirmed (accept H1)</b>'; break } }
    $('spO').innerHTML = `Log-likelihood ratio ${llr.toFixed(2)} after ${at || xs.length} trades; bounds ${B.toFixed(2)} / ${A.toFixed(2)}. Verdict: ${verdict}. Expected trades to a decision when the edge is real ≈ ${Math.ceil(A / (mu * mu / (2 * sd * sd)))}.`;
  }
};
const OUT = { fwd: 'fvO', corr: 'icO', har: 'hrO', kelly: 'kO', dsr: 'dsO', poi: 'poO', sprt: 'spO' };
function proCalc() { for (const k in CALC) try { if ($(OUT[k])) CALC[k]() } catch (e) { console.error(k, e) } }

/* ---------- P15: your market, measured ---------- */
const f1 = (v, n = 1, u = '') => v == null || !isFinite(v) ? '—' : (+v).toFixed(n) + u;
const pc = (v, n = 1) => v == null || !isFinite(v) ? '—' : (v * 100).toFixed(n) + '%';
const conf = (n, good, thin, unit = '') => `<span class="pill ${n >= good ? 'ok' : n >= thin ? 'thin' : 'few'}">n = ${n ?? 0}${unit ? ' ' + unit : ''}${n >= good ? '' : n >= thin ? ', thin' : ', too few to act on'}</span>`;
const tbl = (head, rows) => { const rt = head.map(h => h[0] == '>'); return `<div class=tw><table><tr>${head.map((h, i) => `<th${rt[i] ? ' class=r' : ''}>${h.replace(/^>/, '')}</th>`).join('')}</tr>${rows.map(r => `<tr>${r.map((c, i) => `<td${rt[i] ? ' class=r' : ''}>${c ?? '—'}</td>`).join('')}</tr>`).join('')}</table></div>` };
const put = (id, h) => { const e = $(id); if (e) e.innerHTML = h };
function renderCalib() {
  if (!$('calSrc')) return;
  if (!CAL) { put('calSrc', '<div class="box warn"><b>No calibration file yet</b>Run <code>python3 pro_calibrate.py</code> in the 1OPTIONS folder, then reload. It reads only what the server already recorded and writes <code>guide/pro_calib.js</code>.</div>'); return }
  const S = CAL.sources || {};
  put('calSrc', `<p class=mu>Generated ${CAL.generated} for ${CAL.sym}. Re-run <code>python3 pro_calibrate.py</code> after more days are recorded; every number below updates.</p>` +
    tbl(['Source', '>Days', 'Range', 'Feeds'], [['1-minute index bars', S.minute_days, (S.minute_range || []).join(' → '), 'clock, weekday, day type, pinning'],
      ['Daily index + India VIX (bhavcopy)', S.daily_days, (S.daily_range || []).join(' → '), 'VRP, HAR, empirical tails'],
      ['Chain snapshots (replay)', S.replay_days, (S.replay_range || []).join(' → '), 'surface, spot-vol, IV clock, OI, book shift, markouts']]) +
    (Object.keys(CAL.errors || {}).length ? `<div class="box warn"><b>Studies that failed</b>${Object.entries(CAL.errors).map(([k, v]) => `${k}: ${v}`).join('<br>')}</div>` : ''));
  const c = CAL.clock;
  if (c) {
    const bars = c.u15.map((y, i) => ({ x: 9.25 + i * .25 + .125, y }));
    put('calClock', `<div class=fig>${plot([{ x: [9.25, 15.5], y: [1, 1], k: 3, dash: 1, lab: 'flat clock' }], { bars, x0: 9.25, x1: 15.5, fx: hm, yl: 'variance intensity per minute (1 = flat)', xl: 'time (IST)', h: 210, al: 'variance clock' })}</div>` +
      `<div class=cap>Your intraday variance clock ${conf(c.n_days, 60, 20, 'days')}: mean squared 1-minute index return by 15-minute bucket, divided by the session average. ${c.extrapolated_from ? `The index feed in your files stops updating near ${c.feed_freeze_from || c.extrapolated_from} and the last bar jumps to the official close, so minutes after ${c.extrapolated_from} are extrapolated from the 20 live minutes before.` : ''}</div>` +
      tbl(['Measure', '>Your data', '>Flat clock', 'Use'], [
        ['First 15 minutes, share of session variance', pc(c.first_15_share), pc(15 / 375), 'Do not judge a day by its first 15 minutes; they are the noisiest of the day'],
        ['First hour', pc(c.first_hour_share), pc(60 / 375), 'Theta sold at 09:15 pays for this hour first'],
        ['11:45–13:30 (lunch)', pc(c.lunch_share), pc(105 / 375), 'Quietest stretch: the cheapest time to buy gamma, the worst to own it'],
        ['Last 30 minutes (settlement window)', pc(c.last_30_share), pc(30 / 375), 'Counts one-third toward expiry settlement'],
        ['Overnight share of daily variance', pc(c.overnight_share), '62% (calendar clock)', 'Calendar clocks over-weight nights, so calendar IV drifts down through the session and jumps at the open'],
        ['Weekday night, in sessions', f1(c.overnight_w, 2), '—', `${c.n_overnight} nights`],
        ['Weekend, in sessions', f1(c.weekend_w, 2), '—', `${c.n_weekend} weekends: ${c.weekend_w != null && c.overnight_w != null ? (c.weekend_w < c.overnight_w * 1.3 ? 'a weekend carries no more variance than a normal night, so most "weekend theta" is free' : 'weekends carry extra variance: keep paying for them') : ''}`],
        ['Session RV (annualised, no nights)', f1(c.session_vol_ann, 1, '%'), '—', 'Compare with ATM IV on a trading clock, not a calendar clock'],
        ['Official close vs last live print', c.close_jump && c.close_jump.mean_abs_bps != null ? `${f1(c.close_jump.mean_abs_bps)} bps avg (σ ${f1(c.close_jump.sd_bps)})` : '—', '0', 'The settlement average at work: expiring strikes settle up to this far from the last tick you saw']]));
  }
  const v = CAL.ver;
  if (v && v.rank_corr != null) put('calVer', `<p>Variance-elapsed ratio at 11:00 against the full day, ${conf(v.n_days, 60, 25, 'days')}. Rank correlation <b>${f1(v.rank_corr, 2)}</b>: the first 105 minutes, measured on your clock against the VIX-implied day, already sort the day.</p>` +
    tbl(['VER at 11:00 (÷ your median day)', '>Mean full-day VER (÷ median)', 'Read'], [
      [`below ${f1(v.ver11_cut_rel[0], 2)}`, f1(v.close_rel_by_tercile[0], 2) + '×', `Quiet day: ${pc(v.bottom_tercile_days_below_median, 0)} ended below median. Sellers' day; fade extremes.`],
      [`${f1(v.ver11_cut_rel[0], 2)}–${f1(v.ver11_cut_rel[1], 2)}`, f1(v.close_rel_by_tercile[1], 2) + '×', 'Normal: no edge from the ratio alone.'],
      [`above ${f1(v.ver11_cut_rel[1], 2)}`, f1(v.close_rel_by_tercile[2], 2) + '×', `Realising day: ${pc(v.top_tercile_days_above_median, 0)} ended above median. No new short gamma; trend playbooks.`]]) +
    `<p class=cap>Your median full-day VER is ${f1(v.median_close, 2)}, not 1, because VIX also prices nights and a risk premium. Read VER relative to that median.</p>`);
  const w = CAL.weekday;
  if (w) put('calWk', tbl(['Day', '>n', '>Session RV', '>15:00–close range (bps)'], Object.entries(w).map(([d, x]) => [d, x.n, f1(x.rv, 1, '%'), f1(x.close_range_bps, 1)])) +
    '<p class=cap>Thursday (the Sensex weekly) and Tuesday (the Nifty weekly) are the days to compare with the rest. Differences of a vol point or two are noise at these sample sizes.</p>');
  const ic = CAL.ivclock;
  if (ic && ic.calendar) {
    const xs = ic.calendar.map((_, i) => 9.25 + i * .25 + .125), ok = a => a.map(y => y == null ? 0 : y);
    put('calIvc', `<div class=fig>${plot([{ x: xs, y: ok(ic.calendar), k: 2, lab: 'ATM IV, calendar clock' }, { x: xs, y: ok(ic.trading), k: 1, lab: 'ATM IV, your trading clock' }], { zero: 1, x0: 9.25, x1: 15.5, fx: hm, yl: 'vol pts vs the day\'s average', xl: 'time (IST)', h: 200, al: 'IV clock artifact' })}</div>` +
      `<div class=cap>Average intraday path of the front weekly's ATM IV ${conf(ic.n_days, 20, 8, 'days')}. Where the two lines separate, the move is the clock; where they move together, the market repriced. ${ic.n_days < 8 ? 'With this few days a single trending afternoon dominates the shape.' : ''}</div>`);
  }
  const s = SURF;
  if (s) {
    const C = s.curve, P = s.points, kk = C.K;
    const lines = [{ x: kk, y: C.svi, k: 1, lab: `SVI (rmse ${f1(s.svi_rmse, 2)})` }, { x: kk, y: C.sabr, k: 2, lab: `SABR β=1 (rmse ${f1(s.sabr_rmse, 2)})` }];
    const mi = C.merton.map((y, i) => [kk[i], y]).filter(p => p[1] != null);
    if (mi.length > 5) lines.push({ x: mi.map(p => p[0]), y: mi.map(p => p[1]), k: 3, lab: `Merton (rmse ${f1(s.merton_rmse, 2)})` });
    lines.push({ x: P.K, y: P.iv, k: 4, dash: 1, lab: `market OTM mids (${s.n})` });
    const m = s.merton, b = s.bkm || {}, ev = CAL.events || {};
    put('calSurf', `<div class=fig>${plot(lines, { vx: [s.F], fx: v => v.toFixed(0), yl: 'implied vol % (Black-76 on the parity forward)', xl: 'strike', al: 'fitted smile' })}</div>` +
      `<div class=cap>Front weekly at ${s.t}, expiry ${s.expiry}, ${f1(s.dte_trading, 2)} sessions left. Forward ${f1(s.F, 1)} from put–call parity (spot ${f1(s.spot, 1)}).</div>` +
      tbl(['Measure', '>Value', 'Read'], [
        ['ATM IV, calendar clock / your trading clock', `${f1(s.atm_iv, 2)}% / ${f1(ev.iv_trading_clock, 2)}%`, 'The second is comparable across days of the week and times of day'],
        ['India VIX', f1(s.vix, 2), ''],
        ['25Δ risk reversal (call − put)', f1(s.rr25, 2), s.rr25 > 0 ? 'Calls over puts: upside is being bought (short-covering or event call demand)' : 'Puts over calls: the usual Nifty fear premium'],
        ['25Δ butterfly', f1(s.bf25, 2), 'Price of curvature (volga): what wing sellers collect'],
        ['ATM skew slope dσ/dk (vol pts per 1%)', f1(s.skew_slope, 2), Math.abs(s.skew_slope) < .2 ? 'Near-symmetric at the money: a smile, not a smirk (normal for 1–3 day options)' : s.skew_slope < 0 ? 'Smirk: downside strikes richer' : 'Upside strikes richer near ATM'],
        ['Implied skewness / kurtosis (BKM)', `${f1(b.skew, 2)} / ${f1(b.kurt, 1)}`, 'A lognormal gives 0 / 3. The kurtosis is the market\'s price of a gap'],
        ['Put–call IV gap within 1% of F', f1(s.pc_gap, 2) + ' vol', 'Error bar of this surface (stale quotes, forward error)'],
        ['SVI butterfly-arbitrage check', s.butterfly_arb_free ? 'pass' : `<b class=neg>fails (min g ${f1(s.min_g, 3)})</b>`, 'Density non-negative across the fitted range'],
        ['SABR (α, ρ, ν)', `${f1(s.sabr.alpha * 100, 1)}%, ${f1(s.sabr.rho, 2)}, ${f1(s.sabr.nu, 2)}`, 'ν = vol of vol the smile implies; ρ = spot-vol correlation it implies'],
        ['Merton jumps (σ, λ/yr, mean, size)', `${f1(m.sigma * 100, 1)}%, ${f1(m.lam, 1)}, ${pf(m.muJ * 100, 2)}%, ±${f1(m.deltaJ * 100, 2)}%`, `The chain prices about ${f1(m.lam * s.T, 3)} jumps before expiry, each ±${f1(m.deltaJ * 100, 1)}% (≈ ${Math.round(m.deltaJ * s.F)} pts)`]]));
  }
  const ss = CAL.surface_series;
  if (ss && ss.length) put('calSer', tbl(['Snapshot', '>Spot', '>ATM IV', '>VIX', '>RR25', '>BF25', '>Slope', '>BKM skew', '>Kurt', '>SVI rmse', 'Arb-free'],
    ss.map(x => [x.t, f1(x.spot, 0), f1(x.atm_iv, 2), f1(x.vix, 2), f1(x.rr25, 2), f1(x.bf25, 2), f1(x.skew_slope, 2), f1(x.bkm_skew, 2), f1(x.bkm_kurt, 1), f1(x.svi_rmse, 2), x.butterfly_arb_free ? 'yes' : 'no'])));
  const sv = CAL.spotvol;
  if (sv) {
    const rd = r => r == null ? '—' : r < 0 ? 'opposite sign to the skew slope' : r < .5 ? 'closer to sticky delta' : r < 1.5 ? 'about sticky strike' : 'local-vol-like or steeper';
    put('calSv', tbl(['Move', '>β (vol pts per 1%)', '>÷ skew slope', 'Regime', '>n (5-min steps)'], [
      ['Down moves', f1(sv.beta_down, 2), f1(sv.ratio_down, 2), rd(sv.ratio_down), sv.n_down], ['Up moves', f1(sv.beta_up, 2), f1(sv.ratio_up, 2), rd(sv.ratio_up), sv.n_up]]) +
      `<p class=cap>β: change in the trading-clock ATM IV per +1% in the forward, regressed on 5-minute steps ${conf(sv.n_days, 20, 8, 'days')}. A negative β on down moves means IV rises as spot falls. Ratio 0 = sticky delta, 1 = sticky strike, about 2 = local vol. A ratio with an odd sign means the skew slope and the realised spot-vol move disagree; on a near-symmetric weekly smile it carries little information. Intraday β on small moves does not extrapolate to gaps, so the stress grid keeps its crash coupling.</p>`);
  }
  const e = CAL.events;
  if (e) put('calEv', tbl(['Measure', '>Value'], [['Front weekly total variance', f1(e.weekly_total_var * 1e4, 3) + ' ×10⁻⁴'], ['Variance units to expiry (your clock)', f1(e.units_to_expiry, 2)],
    ['Base vol from VIX, trading clock', f1(e.base_vol_trading_clock, 2, '%')], ['Excess (event) variance', f1(e.lump_var * 1e4, 3) + ' ×10⁻⁴'],
    ['Implied event move', e.event_move_pct != null ? `±${f1(e.event_move_pct, 2)}% ≈ ±${Math.round(e.event_move_pts)} pts` : 'none (weekly at or below base)'],
    ['Events listed in guide/events.txt before expiry', (e.events_listed || []).map(x => `${x.date} ${x.label}`).join('<br>') || 'none listed']]) +
    `<p class=cap>${e.note}. Compare the implied event move with the average absolute move on past occurrences of the same event. If you listed nothing and the excess is large, the market is pricing something you have not named, which is information in itself.</p>`);
  const t = CAL.tails;
  if (t) put('calTail', tbl(['Move by expiry', '>Your surface (RN)', '>Merton fit', '>Lognormal at ATM IV', '>Empirical', '>RN ÷ empirical'], t.rows.map(r => [pf(r.move, 0) + '%', pc(r.rn_svi, 2), pc(r.merton, 2), pc(r.lognormal_atm, 2), pc(r.empirical, 1), r.empirical ? f1(r.rn_svi / r.empirical, 1) + '×' : '∞'])) +
    `<p class=cap>Horizon ${t.horizon_sessions} session(s), the fitted weekly's remaining life. Empirical = share of overlapping ${t.horizon_sessions}-session windows in your daily history with that move ${conf(t.n_windows, 500, 120, 'windows')}. RN far above empirical is the tail premium put and call sellers collect; with a short history, the empirical column has not yet seen a real gap, so treat ∞ as "unmeasured", not "free".</p>`);
  const vr = CAL.vrp, h = CAL.har;
  put('calVrp', (vr && vr.mean_vrp != null ? tbl(['VIX vs next 21 sessions\' realised', '>Value'], [['Mean VIX', f1(vr.mean_vix, 2)], ['Mean realised (close-to-close)', f1(vr.mean_rv, 2)], ['Mean VRP (vol pts)', f1(vr.mean_vrp, 2)], ['Share of days VIX > later RV', pc(vr.share_positive, 0)], ['VIX² ÷ RV²', f1(vr.var_ratio, 2)]]) + `<p class=cap>${conf(vr.n, 500, 60, 'windows')} ${vr.note}.</p>` : '') +
    (h && h.c != null ? tbl(['HAR-RV fit (daily proxy)', '>Value'], [['c, β_d, β_w, β_m', `${f1(h.c, 2)}, ${f1(h.bd, 2)}, ${f1(h.bw, 2)}, ${f1(h.bm, 2)}`], ['R² in sample', f1(h.r2, 2)], ['QLIKE: HAR vs 22-day average', `${f1(h.qlike, 3)} vs ${f1(h.qlike_naive_m, 3)}`], ['Forecast RV / current VIX', `${f1(h.forecast, 1)}% / ${f1(h.last.iv, 1)}%`], ['Forecast RMSE (vol pts)', f1(h.rmse, 2)]]) +
      `<p class=cap>${conf(h.n, 500, 60, 'days')} Fitted on ${h.proxy}, which is noisy; the coefficients are loaded into the HAR tool in P9. ${h.forecast != null && h.last.iv != null ? (h.last.iv - h.forecast > h.rmse ? 'VIX exceeds the forecast by more than its error: premium is rich.' : 'VIX − forecast is inside the forecast error: no VRP signal today.') : ''}</p>` : ''));
  const bk = CAL.book_shift;
  if (bk && bk.by_h) put('calBook', tbl(['Horizon (snapshots ≈ minutes)', '>IC', '>Balanced hit %', '>n'], Object.entries(bk.by_h).map(([k, x]) => [k, f1(x.ic, 3), f1(x.balanced, 1), x.n])) +
    `<p class=cap>The grid's book-shift read (<code>_bfZ</code>, via <code>bias_eval.py</code>) against the index ${conf(bk.n_days, 40, 15, 'days')}. Peak at ${bk.peak_h} snapshot(s)${bk.half_life_h ? `, halves by ${bk.half_life_h}` : ''}: act on it within that window or not at all.</p>`);
  const ag = CAL.aggressor;
  if (ag) put('calOi', tbl(['Reading (ATM ±5 strikes, 5-snapshot window)', '>IC with next 15-min futures move'], [['Aggressor side, weighted by |ΔOI|', f1(ag.ic_aggressor, 3)], ['Price/OI quadrant (retail reading)', f1(ag.ic_price_quadrant, 3)], ['Writers\' reading (call OI up = resistance)', f1(ag.ic_writer_reading, 3)]]) +
    `<p class=cap>${conf(ag.n_days, 40, 15, 'days')}, ${ag.n} windows (overlapping). |IC| below about 0.05 is noise at this size. A negative IC means the reading worked as a contrarian signal in this sample.</p>`);
  const mk = CAL.markouts;
  if (mk) put('calMk', mk.n_matched ? tbl(['After fill', '>Mean markout (pts, + = in your favour)', '>n'], Object.entries(mk.markout_pts).map(([k, x]) => [k == '0' ? 'at fill (vs mid)' : `+${k} min`, f1(x, 2), mk.n_by_h[k]])) +
    `<p class=cap>${mk.n_matched} of ${mk.n_fills_total} fills matched to a chain snapshot; modes: ${mk.modes.join(', ')}. ${mk.modes.includes('paper') ? 'Paper fills are simulated, so they cannot show real adverse selection; run this again on live fills.' : ''}</p>` : '<p class=mu>No fills overlap the recorded chain days yet.</p>');
  const pn = CAL.pinning;
  if (pn) put('calPin', pn.n_expiries ? tbl(['Expiry', '>Max-OI strike (prior close)', '>Distance 12:30', '>Distance 15:29', 'Pinned?'], pn.rows.map(r => [r.expiry, f1(r.max_oi_strike, 0), f1(r.dist_1230, 0), f1(r.dist_1529, 0), r.dist_1529 < r.dist_1230 ? 'closer' : 'further'])) + `<p class=cap>${conf(pn.n_expiries, 30, 10, 'expiries')}</p>` : '<p class=mu>No expiry day yet has both a prior-close OI file and minute bars.</p>');
  const pa = CAL.participant;
  if (pa && pa.series && pa.series.length) put('calPart', tbl(['Date', '>Client', '>Pro', '>FII', '>DII', '>FII index-futures long %'], pa.series.slice(-12).map(r => [r.date, ...['Client', 'Pro', 'FII', 'DII'].map(k => `<span class=${sc(r[k])}>${Math.round(r[k]).toLocaleString('en-IN')}</span>`), pc(r.fii_fut_long_ratio, 0)])) +
    `<p class=cap>Net index options (long − short, calls + puts) per category: negative = net short options = short gamma. The P7 tool is pre-filled with ${pa.latest_date}. ${pa.series.length && pa.series.at(-1).fii_fut_long_ratio < .2 ? 'FII index-futures long ratio below 20%: short-covering fuel is stored.' : ''}</p>`);
}

/* ---------- wiring ---------- */
const _baseRender = renderAll;
renderAll = function () {
  _baseRender();
  const run = (f, ...a) => { try { f(...a) } catch (e) { console.error(f.name, e) } };
  document.querySelectorAll('[data-gf]').forEach(e => run(fingerprint, e));
  document.querySelectorAll('[data-rnd]').forEach(e => run(rndTable, e));
  document.querySelectorAll('[data-odte]').forEach(e => run(odteTable, e));
  document.querySelectorAll('[data-clk]').forEach(e => run(clockTable, e));
  run(stress); run(explain); run(hedgeSim); run(proCalc); run(renderCalib);
};
document.addEventListener('DOMContentLoaded', () => {
  const setv = (id, v) => { const e = $(id); if (e && v != null && isFinite(v)) e.value = v };
  if (CAL) {
    const h = CAL.har;
    if (h && h.c != null) { setv('hrC', +h.c.toFixed(2)); setv('hrBd', +h.bd.toFixed(2)); setv('hrBw', +h.bw.toFixed(2)); setv('hrBm', +h.bm.toFixed(2)); setv('hrD', +h.last.d.toFixed(1)); setv('hrW', +h.last.w.toFixed(1)); setv('hrM', +h.last.m.toFixed(1)); setv('hrI', h.last.iv) }
    const pa = CAL.participant && CAL.participant.latest;
    if (pa) ['Client', 'Pro', 'FII', 'DII'].forEach((c, i) => (pa[c] || []).forEach((v, k) => setv(`po${i}${k}`, Math.round(v))));
    if (SURF) { setv('gS', Math.round(SURF.F)); setv('gV', +SURF.atm_iv.toFixed(1)) }
    document.querySelectorAll('[data-calpill]').forEach(e => {
      const k = e.dataset.calpill, c = CAL[k];
      if (k == 'clock' && c) e.textContent = `your clock · ${c.n_days} days`;
      if (k == 'har' && c && c.c != null) e.textContent = `fitted on your data · n=${c.n}`;
      if (k == 'model' && SURF) e.textContent = `your surface · ${SURF.t}`;
      if (c || (k == 'model' && SURF)) e.classList.add('ok');
    });
  }
  const gm = $('gM');
  if (gm) { if (!SURF) { gm.value = 'toy'; gm.options[1].disabled = true } gm.onchange = () => { MODEL = gm.value; renderAll() } }
  fillSel('sgP', 0); fillSel('peP', 2);
  linkSel('sgP', 'sgL', stress); linkSel('peP', 'peL', explain);
  const F = { sg: stress, pe: explain, hs: hedgeSim };
  document.querySelectorAll('[data-k]').forEach(i => i.addEventListener(i.type == 'checkbox' ? 'change' : 'input', () => { const k = i.dataset.k; try { (F[k] || CALC[k])() } catch (e) { console.error(k, e) } }));
  if (CAL) setG(); else { stress(); explain() }
}, { once: true });
