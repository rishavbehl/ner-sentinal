/* ==========================================================================
   NER Logistics Sentinel — dashboard
   Pure vanilla JS + hand-rolled SVG map engine. No Leaflet, no CDN, no tiles:
   the whole dashboard runs with zero network access beyond this server, so the
   demo cannot be broken by venue wifi.
   ========================================================================== */
'use strict';

const SVGNS = 'http://www.w3.org/2000/svg';
const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const el = (t, a = {}, p) => { const n = document.createElementNS(SVGNS, t);
  for (const k in a) n.setAttribute(k, a[k]); if (p) p.appendChild(n); return n; };
const fmt  = (n, d = 1) => (n === null || n === undefined || isNaN(n)) ? '—' : Number(n).toFixed(d);
const esc  = s => String(s ?? '').replace(/[&<>"]/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/* ---------------- state ---------------- */
const S = {
  nodes: [], nodeById: {}, segments: [], segById: {},
  summary: null, heat: null, crit: null, acc: null, metrics: null,
  routes: [], activeRoute: 0, selectedSeg: null, incidents: [],
  fleet: {}, fleetAlerts: [], fleetPaused: false, es: null, liveGps: [],
  profile: 'balanced', overrides: null, whatifStates: new Set(),
  layers: { seg: 1, heat: 0, crit: 0, acc: 0, fleet: 1, inc: 0, lbl: 1, hull: 1 },
  langs: {}, lang: ''
};

/* ---------------- colour ---------------- */
function sevColor(s, label) {
  if (label === 2) return '#ef4444';
  if (s < 0.12) return '#2dd4a7';
  if (s < 0.25) return '#a3e635';
  if (s < 0.40) return '#fbbf24';
  if (s < 0.60) return '#fb923c';
  return '#ef4444';
}
const gradeCls = g => ['A', 'B', 'C', 'D', 'E', 'F'].includes(g) ? g : 'F';

/* ==========================================================================
   MAP ENGINE
   ========================================================================== */
const MAP = {
  svg: null, cam: null, W: 1500, H: 1180,
  bounds: null, scale: 1, tx: 0, ty: 0, dragging: false, lastFit: null,
  layers: {},

  init() {
    this.svg = $('#map');
    this.svg.setAttribute('viewBox', `0 0 ${this.W} ${this.H}`);
    this.svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

    const defs = el('defs', {}, this.svg);
    defs.innerHTML =
      `<filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
         <feGaussianBlur stdDeviation="4" result="b"/>
         <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
       </filter>
       <radialGradient id="heatg"><stop offset="0%" stop-opacity=".85"/>
         <stop offset="100%" stop-opacity="0"/></radialGradient>`;

    this.cam = el('g', { id: 'cam' }, this.svg);
    for (const n of ['hull', 'grat', 'heat', 'seg', 'crit', 'route',
                     'node', 'inc', 'fleet', 'lbl'])
      this.layers[n] = el('g', { id: 'L-' + n }, this.cam);

    /* pan / zoom.
       NOTE: capture is taken only AFTER the pointer has actually moved past a
       small threshold. Capturing on pointerdown retargets the following click
       to the <svg>, which silently kills every click handler on the segments
       underneath — a genuinely nasty bug to find. The threshold also means a
       drag never fires a segment click. */
    const DRAG_PX = 3;
    let sx = 0, sy = 0, otx = 0, oty = 0, down = false, pid = null;
    this.movedFar = false;

    this.svg.addEventListener('pointerdown', e => {
      down = true; pid = e.pointerId; this.movedFar = false;
      sx = e.clientX; sy = e.clientY; otx = this.tx; oty = this.ty;
    });
    this.svg.addEventListener('pointermove', e => {
      if (!down) return;
      const dx = e.clientX - sx, dy = e.clientY - sy;
      if (!this.movedFar && Math.hypot(dx, dy) < DRAG_PX) return;
      if (!this.movedFar) {
        this.movedFar = true; this.dragging = true;
        this.svg.classList.add('drag');
        try { this.svg.setPointerCapture(pid); } catch (_) {}
      }
      const r = this.svg.getBoundingClientRect();
      const k = this.W / r.width;
      this.tx = otx + dx * k;
      this.ty = oty + dy * k;
      this.apply();
    });
    const end = () => {
      down = false; this.dragging = false;
      this.svg.classList.remove('drag');
      if (pid !== null) { try { this.svg.releasePointerCapture(pid); } catch (_) {} }
      pid = null;
      // let the click that follows pointerup read movedFar, then reset
      setTimeout(() => { this.movedFar = false; }, 0);
    };
    this.svg.addEventListener('pointerup', end);
    this.svg.addEventListener('pointercancel', end);

    this.svg.addEventListener('wheel', e => {
      e.preventDefault();
      const r = this.svg.getBoundingClientRect();
      const k = this.W / r.width;
      const mx = (e.clientX - r.left) * k, my = (e.clientY - r.top) * k;
      const f = e.deltaY < 0 ? 1.16 : 1 / 1.16;
      const ns = Math.max(0.55, Math.min(14, this.scale * f));
      const ratio = ns / this.scale;
      this.tx = mx - (mx - this.tx) * ratio;
      this.ty = my - (my - this.ty) * ratio;
      this.scale = ns; this.apply();
    }, { passive: false });

    $('#zin').onclick  = () => this.zoomBy(1.35);
    $('#zout').onclick = () => this.zoomBy(1 / 1.35);
    $('#zfit').onclick = () => this.fit();
  },

  zoomBy(f) {
    const cx = this.W / 2, cy = this.H / 2;
    const ns = Math.max(0.55, Math.min(14, this.scale * f));
    const r = ns / this.scale;
    this.tx = cx - (cx - this.tx) * r;
    this.ty = cy - (cy - this.ty) * r;
    this.scale = ns; this.apply();
  },

  apply() {
    this.cam.setAttribute('transform',
      `translate(${this.tx} ${this.ty}) scale(${this.scale})`);
    const inv = 1 / this.scale;
    $$('#L-lbl text').forEach(t => {
      t.style.fontSize = (t.dataset.fs || 9) * Math.min(inv, 1.7) + 'px';
      t.style.display = this.scale < 1.25 && t.dataset.minor === '1' ? 'none' : '';
    });
    $$('#L-fleet .trucklbl').forEach(t => {
      t.style.fontSize = 8.5 * Math.min(inv, 1.8) + 'px';
    });
    this.scalebar();
    if (this._lastLblScale === undefined) this._lastLblScale = this.scale;
    const crossed = (this._lastLblScale < 1.8) !== (this.scale < 1.8);
    this._lastLblScale = this.scale;
    if (crossed && typeof drawFleet === 'function') drawFleet();
  },

  fit(pad = 52) {
    this.scale = 1; this.tx = 0; this.ty = 0; this.apply();
  },

  /* --- projection: Web Mercator fitted to the node bounding box --- */
  setBounds(nodes) {
    const lats = nodes.map(n => n.lat), lons = nodes.map(n => n.lon);
    /* Web Mercator northing. The 180/π factor is essential: x is carried in
       DEGREES of longitude, so y must be expressed in the same angular unit or
       the whole map collapses to a horizontal line. */
    const merc = l => (180 / Math.PI) *
      Math.log(Math.tan(Math.PI / 4 + (l * Math.PI / 180) / 2));
    const b = {
      minLon: Math.min(...lons), maxLon: Math.max(...lons),
      minLat: Math.min(...lats), maxLat: Math.max(...lats)
    };
    b.minY = merc(b.minLat); b.maxY = merc(b.maxLat);
    const pad = 0.055;
    const lonSpan = (b.maxLon - b.minLon), ySpan = (b.maxY - b.minY);
    b.minLon -= lonSpan * pad; b.maxLon += lonSpan * pad;
    b.minY  -= ySpan * pad;    b.maxY  += ySpan * pad;
    const aw = this.W * 0.96, ah = this.H * 0.96;
    const sx = aw / (b.maxLon - b.minLon), sy = ah / (b.maxY - b.minY);
    b.k = Math.min(sx, sy);
    b.ox = (this.W - (b.maxLon - b.minLon) * b.k) / 2;
    b.oy = (this.H - (b.maxY - b.minY) * b.k) / 2;
    b.merc = merc;
    this.bounds = b;
  },

  p(lat, lon) {
    const b = this.bounds;
    return [b.ox + (lon - b.minLon) * b.k,
            b.oy + (b.maxY - b.merc(lat)) * b.k];
  },

  kmPerUnit() {   // at the centre latitude
    const b = this.bounds;
    const latC = (b.minLat + b.maxLat) / 2;
    const a = this.p(latC, b.minLon), c = this.p(latC, b.minLon + 1);
    const kmPerDeg = 111.32 * Math.cos(latC * Math.PI / 180);
    return kmPerDeg / (c[0] - a[0]);
  },

  scalebar() {
    if (!this.bounds) return;
    const kmPerUnit = this.kmPerUnit();
    const r = this.svg.getBoundingClientRect();
    const unitsPerPx = (this.W / r.width) / this.scale;
    const kmPerPx = kmPerUnit * unitsPerPx;
    const targets = [10, 25, 50, 100, 200, 400, 800];
    let km = targets.find(t => t / kmPerPx >= 48) || 800;
    const px = Math.round(km / kmPerPx);
    $('#sbar').style.width = Math.min(px, 170) + 'px';
    $('#sbtxt').textContent = km + ' km';
  },

  clear(layer) { const g = this.layers[layer]; while (g.firstChild) g.removeChild(g.firstChild); }
};

/* ==========================================================================
   TOOLTIP
   ========================================================================== */
const TIP = {
  node: null,
  show(html, ev) {
    if (!this.node) this.node = $('#tip');
    this.node.innerHTML = html;
    this.node.classList.add('on');
    const wrap = $('#mapwrap').getBoundingClientRect();
    let x = ev.clientX - wrap.left + 14, y = ev.clientY - wrap.top + 14;
    const r = this.node.getBoundingClientRect();
    if (x + r.width > wrap.width - 8) x = ev.clientX - wrap.left - r.width - 14;
    if (y + r.height > wrap.height - 8) y = ev.clientY - wrap.top - r.height - 14;
    this.node.style.left = Math.max(6, x) + 'px';
    this.node.style.top = Math.max(6, y) + 'px';
  },
  hide() { if (this.node) this.node.classList.remove('on'); }
};

function banner(msg, kind = '', ms = 5200) {
  const b = $('#banner');
  b.textContent = msg;
  b.className = 'on ' + kind;
  clearTimeout(b._t);
  if (ms) b._t = setTimeout(() => b.className = '', ms);
}

/* ==========================================================================
   API
   ========================================================================== */
function ovQuery(extra = {}) {
  const q = new URLSearchParams();
  const o = S.overrides;
  if (o) for (const k in o) {
    if (k === 'states') { if (o.states?.length) q.set('states', o.states.join(',')); }
    else q.set(k, o[k]);
  }
  for (const k in extra) if (extra[k] !== undefined && extra[k] !== '') q.set(k, extra[k]);
  const s = q.toString();
  return s ? '?' + s : '';
}
async function api(path, extra = {}) {
  const r = await fetch('/api' + path + ovQuery(extra));
  const j = await r.json().catch(() => ({ error: 'bad json' }));
  if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
  return j;
}

/* ==========================================================================
   RENDER: state extents + graticule
   ========================================================================== */
const STATE_COLOR = {
  AS: '#38bdf8', ML: '#a78bfa', AR: '#2dd4a7', NL: '#fbbf24',
  MN: '#f472b6', MZ: '#fb923c', TR: '#818cf8', SK: '#34d399', WB: '#94a3b8'
};

function hull(pts) {
  if (pts.length < 3) return pts;
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i];
    while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
// expand a hull outward from its centroid so it reads as an area, not a web
function expand(h, f = 1.14) {
  const cx = h.reduce((s, p) => s + p[0], 0) / h.length;
  const cy = h.reduce((s, p) => s + p[1], 0) / h.length;
  return h.map(p => [cx + (p[0] - cx) * f, cy + (p[1] - cy) * f]);
}

function drawHulls() {
  MAP.clear('hull');
  if (!S.layers.hull) return;
  const byState = {};
  S.nodes.forEach(n => (byState[n.state] ||= []).push(MAP.p(n.lat, n.lon)));
  for (const st in byState) {
    const pts = byState[st];
    if (pts.length < 3) continue;
    const h = expand(hull(pts));
    el('polygon', {
      class: 'statehull', points: h.map(p => p.join(',')).join(' '),
      stroke: (STATE_COLOR[st] || '#6b819a') + '44',
      fill: (STATE_COLOR[st] || '#6b819a') + '0a'
    }, MAP.layers.hull);
    const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
    const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
    const t = el('text', { class: 'statelbl', x: cx, y: cy,
      'text-anchor': 'middle', fill: (STATE_COLOR[st] || '#6b819a') + '55' },
      MAP.layers.hull);
    t.textContent = st;
  }
}

/* ==========================================================================
   RENDER: segments
   ========================================================================== */
function segWidth(s) {
  return s.road_class === 'NH' ? (s.lanes >= 4 ? 4.6 : 3.4)
       : s.road_class === 'SH' ? 2.5 : 1.7;
}

function drawSegments() {
  MAP.clear('seg');
  if (!S.layers.seg) return;
  const g = MAP.layers.seg;
  for (const s of S.segments) {
    const [x1, y1] = MAP.p(s.coords[0][0], s.coords[0][1]);
    const [x2, y2] = MAP.p(s.coords[1][0], s.coords[1][1]);
    const col = sevColor(s.severity, s.risk_label);
    const w = segWidth(s);

    if (s.risk_label === 2) {
      el('line', { x1, y1, x2, y2, class: 'seg', stroke: col,
        'stroke-width': w + 5, opacity: .18, filter: 'url(#glow)',
        'vector-effect': 'non-scaling-stroke' }, g);
    }
    const ln = el('line', {
      x1, y1, x2, y2, stroke: col, 'stroke-width': w,
      class: 'seg' + (s.risk_label === 2 ? ' blockdash' : ''),
      opacity: s.risk_label === 0 ? .82 : .96,
      'vector-effect': 'non-scaling-stroke'
    }, g);
    ln.dataset.rid = s.road_id;

    const hit = el('line', { x1, y1, x2, y2, class: 'seghit',
      'vector-effect': 'non-scaling-stroke' }, g);
    hit.dataset.rid = s.road_id;
    hit.addEventListener('mousemove', ev => TIP.show(segTip(s), ev));
    hit.addEventListener('mouseleave', () => TIP.hide());
    hit.addEventListener('click', ev => {
      if (MAP.movedFar) return;           // that was a pan, not a selection
      ev.stopPropagation(); openSegment(s.road_id);
    });
  }
  if (S.selectedSeg) highlightSeg(S.selectedSeg);
}

function segTip(s) {
  return `<div class="tt">${esc(s.name)}</div>
    <div class="tr"><span>${esc(s.corridor)} · ${esc(s.road_class)}</span>
      <b style="color:${sevColor(s.severity, s.risk_label)}">${s.risk_status.toUpperCase()}</b></div>
    <div class="tr"><span>Severity</span><b>${fmt(s.severity, 2)}</b></div>
    <div class="tr"><span>P(disruption)</span><b>${fmt(s.risk_score * 100, 0)}%</b></div>
    <div class="tr"><span>Predicted delay</span><b>${fmt(s.delay_hours, 1)} h</b></div>
    <div class="tr"><span>Length · terrain</span><b>${fmt(s.length_km, 0)} km · ${esc(s.terrain)}</b></div>
    <div class="tr"><span>Rain 24h / soil</span><b>${fmt(s.weather.rainfall_24h_mm, 0)} / ${fmt(s.weather.api_7d, 0)} mm</b></div>
    ${s.disruption ? '<div class="tr" style="color:#ef4444"><span>Blockade active</span><b>yes</b></div>' : ''}
    <div class="tr" style="margin-top:4px;color:#6b819a"><span>click for full explanation</span></div>`;
}

function highlightSeg(rid) {
  MAP.clear('crit');
  const s = S.segById[rid];
  if (!s) return;
  const [x1, y1] = MAP.p(s.coords[0][0], s.coords[0][1]);
  const [x2, y2] = MAP.p(s.coords[1][0], s.coords[1][1]);
  el('line', { x1, y1, x2, y2, class: 'segsel', 'stroke-width': segWidth(s) + 6,
    'stroke-opacity': .35, 'vector-effect': 'non-scaling-stroke' }, MAP.layers.crit);
}

/* ==========================================================================
   RENDER: nodes + labels
   ========================================================================== */
function drawNodes() {
  MAP.clear('node'); MAP.clear('lbl');
  const KIND = {
    hub:    { r: 5.4, f: '#38bdf8', s: '#0b1b26' },
    port:   { r: 5.0, f: '#a78bfa', s: '#140f26' },
    depot:  { r: 4.4, f: '#34d399', s: '#0a1f18' },
    border: { r: 4.2, f: '#f472b6', s: '#200d18' },
    pass:   { r: 4.8, f: '#fb923c', s: '#231203' },
    town:   { r: 3.0, f: '#7f9bb8', s: '#0a1118' }
  };
  for (const n of S.nodes) {
    const [x, y] = MAP.p(n.lat, n.lon);
    const k = KIND[n.kind] || KIND.town;
    const c = el('circle', { cx: x, cy: y, r: k.r, fill: k.f, stroke: k.s,
      'stroke-width': 1.4, class: 'nodedot' }, MAP.layers.node);
    c.addEventListener('mousemove', ev => TIP.show(
      `<div class="tt">${esc(n.name)}</div>
       <div class="tr"><span>${esc(n.state_name || n.state)}</span><b>${esc(n.kind)}</b></div>
       <div class="tr"><span>Elevation</span><b>${n.elev_m} m</b></div>
       <div class="tr"><span>Population</span><b>${n.population_k}k</b></div>
       <div class="tr" style="margin-top:4px;color:#6b819a"><span>click: set as origin · shift-click: destination</span></div>`, ev));
    c.addEventListener('mouseleave', () => TIP.hide());
    c.addEventListener('click', ev => {
      if (MAP.movedFar) return;
      ev.stopPropagation();
      if (ev.shiftKey) { $('#dest').value = n.id; }
      else { $('#orig').value = n.id; }
      banner(`${ev.shiftKey ? 'Destination' : 'Origin'} set to ${n.name}`, 'info', 2200);
    });

    if (S.layers.lbl) {
      const minor = ['town'].includes(n.kind) && n.population_k < 90 ? '1' : '0';
      const t = el('text', { class: 'nodelbl', x: x + k.r + 3, y: y + 3.2 },
        MAP.layers.lbl);
      t.textContent = n.name.replace(/ \(.*\)$/, '');
      t.dataset.fs = n.kind === 'hub' ? 10.5 : 9;
      t.dataset.minor = minor;
      t.style.fontSize = t.dataset.fs + 'px';
      if (n.kind === 'hub') t.style.fill = '#e8f4ff';
    }
  }
  MAP.apply();
}

/* ==========================================================================
   RENDER: heat / criticality / accessibility overlays
   ========================================================================== */
function drawHeat() {
  MAP.clear('heat');
  if (!S.layers.heat || !S.heat) return;
  for (const d of S.heat.districts) {
    if (d.lat == null) continue;
    const [x, y] = MAP.p(d.lat, d.lon);
    const col = sevColor(d.mean_severity, d.km_blocked > 0 ? 2 : 0);
    const r = 16 + Math.min(d.km / 11, 34);
    const c = el('circle', { cx: x, cy: y, r, fill: col, opacity: .2,
      class: 'heatc' }, MAP.layers.heat);
    el('circle', { cx: x, cy: y, r: r * .45, fill: col, opacity: .3,
      class: 'heatc' }, MAP.layers.heat);
  }
}

function drawCrit() {
  if (!S.layers.crit || !S.crit) { if (!S.selectedSeg) MAP.clear('crit'); return; }
  MAP.clear('crit');
  const top = S.crit.ranked_by_live_exposure.slice(0, 10);
  top.forEach((r, i) => {
    const [x1, y1] = MAP.p(r.coords[0][0], r.coords[0][1]);
    const [x2, y2] = MAP.p(r.coords[1][0], r.coords[1][1]);
    el('line', { x1, y1, x2, y2, stroke: '#f472b6',
      'stroke-width': 9 - i * .45, opacity: .32, filter: 'url(#glow)',
      'stroke-linecap': 'round', 'vector-effect': 'non-scaling-stroke',
      'pointer-events': 'none' }, MAP.layers.crit);
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    const g = el('g', {}, MAP.layers.crit);
    el('circle', { cx: mx, cy: my, r: 8.5, fill: '#f472b6', stroke: '#1b0a14',
      'stroke-width': 1.5 }, g);
    const t = el('text', { x: mx, y: my + 3.2, 'text-anchor': 'middle',
      style: 'font:700 9px var(--mono);fill:#1b0a14;pointer-events:none' }, g);
    t.textContent = i + 1;
    g.addEventListener('mousemove', ev => TIP.show(
      `<div class="tt">#${i + 1} choke point · ${esc(r.corridor)}</div>
       <div class="tr"><span>${esc(r.name)}</span><b>${r.current_risk_status.toUpperCase()}</b></div>
       <div class="tr"><span>Isolates</span><b>${r.isolates_nodes} nodes</b></div>
       <div class="tr"><span>People cut off</span><b>${r.population_cut_off_k.toLocaleString()}k</b></div>
       <div class="tr"><span>Extra network time</span><b>${fmt(r.extra_network_hours, 1)} h</b></div>
       <div class="tr"><span>Live exposure</span><b>${fmt(r.exposure_score, 0)}</b></div>`, ev));
    g.addEventListener('mouseleave', () => TIP.hide());
    g.style.cursor = 'pointer';
    g.addEventListener('click', e => { e.stopPropagation(); openSegment(r.road_id); });
  });
}

function drawAcc() {
  if (!S.layers.acc || !S.acc) return;
  for (const d of S.acc.districts) {
    if (!d.reachable || d.lat == null) continue;
    const [x, y] = MAP.p(d.lat, d.lon);
    const col = { A: '#2dd4a7', B: '#a3e635', C: '#fbbf24', D: '#fb923c',
                  E: '#ef4444', F: '#7f1d1d' }[d.grade] || '#6b819a';
    const g = el('g', {}, MAP.layers.heat);
    el('rect', { x: x - 7, y: y - 14.5, width: 14, height: 12, rx: 3.5,
      fill: col, opacity: .92 }, g);
    const t = el('text', { x, y: y - 5.5, 'text-anchor': 'middle',
      style: 'font:700 9px var(--mono);fill:#06121a;pointer-events:none' }, g);
    t.textContent = d.grade;
  }
}

function drawIncidents() {
  MAP.clear('inc');
  if (!S.layers.inc || !S.incidents.length) return;
  for (const i of S.incidents) {
    const [x, y] = MAP.p(i.lat, i.lon);
    const col = i.severity === 'high' ? '#ef4444'
              : i.severity === 'medium' ? '#fbbf24' : '#a3e635';
    const c = el('circle', { cx: x, cy: y, r: i.severity === 'high' ? 3.6 : 2.6,
      fill: col, opacity: .72, stroke: '#07101a', 'stroke-width': .8 },
      MAP.layers.inc);
    c.style.cursor = 'pointer';
    c.addEventListener('mousemove', ev => TIP.show(
      `<div class="tt">${esc(i.issue_type.replace(/_/g, ' '))}</div>
       <div class="tr"><span>${esc(i.district)}</span><b>${esc(i.severity)}</b></div>
       <div class="tr"><span>Reported</span><b>${esc(i.reported_at.slice(0, 16).replace('T', ' '))}</b></div>
       <div class="tr"><span>Source</span><b>${esc(i.reporter_id)}</b></div>`, ev));
    c.addEventListener('mouseleave', () => TIP.hide());
  }
}

/* ==========================================================================
   RENDER: routes
   ========================================================================== */
function drawRoutes() {
  MAP.clear('route');
  if (!S.routes.length) return;
  const COL = ['#38bdf8', '#a78bfa', '#f472b6'];
  S.routes.forEach((r, i) => {
    if (i !== S.activeRoute && i > 2) return;
    const pts = r.geometry.map(c => MAP.p(c[0], c[1]));
    const d = 'M' + pts.map(p => p.join(' ')).join(' L');
    const on = i === S.activeRoute;
    if (on) el('path', { d, class: 'routeglow', stroke: COL[i % 3],
      'stroke-width': 13, 'vector-effect': 'non-scaling-stroke' }, MAP.layers.route);
    el('path', { d, class: 'routeline', stroke: COL[i % 3],
      'stroke-width': on ? 4.2 : 2.2, opacity: on ? .98 : .42,
      'stroke-dasharray': on ? '' : '7 6',
      'vector-effect': 'non-scaling-stroke' }, MAP.layers.route);
  });
  const best = S.routes[S.activeRoute];
  if (best) {
    [[best.geometry[0], '#2dd4a7', 'O'],
     [best.geometry[best.geometry.length - 1], '#ef4444', 'D']].forEach(([c, col, lbl]) => {
      const [x, y] = MAP.p(c[0], c[1]);
      el('circle', { cx: x, cy: y, r: 8, fill: col, stroke: '#06121a',
        'stroke-width': 2 }, MAP.layers.route);
      const t = el('text', { x, y: y + 3.4, 'text-anchor': 'middle',
        style: 'font:700 9px var(--mono);fill:#06121a;pointer-events:none' },
        MAP.layers.route);
      t.textContent = lbl;
    });
  }
}

/* ==========================================================================
   RENDER: fleet
   ========================================================================== */
function drawFleet() {
  MAP.clear('fleet');
  if (!S.layers.fleet) return;

  // ---- REAL GPS-tracked devices, drawn distinctly from the simulation ----
  for (const v of S.liveGps) {
    if (!v.position) continue;
    const [x, y] = MAP.p(v.position[0], v.position[1]);
    const col = v.in_risk_zone
      ? (v.risk_status === 'blocked' ? '#ef4444' : '#fbbf24') : '#2dd4a7';

    if (v.trail && v.trail.length > 1) {
      const pts = v.trail.map(c => MAP.p(c[0], c[1]));
      el('path', { d: 'M' + pts.map(p => p.join(' ')).join(' L'),
        fill: 'none', stroke: col, 'stroke-width': 2, opacity: .45,
        'stroke-linecap': 'round', 'stroke-dasharray': '4 4',
        'pointer-events': 'none',
        'vector-effect': 'non-scaling-stroke' }, MAP.layers.fleet);
    }
    const g = el('g', { class: 'truck', transform: `translate(${x} ${y})` },
      MAP.layers.fleet);
    el('circle', { cx: 0, cy: 0, r: 13, fill: col, opacity: .2,
      filter: 'url(#glow)' }, g);
    // hexagon = real device, triangle = simulated truck
    el('polygon', { points: '0,-8 7,-4 7,4 0,8 -7,4 -7,-4', fill: col,
      stroke: '#04121a', 'stroke-width': 1.6 }, g);
    el('circle', { cx: 0, cy: 0, r: 2.6, fill: '#04121a' }, g);
    g.addEventListener('mousemove', ev => TIP.show(
      `<div class="tt">📡 ${esc(v.id)} · LIVE GPS</div>
       <div class="tr"><span>${esc(v.label || '')}</span><b>${esc(v.cargo_type || '')}</b></div>
       <div class="tr"><span>Status</span><b style="color:${col}">${esc(v.risk_status || 'off network')}</b></div>
       <div class="tr"><span>Speed</span><b>${v.speed_kmph == null ? '—' : fmt(v.speed_kmph, 0) + ' km/h'}</b></div>
       <div class="tr"><span>Accuracy</span><b>±${fmt(v.accuracy_m, 0)} m</b></div>
       <div class="tr"><span>Last fix</span><b>${fmt(v.age_minutes, 1)} min ago</b></div>
       <div class="tr"><span>Fixes</span><b>${v.points}</b></div>`, ev));
    g.addEventListener('mouseleave', () => TIP.hide());
    g.addEventListener('click', e => {
      e.stopPropagation();
      if (v.road_id) openSegment(v.road_id);
    });

    const lt = el('text', { class: 'trucklbl', x: x + 11, y: y - 8 },
      MAP.layers.fleet);
    lt.textContent = v.id;
  }

  for (const id in S.fleet) {
    const v = S.fleet[id];
    if (!v.position) continue;
    const [x, y] = MAP.p(v.position[0], v.position[1]);
    const col = v.status === 'arrived' ? '#6b819a'
              : v.in_risk_zone ? '#ef4444'
              : v.status === 'halted' ? '#fbbf24' : '#38bdf8';
    const g = el('g', { class: 'truck',
      transform: `translate(${x} ${y}) rotate(${v.heading || 0})` },
      MAP.layers.fleet);
    if (v.in_risk_zone)
      el('circle', { cx: 0, cy: 0, r: 11, fill: col, opacity: .22,
        filter: 'url(#glow)' }, g);
    el('path', { d: 'M0,-7 L5,6 L0,3.2 L-5,6 Z', fill: col, stroke: '#06121a',
      'stroke-width': 1.1 }, g);
    g.addEventListener('mousemove', ev => TIP.show(
      `<div class="tt">${esc(v.id)}</div>
       <div class="tr"><span>${esc(v.label)}</span><b>${fmt(v.progress_pct, 0)}%</b></div>
       <div class="tr"><span>Cargo</span><b>${esc(v.cargo_label)}</b></div>
       <div class="tr"><span>Status</span><b style="color:${col}">${esc(v.status)}</b></div>
       <div class="tr"><span>Remaining</span><b>${fmt(v.remaining_km, 0)} km · ${fmt(v.eta_remaining_hours, 1)} h</b></div>
       ${v.current_segment ? `<div class="tr"><span>On</span><b>${esc(v.current_segment.corridor)}</b></div>` : ''}
       ${v.in_risk_zone ? '<div class="tr" style="color:#ef4444"><span>IN RISK ZONE</span><b>!</b></div>' : ''}`, ev));
    g.addEventListener('mouseleave', () => TIP.hide());

    // plate labels only once zoomed in, otherwise they read as map noise
    if (MAP.scale >= 1.8) {
      const lt = el('text', { class: 'trucklbl', x: x + 9, y: y - 6 },
        MAP.layers.fleet);
      lt.textContent = v.id;
    }
  }
}

/* ==========================================================================
   LEFT PANEL: route
   ========================================================================== */
function fillNodeSelects() {
  const byState = {};
  S.nodes.forEach(n => (byState[n.state_name || n.state] ||= []).push(n));
  const build = (sel, def) => {
    sel.innerHTML = '';
    Object.keys(byState).sort().forEach(st => {
      const og = document.createElement('optgroup');
      og.label = st;
      byState[st].forEach(n => {
        const o = document.createElement('option');
        o.value = n.id;
        o.textContent = n.name + (n.kind !== 'town' ? `  (${n.kind})` : '');
        og.appendChild(o);
      });
      sel.appendChild(og);
    });
    sel.value = def;
  };
  build($('#orig'), 'GUWAHATI');
  build($('#dest'), 'AIZAWL');
}

async function planRoute() {
  const o = $('#orig').value, d = $('#dest').value;
  if (o === d) { banner('Origin and destination are the same', 'warn'); return; }
  const box = $('#routesum');
  box.innerHTML = '<div class="center"><span class="spin"></span> planning…</div>';
  try {
    const j = await api('/route', { origin: o, dest: d, profile: S.profile, k: 3,
      cargo: $('#cargo').value });
    S.routes = j.routes || []; S.activeRoute = 0;
    drawRoutes();
    box.innerHTML = renderRouteCards(j);
    wireRouteCards();
    openRouteDetail(j);
    if (j.fallback_note) banner(j.fallback_note, 'warn', 9000);
  } catch (e) {
    box.innerHTML = `<div class="note crit">${esc(e.message)}</div>`;
  }
}

function renderRouteCards(j) {
  if (!j.routes?.length) return `<div class="note crit">No road route found.</div>`;
  let h = `<div class="sect">${esc(j.origin_name)} → ${esc(j.dest_name)}</div>`;
  if (j.fallback_note) h += `<div class="note warn">${esc(j.fallback_note)}</div>`;
  j.routes.forEach((r, i) => {
    h += `<div class="rt r${Math.min(r.rank, 3)} ${i === 0 ? 'on' : ''}" data-i="${i}">
      <div class="rth">
        <span class="rtt">${r.recommended ? '★ ' : ''}Option ${r.rank}</span>
        <span class="badge b-${r.status === 'open' ? 'safe' : r.status === 'risky' ? 'risky' : 'blocked'}">${esc(r.status)}</span>
      </div>
      <div class="grid3">
        <div class="g"><div class="gv">${esc(r.eta_text)}</div><div class="gl">ETA</div></div>
        <div class="g"><div class="gv">${fmt(r.distance_km, 0)}</div><div class="gl">km</div></div>
        <div class="g"><div class="gv" style="color:${sevColor(r.max_risk, 0)}">${fmt(r.max_risk, 2)}</div><div class="gl">peak risk</div></div>
      </div>
      <div class="kv"><span>Free-flow + ML delay</span><span>${fmt(r.freeflow_hours, 1)} + ${fmt(r.predicted_delay_hours, 1)} h</span></div>
      <div class="kv"><span>Risky / blocked segs</span><span>${r.n_risky} / ${r.n_blocked}</span></div>
      <div class="kv"><span>Composite score</span><span>${fmt(r.composite_score, 1)}</span></div>
      <div class="via">via ${esc(r.path_names.slice(1, -1).join(' · ')) || 'direct'}</div>
    </div>`;
  });
  if (j.comparison_note) h += `<div class="note">${esc(j.comparison_note)}</div>`;
  if (j.cargo_check) {
    const c = j.cargo_check;
    h += `<div class="note ${c.severity === 'critical' ? 'crit' : c.severity === 'warn' ? 'warn' : ''}">
      <b>${esc(c.cargo_label)} — ${esc(c.verdict).toUpperCase()}</b><br>
      ETA ${fmt(c.predicted_eta_hours, 1)} h against a ${c.max_transit_hours} h window
      (margin ${fmt(c.margin_hours, 1)} h).<br>${esc(c.advice)}</div>`;
  }
  return h;
}

function wireRouteCards() {
  $$('#routesum .rt').forEach(c => c.onclick = () => {
    S.activeRoute = +c.dataset.i;
    $$('#routesum .rt').forEach(x => x.classList.toggle('on', x === c));
    drawRoutes();
    openRouteDetail(null, S.routes[S.activeRoute]);
  });
}

/* ==========================================================================
   RIGHT PANEL
   ========================================================================== */
function setRight(title, html) {
  $('#rtitle').textContent = title;
  $('#rbody').innerHTML = html;
  $('#right').classList.add('show');
}
$('#rclose').onclick = () => $('#right').classList.remove('show');

/* ---------- segment + XAI ---------- */
async function openSegment(rid) {
  S.selectedSeg = rid; highlightSeg(rid);
  setRight('Segment', '<div class="center"><span class="spin"></span> explaining…</div>');
  try {
    const j = await api('/segment/' + encodeURIComponent(rid));
    setRight(j.segment.corridor, renderSegment(j));
  } catch (e) {
    setRight('Segment', `<div class="note crit">${esc(e.message)}</div>`);
  }
}

function renderSegment(j) {
  const s = j.segment, rx = j.risk_explanation, dx = j.delay_explanation;
  const col = sevColor(s.severity, s.risk_label);
  const bcls = s.risk_label === 2 ? 'b-blocked' : s.risk_label === 1 ? 'b-risky' : 'b-safe';

  let h = `<div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px">
      <div style="font-size:13.5px;font-weight:650">${esc(s.name)}</div>
      <span class="badge ${bcls}">${esc(s.risk_status)}</span>
    </div>
    <div class="big" style="color:${col}">${fmt(rx.failure_probability * 100, 1)}%</div>
    <div class="muted">probability of disruption (risky or blocked)</div>
    <div class="stack">
      <i style="width:${rx.class_probabilities[0] * 100}%;background:#2dd4a7"></i>
      <i style="width:${rx.class_probabilities[1] * 100}%;background:#fbbf24"></i>
      <i style="width:${rx.class_probabilities[2] * 100}%;background:#ef4444"></i></div>
    <div class="muted" style="font-family:var(--mono);font-size:10.5px">
      safe ${fmt(rx.class_probabilities[0] * 100, 0)}% ·
      risky ${fmt(rx.class_probabilities[1] * 100, 0)}% ·
      blocked ${fmt(rx.class_probabilities[2] * 100, 0)}%</div>
  </div>`;

  h += `<div class="card tight">
    <div class="kv"><span>Predicted delay</span><span>${fmt(s.delay_hours, 2)} h</span></div>
    <div class="kv"><span>Free-flow traverse</span><span>${fmt(s.freeflow_hours, 2)} h</span></div>
    <div class="kv"><span>Length · terrain</span><span>${fmt(s.length_km, 1)} km · ${esc(s.terrain)}</span></div>
    <div class="kv"><span>Class · lanes</span><span>${esc(s.road_class)} · ${s.lanes}</span></div>
    <div class="kv"><span>Rain 24 h</span><span>${fmt(s.weather.rainfall_24h_mm, 1)} mm</span></div>
    <div class="kv"><span>Soil saturation (API)</span><span>${fmt(s.weather.api_7d, 1)} mm</span></div>
    <div class="kv"><span>Condition</span><span>${esc(s.weather.condition)}</span></div>
  </div>`;

  /* ---- the XAI panel ---- */
  h += `<div class="sect">Why — exact model attribution</div>
    <div class="note"><b>${esc(rx.method)}.</b>
      Baseline ${fmt(rx.baseline_failure_probability * 100, 1)}% + the
      contributions below reconstruct the model output to
      ${rx.additivity_check.abs_error.toExponential(1)} absolute error
      ${rx.additivity_check.exact ? '— <b>verified exact</b>' : ''}.
      These are not correlations; they are this prediction, decomposed.</div>`;

  const maxc = Math.max(...rx.drivers.map(d => Math.abs(d.contribution)), 1e-6);
  rx.drivers.forEach(d => {
    const up = d.contribution > 0;
    const w = Math.abs(d.contribution) / maxc * 100;
    h += `<div class="drv">
      <div class="drvh">
        <span class="n">${esc(d.label)}</span>
        <span class="v">${esc(d.value_text)}</span>
      </div>
      <div class="bar">${up
        ? `<i class="up" style="width:${w}%"></i>`
        : `<i class="dn" style="width:${w}%"></i>`}</div>
      <div class="drvn">${esc(d.narrative)}</div>
      <div class="pct">${up ? '+' : ''}${fmt(d.contribution, 4)} · ${d.contribution_pct}% of the explained signal</div>
    </div>`;
  });

  h += `<div class="sect">Delay attribution</div><div class="card tight">
    <div class="kv"><span>Baseline</span><span>${fmt(dx.baseline_hours, 2)} h</span></div>`;
  dx.drivers.forEach(d => {
    h += `<div class="kv"><span>${esc(d.label)} <span class="muted">(${esc(d.value_text)})</span></span>
      <span style="color:${d.hours > 0 ? '#fb923c' : '#2dd4a7'}">${d.hours > 0 ? '+' : ''}${fmt(d.hours, 2)} h</span></div>`;
  });
  h += `<div class="kv" style="border-top:1px solid var(--line2);margin-top:5px;padding-top:6px">
    <span><b>Predicted</b></span><span><b>${fmt(dx.predicted_delay_hours, 2)} h</b></span></div></div>`;

  if (j.alert) {
    h += `<div class="sect">Alert as issued</div>`;
    for (const lg in j.alert.text) {
      const t = j.alert.text[lg];
      h += `<div class="al ${j.alert.severity === 'critical' ? 'crit' : ''}">
        <div class="alh"><span class="alt">${esc(t.title)}</span>
          <span class="badge b-info">${esc(lg)}</span></div>
        <div class="alb">${esc(t.body)}</div>
        ${j.alert.review_pending_languages.includes(lg)
          ? '<div class="revflag">⚠ translation pending native-speaker review</div>' : ''}
      </div>`;
    }
  }

  if (s.notes) h += `<div class="sect">Corridor note</div>
    <div class="muted">${esc(s.notes)}</div>`;

  if (j.recent_incidents?.length) {
    h += `<div class="sect">Recent incidents here (${j.recent_incidents.length})</div>
      <table class="t"><tr><th>When</th><th>Type</th><th>Sev</th></tr>`;
    j.recent_incidents.slice(0, 8).forEach(i => {
      h += `<tr><td class="mono" style="font-size:10.5px">${esc(i.reported_at.slice(5, 16).replace('T', ' '))}</td>
        <td>${esc(i.issue_type.replace(/_/g, ' '))}</td>
        <td><span class="badge b-${i.severity === 'high' ? 'blocked' : i.severity === 'medium' ? 'risky' : 'safe'}">${esc(i.severity)}</span></td></tr>`;
    });
    h += `</table>`;
  }
  return h;
}

/* ---------- route detail ---------- */
function openRouteDetail(j, single) {
  const r = single || j?.routes?.[0];
  if (!r) return;
  let h = '';
  if (j?.profile) h += `<div class="note"><b>${esc(j.profile.label)}.</b> ${esc(j.profile.description)}</div>`;

  h += `<div class="card">
    <div class="big">${esc(r.eta_text)}</div>
    <div class="muted">predicted arrival · ${fmt(r.distance_km, 0)} km · ${r.segments.length} segments</div>
    <div class="kv" style="margin-top:8px"><span>Free-flow driving</span><span>${fmt(r.freeflow_hours, 2)} h</span></div>
    <div class="kv"><span>ML route delay</span><span>+${fmt(r.predicted_delay_hours, 2)} h</span></div>
    <div class="kv"><span>Σ segment delays (naive)</span><span>${fmt(r.sum_of_segment_delays, 2)} h</span></div>
    <div class="kv"><span>Peak segment risk</span><span style="color:${sevColor(r.max_risk, 0)}">${fmt(r.max_risk, 3)}</span></div>
    <div class="kv"><span>Inter-state checkposts</span><span>${r.n_state_crossings}</span></div>
    <div class="kv"><span>Hill / pass share</span><span>${fmt(r.frac_hilly * 100, 0)}%</span></div>
  </div>`;

  h += `<div class="note"><b>Why the route model, not a sum?</b> The naive sum of
    segment delays is ${fmt(r.sum_of_segment_delays, 1)} h; the route model says
    ${fmt(r.predicted_delay_hours, 1)} h. Queues cascade, blocked segments force
    re-planning, checkposts add fixed dwell and hill sections effectively close
    after dark — none of which a sum can capture.</div>`;

  if (r.delay_explanation) {
    h += `<div class="sect">Route delay attribution</div><div class="card tight">`;
    r.delay_explanation.drivers.forEach(d => {
      h += `<div class="kv"><span>${esc(d.label)}</span>
        <span style="color:${d.hours > 0 ? '#fb923c' : '#2dd4a7'}">${d.hours > 0 ? '+' : ''}${fmt(d.hours, 2)} h</span></div>`;
    });
    h += `</div>`;
  }

  h += `<div class="sect">Worst segment on route</div>
    <div class="al ${r.worst_segment.risk_status === 'blocked' ? 'crit' : ''}"
         onclick="openSegment('${esc(r.worst_segment.road_id)}')">
      <div class="alh"><span class="alt">${esc(r.worst_segment.name)}</span>
        <span class="badge b-${r.worst_segment.risk_status === 'blocked' ? 'blocked' : 'risky'}">${esc(r.worst_segment.risk_status)}</span></div>
      <div class="alb">${esc(r.worst_segment.corridor)} · severity ${fmt(r.worst_segment.risk_score, 2)} — click for the full explanation</div>
    </div>`;

  h += `<div class="sect">Segment-by-segment</div>
    <table class="t"><tr><th>Segment</th><th>km</th><th>h</th><th>Risk</th></tr>`;
  r.segments.forEach(s => {
    h += `<tr onclick="openSegment('${esc(s.road_id)}')">
      <td>${esc(s.name)}<div class="muted" style="font-size:9.5px">${esc(s.corridor)}</div></td>
      <td class="num">${fmt(s.length_km, 0)}</td>
      <td class="num">${fmt(s.travel_hours, 1)}</td>
      <td><span class="badge b-${s.risk_status === 'blocked' ? 'blocked' : s.risk_status === 'risky' ? 'risky' : 'safe'}">${fmt(s.risk_score, 2)}</span></td></tr>`;
  });
  h += `</table>`;
  setRight('Route ' + (r.rank || 1), h);
}
window.openSegment = openSegment;
window.S = S; window.MAP = MAP;   // exposed for automated UI tests

/* ==========================================================================
   COMPARE + DEPARTURE
   ========================================================================== */
async function compareProfiles() {
  const o = $('#orig').value, d = $('#dest').value;
  setRight('Profile comparison', '<div class="center"><span class="spin"></span> running four optimisers…</div>');
  try {
    const j = await api('/route/compare', { origin: o, dest: d, profile: S.profile });
    S.routes = j.routes || []; S.activeRoute = 0; drawRoutes();
    let h = '';
    if (j.no_alternative_note)
      h += `<div class="note crit"><b>No alternative exists.</b> ${esc(j.no_alternative_note.replace(/^All four[^.]*\. /, ''))}</div>`;
    h += `<div class="sect">Same origin-destination, four objectives</div>
      <table class="t"><tr><th>Profile</th><th>ETA</th><th>km</th><th>Peak risk</th><th>Blk</th></tr>`;
    j.by_profile.forEach(p => {
      h += `<tr data-geo='${encodeURIComponent(JSON.stringify(p.geometry))}'>
        <td><b>${esc(p.label)}</b></td>
        <td class="num">${esc(p.eta_text)}</td>
        <td class="num">${fmt(p.distance_km, 0)}</td>
        <td class="num" style="color:${sevColor(p.max_risk, 0)}">${fmt(p.max_risk, 2)}</td>
        <td class="num">${p.n_blocked}</td></tr>`;
    });
    h += `</table>`;

    const byp = j.by_profile;
    if (byp.length >= 2) {
      const fast = byp.find(p => p.profile === 'fastest');
      const safe = byp.find(p => p.profile === 'safest');
      if (fast && safe) {
        const dkm = safe.distance_km - fast.distance_km;
        const dh = safe.eta_hours - fast.eta_hours;
        const dr = fast.max_risk - safe.max_risk;
        h += `<div class="note">${Math.abs(dkm) < 1
          ? '<b>Fastest and safest coincide</b> — there is no trade-off to make on this pair today.'
          : `<b>The trade-off, quantified.</b> Choosing <i>safest</i> over <i>fastest</i>
             costs ${fmt(Math.abs(dkm), 0)} km and ${fmt(Math.abs(dh), 1)} h,
             and buys a ${fmt(dr, 2)} reduction in peak segment risk
             (${fmt(dr / Math.max(fast.max_risk, .001) * 100, 0)}% lower).`}</div>`;
      }
    }
    if (j.comparison_note) h += `<div class="note">${esc(j.comparison_note)}</div>`;
    setRight('Profile comparison', h);
  } catch (e) {
    setRight('Profile comparison', `<div class="note crit">${esc(e.message)}</div>`);
  }
}

async function departureSweep() {
  const o = $('#orig').value, d = $('#dest').value;
  setRight('Departure optimiser',
    '<div class="center"><span class="spin"></span> sweeping the forecast horizon…<br><span class="muted">re-scoring the whole network at every window</span></div>');
  try {
    const j = await api('/departure', { origin: o, dest: d, profile: S.profile,
      horizon: 72 });
    const f = j.windows.filter(w => w.feasible);
    if (!f.length) { setRight('Departure optimiser', '<div class="note crit">No feasible window.</div>'); return; }
    const min = Math.min(...f.map(w => w.eta_hours));
    const max = Math.max(...f.map(w => w.eta_hours));
    const best = j.best_by_safety;

    let h = `<div class="note"><b>${esc(j.origin_name)} → ${esc(j.dest_name)}</b><br>
      ${esc(j.recommendation || '')}</div>`;
    h += `<div class="sect">Predicted transit by departure time</div><div class="dep">`;
    j.windows.forEach(w => {
      if (!w.feasible) { h += `<div class="depb" style="height:6px;background:#33475e"
        title="infeasible"></div>`; return; }
      const hgt = 12 + (w.eta_hours - min) / Math.max(max - min, .01) * 88;
      const isBest = best && w.depart_ts === best.depart_ts;
      h += `<div class="depb ${isBest ? 'best' : ''} ${w === j.windows[0] ? 'now' : ''}"
        style="height:${hgt}px;background:${isBest ? '' : sevColor(w.max_risk, w.n_blocked ? 2 : 0)}"
        title="${w.depart_ts.slice(5, 16).replace('T', ' ')} → ${fmt(w.eta_hours, 1)} h, peak risk ${fmt(w.max_risk, 2)}"></div>`;
    });
    h += `</div><div class="depax"><span>now</span><span>+36 h</span><span>+72 h</span></div>
      <div class="muted" style="margin-top:6px">Bar height = predicted transit
      hours. Colour = peak segment risk. Green = best window.</div>`;

    h += `<div class="sect">Windows</div><table class="t">
      <tr><th>Depart</th><th>Transit</th><th>Peak risk</th><th>Blk</th></tr>`;
    j.windows.forEach(w => {
      if (!w.feasible) return;
      const isBest = best && w.depart_ts === best.depart_ts;
      h += `<tr style="${isBest ? 'background:rgba(45,212,167,.08)' : ''}">
        <td class="mono">${esc(w.depart_ts.slice(5, 16).replace('T', ' '))}${isBest ? ' ★' : ''}</td>
        <td class="num">${fmt(w.eta_hours, 1)} h</td>
        <td class="num" style="color:${sevColor(w.max_risk, 0)}">${fmt(w.max_risk, 2)}</td>
        <td class="num">${w.n_blocked}</td></tr>`;
    });
    h += `</table>`;
    h += `<div class="note"><b>Why this matters.</b> Holding a convoy costs money;
      driving into a failing slope costs the load, the vehicle, and sometimes the
      crew. This is the one screen that turns a risk model into a dispatch
      decision.</div>`;
    setRight('Departure optimiser', h);
  } catch (e) {
    setRight('Departure optimiser', `<div class="note crit">${esc(e.message)}</div>`);
  }
}

/* ==========================================================================
   PANES: risk / crit / acc / fleet / whatif / model
   ========================================================================== */
function renderNetStat() {
  const s = S.summary;
  if (!s) return;
  $('#kpis').innerHTML = `
    <div class="kpi s"><div class="v">${s.safe}</div><div class="l">Safe</div></div>
    <div class="kpi r"><div class="v">${s.risky}</div><div class="l">Risky</div></div>
    <div class="kpi b"><div class="v">${s.blocked}</div><div class="l">Blocked</div></div>
    <div class="kpi k"><div class="v">${fmt(s.km_blocked, 0)}</div><div class="l">km cut</div></div>
    <div class="kpi"><div class="v">${fmt(s.mean_severity, 2)}</div><div class="l">Mean sev</div></div>
    <div class="kpi"><div class="v">${fmt(s.network_km, 0)}</div><div class="l">Network km</div></div>`;

  const tot = s.safe + s.risky + s.blocked;
  $('#netstat').innerHTML = `
    <div class="stack">
      <i style="width:${s.safe / tot * 100}%;background:#2dd4a7"></i>
      <i style="width:${s.risky / tot * 100}%;background:#fbbf24"></i>
      <i style="width:${s.blocked / tot * 100}%;background:#ef4444"></i></div>
    <div class="kv"><span>Segments scored</span><span>${s.segments_scored}</span></div>
    <div class="kv"><span>Network length</span><span>${fmt(s.network_km, 0)} km</span></div>
    <div class="kv"><span>Length predicted cut</span><span style="color:#ef4444">${fmt(s.km_blocked, 0)} km</span></div>
    <div class="kv"><span>Mean severity</span><span>${fmt(s.mean_severity, 3)}</span></div>
    <div class="kv"><span>Monsoon window</span><span>${s.is_monsoon ? 'yes' : 'no'}</span></div>
    ${s.counterfactual ? '<div class="note warn" style="margin-top:8px"><b>Counterfactual scenario active</b> — not live data.</div>' : ''}
    ${s.active_disruptions?.length ? s.active_disruptions.map(d =>
      `<div class="note crit" style="margin-top:8px"><b>${esc(d.corridor)}</b> — ${esc(d.note)}</div>`).join('') : ''}`;
}

function renderHotspots() {
  if (!S.heat) return;
  let h = `<table class="t"><tr><th>District</th><th>Sev</th><th>Cut km</th></tr>`;
  S.heat.hotspots.forEach(d => {
    h += `<tr data-lat="${d.lat}" data-lon="${d.lon}">
      <td>${esc(d.district)}<div class="muted" style="font-size:9.5px">${esc(d.state)} · ${d.segments} segs</div></td>
      <td class="num" style="color:${sevColor(d.mean_severity, 0)}">${fmt(d.mean_severity, 2)}</td>
      <td class="num">${fmt(d.km_blocked, 0)}</td></tr>`;
  });
  h += `</table>`;
  $('#hotspots').innerHTML = h;
  $$('#hotspots tr[data-lat]').forEach(tr => tr.onclick = () =>
    zoomTo(+tr.dataset.lat, +tr.dataset.lon, 4.2));
}

function zoomTo(lat, lon, z = 3.6) {
  const [x, y] = MAP.p(lat, lon);
  MAP.scale = z;
  MAP.tx = MAP.W / 2 - x * z;
  MAP.ty = MAP.H / 2 - y * z;
  MAP.apply();
}

async function renderAlerts() {
  const box = $('#alertfeed');
  box.innerHTML = '<div class="center"><span class="spin"></span></div>';
  try {
    const j = await api('/alerts', { limit: 24, lang: S.lang });
    if (!j.alerts.length) { box.innerHTML = '<div class="muted">Network clear — no alerts.</div>'; return; }
    box.innerHTML = j.alerts.map(a => {
      const en = a.text.en || Object.values(a.text)[0];
      const others = Object.keys(a.text).filter(k => k !== 'en');
      return `<div class="al ${a.severity === 'critical' ? 'crit' : ''}"
        ${a.target && a.scope === 'segment' ? `onclick="openSegment('${esc(a.target)}')"` : ''}>
        <div class="alh"><span class="alt">${esc(en.title)}</span>
          <span class="badge b-${a.severity === 'critical' ? 'blocked' : 'risky'}">${esc(a.severity)}</span></div>
        <div class="alb">${esc(en.body)}</div>
        ${others.map(lg => `<div class="allocal">${esc(a.text[lg].body)}
          ${a.review_pending_languages?.includes(lg)
            ? '<span class="revflag">⚠ pending native review</span>' : ''}</div>`).join('')}
        <div class="alm">${esc(a.scope)}${a.corridor ? ' · ' + esc(a.corridor) : ''} · ${esc((a.languages || []).join('/'))}</div>
      </div>`;
    }).join('');
  } catch (e) { box.innerHTML = `<div class="note crit">${esc(e.message)}</div>`; }
}

async function loadCrit() {
  const box = $('#crittbl');
  box.innerHTML = '<div class="center"><span class="spin"></span> deleting every segment and re-solving…</div>';
  try {
    S.crit = await api('/criticality');
    renderCrit();
    if (S.layers.crit) drawCrit();
  } catch (e) { box.innerHTML = `<div class="note crit">${esc(e.message)}</div>`; }
}

function renderCrit() {
  const c = S.crit; if (!c) return;
  const mode = $('#critmode .chip.on').dataset.v;
  const rows = mode === 'exposure' ? c.ranked_by_live_exposure : c.ranked_by_criticality;
  $('#critmeta').innerHTML = `Supply hub <b>${esc(c.hub_name)}</b> ·
    ${c.nodes_reachable_from_hub}/${c.nodes_total} nodes reachable ·
    ${c.structural_bridge_count} cut edges in the network.
    <div style="margin-top:6px">${esc(c.interpretation)}</div>`;
  let h = `<table class="t"><tr><th>#</th><th>Segment</th><th>Isolates</th><th>${mode === 'exposure' ? 'Exposure' : 'Score'}</th></tr>`;
  rows.slice(0, 18).forEach((r, i) => {
    h += `<tr data-rid="${esc(r.road_id)}">
      <td class="mono">${i + 1}</td>
      <td>${esc(r.name)}
        <div class="muted" style="font-size:9.5px">${esc(r.corridor)} ·
          ${esc(r.current_risk_status)}${r.is_structural_bridge ? ' · <b style="color:#f472b6">CUT EDGE</b>' : ''}</div></td>
      <td class="num">${r.isolates_nodes ? r.isolates_nodes + ' / ' + r.population_cut_off_k + 'k' : '—'}</td>
      <td class="num">${fmt(mode === 'exposure' ? r.exposure_score : r.criticality_score, 0)}</td></tr>`;
  });
  h += `</table>`;
  if (c.articulation_points?.length)
    h += `<div class="sect">Articulation points (${c.articulation_points.length})</div>
      <div class="muted">Removing any one of these towns disconnects the network:
      ${esc(c.articulation_points.slice(0, 22).join(', '))}${c.articulation_points.length > 22 ? '…' : ''}</div>`;
  $('#crittbl').innerHTML = h;
  $$('#crittbl tr[data-rid]').forEach(tr => tr.onclick = () => openSegment(tr.dataset.rid));
}

async function loadAcc() {
  $('#acctbl').innerHTML = '<div class="center"><span class="spin"></span> scoring every district…</div>';
  try {
    S.acc = await api('/accessibility');
    renderAcc();
    if (S.layers.acc) { drawHeat(); drawAcc(); }
  } catch (e) { $('#acctbl').innerHTML = `<div class="note crit">${esc(e.message)}</div>`; }
}

function renderAcc() {
  const a = S.acc; if (!a) return;
  let h = `<div class="sect">By state (worst first)</div><table class="t">
    <tr><th>State</th><th>Mean index</th><th>Districts</th></tr>`;
  a.by_state.forEach(s => {
    const g = s.mean_index >= 80 ? 'A' : s.mean_index >= 65 ? 'B'
            : s.mean_index >= 50 ? 'C' : s.mean_index >= 35 ? 'D' : 'E';
    h += `<tr><td>${esc(s.state_name)}</td>
      <td class="num"><span class="gr ${g}">${g}</span> ${fmt(s.mean_index, 1)}</td>
      <td class="num">${s.districts}</td></tr>`;
  });
  h += `</table>`;
  $('#accstate').innerHTML = h;

  let t = `<table class="t"><tr><th>District</th><th>Index</th><th>From hub</th><th>Routes</th></tr>`;
  a.most_vulnerable.forEach(d => {
    t += `<tr data-nid="${esc(d.node_id)}" data-lat="${d.lat}" data-lon="${d.lon}">
      <td>${esc(d.district)}<div class="muted" style="font-size:9.5px">${esc(d.state)}${d.on_structural_cut_edge ? ' · <b style="color:#f472b6">cut edge</b>' : ''}</div></td>
      <td class="num"><span class="gr ${gradeCls(d.grade)}">${d.grade}</span> ${fmt(d.index, 0)}</td>
      <td class="num">${esc(d.travel_text || '—')}</td>
      <td class="num">${d.independent_routes ?? '—'}</td></tr>`;
  });
  t += `</table>`;
  $('#acctbl').innerHTML = t;
  $$('#acctbl tr[data-nid]').forEach(tr => tr.onclick = () => {
    zoomTo(+tr.dataset.lat, +tr.dataset.lon, 4);
    openAccDetail(tr.dataset.nid);
  });
}

function openAccDetail(nid) {
  const d = S.acc.districts.find(x => x.node_id === nid);
  if (!d) return;
  const W = S.acc.weights;
  let h = `<div class="card">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
      <span class="gr ${gradeCls(d.grade)}" style="width:34px;height:34px;font-size:17px">${d.grade}</span>
      <div><div class="big" style="font-size:22px">${fmt(d.index, 1)}</div>
        <div class="muted">accessibility index · ${esc(d.state_name)}</div></div>
    </div>
    <div class="kv"><span>Travel time from hub</span><span>${esc(d.travel_text || '—')}</span></div>
    <div class="kv"><span>Independent corridors</span><span>${d.independent_routes}</span></div>
    <div class="kv"><span>On a structural cut edge</span><span style="color:${d.on_structural_cut_edge ? '#ef4444' : '#2dd4a7'}">${d.on_structural_cut_edge ? 'YES' : 'no'}</span></div>
  </div>`;
  h += `<div class="note">${esc(d.redundancy_note || '')}</div>`;
  h += `<div class="sect">Index decomposition</div>`;
  const names = { reach: 'Reach (time from hub)', reliability: 'Reliability (route risk)',
    redundancy: 'Redundancy (alternatives)', terrain: 'Terrain burden',
    disruption: 'Live disruption' };
  for (const k in d.components) {
    const v = d.components[k];
    h += `<div class="drv"><div class="drvh"><span class="n">${names[k] || k}</span>
      <span class="v">${fmt(v, 1)} × ${W[k]}</span></div>
      <div class="bar"><i style="width:${v}%;background:linear-gradient(90deg,#0ea5e9,#2dd4a7)"></i></div>
      <div class="pct">contributes ${fmt(v * W[k], 1)} points</div></div>`;
  }
  if (d.bottleneck) {
    h += `<div class="sect">Binding constraint</div>
      <div class="al ${d.bottleneck.risk_status === 'blocked' ? 'crit' : ''}"
        onclick="openSegment('${esc(d.bottleneck.road_id)}')">
        <div class="alh"><span class="alt">${esc(d.bottleneck.name)}</span>
          <span class="badge b-${d.bottleneck.risk_status === 'blocked' ? 'blocked' : 'risky'}">${esc(d.bottleneck.risk_status)}</span></div>
        <div class="alb">${esc(d.bottleneck.corridor)} — the worst segment on this
          district's best route. Fix this and the index moves.</div></div>`;
  }
  setRight(d.district, h);
}

/* ---------- fleet ---------- */
function renderFleet(summary) {
  if (summary) {
    $('#fleetstat').innerHTML = `
      <div class="kv"><span>Vehicles</span><span>${summary.total}</span></div>
      <div class="kv"><span>En route / halted</span><span>${summary.en_route} / ${summary.halted}</span></div>
      <div class="kv"><span>Arrived</span><span>${summary.arrived}</span></div>
      <div class="kv"><span>In risk zone</span><span style="color:${summary.in_risk_zone ? '#ef4444' : '#2dd4a7'}">${summary.in_risk_zone}</span></div>
      <div class="kv" style="border-top:1px solid var(--line2);margin-top:5px;padding-top:6px">
        <span>📡 Live GPS devices</span>
        <span style="color:${summary.live_gps_active ? '#2dd4a7' : 'var(--ink3)'}">${summary.live_gps_active ?? 0} active</span></div>
      ${(summary.live_gps_active ?? 0) === 0
        ? '<div class="muted" style="margin-top:6px">No real devices reporting. Open <b>/track</b> on a phone to stream live GPS into this map.</div>'
        : `<div class="kv"><span>Live devices in risk zone</span><span style="color:${summary.live_gps_in_risk ? '#ef4444' : '#2dd4a7'}">${summary.live_gps_in_risk}</span></div>`}`;
  }
  const live = S.liveGps.map(v => `
    <div class="vh ${v.in_risk_zone ? 'risk' : ''}" data-gps="${esc(v.id)}"
         style="border-color:${v.in_risk_zone ? '' : 'rgba(45,212,167,.4)'}">
      <div class="vhh"><span class="plate">📡 ${esc(v.id)}</span>
        <span class="badge b-${v.in_risk_zone ? (v.risk_status === 'blocked' ? 'blocked' : 'risky') : 'safe'}">LIVE GPS</span></div>
      <div class="muted" style="font-size:11px">${esc(v.cargo_type || 'general')} ·
        ${v.risk_status ? esc(v.risk_status) : 'off network'} ·
        ${fmt(v.age_minutes, 1)} min ago</div>
      <div class="vhm" style="margin-top:5px">
        <span>${v.speed_kmph == null ? '—' : fmt(v.speed_kmph, 0) + ' km/h'}</span>
        <span>±${fmt(v.accuracy_m, 0)} m</span>
        <span>${v.points} fixes</span></div>
    </div>`).join('');

  const vs = Object.values(S.fleet);
  $('#vehlist').innerHTML = (live || '') + vs.map(v => `
    <div class="vh ${v.in_risk_zone ? 'risk' : ''}" data-vid="${esc(v.id)}">
      <div class="vhh"><span class="plate">${esc(v.id)}</span>
        <span class="badge b-${v.status === 'arrived' ? 'safe' : v.in_risk_zone ? 'blocked' : v.status === 'halted' ? 'risky' : 'info'}">${esc(v.status)}</span></div>
      <div class="muted" style="font-size:11px">${esc(v.label)} · ${esc(v.cargo_label)}</div>
      <div class="prog"><i style="width:${v.progress_pct}%"></i></div>
      <div class="vhm"><span>${fmt(v.progress_pct, 0)}%</span>
        <span>${fmt(v.remaining_km, 0)} km left</span>
        <span>${fmt(v.eta_remaining_hours, 1)} h</span></div>
      ${v.current_segment ? `<div class="muted" style="font-size:10px;margin-top:3px">on ${esc(v.current_segment.corridor)} · ${esc(v.current_segment.risk_status)}</div>` : ''}
    </div>`).join('');
  $$('#vehlist .vh').forEach(c => c.onclick = () => {
    if (c.dataset.gps) {
      const g = S.liveGps.find(x => x.id === c.dataset.gps);
      if (g?.position) zoomTo(g.position[0], g.position[1], 7);
      return;
    }
    const v = S.fleet[c.dataset.vid];
    if (v?.position) zoomTo(v.position[0], v.position[1], 5);
  });

  $('#fleetalerts').innerHTML = S.fleetAlerts.length
    ? S.fleetAlerts.slice(0, 14).map(a => `
      <div class="al ${a.severity === 'critical' ? 'crit' : ''}"
        ${a.road_id ? `onclick="openSegment('${esc(a.road_id)}')"` : ''}>
        <div class="alh"><span class="alt">${esc(a.vehicle)}</span>
          <span class="badge b-${a.kind === 'geofence_breach' ? 'blocked' : a.kind === 'cargo_viability' ? 'warn' : 'risky'}">${esc(a.kind.replace(/_/g, ' '))}</span></div>
        <div class="alb">${esc(a.message)}</div>
        <div class="alm">${esc(a.at?.slice(5, 16).replace('T', ' ') || '')}</div></div>`).join('')
    : '<div class="muted">No fleet alerts yet.</div>';
}

function startStream() {
  if (S.es) { S.es.close(); S.es = null; }
  const es = new EventSource('/api/fleet/stream');
  S.es = es;
  es.addEventListener('snapshot', e => {
    const d = JSON.parse(e.data);
    d.vehicles.forEach(v => S.fleet[v.id] = v);
    S.liveGps = d.live_gps || [];
    S.fleetAlerts = d.alerts || [];
    drawFleet(); renderFleet(null);
  });
  es.addEventListener('tick', e => {
    if (S.fleetPaused) return;
    const d = JSON.parse(e.data);
    d.vehicles.forEach(v => S.fleet[v.id] = v);
    S.liveGps = d.live_gps || [];
    if (d.new_alerts?.length) {
      S.fleetAlerts = d.new_alerts.concat(S.fleetAlerts).slice(0, 60);
      const crit = d.new_alerts.find(a => a.severity === 'critical');
      if (crit) banner(crit.message, '', 7000);
    }
    $('#clk').textContent = d.sim_time.slice(11, 16);
    $('#clkd').textContent = new Date(d.sim_time).toDateString().slice(0, 11) + ' · sim';
    drawFleet(); renderFleet(d.summary);
  });
  es.onerror = () => { $('#livelbl').textContent = 'reconnecting'; };
  es.onopen = () => { $('#livelbl').textContent = 'live'; };
}

/* ---------- what-if ---------- */
function whatifParams() {
  const rain = +$('#s_rain').value, apiv = +$('#s_api').value, tmp = +$('#s_temp').value;
  const o = {};
  if (rain > 0) o.rain_24h_set = rain;
  if (apiv > 0) o.api_set = apiv;
  if (tmp !== 0) o.temperature_delta = tmp;
  if (S.whatifStates.size) o.states = Array.from(S.whatifStates);
  return Object.keys(o).length ? o : null;
}

async function runWhatif() {
  const o = whatifParams();
  if (!o) { banner('Move at least one slider first', 'warn'); return; }
  const box = $('#whatifout');
  box.innerHTML = '<div class="center"><span class="spin"></span> re-scoring the network…</div>';
  try {
    const prev = S.overrides; S.overrides = null;
    const q = new URLSearchParams();
    for (const k in o) q.set(k, k === 'states' ? o.states.join(',') : o[k]);
    const r = await fetch('/api/whatif?' + q.toString());
    const j = await r.json();
    if (!r.ok) throw new Error(j.error);
    S.overrides = o;
    await refreshNetwork();
    box.innerHTML = renderWhatif(j);
    banner(`Scenario active — ${j.delta.blocked >= 0 ? '+' : ''}${j.delta.blocked} blocked segments, ${fmt(j.delta.km_blocked, 0)} km more cut`,
      j.delta.blocked > 0 ? '' : 'info', 8000);
  } catch (e) {
    box.innerHTML = `<div class="note crit">${esc(e.message)}</div>`;
  }
}

function renderWhatif(j) {
  const d = j.delta, b = j.baseline, c = j.counterfactual;
  let h = `<div class="sect">Baseline → scenario</div>
    <table class="t"><tr><th></th><th>Live</th><th>Scenario</th><th>Δ</th></tr>
    <tr><td>Safe</td><td class="num">${b.safe}</td><td class="num">${c.safe}</td><td class="num" style="color:${d.safe < 0 ? '#ef4444' : '#2dd4a7'}">${d.safe >= 0 ? '+' : ''}${d.safe}</td></tr>
    <tr><td>Risky</td><td class="num">${b.risky}</td><td class="num">${c.risky}</td><td class="num">${d.risky >= 0 ? '+' : ''}${d.risky}</td></tr>
    <tr><td>Blocked</td><td class="num">${b.blocked}</td><td class="num">${c.blocked}</td><td class="num" style="color:${d.blocked > 0 ? '#ef4444' : '#2dd4a7'}">${d.blocked >= 0 ? '+' : ''}${d.blocked}</td></tr>
    <tr><td>km cut</td><td class="num">${fmt(b.km_blocked, 0)}</td><td class="num">${fmt(c.km_blocked, 0)}</td><td class="num" style="color:#ef4444">${d.km_blocked >= 0 ? '+' : ''}${fmt(d.km_blocked, 0)}</td></tr>
    </table>`;

  if (j.accessibility_impact?.length) {
    h += `<div class="sect">Districts that lose the most access</div><table class="t">
      <tr><th>District</th><th>Index</th><th>Drop</th></tr>`;
    j.accessibility_impact.slice(0, 10).forEach(x => {
      h += `<tr><td>${esc(x.district)}<div class="muted" style="font-size:9.5px">${esc(x.state)} · ${esc(x.grade_before)} → ${esc(x.grade_after)}</div></td>
        <td class="num">${fmt(x.index_before, 0)} → ${fmt(x.index_after, 0)}</td>
        <td class="num" style="color:#ef4444">−${fmt(x.drop, 1)}</td></tr>`;
    });
    h += `</table>`;
  }

  if (j.segments_changed?.length) {
    h += `<div class="sect">Segments that flipped (${j.n_segments_changed})</div>
      <table class="t"><tr><th>Segment</th><th>Change</th><th>Delay</th></tr>`;
    j.segments_changed.slice(0, 14).forEach(s => {
      h += `<tr data-rid="${esc(s.road_id)}"><td>${esc(s.name)}
        <div class="muted" style="font-size:9.5px">${esc(s.corridor)}</div></td>
        <td><span class="badge b-${s.from === 'safe' ? 'safe' : s.from === 'risky' ? 'risky' : 'blocked'}">${esc(s.from)}</span>
          → <span class="badge b-${s.to === 'safe' ? 'safe' : s.to === 'risky' ? 'risky' : 'blocked'}">${esc(s.to)}</span></td>
        <td class="num">${fmt(s.delay_before, 1)} → ${fmt(s.delay_after, 1)} h</td></tr>`;
    });
    h += `</table>`;
  }
  setTimeout(() => $$('#whatifout tr[data-rid]').forEach(tr =>
    tr.onclick = () => openSegment(tr.dataset.rid)), 0);
  return h;
}

/* ---------- model card ---------- */
function renderModel() {
  const m = S.metrics;
  if (!m) return;
  const t = m.risk_model_temporal, r = m.risk_model_random,
        hgb = m.risk_model_challenger_hgb_temporal,
        dl = m.delay_model_temporal, rt = m.route_delay_model,
        bl = m.route_delay_naive_mean_baseline;

  let h = `<div class="note"><b>Evaluated on a TEMPORAL split</b> — trained on the
    first 80% of the timeline, tested on the last 20%. That is the honest number,
    because in production you always predict forward in time. The random split is
    shown alongside so the optimism gap is visible rather than hidden.</div>`;

  h += `<div class="sect">Risk classifier — safe / risky / blocked</div>
    <div class="card tight">
      <div class="kv"><span>Accuracy (temporal)</span><span>${fmt(t.accuracy * 100, 1)}%</span></div>
      <div class="kv"><span>Accuracy (random)</span><span class="muted">${fmt(r.accuracy * 100, 1)}%</span></div>
      <div class="kv"><span>Macro F1</span><span>${fmt(t.f1_macro, 3)}</span></div>
      <div class="kv"><span>Balanced accuracy</span><span>${fmt(t.balanced_accuracy, 3)}</span></div>
      <div class="kv"><span><b>Recall on BLOCKED</b></span><span><b style="color:#2dd4a7">${fmt(t.blocked_recall * 100, 1)}%</b></span></div>
      <div class="kv"><span>AUC, will-disrupt</span><span>${fmt(t.failure_auc, 3)}</span></div>
      <div class="kv"><span>Brier (calibration)</span><span>${fmt(t.failure_brier_score, 3)}</span></div>
      <div class="kv"><span>Test rows</span><span>${(t.n_test || 0).toLocaleString()}</span></div>
    </div>
    <div class="muted">Recall on <i>blocked</i> is the metric that matters: a missed
    closure sends a convoy into a landslide. We weight classes to protect it.</div>`;

  h += `<div class="sect">Per class (temporal)</div><table class="t">
    <tr><th>Class</th><th>Prec</th><th>Recall</th><th>F1</th><th>n</th></tr>`;
  for (const k in t.per_class) {
    const c = t.per_class[k];
    h += `<tr><td><span class="badge b-${k}">${k}</span></td>
      <td class="num">${fmt(c.precision, 2)}</td><td class="num">${fmt(c.recall, 2)}</td>
      <td class="num">${fmt(c.f1, 2)}</td><td class="num">${c.support.toLocaleString()}</td></tr>`;
  }
  h += `</table>`;

  h += `<div class="sect">Confusion matrix</div><table class="t">
    <tr><th>actual ↓ / pred →</th><th>safe</th><th>risky</th><th>blocked</th></tr>`;
  ['safe', 'risky', 'blocked'].forEach((nm, i) => {
    h += `<tr><td>${nm}</td>` + t.confusion_matrix[i].map((v, jx) =>
      `<td class="num" style="${i === jx ? 'color:#2dd4a7;font-weight:700' : ''}">${v.toLocaleString()}</td>`).join('') + `</tr>`;
  });
  h += `</table>`;

  if (hgb) h += `<div class="sect">Challenger model</div>
    <div class="card tight">
      <div class="kv"><span>HistGradientBoosting acc</span><span>${fmt(hgb.accuracy * 100, 1)}%</span></div>
      <div class="kv"><span>Shipped RandomForest acc</span><span>${fmt(t.accuracy * 100, 1)}%</span></div>
    </div>
    <div class="note warn">${esc(m.model_selection_note || '')}</div>`;

  h += `<div class="sect">Delay regressors</div><div class="card tight">
    <div class="kv"><span>Segment MAE</span><span>${fmt(dl.mae_hours, 2)} h</span></div>
    <div class="kv"><span>Segment R²</span><span>${fmt(dl.r2, 3)}</span></div>
    <div class="kv"><span>Within 1 h</span><span>${fmt(dl.within_1h_pct, 1)}%</span></div>
    <div class="kv" style="margin-top:6px;border-top:1px solid var(--line2);padding-top:6px">
      <span>Route MAE</span><span>${fmt(rt.mae_hours, 2)} h</span></div>
    <div class="kv"><span>Route R²</span><span>${fmt(rt.r2, 3)}</span></div>
    <div class="kv"><span>Mean-baseline MAE</span><span class="muted">${fmt(bl?.mae_hours, 2)} h</span></div>
  </div>`;

  if (m.cascade_note) h += `<div class="note"><b>Two-stage cascade.</b> ${esc(m.cascade_note)}</div>`;

  h += `<div class="sect">What the model actually learned</div>
    <div class="muted" style="margin-bottom:8px">Permutation importance on the
    held-out temporal fold (F1-macro drop when a feature is shuffled).</div>`;
  const imp = m.global_importance_permutation || [];
  const mx = Math.max(...imp.map(d => d.importance), 1e-9);
  imp.slice(0, 12).forEach(d => {
    h += `<div class="drv" style="padding-bottom:7px;margin-bottom:7px">
      <div class="drvh"><span class="n" style="font-size:12px">${esc(d.feature)}</span>
        <span class="v">${fmt(d.importance, 4)}</span></div>
      <div class="bar"><i style="width:${d.importance / mx * 100}%;background:linear-gradient(90deg,#0ea5e9,#38bdf8)"></i></div></div>`;
  });
  h += `<div class="note"><b>Read the top of that list.</b> The strongest driver is
    <i>api_7d</i> — seven-day antecedent rainfall, i.e. soil saturation. Nobody
    told the model that; it recovered the actual physics of Himalayan slope
    failure from the data. Slopes do not fail because of today's rain, they fail
    because today's rain lands on ground already saturated. That is why this
    feature exists in the first place.</div>`;

  h += `<div class="sect">Training set</div><div class="card tight">
    <div class="kv"><span>Observations</span><span>${(m.n_observations || 0).toLocaleString()}</span></div>
    <div class="kv"><span>Route samples</span><span>${(m.n_route_samples || 0).toLocaleString()}</span></div>
    <div class="kv"><span>Features</span><span>${(m.feature_columns || []).length}</span></div>
    <div class="kv"><span>Temporal cut</span><span class="mono" style="font-size:10.5px">${esc((m.temporal_cut || '').slice(0, 10))}</span></div>
    <div class="kv"><span>Class balance</span><span>${m.class_balance ? Object.values(m.class_balance).map(v => v.toLocaleString()).join(' / ') : '—'}</span></div>
  </div>`;
  $('#modelout').innerHTML = h;
}

/* ==========================================================================
   DATA LOADING
   ========================================================================== */
async function refreshNetwork() {
  const [net, heat] = await Promise.all([
    api('/network', { slim: 1 }), api('/heatmap')
  ]);
  S.segments = net.segments;
  S.segById = {}; S.segments.forEach(s => S.segById[s.road_id] = s);
  S.summary = net.summary;
  S.heat = heat;
  renderNetStat(); renderHotspots();
  drawSegments(); drawHeat(); if (S.layers.acc) drawAcc();
  if (!S.overrides) {
    $('#clk').textContent = net.ts.slice(11, 16);
    $('#clkd').textContent = new Date(net.ts).toDateString().slice(0, 11);
  }
}

async function boot() {
  MAP.init();

  const nodes = await api('/nodes');
  S.nodes = nodes.nodes;
  S.nodes.forEach(n => S.nodeById[n.id] = n);
  MAP.setBounds(S.nodes);
  drawHulls(); drawNodes();
  fillNodeSelects();

  // state chips for what-if
  const states = {};
  S.nodes.forEach(n => states[n.state] = n.state_name);
  $('#statechips').innerHTML = Object.keys(states).sort().map(s =>
    `<button class="chip" data-v="${s}">${s}</button>`).join('');
  $$('#statechips .chip').forEach(c => c.onclick = () => {
    c.classList.toggle('on');
    c.classList.contains('on') ? S.whatifStates.add(c.dataset.v)
                               : S.whatifStates.delete(c.dataset.v);
  });

  await refreshNetwork();

  api('/metrics').then(m => { S.metrics = m; renderModel(); })
    .catch(() => $('#modelout').innerHTML = '<div class="note crit">metrics unavailable</div>');
  api('/languages').then(j => {
    S.langs = j.languages;
    const sel = $('#lang');
    for (const k in j.languages) {
      const o = document.createElement('option');
      o.value = k; o.textContent = j.languages[k].name +
        (j.languages[k].review_pending ? ' ⚠' : '');
      sel.appendChild(o);
    }
  }).catch(() => {});
  api('/incidents', { limit: 400 }).then(j => { S.incidents = j.incidents; drawIncidents(); })
    .catch(() => {});

  renderAlerts();
  startStream();
  banner('Click any road for the model\'s reasoning · shift-click a town to set destination',
    'info', 7000);
}

/* ==========================================================================
   WIRING
   ========================================================================== */
$$('#tabs .tab').forEach(t => t.onclick = () => {
  $$('#tabs .tab').forEach(x => x.classList.toggle('on', x === t));
  $$('.pane').forEach(p => p.classList.toggle('on', p.dataset.p === t.dataset.p));
  if (t.dataset.p === 'crit' && !S.crit) loadCrit();
  if (t.dataset.p === 'acc' && !S.acc) loadAcc();
});
$$('#profchips .chip').forEach(c => c.onclick = () => {
  $$('#profchips .chip').forEach(x => x.classList.toggle('on', x === c));
  S.profile = c.dataset.v;
});
$$('#critmode .chip').forEach(c => c.onclick = () => {
  $$('#critmode .chip').forEach(x => x.classList.toggle('on', x === c));
  renderCrit();
});
$('#goroute').onclick    = planRoute;
$('#gocompare').onclick  = compareProfiles;
$('#godep').onclick      = departureSweep;
$('#lang').onchange      = e => { S.lang = e.target.value; renderAlerts(); };

const LMAP = { L_seg: 'seg', L_heat: 'heat', L_crit: 'crit', L_acc: 'acc',
  L_fleet: 'fleet', L_inc: 'inc', L_lbl: 'lbl', L_hull: 'hull' };
for (const id in LMAP) {
  $('#' + id).onchange = async e => {
    S.layers[LMAP[id]] = e.target.checked ? 1 : 0;
    if (LMAP[id] === 'crit' && e.target.checked && !S.crit) await loadCrit();
    if (LMAP[id] === 'acc' && e.target.checked && !S.acc) await loadAcc();
    drawSegments(); drawHeat(); drawCrit(); drawIncidents(); drawFleet();
    drawHulls(); drawNodes(); drawRoutes();
    if (S.layers.acc) drawAcc();
  };
}

['rain', 'api', 'temp'].forEach(k => {
  const s = $('#s_' + k), v = $('#v_' + k);
  s.oninput = () => {
    const n = +s.value;
    v.textContent = k === 'temp' ? (n > 0 ? '+' : '') + n + ' °C'
                  : n === 0 ? 'off' : n + ' mm';
  };
});
$('#gowhatif').onclick = runWhatif;
$('#presetmonsoon').onclick = () => {
  $('#s_rain').value = 210; $('#s_api').value = 340; $('#s_temp').value = 0;
  ['rain', 'api', 'temp'].forEach(k => $('#s_' + k).dispatchEvent(new Event('input')));
  S.whatifStates = new Set(['ML', 'AS']);
  $$('#statechips .chip').forEach(c =>
    c.classList.toggle('on', S.whatifStates.has(c.dataset.v)));
  runWhatif();
};
$('#resetwhatif').onclick = async () => {
  S.overrides = null; S.whatifStates.clear();
  ['rain', 'api'].forEach(k => $('#s_' + k).value = 0);
  $('#s_temp').value = 0;
  ['rain', 'api', 'temp'].forEach(k => $('#s_' + k).dispatchEvent(new Event('input')));
  $$('#statechips .chip').forEach(c => c.classList.remove('on'));
  $('#whatifout').innerHTML = '';
  S.crit = null; S.acc = null;
  await refreshNetwork(); renderAlerts();
  banner('Back to live data', 'info', 2600);
};

$('#fleettoggle').onclick = e => {
  S.fleetPaused = !S.fleetPaused;
  e.target.textContent = S.fleetPaused ? 'Resume stream' : 'Pause stream';
};
$('#fleetreset').onclick = async () => {
  await fetch('/api/fleet/reset', { method: 'POST' });
  S.fleet = {}; S.fleetAlerts = [];
  banner('Fleet re-planned on the current network', 'info', 3000);
};

$('#map').addEventListener('click', e => {
  if (e.target.id === 'map') { S.selectedSeg = null; MAP.clear('crit'); drawCrit(); }
});
window.addEventListener('resize', () => MAP.scalebar());

boot().catch(e => {
  document.body.innerHTML =
    `<div style="padding:40px;font:14px system-ui;color:#e8eef6">
      <h2>Backend not ready</h2>
      <p style="color:#9fb3c8">${esc(e.message)}</p>
      <p style="color:#9fb3c8">Run <code style="color:#38bdf8">python3 scripts/pipeline.py</code>
      to build the dataset and train the models, then restart the server.</p></div>`;
});
