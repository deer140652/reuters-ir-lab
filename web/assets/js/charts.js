/*
 * charts.js
 * ---------
 * 純 SVG 圖表（不使用外部套件）：分組長條圖、正負長條圖、水平長條圖、熱圖、散佈圖，
 * 以及 PCA（power iteration）與精確版 t-SNE。
 * 所有圖形元素帶 data-tip 屬性，滑鼠移上去會顯示提示框。
 * 對外公開為 Charts。
 */
(function (global) {
  "use strict";

  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const f3 = (x) => (x == null || isNaN(x) ? "–" : x.toFixed(3));

  // ---- 提示框 ----
  function bindTips(root) {
    const tip = document.getElementById("chart-tip");
    if (!tip || !root) return;
    root.addEventListener("mousemove", (e) => {
      const t = e.target.closest("[data-tip]");
      if (!t) { tip.style.display = "none"; return; }
      tip.innerHTML = t.getAttribute("data-tip");
      tip.style.display = "block";
      const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
      const y = e.clientY + 16 + tip.offsetHeight > window.innerHeight ? e.clientY - tip.offsetHeight - 10 : e.clientY + 16;
      tip.style.left = x + "px"; tip.style.top = y + "px";
    });
    root.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  }

  // 只有上方圓角的長條（貼齊基準線）
  function barPath(x, y, w, h, r) {
    if (h <= 0.5) return `M${x},${y + h}h${w}`;
    r = Math.min(r, w / 2, h);
    return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
  }
  // 水平長條（右側圓角，起點在 x0）
  function hbarPath(x0, y, len, h, r) {
    const dir = len >= 0 ? 1 : -1, L = Math.abs(len);
    if (L <= 0.5) return `M${x0},${y}v${h}`;
    r = Math.min(r, h / 2, L);
    const x1 = x0 + dir * L;
    return `M${x0},${y}H${x1 - dir * r}Q${x1},${y} ${x1},${y + r}V${y + h - r}Q${x1},${y + h} ${x1 - dir * r},${y + h}H${x0}Z`;
  }

  // ---- 分組長條圖 ----
  // opts: { cats: [{label, sub}], series: [{name, color, values}], max, height, ref: [value per cat] | number, refLabel, fmt, yLabel }
  function groupedBar(el, opts) {
    const nLines = Math.max(1, ...opts.cats.map((c) => (c.lines ? c.lines.length : 1 + (c.sub ? 1 : 0))));
    const W = opts.width || 640, H = opts.height || 300, ml = 40, mr = 10, mt = 14, mb = 18 + nLines * 14;
    const max = opts.max || Math.max(...opts.series.flatMap((s) => s.values)) || 1;
    const nC = opts.cats.length, nS = opts.series.length;
    const cw = (W - ml - mr) / nC, gap = 2, bw = Math.min(26, (cw * 0.78 - gap * (nS - 1)) / nS);
    const Y = (v) => mt + (H - mt - mb) * (1 - v / max);
    const fmt = opts.fmt || f3;
    let g = "";
    const ticks = opts.ticks || [0, 0.25, 0.5, 0.75, 1].map((t) => t * max);
    ticks.forEach((t) => {
      g += `<line class="grid" x1="${ml}" x2="${W - mr}" y1="${Y(t)}" y2="${Y(t)}"/><text class="tick" x="${ml - 6}" y="${Y(t) + 4}" text-anchor="end">${opts.tickFmt ? opts.tickFmt(t) : +t.toFixed(2)}</text>`;
    });
    opts.cats.forEach((c, i) => {
      const x0 = ml + i * cw + (cw - (bw * nS + gap * (nS - 1))) / 2;
      opts.series.forEach((s, j) => {
        const v = s.values[i];
        const x = x0 + j * (bw + gap), y = Y(Math.max(0, v));
        const tip = `<b>${esc(c.tipLabel || c.label)}</b><br>${esc(s.name)}：${fmt(v)}${s.tips && s.tips[i] ? "<br>" + s.tips[i] : ""}`;
        g += `<path class="bar" d="${barPath(x, y, bw, Y(0) - y, 4)}" fill="${s.color}" data-tip="${esc(tip)}"/>`;
        if (opts.showValues) g += `<text class="val-label" x="${x + bw / 2}" y="${y - 3}" text-anchor="middle">${opts.valueFmt ? opts.valueFmt(v) : fmt(v)}</text>`;
        // 透明的大範圍感應區
        g += `<rect x="${x - 1}" y="${mt}" width="${bw + 2}" height="${H - mt - mb}" fill="transparent" data-tip="${esc(tip)}"/>`;
      });
      if (opts.ref != null) {
        const rv = Array.isArray(opts.ref) ? opts.ref[i] : opts.ref;
        if (rv != null) g += `<line class="ref-line" x1="${ml + i * cw + 4}" x2="${ml + (i + 1) * cw - 4}" y1="${Y(rv)}" y2="${Y(rv)}" data-tip="${esc(`${opts.refLabel || "參考線"}：${fmt(rv)}`)}"/>`;
      }
      const lines = c.lines || [c.label].concat(c.sub ? [c.sub] : []);
      lines.forEach((ln, li) => {
        const last = li === lines.length - 1 && lines.length > 1;
        g += `<text class="${last ? "tick" : "cat-label"}" x="${ml + i * cw + cw / 2}" y="${H - mb + 16 + li * 14}" text-anchor="middle">${esc(ln)}</text>`;
      });
    });
    g += `<line class="base" x1="${ml}" x2="${W - mr}" y1="${Y(0)}" y2="${Y(0)}"/>`;
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img">${g}</svg>`;
    bindTips(el);
  }

  // ---- 正負水平長條圖（勝負圖）----
  // items: [{label, value, tip}]，依 value 由大到小
  function divergingBar(el, items, opts) {
    opts = opts || {};
    const rowH = 19, W = opts.width || 640, ml = opts.labelWidth || 250, mr = 46, mt = 8;
    const H = mt + items.length * rowH + 26;
    const ext = Math.max(0.05, ...items.map((d) => Math.abs(d.value)));
    const lim = Math.ceil(ext * 10) / 10;
    const cx = ml + (W - ml - mr) / 2, half = (W - ml - mr) / 2;
    const X = (v) => cx + (v / lim) * half;
    let g = "";
    [-lim, -lim / 2, 0, lim / 2, lim].forEach((t) => {
      g += `<line class="grid" x1="${X(t)}" x2="${X(t)}" y1="${mt}" y2="${H - 22}"/><text class="tick" x="${X(t)}" y="${H - 8}" text-anchor="middle">${t > 0 ? "+" : ""}${+t.toFixed(2)}</text>`;
    });
    items.forEach((d, i) => {
      const y = mt + i * rowH;
      const col = d.value >= 0 ? opts.pos : opts.neg;
      g += `<text class="cat-label" x="${ml - 8}" y="${y + 13}" text-anchor="end">${esc(d.label)}</text>`;
      g += `<path class="bar" d="${hbarPath(cx, y + 3, X(d.value) - cx, rowH - 6, 3)}" fill="${col}" data-tip="${esc(d.tip || "")}"/>`;
      g += `<text class="val-label" x="${d.value >= 0 ? X(d.value) + 4 : X(d.value) - 4}" y="${y + 13}" text-anchor="${d.value >= 0 ? "start" : "end"}">${d.value > 0 ? "+" : ""}${d.value.toFixed(2)}</text>`;
      g += `<rect x="0" y="${y}" width="${W}" height="${rowH}" fill="transparent" data-tip="${esc(d.tip || "")}"/>`;
    });
    g += `<line class="base" x1="${cx}" x2="${cx}" y1="${mt}" y2="${H - 22}"/>`;
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img">${g}</svg>`;
    bindTips(el);
  }

  // ---- 水平長條圖（文件向量組成）----
  // items: [{label, value, color, tip}]
  function hbar(el, items, opts) {
    opts = opts || {};
    const rowH = 20, W = opts.width || 520, ml = 120, mr = 50, mt = 4;
    const H = mt + items.length * rowH + 8;
    const max = Math.max(...items.map((d) => d.value)) || 1;
    const X = (v) => ml + (v / max) * (W - ml - mr);
    let g = "";
    items.forEach((d, i) => {
      const y = mt + i * rowH;
      g += `<text class="cat-label" x="${ml - 8}" y="${y + 14}" text-anchor="end" style="font-family:var(--mono)">${esc(d.label)}</text>`;
      g += `<path class="bar" d="${hbarPath(ml, y + 4, X(d.value) - ml, rowH - 8, 3)}" fill="${d.color}" data-tip="${esc(d.tip || "")}"/>`;
      g += `<text class="val-label" x="${X(d.value) + 4}" y="${y + 14}">${d.value.toFixed(2)}</text>`;
      g += `<rect x="0" y="${y}" width="${W}" height="${rowH}" fill="transparent" data-tip="${esc(d.tip || "")}"/>`;
    });
    g += `<line class="base" x1="${ml}" x2="${ml}" y1="${mt}" y2="${H - 4}"/>`;
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img">${g}</svg>`;
    bindTips(el);
  }

  // ---- 熱圖 ----
  // rows: [label], cols: [label], cells[r][c] = { v, tip }；正值藍、負值橘，中間灰
  function heatmap(el, rows, cols, cells, opts) {
    opts = opts || {};
    const cellW = 46, cellH = 30, ml = 96, mt = 74, W = ml + cols.length * cellW + 10, H = mt + rows.length * cellH + 8;
    const ext = Math.max(1e-6, ...cells.flat().map((c) => Math.abs(c.v)));
    const color = (v) => {
      const t = Math.min(1, Math.abs(v) / ext);
      // 從中性灰 #ecebe6 漸變到藍 #1c5cab（正）或橘 #c24d1f（負）
      const a = [236, 235, 230], b = v >= 0 ? [28, 92, 171] : [194, 77, 31];
      const c = a.map((x, i) => Math.round(x + (b[i] - x) * Math.pow(t, 0.8)));
      return `rgb(${c.join(",")})`;
    };
    let g = "";
    cols.forEach((c, j) => {
      const x = ml + j * cellW + cellW / 2;
      g += `<text class="cat-label" transform="translate(${x + 4},${mt - 6}) rotate(-45)" style="font-family:var(--mono)">${esc(c)}</text>`;
    });
    rows.forEach((r, i) => {
      const y = mt + i * cellH;
      g += `<text class="cat-label" x="${ml - 8}" y="${y + cellH / 2 + 4}" text-anchor="end" style="font-family:var(--mono);font-weight:700">${esc(r)}</text>`;
      cols.forEach((c, j) => {
        const cell = cells[i][j];
        const x = ml + j * cellW;
        const fill = color(cell.v);
        const t = Math.abs(cell.v) / ext;
        g += `<rect class="cell" x="${x + 1}" y="${y + 1}" width="${cellW - 2}" height="${cellH - 2}" rx="3" fill="${fill}" data-tip="${esc(cell.tip)}"/>`;
        g += `<text x="${x + cellW / 2}" y="${y + cellH / 2 + 4}" text-anchor="middle" style="font-family:var(--mono);font-size:10px;pointer-events:none;fill:${t > 0.55 ? "#fff" : "#3b4256"}">${cell.label}</text>`;
      });
    });
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" style="max-width:${W}px">${g}</svg>`;
    bindTips(el);
  }

  // ---- PCA 與 t-SNE----
  function seededRandom(seed) {
    let s = seed >>> 0 || 1;
    return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  }

  function pca2D(X) {
    const n = X.length, D = X[0].length;
    const mean = new Float64Array(D);
    X.forEach((r) => { for (let d = 0; d < D; d++) mean[d] += r[d] / n; });
    const C = X.map((r) => { const c = new Float64Array(D); for (let d = 0; d < D; d++) c[d] = r[d] - mean[d]; return c; });
    let total = 0;
    C.forEach((r) => { for (let d = 0; d < D; d++) total += r[d] * r[d]; });
    function component(orth) {
      let v = new Float64Array(D);
      for (let d = 0; d < D; d++) v[d] = Math.sin(d * 12.9898 + 1);
      let lambda = 0;
      for (let it = 0; it < 200; it++) {
        const proj = C.map((r) => { let s = 0; for (let d = 0; d < D; d++) s += r[d] * v[d]; return s; });
        const nv = new Float64Array(D);
        C.forEach((r, i) => { for (let d = 0; d < D; d++) nv[d] += r[d] * proj[i]; });
        if (orth) { let p = 0; for (let d = 0; d < D; d++) p += nv[d] * orth[d]; for (let d = 0; d < D; d++) nv[d] -= p * orth[d]; }
        let norm = 0; for (let d = 0; d < D; d++) norm += nv[d] * nv[d];
        norm = Math.sqrt(norm) || 1;
        lambda = norm;
        for (let d = 0; d < D; d++) nv[d] /= norm;
        v = nv;
      }
      return { v, lambda };
    }
    const c1 = component(null), c2 = component(c1.v);
    const pts = C.map((r) => {
      let x = 0, y = 0;
      for (let d = 0; d < D; d++) { x += r[d] * c1.v[d]; y += r[d] * c2.v[d]; }
      return [x, y];
    });
    return { pts, explained: total ? [c1.lambda / total, c2.lambda / total] : [0, 0] };
  }

  function tsne2D(X, perplexity, seed) {
    const n = X.length;
    const perp = Math.max(2, Math.min(perplexity, (n - 1) / 3));
    const D2 = [];
    for (let i = 0; i < n; i++) {
      D2.push(new Float64Array(n));
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        let s = 0;
        for (let d = 0; d < X[i].length; d++) { const t = X[i][d] - X[j][d]; s += t * t; }
        D2[i][j] = s;
      }
    }
    const P = [];
    const logU = Math.log(perp);
    for (let i = 0; i < n; i++) {
      let beta = 1, lo = -Infinity, hi = Infinity;
      const row = new Float64Array(n);
      for (let it = 0; it < 60; it++) {
        let sum = 0;
        for (let j = 0; j < n; j++) { row[j] = i === j ? 0 : Math.exp(-D2[i][j] * beta); sum += row[j]; }
        sum = sum || 1e-12;
        let H = 0;
        for (let j = 0; j < n; j++) { row[j] /= sum; if (row[j] > 1e-12) H -= row[j] * Math.log(row[j]); }
        if (Math.abs(H - logU) < 1e-5) break;
        if (H > logU) { lo = beta; beta = hi === Infinity ? beta * 2 : (beta + hi) / 2; }
        else { hi = beta; beta = lo === -Infinity ? beta / 2 : (beta + lo) / 2; }
      }
      P.push(row);
    }
    const Ps = [];
    for (let i = 0; i < n; i++) {
      Ps.push(new Float64Array(n));
      for (let j = 0; j < n; j++) Ps[i][j] = Math.max((P[i][j] + P[j][i]) / (2 * n), 1e-12);
    }
    const rnd = seededRandom(seed || 20261006);
    const gauss = () => { let u = 0, v = 0; while (!u) u = rnd(); while (!v) v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const Y = [], gains = [], upd = [];
    for (let i = 0; i < n; i++) { Y.push([gauss() * 1e-4, gauss() * 1e-4]); gains.push([1, 1]); upd.push([0, 0]); }
    const ITERS = 1000, lr = Math.max(n / 12, 50);
    const num = [];
    for (let i = 0; i < n; i++) num.push(new Float64Array(n));
    for (let it = 0; it < ITERS; it++) {
      const exag = it < 250 ? 12 : 1;
      const mom = it < 250 ? 0.5 : 0.8;
      let Z = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        if (i === j) { num[i][j] = 0; continue; }
        const dx = Y[i][0] - Y[j][0], dy = Y[i][1] - Y[j][1];
        num[i][j] = 1 / (1 + dx * dx + dy * dy);
        Z += num[i][j];
      }
      const grad = [];
      for (let i = 0; i < n; i++) {
        let gx = 0, gy = 0;
        for (let j = 0; j < n; j++) {
          if (i === j) continue;
          const q = Math.max(num[i][j] / Z, 1e-12);
          const m = (exag * Ps[i][j] - q) * num[i][j];
          gx += 4 * m * (Y[i][0] - Y[j][0]);
          gy += 4 * m * (Y[i][1] - Y[j][1]);
        }
        grad.push([gx, gy]);
      }
      for (let i = 0; i < n; i++) {
        for (let d = 0; d < 2; d++) {
          const g = grad[i][d];
          gains[i][d] = Math.sign(g) !== Math.sign(upd[i][d]) ? gains[i][d] + 0.2 : Math.max(gains[i][d] * 0.8, 0.01);
          upd[i][d] = mom * upd[i][d] - lr * gains[i][d] * g;
          Y[i][d] += upd[i][d];
        }
      }
      let mx = 0, my = 0;
      for (let i = 0; i < n; i++) { mx += Y[i][0] / n; my += Y[i][1] / n; }
      for (let i = 0; i < n; i++) { Y[i][0] -= mx; Y[i][1] -= my; }
    }
    return Y;
  }

  // 散佈圖：items [{word, group, isQuery, tip}]，colors[group]
  function scatter(el, points, items, colors) {
    const W = 620, H = 440, ml = 16, mr = 90, mt = 16, mb = 16;
    const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const X = (x) => ml + ((x - x0) / (x1 - x0 || 1)) * (W - ml - mr);
    const Y = (y) => H - mb - ((y - y0) / (y1 - y0 || 1)) * (H - mt - mb);
    const placed = [];
    const overlaps = (b) => placed.some((q) => b.x < q.x + q.w && b.x + b.w > q.x && b.y < q.y + q.h && b.y + b.h > q.y);
    const OFFSETS = [0];
    for (let o = 12; o <= 60; o += 12) OFFSETS.push(o, -o);
    const order = items.map((_, i) => i).sort((a, b) => (items[b].isQuery ? 1 : 0) - (items[a].isQuery ? 1 : 0));
    const dyOf = new Array(items.length).fill(0);
    order.forEach((i) => {
      const it = items[i], fs = it.isQuery ? 12.5 : 10.5;
      const w = it.word.length * fs * 0.62 + 2;
      const box = (dy) => ({ x: X(points[i][0]) + 6, y: Y(points[i][1]) + dy - fs * 0.85, w, h: fs + 2 });
      const dy = OFFSETS.find((o) => { const b = box(o); return b.y > 2 && b.y + b.h < H - 2 && !overlaps(b); });
      dyOf[i] = dy === undefined ? 0 : dy;
      placed.push(box(dyOf[i]));
    });
    let g = `<rect x="0.5" y="0.5" width="${W - 1}" height="${H - 1}" fill="none" stroke="#E1DED4"/>`;
    items.forEach((it, i) => {
      const cx = X(points[i][0]), cy = Y(points[i][1]), col = colors[it.group % colors.length];
      if (dyOf[i]) g += `<line x1="${cx}" y1="${cy}" x2="${cx + 6}" y2="${cy + dyOf[i]}" stroke="#bbb" stroke-width="0.8"/>`;
      g += `<circle class="pt" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${it.isQuery ? 6 : 4.2}" fill="${col}" stroke="#FFFDF8" stroke-width="2" data-tip="${esc(it.tip || it.word)}"/>`;
      g += `<text x="${(cx + 7).toFixed(1)}" y="${(cy + dyOf[i] + 4).toFixed(1)}" style="font-family:var(--mono);font-size:${it.isQuery ? 12.5 : 10.5}px;font-weight:${it.isQuery ? 700 : 400};fill:#2b3550">${esc(it.word)}</text>`;
    });
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img">${g}</svg>`;
    bindTips(el);
  }

  global.Charts = { esc, f3, bindTips, groupedBar, divergingBar, hbar, heatmap, pca2D, tsne2D, scatter };
})(window);
