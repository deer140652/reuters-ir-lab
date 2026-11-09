/*
 * embed-core.js
 * -------------
 * 方法 B：TF-IDF 加權詞向量（TF-IDF Weighted Word Embedding）
 *
 *   文件向量  d = Σ_t w_t · v̂_t       （w_t = 方法 A 的 TF-IDF 權重，v̂_t = 單位長度的詞向量）
 *   查詢向量  q = Σ_i w_i · v̂_i
 *   分數      cos(q, d) = q·d / (|q| |d|)
 *
 * 因為 q 與 d 都是詞向量的加權和，分數可以精確拆成「查詢詞 i × 文件詞 j」的貢獻：
 *   score = Σ_i Σ_j  w_i · w_j · (v̂_i · v̂_j) / (|q| |d|)
 * 網頁上的「分數拆解熱圖」就是畫這個矩陣。
 *
 * 不在詞向量詞彙表（訓練集詞頻 ≥ 5）中的詞，不會進入文件或查詢向量。
 *
 * 另外包含三個實驗：
 *   - mask：把指定的詞從所有文件中「遮掉」後重新排名（遮字實驗）
 *   - overlapAt：兩個查詢的前 k 名重疊數（反義查詢實驗）
 *   - zero-overlap：B 的前 k 名中，與查詢沒有任何共同詞的文件
 *
 * 純 JavaScript，網頁與 Node 都能使用。對外公開為 EmbedCore。
 */
(function (global) {
  "use strict";

  // 建立所有文件的「未正規化」向量（存成一個大的 Float32Array）與長度
  function buildDocVectors(index, model) {
    const D = model.dim, N = index.ids.length;
    const raw = new Float32Array(N * D);
    const norm = new Float32Array(N);
    const pos = new Map();
    const coverage = new Float32Array(N); // 有詞向量的詞，佔文件 TF-IDF 權重的比例
    index.ids.forEach((id, n) => {
      pos.set(id, n);
      const off = n * D;
      let wIn = 0, wAll = 0;
      index.docWeights.get(id).forEach((w, t) => {
        wAll += w;
        const v = model.lookup(t);
        if (v === undefined) return;
        wIn += w;
        const vo = v * D;
        for (let k = 0; k < D; k++) raw[off + k] += w * model.vecs[vo + k];
      });
      let s = 0;
      for (let k = 0; k < D; k++) s += raw[off + k] * raw[off + k];
      norm[n] = Math.sqrt(s);
      coverage[n] = wAll ? wIn / wAll : 0;
    });
    return { model, index, D, raw, norm, pos, coverage };
  }

  function queryVector(qTerms, dv) {
    const { model, index, D } = dv;
    const qw = IRCore.queryWeights(qTerms, index);
    const vec = new Float32Array(D);
    const used = [], oov = [];
    qw.forEach((w, t) => {
      const v = model.lookup(t);
      if (v === undefined) { oov.push(t); return; }
      used.push([t, w]);
      const vo = v * D;
      for (let k = 0; k < D; k++) vec[k] += w * model.vecs[vo + k];
    });
    let s = 0;
    for (let k = 0; k < D; k++) s += vec[k] * vec[k];
    return { vec, norm: Math.sqrt(s), used, oov, weights: qw };
  }

  // 排名；opts.mask：要從文件中遮掉的詞（Set）
  function rank(qTerms, dv, opts) {
    opts = opts || {};
    const q = queryVector(qTerms, dv);
    const { D, raw, norm, index, model } = dv;
    const out = [];
    if (!q.norm) return out;
    const mask = opts.mask && opts.mask.size ? opts.mask : null;
    const maskVecs = [];
    if (mask) mask.forEach((t) => { const v = model.lookup(t); if (v !== undefined) maskVecs.push([t, v * D]); });
    const tmp = new Float32Array(D);
    index.ids.forEach((id, n) => {
      const off = n * D;
      let dot = 0, dn = norm[n];
      let touched = false;
      if (mask) {
        const dw = index.docWeights.get(id);
        maskVecs.forEach(([t, vo]) => {
          const w = dw.get(t);
          if (!w) return;
          if (!touched) { for (let k = 0; k < D; k++) tmp[k] = raw[off + k]; touched = true; }
          for (let k = 0; k < D; k++) tmp[k] -= w * model.vecs[vo + k];
        });
      }
      if (touched) {
        let s = 0;
        for (let k = 0; k < D; k++) { dot += q.vec[k] * tmp[k]; s += tmp[k] * tmp[k]; }
        dn = Math.sqrt(s);
      } else {
        for (let k = 0; k < D; k++) dot += q.vec[k] * raw[off + k];
      }
      if (dn > 1e-9) out.push({ id, score: dot / (q.norm * dn) });
    });
    out.sort((a, b) => b.score - a.score || (a.id < b.id ? -1 : 1));
    return out;
  }

  // TF-IDF 排名（也支援遮字）
  function rankTfidfMasked(qTerms, index, mask) {
    if (!mask || !mask.size) return IRCore.rankTfidf(qTerms, index);
    const q = IRCore.queryWeights(qTerms.filter((t) => !mask.has(t)), index);
    let qn = 0;
    q.forEach((x) => { qn += x * x; });
    qn = Math.sqrt(qn);
    if (!qn) return [];
    const acc = new Map();
    q.forEach((qw, t) => (index.postings.get(t) || []).forEach(([id, dw]) => acc.set(id, (acc.get(id) || 0) + qw * dw)));
    const out = [];
    acc.forEach((dot, id) => {
      // 文件長度要扣掉被遮掉的詞
      let ss = index.docNorm.get(id) ** 2;
      mask.forEach((t) => { const w = index.docWeights.get(id).get(t); if (w) ss -= w * w; });
      const dn = Math.sqrt(Math.max(ss, 1e-12));
      out.push({ id, score: dot / (qn * dn) });
    });
    out.sort((a, b) => b.score - a.score || (a.id < b.id ? -1 : 1));
    return out;
  }

  // 分數拆解：回傳查詢詞 × 文件詞的貢獻矩陣（文件詞取 |貢獻| 最大的前 topN 個）
  function decompose(qTerms, docId, dv, topN) {
    const { model, index, D } = dv;
    const q = queryVector(qTerms, dv);
    const n = dv.pos.get(docId);
    const dNorm = dv.norm[n];
    if (!q.norm || !dNorm) return null;
    const docTerms = [];
    index.docWeights.get(docId).forEach((w, t) => {
      const v = model.lookup(t);
      if (v !== undefined) docTerms.push([t, w, v]);
    });
    const cells = docTerms.map(([t, w, v]) => {
      const row = q.used.map(([qt, qw]) => {
        const qi = model.lookup(qt) * D, vj = v * D;
        let c = 0;
        for (let k = 0; k < D; k++) c += model.vecs[qi + k] * model.vecs[vj + k];
        return { cos: c, contrib: (qw * w * c) / (q.norm * dNorm) };
      });
      return { term: t, weight: w, row, total: row.reduce((s, x) => s + x.contrib, 0) };
    });
    const score = cells.reduce((s, c) => s + c.total, 0);
    cells.sort((a, b) => Math.abs(b.total) - Math.abs(a.total));
    return {
      score, queryTerms: q.used, oov: q.oov,
      docTerms: cells.slice(0, topN || 12), totalDocTerms: cells.length,
      shown: cells.slice(0, topN || 12).reduce((s, c) => s + c.total, 0),
    };
  }

  // 與查詢沒有任何共同詞的文件
  function sharesNoTerm(docId, qTerms, index) {
    const dw = index.docWeights.get(docId);
    return !qTerms.some((t) => dw.has(t));
  }

  global.EmbedCore = { buildDocVectors, queryVector, rank, rankTfidfMasked, decompose, sharesNoTerm };
})(typeof window !== "undefined" ? window : globalThis);
