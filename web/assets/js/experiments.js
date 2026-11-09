/*
 * experiments.js
 * --------------
 * 所有實驗的計算（網頁與 Node 腳本共用，數字完全一致）：
 *   1. 主要評估：每個查詢 × 每個方法的 P@10、MRR、nDCG@10
 *   2. 「沒有共同字」的文件：B 的前 10 名中與查詢沒有共同詞的文件
 *   3. 反義查詢：兩個意思相反的查詢，前 10 名重疊幾篇
 *   4. 遮字實驗：把查詢詞從所有文件中遮掉，再看各方法還能不能找到相關文件
 *
 * 環境 env = { docs, index, dv: { cbow, sg }, evalSet }
 * 對外公開為 Experiments。
 */
(function (global) {
  "use strict";

  const METHODS = [
    { id: "tfidf", label: "A：TF-IDF", short: "TF-IDF" },
    { id: "cbow", label: "B：TF-IDF 加權詞向量（CBOW）", short: "B-CBOW" },
    { id: "sg", label: "B：TF-IDF 加權詞向量（Skip-Gram）", short: "B-SG" },
  ];
  const K = 10;

  function rankBy(method, qTerms, env, mask) {
    if (method === "tfidf") return EmbedCore.rankTfidfMasked(qTerms, env.index, mask);
    return EmbedCore.rank(qTerms, env.dv[method], { mask });
  }

  // 主題 → { 文件 id: 1 }
  function qrelsFor(topic, env) {
    if (!env._qrels) env._qrels = {};
    if (!env._qrels[topic]) {
      const j = {};
      env.docs.forEach((d) => { if (d.topics.includes(topic)) j[d.id] = 1; });
      env._qrels[topic] = j;
    }
    return env._qrels[topic];
  }

  function metrics(ranked, judg) {
    const ids = ranked.map((r) => r.id);
    return {
      p10: EvalCore.precisionAt(ids, judg, K),
      rr: EvalCore.reciprocalRank(ids, judg),
      ndcg10: EvalCore.ndcgAt(ids, judg, K),
      ap: EvalCore.averagePrecision(ids, judg),
      retrieved: ids.length,
    };
  }

  // 1 + 2：主要評估
  function mainEval(env) {
    const rows = [];
    env.evalSet.queries.forEach((q) => {
      const qTerms = IRCore.terms(q.text);
      const judg = qrelsFor(q.topic, env);
      const row = { q, qTerms, R: EvalCore.numRelevant(judg), m: {}, top: {} };
      METHODS.forEach((M) => {
        const ranked = rankBy(M.id, qTerms, env);
        row.m[M.id] = metrics(ranked, judg);
        row.top[M.id] = ranked.slice(0, K).map((r) => ({
          id: r.id, score: r.score, rel: !!judg[r.id],
          noShared: EmbedCore.sharesNoTerm(r.id, qTerms, env.index),
        }));
      });
      rows.push(row);
    });
    return rows;
  }

  function summarize(rows, filter) {
    const sel = rows.filter(filter || (() => true));
    const out = {};
    METHODS.forEach((M) => {
      out[M.id] = {
        n: sel.length,
        p10: EvalCore.mean(sel.map((r) => r.m[M.id]), "p10"),
        mrr: EvalCore.mean(sel.map((r) => r.m[M.id]), "rr"),
        ndcg10: EvalCore.mean(sel.map((r) => r.m[M.id]), "ndcg10"),
        map: EvalCore.mean(sel.map((r) => r.m[M.id]), "ap"),
      };
    });
    return out;
  }

  // 「沒有共同字」的統計：B 前 10 名中與查詢沒有共同詞的文件數、其中相關的數量
  function zeroOverlapStats(rows) {
    const out = {};
    METHODS.forEach((M) => {
      let n = 0, rel = 0;
      rows.forEach((r) => r.top[M.id].forEach((d) => { if (d.noShared) { n++; if (d.rel) rel++; } }));
      out[M.id] = { n, rel, total: rows.length * K };
    });
    return out;
  }

  // 3：反義查詢
  function antonymEval(env) {
    return env.evalSet.antonymPairs.map((p) => {
      const ta = IRCore.terms(p.a), tb = IRCore.terms(p.b);
      const r = { p, m: {} };
      METHODS.forEach((M) => {
        const ra = rankBy(M.id, ta, env).slice(0, K).map((x) => x.id);
        const rb = rankBy(M.id, tb, env).slice(0, K).map((x) => x.id);
        r.m[M.id] = { overlap: EvalCore.overlapAt(ra, rb, K), a: ra, b: rb };
      });
      return r;
    });
  }

  // 4：遮字實驗
  //   mode = "all"：遮掉查詢的所有詞（TF-IDF 必然找不到任何文件）
  //   mode = "key"：只遮掉一個「關鍵詞」＝在相關文件中出現最多次的查詢詞
  function keyTerm(qTerms, judg, env) {
    let best = null, bestN = -1;
    Array.from(new Set(qTerms)).forEach((t) => {
      const post = env.index.postings.get(t) || [];
      const n = post.filter(([id]) => judg[id]).length;
      if (n > bestN) { best = t; bestN = n; }
    });
    return best;
  }

  function maskEval(env, mode) {
    return env.evalSet.queries.map((q) => {
      const qTerms = IRCore.terms(q.text);
      const judg = qrelsFor(q.topic, env);
      const mask = new Set(mode === "all" ? qTerms : [keyTerm(qTerms, judg, env)]);
      const row = { q, qTerms, mask: Array.from(mask), before: {}, after: {} };
      // 被遮掉的詞實際影響了幾篇相關文件
      row.affectedRel = Object.keys(judg).filter((id) => Array.from(mask).some((t) => env.index.docWeights.get(id).has(t))).length;
      METHODS.forEach((M) => {
        row.before[M.id] = metrics(rankBy(M.id, qTerms, env), judg);
        row.after[M.id] = metrics(rankBy(M.id, qTerms, env, mask), judg);
      });
      return row;
    });
  }

  // 隨機排序的期望 P@10（＝相關文件比例），作為遮字實驗的基準線
  function randomBaseline(topic, env) {
    return EvalCore.numRelevant(qrelsFor(topic, env)) / env.docs.length;
  }

  global.Experiments = {
    METHODS, K, rankBy, qrelsFor, metrics, mainEval, summarize, zeroOverlapStats,
    antonymEval, maskEval, keyTerm, randomBaseline,
  };
})(typeof window !== "undefined" ? window : globalThis);
