/*
 * ir-core.js
 * ----------
 * 檢索系統的核心：斷句、斷詞、停用詞，以及 TF-IDF 與 cosine 計分。
 * 所有方法（TF-IDF、Word2Vec 訓練、詞向量文件表示）都使用這一套前處理：
 *   小寫 → 斷詞（去標點、去數字）→ 去停用詞；不做詞幹還原。
 *
 * 純 JavaScript，網頁與 Node 都能使用。對外公開為 IRCore。
 */
(function (global) {
  "use strict";

  // -------------------------------------------------------------------
  // 停用詞（127 個 SMART／標準 IR 停用詞）
  // -------------------------------------------------------------------
  const STOPWORDS = new Set(`
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
said says also will
`.split(/\s+/).filter(Boolean));

  // -------------------------------------------------------------------
  // 斷句：句點／問號／驚嘆號後接空白與大寫字母（或段落換行）才算句子結束。
  // 縮寫（U.S.、Inc.、Corp. …）不切。
  // -------------------------------------------------------------------
  const ABBR = new Set(["u.s", "u.k", "inc", "corp", "co", "ltd", "dr", "mr", "mrs", "ms",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "no", "vs", "st", "jr", "sr", "e.c", "u.n", "bhd", "plc", "dlr", "pct"]);

  function splitSentences(text) {
    const paras = String(text || "").split(/\n\s{2,}|\n\s*\n/);
    const out = [];
    paras.forEach((p) => {
      p = p.replace(/\s+/g, " ").trim();
      if (!p) return;
      let start = 0;
      const re = /[.!?]+(?=\s+["'(]?[A-Z0-9])/g;
      let m;
      while ((m = re.exec(p)) !== null) {
        const before = p.slice(start, m.index);
        const last = (before.match(/([A-Za-z.]+)$/) || [""])[0].toLowerCase().replace(/\.$/, "");
        if (ABBR.has(last) || /^[a-z]$/.test(last)) continue;   // 縮寫或單一大寫字母
        const s = p.slice(start, m.index + m[0].length).trim();
        if (s) out.push(s);
        start = m.index + m[0].length;
      }
      const tail = p.slice(start).trim();
      if (tail) out.push(tail);
    });
    return out;
  }

  // -------------------------------------------------------------------
  // 斷詞：小寫；縮寫 U.S.、U.K.、E.C. 保留為一個詞（u.s.）；
  // 其他詞必須以字母開頭（純數字不會成為詞），字中可含連字號與撇號。
  // -------------------------------------------------------------------
  const TOKEN_RE = /[A-Za-z](?:\.[A-Za-z])+\.?|[A-Za-z][A-Za-z0-9'\-]*[A-Za-z0-9]|[A-Za-z]/g;

  function tokenize(text) {
    const out = [];
    let m;
    TOKEN_RE.lastIndex = 0;
    while ((m = TOKEN_RE.exec(text)) !== null) {
      let t = m[0].toLowerCase();
      if (t.includes(".")) { t = t.replace(/\.?$/, "."); }       // u.s → u.s.
      else t = t.replace(/'s$/, "").replace(/'$/, "");            // 所有格
      out.push(t);
    }
    return out;
  }

  function removeStopwords(tokens) {
    return tokens.filter((t) => t.length > 1 && !STOPWORDS.has(t));
  }

  // 統一的前處理：文字 → 索引詞
  function terms(text) {
    return removeStopwords(tokenize(text || ""));
  }

  // -------------------------------------------------------------------
  // TF-IDF（方法 A）
  //   TF  = 1 + ln(tf)（tf > 0）
  //   IDF = ln(N / df)
  //   文件與查詢使用同一套權重，以 cosine（L2 正規化後內積）計分
  // -------------------------------------------------------------------
  const tfWeight = (tf) => (tf > 0 ? 1 + Math.log(tf) : 0);
  const idfWeight = (N, df) => (df > 0 ? Math.log(N / df) : 0);

  // docs: [{ id, terms: [...] }]
  function buildIndex(docs) {
    const N = docs.length;
    const df = new Map();
    const docTF = new Map();
    const docLen = new Map();
    docs.forEach((d) => {
      const c = new Map();
      d.terms.forEach((t) => c.set(t, (c.get(t) || 0) + 1));
      docTF.set(d.id, c);
      docLen.set(d.id, d.terms.length);
      c.forEach((_, t) => df.set(t, (df.get(t) || 0) + 1));
    });
    const idf = new Map();
    df.forEach((v, t) => idf.set(t, idfWeight(N, v)));

    const postings = new Map();   // term → [[docId, weight], ...]
    const docWeights = new Map(); // docId → Map(term → weight)
    const docNorm = new Map();
    docs.forEach((d) => {
      const w = new Map();
      let ss = 0;
      docTF.get(d.id).forEach((tf, t) => {
        const x = tfWeight(tf) * idf.get(t);
        if (!x) return;
        w.set(t, x);
        ss += x * x;
        if (!postings.has(t)) postings.set(t, []);
        postings.get(t).push([d.id, x]);
      });
      docWeights.set(d.id, w);
      docNorm.set(d.id, Math.sqrt(ss) || 1);
    });
    return { N, df, idf, docTF, docLen, docWeights, docNorm, postings, ids: docs.map((d) => d.id) };
  }

  // 查詢向量：與文件相同的 TF × IDF；不在詞彙中的詞忽略
  function queryWeights(qTerms, index) {
    const c = new Map();
    qTerms.forEach((t) => c.set(t, (c.get(t) || 0) + 1));
    const w = new Map();
    c.forEach((tf, t) => {
      const idf = index.idf.get(t);
      if (idf) w.set(t, tfWeight(tf) * idf);
    });
    return w;
  }

  // cosine 排名；回傳 [{ id, score }]（分數相同時依 id 排序，結果可重現）
  function rankTfidf(qTerms, index) {
    const q = queryWeights(qTerms, index);
    let qn = 0;
    q.forEach((x) => { qn += x * x; });
    qn = Math.sqrt(qn) || 1;
    const acc = new Map();
    q.forEach((qw, t) => {
      (index.postings.get(t) || []).forEach(([id, dw]) => acc.set(id, (acc.get(id) || 0) + qw * dw));
    });
    const out = [];
    acc.forEach((dot, id) => out.push({ id, score: dot / (qn * index.docNorm.get(id)) }));
    out.sort((a, b) => b.score - a.score || (a.id < b.id ? -1 : 1));
    return out;
  }

  global.IRCore = {
    STOPWORDS, splitSentences, tokenize, removeStopwords, terms,
    tfWeight, idfWeight, buildIndex, queryWeights, rankTfidf,
  };
})(typeof window !== "undefined" ? window : globalThis);
