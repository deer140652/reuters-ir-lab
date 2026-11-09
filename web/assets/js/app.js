/*
 * app.js
 * ------
 * 網頁主程式：載入資料 → 建立 TF-IDF 索引與文件向量 → 執行全部實驗 → 繪製各區塊。
 * 計算都在 ir-core.js、embed-core.js、experiments.js（與 Node 腳本共用）。
 */
(function () {
  "use strict";
  const { esc, f3 } = Charts;
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const fmtInt = (n) => Number(n).toLocaleString("en-US");
  const pct = (x, d) => (x * 100).toFixed(d == null ? 1 : d) + "%";

  const TOPIC_ZH = { earn: "財報", acq: "併購", crude: "原油", grain: "穀物", trade: "貿易" };
  const TOPICS5 = ["earn", "acq", "crude", "grain", "trade"];
  const MHEX = { tfidf: "#eb6834", cbow: "#1baf7a", sg: "#2a78d6" };
  const MNAME = { tfidf: "TF-IDF", cbow: "CBOW", sg: "Skip-Gram" };
  const GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300", "#e34948"];
  const DEMO_QUERIES = ["C4", "C1", "G4", "A4", "T1", "E5"];

  const S = { search: { method: "sg", ranked: [], qTerms: [], topic: "", selected: null } };

  // -------------------------------------------------------------------
  // 載入
  // -------------------------------------------------------------------
  async function fetchJSON(u) { const r = await fetch(u); if (!r.ok) throw new Error(u + " " + r.status); return r.json(); }
  async function loadModel(kind) {
    const [meta, buf] = await Promise.all([
      fetchJSON(`data/w2v-${kind}.json`),
      fetch(`data/w2v-${kind}.bin`).then((r) => { if (!r.ok) throw new Error(kind + ".bin"); return r.arrayBuffer(); }),
    ]);
    return Word2Vec.fromStored(meta, new Float32Array(buf));
  }

  async function init() {
    const [corpus, evalSet, notes, param, cbow, sg] = await Promise.all([
      fetchJSON("data/corpus.json"), fetchJSON("data/eval-set.json"), fetchJSON("data/neighbor-notes.json"),
      fetchJSON("data/param-sensitivity.json"), loadModel("cbow"), loadModel("sg"),
    ]);
    S.param = param;
    S.corpus = corpus; S.docs = corpus.documents; S.evalSet = evalSet; S.notes = notes;
    S.byId = new Map(S.docs.map((d) => [d.id, d]));
    S.models = { cbow, sg };
    S.index = IRCore.buildIndex(S.docs.map((d) => ({ id: d.id, terms: IRCore.terms(d.title + ".\n  " + d.body) })));
    S.dv = { cbow: EmbedCore.buildDocVectors(S.index, cbow), sg: EmbedCore.buildDocVectors(S.index, sg) };
    S.env = { docs: S.docs, index: S.index, dv: S.dv, evalSet };
    S.rows = Experiments.mainEval(S.env);
    S.ant = Experiments.antonymEval(S.env);
    S.mask = { key: Experiments.maskEval(S.env, "key"), all: Experiments.maskEval(S.env, "all") };
    S.sum = {
      all: Experiments.summarize(S.rows),
      keyword: Experiments.summarize(S.rows, (r) => r.q.type === "keyword"),
      paraphrase: Experiments.summarize(S.rows, (r) => r.q.type === "paraphrase"),
    };

    renderOverview();
    initSearch();
    renderW2V();
    renderEval();
    renderParam();
    renderTopic();
    renderConclusion();
    $("#loading").style.display = "none";
    initNavHighlight();
  }

  // -------------------------------------------------------------------
  // 共用
  // -------------------------------------------------------------------
  const avgOverlap = (k) => S.ant.reduce((s, a) => s + a.m[k].overlap, 0) / S.ant.length;
  const maskAvg = (mode, when, k) => S.mask[mode].reduce((s, r) => s + r[when][k].ndcg10, 0) / S.mask[mode].length;
  const randomBase = () => S.rows.reduce((s, r) => s + Experiments.randomBaseline(r.q.topic, S.env), 0) / S.rows.length;

  function topicTags(d) {
    return d.topics.length
      ? d.topics.map((t) => `<span class="topic-tag${TOPICS5.includes(t) ? " t5" : ""}">${esc(t)}</span>`).join(" ")
      : `<span class="topic-tag">無主題</span>`;
  }
  function docTitle(d) { return d.title || d.body.split("\n")[0].slice(0, 90); }

  function highlight(text, terms) {
    if (!terms.length) return esc(text);
    const set = new Set(terms);
    let out = "", last = 0, m;
    const re = /[A-Za-z](?:\.[A-Za-z])+\.?|[A-Za-z][A-Za-z0-9'\-]*[A-Za-z0-9]|[A-Za-z]/g;
    while ((m = re.exec(text)) !== null) {
      const tok = IRCore.tokenize(m[0])[0];
      if (tok && set.has(tok)) {
        out += esc(text.slice(last, m.index)) + "<mark>" + esc(m[0]) + "</mark>";
        last = m.index + m[0].length;
      }
    }
    return out + esc(text.slice(last));
  }

  function snippet(d, terms) {
    if (!d.body.trim()) return "";
    const set = new Set(terms);
    const sents = IRCore.splitSentences(d.body);
    let s = sents.find((x) => IRCore.terms(x).some((t) => set.has(t))) || sents[0] || "";
    if (s.length > 220) s = s.slice(0, 220) + "…";
    return highlight(s, terms);
  }

  const rankFor = (method, qTerms, mask) => Experiments.rankBy(method, qTerms, S.env, mask);

  // 查詢詞在文件中最相近的詞（文件中沒有出現的查詢詞才列出）
  function semMatches(qTerms, docId, method) {
    if (method === "tfidf") return [];
    const model = S.models[method], dw = S.index.docWeights.get(docId), D = model.dim;
    const out = [];
    Array.from(new Set(qTerms)).forEach((qt) => {
      if (dw.has(qt)) return;
      const qi = model.lookup(qt);
      if (qi === undefined) return;
      let best = null, bs = -2;
      dw.forEach((_, t) => {
        const ti = model.lookup(t);
        if (ti === undefined) return;
        let c = 0;
        for (let k = 0; k < D; k++) c += model.vecs[qi * D + k] * model.vecs[ti * D + k];
        if (c > bs) { bs = c; best = t; }
      });
      if (best) out.push({ q: qt, d: best, cos: bs });
    });
    return out;
  }

  // -------------------------------------------------------------------
  // 01 資料集與實驗設計
  // -------------------------------------------------------------------
  function renderOverview() {
    const st = S.corpus.stats, m = S.models.sg;
    $("#ov-stats").innerHTML = [
      [fmtInt(st.test_docs), "測試集新聞：檢索的對象"],
      [fmtInt(st.train_docs), "訓練集新聞：學習詞向量"],
      [fmtInt(m.stats.trainTokens), "詞向量的訓練詞數"],
      [fmtInt(m.vocab.length), "詞向量詞彙（出現 ≥ 5 次）"],
    ].map(([v, l]) => `<div class="stat-block"><strong>${v}</strong><span>${l}</span></div>`).join("");
    $("#ov-topics").innerHTML = "測試集中五個主題的篇數：" + TOPICS5.map((t) => `<strong>${t}</strong>（${TOPIC_ZH[t]}）${fmtInt(st.test_topics[t])}`).join("、") +
      `；其餘 ${fmtInt(st.test_topics.other + st.test_topics.none)} 篇屬於其他主題或沒有主題。`;
    const c = S.models.cbow, s = S.models.sg;
    $("#w2v-train-line").textContent = `Word2Vec（CBOW、Skip-Gram）只用訓練集訓練，兩個模型參數相同：維度 ${s.params.dim}、視窗 ${s.params.window}、負樣本 ${s.params.negative}、${s.params.epochs} epochs。` +
      `CBOW 訓練樣本 ${fmtInt(c.examples)}、${(c.elapsed / 1000).toFixed(0)} 秒；Skip-Gram ${fmtInt(s.examples)}、${(s.elapsed / 1000).toFixed(0)} 秒（約 ${(s.examples / c.examples).toFixed(1)} 倍的樣本）。`;
  }

  // -------------------------------------------------------------------
  // 02 新聞檢索 + 分數拆解
  // -------------------------------------------------------------------
  function initSearch() {
    const qs = DEMO_QUERIES.map((id) => S.evalSet.queries.find((q) => q.id === id));
    $("#search-chips").innerHTML = qs.map((q) =>
      `<button type="button" class="chip" data-q="${esc(q.text)}">${esc(q.text)}</button>`).join("");
    $("#search-chips").addEventListener("click", (e) => {
      const b = e.target.closest(".chip"); if (!b) return;
      $("#search-input").value = b.dataset.q; doSearch();
    });
    $("#search-form").addEventListener("submit", (e) => { e.preventDefault(); doSearch(); });
    $("#search-method").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      $$("#search-method button").forEach((x) => x.classList.toggle("on", x === b));
      S.search.method = b.dataset.m; doSearch(S.search.selected);
    });
    $("#results").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-act=doc]");
      if (btn) { openDoc(btn.dataset.id); return; }
      const r = e.target.closest(".result"); if (!r) return;
      selectDoc(r.dataset.id);
    });
    $("#modal-close").addEventListener("click", closeModal);
    $("#modal-back").addEventListener("click", (e) => { if (e.target.id === "modal-back") closeModal(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });
    $("#search-input").value = "petroleum supply";
    doSearch();
  }

  function doSearch(keepId) {
    const text = $("#search-input").value.trim();
    const qTerms = IRCore.terms(text);
    const q = S.evalSet.queries.find((x) => x.text === text.toLowerCase());
    Object.assign(S.search, { qTerms, topic: q ? q.topic : "" });
    S.search.ranked = qTerms.length ? rankFor(S.search.method, qTerms).slice(0, 10) : [];
    renderResults();
    const ids = S.search.ranked.map((r) => r.id);
    let pick = keepId && ids.includes(keepId) ? keepId : null;
    if (!pick) {
      const judg = S.search.topic ? Experiments.qrelsFor(S.search.topic, S.env) : null;
      const ok = (r, needRel) => EmbedCore.sharesNoTerm(r.id, qTerms, S.index) && S.index.docWeights.get(r.id).size >= 15 && (!needRel || (judg && judg[r.id]));
      const best = S.search.ranked.find((r) => ok(r, true)) || S.search.ranked.find((r) => ok(r, false)) || S.search.ranked[0];
      pick = best ? best.id : null;
    }
    selectDoc(pick);
  }

  function renderResults() {
    const { ranked, qTerms, method, topic } = S.search;
    const judg = topic ? Experiments.qrelsFor(topic, S.env) : null;
    if (!qTerms.length) { $("#result-count").textContent = "請輸入至少一個非停用詞。"; $("#results").innerHTML = ""; return; }
    const nTf = rankFor("tfidf", qTerms).length;
    $("#result-count").innerHTML = `${MNAME[method]} 前 10 名` +
      (method === "tfidf" ? `（共 ${fmtInt(nTf)} 篇含有查詢詞）` : `（TF-IDF 只能從 ${fmtInt(nTf)} 篇含有查詢詞的新聞中排序）`) +
      (judg ? `　·　屬於 <strong>${topic}</strong> 的有 ${ranked.filter((r) => judg[r.id]).length} 篇` : "");
    $("#results").innerHTML = ranked.map((r, i) => {
      const d = S.byId.get(r.id);
      const noShare = EmbedCore.sharesNoTerm(r.id, qTerms, S.index);
      const sm = semMatches(qTerms, r.id, method);
      return `<article class="result" data-id="${d.id}">
        <div class="result-head"><span class="result-rank">${i + 1}</span>
          <span class="result-title">${highlight(docTitle(d), qTerms)}</span>
          <span class="result-score">${r.score.toFixed(3)}</span></div>
        <div class="result-meta">${topicTags(d)}
          ${judg ? (judg[r.id] ? `<span class="badge rel">相關</span>` : `<span class="badge nrel">不相關</span>`) : ""}
          ${noShare ? `<span class="badge noshare">無共同字</span>` : ""}
          ${sm.map((x) => `<span class="sem-match">${esc(x.q)} ≈ ${esc(x.d)} ${x.cos.toFixed(2)}</span>`).join(" ")}</div>
        <div class="result-snippet">${snippet(d, qTerms)}</div>
        <div class="result-actions"><button type="button" class="link" data-act="doc" data-id="${d.id}">全文</button></div>
      </article>`;
    }).join("") || `<p class="empty-state">${method === "tfidf" ? "沒有任何新聞含有這些查詢詞。" : "查詢詞都沒有詞向量（訓練集出現不到 5 次）。"}</p>`;
  }

  // 全文視窗：標題、日期、主題，內文中的查詢詞反白
  function openDoc(id) {
    const d = S.byId.get(id), q = S.search.qTerms;
    $("#modal-title").innerHTML = highlight(docTitle(d), q);
    $("#modal-meta").innerHTML = `NEWID ${esc(d.id)}　·　${esc(d.date)}　·　${topicTags(d)}`;
    $("#modal-body").innerHTML = d.body.trim() ? highlight(d.body, q) : `<span class="muted">（這篇是只有標題的簡訊，原始資料沒有內文。）</span>`;
    $("#modal-back").classList.add("open");
  }
  function closeModal() { $("#modal-back").classList.remove("open"); }

  function selectDoc(docId) {
    S.search.selected = docId;
    $$("#results .result").forEach((el) => el.classList.toggle("active", el.dataset.id === docId));
    renderPanel(docId);
  }

  function renderPanel(docId) {
    const qTerms = S.search.qTerms;
    const model = S.search.method === "tfidf" ? "sg" : S.search.method;
    const clear = (msg) => {
      $("#panel-doc").innerHTML = msg;
      ["#calc-heat", "#calc-tfidf", "#calc-docvec"].forEach((s) => { $(s).innerHTML = ""; });
      ["#calc-heat-meta", "#calc-tfidf-sum", "#calc-docvec-meta"].forEach((s) => { $(s).textContent = ""; });
      $("#calc-finding").innerHTML = "";
    };
    if (!docId || !qTerms.length) { clear(`<span class="muted">沒有可分析的新聞。</span>`); return; }
    const d = S.byId.get(docId), idx = S.index, dv = S.dv[model];
    const rA = rankFor("tfidf", qTerms), rB = rankFor(model, qTerms);
    const posA = rA.findIndex((r) => r.id === docId), posB = rB.findIndex((r) => r.id === docId);
    const dw = idx.docWeights.get(docId);

    // TF-IDF 逐步計算
    const dtf = idx.docTF.get(docId), qw = IRCore.queryWeights(qTerms, idx);
    const qtf = new Map(); qTerms.forEach((t) => qtf.set(t, (qtf.get(t) || 0) + 1));
    const rows = Array.from(qtf.keys());
    Array.from(dw.entries()).sort((x, y) => y[1] - x[1]).slice(0, 8).forEach(([t]) => { if (!rows.includes(t)) rows.push(t); });
    let dot = 0, qn = 0;
    qw.forEach((x) => { qn += x * x; });
    qn = Math.sqrt(qn);
    let h = `<table class="data"><thead><tr><th>詞</th><th class="num">tf<sub>q</sub></th><th class="num">tf<sub>d</sub></th><th class="num">df</th><th class="num">idf</th><th class="num">w<sub>q</sub></th><th class="num">w<sub>d</sub></th><th class="num">w<sub>q</sub>·w<sub>d</sub></th></tr></thead><tbody>`;
    rows.forEach((t) => {
      const df = idx.df.get(t) || 0, idf = idx.idf.get(t) || 0, wq = qw.get(t) || 0, wd = dw.get(t) || 0;
      dot += wq * wd;
      h += `<tr${qtf.has(t) ? ' style="background:#FFF6D6"' : ""}><td class="mono">${qtf.has(t) ? "<strong>" + esc(t) + "</strong>" : esc(t)}</td><td class="num">${qtf.get(t) || 0}</td><td class="num">${dtf.get(t) || 0}</td><td class="num">${df}</td><td class="num">${df ? idf.toFixed(2) : "–"}</td><td class="num">${wq ? wq.toFixed(2) : 0}</td><td class="num">${wd ? wd.toFixed(2) : 0}</td><td class="num">${wq * wd ? (wq * wd).toFixed(2) : 0}</td></tr>`;
    });
    $("#calc-tfidf").innerHTML = h + `</tbody></table>`;
    const dn = idx.docNorm.get(docId), cosA = qn ? dot / (qn * dn) : 0;
    $("#calc-tfidf-sum").textContent = `tf 權重 = 1 + ln(tf)，idf = ln(${fmtInt(idx.N)} / df)\ncos(q, d) = ${dot.toFixed(3)} / (${qn.toFixed(3)} × ${dn.toFixed(3)}) = ${cosA.toFixed(4)}` +
      `　（|d| 用全部 ${dw.size} 個詞計算）`;

    // 文件向量組成
    const m = S.models[model];
    const top = Array.from(dw.entries()).sort((x, y) => y[1] - x[1]).slice(0, 10);
    Charts.hbar($("#calc-docvec"), top.map(([t, w]) => {
      const has = m.lookup(t) !== undefined;
      return { label: t, value: w, color: has ? "#1F3A5F" : "#C9C4B5", tip: `<b>${esc(t)}</b><br>TF-IDF 權重 ${w.toFixed(3)}<br>${has ? "有詞向量" : "沒有詞向量"}` };
    }), { width: 460 });
    $("#calc-docvec-meta").textContent = `有詞向量的詞佔這篇新聞 TF-IDF 權重的 ${pct(dv.coverage[dv.pos.get(docId)])}。`;

    // 分數拆解熱圖
    const dec = EmbedCore.decompose(qTerms, docId, dv, 8);
    $("#panel-doc").innerHTML = `<strong>${esc(docTitle(d).slice(0, 90))}</strong><br>${topicTags(d)}　` +
      `TF-IDF ${posA >= 0 ? `第 ${posA + 1} 名（${cosA.toFixed(3)}）` : "找不到"}　·　${MNAME[model]} 第 ${posB + 1} 名（${dec ? dec.score.toFixed(3) : "–"}）`;
    if (!dec) { $("#calc-heat").innerHTML = `<p class="empty-state">查詢詞都沒有詞向量。</p>`; return; }
    const cols = dec.docTerms.map((c) => c.term), rowsQ = dec.queryTerms.map(([t]) => t);
    const cells = rowsQ.map((qt, i) => dec.docTerms.map((c) => ({
      v: c.row[i].contrib, label: c.row[i].cos.toFixed(2),
      tip: `<b>${esc(qt)} × ${esc(c.term)}</b><br>cos = ${c.row[i].cos.toFixed(3)}<br>貢獻 ${c.row[i].contrib.toFixed(4)}（${pct(c.row[i].contrib / dec.score)}）`,
    })));
    Charts.heatmap($("#calc-heat"), rowsQ, cols, cells);
    $("#calc-heat-meta").textContent = `${MNAME[model]}｜格內數字為兩個詞向量的 cosine，顏色深淺為對分數的貢獻 w_q·w_d·cos / (|q||d|)。圖中 ${cols.length} 個詞佔分數的 ${pct(dec.shown / dec.score, 0)}，其餘 ${dec.totalDocTerms - cols.length} 個詞佔 ${pct(1 - dec.shown / dec.score, 0)}。` +
      (S.search.method === "tfidf" ? "（TF-IDF 模式下顯示 Skip-Gram 的拆解）" : "");
    const flat = [];
    dec.docTerms.forEach((c) => c.row.forEach((x, i) => flat.push({ q: rowsQ[i], d: c.term, ...x })));
    flat.sort((a, b) => b.contrib - a.contrib);
    const shared = qTerms.filter((t) => dw.has(t));
    $("#calc-finding").innerHTML = `<p>${shared.length ? `共同的字：${shared.map((t) => `<code>${esc(t)}</code>`).join(" ")}。` : "<strong>這篇新聞沒有任何查詢詞</strong>，TF-IDF 分數為 0。"}` +
      `詞向量的分數主要來自 ${flat.slice(0, 3).map((x) => `<code>${esc(x.q)} × ${esc(x.d)}</code>`).join("、")}，三組合計 ${pct(flat.slice(0, 3).reduce((s, x) => s + x.contrib, 0) / dec.score, 0)}：分數是很多組「還算接近」的詞累積出來的。</p>`;
  }

  // -------------------------------------------------------------------
  // 03 詞的鄰居
  // -------------------------------------------------------------------
  function nbChip(word, score, freq, cat) {
    return `<span class="nb ${cat || "none"}" title="訓練集出現 ${freq} 次">${esc(word)} <small>${score.toFixed(2)}</small></span>`;
  }

  function renderW2V() {
    const cats = S.notes.categories;
    $("#nb-legend").innerHTML = ["sem", "ctx", "odd"].map((c) => `<span><span class="nb ${c}">${cats[c].label}</span> ${esc(cats[c].desc)}</span>`).join("");
    let h = `<table class="data"><thead><tr><th>詞（主題）</th><th>CBOW</th><th>Skip-Gram</th><th style="min-width:320px">觀察</th></tr></thead><tbody>`;
    S.evalSet.neighborWords.forEach((w) => {
      const note = S.notes.words[w] || {};
      const cell = (k) => (S.models[k].mostSimilar(w, 5) || []).map((x) => nbChip(x.word, x.score, x.freq, note[k] && note[k][x.word])).join("");
      h += `<tr><td style="white-space:nowrap"><span class="nb-word">${esc(w)}</span><br><span class="small muted">${esc(note.topic || "")} · ${fmtInt(S.models.sg.freq[S.models.sg.lookup(w)])} 次</span></td><td>${cell("cbow")}</td><td>${cell("sg")}</td><td class="nb-note">${esc(note.brief || "")}</td></tr>`;
    });
    $("#nb-table").innerHTML = h + `</tbody></table><p class="small muted">${esc(S.notes.note)}</p>`;

    const pairCat = (i) => (i < 5 ? "同類" : i < 10 ? "反義" : "無關");
    const pairs = S.evalSet.wordPairs;
    Charts.groupedBar($("#pair-chart"), {
      cats: pairs.map((p, i) => ({ label: p[0], lines: [p[0], p[1], pairCat(i)], tipLabel: `${p[0]} – ${p[1]}（${pairCat(i)}）` })),
      series: ["cbow", "sg"].map((k) => ({ name: MNAME[k], color: MHEX[k], values: pairs.map((p) => S.models[k].similarity(p[0], p[1]) || 0) })),
      max: 1, height: 320, width: 640, fmt: (v) => v.toFixed(3),
    });

    $("#viz-go").addEventListener("click", renderViz);
    $("#viz-model").addEventListener("change", renderViz);
    $("#viz-words").addEventListener("keydown", (e) => { if (e.key === "Enter") renderViz(); });
    $("#viz-mode").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      $$("#viz-mode button").forEach((x) => x.classList.toggle("on", x === b)); renderViz();
    });
    renderViz();
  }

  function renderViz() {
    const k = $("#viz-model").value, model = S.models[k];
    const mode = $$("#viz-mode button").find((b) => b.classList.contains("on")).dataset.v;
    const words = $("#viz-words").value.split(/[,，\s]+/).map((x) => x.trim().toLowerCase()).filter(Boolean);
    const items = [], seen = new Set(), missing = [];
    words.forEach((w) => {
      const id = model.lookup(w);
      if (id === undefined) { missing.push(w); return; }
      const group = items.filter((x) => x.isQuery).length;
      if (!seen.has(id)) { seen.add(id); items.push({ id, word: w, group, isQuery: true }); }
      (model.mostSimilar(w, 5) || []).forEach((r) => {
        if (seen.has(r.id)) return;
        seen.add(r.id);
        items.push({ id: r.id, word: r.word, group, isQuery: false, tip: `<b>${esc(r.word)}</b><br>${esc(w)} 的鄰居，cos ${r.score.toFixed(3)}` });
      });
    });
    if (items.length < 4) { $("#viz-chart").innerHTML = `<p class="empty-state">至少需要 4 個詞。</p>`; return; }
    const X = items.map((it) => Array.from(model.vectorOf(it.id)));
    let pts, meta;
    if (mode === "pca") {
      const p = Charts.pca2D(X); pts = p.pts;
      meta = `PCA 前兩個主成分保留 ${pct(p.explained[0] + p.explained[1], 0)} 的變異，呈現整體方向，群集較容易重疊。`;
    } else {
      pts = Charts.tsne2D(X, 8, 20261006);
      meta = "t-SNE（perplexity 8）保留局部的鄰近關係，群與群之間的距離沒有意義。";
    }
    Charts.scatter($("#viz-chart"), pts, items, GROUP_COLORS);
    $("#viz-legend").innerHTML = items.filter((x) => x.isQuery).map((g) => `<span><span class="sw" style="background:${GROUP_COLORS[g.group % GROUP_COLORS.length]}"></span>${esc(g.word)}</span>`).join("") +
      (missing.length ? `<span class="muted">不在詞彙中：${missing.map(esc).join("、")}</span>` : "");
    $("#viz-meta").textContent = `${MNAME[k]}｜${meta}`;
  }

  // -------------------------------------------------------------------
  // 04 檢索效果
  // -------------------------------------------------------------------
  function renderEval() {
    const M = Experiments.METHODS;
    const groups = [["全部查詢", "all"], ["一般查詢：用常見字", "keyword"], ["換字查詢：避開常見字", "paraphrase"]];
    const metrics = [["p10", "P@10"], ["mrr", "MRR"], ["ndcg10", "nDCG@10"]];
    const sep = ' style="border-left:1px solid var(--line)"';
    let h = `<table class="data"><thead><tr><th rowspan="2">查詢</th><th rowspan="2" class="num">數量</th>${metrics.map(([, l]) => `<th colspan="3" style="text-align:center;border-left:1px solid var(--line)">${l}</th>`).join("")}</tr><tr>${metrics.map(() => M.map((m, i) => `<th class="num"${i === 0 ? sep : ""}>${MNAME[m.id]}</th>`).join("")).join("")}</tr></thead><tbody>`;
    groups.forEach(([label, key]) => {
      const s = S.sum[key];
      h += `<tr><td>${label}</td><td class="num">${s.tfidf.n}</td>`;
      metrics.forEach(([k]) => {
        const best = Math.max(...M.map((m) => +s[m.id][k].toFixed(3)));
        M.forEach((m, i) => { const v = s[m.id][k]; h += `<td class="num${+v.toFixed(3) === best ? " best" : ""}"${i === 0 ? sep : ""}>${f3(v)}</td>`; });
      });
      h += `</tr>`;
    });
    $("#eval-main").innerHTML = h + `</tbody></table><p class="small muted">P@10：前 10 名中相關的比例。MRR：第一篇相關新聞名次的倒數。nDCG@10：考慮名次的前 10 名品質。粗體為該列最高。</p>`;
    const { keyword: sK, paraphrase: sP } = S.sum;
    $("#eval-finding").innerHTML = `<p><strong>差距幾乎全部來自換字查詢。</strong>用常見字查詢時，TF-IDF（${f3(sK.tfidf.ndcg10)}）與 Skip-Gram（${f3(sK.sg.ndcg10)}）幾乎相同，CBOW（${f3(sK.cbow.ndcg10)}）略低；查詢用字和新聞不同時，TF-IDF 掉到 ${f3(sP.tfidf.ndcg10)}，兩種詞向量都維持在 0.9 左右。</p>`;

    // 勝負圖
    const items = S.rows.map((r) => ({
      label: `${r.q.type === "paraphrase" ? "★ " : ""}${r.q.text}`,
      value: r.m.sg.ndcg10 - r.m.tfidf.ndcg10,
      tip: `<b>${esc(r.q.text)}</b>（${r.q.topic}）<br>TF-IDF ${f3(r.m.tfidf.ndcg10)} → Skip-Gram ${f3(r.m.sg.ndcg10)}`,
    })).sort((a, b) => b.value - a.value);
    Charts.divergingBar($("#wl-chart"), items, { pos: MHEX.sg, neg: MHEX.tfidf, labelWidth: 250, width: 620 });

    // 無共同字
    const zo = Experiments.zeroOverlapStats(S.rows).sg;
    $("#zo-stats").innerHTML = [
      [`${zo.n} 篇`, `佔前 10 名的 ${pct(zo.n / zo.total, 0)}（25 個查詢共 ${zo.total} 篇）`],
      [pct(zo.rel / zo.n, 0), `是相關的新聞（${zo.rel} 篇）`],
    ].map(([v, l]) => `<div class="stat-block"><strong>${v}</strong><span>${l}</span></div>`).join("");

    // 排錯的例子
    const cases = [
      { id: "T1", method: "sg", text: "混入「經常帳」與「匯率」的新聞（bop、money-fx）。deficit ≈ surplus，而 trade 和 current account 常出現在同一則總體經濟新聞裡。" },
      { id: "C4", method: "tfidf", text: "反過來，TF-IDF 被字面騙了：British Petroleum、Dome Petroleum 的併購新聞，以及 money supply（貨幣供給）。" },
    ];
    $("#fail-cases").innerHTML = cases.map((c) => {
      const r = S.rows.find((x) => x.q.id === c.id);
      const wrong = r.top[c.method].map((d, i) => ({ d, i })).filter((x) => !x.d.rel).slice(0, 4);
      return `<div class="fail-case"><p><strong>「${esc(r.q.text)}」· ${MNAME[c.method]}</strong>　<span class="muted">nDCG@10 ${f3(r.m[c.method].ndcg10)}</span></p><p>${c.text}</p>
        <ul class="mini-list">${wrong.map(({ d, i }) => { const doc = S.byId.get(d.id); return `<li><span class="r">${i + 1}</span><span class="t">${esc(docTitle(doc).slice(0, 62))}</span><span class="small muted">${esc(doc.topics.slice(0, 2).join(","))}</span></li>`; }).join("")}</ul></div>`;
    }).join("");
  }


  // 架構與參數敏感度（數字由 scripts/param_sensitivity.js 離線計算）
  function renderParam() {
    const rows = S.param.rows, T = rows[0].tfidf;
    const keys = [["all", "全部"], ["keyword", "一般"], ["paraphrase", "換字"]];
    const best = {};
    ["cbow", "sg"].forEach((k) => keys.forEach(([g]) => { best[k + g] = Math.max(...rows.map((r) => +r.models[k].ndcg10[g].toFixed(3))); }));
    const sep = ' style="border-left:1px solid var(--line)"';
    let h = `<table class="data"><thead><tr><th rowspan="2">視窗</th><th rowspan="2">維度</th>` +
      `<th colspan="3" style="text-align:center;border-left:1px solid var(--line)">CBOW　nDCG@10</th><th colspan="3" style="text-align:center;border-left:1px solid var(--line)">Skip-Gram　nDCG@10</th>` +
      `<th rowspan="2" class="num"${sep}>Skip-Gram<br>− CBOW</th><th rowspan="2" class="num">訓練時間<br>CBOW／SG</th></tr><tr>` +
      ["cbow", "sg"].map(() => keys.map(([, l], i) => `<th class="num"${i === 0 ? sep : ""}>${l}</th>`).join("")).join("") + `</tr></thead><tbody>`;
    h += `<tr class="group-row"><td colspan="2">TF-IDF（參考）</td>` + ["cbow", "sg"].map(() => keys.map(([g], i) => `<td class="num"${i === 0 ? sep : ""}>${f3(T[g])}</td>`).join("")).join("") + `<td${sep}></td><td></td></tr>`;
    rows.forEach((r) => {
      h += `<tr><td class="num">${r.window}</td><td class="num">${r.dim}${r.isDefault ? " ＊" : ""}</td>` +
        ["cbow", "sg"].map((k) => keys.map(([g], i) => { const v = r.models[k].ndcg10[g]; return `<td class="num${+v.toFixed(3) === best[k + g] ? " best" : ""}"${i === 0 ? sep : ""}>${f3(v)}</td>`; }).join("")).join("") +
        `<td class="num"${sep}>${(r.models.sg.ndcg10.all - r.models.cbow.ndcg10.all >= 0 ? "+" : "")}${(r.models.sg.ndcg10.all - r.models.cbow.ndcg10.all).toFixed(3)}</td>` +
        `<td class="num">${r.models.cbow.seconds.toFixed(0)}／${r.models.sg.seconds.toFixed(0)} 秒</td></tr>`;
    });
    $("#param-table").innerHTML = h + `</tbody></table><p class="small muted">＊ 網站內建的預設模型。${esc(S.param.note)} 粗體為該欄最高。</p>`;

    const avg = (f) => { const v = rows.filter(f).flatMap((r) => [r.models.cbow.ndcg10.all, r.models.sg.ndcg10.all]); return v.reduce((a, b) => a + b, 0) / v.length; };
    const winGain = avg((r) => r.window === 5) - avg((r) => r.window === 2);
    const dimGain = avg((r) => r.dim === 100) - avg((r) => r.dim === 50);
    const gaps = rows.map((r) => r.models.sg.ndcg10.all - r.models.cbow.ndcg10.all);
    const kwBelow = rows.flatMap((r) => ["cbow", "sg"].map((k) => r.models[k].ndcg10.keyword < T.keyword)).filter(Boolean).length;
    const allAbove = rows.every((r) => r.models.cbow.ndcg10.all > T.all && r.models.sg.ndcg10.all > T.all);
    $("#param-finding").innerHTML = `<p><strong>視窗大小的影響大於架構。</strong>視窗由 2 增為 5，nDCG@10 平均提高 ${winGain.toFixed(3)}；維度由 50 增為 100 的平均變化為 ${(dimGain >= 0 ? "+" : "") + dimGain.toFixed(3)}；Skip-Gram 與 CBOW 的差距介於 ${Math.min(...gaps).toFixed(3)} 至 ${Math.max(...gaps).toFixed(3)}。較大的視窗納入更廣的上下文，詞向量更接近「主題」層次的相似度，正好符合以主題判斷相關性的檢索任務。</p>` +
      `<p>四組設定中 Skip-Gram 皆略高於 CBOW，但差距會隨設定改變（視窗 2、維度 100 時僅 ${gaps[1].toFixed(3)}），且只來自單一隨機種子與 25 個查詢。${allAbove ? "所有設定的整體分數都高於 TF-IDF；" : ""}但在一般查詢上，8 個模型中有 ${kwBelow} 個低於 TF-IDF（${f3(T.keyword)}），視窗 2 的模型全部低於 TF-IDF。詞向量的效果取決於訓練設定，不能預設某一種架構或參數一定較好。</p>`;
  }

  // -------------------------------------------------------------------
  // 05 主題與字
  // -------------------------------------------------------------------
  function renderTopic() {
    const M = Experiments.METHODS;
    $("#ant-legend").innerHTML = M.map((m) => `<span><span class="sw" style="background:${MHEX[m.id]}"></span>${MNAME[m.id]}</span>`).join("");
    Charts.groupedBar($("#ant-chart"), {
      cats: S.ant.map((a) => ({ label: a.p.a, lines: [a.p.a, a.p.b], tipLabel: `${a.p.a} ／ ${a.p.b}` })),
      series: M.map((m) => ({ name: MNAME[m.id], color: MHEX[m.id], values: S.ant.map((a) => a.m[m.id].overlap) })),
      max: 10, ticks: [0, 2, 4, 6, 8, 10], tickFmt: (t) => t, height: 290, width: 640, showValues: true, valueFmt: (v) => v, fmt: (v) => `${v} / 10 篇相同`,
    });
    $("#ant-finding").innerHTML = `7 組反義查詢平均有 <strong>${avgOverlap("sg").toFixed(1)}</strong> 篇相同（Skip-Gram）、${avgOverlap("cbow").toFixed(1)} 篇（CBOW），TF-IDF 只有 ${avgOverlap("tfidf").toFixed(1)} 篇。dividend raised／cut 是例外：兩個查詢共用權重最高的 dividend，三種方法都重疊 6 篇。`;

    $("#mask-mode").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      $$("#mask-mode button").forEach((x) => x.classList.toggle("on", x === b)); renderMaskChart();
    });
    renderMaskChart();
    $("#topic-finding").innerHTML = `<p><strong>兩個實驗是同一件事的兩面。</strong>詞向量方法依整篇新聞的「主題」排序，幾乎不依賴查詢詞本身：所以用字不同（petroleum／oil）也找得到，意思相反（surplus／deficit）也分不出來。TF-IDF 正好相反，只認字面。</p>`;
  }

  function renderMaskChart() {
    const mode = $$("#mask-mode button").find((b) => b.classList.contains("on")).dataset.v;
    const M = Experiments.METHODS;
    Charts.groupedBar($("#mask-chart"), {
      cats: M.map((m) => ({ label: MNAME[m.id] })),
      series: [
        { name: "刪除前", color: "#C9C4B5", values: M.map((m) => maskAvg(mode, "before", m.id)) },
        { name: "刪除後", color: "#1F3A5F", values: M.map((m) => maskAvg(mode, "after", m.id)) },
      ],
      max: 1, height: 270, ref: randomBase(), refLabel: "隨機排序的期望值", showValues: true, valueFmt: (v) => v.toFixed(2),
    });
    const kr = S.mask.key;
    const drops = kr.filter((r) => r.after.tfidf.ndcg10 < r.before.tfidf.ndcg10 - 0.3).map((r) => r.mask[0]);
    $("#mask-finding").innerHTML = mode === "key"
      ? `每個查詢只刪掉在相關新聞中最常出現的那個詞。TF-IDF 平均從 ${f3(maskAvg("key", "before", "tfidf"))} 掉到 ${f3(maskAvg("key", "after", "tfidf"))}（刪掉 ${drops.map(esc).join("、")} 時下降最多），兩種詞向量幾乎不變。`
      : `刪掉所有查詢詞後，TF-IDF 一篇都找不到；Skip-Gram 仍有 ${f3(maskAvg("all", "after", "sg"))}、CBOW ${f3(maskAvg("all", "after", "cbow"))}，遠高於隨機排序的 ${randomBase().toFixed(2)}。`;
  }

  // -------------------------------------------------------------------
  // 06 三個觀察
  // -------------------------------------------------------------------
  function renderConclusion() {
    const { all: sA, keyword: sK, paraphrase: sP } = S.sum;
    const zo = Experiments.zeroOverlapStats(S.rows).sg;
    $("#concl-body").innerHTML = `
      <ol class="concl-list">
        <li><strong>詞向量的價值在「用字不同」的時候。</strong>整體 nDCG@10：TF-IDF ${f3(sA.tfidf.ndcg10)}、CBOW ${f3(sA.cbow.ndcg10)}、Skip-Gram ${f3(sA.sg.ndcg10)}。但用常見字查詢時三者相近（${f3(sK.tfidf.ndcg10)}／${f3(sK.cbow.ndcg10)}／${f3(sK.sg.ndcg10)}），差距來自換字查詢（${f3(sP.tfidf.ndcg10)} → ${f3(sP.sg.ndcg10)}）。Skip-Gram 前 10 名有 ${pct(zo.n / zo.total, 0)} 的新聞沒有任何查詢詞，其中 ${pct(zo.rel / zo.n, 0)} 是相關的。</li>
        <li><strong>向量接近，代表「出現在同一類新聞」，不代表意思相同。</strong>profit 的鄰居是 loss、deficit 的鄰居是 surplus，petroleum 的鄰居是公司名稱。到了檢索，意思相反的查詢有 ${avgOverlap("sg").toFixed(1)} / 10 篇結果相同。</li>
        <li><strong>CBOW 與 Skip-Gram 各有傾向，訓練設定的影響大於架構。</strong>CBOW 的最近鄰較像同義詞，Skip-Gram 較偏向事件與搭配詞。預設設定下 Skip-Gram 的檢索較佳（${f3(sA.sg.ndcg10)} 對 ${f3(sA.cbow.ndcg10)}），但需要約 ${(S.models.sg.examples / S.models.cbow.examples).toFixed(1)} 倍的訓練樣本；改變視窗大小造成的差異，大於兩種架構之間的差異。</li>
      </ol>
      <p class="small muted" style="margin-top:12px"><strong>限制：</strong>相關性以主題標籤判斷，對「抓主題」的詞向量方法較有利；earn、acq 的相關新聞很多，三種方法都接近滿分。訓練語料約 72 萬詞，鄰居容易受少數大新聞影響，結論不宜推論到大型語料或現代語言模型。</p>`;
  }

  // -------------------------------------------------------------------
  function initNavHighlight() {
    const links = $$(".section-nav a");
    const secs = links.map((a) => document.querySelector(a.getAttribute("href")));
    const onScroll = () => {
      let cur = 0;
      secs.forEach((s, i) => { if (s.getBoundingClientRect().top < 120) cur = i; });
      links.forEach((a, i) => a.classList.toggle("active", i === cur));
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  init().catch((e) => {
    console.error(e);
    $("#loading").textContent = "載入失敗：" + e.message + "（請透過網頁伺服器開啟，直接開啟檔案會被瀏覽器擋下）";
  });
})();
