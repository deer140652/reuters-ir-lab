#!/usr/bin/env node
/*
 * run_experiments.js
 * ------------------
 * 在 Node 中跑一次網頁上的所有實驗（與網頁使用同一份程式），印出結果並寫入
 * web/data/results-summary.json，方便撰寫報告與核對網頁上的數字。
 *
 * 用法：node scripts/run_experiments.js [--suffix ""]
 */
"use strict";
const fs = require("fs");
const path = require("path");
const ROOT = path.resolve(__dirname, "..");
const JS = path.join(ROOT, "web", "assets", "js");
const DATA = path.join(ROOT, "web", "data");
["ir-core.js", "eval-core.js", "word2vec.js", "embed-core.js", "experiments.js"].forEach((f) => require(path.join(JS, f)));

function arg(name, def) {
  const i = process.argv.indexOf(name);
  return i > 0 && process.argv[i + 1] !== undefined ? process.argv[i + 1] : def;
}

function loadModel(kind, suffix) {
  const meta = JSON.parse(fs.readFileSync(path.join(DATA, `w2v-${kind}${suffix}.json`), "utf8"));
  const buf = fs.readFileSync(path.join(DATA, `w2v-${kind}${suffix}.bin`));
  const vec = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  return Word2Vec.fromStored(meta, vec);
}

function buildEnv(suffix) {
  const corpus = JSON.parse(fs.readFileSync(path.join(DATA, "corpus.json"), "utf8"));
  const evalSet = JSON.parse(fs.readFileSync(path.join(DATA, "eval-set.json"), "utf8"));
  const docs = corpus.documents;
  const index = IRCore.buildIndex(docs.map((d) => ({ id: d.id, terms: IRCore.terms(d.title + ".\n  " + d.body) })));
  const models = { cbow: loadModel("cbow", suffix), sg: loadModel("sg", suffix) };
  const dv = { cbow: EmbedCore.buildDocVectors(index, models.cbow), sg: EmbedCore.buildDocVectors(index, models.sg) };
  return { docs, index, dv, models, evalSet };
}

const f3 = (x) => x.toFixed(3);

function main() {
  const suffix = arg("--suffix", "");
  const env = buildEnv(suffix);
  const M = Experiments.METHODS;
  const rows = Experiments.mainEval(env);

  console.log("== 每個查詢的 nDCG@10 ==");
  rows.forEach((r) => console.log(r.q.id, r.q.type.padEnd(10), r.q.text.padEnd(42), M.map((m) => f3(r.m[m.id].ndcg10)).join("  "), " P@10", M.map((m) => r.m[m.id].p10.toFixed(1)).join(" ")));
  const groups = { all: null, keyword: (r) => r.q.type === "keyword", paraphrase: (r) => r.q.type === "paraphrase" };
  ["earn", "acq", "crude", "grain", "trade"].forEach((t) => { groups[t] = (r) => r.q.topic === t; });
  const summary = {};
  console.log("\n== 平均（P@10 / MRR / nDCG@10 / MAP）==");
  Object.entries(groups).forEach(([g, f]) => {
    summary[g] = Experiments.summarize(rows, f || undefined);
    console.log(g.padEnd(11), M.map((m) => `${m.short}: ${f3(summary[g][m.id].p10)} ${f3(summary[g][m.id].mrr)} ${f3(summary[g][m.id].ndcg10)} ${f3(summary[g][m.id].map)}`).join(" | "));
  });
  const zo = Experiments.zeroOverlapStats(rows);
  console.log("\n== 前 10 名中沒有共同字的文件 ==", JSON.stringify(zo));

  console.log("\n== 反義查詢：前 10 名重疊篇數 ==");
  const ant = Experiments.antonymEval(env);
  ant.forEach((a) => console.log(a.p.id, `${a.p.a} / ${a.p.b}`.padEnd(34), M.map((m) => `${m.short}=${a.m[m.id].overlap}`).join("  ")));

  const masks = {};
  ["key", "all"].forEach((mode) => {
    console.log(`\n== 遮字實驗（${mode}）：nDCG@10 遮前 → 遮後 ==`);
    const mr = Experiments.maskEval(env, mode);
    masks[mode] = mr;
    mr.forEach((r) => console.log(r.q.id, `[${r.mask.join(",")}]`.padEnd(30), M.map((m) => `${m.short} ${f3(r.before[m.id].ndcg10)}→${f3(r.after[m.id].ndcg10)}`).join("  ")));
    const avg = (k, id) => mr.reduce((s, r) => s + r[k][id].ndcg10, 0) / mr.length;
    const avgP = (k, id) => mr.reduce((s, r) => s + r[k][id].p10, 0) / mr.length;
    console.log("平均 nDCG", M.map((m) => `${m.short} ${f3(avg("before", m.id))}→${f3(avg("after", m.id))}`).join("  "));
    console.log("平均 P@10", M.map((m) => `${m.short} ${f3(avgP("before", m.id))}→${f3(avgP("after", m.id))}`).join("  "));
    const rb = mr.reduce((s, r) => s + Experiments.randomBaseline(r.q.topic, env), 0) / mr.length;
    console.log("隨機排序期望 P@10", f3(rb));
  });

  const out = {
    suffix, summary, zeroOverlap: zo,
    perQuery: rows.map((r) => ({ id: r.q.id, ndcg10: Object.fromEntries(M.map((m) => [m.id, r.m[m.id].ndcg10])) })),
    antonym: ant.map((a) => ({ id: a.p.id, overlap: Object.fromEntries(M.map((m) => [m.id, a.m[m.id].overlap])) })),
  };
  fs.writeFileSync(path.join(DATA, `results-summary${suffix}.json`), JSON.stringify(out, null, 1));
}

main();
