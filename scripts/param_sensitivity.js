#!/usr/bin/env node
/*
 * param_sensitivity.js
 * --------------------
 * 架構與參數敏感度：window 2／5 × 維度 50／100，CBOW 與 Skip-Gram 各訓練一次，
 * 用同一組 25 個查詢比較檢索效果（nDCG@10 等）。
 *
 * window 5、維度 100 直接載入網站內建的預設模型；其餘 6 個模型在記憶體中訓練，
 * 只把評估結果寫入 web/data/param-sensitivity.json（不輸出模型檔）。
 *
 * 用法：node scripts/param_sensitivity.js <SGML 資料夾>
 */
"use strict";
const fs = require("fs");
const path = require("path");
const ROOT = path.resolve(__dirname, "..");
const JS = path.join(ROOT, "web", "assets", "js");
const DATA = path.join(ROOT, "web", "data");
["ir-core.js", "eval-core.js", "word2vec.js", "embed-core.js", "experiments.js"].forEach((f) => require(path.join(JS, f)));
const { parseFile } = require("./build_data.js");

const CONFIGS = [
  { window: 2, dim: 50 }, { window: 2, dim: 100 },
  { window: 5, dim: 50 }, { window: 5, dim: 100, default: true },
];

function loadModel(kind) {
  const meta = JSON.parse(fs.readFileSync(path.join(DATA, `w2v-${kind}.json`), "utf8"));
  const buf = fs.readFileSync(path.join(DATA, `w2v-${kind}.bin`));
  return Word2Vec.fromStored(meta, new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)));
}

function main() {
  const dir = process.argv[2];
  if (!dir) { console.error("用法：node scripts/param_sensitivity.js <SGML 資料夾>"); process.exit(1); }
  const files = fs.readdirSync(dir).filter((f) => /^reut2-\d+\.sgm$/.test(f)).sort();
  let all = [];
  files.forEach((f) => { all = all.concat(parseFile(path.join(dir, f))); });
  const train = all.filter((d) => d.split === "TRAIN" && d.hasTopics === "YES");
  const cw = Word2Vec.buildCorpus(train.map((d) => (d.title ? d.title + ".\n  " : "") + d.body), { minCount: 5 });

  const corpus = JSON.parse(fs.readFileSync(path.join(DATA, "corpus.json"), "utf8"));
  const evalSet = JSON.parse(fs.readFileSync(path.join(DATA, "eval-set.json"), "utf8"));
  const docs = corpus.documents;
  const index = IRCore.buildIndex(docs.map((d) => ({ id: d.id, terms: IRCore.terms(d.title + ".\n  " + d.body) })));

  const rows = [];
  CONFIGS.forEach((c) => {
    const models = {};
    ["cbow", "sg"].forEach((kind) => {
      if (c.default) { models[kind] = loadModel(kind); return; }
      const params = { model: kind, dim: c.dim, window: c.window, negative: 5, epochs: 5, lr: kind === "sg" ? 0.025 : 0.05, sample: 0.001, seed: 20261105 };
      let res = null;
      Word2Vec.trainCore(Object.assign({ ids: cw.ids, sentStarts: cw.sentStarts, freq: cw.freq }, params), (m) => { if (m.type === "done") res = m; });
      models[kind] = new Word2Vec.Model(cw, res.vectors, params, res);
    });
    const dv = { cbow: EmbedCore.buildDocVectors(index, models.cbow), sg: EmbedCore.buildDocVectors(index, models.sg) };
    const env = { docs, index, dv, evalSet };
    const ev = Experiments.mainEval(env);
    const sum = (f) => Experiments.summarize(ev, f);
    const S = { all: sum(), keyword: sum((r) => r.q.type === "keyword"), paraphrase: sum((r) => r.q.type === "paraphrase") };
    const row = { window: c.window, dim: c.dim, isDefault: !!c.default, models: {} };
    ["cbow", "sg"].forEach((k) => {
      row.models[k] = {
        ndcg10: { all: S.all[k].ndcg10, keyword: S.keyword[k].ndcg10, paraphrase: S.paraphrase[k].ndcg10 },
        p10: S.all[k].p10, mrr: S.all[k].mrr,
        seconds: models[k].elapsed / 1000, examples: models[k].examples,
        wins: ev.filter((r) => r.m[k].ndcg10 > r.m.tfidf.ndcg10 + 1e-9).length,
        losses: ev.filter((r) => r.m[k].ndcg10 < r.m.tfidf.ndcg10 - 1e-9).length,
      };
    });
    row.tfidf = { all: S.all.tfidf.ndcg10, keyword: S.keyword.tfidf.ndcg10, paraphrase: S.paraphrase.tfidf.ndcg10 };
    rows.push(row);
    console.log(`window ${c.window} dim ${c.dim}：` + ["cbow", "sg"].map((k) =>
      `${k} 全部 ${row.models[k].ndcg10.all.toFixed(3)} 一般 ${row.models[k].ndcg10.keyword.toFixed(3)} 換字 ${row.models[k].ndcg10.paraphrase.toFixed(3)}（${row.models[k].seconds.toFixed(0)} 秒）`).join("｜"));
  });
  fs.writeFileSync(path.join(DATA, "param-sensitivity.json"), JSON.stringify({
    note: "訓練集 9,603 篇；負樣本 5、5 epochs、最小詞頻 5、次抽樣 1e-3、隨機種子 20261105。",
    rows,
  }, null, 1));
  console.log("→ web/data/param-sensitivity.json");
}

main();
