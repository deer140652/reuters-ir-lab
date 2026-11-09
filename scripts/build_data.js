#!/usr/bin/env node
/*
 * build_data.js
 * -------------
 * 由 Reuters-21578 原始 SGML（reut2-000.sgm … reut2-021.sgm）產生網站用的資料：
 *
 *   web/data/corpus.json        ModApte 測試集 3,299 篇（檢索與評估的文件庫）
 *   web/data/w2v-cbow.json/.bin  CBOW 模型（只用 ModApte 訓練集 9,603 篇訓練）
 *   web/data/w2v-sg.json/.bin    Skip-Gram 模型（參數與 CBOW 相同）
 *
 * ModApte 切分：LEWISSPLIT="TRAIN" 且 TOPICS="YES" → 訓練集；
 *               LEWISSPLIT="TEST"  且 TOPICS="YES" → 測試集。
 *
 * 直接載入網頁端的 ir-core.js 與 word2vec.js，前處理與訓練演算法與網頁完全相同。
 *
 * 用法：node scripts/build_data.js <SGML 資料夾> [--epochs 5] [--skip-model]
 */
"use strict";

const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const JS = path.join(ROOT, "web", "assets", "js");
const OUT = path.join(ROOT, "web", "data");
require(path.join(JS, "ir-core.js"));
require(path.join(JS, "word2vec.js"));

const TOPICS5 = ["earn", "acq", "crude", "grain", "trade"];

function arg(name, def) {
  const i = process.argv.indexOf(name);
  return i > 0 && process.argv[i + 1] ? process.argv[i + 1] : def;
}

function decode(s) {
  return s
    .replace(/&#(\d+);/g, (_, n) => (+n < 32 ? " " : String.fromCharCode(+n)))
    .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&").replace(/&quot;/g, '"');
}

function tag(block, name) {
  const m = block.match(new RegExp(`<${name}(?:\\s[^>]*)?>([\\s\\S]*?)</${name}>`));
  return m ? m[1] : "";
}

function parseFile(file) {
  const s = fs.readFileSync(file, "latin1");
  const out = [];
  const re = /<REUTERS ([^>]*)>([\s\S]*?)<\/REUTERS>/g;
  let m;
  while ((m = re.exec(s)) !== null) {
    const attr = m[1], body = m[2];
    const at = (k) => (attr.match(new RegExp(`${k}="([^"]*)"`)) || [])[1];
    const topics = (tag(body, "TOPICS").match(/<D>(.*?)<\/D>/g) || []).map((x) => x.replace(/<\/?D>/g, ""));
    const text = tag(body, "TEXT");
    const title = decode(tag(text, "TITLE")).replace(/\s+/g, " ").trim();
    let content = tag(text, "BODY");
    if (!content && !/<TITLE>/.test(text)) content = text;              // TYPE="UNPROC"：沒有標題與內文標記
    content = decode(content)
      .replace(/^\s*Blah blah blah\.?\s*$/i, "")                        // TYPE="BRIEF"：只有標題，內文為填充字
      .replace(/\s*Reuter\s*$/i, "")                                   // 每篇結尾的「Reuter」署名
      .replace(/\s*REUTER\s*$/i, "")
      .trim();
    out.push({
      id: at("NEWID"), split: at("LEWISSPLIT"), hasTopics: at("TOPICS"),
      date: tag(body, "DATE").trim().slice(0, 11), title, body: content, topics,
    });
  }
  return out;
}

function main() {
  const dir = process.argv[2];
  if (!dir || dir.startsWith("--")) {
    console.error("用法：node scripts/build_data.js <SGML 資料夾> [--epochs 5] [--skip-model]");
    process.exit(1);
  }
  const files = fs.readdirSync(dir).filter((f) => /^reut2-\d+\.sgm$/.test(f)).sort();
  let all = [];
  files.forEach((f) => { all = all.concat(parseFile(path.join(dir, f))); });
  const train = all.filter((d) => d.split === "TRAIN" && d.hasTopics === "YES");
  const test = all.filter((d) => d.split === "TEST" && d.hasTopics === "YES");
  console.log(`解析 ${files.length} 個檔案、${all.length} 篇；ModApte 訓練 ${train.length} 篇、測試 ${test.length} 篇`);

  const countTopics = (docs) => {
    const c = {};
    TOPICS5.forEach((t) => { c[t] = docs.filter((d) => d.topics.includes(t)).length; });
    c.other = docs.filter((d) => d.topics.length && !d.topics.some((t) => TOPICS5.includes(t))).length;
    c.none = docs.filter((d) => !d.topics.length).length;
    return c;
  };

  fs.mkdirSync(OUT, { recursive: true });
  const corpus = {
    source: "Reuters-21578, Distribution 1.0 — ModApte split (test set)",
    built_at: new Date().toISOString(),
    topics5: TOPICS5,
    stats: {
      train_docs: train.length, test_docs: test.length,
      train_topics: countTopics(train), test_topics: countTopics(test),
    },
    documents: test.map((d) => ({ id: d.id, date: d.date, title: d.title, body: d.body, topics: d.topics })),
  };
  fs.writeFileSync(path.join(OUT, "corpus.json"), JSON.stringify(corpus));
  console.log("  測試集主題：", corpus.stats.test_topics);
  console.log("  訓練集主題：", corpus.stats.train_topics);
  console.log(`  → web/data/corpus.json（${(fs.statSync(path.join(OUT, "corpus.json")).size / 1e6).toFixed(1)} MB）`);

  if (process.argv.includes("--skip-model")) return;

  // ---- Word2Vec：只用訓練集 ----
  const texts = train.map((d) => (d.title ? d.title + ".\n  " : "") + d.body);
  const cw = Word2Vec.buildCorpus(texts, { minCount: 5 });
  console.log(`Word2Vec 前處理：${cw.stats.numSentences} 句、訓練詞元 ${cw.ids.length}、詞彙 ${cw.vocab.length}（原始 ${cw.stats.rawVocab}）`);
  const epochs = parseInt(arg("--epochs", "5"), 10);
  const seed = parseInt(arg("--seed", "20261105"), 10);
  const suffix = arg("--suffix", "");

  ["cbow", "sg"].forEach((model) => {
    const params = {
      model, dim: 100, window: 5, negative: 5, epochs,
      lr: model === "sg" ? 0.025 : 0.05, sample: 0.001, seed,
    };
    let res = null;
    Word2Vec.trainCore(Object.assign({ ids: cw.ids, sentStarts: cw.sentStarts, freq: cw.freq }, params), (msg) => {
      if (msg.type === "epoch") console.log(`  [${model}] epoch ${msg.epoch} 平均 loss ${msg.loss.toFixed(4)}`);
      else if (msg.type === "done") res = msg;
    });
    const m = new Word2Vec.Model(cw, res.vectors, params, res);
    m.source = "default";
    m.trainedAt = new Date().toISOString();
    m.label = m.describe();
    const st = m.toStorable();
    st.meta.stats.trainDocs = train.length;
    fs.writeFileSync(path.join(OUT, `w2v-${model}${suffix}.json`), JSON.stringify(st.meta));
    const buf = Buffer.from(st.vectors.buffer, st.vectors.byteOffset, st.vectors.byteLength);
    fs.writeFileSync(path.join(OUT, `w2v-${model}${suffix}.bin`), buf);
    console.log(`  [${model}] 完成 ${(res.elapsed / 1000).toFixed(1)} 秒、樣本 ${res.examples}、${(buf.length / 1e6).toFixed(1)} MB`);
    ["oil", "petroleum", "profit", "wheat", "merger", "deficit"].forEach((w) => {
      const r = m.mostSimilar(w, 5);
      console.log(`    ${w} → ${r ? r.map((x) => `${x.word}(${x.score.toFixed(2)})`).join(", ") : "（不在詞彙中）"}`);
    });
  });
}

if (require.main === module) main();
module.exports = { parseFile };
