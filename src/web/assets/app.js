// The question page: loads its choices from /api/config, sends each question with the earlier turns to /api/ask,
// and sets every reply as a box with its evidence table. The server keeps no conversation; this page does.
"use strict";

const $ = (id) => document.getElementById(id);
const thread = $("thread"), empty = $("empty"), form = $("compose"), field = $("question"), ask = $("ask");
const modeSelect = $("mode"), modelSelect = $("model"), threshold = $("threshold");
const resets = [$("reset"), $("reset-bar")];
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const history = [];  // earlier {question, answer} pairs, sent with each question
let modes = [], busy = false, count = 0;

function option(value, text) {
  const o = document.createElement("option");
  o.value = value;
  o.textContent = text;
  return o;
}

async function loadConfig() {
  const cfg = await (await fetch("/api/config")).json();
  modes = cfg.modes;
  modes.forEach((m) => modeSelect.append(option(m.name, m.name)));
  modeSelect.value = cfg.default_mode;
  cfg.models.forEach((m) => modelSelect.append(option(m, m.replace(/^[^/]+\//, ""))));
  modelSelect.value = cfg.default_model;
  setThreshold();
  $("intro").textContent = cfg.intro;
  $("disclaimer").textContent = cfg.disclaimer;
  const rows = $("examples");
  cfg.examples.forEach((q) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "example";
    b.textContent = q;
    b.addEventListener("click", () => { field.value = q; submit(); });
    rows.insertRow().insertCell().append(b);
  });
}

function setThreshold() {
  const m = modes.find((x) => x.name === modeSelect.value);
  if (m) threshold.value = m.threshold.toFixed(2);
}

function seconds(s) {
  return s < 10 ? `${s.toFixed(2)} s` : `${s.toFixed(1)} s`;
}

// A file name has no break point: allow one after each underscore, as text nodes and <wbr>, never as HTML.
function breakable(el, text) {
  text.split("_").forEach((part, i, all) => {
    el.append(part + (i < all.length - 1 ? "_" : ""));
    if (i < all.length - 1) el.append(document.createElement("wbr"));
  });
}

function newTurn(question) {
  const node = $("turn-template").content.firstElementChild.cloneNode(true);
  count += 1;
  node.id = `t${count}`;
  node.querySelector(".qnum").textContent = count;
  node.querySelector(".question-text").textContent = question;
  return node;
}

function setPending(node) {
  const box = node.querySelector(".box"), start = performance.now();
  box.className = "box pending";
  node.querySelector(".verdict").textContent = "Searching the 50 documents";
  const clock = node.querySelector(".box-meta");
  clock.className = "box-meta clock";
  node.querySelector(".statement").textContent =
    "Finding passages and scoring them. If the gate lets the question through, the model writes the answer.";
  const tick = () => { clock.textContent = seconds((performance.now() - start) / 1000); };
  tick();
  return setInterval(tick, 100);
}

function headMeta(res) {
  const parts = [];
  const gate = res.rerank_score ?? res.best_similarity;
  if (gate != null) parts.push(`Gate ${gate.toFixed(2)}`);
  if (res.search_s != null) parts.push(`search ${seconds(res.search_s)}`);
  if (res.model_s != null) parts.push(`model ${seconds(res.model_s)}`);
  return parts.join(" · ");
}

function evidenceRows(node, res, turnId) {
  const body = node.querySelector(".evidence").tBodies[0];
  res.sources.forEach((s, i) => {
    const tr = body.insertRow();
    tr.id = `${turnId}-r${s.rank}`;
    tr.style.setProperty("--i", i);
    if (s.cited) tr.className = "cited";
    const num = tr.insertCell();
    num.className = "num";
    num.textContent = s.rank;
    const passage = tr.insertCell();
    passage.className = "passage";
    breakable(passage, s.label);
    const wrap = document.createElement("span"), bar = document.createElement("span"), fill = document.createElement("i");
    wrap.className = "sim";
    bar.className = "sim-bar";
    fill.style.setProperty("--v", s.similarity == null ? 0 : Math.max(0, Math.min(1, s.similarity)));
    bar.append(fill);
    wrap.append(bar, s.similarity == null ? "not scored" : s.similarity.toFixed(3));
    tr.insertCell().append(wrap);
    const cited = tr.insertCell();
    if (s.cited) {
      const mark = document.createElement("span");
      mark.className = "cited-mark";
      mark.textContent = "Cited";
      cited.append(mark);
    }
  });
}

function timingRow(node, res) {
  const search = res.search_s ?? 0, model = res.model_s;
  const text = node.querySelector(".timing-text");
  const part = (label, value, cls) => {
    const b = document.createElement("b");
    b.className = cls;
    b.textContent = value;
    text.append(`${label} `, b);
  };
  text.replaceChildren();
  part("Search and gate", seconds(search), "k-search");
  text.append(" · ", model != null ? "" : res.kind === "error" ? "Model call failed" : "Model not called");
  if (model != null) part("Model reply", seconds(model), "k-model");
  node.querySelector(".timing").classList.toggle("failed", res.kind === "error");
  const total = search + (model ?? 0) || 1;
  node.querySelector(".t-search").style.width = `${(100 * search) / total}%`;
  node.querySelector(".t-model").style.width = `${(100 * (model ?? 0)) / total}%`;
}

// Each citation marker points at its row of the evidence table; hovering or focusing it lights the row.
function linkCitations(node, turnId) {
  node.querySelectorAll(".cite a[data-row]").forEach((a) => {
    const row = () => document.getElementById(`${turnId}-r${a.dataset.row}`);
    a.href = `#${turnId}-r${a.dataset.row}`;
    a.setAttribute("aria-label", `Passage ${a.dataset.row}`);
    const on = () => row()?.classList.add("lit"), off = () => row()?.classList.remove("lit");
    a.addEventListener("mouseenter", on);
    a.addEventListener("focus", on);
    a.addEventListener("mouseleave", off);
    a.addEventListener("blur", off);
  });
}

function fill(node, res) {
  const box = node.querySelector(".box"), turnId = node.id;
  box.className = `box ${res.kind}`;
  box.style.animation = "none";
  void box.offsetWidth;  // restart the entrance for the finished box
  box.style.animation = "";
  const answered = res.kind === "answered";
  const cited = (res.sources || []).filter((s) => s.cited).length;
  node.querySelector(".verdict").textContent =
    answered ? `Answer · ${cited} of ${res.sources.length} passages cited` : res.label || "Not answered";
  const meta = node.querySelector(".box-meta");
  meta.className = "box-meta";
  meta.textContent = headMeta(res);
  const statement = node.querySelector(".statement"), detail = node.querySelector(".detail");
  if (res.kind === "error" || res.kind === "empty") {
    statement.textContent = res.detail;
    detail.hidden = true;
  } else {
    statement.innerHTML = res.answer_html;  // escaped by the server
    detail.textContent = res.detail;
    detail.hidden = answered;  // the head already says how many passages are cited
    linkCitations(node, turnId);
  }
  if (res.sources && res.sources.length) {
    evidenceRows(node, res, turnId);
    timingRow(node, res);
    node.querySelector(".evidence").hidden = false;
  }
}

function markFollowUp(node) {  // the server searched this question joined to the previous one
  const note = node.querySelector(".follow-up");
  note.textContent = `Follow-up to question ${count - 1}: alone it was refused, so it was searched joined to that question.`;
  note.hidden = false;
}

async function submit() {
  const question = field.value.trim();
  if (!question || busy) { field.focus(); return; }
  busy = true;
  ask.disabled = true;
  ask.textContent = "Asking";
  empty.hidden = true;
  const node = newTurn(question);
  thread.append(node);
  const timer = setPending(node);
  node.scrollIntoView({ behavior: reducedMotion.matches ? "auto" : "smooth", block: "start" });
  field.value = "";
  autosize();
  let res;
  try {
    const reply = await fetch("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, model: modelSelect.value, mode: modeSelect.value,
                             threshold: Number(threshold.value), history }),
    });
    res = reply.ok ? await reply.json()
      : { kind: "error", label: "Not answered", detail: `The server refused the question (HTTP ${reply.status}). Check that the gate is between 0 and 1.` };
  } catch (err) {
    res = { kind: "error", label: "Not answered", detail: "The server did not answer. Check that the app is still running, then ask again." };
  }
  clearInterval(timer);
  fill(node, res);
  if (res.follow_up) markFollowUp(node);
  if (res.kind === "answered" || res.kind === "refused" || res.kind === "uncited") {
    history.push({ question, answer: res.answer });
  }
  busy = false;
  ask.disabled = false;
  ask.textContent = "Ask";
  resets.forEach((b) => { b.disabled = false; });
  field.focus();
}

function autosize() {
  if (CSS.supports("field-sizing", "content")) return;
  field.style.height = "auto";
  field.style.height = `${field.scrollHeight + 4}px`;
}

form.addEventListener("submit", (e) => { e.preventDefault(); submit(); });
field.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(); }
});
field.addEventListener("input", autosize);
modeSelect.addEventListener("change", setThreshold);
const wide = matchMedia("(min-width: 761px)");  // the bar shows the settings; a phone folds them behind "Settings"
const placeSettings = () => { $("settings").open = wide.matches; };
wide.addEventListener("change", placeSettings);
placeSettings();
resets.forEach((b) => b.addEventListener("click", () => {
  history.length = 0;
  count = 0;
  thread.querySelectorAll(".turn").forEach((t) => t.remove());
  empty.hidden = false;
  resets.forEach((x) => { x.disabled = true; });
  field.focus();
}));

loadConfig();
