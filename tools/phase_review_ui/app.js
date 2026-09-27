const stepDefinitions = [
  { key: "closing", title: "闭爪区间", hint: "开始闭合 → 基本稳定", fields: [["start_frame", "开始"], ["end_frame", "稳定"]] },
  { key: "lift_start", title: "抬升开始", hint: "机械臂真正开始抬杯", fields: [["frame", "边界"]] },
  { key: "release", title: "释放区间", hint: "开始张开 → 基本稳定", fields: [["start_frame", "开始"], ["end_frame", "稳定"]] },
];
const state = { fps: 30, episodes: {}, current: 0, activeStep: 0, saveTimer: null };
const $ = (selector) => document.querySelector(selector);
const video = $("#video");
function currentFrame() { return Math.max(0, Math.round(video.currentTime * state.fps)); }
function escapeHtml(value) { return String(value).replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("<", "&lt;").replaceAll(">", "&gt;"); }
function episodeSummary(episode) {
  return episode.reviewed ? "complete" : "pending";
}
function renderEpisodeList() {
  const filter = $("#filter").value;
  const query = $("#episodeSearch").value.trim();
  const html = Object.values(state.episodes).filter((episode) => {
    const summary = episodeSummary(episode);
    return (!query || String(episode.episode_index).includes(query)) && (filter === "all" || filter === summary || (filter === "issues" && summary === "issue"));
  }).map((episode) => {
    const summary = episodeSummary(episode);
    return `<button class="episode-item ${summary} ${episode.episode_index === state.current ? "active" : ""}" data-episode="${episode.episode_index}"><span>Episode ${String(episode.episode_index).padStart(3, "0")}</span><span class="count">${episode.reviewed ? "已确认" : "待审"}</span></button>`;
  }).join("");
  $("#episodeList").innerHTML = html || `<p class="empty">没有匹配片段</p>`;
  document.querySelectorAll("[data-episode]").forEach((button) => button.addEventListener("click", () => loadEpisode(Number(button.dataset.episode))));
}
function frameControls(definition, step, episode) {
  return definition.fields.map(([field, label]) => `<div class="boundary-row">
    <span class="boundary-label">${label}</span>
    <input class="frame-input" data-frame-input="${definition.key}" data-field="${field}" type="number" min="0" max="${episode.length - 1}" value="${step[field]}" aria-label="${definition.title}${label}帧" />
    <button class="seek-button" data-seek="${definition.key}" data-field="${field}">跳转</button>
    <button class="capture-button" data-capture="${definition.key}" data-field="${field}">取当前帧</button>
  </div>`).join("");
}
function renderSteps() {
  const episode = state.episodes[String(state.current)];
  $("#stepList").innerHTML = stepDefinitions.map((definition, index) => {
    const step = episode.steps[definition.key];
    return `<article class="step-card ${index === state.activeStep ? "active" : ""}" data-step="${index}">
      <div class="step-title"><strong>${index + 1}. ${definition.title}</strong><span>${definition.hint}</span></div>
      ${frameControls(definition, step, episode)}
      <input class="step-note" data-note="${definition.key}" value="${escapeHtml(step.note || "")}" placeholder="该步骤备注（可选）" />
    </article>`;
  }).join("");
  bindStepEvents();
}
function bindStepEvents() {
  document.querySelectorAll(".step-card").forEach((card) => card.addEventListener("click", (event) => {
    if (event.target.closest("button, input")) return;
    state.activeStep = Number(card.dataset.step);
    document.querySelectorAll(".step-card").forEach((item) => item.classList.toggle("active", item === card));
  }));
  document.querySelectorAll("[data-seek]").forEach((button) => button.addEventListener("click", () => seekTo(state.episodes[String(state.current)].steps[button.dataset.seek][button.dataset.field])));
  document.querySelectorAll("[data-capture]").forEach((button) => button.addEventListener("click", () => updateFrame(button.dataset.capture, button.dataset.field, currentFrame())));
  document.querySelectorAll("[data-frame-input]").forEach((input) => input.addEventListener("change", () => updateFrame(input.dataset.frameInput, input.dataset.field, Number(input.value))));
  document.querySelectorAll("[data-note]").forEach((input) => input.addEventListener("input", () => { state.episodes[String(state.current)].steps[input.dataset.note].note = input.value; changed(); }));
}
function updateFrame(key, field, frame) {
  const episode = state.episodes[String(state.current)];
  episode.steps[key][field] = Math.max(0, Math.min(episode.length - 1, Math.round(frame)));
  renderSteps();
  changed();
}
function seekTo(frame) { video.currentTime = frame / state.fps; video.pause(); }
function loadEpisode(index) {
  state.current = index;
  state.activeStep = 0;
  const episode = state.episodes[String(index)];
  $("#episodeLabel").textContent = `EPISODE ${String(index).padStart(3, "0")}`;
  $("#diagnostic").hidden = !episode.load_diagnostic;
  $("#episodeNote").value = episode.note || "";
  video.src = `/videos/episode_${String(index).padStart(3, "0")}.mp4`;
  renderEpisodeList(); renderSteps();
}
function updateProgress() {
  const episodes = Object.values(state.episodes);
  const complete = episodes.filter((episode) => episodeSummary(episode) === "complete").length;
  $("#progressText").textContent = `${complete} / ${episodes.length}`;
  $("#progressBar").style.width = `${complete / episodes.length * 100}%`;
}
function changed(rerenderList = true) {
  state.episodes[String(state.current)].reviewed = false;
  $("#saveState").textContent = "有未保存更改";
  if (rerenderList) renderEpisodeList();
  updateProgress(); clearTimeout(state.saveTimer); state.saveTimer = setTimeout(save, 450);
}
async function save() {
  $("#saveState").textContent = "保存中…";
  try {
    const response = await fetch("/api/save", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ schema_version: 3, episodes: state.episodes }) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "保存失败");
    $("#saveState").textContent = "已自动保存";
    return true;
  } catch (error) { $("#saveState").textContent = `保存失败：${error.message}`; return false; }
}
function moveFrame(delta) { seekTo(Math.max(0, Math.min(state.episodes[String(state.current)].length - 1, currentFrame() + delta))); }
function nextPending() {
  const indices = Object.keys(state.episodes).map(Number).sort((a, b) => a - b);
  const next = indices.find((index) => index > state.current && episodeSummary(state.episodes[String(index)]) !== "complete") ?? indices.find((index) => episodeSummary(state.episodes[String(index)]) !== "complete");
  if (next !== undefined) loadEpisode(next);
}
$("#playPause").addEventListener("click", () => video.paused ? video.play() : video.pause());
document.querySelectorAll("[data-step-frame]").forEach((button) => button.addEventListener("click", () => moveFrame(Number(button.dataset.stepFrame))));
$("#filter").addEventListener("change", renderEpisodeList);
$("#episodeSearch").addEventListener("input", renderEpisodeList);
$("#previousEpisode").addEventListener("click", () => loadEpisode(Math.max(0, state.current - 1)));
$("#nextPending").addEventListener("click", async () => {
  state.episodes[String(state.current)].reviewed = true;
  renderEpisodeList(); updateProgress();
  if (await save()) nextPending();
});
$("#episodeNote").addEventListener("input", (event) => { state.episodes[String(state.current)].note = event.target.value; changed(); });
video.addEventListener("timeupdate", () => { const episode = state.episodes[String(state.current)]; if (episode) $("#frameReadout").textContent = `帧 ${currentFrame()} / ${episode.length - 1}`; });
window.addEventListener("keydown", (event) => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  if (event.code === "Space") { event.preventDefault(); video.paused ? video.play() : video.pause(); }
  if (event.key === "ArrowLeft") moveFrame(-1);
  if (event.key === "ArrowRight") moveFrame(1);
  if (/^[1-3]$/.test(event.key)) { state.activeStep = Number(event.key) - 1; renderSteps(); }
  if (event.key.toLowerCase() === "f") { const definition = stepDefinitions[state.activeStep]; updateFrame(definition.key, definition.fields[0][0], currentFrame()); }
});
fetch("/api/bootstrap").then((response) => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); }).then((data) => {
  state.fps = data.fps; state.episodes = data.episodes; $(".workspace").hidden = false; $("#saveState").textContent = "已载入"; updateProgress(); loadEpisode(0);
}).catch((error) => { $("#errorState").hidden = false; $("#errorState p").textContent = error.message; });
