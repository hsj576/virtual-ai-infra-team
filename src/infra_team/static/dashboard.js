const $ = id => document.getElementById(id);
let currentData = null;
let selectedReplay = null;
let loading = false;
let sessionToken = "";
let chatMode = "deployed";
let chatBusy = false;
let pendingAction = null;
const histories = {deployed: [], planner: []};

const esc = value => String(value ?? "").replace(/[&<>'"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"}[c]));
const num = (value, digits = 2) => value === null || value === undefined || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(digits);
const compact = value => { const n = Number(value); if (!Number.isFinite(n)) return "—"; return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(Math.round(n)); };
const duration = seconds => { const s = Number(seconds); if (!Number.isFinite(s)) return "—"; if (s < 60) return `${Math.round(s)}s`; return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`; };
const dateText = value => { if (!value) return "—"; const d = new Date(value); return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString("zh-CN", {hour12: false}); };
const label = id => ({baseline: "Baseline", dflash2_default: "DFlash2 native", dflash2_block4: "DFlash2 block 4", dflash2_block6: "DFlash2 block 6"}[id] || id || "—");
const checkName = id => ({arithmetic: "确定性算术", json_schema: "JSON 结构", instruction_following: "指令遵循", code_generation: "代码语义"}[id] || id);
const evidenceLabel = run => run?.benchmark_mode === "formal" ? `正式复测 · ${run.repeats} 次` : run?.benchmark_mode === "smoke" ? `Smoke · ${run.repeats || 1} 次` : "证据未标注";
const stateText = state => ({TRIGGERED: "收到升级任务", DISCOVERING: "发现可信能力", PREFLIGHTING: "检查兼容性与资源", PREPARING: "安全准备候选", READY: "候选已准备", PLANNING: "生成实验计划", PLAN_FROZEN: "计划已冻结", MAINTENANCE: "进入维护窗口", EXPERIMENTING: "执行本机实验", SELECTING: "独立比较结果", PROMOTING: "升级稳定服务", ONLINE_VERIFYING: "在线质量复验", REMEMBERING: "写入本地记忆", COMPLETED: "升级完成", NO_IMPROVEMENT: "保留当前版本", BASELINE_RESTORED: "已恢复旧版本", ROLLBACK_FAILED: "需要人工恢复", PREPARE_FAILED: "候选准备失败", CANDIDATE_FAILED: "候选实验失败", BASELINE_FAILED: "基线不可用", DISCOVERY_REJECTED: "候选不符合策略", PLAN_REJECTED: "计划被拒绝", PROMOTION_FAILED: "升级失败"}[state] || state || "未知状态");
const stateTone = state => ["COMPLETED", "ONLINE_VERIFYING", "REMEMBERING"].includes(state) ? "green" : ["ROLLBACK_FAILED", "BASELINE_FAILED", "PROMOTION_FAILED"].includes(state) ? "red" : ["NO_IMPROVEMENT", "BASELINE_RESTORED", "PREPARE_FAILED", "CANDIDATE_FAILED", "DISCOVERY_REJECTED", "PLAN_REJECTED"].includes(state) ? "amber" : "blue";

function toast(text) {
  const el = $("toast");
  el.textContent = text;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 1800);
}

function renderProduct(data) {
  const service = data.service || {};
  const latest = data.evolution?.latest || {};
  const watch = data.watch || {};
  $("productServiceState").textContent = service.healthy ? "正在稳定服务" : "等待启动";
  $("productServiceDetail").textContent = data.stable_api?.endpoint || "稳定 API 尚未部署";
  if (latest.status === "COMPLETED") {
    $("productUpgradeOutcome").textContent = `+${num(latest.speedup_percent)}%`;
    $("productUpgradeDetail").textContent = `${label(latest.selected_id)} · 质量 ${latest.run?.quality?.passed || 0}/${latest.run?.quality?.total || 4}`;
  } else if (latest.status) {
    $("productUpgradeOutcome").textContent = stateText(latest.status);
    $("productUpgradeDetail").textContent = latest.terminal ? "已保存真实运行证据" : "正在执行完整本机验证";
  } else {
    $("productUpgradeOutcome").textContent = "尚无记录";
    $("productUpgradeDetail").textContent = "点击检查并升级开始第一次演进";
  }
  $("productWatchState").textContent = stateText(watch.status === "NO_HISTORY" ? "等待首次检查" : watch.status);
  $("productWatchDetail").textContent = watch.next_attempt_at ? `下次允许：${dateText(watch.next_attempt_at)}` : watch.manifest_id ? `候选：${watch.manifest_id}` : "等待可信候选";
  $("humanDecisions").textContent = String(data.product?.human_decisions_after_start ?? 0);
}

function renderService(data) {
  const service = data.service || {};
  const healthy = !!service.healthy;
  const spec = service.spec || {};
  const run = data.latest_run || {};
  const active = data.recipes?.active || {};
  $("serviceTitle").textContent = healthy ? "本地 AI 正在提供服务" : "本地服务不可用";
  $("serviceDetail").textContent = healthy ? `${spec.model || "未知模型"} · ${spec.host || "127.0.0.1"}:${spec.port || "—"}` : (service.health_detail || service.status || "未部署");
  $("statusIcon").className = `status-icon ${healthy ? "good" : "bad"}`;
  $("livePill").className = `live-pill ${healthy ? "good" : "bad"}`;
  $("livePill").innerHTML = `<span class="pulse"></span><span>${healthy ? "LIVE" : "OFFLINE"}</span>`;
  const endpoint = data.stable_api?.endpoint || (service.base_url ? `${service.base_url}/v1` : "—");
  $("apiEndpoint").textContent = endpoint;
  $("copyApi").disabled = endpoint === "—";
  const draft = spec.draft_model;
  $("recipeName").textContent = active.candidate_id ? label(active.candidate_id) : draft ? "DFlash2 speculative" : "Autoregressive baseline";
  $("recipeModel").textContent = draft ? `${active.manifest_id || "verified local candidate"}${spec.draft_block_size ? ` · block ${spec.draft_block_size}` : " · native block"}` : (spec.model || "—");
  $("recipeSpeedup").textContent = run.speedup_percent !== null && run.speedup_percent !== undefined ? `+${num(run.speedup_percent)}%` : "—";
  $("recipeEvidenceLabel").textContent = evidenceLabel(run);
  const quality = run.quality || {};
  $("recipeQuality").textContent = active.quality_score || (quality.total ? `${quality.passed || 0}/${quality.total}` : "—");
  $("stableApiProof").textContent = data.stable_api?.unchanged ? "✓ 升级前后 API 未改变" : "稳定 API 等待演进证据";
  $("recoveryProof").textContent = data.recipes?.recoverable ? `✓ Previous Recipe：${label(data.recipes.previous?.candidate_id)}` : "Previous Recipe：尚无";
}

function renderMetrics(data) {
  const metrics = data.metrics || {};
  const latest = metrics.latest || {};
  $("liveTps").textContent = num(latest.decode_tok_s);
  $("liveTtft").textContent = num(latest.ttft_s, 2);
  $("liveMemory").textContent = num(latest.peak_memory_gb, 2);
}

function renderBars(run) {
  const el = $("candidateBars");
  if (!run || !run.candidates?.length) {
    el.innerHTML = '<div class="empty">暂无有效实验结果</div>';
    return;
  }
  const max = Math.max(...run.candidates.map(item => Number(item.generation_tps) || 0), 1);
  el.innerHTML = run.candidates.map(row => {
    const selected = row.id === run.selected_id;
    const baseline = row.id === "baseline";
    const width = Math.max(2, (Number(row.generation_tps) || 0) / max * 100);
    const desc = baseline ? "参考配置" : row.draft_block_size ? `draft block ${row.draft_block_size}` : "native block";
    return `<div class="bar-row"><div class="bar-name"><div class="bar-title">${esc(label(row.id))}</div><div class="bar-desc">${esc(desc)} · ${num(row.peak_memory_gb, 2)} GB</div></div><div class="track"><div class="bar ${selected ? "selected" : baseline ? "base" : ""}" style="width:${width}%"></div></div><div class="bar-value">${num(row.generation_tps)}<span class="bar-speedup">${baseline ? "reference" : `${Number(row.speedup_percent) >= 0 ? "+" : ""}${num(row.speedup_percent)}%`}</span></div></div>`;
  }).join("");
  $("runBadge").textContent = `Run ${run.id} · ${evidenceLabel(run)}`;
}

function renderQuality(run) {
  const quality = run?.quality || {};
  const checks = quality.checks || [];
  $("qualityBadge").textContent = quality.total ? `${quality.passed || 0}/${quality.total} ${quality.quality_pass ? "PASS" : "FAIL"}` : "暂无";
  $("qualityBadge").className = `badge ${quality.quality_pass ? "green" : "red"}`;
  $("gateScore").textContent = quality.total ? `${quality.passed || 0}/${quality.total}` : "—";
  $("postRestartText").textContent = quality.post_restart_verified ? "已在晋级服务上复验" : "来自候选实验";
  const el = $("qualityGates");
  if (!checks.length) {
    el.innerHTML = '<div class="empty">暂无质量结果</div>';
    return;
  }
  el.innerHTML = checks.map(item => `<div class="gate"><div class="gate-icon ${item.passed ? "" : "bad"}">${item.passed ? "✓" : "×"}</div><div><div class="gate-title">${esc(checkName(item.id))}</div><div class="gate-detail">${esc(item.detail || "—")}</div></div></div>`).join("");
}

function renderEvolution(evolution) {
  const evo = evolution || {};
  $("evolutionTitle").textContent = evo.id ? `演进 ${evo.id}` : "最近一次演进";
  $("evolutionSubtitle").textContent = evo.terminal ? "从不可变 Artifact 读取，不会修改当前服务" : "正在读取实时事件";
  $("evolutionModeBadge").textContent = evo.label || (evo.mode === "live" ? "实时演进" : "历史真实运行回放");
  $("evolutionModeBadge").className = `badge ${evo.mode === "live" ? "green" : "replay"}`;
  $("replayNotice").textContent = evo.mode === "live" ? "这是当前真实演进状态，页面按本地 Artifact 刷新。" : "这是历史真实运行回放，不会修改当前服务，也不会被伪装成 60 秒实时实验。";
  const events = evo.events || [];
  $("timeline").innerHTML = events.length ? events.map((event, index) => `<div class="step ${index === events.length - 1 && !evo.terminal ? "active" : ""}"><div class="step-dot"></div><div><div class="step-title">${esc(stateText(event.state))}</div><div class="step-detail">${esc(event.message || "已记录")}</div></div><div class="step-time">${esc(event.time ? event.time.slice(11, 19) : "—")}</div></div>`).join("") : '<div class="empty">暂无演进事件</div>';
  const tone = stateTone(evo.status);
  $("transitionBadge").textContent = stateText(evo.status);
  $("transitionBadge").className = `badge ${tone}`;
  $("evidenceApi").textContent = evo.stable_api?.unchanged ? "未改变" : "待验证";
  const quality = evo.run?.quality || {};
  $("evidenceQuality").textContent = quality.total ? `${quality.passed || 0}/${quality.total}` : "—";
  $("evidenceRecovery").textContent = evo.recipes?.recoverable ? "已保留" : "尚无";
  $("evidenceMemory").textContent = `${evo.memory?.prior_hits || 0} 条`;
  const tech = evo.technical || {};
  const rows = [["Manifest", tech.manifest_id], ["固定 revision", tech.source_revision], ["Manifest hash", tech.manifest_hash], ["Supervisor run", tech.supervisor_run_id], ["Artifact", evo.artifact_ref], ["证据文件", tech.artifacts?.join(", ")]];
  $("technicalDetails").innerHTML = rows.map(([key, value]) => `<div class="system-row"><div class="system-key">${esc(key)}</div><div class="system-value mono wrap">${esc(value || "—")}</div></div>`).join("");
}

function renderEvolutionList(evolution) {
  const list = evolution?.recent || [];
  const el = $("evolutionList");
  if (!list.length) {
    el.innerHTML = '<div class="empty">暂无演进记录</div>';
    return;
  }
  el.innerHTML = list.map(item => `<button class="evolution-row ${selectedReplay?.id === item.id ? "selected" : ""}" type="button" data-evolution-id="${esc(item.id)}"><span><strong>${esc(stateText(item.status))}</strong><small>${esc(item.id)}</small></span><span><strong>${item.speedup_percent === null || item.speedup_percent === undefined ? "—" : `+${num(item.speedup_percent)}%`}</strong><small>历史真实运行回放</small></span></button>`).join("");
  el.querySelectorAll("[data-evolution-id]").forEach(button => button.addEventListener("click", () => loadReplay(button.dataset.evolutionId)));
}

function renderWatch(data) {
  const watch = data.watch || {};
  const memory = data.memory || {};
  $("watchBadge").textContent = stateText(watch.status === "NO_HISTORY" ? "等待首次检查" : watch.status);
  $("watchBadge").className = `badge ${stateTone(watch.status)}`;
  const rows = [["当前状态", stateText(watch.status === "NO_HISTORY" ? "等待首次检查" : watch.status)], ["可信候选", watch.manifest_id], ["下次允许", watch.next_attempt_at ? dateText(watch.next_attempt_at) : "无需重试"], ["失败次数", watch.attempt_count ?? 0], ["最近演进", watch.last_evolution_id]];
  $("watchSummary").innerHTML = rows.map(([key, value]) => `<div class="system-row"><div class="system-key">${esc(key)}</div><div class="system-value">${esc(value ?? "—")}</div></div>`).join("");
  $("memoryExperiments").textContent = compact(memory.candidate_runs || 0);
  $("memoryPromotions").textContent = compact(memory.promotions || 0);
  $("memoryEvolutions").textContent = compact(memory.evolution_runs || 0);
}

function renderSystem(run) {
  const env = run?.environment || {};
  const packages = env.packages || {};
  const rows = [["芯片", env.chip], ["GPU 核心", env.gpu_cores ? `${env.gpu_cores} cores` : null], ["统一内存", env.unified_memory_gb ? `${env.unified_memory_gb} GB` : null], ["macOS", env.macos_version], ["MLX / MLX-VLM", packages.mlx && packages["mlx-vlm"] ? `${packages.mlx} / ${packages["mlx-vlm"]}` : null], ["可用磁盘", env.disk_free_gb ? `${num(env.disk_free_gb, 1)} GB` : null]];
  $("systemList").innerHTML = rows.map(([key, value]) => `<div class="system-row"><div class="system-key">${esc(key)}</div><div class="system-value">${esc(value || "—")}</div></div>`).join("");
  $("plannerSource").textContent = run?.planner_source === "target_service" ? "当前本地 Target 模型" : (run?.planner_source || "—");
  $("plannerHypothesis").textContent = run?.hypothesis || "暂无实验假设";
}

function renderControl(data) {
  const service = data.service || {};
  const task = data.control?.evolution || data.control?.optimization || {};
  const running = !!service.healthy;
  const busy = !!task.running;
  ["startButton", "panelStart"].forEach(id => $(id).disabled = running || busy);
  ["stopButton", "panelStop"].forEach(id => $(id).disabled = !running || busy);
  ["optimizeButton", "panelOptimize", "heroEvolve"].forEach(id => $(id).disabled = !running || busy);
  $("taskStatus").textContent = task.status === "running" ? "运行中" : task.status === "completed" ? "已完成" : task.status === "failed" ? "失败" : "空闲";
  $("taskLog").textContent = task.log_tail || "暂无后台任务。完整实验需要数分钟，不会伪装成 60 秒实时完成。";
  $("sendButton").disabled = chatBusy || !running;
  $("chatInput").disabled = chatBusy || !running;
  $("chatInput").placeholder = !running ? "请先启动模型服务" : chatMode === "planner" ? "询问升级状态，或输入 /help" : "输入消息；Shift+Enter 换行";
}

function render(data) {
  currentData = data;
  const evolution = selectedReplay || data.evolution?.latest || {};
  const run = evolution.run || data.latest_run;
  renderProduct(data);
  renderService(data);
  renderMetrics(data);
  renderEvolution(evolution);
  renderBars(run);
  renderQuality(run);
  renderWatch(data);
  renderSystem(run);
  renderEvolutionList(data.evolution);
  renderControl(data);
  const generated = new Date(data.generated_at);
  $("generatedAt").textContent = `最后同步 ${generated.toLocaleTimeString("zh-CN", {hour12: false})} · 数据来自本机服务与不可变 Artifact`;
  $("errorBanner").classList.remove("show");
}

async function loadReplay(evolutionId) {
  try {
    const response = await fetch(`/api/evolutions/${encodeURIComponent(evolutionId)}`, {cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    selectedReplay = await response.json();
    render(currentData);
    toast("已切换到历史真实运行回放");
  } catch (error) {
    toast(`无法读取回放：${error.message}`);
  }
}

function appendMessage(role, content, pending = false) {
  const el = document.createElement("div");
  el.className = `message ${role === "user" ? "user" : ""} ${pending ? "pending" : ""}`;
  el.innerHTML = `<div class="avatar">${role === "user" ? "YOU" : chatMode === "planner" ? "P" : "AI"}</div><div class="bubble">${esc(content)}</div>`;
  $("messages").appendChild(el);
  $("messages").scrollTop = $("messages").scrollHeight;
  return el;
}

function resetChatView() {
  const history = histories[chatMode];
  $("messages").innerHTML = "";
  if (!history.length) appendMessage("assistant", chatMode === "planner" ? "我是本地升级顾问。我可以解释证据和提出固定操作建议；输入 /help 查看命令。" : "这里实时调用当前已晋级服务，与历史演进回放相互独立。");
  else history.forEach(message => appendMessage(message.role, message.content));
  const planner = chatMode === "planner";
  $("chatSubtitle").textContent = planner ? "解释本机证据并提出受控升级建议" : "实时调用同一个稳定 API";
  $("quickPrompts").innerHTML = (planner ? ["/status", "为什么采用当前配置？", "下一次检查是什么时候？", "/evolve"] : ["介绍一下当前服务", "解释投机解码", "写一个 Python 快速排序"]).map(text => `<button class="quick" type="button">${esc(text)}</button>`).join("");
  bindQuickPrompts();
}

function bindQuickPrompts() {
  $("quickPrompts").querySelectorAll(".quick").forEach(button => button.addEventListener("click", () => { $("chatInput").value = button.textContent; $("chatInput").focus(); }));
}

async function ensureSession() {
  if (sessionToken) return;
  const response = await fetch("/api/session", {cache: "no-store"});
  if (!response.ok) throw new Error("无法建立本地控制会话");
  sessionToken = (await response.json()).token;
}

async function controlPost(path, body = {}) {
  await ensureSession();
  const response = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json", "X-Infra-Control-Token": sessionToken}, body: JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

async function streamControlPost(path, body, onEvent) {
  await ensureSession();
  const response = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json", "X-Infra-Control-Token": sessionToken}, body: JSON.stringify(body)});
  if (!response.ok) { let message = `HTTP ${response.status}`; try { message = (await response.json()).error || message; } catch {} throw new Error(message); }
  if (!response.body) throw new Error("浏览器不支持流式响应");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const {value, done} = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.type === "error") throw new Error(event.error || "流式响应失败");
      onEvent(event);
    }
    if (done) break;
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

function openConfirm(action) {
  pendingAction = action;
  const config = {
    start: {title: "启动本地 AI", text: "将使用上次保存的模型与 Recipe 启动稳定 API。", danger: false},
    stop: {title: "停止本地 AI", text: "模型 API 将停止；Dashboard 仍保持运行。", danger: true},
    evolve: {title: "检查并升级本地 AI", text: "系统将发现可信候选，在本机真实测试，只有质量、速度和资源都通过才升级；失败会恢复旧配置。", danger: false},
  }[action];
  $("modalTitle").textContent = config.title;
  $("modalText").textContent = config.text;
  $("modalOptions").hidden = action !== "evolve";
  $("modalConfirm").className = `modal-action ${config.danger ? "danger" : "confirm"}`;
  $("confirmModal").classList.add("show");
}

function closeConfirm() {
  pendingAction = null;
  $("confirmModal").classList.remove("show");
}

async function executeAction(action) {
  const endpoint = {start: "/api/control/start", stop: "/api/control/stop", evolve: "/api/control/evolve"}[action];
  const body = action === "evolve" ? {repeats: Number($("repeatsSelect").value)} : {};
  try {
    const result = await controlPost(endpoint, body);
    selectedReplay = null;
    toast(result.message || "操作已提交");
    if (chatMode === "planner") {
      const message = {role: "assistant", content: result.message || "操作已提交"};
      histories.planner.push(message);
      appendMessage(message.role, message.content);
    }
    await refresh();
  } catch (error) {
    toast(error.message);
    $("errorBanner").textContent = `操作失败：${error.message}`;
    $("errorBanner").classList.add("show");
  }
}

async function sendChat() {
  const input = $("chatInput");
  const content = input.value.trim();
  if (!content || chatBusy) return;
  const normalized = content.toLowerCase();
  if (["/start", "/stop", "/evolve", "/optimize"].includes(normalized)) {
    input.value = "";
    openConfirm(normalized === "/optimize" ? "evolve" : normalized.slice(1));
    return;
  }
  const modeAtSend = chatMode;
  const user = {role: "user", content};
  histories[modeAtSend].push(user);
  appendMessage("user", content);
  input.value = "";
  chatBusy = true;
  renderControl(currentData || {});
  const pending = appendMessage("assistant", "等待首个 token…", true);
  const bubble = pending.querySelector(".bubble");
  pending.classList.add("streaming");
  let answer = "";
  let requestedAction = null;
  let started = false;
  try {
    await streamControlPost("/api/chat/stream", {mode: modeAtSend, messages: histories[modeAtSend].slice(-24)}, event => {
      if (event.type === "delta") {
        if (!started) { answer = ""; bubble.textContent = ""; pending.classList.remove("pending"); started = true; }
        answer += event.content || "";
        bubble.textContent = answer;
        $("messages").scrollTop = $("messages").scrollHeight;
      } else if (event.type === "action") requestedAction = event.action;
    });
    pending.classList.remove("streaming", "pending");
    if (!answer) { answer = "模型没有返回文本"; bubble.textContent = answer; }
    histories[modeAtSend].push({role: "assistant", content: answer});
    if (requestedAction) openConfirm(requestedAction);
  } catch (error) {
    pending.classList.remove("streaming", "pending");
    bubble.textContent = `请求失败：${error.message}`;
  } finally {
    chatBusy = false;
    renderControl(currentData || {});
  }
}

async function refresh() {
  if (loading) return;
  loading = true;
  $("refreshButton").disabled = true;
  try {
    const response = await fetch("/api/dashboard", {cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
    $("refreshText").textContent = "每 5 秒刷新";
  } catch (error) {
    $("errorBanner").textContent = `无法读取 Dashboard 数据：${error.message}`;
    $("errorBanner").classList.add("show");
    $("refreshText").textContent = "刷新失败";
  } finally {
    loading = false;
    $("refreshButton").disabled = false;
  }
}

$("refreshButton").addEventListener("click", () => { selectedReplay = null; refresh(); });
$("copyApi").addEventListener("click", async () => { const value = $("apiEndpoint").textContent; if (value === "—") return; try { await navigator.clipboard.writeText(value); toast("API 地址已复制"); } catch { toast("复制失败"); } });
[["startButton", "start"], ["panelStart", "start"], ["stopButton", "stop"], ["panelStop", "stop"], ["optimizeButton", "evolve"], ["panelOptimize", "evolve"], ["heroEvolve", "evolve"]].forEach(([id, action]) => $(id).addEventListener("click", () => openConfirm(action)));
$("modalCancel").addEventListener("click", closeConfirm);
$("confirmModal").addEventListener("click", event => { if (event.target === $("confirmModal")) closeConfirm(); });
$("modalConfirm").addEventListener("click", async () => { const action = pendingAction; closeConfirm(); if (action) await executeAction(action); });
document.querySelectorAll(".mode-tab").forEach(tab => tab.addEventListener("click", () => { document.querySelectorAll(".mode-tab").forEach(item => item.classList.remove("active")); tab.classList.add("active"); chatMode = tab.dataset.mode; resetChatView(); renderControl(currentData || {}); }));
$("chatForm").addEventListener("submit", event => { event.preventDefault(); sendChat(); });
$("chatInput").addEventListener("keydown", event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendChat(); } });
bindQuickPrompts();
ensureSession().catch(error => toast(error.message));
refresh();
setInterval(refresh, 5000);
