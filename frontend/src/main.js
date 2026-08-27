import "./styles/style.css";
import { API_BASE_URL, buildApiUrl, apiFetch } from "./api/http";
import LineDrawer from "./ui/lineDrawer";

let lineDrawer = null;
let realtimeChart = null;
let pollTimer = null;
let currentVideoSource = null;
let sourceRefreshTimer = null;
let lastCameras = [];
let lastVideos = [];

const req = (path, options = {}) => apiFetch(path, options);
const unwrap = (res) => {
  if (!res || res.status !== "ok") throw new Error(res?.message || "请求失败");
  return res.data;
};

function toast(msg, type = "") {
  const old = document.querySelector(".toast");
  if (old) old.remove();
  const t = document.createElement("div");
  t.className = `toast ${type}`.trim();
  t.textContent = msg;
  document.body.appendChild(t);
  requestAnimationFrame(() => t.classList.add("show"));
  setTimeout(() => {
    t.classList.remove("show");
    setTimeout(() => t.remove(), 280);
  }, 2600);
}

function openTab(tabId, btn) {
  document
    .querySelectorAll(".tab-body")
    .forEach((x) => x.classList.remove("active"));
  document
    .querySelectorAll(".tab")
    .forEach((x) => x.classList.remove("active"));
  document.getElementById(tabId)?.classList.add("active");
  btn?.classList.add("active");
}
window.openTab = openTab;

function setStatusDot(status) {
  const dot = document.querySelector(".status-dot");
  const text = document.getElementById("statusText");
  if (!dot || !text) return;

  if (status.is_running && status.is_paused) {
    dot.className = "status-dot paused";
    text.textContent = "已暂停";
  } else if (status.is_running) {
    dot.className = "status-dot running";
    text.textContent = "计数中";
  } else if (status.model_loaded) {
    dot.className = "status-dot ready";
    text.textContent = "就绪";
  } else {
    dot.className = "status-dot";
    text.textContent = "未就绪";
  }
}

function syncButtons(status) {
  const btnDraw = document.getElementById("btnDrawLine");
  const btnEdit = document.getElementById("btnEditLine");
  const btnClear = document.getElementById("btnClearLine");
  const btnStart = document.getElementById("btnStart");
  const btnPause = document.getElementById("btnPause");
  const btnStop = document.getElementById("btnStop");
  const btnReset = document.getElementById("btnReset");
  const mode = document.getElementById("modeSelect");

  if (btnDraw) btnDraw.disabled = !status.has_source;
  if (btnEdit) btnEdit.disabled = !status.has_line;
  if (btnClear) btnClear.disabled = !status.has_line;
  if (mode) mode.disabled = !status.has_source;

  if (btnStart)
    btnStart.disabled = !(
      status.model_loaded &&
      status.has_source &&
      status.has_line &&
      !status.is_running
    );
  if (btnPause) btnPause.disabled = !status.is_running;
  if (btnStop) btnStop.disabled = !status.is_running;
  if (btnReset) btnReset.disabled = status.is_running;
}

function setLineHint(text, ok = false) {
  const el = document.getElementById("lineStatus");
  if (!el) return;
  el.textContent = text;
  el.className = ok ? "line-hint ok" : "line-hint";
}

function showFeed() {
  const img = document.getElementById("videoFeed");
  const placeholder = document.getElementById("videoPlaceholder");
  if (!img) return;
  img.src = `${buildApiUrl("/video/mjpeg_feed")}?t=${Date.now()}`;
  if (placeholder) placeholder.style.display = "none";
}

async function refreshStatus() {
  const status = unwrap(await req("/detection/status"));
  setStatusDot(status);
  syncButtons(status);
  return status;
}

function updateClasses(modelInfo) {
  const toggle = document.getElementById("pedestrianToggle");
  if (!toggle || !modelInfo?.classes?.length) return;

  // 专用系统：同步模型加载后的类别开启状态
  const enabled = new Set(modelInfo.enabled_classes || modelInfo.classes);
  toggle.checked = enabled.has("person") || enabled.size > 0;
}

async function syncClasses() {
  const enabled = [
    ...document.querySelectorAll("#classContainer input:checked"),
  ].map((x) => x.value);
  try {
    await req("/detection/classes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
  } catch (err) {
    toast(err.message, "error");
  }
}
window.syncClasses = syncClasses;

async function fetchLocalModels() {
  const sel = document.getElementById("modelSelect");
  if (!sel) return;
  // Only show loading if no model options have been loaded yet
  if (sel.options.length <= 1) {
    sel.innerHTML = '<option value="">-- 选择服务器模型 --</option><option value="" disabled>加载模型列表中...</option>';
  }
  try {
    const data = unwrap(await req("/video/model/list"));
    if (!sel) return;
    sel.innerHTML = '<option value="">-- 选择服务器模型 --</option>';
    for (const m of data.models || []) {
      sel.innerHTML += `<option value="${m.path}">${m.name}</option>`;
    }
    if (!data.models || data.models.length === 0) {
      sel.innerHTML += '<option value="" disabled>未找到模型文件</option>';
    }
  } catch {
    if (sel) {
      sel.innerHTML = '<option value="">-- 选择服务器模型 --</option><option value="" disabled>模型列表加载失败（点击重试）</option>';
    }
  }
}

async function uploadModel(e) {
  const file = e.target.files?.[0];
  if (!file) return;

  const statusEl = document.getElementById("modelStatus");
  if (statusEl) {
    statusEl.textContent = "上传中...";
    statusEl.className = "badge";
  }

  try {
    const fd = new FormData();
    fd.append("model", file);
    const modelInfo = unwrap(
      await req("/video/model/load", { method: "POST", body: fd }),
    );

    if (statusEl) {
      statusEl.textContent = `✅ ${file.name}`;
      statusEl.className = "badge ok";
    }
    updateClasses(modelInfo);
    await refreshStatus();
    toast("模型加载成功", "success");
  } catch (err) {
    if (statusEl) statusEl.textContent = "上传失败";
    toast(err.message, "error");
  } finally {
    e.target.value = "";
  }
}

async function loadLocalModel() {
  const sel = document.getElementById("modelSelect");
  const path = sel?.value;
  if (!path) return toast("请先选择模型", "error");

  const statusEl = document.getElementById("modelStatus");
  if (statusEl) {
    statusEl.textContent = "加载中...";
    statusEl.className = "badge";
  }

  try {
    const modelInfo = unwrap(
      await req("/video/model/load_local", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path }),
      }),
    );

    if (statusEl) {
      statusEl.textContent = `✅ ${path.split(/[\\/]/).pop()}`;
      statusEl.className = "badge ok";
    }
    updateClasses(modelInfo);
    await refreshStatus();
    toast("模型加载成功", "success");
  } catch (err) {
    if (statusEl) statusEl.textContent = "加载失败";
    toast(err.message, "error");
  }
}
window.loadLocalModel = loadLocalModel;

function renderSourcesSelect({
  cameras = lastCameras,
  videos = lastVideos,
  scanning = false,
  loading = false,
} = {}) {
  const sel = document.getElementById("cameraSelect");
  const useBtn = document.getElementById("btnUseCamera");
  if (!sel || !useBtn) return;

  sel.innerHTML = '<option value="">-- 选择视频源 --</option>';

  if (loading) {
    sel.innerHTML += '<option value="" disabled>加载视频源中...</option>';
  }

  if (scanning) {
    sel.innerHTML += '<option value="" disabled>正在扫描摄像头...</option>';
  }

  if (cameras && cameras.length > 0) {
    const group = document.createElement("optgroup");
    group.label = "摄像头";
    cameras.forEach((c) => {
      group.innerHTML += `<option value="${c}">摄像头 ${c}</option>`;
    });
    sel.appendChild(group);
  }

  if (videos && videos.length > 0) {
    const group = document.createElement("optgroup");
    group.label = "服务器视频";
    videos.forEach((v) => {
      group.innerHTML += `<option value="${v.path}">${v.name} (${v.size_mb}MB)</option>`;
    });
    sel.appendChild(group);
  }

  sel.disabled = false;
  useBtn.disabled = false;
}

async function fetchSources() {
  // Only show loading if no sources have been loaded yet
  const sel = document.getElementById("cameraSelect");
  if (sel && lastCameras.length === 0 && lastVideos.length === 0) {
    renderSourcesSelect({ loading: true });
  }

  let scanning = false;

  // Request cameras and videos independently, and render whichever returns first.
  const cameraPromise = (async () => {
    try {
      const camData = unwrap(await req("/video/cameras", { timeoutMs: 4000 }));
      lastCameras = camData.cameras || [];
      scanning = !!camData.scanning;
      renderSourcesSelect({ cameras: lastCameras, videos: lastVideos, scanning });
      return true;
    } catch (err) {
      renderSourcesSelect({ cameras: lastCameras, videos: lastVideos, scanning: false });
      return false;
    }
  })();

  const videosPromise = (async () => {
    try {
      const videoData = unwrap(await req("/video/videos", { timeoutMs: 4000 }));
      lastVideos = videoData.videos || [];
      renderSourcesSelect({ cameras: lastCameras, videos: lastVideos, scanning });
      return true;
    } catch (err) {
      // Fallback for older backend versions without /video/videos.
      try {
        const srcData = unwrap(await req("/video/sources", { timeoutMs: 6000 }));
        lastVideos = srcData.videos || [];
        renderSourcesSelect({ cameras: lastCameras, videos: lastVideos, scanning });
        return true;
      } catch {
        renderSourcesSelect({ cameras: lastCameras, videos: lastVideos, scanning });
        return false;
      }
    }
  })();

  await Promise.allSettled([cameraPromise, videosPromise]);

  // Continue polling while backend is scanning cameras.
  if (scanning) {
    if (sourceRefreshTimer) clearTimeout(sourceRefreshTimer);
    sourceRefreshTimer = setTimeout(() => {
      sourceRefreshTimer = null;
      fetchSources();
    }, 1000);
  }
}
window.fetchSources = fetchSources;

async function setSource(source) {
  unwrap(
    await req("/video/source/set", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source }),
    }),
  );

  const sourceEl = document.getElementById("sourceStatus");
  if (sourceEl) {
    sourceEl.textContent = `✅ ${source}`;
    sourceEl.className = "badge ok";
  }

  currentVideoSource = source;
  showFeed();
  await refreshStatus();
  toast("视频源设置成功", "success");
}

async function useCamera() {
  const source = document.getElementById("cameraSelect")?.value;
  if (!source) return;
  try {
    await setSource(source);
  } catch (err) {
    toast(err.message, "error");
  }
}
window.useCamera = useCamera;

async function uploadVideo(e) {
  const file = e.target.files?.[0];
  if (!file) return;

  const statusEl = document.getElementById("sourceStatus");
  if (statusEl) {
    statusEl.textContent = "上传中...";
    statusEl.className = "badge";
  }

  try {
    const fd = new FormData();
    fd.append("video", file);
    const data = unwrap(
      await req("/video/source/load", { method: "POST", body: fd }),
    );

    if (statusEl) {
      statusEl.textContent = `✅ ${data.filename}`;
      statusEl.className = "badge ok";
    }

    currentVideoSource = data.source;
    showFeed();
    await refreshStatus();
    toast("视频上传成功", "success");
  } catch (err) {
    if (statusEl) statusEl.textContent = "上传失败";
    toast(err.message, "error");
  } finally {
    e.target.value = "";
  }
}

async function onLineSet(p1, p2) {
  try {
    await req("/settings/line/set", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ point1: p1, point2: p2, reset_counts: true }),
    });

    document.getElementById("btnDrawLine")?.classList.remove("active");
    setLineHint("✅ 计数线已设置", true);
    await refreshStatus();
  } catch (err) {
    toast(err.message, "error");
  }
}

async function clearLine() {
  try {
    lineDrawer.clear();
    await req("/settings/line/clear", { method: "POST" });
    setLineHint("未设置计数线", false);
    document.getElementById("btnEditLine")?.classList.remove("active");
    await refreshStatus();
  } catch (err) {
    toast(err.message, "error");
  }
}

async function startCounting() {
  if (document.getElementById("modeSelect")?.value === "fast") {
    return startFastAnalysis();
  }

  try {
    await req("/detection/count/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    await refreshStatus();
    showFeed();
    toast("计数开始", "success");
  } catch (err) {
    toast(err.message, "error");
  }
}
window.startCounting = startCounting;

async function pauseCounting() {
  try {
    const data = unwrap(
      await req("/detection/count/pause", { method: "POST" }),
    );
    const btn = document.getElementById("btnPause");
    if (btn) {
      if (data.is_paused) {
        btn.textContent = "▶ 继续";
        btn.className = "btn go";
      } else {
        btn.textContent = "⏸ 暂停";
        btn.className = "btn warn";
      }
    }
    await refreshStatus();
  } catch (err) {
    toast(err.message, "error");
  }
}
window.pauseCounting = pauseCounting;

async function stopCounting() {
  try {
    await req("/detection/count/stop", { method: "POST" });
    await refreshStatus();
    toast("计数停止", "success");
  } catch (err) {
    toast(err.message, "error");
  }
}
window.stopCounting = stopCounting;

async function resetCount() {
  await req("/detection/count/reset", { method: "POST" });
  document.getElementById("statAtoB").textContent = "0";
  document.getElementById("statBtoA").textContent = "0";
  document.getElementById("statTotal").textContent = "0";
  clearChart();
  toast("已重置", "success");
  await refreshStatus();
}
window.resetCount = resetCount;

function initRealtimeChart() {
  const canvas = document.getElementById("realtimeChart");
  if (!canvas || !window.Chart) return;

  const ctx = canvas.getContext("2d");
  realtimeChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "A→B",
          data: [],
          borderColor: "#6366f1",
          backgroundColor: "rgba(99,102,241,.1)",
          borderWidth: 2,
          tension: 0.3,
          fill: true,
          pointRadius: 0,
        },
        {
          label: "B→A",
          data: [],
          borderColor: "#ef4444",
          backgroundColor: "rgba(239,68,68,.1)",
          borderWidth: 2,
          tension: 0.3,
          fill: true,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 180 },
      plugins: {
        legend: {
          labels: { color: "#8b92a5", font: { size: 11 }, padding: 8 },
        },
      },
      scales: {
        x: {
          display: true,
          grid: { display: false },
          ticks: { maxTicksLimit: 6, color: "#5c6278", font: { size: 10 } },
        },
        y: {
          display: true,
          beginAtZero: true,
          grid: { color: "rgba(255,255,255,.04)" },
          ticks: { stepSize: 1, color: "#5c6278", font: { size: 10 } },
        },
      },
    },
  });
}

let chartN = 0;
function updateChart(stats) {
  if (!realtimeChart || !stats) return;
  chartN++;
  realtimeChart.data.labels.push(`${chartN}s`);
  realtimeChart.data.datasets[0].data.push(stats.count_a_to_b || 0);
  realtimeChart.data.datasets[1].data.push(stats.count_b_to_a || 0);
  if (realtimeChart.data.labels.length > 60) {
    realtimeChart.data.labels.shift();
    realtimeChart.data.datasets[0].data.shift();
    realtimeChart.data.datasets[1].data.shift();
  }
  realtimeChart.update("none");
}

function clearChart() {
  if (!realtimeChart) return;
  chartN = 0;
  realtimeChart.data.labels = [];
  realtimeChart.data.datasets.forEach((ds) => (ds.data = []));
  realtimeChart.update();
}

async function fetchSettings() {
  try {
    const s = unwrap(await req("/settings"));
    if (s.yolo)
      document.getElementById("settConfidence").value = s.yolo.confidence;
    if (s.tracking) {
      document.getElementById("settMaxAge").value = s.tracking.max_age;
      document.getElementById("settMinHits").value = s.tracking.min_hits;
    }
    if (s.display) {
      document.getElementById("settInvisible").value =
        s.display.invisible_threshold;
      document.getElementById("settShowBbox").checked = !!s.display.show_bbox;
      document.getElementById("settShowLabel").checked = !!s.display.show_label;
      document.getElementById("settShowCenter").checked =
        !!s.display.show_center;
      document.getElementById("settShowTrajectory").checked =
        !!s.display.show_trajectory;
      document.getElementById("settStatsFontScale").value =
        s.display.stats_font_scale;
      document.getElementById("settLineThickness").value =
        s.display.line_thickness;
      document.getElementById("settBboxThickness").value =
        s.display.bbox_thickness;
      document.getElementById("settLabelFontScale").value =
        s.display.label_font_scale;
      document.getElementById("settCenterSize").value = s.display.center_size;
    }
  } catch {
    // ignore
  }
}

function settingsPayload() {
  return {
    yolo: {
      confidence: parseFloat(document.getElementById("settConfidence").value),
    },
    tracking: {
      max_age: parseInt(document.getElementById("settMaxAge").value, 10),
      min_hits: parseInt(document.getElementById("settMinHits").value, 10),
    },
    display: {
      invisible_threshold: parseInt(
        document.getElementById("settInvisible").value,
        10,
      ),
      show_bbox: document.getElementById("settShowBbox").checked,
      show_label: document.getElementById("settShowLabel").checked,
      show_center: document.getElementById("settShowCenter").checked,
      show_trajectory: document.getElementById("settShowTrajectory").checked,
      stats_font_scale: parseFloat(
        document.getElementById("settStatsFontScale").value,
      ),
      line_thickness: parseInt(
        document.getElementById("settLineThickness").value,
        10,
      ),
      bbox_thickness: parseInt(
        document.getElementById("settBboxThickness").value,
        10,
      ),
      label_font_scale: parseFloat(
        document.getElementById("settLabelFontScale").value,
      ),
      center_size: parseInt(
        document.getElementById("settCenterSize").value,
        10,
      ),
    },
  };
}

async function applySettings() {
  try {
    await req("/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settingsPayload()),
    });
    toast("设置已应用", "success");
  } catch (err) {
    toast(err.message, "error");
  }
}
window.applySettings = applySettings;

async function loadPreset(name) {
  try {
    await req("/settings/preset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    await fetchSettings();
    toast("预设已加载", "success");
  } catch (err) {
    toast(err.message, "error");
  }
}
window.loadPreset = loadPreset;

async function saveConfig() {
  try {
    await applySettings();
    await req("/settings/save", { method: "POST" });
    toast("配置已保存", "success");
  } catch (err) {
    toast(err.message, "error");
  }
}
window.saveConfig = saveConfig;

async function saveLog() {
  try {
    const data = unwrap(await req("/logs/save", { method: "POST" }));
    toast(data?.csv ? `日志已保存 ${data.csv}` : "日志已保存", "success");
  } catch (err) {
    toast(err.message, "error");
  }
}
window.saveLog = saveLog;

async function listLogs() {
  try {
    const data = unwrap(await req("/logs/list"));
    const box = document.getElementById("logList");
    if (!box) return;

    if (!(data.files || []).length) {
      box.innerHTML = '<span class="muted">暂无日志</span>';
      box.style.display = "block";
      return;
    }

    box.innerHTML = `<div class="log-list-inner">${data.files
      .map(
        (f) =>
          `<div class="log-entry"><span>${f}</span><a href="${buildApiUrl(`/logs/download/${encodeURIComponent(f)}`)}" target="_blank" rel="noreferrer">下载</a></div>`,
      )
      .join("")}</div>`;
    box.style.display = "block";
  } catch (err) {
    toast(err.message, "error");
  }
}
window.listLogs = listLogs;

async function startFastAnalysis() {
  if (!currentVideoSource) return toast("请先设置视频源", "error");

  try {
    await req("/logs/analysis/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_path: currentVideoSource }),
    });

    document.getElementById("analysisModal")?.classList.remove("hidden");
    pollAnalysisProgress();
  } catch (err) {
    toast(err.message, "error");
  }
}

function pollAnalysisProgress() {
  const iv = setInterval(async () => {
    try {
      const d = unwrap(await req("/logs/analysis/progress"));
      document.getElementById("analysisProgress").style.width =
        `${d.percentage || 0}%`;
      document.getElementById("analysisStatus").textContent =
        d.status || "分析中...";
      if (!d.running) {
        clearInterval(iv);
        setTimeout(
          () =>
            document.getElementById("analysisModal")?.classList.add("hidden"),
          1000,
        );
        toast(d.status || "分析完成", "success");
      }
    } catch {
      clearInterval(iv);
    }
  }, 500);
}

function stopAnalysis() {
  document.getElementById("analysisModal")?.classList.add("hidden");
}
window.stopAnalysis = stopAnalysis;

async function refreshDetailed() {
  if (!document.getElementById("tab-stats")?.classList.contains("active"))
    return;

  try {
    const d = unwrap(await req("/detection/stats/detailed"));
    const ts = d.total_stats || {};
    document.getElementById("detailedCurrent").textContent =
      d.current_objects || 0;
    document.getElementById("detailedAtoB").textContent = ts.a_to_b || 0;
    document.getElementById("detailedBtoA").textContent = ts.b_to_a || 0;
    document.getElementById("detailedTotal").textContent = ts.total || 0;
    document.getElementById("detailedAvg").textContent = (
      d.avg_per_minute || 0
    ).toFixed(1);

    const box = document.getElementById("classStatsContainer");
    let html = "";
    for (const [name, row] of Object.entries(d.class_stats || {})) {
      html += `<div class="cls-card"><div class="cls-card-h">${name}</div><div class="cls-card-b">
        <div class="cls-item"><div class="v">${row.a_to_b || 0}</div><div class="l">A→B</div></div>
        <div class="cls-item"><div class="v">${row.b_to_a || 0}</div><div class="l">B→A</div></div>
        <div class="cls-item"><div class="v">${row.total || 0}</div><div class="l">总计</div></div>
      </div></div>`;
    }
    box.innerHTML = html;
  } catch {
    // ignore
  }
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const stats = unwrap(await req("/detection/stats"));
      const status = await refreshStatus();

      document.getElementById("statAtoB").textContent = stats.count_a_to_b || 0;
      document.getElementById("statBtoA").textContent = stats.count_b_to_a || 0;
      document.getElementById("statTotal").textContent = stats.total || 0;

      if (status.is_running && !status.is_paused) updateChart(stats);
      await refreshDetailed();
    } catch {
      // ignore
    }
  }, 1000);
}

async function bindEvents() {
  document.getElementById("modelFile")?.addEventListener("change", uploadModel);
  document.getElementById("videoFile")?.addEventListener("change", uploadVideo);
  document.getElementById("modelSelect")?.addEventListener("focus", fetchLocalModels);
  document.getElementById("cameraSelect")?.addEventListener("focus", fetchSources);

  document.getElementById("btnDrawLine")?.addEventListener("click", () => {
    if (!lineDrawer.startDraw()) {
      toast("视频画面尚未加载，请稍后再试", "error");
      return;
    }
    document.getElementById("btnDrawLine")?.classList.add("active");
    setLineHint("请在视频上拖拽绘制", false);
  });

  document.getElementById("btnEditLine")?.addEventListener("click", () => {
    const btn = document.getElementById("btnEditLine");
    if (!btn) return;
    if (btn.classList.contains("active")) {
      lineDrawer.stopEdit();
      btn.classList.remove("active");
    } else if (lineDrawer.startEdit()) {
      btn.classList.add("active");
    }
  });

  document.getElementById("btnClearLine")?.addEventListener("click", clearLine);
}

async function bootstrap() {
  lineDrawer = new LineDrawer("lineCanvas", "videoFeed");
  lineDrawer.onLineSet = onLineSet;

  await bindEvents();
  initRealtimeChart();
  await Promise.allSettled([fetchLocalModels(), fetchSources(), fetchSettings()]);

  try {
    const status = await refreshStatus();
    if (status.has_source) showFeed();

    if (status.has_line) {
      try {
        const line = unwrap(await req("/settings/line"))?.line;
        if (line?.point1 && line?.point2)
          lineDrawer.setLineFromImage(line.point1, line.point2);
      } catch {
        // ignore
      }
      setLineHint("✅ 计数线已设置", true);
    }
  } catch {
    // ignore
  }

  startPolling();
  toast(`已连接 ${API_BASE_URL}`, "success");
}

document.addEventListener("DOMContentLoaded", () => {
  bootstrap().catch((err) => {
    console.error(err);
    toast(err.message || "前端初始化失败", "error");
  });
});
