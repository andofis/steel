(() => {
  const byId = (id) => document.getElementById(id);
  const select = byId("pipeTemplateSelect");
  const record = byId("pipeRecordTemplate");
  const backend = byId("pipeInferenceBackend");
  const toggle = byId("pipeInspectionToggle");
  const form = byId("pipeTemplateForm");
  const limits = byId("pipeInspectionLimits");
  const status = byId("pipeInspectionStatus");
  const verdict = byId("pipeVerdict");
  const verdictDetail = byId("pipeVerdictDetail");
  const lineState = byId("pipeLineState");
  const image = byId("unwrappedImage");
  const imageMeta = byId("pipeImageMeta");
  const placeholder = byId("pipeImagePlaceholder");
  const scroll = byId("pipeImageScroll");
  const zoomButtons = document.querySelectorAll("[data-zoom]");
  const canvas = byId("pipeInspectionOverlay");
  const findingNames = { ok: "合格", missing: "漏孔", extra: "多孔", shifted: "孔位偏移",
    size: "面积异常", shape: "形状异常" };
  let settings = null;
  let pipeId = "";
  let capturedAt = "";
  let retryImage = false;
  let zoom = null;
  let result = null;
  let busy = false;
  let error = "";
  let request = null;
  let lastChecked = "";

  async function readJson(path, options) {
    const response = await fetch(path, options);
    const data = await response.json();
    if (!response.ok || data.ok === false) throw new Error(data.error || "检测服务暂不可用");
    return data;
  }

  function api(path = "", body, signal) {
    const options = { signal };
    if (body !== undefined) Object.assign(options, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    return readJson("/api/inspection" + path, options);
  }

  function updateControls() {
    const locked = !settings || settings.enabled || busy || settings?.recording;
    select.disabled = backend.disabled = locked;
    for (const control of limits.elements) control.disabled = locked;
    for (const control of form.elements) control.disabled = locked || image.hidden;
    for (const button of zoomButtons) button.disabled = image.hidden;
    record.disabled = locked || !pipeId || image.hidden;
    toggle.disabled = !settings?.active_template || busy || settings.recording;
    toggle.textContent = settings?.enabled ? "停止流水线检测" : "开启流水线检测";
    toggle.setAttribute("aria-pressed", String(Boolean(settings?.enabled)));
  }

  function showSettings(next) {
    settings = next;
    lineState.textContent = settings.enabled ? "流水线运行中" : "流水线已停止";
    lineState.dataset.enabled = String(settings.enabled);
    select.replaceChildren(new Option("请选择模板", ""));
    for (const template of settings.templates) {
      select.add(new Option(`${template.name} · ${template.hole_count} 孔`, template.id));
    }
    select.value = settings.active_template;
    backend.value = settings.backend;
    for (const [key, value] of Object.entries(settings.tolerances)) {
      const input = limits.elements.namedItem(key);
      if (input) input.value = key === "shape_iou" ? Math.round(value * 1000) / 10 : value;
    }
    updateControls();
  }

  function showStatus() {
    const kind = error || settings?.error ? "error" : result?.status || "";
    status.dataset.status = kind;
    const labels = { pending: "正在检测", template: "已录入样管", pass: "通过", fail: "不通过",
      review: "待复核", error: "检测失败" };
    let title = labels[kind];
    let detail = error || settings?.error || "";
    if (!detail && result) {
      const counts = new Map();
      for (const finding of result.findings || []) {
        counts.set(finding.kind, (counts.get(finding.kind) || 0) + 1);
      }
      const found = Array.from(counts, ([key, count]) => `${findingNames[key] || key} ${count}`).join(" · ");
      const timing = Number.isFinite(result.after_last_frame_ms) ?
        `末帧至判定 ${Math.round(result.after_last_frame_ms)} ms` : "";
      const pose = result.alignment;
      const transform = Number.isFinite(pose?.rotation_deg) && Number.isFinite(pose?.axial_pct) ?
        `${pose.flipped ? "掉头" : "同向"}，转角 ${pose.rotation_deg.toFixed(1)}°，轴移 ${pose.axial_pct.toFixed(2)}%` : "";
      const offImage = result.pipe_id === pipeId ? "" : result.pipe_id;
      detail = [offImage, result.template_name, found, result.reason, transform, timing].filter(Boolean).join(" · ");
    }
    if (!title && busy) title = "正在更新检测设置";
    if (!title && settings) {
      title = settings.enabled ? "等待下一根钢管" : "等待检测";
      if (!settings.templates.length) detail = "先采集一根合格样管，再录入模板。";
    }
    verdict.textContent = title || "正在读取检测设置";
    verdictDetail.textContent = detail;
    verdictDetail.hidden = !detail;
  }

  function fitScale() {
    return Math.min((scroll.clientWidth - 24) / image.naturalWidth,
      (scroll.clientHeight - 24) / image.naturalHeight, 1);
  }

  function sizeImage() {
    if (!image.hidden) {
      image.parentElement.style.width = `${Math.max(1, Math.floor(image.naturalWidth * (zoom ?? fitScale())))}px`;
    }
  }

  function drawOverlay() {
    const { width, height, findings } = result || {};
    canvas.hidden = true;
    if (result?.pipe_id !== pipeId || image.hidden || !findings?.length ||
        image.naturalWidth !== width || image.naturalHeight !== height) return;
    const scale = Math.min(1, Math.sqrt(5000000 / (width * height)));
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const ctx = canvas.getContext("2d");
    ctx.scale(canvas.width / width, canvas.height / height);
    const unit = width / 500;
    const shifts = [-width, 0, width];
    ctx.lineWidth = Math.max(1.5, 1.6 * unit);
    ctx.font = `600 ${Math.max(13, 11 * unit)}px "Microsoft YaHei", sans-serif`;
    for (const finding of findings) {
      const [cx, cy] = finding.center;
      const ok = finding.kind === "ok";
      const missing = finding.kind === "missing";
      const contours = (finding.contours || []).map((polygon) => polygon.map(([x, y]) =>
        [cx + ((((x - cx + width / 2) % width) + width) % width) - width / 2, y]));
      const points = contours.flat();
      ctx.strokeStyle = ok ? "#60f2a7" : "#ff555f";
      ctx.fillStyle = ok ? "#19ba7060" : "#ff555f40";
      for (const shift of shifts) {
        for (const polygon of contours) {
          if (polygon.length < 3 || missing) continue;
          ctx.beginPath();
          ctx.moveTo(polygon[0][0] + shift, polygon[0][1]);
          for (const [x, y] of polygon.slice(1)) ctx.lineTo(x + shift, y);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }
        if (ok) continue;
        const margin = 3 * unit;
        const left = (points.length ? Math.min(...points.map((p) => p[0])) : cx - 5 * unit) - margin + shift;
        const right = (points.length ? Math.max(...points.map((p) => p[0])) : cx + 5 * unit) + margin + shift;
        const top = (points.length ? Math.min(...points.map((p) => p[1])) : cy - 5 * unit) - margin;
        const bottom = (points.length ? Math.max(...points.map((p) => p[1])) : cy + 5 * unit) + margin;
        if (right < 0 || left >= width) continue;
        if (missing) ctx.setLineDash([4 * unit, 3 * unit]);
        ctx.strokeRect(left, top, right - left, bottom - top);
        ctx.setLineDash([]);
      }
      if (!ok) {
        const label = findingNames[finding.kind] || finding.kind;
        const textWidth = ctx.measureText(label).width;
        const x = Math.max(2 * unit, Math.min(width - textWidth - 3 * unit, cx + 12 * unit));
        const y = Math.min(height - 4 * unit, Math.max(11 * unit, cy));
        ctx.strokeStyle = "#151010";
        ctx.fillStyle = "#ff878e";
        ctx.strokeText(label, x, y);
        ctx.fillText(label, x, y);
      }
    }
    canvas.hidden = false;
  }

  async function watchResult() {
    request?.abort();
    if (!pipeId) return;
    const controller = new AbortController();
    request = controller;
    const query = new URLSearchParams({ pipe: pipeId, wait: "1" });
    try {
      do {
        const data = await api(`/result?${query}`, undefined, controller.signal);
        if (request !== controller) return;
        result = data.result;
        showStatus();
        drawOverlay();
      } while (result?.status === "pending");
    } catch (err) {
      if (err.name !== "AbortError" && request === controller) {
        error = err.message;
        showStatus();
      }
    }
  }

  async function change(path, body) {
    busy = true;
    error = "";
    updateControls();
    showStatus();
    try {
      showSettings((await api(path, body)).state);
      return true;
    } catch (err) {
      if (settings) showSettings(settings);
      error = err.message;
      return false;
    } finally {
      busy = false;
      updateControls();
      showStatus();
    }
  }

  function showCapture(capture) {
    if (!capture?.pipe_id || !capture.url) return;
    const stamp = capture.captured_at || "";
    if (stamp && stamp < capturedAt) return;
    const changed = pipeId !== capture.pipe_id;
    if (!changed && !retryImage) return;
    pipeId = capture.pipe_id;
    capturedAt = stamp;
    retryImage = false;
    image.hidden = true;
    placeholder.hidden = false;
    placeholder.querySelector("strong").textContent = "正在加载展开图";
    placeholder.querySelector("p").textContent = pipeId;
    imageMeta.textContent = `${pipeId} · ${capture.width} × ${capture.height}`;
    if (changed) {
      result = settings?.enabled ? { pipe_id: pipeId, status: "pending" } : null;
      error = "";
      form.hidden = true;
      form.reset();
      zoom = null;
      scroll.scrollTo(0, 0);
    }
    const url = new URL(capture.url, window.location.href);
    url.searchParams.set("pipe", pipeId);
    image.src = url.href;
    updateControls();
    showStatus();
    drawOverlay();
    if (changed) void watchResult();
  }

  async function watchCaptures() {
    for (;;) {
      try {
        const query = new URLSearchParams({ after: pipeId });
        const data = await readJson(`/api/unwrapped/wait?${query}`);
        showCapture(data.result);
      } catch {
        await new Promise(resolve => setTimeout(resolve, 1500));
      }
    }
  }

  async function pollCaptureState() {
    try {
      const { state } = await readJson("/api/state");
      showCapture(state.latest_unwrapped);
      // A failed capture can have a verdict even when it produced no panorama.
      const attempt = state.save_pipe_id;
      if (settings?.enabled && attempt && attempt !== pipeId && attempt !== lastChecked &&
          ["raw_saving", "raw_saved", "completed", "failed"].includes(state.save_phase)) {
        const displayed = pipeId;
        const data = await api(`/result?${new URLSearchParams({ pipe: attempt })}`);
        if (displayed === pipeId && data.result?.status === "error") {
          lastChecked = attempt;
          request?.abort();
          result = data.result;
          showStatus();
          drawOverlay();
        }
      }
    } catch {
      // The notification request also brings in new panoramas.
    } finally {
      setTimeout(pollCaptureState, 1500);
    }
  }

  image.addEventListener("load", () => {
    image.hidden = false;
    placeholder.hidden = true;
    sizeImage();
    updateControls();
    drawOverlay();
  });
  image.addEventListener("error", () => {
    retryImage = true;
    placeholder.querySelector("strong").textContent = "展开图读取失败";
    placeholder.querySelector("p").textContent = "正在重新获取当前批次。";
  });
  new ResizeObserver(sizeImage).observe(scroll);
  for (const button of zoomButtons) {
    button.addEventListener("click", () => {
      const action = button.dataset.zoom;
      const scale = zoom ?? fitScale();
      if (action === "fit") zoom = null;
      else if (action === "actual") zoom = 1;
      else zoom = Math.min(4, Math.max(.02, scale * (action === "in" ? 1.35 : 1 / 1.35)));
      sizeImage();
    });
  }
  select.addEventListener("change", async () => {
    if (await change("/config", { active_template: select.value })) void watchResult();
  });
  backend.addEventListener("change", () => change("/config", { backend: backend.value }));
  toggle.addEventListener("click", () => change("/enabled", { enabled: !settings.enabled }));
  record.addEventListener("click", () => {
    form.hidden = false;
    form.elements.namedItem("name").focus();
  });
  byId("pipeCancelTemplate").addEventListener("click", () => { form.hidden = true; });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (busy || !pipeId) return;
    const name = form.elements.namedItem("name").value;
    if (await change("/templates", { name, pipe_id: pipeId })) {
      form.reset();
      form.hidden = true;
      void watchResult();
    }
  });
  limits.addEventListener("submit", (event) => {
    event.preventDefault();
    const values = Object.fromEntries(Array.from(new FormData(limits), ([key, value]) =>
      [key, Number(value) / (key === "shape_iou" ? 100 : 1)]));
    void change("/config", { tolerances: values });
  });
  api().then((data) => { showSettings(data.state); showStatus(); }).catch((err) => {
    error = err.message;
    showStatus();
  });
  void watchCaptures();
  void pollCaptureState();
})();
