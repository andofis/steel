(() => {
  "use strict";

  const CAMERA_SERIALS = ["DB0959183", "DB2692106", "DB0426965", "DB0426961"];
  const CAMERA_NAMES = ["cam_0", "cam_90", "cam_180", "cam_270"];
  const STORAGE_KEY = "hikrobotCapture.ui.v1";

  const ui = {
    singleViewTab: document.getElementById("singleViewTab"),
    quadViewTab: document.getElementById("quadViewTab"),
    unwrappedViewTab: document.getElementById("unwrappedViewTab"),
    unwrappedView: document.getElementById("unwrappedView"),
    unwrappedImage: document.getElementById("unwrappedImage"),
    unwrapSizeForm: document.getElementById("unwrapSizeForm"),
    unwrapSizePreset: document.getElementById("unwrapSizePreset"),
    unwrapSizeActive: document.getElementById("unwrapSizeActive"),
    unwrapWidth: document.getElementById("unwrapWidth"),
    unwrapHeight: document.getElementById("unwrapHeight"),
    unwrapCustomDims: document.getElementById("unwrapCustomDims"),
    unwrapSizeHint: document.getElementById("unwrapSizeHint"),
    applyUnwrapSize: document.getElementById("applyUnwrapSize"),
    unwrappedMeta: document.getElementById("unwrappedMeta"),
    unwrappedPlaceholder: document.getElementById("unwrappedPlaceholder"),
    unwrappedScroll: document.getElementById("unwrappedScroll"),
    unwrappedZoomOutBtn: document.getElementById("unwrappedZoomOutBtn"),
    unwrappedFitBtn: document.getElementById("unwrappedFitBtn"),
    unwrappedZoomInBtn: document.getElementById("unwrappedZoomInBtn"),
    unwrappedOpenOutputBtn: document.getElementById("unwrappedOpenOutputBtn"),
    clockTime: document.getElementById("clockTime"),
    clockDate: document.getElementById("clockDate"),
    sdkBadgeDot: document.getElementById("sdkBadgeDot"),
    sdkBadgeText: document.getElementById("sdkBadgeText"),
    onlineSummary: document.getElementById("onlineSummary"),
    refreshBtn: document.getElementById("refreshBtn"),
    connectAllBtn: document.getElementById("connectAllBtn"),
    disconnectAllBtn: document.getElementById("disconnectAllBtn"),
    cameraList: document.getElementById("cameraList"),
    viewGrid: document.getElementById("viewGrid"),
    stageCameraName: document.getElementById("stageCameraName"),
    stageCameraDetail: document.getElementById("stageCameraDetail"),
    stageLiveDot: document.getElementById("stageLiveDot"),
    previewBtn: document.getElementById("previewBtn"),
    acquireBtn: document.getElementById("acquireBtn"),
    batchFlash: document.getElementById("batchFlash"),
    currentModeLabel: document.getElementById("currentModeLabel"),
    pipeCount: document.getElementById("pipeCount"),
    savedCount: document.getElementById("savedCount"),
    outputPath: document.getElementById("outputPath"),
    selectedIndexChip: document.getElementById("selectedIndexChip"),
    controlDeviceState: document.getElementById("controlDeviceState"),
    controlDeviceName: document.getElementById("controlDeviceName"),
    controlDeviceSerial: document.getElementById("controlDeviceSerial"),
    parameterState: document.getElementById("parameterState"),
    exposureInput: document.getElementById("exposureInput"),
    gainInput: document.getElementById("gainInput"),
    lineRateInput: document.getElementById("lineRateInput"),
    triggerSourceSelect: document.getElementById("triggerSourceSelect"),
    polaritySelect: document.getElementById("polaritySelect"),
    applyParamsBtn: document.getElementById("applyParamsBtn"),
    acquisitionState: document.getElementById("acquisitionState"),
    headDelayInput: document.getElementById("headDelayInput"),
    tailDelayInput: document.getElementById("tailDelayInput"),
    noSignalInput: document.getElementById("noSignalInput"),
    streamAlignInput: document.getElementById("streamAlignInput"),
    startAcquisitionBtn: document.getElementById("startAcquisitionBtn"),
    storageState: document.getElementById("storageState"),
    outputDirInput: document.getElementById("outputDirInput"),
    folderPrefixInput: document.getElementById("folderPrefixInput"),
    chooseOutputDirBtn: document.getElementById("chooseOutputDirBtn"),
    storageName: document.getElementById("storageName"),
    openOutputBtn: document.getElementById("openOutputBtn"),
    clearLogBtn: document.getElementById("clearLogBtn"),
    eventList: document.getElementById("eventList"),
    footerDot: document.getElementById("footerDot"),
    footerStatus: document.getElementById("footerStatus"),
    footerMode: document.getElementById("footerMode"),
    footerOnline: document.getElementById("footerOnline"),
    footerPipes: document.getElementById("footerPipes"),
    footerDirectory: document.getElementById("footerDirectory"),
    historySummary: document.getElementById("historySummary"),
    historyList: document.getElementById("historyList"),
    historyEmpty: document.getElementById("historyEmpty"),
    toastStack: document.getElementById("toastStack"),
    cameraCardTemplate: document.getElementById("cameraCardTemplate")
  };

  const savedUi = (() => {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); }
    catch { return {}; }
  })();
  const state = {
    selected: Number.isInteger(savedUi.selected) ? savedUi.selected : 0,
    view: savedUi.view === "quad" ? "quad" : savedUi.view === "unwrapped" ? "unwrapped" : "single",
    server: null,
    cards: [],
    viewports: [],
    previewImages: [null, null, null, null],
    previewInFlight: [false, false, false, false],
    inputDirty: false,
    unwrapSizeDirty: false,
    unwrapSizeSaving: false,
    actionBusy: false,
    previousPipeCount: 0,
    latestUnwrappedPipeId: "",
    latestUnwrappedAt: "",
    retryUnwrappedImage: false,
    pendingUnwrappedReveal: false,
    unwrappedZoom: 1,
    logsSignature: "",
    historySignature: "",
    pollTimer: null,
    previewTimer: null,
    renderTimer: null
  };

  let flashTimer = null;

  function saveUiState() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ selected: state.selected, view: state.view }));
  }

  function buildUi() {
    for (let index = 0; index < 4; index += 1) {
      const card = ui.cameraCardTemplate.content.firstElementChild.cloneNode(true);
      card.dataset.index = String(index);
      card.querySelector(".camera-name").textContent = CAMERA_NAMES[index];
      card.querySelector(".camera-serial").textContent = `SN ${CAMERA_SERIALS[index]}`;
      card.querySelector(".camera-select-surface").addEventListener("click", () => selectCamera(index));
      card.querySelector(".camera-connect-button").addEventListener("click", (event) => {
        event.stopPropagation();
        toggleConnection(index);
      });
      ui.cameraList.appendChild(card);
      state.cards.push(card);

      const viewport = document.createElement("div");
      viewport.className = "viewport offline";
      viewport.dataset.index = String(index);
      viewport.innerHTML = `
        <canvas></canvas>
        <div class="viewport-topline">
          <span class="viewport-id"><span class="viewport-live-dot"></span><span class="viewport-name">${CAMERA_NAMES[index]}</span></span>
          <span class="viewport-badge">NO SIGNAL</span>
        </div>
        <div class="viewport-bottomline">
          <span class="viewport-meta viewport-resolution">-- × --</span>
          <span class="viewport-meta viewport-stats">MVS 未连接</span>
        </div>
        <div class="viewport-empty">
          <div class="empty-core">
            <strong>未检测到相机</strong>
            <span>CHECK MVS / POE / CAMERA IP</span>
          </div>
        </div>`;
      viewport.addEventListener("click", () => selectCamera(index));
      ui.viewGrid.appendChild(viewport);
      state.viewports.push(viewport);
    }
    state.selected = Math.min(3, Math.max(0, state.selected));
    setView(state.view, false);
  }

  function selectCamera(index) {
    state.selected = index;
    state.inputDirty = false;
    if (state.view === "single") updateViewportVisibility();
    updateSelectedCard();
    renderCameraInfo();
    syncInputsFromConfig(state.server);
    updateActionButtons();
    saveUiState();
  }

  function setView(view, persist = true) {
    state.view = ["single", "quad", "unwrapped"].includes(view) ? view : "single";
    const showCameras = state.view !== "unwrapped";
    ui.viewGrid.classList.toggle("hidden", !showCameras);
    ui.unwrappedView.classList.toggle("hidden", showCameras);
    ui.viewGrid.classList.toggle("quad", state.view === "quad");
    ui.viewGrid.classList.toggle("single", state.view !== "quad");
    ui.singleViewTab.classList.toggle("active", state.view === "single");
    ui.quadViewTab.classList.toggle("active", state.view === "quad");
    ui.unwrappedViewTab.classList.toggle("active", state.view === "unwrapped");
    updateViewportVisibility();
    updateSelectedCard();
    if (showCameras) renderFrames();
    if (persist) saveUiState();
  }

  function setUnwrappedZoom(value) {
    state.unwrappedZoom = Math.min(4, Math.max(0.5, Number(value) || 1));
    ui.unwrappedImage.parentElement.style.width = `${Math.round(state.unwrappedZoom * 100)}%`;
  }

  function showUnwrapSize(spec) {
    if (!spec) return;
    const select = ui.unwrapSizePreset;
    const signature = `${spec.max_width}x${spec.max_height}`;
    if (select.dataset.reference !== signature) {
      const options = spec.presets.map(p => new Option(`${p.label}　${p.width} × ${p.height}`, `${p.width}x${p.height}`));
      select.replaceChildren(...options, new Option("自定义宽高…", "custom"));
      select.dataset.reference = signature;
      ui.unwrapWidth.min = spec.min_width; ui.unwrapWidth.max = spec.max_width;
      ui.unwrapHeight.min = spec.min_height; ui.unwrapHeight.max = spec.max_height;
    }
    // Keep a partially typed size intact when the camera changes or /api/state is polled.
    if (!state.unwrapSizeDirty) {
      const match = spec.presets.some(p => p.width === spec.width && p.height === spec.height);
      select.value = match ? `${spec.width}x${spec.height}` : "custom";
      ui.unwrapWidth.value = spec.width;
      ui.unwrapHeight.value = spec.height;
    }
    const custom = select.value === "custom";
    if (!custom) [ui.unwrapWidth.value, ui.unwrapHeight.value] = select.value.split("x");
    ui.unwrapCustomDims.hidden = !custom;
    select.disabled = state.unwrapSizeSaving;
    ui.unwrapWidth.disabled = ui.unwrapHeight.disabled = !custom || state.unwrapSizeSaving;
    const w = ui.unwrapWidth.valueAsNumber, h = ui.unwrapHeight.valueAsNumber;
    const valid = Number.isInteger(w) && w >= spec.min_width && w <= spec.max_width &&
      Number.isInteger(h) && h >= spec.min_height && h <= spec.max_height;
    ui.unwrapSizeActive.textContent = `当前 ${spec.width} × ${spec.height}`;
    ui.unwrapSizeActive.className = "tiny-status ok";
    ui.applyUnwrapSize.disabled = state.unwrapSizeSaving || !valid || (w === spec.width && h === spec.height);
    ui.unwrapSizeHint.textContent = valid
      ? `输出像素量为原档的 ${Number((100 * w * h / (spec.max_width * spec.max_height)).toFixed(1))}%。` +
        (Math.abs(w / spec.max_width - h / spec.max_height) > 0.001 ? "宽高未等比缩放，孔的外形比例会改变。" : "")
      : `请输入整数：宽 ${spec.min_width}–${spec.max_width}，高 ${spec.min_height}–${spec.max_height}。`;
  }

  async function saveUnwrapSize(event) {
    event.preventDefault();
    if (ui.applyUnwrapSize.disabled || !state.server?.unwrap_size) return;
    const size = {width: ui.unwrapWidth.valueAsNumber, height: ui.unwrapHeight.valueAsNumber};
    state.unwrapSizeSaving = true;
    showUnwrapSize(state.server.unwrap_size);
    try {
      await apiRequest("/api/unwrap-size", size);
      state.unwrapSizeDirty = false;
      showToast("展开尺寸已保存", `下一根钢管起使用 ${size.width} × ${size.height}。`, "success");
    } catch (error) {
      showToast("展开尺寸未保存", error.message, "error");
    } finally {
      state.unwrapSizeSaving = false;
      showUnwrapSize(state.server.unwrap_size);
    }
  }

  function updateUnwrappedResult(result, reveal = false) {
    if (!result || !result.url) return false;
    const pipeId = String(result.pipe_id || "");
    const changed = pipeId !== state.latestUnwrappedPipeId;
    const stamp = result.captured_at || "";
    // A full-state response already in flight must not undo a newer result notification.
    if (changed && stamp && stamp < state.latestUnwrappedAt) return false;
    if (changed || state.retryUnwrappedImage) {
      state.latestUnwrappedPipeId = pipeId;
      state.latestUnwrappedAt = stamp;
      state.retryUnwrappedImage = false;
      // Do not show the previous pipe under the new batch number during fetch/decode.
      ui.unwrappedImage.hidden = true;
      ui.unwrappedPlaceholder.hidden = false;
      ui.unwrappedPlaceholder.querySelector("strong").textContent = "正在加载展开图";
      ui.unwrappedPlaceholder.querySelector("p").textContent = pipeId;
      ui.unwrappedImage.onload = () => {
        ui.unwrappedImage.hidden = false;
        ui.unwrappedPlaceholder.hidden = true;
      };
      ui.unwrappedImage.onerror = () => {
        state.retryUnwrappedImage = true;
        ui.unwrappedImage.hidden = true;
        ui.unwrappedPlaceholder.hidden = false;
        ui.unwrappedPlaceholder.querySelector("strong").textContent = "展开图读取失败";
        ui.unwrappedPlaceholder.querySelector("p").textContent = "正在重新获取当前批次结果。";
      };
      const url = new URL(result.url, window.location.href);
      url.searchParams.set("pipe", pipeId);
      ui.unwrappedImage.src = url.href;
      if (changed) setUnwrappedZoom(1);
    }
    const meta = [`${result.width || "?"} × ${result.height || "?"}`, pipeId || "最新结果"];
    if (result.result_ready_ms) meta.push(`触发至可取图 ${Math.round(result.result_ready_ms)} ms`);
    else if (result.elapsed_ms) meta.push(`展开算法 ${Math.round(result.elapsed_ms)} ms`);
    const seam = result.panorama_seam;
    if (seam?.status === "no_clear_column") meta.push(`无共同空隙，切口仍经过 ${seam.crossing_candidates} 个候选孔，请复核`);
    else if (seam?.status === "limited_clearance") meta.push("切口已避孔，但边缘余量不足");
    else if (seam?.status === "no_candidates") meta.push("未找到孔候选，保留标定起点");
    else if (seam?.status === "shifted") meta.push(`已避开孔口（圆周起点 ${seam.cut_x} px）`);
    if (result.archive_status === "failed") meta.push("展开图归档失败，请检查磁盘和日志");
    else if (result.archive_status === "pending") meta.push("展开图后台归档中");
    if (result.raw_archive_error) meta.push("原图归档失败，请检查磁盘和日志");
    else if (result.raw_images_ready === false) meta.push("四张原图待保存");
    ui.unwrappedMeta.textContent = meta.join(" · ");
    if (reveal) setView("unwrapped");
    return changed;
  }

  async function watchUnwrappedResults() {
    // One outstanding lightweight request per tab. Normal state polling remains the fallback.
    let after = state.latestUnwrappedPipeId;
    for (;;) {
      try {
        const response = await fetch(`/api/unwrapped/wait?after=${encodeURIComponent(after)}`);
        if (!response.ok) throw new Error("结果通知暂不可用");
        const { result } = await response.json();
        if (result) after = result.pipe_id;
        if (result && updateUnwrappedResult(result)) {
          setView("unwrapped");
          showCaptureFlash("展开图已生成");
        }
      } catch {
        await new Promise(resolve => window.setTimeout(resolve, 1500));
      }
    }
  }

  function updateViewportVisibility() {
    state.viewports.forEach((viewport, index) => {
      viewport.classList.toggle("hidden", state.view === "single" && index !== state.selected);
    });
  }

  function updateSelectedCard() {
    state.cards.forEach((card, index) => card.classList.toggle("selected", index === state.selected));
    ui.selectedIndexChip.textContent = `CAM ${state.selected}`;
  }
  function createNoiseTile() {
    const canvas = document.createElement("canvas");
    canvas.width = 128;
    canvas.height = 128;
    const context = canvas.getContext("2d");
    const image = context.createImageData(128, 128);
    let seed = 90210;
    for (let offset = 0; offset < image.data.length; offset += 4) {
      seed = (seed * 16807) % 2147483647;
      const value = 92 + (seed % 70);
      image.data[offset] = value;
      image.data[offset + 1] = value + 2;
      image.data[offset + 2] = value + 3;
      image.data[offset + 3] = 255;
    }
    context.putImageData(image, 0, 0);
    return canvas;
  }

  const noiseTile = createNoiseTile();
  let renderCounter = 0;

  function drawDemoFrame(context, width, height, index, time) {
    const phase = time / 1400 + index * 0.4;
    const background = context.createLinearGradient(0, 0, width, height);
    background.addColorStop(0, "#0a1012");
    background.addColorStop(0.48, "#151c1d");
    background.addColorStop(1, "#060b0d");
    context.fillStyle = background;
    context.fillRect(0, 0, width, height);

    const inset = width * 0.04;
    const metal = context.createLinearGradient(inset, 0, width - inset, 0);
    metal.addColorStop(0, "#161d1f");
    metal.addColorStop(0.09, "#737b79");
    metal.addColorStop(0.22, "#272d2e");
    metal.addColorStop(0.4, "#b9beb8");
    metal.addColorStop(0.52, "#555c5b");
    metal.addColorStop(0.64, "#d0d2cb");
    metal.addColorStop(0.78, "#444b4d");
    metal.addColorStop(0.91, "#909793");
    metal.addColorStop(1, "#161c1e");
    context.fillStyle = metal;
    context.fillRect(inset, 0, width - inset * 2, height);

    context.save();
    context.globalAlpha = 0.16;
    for (let line = 0; line < 52; line += 1) {
      const fraction = Math.abs(Math.sin((line + 2) * (9.73 + index)));
      const x = inset + fraction * (width - inset * 2);
      context.strokeStyle = fraction > 0.52 ? "#f4f4ec" : "#080f11";
      context.lineWidth = fraction > 0.9 ? 1.25 : 0.55;
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x + Math.sin(phase + line) * 2, height);
      context.stroke();
    }
    context.restore();

    const seamX = inset + (width - inset * 2) * (0.57 + Math.sin(phase * 0.62) * 0.02);
    const glow = context.createLinearGradient(seamX - 22, 0, seamX + 22, 0);
    glow.addColorStop(0, "rgba(0,0,0,0)");
    glow.addColorStop(0.42, "rgba(235,240,230,0.22)");
    glow.addColorStop(0.54, "rgba(255,255,246,0.5)");
    glow.addColorStop(0.66, "rgba(18,22,22,0.52)");
    glow.addColorStop(1, "rgba(0,0,0,0)");
    context.fillStyle = glow;
    context.fillRect(seamX - 24, 0, 48, height);

    context.save();
    context.globalCompositeOperation = "overlay";
    context.globalAlpha = 0.1;
    const pattern = context.createPattern(noiseTile, "repeat");
    if (pattern) {
      context.fillStyle = pattern;
      context.fillRect(0, 0, width, height);
    }
    context.restore();

    const sheen = context.createLinearGradient(0, 0, width, 0);
    sheen.addColorStop(0.19 + Math.sin(phase) * 0.08, "rgba(255,255,255,0)");
    sheen.addColorStop(0.5 + Math.sin(phase) * 0.08, "rgba(255,255,255,0.23)");
    sheen.addColorStop(0.82 + Math.sin(phase) * 0.08, "rgba(255,255,255,0)");
    context.fillStyle = sheen;
    context.fillRect(0, 0, width, height);

    context.globalAlpha = 0.1;
    context.fillStyle = "#000";
    for (let y = 0; y < height; y += 4) context.fillRect(0, y, width, 1);
    context.globalAlpha = 1;
  }

  function resizeCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 1.4);
    const width = Math.max(1, Math.round(rect.width * ratio));
    const height = Math.max(1, Math.round(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    return { context: canvas.getContext("2d"), width, height };
  }

  function drawImageContain(context, image, width, height) {
    const scale = Math.min(width / image.naturalWidth, height / image.naturalHeight);
    const drawWidth = image.naturalWidth * scale;
    const drawHeight = image.naturalHeight * scale;
    const x = (width - drawWidth) / 2;
    const y = (height - drawHeight) / 2;
    context.fillStyle = "#020609";
    context.fillRect(0, 0, width, height);
    context.drawImage(image, x, y, drawWidth, drawHeight);
  }

  function drawCamera(cameraIndex, canvas, time) {
    if (!canvas || !canvas.getBoundingClientRect().width) return;
    const camera = state.server?.cameras?.[cameraIndex];
    const { context, width, height } = resizeCanvas(canvas);
    context.clearRect(0, 0, width, height);
    const preview = state.previewImages[cameraIndex];
    if (camera?.status === "online" && preview) {
      drawImageContain(context, preview, width, height);
    } else if (!camera || camera.status !== "online") {
      drawDemoFrame(context, width, height, cameraIndex, time);
    } else {
      context.fillStyle = "#03080b";
      context.fillRect(0, 0, width, height);
    }
  }

  function renderFrames() {
    if (state.view === "unwrapped") return;
    const now = performance.now();
    state.cards.forEach((card, index) => {
      drawCamera(index, card.querySelector(".camera-thumb"), now);
    });
    state.viewports.forEach((viewport, index) => {
      if (state.view === "single" && index !== state.selected) return;
      drawCamera(index, viewport.querySelector("canvas"), now);
    });
  }

  function startRenderLoop() {
    if (state.renderTimer) return;
    const loop = () => {
      renderCounter += 1;
      if (renderCounter % 3 === 0) renderFrames();
      state.renderTimer = requestAnimationFrame(loop);
    };
    state.renderTimer = requestAnimationFrame(loop);
  }
  async function apiRequest(path, body = null) {
    const options = { method: body === null ? "GET" : "POST", headers: {} };
    if (body !== null) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    const response = await fetch(path, options);
    let data = {};
    try { data = await response.json(); } catch { data = {}; }
    if (!response.ok || data.ok === false) {
      throw new Error(data.error || `请求失败 (${response.status})`);
    }
    if (data.state) updateServerState(data.state);
    return data;
  }

  async function runAction(button, pendingText, action) {
    if (state.actionBusy) return;
    state.actionBusy = true;
    const originalHtml = button?.innerHTML;
    if (button) {
      button.disabled = true;
      if (pendingText) button.textContent = pendingText;
    }
    try {
      await action();
    } catch (error) {
      showToast("操作失败", error.message || String(error), "error");
    } finally {
      state.actionBusy = false;
      if (button) {
        button.disabled = false;
        if (originalHtml) button.innerHTML = originalHtml;
      }
      updateActionButtons();
    }
  }

  function collectConfig() {
    return {
      slot: state.selected,
      exposure_us: Number(ui.exposureInput.value),
      gain_db: Number(ui.gainInput.value),
      line_rate_hz: Number(ui.lineRateInput.value),
      trigger_source: ui.triggerSourceSelect.value,
      trigger_polarity: ui.polaritySelect.value,
      head_delay_ms: Number(ui.headDelayInput.value),
      tail_delay_ms: Number(ui.tailDelayInput.value),
      no_signal_ms: Number(ui.noSignalInput.value),
      stream_align_ms: Number(ui.streamAlignInput.value),
      folder_prefix: ui.folderPrefixInput.value.trim() || "pipe",
      output_dir: ui.outputDirInput.value.trim() || "captures"
    };
  }

  async function applyParameters(options = {}) {
    const data = await apiRequest("/api/config", collectConfig());
    state.inputDirty = false;
    const camera = data.state?.cameras?.find((item) => item.slot === state.selected);
    if (camera?.error) {
      ui.parameterState.textContent = "部分失败";
      ui.parameterState.className = "tiny-status warn";
      showToast("参数未完全写入", camera.error, "warn");
      return;
    }
    ui.parameterState.textContent = "已应用";
    ui.parameterState.className = "tiny-status ok";
    if (!options.silent) showToast("参数已应用", `已写入 ${CAMERA_NAMES[state.selected]} 的曝光、增益、行频与触发参数。`, "success");
  }

  async function refreshDevices() {
    const data = await apiRequest("/api/refresh", {});
    if (data.device_count > 0) {
      showToast("相机列表已刷新", `MVS 发现 ${data.device_count} 台相机。`, "success");
    } else {
      showToast("未发现相机", "请检查 PoE 供电、网线、相机 IP 与 MVS 客户端设备列表。", "warn");
    }
  }

  async function connectAll() {
    const data = await apiRequest("/api/connect", {});
    const online = (data.state?.cameras || []).filter((camera) => camera.status === "online").length;
    showToast(online ? "相机连接完成" : "相机连接失败", online ? `已连接 ${online} / 4 台相机。` : "MVS 未发现设备，请检查相机链路。", online ? "success" : "error");
  }

  async function disconnectAll() {
    await apiRequest("/api/disconnect", {});
    showToast("已断开", "全部相机连接已关闭。", "success");
  }

  async function toggleConnection(slot) {
    const camera = state.server?.cameras?.[slot];
    if (!camera) return;
    try {
      if (camera.status === "online") {
        await apiRequest("/api/disconnect/one", { slot });
        showToast(`${camera.name} 已断开`, "该通道不再取流。", "success");
      } else {
        await apiRequest("/api/connect/one", { slot });
        showToast(`${camera.name} 已连接`, "可以开始实时预览或触发采集。", "success");
      }
    } catch (error) {
      showToast("通道操作失败", error.message, "error");
    }
  }

  async function handlePreview() {
    const mode = state.server?.mode || "idle";
    if (mode === "preview") {
      await apiRequest("/api/preview/stop", {});
      showToast("实时预览已停止", "相机已停止取流。", "success");
      return;
    }
    await applyParameters({ silent: true });
    await apiRequest("/api/preview/start", {});
    showToast("实时预览已启动", "四台已连接相机正在连续取流。", "success");
  }

  async function handleAcquisition() {
    const mode = state.server?.mode || "idle";
    if (mode === "acquisition") {
      await apiRequest("/api/acquisition/stop", {});
      showToast("触发采集已停止", "等待中的光电触发已取消。", "success");
      return;
    }
    await applyParameters({ silent: true });
    await apiRequest("/api/acquisition/start", {});
    showToast("已开始等待光电触发", "每来一根钢管将自动保存四张完整图像。", "success");
  }

  async function chooseOutputDirectory() {
    const data = await apiRequest("/api/select-output-dir", {});
    if (data.state) updateServerState(data.state);
    if (data.path) {
      showToast("保存目录已更改", data.path, "success");
    }
  }
  async function openOutputDirectory() {
    const data = await apiRequest("/api/open-output", {});
    showToast("已打开保存目录", data.path || "captures", "success");
  }

  function showCaptureFlash(text) {
    ui.batchFlash.querySelector("strong").textContent = text;
    ui.batchFlash.classList.add("visible");
    window.clearTimeout(flashTimer);
    flashTimer = window.setTimeout(() => ui.batchFlash.classList.remove("visible"), 1600);
  }

  function showToast(title, message, type = "success") {
    const node = document.createElement("div");
    node.className = `toast ${type}`;
    node.innerHTML = `<span class="toast-bar"></span><div class="toast-content"><strong></strong><span></span></div>`;
    node.querySelector("strong").textContent = title;
    node.querySelector(".toast-content span:last-child").textContent = message;
    ui.toastStack.appendChild(node);
    requestAnimationFrame(() => node.classList.add("visible"));
    window.setTimeout(() => {
      node.classList.remove("visible");
      window.setTimeout(() => node.remove(), 220);
    }, 4600);
  }
  function syncInputsFromConfig(server) {
    if (!server || state.inputDirty) return;
    const config = server.camera_configs?.[state.selected] || server.config || {};
    ui.exposureInput.value = config.exposure_us ?? 182;
    ui.gainInput.value = Number(config.gain_db ?? 0).toFixed(1);
    ui.lineRateInput.value = config.line_rate_hz ?? 4100;
    ui.triggerSourceSelect.value = config.trigger_source || "Line0";
    ui.polaritySelect.value = config.trigger_polarity || "LevelLow";
    ui.headDelayInput.value = config.head_delay_ms ?? 0;
    ui.tailDelayInput.value = config.tail_delay_ms ?? 430;
    ui.noSignalInput.value = config.no_signal_ms ?? 80;
    ui.streamAlignInput.value = server.config?.stream_align_ms ?? 1000;
    ui.folderPrefixInput.value = server.config?.folder_prefix || "pipe";
    ui.outputDirInput.value = server.config?.output_dir || "captures";
  }

  function updateServerState(server) {
    if (!server) return;
    const previousPipes = state.previousPipeCount;
    state.server = server;
    showUnwrapSize(server.unwrap_size);
    syncInputsFromConfig(server);
    updateSdkStatus(server);
    updateCameraCards(server);
    renderCameraInfo();
    updateStatusText(server);
    renderLogs(server.logs || []);
    renderCaptureHistory(server);
    updateActionButtons();

    ui.pipeCount.textContent = String(server.pipe_count || 0);
    ui.savedCount.textContent = String(server.saved_count || 0);
    ui.outputPath.textContent = server.output_path || "captures";
    ui.storageName.textContent = server.output_path || "captures";
    ui.footerDirectory.textContent = `输出：${server.output_path || "captures"}`;
    ui.footerPipes.textContent = String(server.pipe_count || 0);

    const resultChanged = updateUnwrappedResult(server.latest_unwrapped, false);
    const sameActiveSave = Boolean(
      server.save_pipe_id &&
      server.latest_unwrapped?.pipe_id === server.save_pipe_id
    );
    const resultReadyPhases = ["unwrapped_ready", "raw_saving", "raw_saved", "completed"];
    if (resultChanged && sameActiveSave && resultReadyPhases.includes(server.save_phase)) {
      setView("unwrapped");
      state.pendingUnwrappedReveal = false;
      showCaptureFlash("展开图已生成");
      showToast(
        "展开图已生成",
        "已按所选尺寸发布展开图；保存状态见结果栏和采集记录。",
        "success",
      );
    }
    if (server.pipe_count > previousPipes) {
      showCaptureFlash("钢管处理完成");
      showToast("钢管处理完成", `累计 ${server.pipe_count} 根钢管，归档结果见采集记录。`, "success");
      state.previousPipeCount = server.pipe_count;
    } else if (state.previousPipeCount === 0) {
      state.previousPipeCount = server.pipe_count || 0;
    }
  }

  function updateSdkStatus(server) {
    if (server.sdk_ready) {
      ui.sdkBadgeText.textContent = `MVS SDK 就绪 · ${server.sdk_version || ""}`;
      ui.sdkBadgeDot.style.background = "var(--accent)";
      ui.sdkBadgeDot.style.boxShadow = "0 0 10px rgba(104, 227, 194, 0.7)";
    } else {
      ui.sdkBadgeText.textContent = "MVS SDK 不可用";
      ui.sdkBadgeDot.style.background = "var(--danger)";
      ui.sdkBadgeDot.style.boxShadow = "0 0 10px rgba(255, 107, 120, 0.65)";
    }
  }

  function updateCameraCards(server) {
    const online = (server.cameras || []).filter((camera) => camera.status === "online").length;
    ui.onlineSummary.textContent = `${online} / 4 在线`;
    ui.footerOnline.textContent = `${online} / 4`;

    (server.cameras || []).forEach((camera, index) => {
      const card = state.cards[index];
      const viewport = state.viewports[index];
      const isOnline = camera.status === "online";
      card.classList.toggle("online", isOnline);
      card.classList.toggle("demo", false);
      card.querySelector(".camera-device").textContent = camera.error
        ? camera.error
        : isOnline ? `${camera.model || "MV-CL024-91GM"} · ${camera.ip || "已连接"}` : "未检测到相机";
      const connectButton = card.querySelector(".camera-connect-button");
      connectButton.textContent = camera.status === "connecting" ? "连接中" : isOnline ? "断开" : "连接";
      connectButton.disabled = camera.status === "connecting" || state.actionBusy;

      viewport.classList.toggle("offline", !isOnline);
      viewport.querySelector(".viewport-badge").textContent = isOnline
        ? (server.preview_paused ? "预览已暂停" : server.mode === "acquisition" ? "TRIGGER" : server.mode === "preview" ? "LIVE" : "MVS")
        : "NO SIGNAL";
      viewport.querySelector(".viewport-resolution").textContent = isOnline && camera.width
        ? `${camera.width} × ${camera.height || "?"}`
        : "-- × --";
      viewport.querySelector(".viewport-stats").textContent = isOnline
        ? `帧 ${camera.frame_count || 0}`
        : "MVS 未连接";
      viewport.querySelector(".empty-core strong").textContent = camera.error ? "相机连接异常" : "未检测到相机";
      viewport.querySelector(".empty-core span").textContent = camera.error ? camera.error : "CHECK MVS / POE / CAMERA IP";
      if (!isOnline) state.previewImages[index] = null;
    });
    updateSelectedCard();
  }

  function renderCameraInfo() {
    const camera = state.server?.cameras?.[state.selected];
    if (!camera) {
      ui.stageCameraName.textContent = CAMERA_NAMES[state.selected];
      ui.stageCameraDetail.textContent = "正在读取 MVS 状态";
      ui.controlDeviceName.textContent = "MV-CL024-91GM";
      ui.controlDeviceSerial.textContent = `SERIAL: ${CAMERA_SERIALS[state.selected]}`;
      ui.controlDeviceState.textContent = "等待连接";
      return;
    }
    const isOnline = camera.status === "online";
    ui.stageCameraName.textContent = camera.name;
    ui.stageCameraDetail.textContent = isOnline
      ? `${camera.model || "MV-CL024-91GM"} · ${camera.serial} · ${camera.ip || "已连接"}`
      : (camera.error || `未发现序列号 ${camera.serial}`);
    ui.controlDeviceName.textContent = camera.model || "MV-CL024-91GM";
    ui.controlDeviceSerial.textContent = `SERIAL: ${camera.serial}`;
    ui.controlDeviceState.textContent = isOnline ? "海康相机在线" : "等待连接";
    ui.controlDeviceState.className = `tiny-status ${isOnline ? "ok" : "warn"}`;
    const mode = state.server?.mode || "idle";
    ui.stageLiveDot.className = `live-indicator${mode === "acquisition" ? " capturing" : isOnline ? " online" : ""}`;
  }

  function updateStatusText(server) {
    const modeLabels = { idle: "空闲", preview: "实时预览", acquisition: "等待光电触发" };
    const modeLabel = modeLabels[server.mode] || "空闲";
    const online = (server.cameras || []).filter((camera) => camera.status === "online").length;
    ui.currentModeLabel.textContent = server.preview_paused ? "展开中，预览已暂停" : modeLabel;
    ui.footerMode.textContent = modeLabel;
    ui.acquisitionState.textContent = server.mode === "acquisition" ? "运行中" : "已停止";
    ui.acquisitionState.className = `tiny-status ${server.mode === "acquisition" ? "warn" : ""}`;
    if (!server.sdk_ready) {
      ui.footerStatus.textContent = server.sdk_error || "MVS SDK 未就绪";
      ui.footerDot.style.background = "var(--danger)";
    } else if (server.preview_paused) {
      ui.footerStatus.textContent = "展开中 · 实时预览已暂停 · 相机采集继续";
      ui.footerDot.style.background = "var(--amber)";
    } else if (server.mode === "acquisition") {
      ui.footerStatus.textContent = "采集中 · 等待 Line0 光电触发";
      ui.footerDot.style.background = "var(--danger)";
    } else if (server.mode === "preview") {
      ui.footerStatus.textContent = `实时预览中 · ${online} 路相机`;
      ui.footerDot.style.background = "var(--accent)";
    } else if (online === 0) {
      ui.footerStatus.textContent = "MVS 就绪 · 等待相机上线";
      ui.footerDot.style.background = "var(--amber)";
    } else {
      ui.footerStatus.textContent = `相机已连接 · ${online} / 4`;
      ui.footerDot.style.background = "var(--accent)";
    }
  }

  function renderLogs(logs) {
    const signature = logs.map((item) => `${item.time}|${item.level}|${item.message}`).join("\n");
    if (signature === state.logsSignature) return;
    state.logsSignature = signature;
    ui.eventList.innerHTML = "";
    [...logs].reverse().slice(0, 35).forEach((entry) => {
      const item = document.createElement("div");
      item.className = "event-item";
      const time = document.createElement("time");
      time.textContent = entry.time || "--:--:--";
      const text = document.createElement("span");
      text.textContent = entry.message || "";
      if (entry.level === "ERROR") text.style.color = "#ff9aa3";
      if (entry.level === "WARN") text.style.color = "#ddbd7d";
      item.append(time, text);
      ui.eventList.appendChild(item);
    });
  }
  function renderCaptureHistory(server) {
    const history = Array.isArray(server.capture_history) ? server.capture_history : [];
    const signature = history.map((item) => `${item.序号}|${item.文件夹}|${item.图像数量}|${item.采集状态}|${item.完成时间}`).join("\n");
    if (signature === state.historySignature) return;
    state.historySignature = signature;
    const totalImages = history.reduce((sum, item) => sum + Number(item.图像数量 || 0), 0);
    ui.historySummary.textContent = `${history.length} 个文件夹 · ${totalImages} 张图像 · 钢管总数 ${server.pipe_count || 0}`;
    ui.historyList.innerHTML = "";
    if (!history.length) {
      const empty = document.createElement("div");
      empty.className = "history-empty";
      empty.textContent = "尚未完成钢管采集";
      ui.historyList.appendChild(empty);
      return;
    }
    [...history].sort((a, b) => Number(b.序号 || 0) - Number(a.序号 || 0)).forEach((item) => {
      const row = document.createElement("div");
      row.className = "history-row";
      const seq = document.createElement("span");
      seq.textContent = String(item.序号 || "");
      const folder = document.createElement("span");
      folder.className = "history-folder";
      folder.textContent = item.文件夹 || "";
      folder.title = item.文件夹路径 || item.文件夹 || "";
      const images = document.createElement("span");
      images.textContent = `${item.图像数量 || 0} / ${item.预期图像数量 || 5}`;
      const status = document.createElement("span");
      status.textContent = `${item.采集状态 || "未知"} · ${item.展开状态 || "未知"}`;
      if (item.归档失败) status.textContent += " · 归档失败";
      status.className = item.采集状态 === "完整" && item.展开状态 === "成功" && !item.归档失败 ? "history-ok" : "history-warn";
      const completed = document.createElement("span");
      completed.textContent = String(item.完成时间 || "").slice(11, 23) || "—";
      row.append(seq, folder, images, status, completed);
      ui.historyList.appendChild(row);
    });
  }

  function setButtonLabel(button, label) {
    if (!button) return;
    const icon = button.querySelector("svg");
    button.textContent = "";
    if (icon) button.appendChild(icon);
    button.appendChild(document.createTextNode(` ${label}`));
  }

  function updateActionButtons() {
    const server = state.server;
    const mode = server?.mode || "idle";
    const online = (server?.cameras || []).filter((camera) => camera.status === "online").length;
    const sdkReady = Boolean(server?.sdk_ready);
    ui.refreshBtn.disabled = state.actionBusy || !sdkReady;
    ui.connectAllBtn.disabled = state.actionBusy || !sdkReady;
    ui.disconnectAllBtn.disabled = state.actionBusy || online === 0;
    ui.applyParamsBtn.disabled = state.actionBusy || online === 0;
    ui.previewBtn.disabled = state.actionBusy || !sdkReady;
    ui.acquireBtn.disabled = state.actionBusy || !sdkReady;
    ui.openOutputBtn.disabled = state.actionBusy;

    setButtonLabel(ui.previewBtn, mode === "preview" ? "停止实时预览" : "开始实时预览");
    setButtonLabel(ui.acquireBtn, mode === "acquisition" ? "停止触发采集" : "开始触发采集");

    ui.startAcquisitionBtn.classList.toggle("running", mode === "acquisition");
    ui.startAcquisitionBtn.querySelector("span").textContent = mode === "acquisition" ? "停止光电触发采集" : "开始等待光电触发";
    ui.parameterState.textContent = state.inputDirty ? "待应用" : "已就绪";
    ui.parameterState.className = `tiny-status ${state.inputDirty ? "warn" : "ok"}`;
  }

  function pollPreviews() {
    if (state.server?.preview_paused) return;
    const cameras = state.server?.cameras || [];
    cameras.forEach((camera, index) => {
      if (camera.status !== "online" || state.previewInFlight[index]) return;
      state.previewInFlight[index] = true;
      const image = new Image();
      image.decoding = "async";
      image.onload = () => {
        state.previewImages[index] = image;
        state.previewInFlight[index] = false;
      };
      image.onerror = () => {
        state.previewInFlight[index] = false;
      };
      image.src = `/api/preview/${index}.bmp?t=${Date.now()}`;
    });
  }

  async function pollState() {
    try {
      const data = await apiRequest("/api/state");
      if (data.state) updateServerState(data.state);
    } catch (error) {
      ui.footerStatus.textContent = "本地采集服务连接中断";
      ui.footerDot.style.background = "var(--danger)";
    }
  }

  function updateClock() {
    const now = new Date();
    const pad = (value) => String(value).padStart(2, "0");
    ui.clockTime.textContent = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
    ui.clockDate.textContent = `${now.getFullYear()}/${pad(now.getMonth() + 1)}/${pad(now.getDate())}`;
  }

  function bindEvents() {
    function editUnwrapSize() {
      state.unwrapSizeDirty = true;
      showUnwrapSize(state.server?.unwrap_size);
    }
    ui.unwrapSizePreset.addEventListener("change", editUnwrapSize);
    ui.unwrapWidth.addEventListener("input", editUnwrapSize);
    ui.unwrapHeight.addEventListener("input", editUnwrapSize);
    ui.unwrapSizeForm.addEventListener("submit", saveUnwrapSize);
    ui.singleViewTab.addEventListener("click", () => setView("single"));
    ui.quadViewTab.addEventListener("click", () => setView("quad"));
    ui.unwrappedViewTab.addEventListener("click", () => setView("unwrapped"));
    ui.unwrappedZoomOutBtn.addEventListener("click", () => setUnwrappedZoom(state.unwrappedZoom - 0.25));
    ui.unwrappedFitBtn.addEventListener("click", () => setUnwrappedZoom(1));
    ui.unwrappedZoomInBtn.addEventListener("click", () => setUnwrappedZoom(state.unwrappedZoom + 0.25));
    ui.unwrappedOpenOutputBtn.addEventListener("click", () => runAction(ui.unwrappedOpenOutputBtn, "打开中", openOutputDirectory));
    ui.refreshBtn.addEventListener("click", () => runAction(ui.refreshBtn, "刷新中", refreshDevices));
    ui.connectAllBtn.addEventListener("click", () => runAction(ui.connectAllBtn, "连接中", connectAll));
    ui.disconnectAllBtn.addEventListener("click", () => runAction(ui.disconnectAllBtn, "断开中", disconnectAll));
    ui.previewBtn.addEventListener("click", () => runAction(ui.previewBtn, "处理中", handlePreview));
    ui.acquireBtn.addEventListener("click", () => runAction(ui.acquireBtn, "处理中", handleAcquisition));
    ui.startAcquisitionBtn.addEventListener("click", () => runAction(ui.startAcquisitionBtn, "处理中", handleAcquisition));
    ui.applyParamsBtn.addEventListener("click", () => runAction(ui.applyParamsBtn, "应用中", () => applyParameters()));
    ui.openOutputBtn.addEventListener("click", () => runAction(ui.openOutputBtn, "打开中", openOutputDirectory));
    ui.chooseOutputDirBtn.addEventListener("click", () => runAction(ui.chooseOutputDirBtn, "选择中", chooseOutputDirectory));

    ui.clearLogBtn.addEventListener("click", () => {
      ui.eventList.innerHTML = "";
      state.logsSignature = (state.server?.logs || []).map((item) => `${item.time}|${item.level}|${item.message}`).join("\n");
      showToast("日志已清空", "新的运行事件仍会继续记录。", "success");
    });

    const markDirty = () => {
      state.inputDirty = true;
      updateActionButtons();
    };
    [ui.exposureInput, ui.gainInput, ui.lineRateInput, ui.triggerSourceSelect, ui.polaritySelect,
      ui.headDelayInput, ui.tailDelayInput, ui.noSignalInput, ui.streamAlignInput, ui.folderPrefixInput, ui.outputDirInput].forEach((control) => {
      control.addEventListener("change", markDirty);
      control.addEventListener("input", markDirty);
    });

    window.addEventListener("resize", () => window.setTimeout(renderFrames, 120));
    document.addEventListener("keydown", (event) => {
      if (event.ctrlKey && event.key === "Enter") {
        event.preventDefault();
        runAction(ui.acquireBtn, "处理中", handleAcquisition);
      }
    });

    window.addEventListener("beforeunload", () => {
      if (state.pollTimer) window.clearInterval(state.pollTimer);
      if (state.previewTimer) window.clearInterval(state.previewTimer);
    });
  }

  async function init() {
    buildUi();
    bindEvents();
    updateClock();
    window.setInterval(updateClock, 1000);
    startRenderLoop();
    await pollState();
    watchUnwrappedResults();
    state.pollTimer = window.setInterval(pollState, 900);
    state.previewTimer = window.setInterval(pollPreviews, 360);
    window.setTimeout(() => {
      const server = state.server;
      if (!server) return;
      if (!server.sdk_ready) {
        showToast("MVS SDK 未就绪", server.sdk_error || "请检查 MVS 安装。", "error");
      } else {
        const online = (server.cameras || []).filter((camera) => camera.status === "online").length;
        if (online === 0) {
          showToast("尚未发现相机", "请确认相机供电、网线和 IP 后点击“刷新”。", "warn");
        }
      }
    }, 800);
  }

  init().catch((error) => {
    showToast("界面初始化失败", error.message || String(error), "error");
  });
})();
