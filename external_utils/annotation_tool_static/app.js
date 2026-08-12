const elements = {
  datasetName: document.querySelector("#dataset-name"),
  datasetProgress: document.querySelector("#dataset-progress"),
  saveStatus: document.querySelector("#save-status"),
  discardButton: document.querySelector("#discard-button"),
  commitButton: document.querySelector("#commit-button"),
  episodeSelect: document.querySelector("#episode-select"),
  previousEpisode: document.querySelector("#previous-episode"),
  nextEpisode: document.querySelector("#next-episode"),
  cameraButtons: document.querySelector("#camera-buttons"),
  frameImage: document.querySelector("#frame-image"),
  imageLoading: document.querySelector("#image-loading"),
  imageError: document.querySelector("#image-error"),
  frameBadge: document.querySelector("#frame-badge"),
  skipTenBack: document.querySelector("#skip-ten-back"),
  previousFrame: document.querySelector("#previous-frame"),
  nextFrame: document.querySelector("#next-frame"),
  skipTen: document.querySelector("#skip-ten"),
  frameSlider: document.querySelector("#frame-slider"),
  annotationMarks: document.querySelector("#annotation-marks"),
  framePosition: document.querySelector("#frame-position"),
  frameTime: document.querySelector("#frame-time"),
  episodeFname: document.querySelector("#episode-fname"),
  episodeOutcome: document.querySelector("#episode-outcome"),
  currentFrameTitle: document.querySelector("#current-frame-title"),
  frameState: document.querySelector("#frame-state"),
  annotationForm: document.querySelector("#annotation-form"),
  annotationInput: document.querySelector("#annotation-input"),
  annotationSuggestions: document.querySelector("#annotation-suggestions"),
  clearButton: document.querySelector("#clear-button"),
  quickLabels: document.querySelector("#quick-labels"),
  episodeProgress: document.querySelector("#episode-progress"),
  episodeProgressText: document.querySelector("#episode-progress-text"),
  toast: document.querySelector("#toast"),
};

let state = null;
let framePosition = 0;
let toastTimer = null;
let frameRequest = 0;
let activeFrameUrl = null;
let scrubTimer = null;
let selectedCameraKey = null;
let loadingTimer = null;
let renderChain = Promise.resolve();
let draggedLabel = null;
const combinedCameraKey = "__front_side__";

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return payload;
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.hidden = false;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 2800);
}

function currentFrame() {
  return state.frames[framePosition];
}

function titleCase(value) {
  return value.replace(/\b\w/g, (character) => character.toUpperCase());
}

function populateEpisodeSelect() {
  elements.episodeSelect.replaceChildren();
  for (const episode of state.episodes) {
    const summary = state.episode_summaries[String(episode)];
    const details = [summary.fname, titleCase(summary.outcome)].filter(Boolean).join(" | ");
    const countIcon = summary.annotation_count > 0 ? "\u{1F7E2}" : "\u26AA";
    const count = `${countIcon} ${summary.annotation_count}`;
    const option = document.createElement("option");
    option.value = episode;
    option.textContent = details ? `${count} | ${episode} | ${details}` : `${count} | ${episode}`;
    option.selected = episode === state.episode_index;
    elements.episodeSelect.append(option);
  }
}

function renderEpisodeSummary() {
  const summary = state.episode_summary;
  elements.episodeFname.textContent = summary.fname || "Unavailable";
  elements.episodeFname.title = summary.fname || "fname is not available";
  elements.episodeOutcome.textContent = titleCase(summary.outcome);
  elements.episodeOutcome.className = `outcome-pill ${summary.outcome.replaceAll(" ", "-")}`;
  const rewardDescription =
    summary.outcome_field && summary.outcome_value !== null
      ? `${summary.outcome_field} = ${summary.outcome_value}`
      : "No sparse_reward or reward value is available";
  elements.episodeOutcome.title = rewardDescription;
}

function cameraLabel(videoKey) {
  const name = videoKey.split(".").pop().replace(/^camera_/, "");
  return titleCase(name.replaceAll("_", " "));
}

function cameraModes() {
  const frontKey = state.video_keys.find((key) => key.endsWith("camera_front"));
  const sideKey = state.video_keys.find((key) => key.endsWith("camera_side"));
  const modes = state.video_keys.map((key) => ({ key, label: cameraLabel(key) }));
  if (frontKey && sideKey) {
    modes.unshift({ key: combinedCameraKey, label: "Front + Side", frontKey, sideKey });
  }
  return modes;
}

function renderCameraButtons() {
  const modes = cameraModes();
  if (!modes.some((mode) => mode.key === selectedCameraKey)) {
    selectedCameraKey = modes[0].key;
  }
  elements.cameraButtons.replaceChildren();
  for (const mode of modes) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = mode.key === selectedCameraKey ? "camera-button selected" : "camera-button";
    button.textContent = mode.label;
    button.title = mode.key === combinedCameraKey ? "Square-cropped front and side cameras" : mode.key;
    button.setAttribute("aria-pressed", String(mode.key === selectedCameraKey));
    button.addEventListener("click", () => {
      selectedCameraKey = mode.key;
      renderCameraButtons();
      renderFrame();
    });
    elements.cameraButtons.append(button);
  }
}

function frameImageUrl(frame) {
  const params = new URLSearchParams({
    episode: state.episode_index,
    frame: frame.frame_index,
  });
  if (selectedCameraKey === combinedCameraKey) {
    const combined = cameraModes().find((mode) => mode.key === combinedCameraKey);
    params.set("front_key", combined.frontKey);
    params.set("side_key", combined.sideKey);
    return `/api/combined-frame?${params}`;
  }
  params.set("video_key", selectedCameraKey);
  return `/api/frame?${params}`;
}

function renderQuickLabels() {
  elements.quickLabels.replaceChildren();
  elements.annotationSuggestions.replaceChildren();
  state.labels.forEach((label, index) => {
    const option = document.createElement("option");
    option.value = label;
    elements.annotationSuggestions.append(option);

    const button = document.createElement("button");
    button.type = "button";
    button.className = "quick-label";
    button.draggable = true;
    button.dataset.label = label;
    button.title = `Apply ${label}`;
    button.innerHTML = `<span class="quick-key">${index + 1}</span><span class="quick-text"></span>`;
    button.querySelector(".quick-text").textContent = label;
    button.addEventListener("click", () => applyAnnotation(label));
    button.addEventListener("dragstart", (event) => {
      draggedLabel = label;
      button.classList.add("dragging");
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", label);
    });
    button.addEventListener("dragend", () => {
      draggedLabel = null;
      button.classList.remove("dragging");
      elements.quickLabels.querySelectorAll(".drag-over").forEach((item) => item.classList.remove("drag-over"));
    });
    button.addEventListener("dragover", (event) => {
      if (draggedLabel && draggedLabel !== label) {
        event.preventDefault();
        event.dataTransfer.dropEffect = "move";
        button.classList.add("drag-over");
      }
    });
    button.addEventListener("dragleave", () => button.classList.remove("drag-over"));
    button.addEventListener("drop", (event) => {
      event.preventDefault();
      button.classList.remove("drag-over");
      reorderLabel(draggedLabel, label);
    });
    elements.quickLabels.append(button);
  });
}

async function reorderLabel(source, target) {
  if (!source || source === target) {
    return;
  }
  const labels = [...state.labels];
  const sourceIndex = labels.indexOf(source);
  const targetIndex = labels.indexOf(target);
  if (sourceIndex < 0 || targetIndex < 0) {
    return;
  }
  labels.splice(sourceIndex, 1);
  labels.splice(targetIndex, 0, source);
  state.labels = labels;
  renderQuickLabels();
  try {
    const result = await request("/api/labels/reorder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ labels }),
    });
    state.labels = result.labels;
    renderQuickLabels();
  } catch (error) {
    showToast(error.message);
    const refreshed = await request(`/api/state?episode=${encodeURIComponent(state.episode_index)}`);
    state.labels = refreshed.labels;
    renderQuickLabels();
  }
}

function renderProgress() {
  const progress = state.progress;
  elements.datasetProgress.textContent = `${progress.total_annotated.toLocaleString()} of ${progress.total_frames.toLocaleString()} frames annotated`;
  elements.episodeProgress.max = Math.max(progress.episode_total, 1);
  elements.episodeProgress.value = progress.episode_annotated;
  elements.episodeProgressText.textContent = `${progress.episode_annotated} / ${progress.episode_total}`;
  elements.saveStatus.textContent = state.pending_count
    ? `${state.pending_count} pending ${state.pending_count === 1 ? "change" : "changes"}`
    : "Dataset is up to date";
  elements.commitButton.disabled = state.pending_count === 0;
  elements.discardButton.disabled = state.pending_count === 0;
}

function renderTimelineMarks() {
  elements.annotationMarks.replaceChildren();
  const denominator = Math.max(state.frames.length - 1, 1);
  state.frames.forEach((frame, index) => {
    if (!frame.annotation) {
      return;
    }
    const mark = document.createElement("button");
    mark.type = "button";
    mark.className = frame.pending ? "annotation-mark pending" : "annotation-mark";
    mark.style.left = `${(index / denominator) * 100}%`;
    mark.title = `Frame ${frame.frame_index}: ${frame.annotation}`;
    mark.setAttribute("aria-label", mark.title);
    mark.addEventListener("click", () => setFramePosition(index));
    elements.annotationMarks.append(mark);
  });
}

async function renderFrameNow() {
  const frame = currentFrame();
  const requestId = ++frameRequest;
  elements.frameSlider.value = framePosition;
  elements.framePosition.textContent = `${framePosition + 1} / ${state.frames.length}`;
  elements.frameTime.textContent = `${frame.timestamp.toFixed(3)} s`;
  elements.frameBadge.textContent = `Frame ${frame.frame_index}`;
  elements.currentFrameTitle.textContent = `Frame ${frame.frame_index}`;
  elements.annotationInput.value = frame.annotation;
  elements.clearButton.disabled = !frame.annotation;
  elements.skipTenBack.disabled = framePosition === 0;
  elements.previousFrame.disabled = framePosition === 0;
  elements.nextFrame.disabled = framePosition === state.frames.length - 1;
  elements.skipTen.disabled = framePosition === state.frames.length - 1;

  elements.frameState.className = "state-pill";
  if (frame.pending) {
    elements.frameState.textContent = "Pending";
    elements.frameState.classList.add("pending");
  } else if (frame.annotation) {
    elements.frameState.textContent = "Annotated";
  } else {
    elements.frameState.textContent = "Unannotated";
    elements.frameState.classList.add("empty");
  }

  window.clearTimeout(loadingTimer);
  const hasDisplayedFrame = Boolean(elements.frameImage.getAttribute("src"));
  elements.imageLoading.hidden = hasDisplayedFrame;
  if (hasDisplayedFrame) {
    loadingTimer = window.setTimeout(() => {
      if (requestId === frameRequest) {
        elements.imageLoading.hidden = false;
      }
    }, 150);
  }
  elements.imageError.hidden = true;
  await loadFrameImage(frameImageUrl(frame), requestId);
}

function setFramePosition(position) {
  renderChain = renderChain.then(async () => {
    framePosition = Math.max(0, Math.min(Number(position), state.frames.length - 1));
    await renderFrameNow();
  });
  return renderChain;
}

function moveFrame(offset) {
  renderChain = renderChain.then(async () => {
    const nextPosition = Math.max(0, Math.min(framePosition + offset, state.frames.length - 1));
    if (nextPosition === framePosition) {
      return;
    }
    framePosition = nextPosition;
    await renderFrameNow();
  });
  return renderChain;
}

function renderFrame() {
  renderChain = renderChain.then(() => renderFrameNow());
  return renderChain;
}

async function loadFrameImage(url, requestId) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      let message = `Frame request failed (${response.status})`;
      try {
        const payload = await response.json();
        message = payload.error || message;
      } catch {
        // Keep the HTTP status when the response is not JSON.
      }
      throw new Error(message);
    }

    const blob = await response.blob();
    if (requestId !== frameRequest) {
      return;
    }
    const objectUrl = URL.createObjectURL(blob);
    const previousUrl = activeFrameUrl;
    activeFrameUrl = objectUrl;
    await new Promise((resolve) => {
      elements.frameImage.onload = () => {
        if (requestId === frameRequest) {
          window.clearTimeout(loadingTimer);
          elements.imageLoading.hidden = true;
        }
        if (previousUrl) {
          URL.revokeObjectURL(previousUrl);
        }
        window.requestAnimationFrame(resolve);
      };
      elements.frameImage.onerror = () => {
        if (requestId === frameRequest) {
          window.clearTimeout(loadingTimer);
          elements.imageLoading.hidden = true;
          elements.imageError.textContent = "The browser could not display the decoded JPEG";
          elements.imageError.hidden = false;
        }
        window.requestAnimationFrame(resolve);
      };
      elements.frameImage.src = objectUrl;
    });
  } catch (error) {
    if (requestId === frameRequest) {
      window.clearTimeout(loadingTimer);
      elements.imageLoading.hidden = true;
      elements.imageError.textContent = error.message;
      elements.imageError.hidden = false;
    }
  }
}

async function loadState(episode = null, preferredFrame = null) {
  const query = episode === null ? "" : `?episode=${encodeURIComponent(episode)}`;
  state = await request(`/api/state${query}`);
  elements.datasetName.textContent = state.dataset;
  document.title = `${state.dataset} - Frame Annotations`;
  populateEpisodeSelect();
  renderCameraButtons();
  elements.frameSlider.max = Math.max(state.frames.length - 1, 0);
  const preferredPosition = state.frames.findIndex((frame) => frame.frame_index === preferredFrame);
  framePosition = preferredPosition >= 0 ? preferredPosition : 0;

  const episodePosition = state.episodes.indexOf(state.episode_index);
  elements.previousEpisode.disabled = episodePosition <= 0;
  elements.nextEpisode.disabled = episodePosition >= state.episodes.length - 1;
  renderEpisodeSummary();
  renderQuickLabels();
  renderTimelineMarks();
  renderProgress();
  renderFrame();
}

async function applyAnnotation(annotation) {
  const frame = currentFrame();
  const oldAnnotated = Boolean(frame.annotation);
  try {
    const result = await request("/api/annotations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        episode_index: state.episode_index,
        frame_index: frame.frame_index,
        annotation,
      }),
    });
    frame.annotation = result.annotation;
    frame.pending = true;
    state.pending_count = result.pending_count;
    state.labels = result.labels;
    const delta = Number(Boolean(frame.annotation)) - Number(oldAnnotated);
    state.progress.episode_annotated += delta;
    state.progress.total_annotated += delta;
    state.episode_summary.annotation_count += delta;
    state.episode_summaries[String(state.episode_index)].annotation_count += delta;
    populateEpisodeSelect();
    renderQuickLabels();
    renderTimelineMarks();
    renderProgress();
    renderFrame();
  } catch (error) {
    showToast(error.message);
  }
}

async function commit(showResult = true) {
  if (!state || state.pending_count === 0) {
    return;
  }
  elements.commitButton.disabled = true;
  elements.discardButton.disabled = true;
  elements.commitButton.textContent = "Writing...";
  try {
    const result = await request("/api/commit", { method: "POST" });
    state.pending_count = result.pending_count;
    state.frames.forEach((frame) => {
      frame.pending = false;
    });
    renderTimelineMarks();
    renderProgress();
    renderFrame();
    if (showResult) {
      showToast(`Wrote ${result.committed} annotations across ${result.files} data files`);
    }
  } catch (error) {
    showToast(error.message);
  } finally {
    elements.commitButton.textContent = "Write dataset";
    elements.commitButton.disabled = state.pending_count === 0;
    elements.discardButton.disabled = state.pending_count === 0;
  }
}

async function discardPending() {
  if (!state || state.pending_count === 0) {
    return;
  }
  if (!window.confirm(`Discard ${state.pending_count} pending changes?`)) {
    return;
  }

  const episode = state.episode_index;
  const frame = currentFrame().frame_index;
  elements.discardButton.disabled = true;
  elements.commitButton.disabled = true;
  elements.discardButton.textContent = "Discarding...";
  try {
    const result = await request("/api/discard", { method: "POST" });
    await loadState(episode, frame);
    showToast(`Discarded ${result.discarded} pending changes`);
  } catch (error) {
    showToast(error.message);
  } finally {
    elements.discardButton.textContent = "Discard pending";
    elements.discardButton.disabled = state.pending_count === 0;
    elements.commitButton.disabled = state.pending_count === 0;
  }
}

async function changeEpisode(offset = 0, explicitEpisode = null) {
  const currentIndex = state.episodes.indexOf(state.episode_index);
  const episode = explicitEpisode === null ? state.episodes[currentIndex + offset] : Number(explicitEpisode);
  if (episode !== undefined) {
    try {
      await loadState(episode);
    } catch (error) {
      showToast(error.message);
    }
  }
}

elements.annotationForm.addEventListener("submit", (event) => {
  event.preventDefault();
  applyAnnotation(elements.annotationInput.value);
});
elements.clearButton.addEventListener("click", () => applyAnnotation(""));
elements.commitButton.addEventListener("click", () => commit(true));
elements.discardButton.addEventListener("click", discardPending);
elements.skipTenBack.addEventListener("click", () => moveFrame(-10));
elements.previousFrame.addEventListener("click", () => moveFrame(-1));
elements.nextFrame.addEventListener("click", () => moveFrame(1));
elements.skipTen.addEventListener("click", () => moveFrame(10));
elements.frameSlider.addEventListener("input", (event) => {
  window.clearTimeout(scrubTimer);
  const position = event.target.value;
  scrubTimer = window.setTimeout(() => setFramePosition(position), 100);
});
elements.frameSlider.addEventListener("change", (event) => {
  window.clearTimeout(scrubTimer);
  setFramePosition(event.target.value);
});
elements.previousEpisode.addEventListener("click", () => changeEpisode(-1));
elements.nextEpisode.addEventListener("click", () => changeEpisode(1));
elements.episodeSelect.addEventListener("change", (event) => changeEpisode(0, event.target.value));

document.addEventListener("keydown", (event) => {
  if (!state) {
    return;
  }
  const isTyping = event.target === elements.annotationInput;
  if (!isTyping && event.key === "ArrowLeft") {
    event.preventDefault();
    moveFrame(-1);
  } else if (!isTyping && event.key === "ArrowRight") {
    event.preventDefault();
    moveFrame(1);
  } else if (!isTyping && /^[1-9]$/.test(event.key)) {
    const label = state.labels[Number(event.key) - 1];
    if (label) {
      event.preventDefault();
      applyAnnotation(label);
    }
  }
});

window.addEventListener("beforeunload", (event) => {
  if (state && state.pending_count > 0) {
    event.preventDefault();
  }
});

loadState().catch((error) => {
  elements.imageLoading.textContent = error.message;
  elements.imageLoading.classList.add("error");
  showToast(error.message);
});
