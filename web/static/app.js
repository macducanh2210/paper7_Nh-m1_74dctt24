const byId = (id) => document.getElementById(id);
const fields = ["fx", "fy", "cx", "cy", "baseline"];
const resultNames = {
  "01_disparity.png": ["Disparity", "Chênh lệch vị trí điểm ảnh trái–phải"],
  "02_depth.png": ["Bản đồ độ sâu", "Quy đổi disparity sang khoảng cách (m)"],
  "03_pointcloud.png": ["Point cloud 3D", "Hai góc nhìn của các điểm được tái tạo"],
  "04_bev.png": ["Bird’s-Eye View", "Mật độ điểm nhìn từ trên xuống"],
  "05_size_vs_depth.png": ["Kích thước theo khoảng cách", "Bề rộng xe trên ảnh và ngoài thực tế"],
  "06_error_vs_depth.png": ["Sai số độ sâu", "Sai số tăng nhanh theo bình phương khoảng cách"],
};
let selectedCalibText = "";
let currentPoints = [];
let yaw = 0.35;
let pitch = -0.12;
let zoom = 1;
let dragging = false;
let previousPointer = null;
let animationFrame = 0;

function showError(message) {
  const box = byId("errorBox");
  box.textContent = message;
  box.hidden = false;
  box.scrollIntoView({ behavior: "smooth", block: "center" });
}

function hideError() {
  byId("errorBox").hidden = true;
}

function setStatus(text, state = "") {
  const badge = byId("resultStatus");
  badge.className = `result-status ${state}`;
  badge.innerHTML = `<span class="status-dot"></span> ${text}`;
}

function updatePreview(side, file) {
  const name = byId(`${side}Name`);
  const card = byId(`${side}File`).closest(".upload-card");
  if (!file) {
    name.textContent = "Chọn tệp ảnh";
    card.classList.remove("has-file");
    return;
  }
  name.textContent = file.name;
  card.classList.add("has-file");
  const image = byId(`${side}Preview`);
  image.src = URL.createObjectURL(file);
  byId("previewPair").hidden = false;
}

function setCalibration(values, message, isValid) {
  for (const key of fields) {
    if (values && Number.isFinite(Number(values[key]))) {
      byId(key).value = Number(values[key]).toFixed(key === "baseline" ? 5 : 3);
    }
  }
  const notice = byId("calibNotice");
  notice.classList.toggle("ok", isValid);
  notice.innerHTML = `<span>${isValid ? "✓" : "i"}</span><p>${message}</p>`;
}

function readCalibration(text) {
  const matrices = {};
  for (const line of text.split(/\r?\n/)) {
    const split = line.indexOf(":");
    if (split < 0) continue;
    const key = line.slice(0, split).trim();
    if (!["P_rect_02", "P_rect_03", "P2", "P3"].includes(key)) continue;
    const values = line.slice(split + 1).trim().split(/\s+/).map(Number);
    if (values.length !== 12 || values.some((value) => !Number.isFinite(value))) {
      throw new Error(`Dòng ${key} cần đúng 12 giá trị số.`);
    }
    matrices[key] = values;
  }
  let left;
  let right;
  if (matrices.P_rect_02 && matrices.P_rect_03) {
    left = matrices.P_rect_02;
    right = matrices.P_rect_03;
  } else if (matrices.P2 && matrices.P3) {
    left = matrices.P2;
    right = matrices.P3;
  } else {
    throw new Error("Không thấy đủ P_rect_02/P_rect_03 hoặc P2/P3.");
  }
  return {
    fx: left[0],
    fy: left[5],
    cx: left[2],
    cy: left[6],
    baseline: (left[3] - right[3]) / left[0],
  };
}

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    const config = await response.json();
    if (config.sample_intrinsics_hint) {
      ["fx", "fy", "cx", "cy"].forEach((key, index) => {
        byId(key).value = Number(config.sample_intrinsics_hint[index]).toFixed(3);
      });
    }
    if (config.sample_calibration) {
      setCalibration(config.sample_calibration, "Đã đọc calibration bộ ảnh mẫu. Bạn có thể chỉnh sửa thông số bên dưới.", true);
      byId("baseline").value = Number(config.sample_calibration.baseline).toFixed(5);
    } else if (config.calibration_warning) {
      setCalibration(null, `Calibration mẫu chưa đủ ma trận stereo. Đang dùng fx/fy/cx/cy tham khảo từ P_rect_00 và baseline KITTI tham khảo 0,54 m; hãy kiểm tra hoặc thay bằng thông số đúng của camera.`, false);
    }
    if (!config.samples_available) {
      byId("sampleButton").disabled = true;
      byId("sampleButton").title = "Không tìm thấy đủ ảnh mẫu trong data/";
    }
  } catch (error) {
    setCalibration(null, "Không lấy được thông tin mẫu. Vẫn có thể tải cặp ảnh và nhập calibration thủ công.", false);
  }
}

function markPipeline(activeIndex = -1, complete = false) {
  document.querySelectorAll(".pipeline-step").forEach((step, index) => {
    step.classList.toggle("active", index === activeIndex);
    step.classList.toggle("done", complete || index < activeIndex);
  });
}

function createGallery(files) {
  const gallery = byId("gallery");
  gallery.replaceChildren();
  for (const [filename, [title, description]] of Object.entries(resultNames)) {
    const article = document.createElement("article");
    article.className = "gallery-item";
    const image = document.createElement("img");
    image.src = `${files[filename]}?v=${Date.now()}`;
    image.alt = title;
    image.loading = "lazy";
    image.addEventListener("click", () => {
      byId("lightboxImage").src = image.src;
      byId("lightboxCaption").textContent = title;
      byId("lightbox").hidden = false;
    });
    const caption = document.createElement("div");
    caption.className = "gallery-caption";
    caption.innerHTML = `<div><b>${title}</b><small>${description}</small></div><span>PNG ↗</span>`;
    article.append(image, caption);
    gallery.append(article);
  }
}

function setMetrics(metrics) {
  byId("metricPoints").textContent = Number(metrics.point_count).toLocaleString("vi-VN");
  byId("metricValid").innerHTML = `${metrics.valid_percent}<small>%</small>`;
  byId("metricDepth").innerHTML = `${metrics.depth_median ?? "—"}<small>m</small>`;
  byId("metricTime").innerHTML = `${metrics.elapsed}<small>s</small>`;
  byId("validBar").style.width = `${Math.min(100, metrics.valid_percent)}%`;
  byId("calculationStrip").innerHTML = [
    `<span class="calc-chip">ẢNH <b>${metrics.width} × ${metrics.height} px</b></span>`,
    `<span class="calc-chip">fx <b>${metrics.fx} px</b></span>`,
    `<span class="calc-chip">baseline <b>${metrics.baseline} m</b></span>`,
    `<span class="calc-chip">z hợp lệ <b>${metrics.depth_min ?? "—"} – ${metrics.depth_max ?? "—"} m</b></span>`,
  ].join("");
}

function setTimings(steps) {
  const target = byId("timingList");
  target.replaceChildren();
  const maxTime = Math.max(0.001, ...Object.values(steps));
  for (const [name, seconds] of Object.entries(steps)) {
    const row = document.createElement("div");
    row.className = "timing-row";
    const label = document.createElement("span");
    label.textContent = name;
    const track = document.createElement("span");
    track.className = "timing-track";
    const bar = document.createElement("i");
    bar.style.width = `${Math.max(2, seconds / maxTime * 100)}%`;
    track.append(bar);
    const value = document.createElement("span");
    value.className = "timing-value";
    value.textContent = `${seconds.toFixed(3)} s`;
    row.append(label, track, value);
    target.append(row);
  }
}

function showResults(data) {
  byId("emptyState").hidden = true;
  byId("results").hidden = false;
  setMetrics(data.metrics);
  setTimings(data.metrics.steps);
  createGallery(data.files);
  byId("plyDownload").href = data.files["point_cloud.ply"];
  byId("cloudCount").textContent = `${data.points.length.toLocaleString("vi-VN")} ĐIỂM HIỂN THỊ`;
  currentPoints = data.points;
  byId("runWarning").hidden = !data.warning;
  if (data.warning) byId("runWarning").textContent = data.warning;
  drawCloud();
  setStatus("HOÀN TẤT", "complete");
  markPipeline(-1, true);
}

async function runDemo(useSample) {
  hideError();
  const left = byId("leftFile").files[0];
  const right = byId("rightFile").files[0];
  if (!useSample && (!left || !right)) {
    showError("Hãy chọn đủ ảnh trái và ảnh phải, hoặc nhấn “Dùng bộ ảnh mẫu”.");
    return;
  }
  const camera = {};
  for (const key of fields) {
    const value = Number(byId(key).value);
    if (!Number.isFinite(value) || (["fx", "fy", "baseline"].includes(key) && value <= 0)) {
      showError(`Giá trị ${key} không hợp lệ. Kiểm tra lại thông số camera.`);
      byId(key).focus();
      return;
    }
    camera[key] = value;
  }

  const form = new FormData();
  form.set("use_sample", String(useSample));
  form.set("scale", byId("scale").value);
  form.set("max_depth", byId("maxDepth").value);
  for (const [key, value] of Object.entries(camera)) form.set(key, String(value));
  if (!useSample) {
    form.set("left", left);
    form.set("right", right);
  }
  const calibration = byId("calibFile").files[0];
  if (calibration) form.set("calib", calibration);
  else if (selectedCalibText) form.set("calib_text", selectedCalibText);

  byId("runButton").disabled = true;
  byId("runButton").querySelector(".run-label").textContent = "ĐANG XỬ LÝ…";
  byId("resultStatus").className = "result-status running";
  setStatus("ĐANG CHẠY CPU", "running");
  byId("emptyState").hidden = false;
  byId("results").hidden = true;
  byId("emptyState").querySelector(".empty-kicker").textContent = "ĐANG TÍNH TOÁN TRÊN CPU";
  byId("emptyState").querySelector("h3").innerHTML = "Đang tái tạo<br><span>không gian 3D…</span>";
  byId("emptyState").querySelector(":scope > p").textContent = "StereoSGBM đang tìm tương ứng giữa hai ảnh. Tốc độ phụ thuộc vào kích thước ảnh và máy tính.";
  byId("emptyState").querySelector(".empty-steps").innerHTML = '<span>01 <b>Đọc ảnh</b></span><i>→</i><span>02 <b>Tính disparity</b></span><i>→</i><span>03 <b>Tạo 3D</b></span>';
  markPipeline(1);
  setTimeout(() => markPipeline(3), 120);

  try {
    const response = await fetch("/api/run", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Xử lý không thành công.");
    showResults(data);
  } catch (error) {
    setStatus("CẦN KIỂM TRA", "");
    markPipeline(-1);
    showError(error.message || "Không kết nối được máy chủ demo.");
    byId("emptyState").querySelector(".empty-kicker").textContent = "CHƯA CÓ KẾT QUẢ";
    byId("emptyState").querySelector("h3").innerHTML = 'Một cặp ảnh.<br><span>Một thế giới có chiều sâu.</span>';
    byId("emptyState").querySelector(":scope > p").innerHTML = 'Chọn bộ ảnh mẫu hoặc tải cặp ảnh stereo của bạn lên, sau đó nhấn <b>Chạy demo</b> để bắt đầu.';
    byId("emptyState").querySelector(".empty-steps").innerHTML = '<span>01 <b>Ảnh trái + phải</b></span><i>→</i><span>02 <b>Disparity</b></span><i>→</i><span>03 <b>Không gian 3D</b></span>';
  } finally {
    byId("runButton").disabled = false;
    byId("runButton").querySelector(".run-label").textContent = "CHẠY DEMO";
  }
}

function drawCloud() {
  const canvas = byId("cloudCanvas");
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#090f15";
  ctx.fillRect(0, 0, rect.width, rect.height);
  const gradient = ctx.createRadialGradient(rect.width * 0.5, rect.height * 0.5, 5, rect.width * 0.5, rect.height * 0.5, rect.width * 0.6);
  gradient.addColorStop(0, "rgba(40,78,62,.23)");
  gradient.addColorStop(1, "rgba(9,15,21,0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, rect.width, rect.height);

  const cosYaw = Math.cos(yaw);
  const sinYaw = Math.sin(yaw);
  const cosPitch = Math.cos(pitch);
  const sinPitch = Math.sin(pitch);
  const focal = Math.min(rect.width, rect.height) * 1.15 * zoom;
  const centerX = rect.width * 0.5;
  const centerY = rect.height * 0.52;
  const rendered = [];
  let maximumDepth = 1;
  for (const point of currentPoints) {
    const x = point[0], y = point[1], z = point[2];
    const rx = x * cosYaw - z * sinYaw;
    const rz = x * sinYaw + z * cosYaw;
    const ry = y * cosPitch - rz * sinPitch;
    const depth = y * sinPitch + rz * cosPitch + 4;
    if (depth <= 0.05) continue;
    const scale = focal / depth;
    const px = centerX + rx * scale;
    const py = centerY + ry * scale;
    if (px < -2 || px > rect.width + 2 || py < -2 || py > rect.height + 2) continue;
    maximumDepth = Math.max(maximumDepth, depth);
    rendered.push({ px, py, depth, color: point });
  }
  rendered.sort((a, b) => b.depth - a.depth);
  for (const p of rendered) {
    const ratio = Math.min(1, p.depth / maximumDepth);
    const red = Math.round(255 * (1 - ratio) + 70 * ratio);
    const green = Math.round(142 * (1 - ratio) + 180 * ratio);
    const blue = Math.round(104 * (1 - ratio) + 232 * ratio);
    const radius = Math.max(1, Math.min(2.5, focal / p.depth * 0.013));
    ctx.fillStyle = `rgba(${red},${green},${blue},${Math.max(0.28, 1 - ratio * .54)})`;
    ctx.fillRect(p.px, p.py, radius, radius);
  }
  ctx.strokeStyle = "rgba(129,185,154,.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(centerX, 13);
  ctx.lineTo(centerX, rect.height - 12);
  ctx.moveTo(12, centerY);
  ctx.lineTo(rect.width - 12, centerY);
  ctx.stroke();
}

byId("leftFile").addEventListener("change", (event) => updatePreview("left", event.target.files[0]));
byId("rightFile").addEventListener("change", (event) => updatePreview("right", event.target.files[0]));
byId("calibFile").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  byId("calibName").textContent = file.name;
  try {
    selectedCalibText = await file.text();
    const values = readCalibration(selectedCalibText);
    setCalibration(values, "Đã nhận diện ma trận chiếu stereo. Các tham số được điền tự động và có thể chỉnh sửa.", true);
  } catch (error) {
    setCalibration(null, `${error.message} Có thể nhập đủ thông số camera bằng tay bên dưới.`, false);
  }
});
byId("sampleButton").addEventListener("click", () => {
  byId("leftFile").value = "";
  byId("rightFile").value = "";
  updatePreview("left", null);
  updatePreview("right", null);
  byId("previewPair").hidden = true;
  byId("inputState").textContent = "ẢNH MẪU";
  byId("inputState").classList.add("ready");
  runDemo(true);
});
byId("runButton").addEventListener("click", () => {
  byId("inputState").textContent = "ẢNH ĐÃ CHỌN";
  byId("inputState").classList.add("ready");
  runDemo(false);
});
byId("scale").addEventListener("input", (event) => {
  byId("scaleValue").textContent = `${Number(event.target.value).toFixed(2)}×`;
});
byId("maxDepth").addEventListener("input", (event) => {
  byId("depthValue").textContent = `${event.target.value} m`;
});
byId("closeLightbox").addEventListener("click", () => { byId("lightbox").hidden = true; });
byId("lightbox").addEventListener("click", (event) => {
  if (event.target === byId("lightbox")) byId("lightbox").hidden = true;
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") byId("lightbox").hidden = true;
});
byId("cloudCanvas").addEventListener("pointerdown", (event) => {
  dragging = true;
  previousPointer = [event.clientX, event.clientY];
  byId("cloudCanvas").setPointerCapture(event.pointerId);
});
byId("cloudCanvas").addEventListener("pointermove", (event) => {
  if (!dragging || !previousPointer) return;
  yaw += (event.clientX - previousPointer[0]) * 0.008;
  pitch = Math.max(-1.15, Math.min(1.15, pitch + (event.clientY - previousPointer[1]) * 0.008));
  previousPointer = [event.clientX, event.clientY];
  cancelAnimationFrame(animationFrame);
  animationFrame = requestAnimationFrame(drawCloud);
});
byId("cloudCanvas").addEventListener("pointerup", () => { dragging = false; previousPointer = null; });
byId("cloudCanvas").addEventListener("pointercancel", () => { dragging = false; previousPointer = null; });
byId("cloudCanvas").addEventListener("wheel", (event) => {
  event.preventDefault();
  zoom = Math.max(0.5, Math.min(2.5, zoom * (event.deltaY < 0 ? 1.1 : 0.9)));
  drawCloud();
}, { passive: false });
window.addEventListener("resize", () => requestAnimationFrame(drawCloud));

loadConfig();
