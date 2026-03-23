const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("upload-form");
const fileInput = document.getElementById("file");

function appendMessage({ type, html }) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${type}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = type === "user" ? "Bạn" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = html;

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

formEl.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    alert("Vui lòng chọn một ảnh lá trước.");
    return;
  }

  const userHtml = `
    <p><strong>Ảnh:</strong> ${file.name}</p>
  `;
  appendMessage({ type: "user", html: userHtml });

  const reader = new FileReader();
  reader.onload = () => {
    const imgHtml = `<img src="${reader.result}" class="image-preview" alt="preview" />`;
    appendMessage({ type: "user", html: imgHtml });
  };
  reader.readAsDataURL(file);

  const btn = formEl.querySelector(".send-btn");
  btn.disabled = true;
  btn.textContent = "Đang phân tích...";

  try {
    const formData = new FormData();
    formData.append("file", file);

    const resp = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || "Lỗi khi gửi ảnh lên server");
    }

    const data = await resp.json();
    const r = data.result;

    const probs = Object.entries(r.raw_probs)
      .map(
        ([k, v]) =>
          `<span>${k.replaceAll("_", " ")}: ${(v * 100).toFixed(1)}%</span>`,
      )
      .join("");

    const botHtml = `
      <p><strong>Kết quả:</strong> ${r.label_vietnamese}</p>
      <p><strong>Độ tin cậy:</strong> ${(r.probability * 100).toFixed(1)}%</p>
      <p>${r.explanation}</p>
      <div class="probs">${probs}</div>
    `;

    appendMessage({ type: "bot", html: botHtml });
  } catch (err) {
    appendMessage({
      type: "bot",
      html: `<p><strong>Lỗi:</strong> ${(err && err.message) || "Không xác định"}</p>`,
    });
  } finally {
    btn.disabled = false;
    btn.textContent = "Gửi";
    formEl.reset();
  }
});

