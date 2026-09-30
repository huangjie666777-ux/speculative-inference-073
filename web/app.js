"use strict";

const form = document.getElementById("generate-form");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const output = document.getElementById("output");
const errorBox = document.getElementById("error");

let activeController = null;

function setStats(stats) {
  document.getElementById("stat-candidates").textContent = stats.candidates ?? 0;
  document.getElementById("stat-accepted").textContent = stats.accepted ?? 0;
  document.getElementById("stat-forwards").textContent = stats.target_forwards ?? 0;
  document.getElementById("stat-elapsed").textContent = stats.elapsed_seconds ?? "-";
}

function setRunning(running, statusText) {
  startBtn.disabled = running;
  stopBtn.disabled = !running;
  document.getElementById("stat-status").textContent = statusText;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideError();
  output.textContent = "";
  setStats({});

  const body = {
    text: document.getElementById("text").value,
    mode: document.getElementById("mode").value,
    max_new_tokens: Number(document.getElementById("max_new_tokens").value),
    draft_steps: Number(document.getElementById("draft_steps").value),
    temperature: Number(document.getElementById("temperature").value),
    seed: Number(document.getElementById("seed").value),
  };

  activeController = new AbortController();
  setRunning(true, "生成中…");
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: activeController.signal,
    });
    if (response.status === 409) {
      showError("服务器忙：已有一个活动生成请求，请稍后再试。");
      setRunning(false, "忙");
      return;
    }
    if (!response.ok || !response.body) {
      const detail = await response.text();
      throw new Error(`HTTP ${response.status}: ${detail}`);
    }

    const reader = response.body.getReader();
    const textDecoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += textDecoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        const dataLine = frame.split("\n").find((line) => line.startsWith("data: "));
        const eventLine = frame.split("\n").find((line) => line.startsWith("event: "));
        if (!dataLine) continue;
        const name = eventLine ? eventLine.slice(7) : "message";
        const data = JSON.parse(dataLine.slice(6));
        if (name === "token") {
          output.textContent += data.text;
          output.scrollTop = output.scrollHeight;
        } else if (name === "done") {
          setStats(data);
          setRunning(false, data.stopped ? "已停止" : "完成");
        } else if (name === "error") {
          showError(data.message);
          setRunning(false, "错误");
        }
      }
    }
  } catch (err) {
    if (err.name === "AbortError") {
      setRunning(false, "已停止");
    } else {
      showError(err.message);
      setRunning(false, "错误");
    }
  } finally {
    activeController = null;
  }
});

stopBtn.addEventListener("click", () => {
  if (activeController) activeController.abort();
});
