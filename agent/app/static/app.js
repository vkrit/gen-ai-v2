const state = {
  sessionId: crypto.randomUUID(),
  patients: [],
};

const patientSelect = document.querySelector("#patient-select");
const nurseInput = document.querySelector("#nurse-id");
const sessionIdLabel = document.querySelector("#session-id");
const messages = document.querySelector("#messages");
const trace = document.querySelector("#trace");
const auditLog = document.querySelector("#audit-log");
const approvalList = document.querySelector("#approval-list");
const approvalCount = document.querySelector("#approval-count");
const messageInput = document.querySelector("#message-input");

function setSessionLabel() {
  sessionIdLabel.textContent = state.sessionId;
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function appendMessage(role, content) {
  const template = document.querySelector("#message-template");
  const node = template.content.firstElementChild.cloneNode(true);
  node.classList.add(role);
  node.querySelector(".role").textContent = role === "user" ? "Nurse" : "Agent";
  node.querySelector(".content").textContent = content;
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
}

function renderTrace(items) {
  trace.innerHTML = "";
  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "trace-item";
    card.innerHTML = `<strong>${escapeHtml(item.title)}</strong><p>${escapeHtml(item.detail)}</p>`;
    trace.appendChild(card);
  });
}

function renderAuditLog(items) {
  auditLog.innerHTML = "";
  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "audit-item";
    card.innerHTML = `
      <strong>${escapeHtml(item.action)} · ${escapeHtml(item.approval_status)}</strong>
      <p>${escapeHtml(item.created_at)} · ${escapeHtml(item.patient_hash)}</p>
      <p>${escapeHtml(JSON.stringify(item.detail))}</p>
    `;
    auditLog.appendChild(card);
  });
}

async function decideApproval(approvalId, decision) {
  const response = await fetch(`/api/approvals/${approvalId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      nurse_id: nurseInput.value || "nurse-demo",
      decision,
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    window.alert(data.detail || "Approval update failed.");
    return;
  }
  await refreshApprovals();
  renderAuditLog(data.audit_log);
}

function renderApprovals(items) {
  approvalCount.textContent = String(items.length);
  approvalList.innerHTML = "";
  if (!items.length) {
    approvalList.innerHTML = "<p class='muted'>No pending write actions.</p>";
    return;
  }

  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "approval-item";
    card.innerHTML = `
      <strong>${escapeHtml(item.tool_name)}</strong>
      <p>${escapeHtml(item.reason)}</p>
      <p>${escapeHtml(item.patient_id)} · ${escapeHtml(item.approval_id)}</p>
      <div class="approval-actions">
        <button data-action="approved">Approve</button>
        <button class="reject" data-action="rejected">Reject</button>
      </div>
    `;
    card.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () => decideApproval(item.approval_id, button.dataset.action));
    });
    approvalList.appendChild(card);
  });
}

async function refreshApprovals() {
  const response = await fetch("/api/approvals");
  const approvals = await response.json();
  renderApprovals(approvals);
}

async function bootstrap() {
  const response = await fetch("/api/bootstrap");
  const data = await response.json();
  state.patients = data.patients;

  patientSelect.innerHTML = data.patients
    .map((patient) => `<option value="${patient.patient_id}">${patient.patient_id} · ${patient.name}</option>`)
    .join("");

  renderApprovals(data.approvals);
  renderAuditLog(data.audit_log);
  setSessionLabel();
}

document.querySelector("#new-session").addEventListener("click", () => {
  state.sessionId = crypto.randomUUID();
  messages.innerHTML = "";
  trace.innerHTML = "";
  setSessionLabel();
});

document.querySelector("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) {
    return;
  }

  appendMessage("user", message);
  messageInput.value = "";

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: state.sessionId,
      nurse_id: nurseInput.value || "nurse-demo",
      patient_id: patientSelect.value,
      message,
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    window.alert(data.detail || "Chat request failed.");
    return;
  }

  appendMessage("assistant", data.reply);
  renderTrace(data.trace);
  renderApprovals(data.approvals);
  renderAuditLog(data.audit_log);
});

bootstrap();
