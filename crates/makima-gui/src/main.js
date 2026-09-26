// Makima Desktop UI Client Script

let invoke = null;

// Rilevamento ambiente Tauri o Browser
async function initTauriIpc() {
  try {
    if (window.__TAURI_INTERNALS__) {
      const core = await import("@tauri-apps/api/core");
      invoke = core.invoke;
    }
  } catch (e) {
    console.warn("Esecuzione in modalità browser standalone (mock attivo).", e);
  }
}

// Fallback Mock per test in browser senza backend Tauri avviato
const mockData = {
  status: {
    version: "0.1.0",
    state: "ready",
    total_observations: 15,
    total_outcomes: 2,
    total_forecasts: 2,
    pending_forecasts: 0
  },
  summaries: [
    {
      target: "framework_release",
      observations_count: 8,
      success_count: 6,
      failure_count: 2,
      probability: 0.70,
      uncertainty_variance: 0.0191,
      entropy_bits: 0.88,
      last_timestamp_sec: 1700604800,
      estimated_daily_rate: 1.14
    },
    {
      target: "daily_build",
      observations_count: 3,
      success_count: 3,
      failure_count: 0,
      probability: 0.80,
      uncertainty_variance: 0.0267,
      entropy_bits: 0.72,
      last_timestamp_sec: 1700172800,
      estimated_daily_rate: 1.50
    },
    {
      target: "api_gateway",
      observations_count: 4,
      success_count: 3,
      failure_count: 1,
      probability: 0.667,
      uncertainty_variance: 0.0317,
      entropy_bits: 0.92,
      last_timestamp_sec: 1700259200,
      estimated_daily_rate: 1.33
    }
  ],
  ledger: [
    {
      id: 1,
      target: "framework_release",
      created_at_sec: 1700600000,
      probability: { value: 0.70 },
      window_desc: "7 days",
      evidence_count: 8,
      model_name: "BayesianConjugate",
      status: { Resolved: { actual: true, brier_score: 0.09, calibration_bucket: "0.7-0.8", resolved_at_sec: 1700650000 } }
    },
    {
      id: 2,
      target: "daily_build",
      created_at_sec: 1700150000,
      probability: { value: 0.80 },
      window_desc: "24 hours",
      evidence_count: 3,
      model_name: "BayesianConjugate",
      status: { Resolved: { actual: true, brier_score: 0.04, calibration_bucket: "0.8-0.9", resolved_at_sec: 1700200000 } }
    }
  ]
};

async function callIpc(cmd, args = {}) {
  if (invoke) {
    return await invoke(cmd, args);
  }
  // Mock implementations
  if (cmd === "get_engine_status") return mockData.status;
  if (cmd === "get_all_target_summaries") return mockData.summaries;
  if (cmd === "get_forecast_ledger") return mockData.ledger;
  if (cmd === "get_target_distribution") {
    const sum = mockData.summaries.find(s => s.target === args.target) || mockData.summaries[0];
    const alpha = 1 + sum.success_count;
    const beta = 1 + sum.failure_count;
    const mean = alpha / (alpha + beta);
    const variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1));
    const stdDev = Math.sqrt(variance);
    const points = [];
    for (let i = 0; i <= 100; i++) {
      const x = 0.005 + (i / 100) * 0.99;
      // Approssimazione PDF
      const y = Math.pow(x, alpha - 1) * Math.pow(1 - x, beta - 1) * 5.0;
      points.push({ x, y: Math.min(y, 25) });
    }
    return {
      target: sum.target,
      alpha,
      beta,
      mean,
      variance,
      entropy_bits: sum.entropy_bits,
      ci_lower_95: Math.max(0.01, mean - 1.96 * stdDev),
      ci_upper_95: Math.min(0.99, mean + 1.96 * stdDev),
      curve_points: points
    };
  }
  if (cmd === "add_observation") {
    mockData.status.total_observations++;
    const existing = mockData.summaries.find(s => s.target === args.target);
    if (existing) {
      existing.observations_count++;
      if (args.success) existing.success_count++; else existing.failure_count++;
      existing.probability = (existing.success_count + 1) / (existing.observations_count + 2);
      return existing;
    } else {
      const newSum = {
        target: args.target,
        observations_count: 1,
        success_count: args.success ? 1 : 0,
        failure_count: args.success ? 0 : 1,
        probability: args.success ? 0.667 : 0.333,
        uncertainty_variance: 0.055,
        entropy_bits: 0.95,
        last_timestamp_sec: Date.now() / 1000,
        estimated_daily_rate: 1.0
      };
      mockData.summaries.push(newSum);
      return newSum;
    }
  }
  if (cmd === "generate_laplace_bulletin") {
    return `# BOLLETTINO PREVISIONALE LAPLACE MAIL\n**Data:** ${new Date().toISOString()}\n**Target Monitorati:** ${mockData.summaries.length}\n\n` +
      mockData.summaries.map(s => `- **${s.target}**: P(p)=${(s.probability*100).toFixed(1)}% (N=${s.observations_count}, Varianza=${s.uncertainty_variance.toFixed(4)})`).join("\n");
  }
  if (cmd === "query_chat") {
    const q = args.query.toLowerCase();
    const sum = mockData.summaries.find(s => q.includes(s.target.toLowerCase()));
    if (sum) {
      return {
        response: `Analisi Bayesiana per **${sum.target}**:\n\n• Probabilità a posteriori $P(p)$: **${(sum.probability * 100).toFixed(1)}%**\n• Incertezza (Varianza): **${sum.uncertainty_variance.toFixed(4)}**\n• Entropia: **${sum.entropy_bits.toFixed(2)} bit**\n• Evidenze: ${sum.observations_count} (${sum.success_count} successi, ${sum.failure_count} fallimenti).`,
        confidence: sum.probability,
        target: sum.target,
        timestamp_sec: Date.now() / 1000
      };
    }
    return {
      response: `Ho elaborato la tua richiesta: "${args.query}". Monitoro ${mockData.summaries.length} target previsionali con calibrazione bayesiana continua.`,
      confidence: 0.9,
      target: null,
      timestamp_sec: Date.now() / 1000
    };
  }
  throw new Error(`Comando non gestito: ${cmd}`);
}

// Stato dell'applicazione frontend
let currentTarget = null;
let currentDistribution = null;

// Gestione Navigazione Tabs
function setupNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  const tabPanels = document.querySelectorAll(".tab-panel");
  const titleEl = document.getElementById("page-title");
  const subEl = document.getElementById("page-subtitle");

  const titles = {
    dashboard: { title: "Dashboard Bayesiana", sub: "Visualizzazione delle distribuzioni a posteriori, stima dell'incertezza e tracciamento eventi." },
    observe: { title: "Registra Evidenza", sub: "Inserimento rapido di evidenze binarie nel database relazionale SQLite con Event Log." },
    ledger: { title: "Ledger Previsioni & Brier Score", sub: "Registro immutabile delle previsioni emesse e calcolo dell'errore quadratico sui Ground Truth." },
    chat: { title: "Chat Assistente Makima", sub: "Interroga il motore probabilistico e l'assistente in linguaggio naturale." },
    bulletin: { title: "Laplace Mail", sub: "Bollettino sintetico consolidato per reportistica previsionale e monitoraggio processi." }
  };

  navItems.forEach(btn => {
    btn.addEventListener("click", () => {
      const tab = btn.getAttribute("data-tab");
      navItems.forEach(b => b.classList.remove("active"));
      tabPanels.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPanel = document.getElementById(`tab-${tab}`);
      if (targetPanel) targetPanel.classList.add("active");

      if (titles[tab]) {
        titleEl.textContent = titles[tab].title;
        subEl.textContent = titles[tab].sub;
      }

      if (tab === "bulletin") loadBulletin();
      if (tab === "dashboard" && currentTarget) loadTargetDistribution(currentTarget);
    });
  });
}

// Rendering Curva Beta su Canvas
function renderBetaCurve(dist) {
  const canvas = document.getElementById("beta-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  
  // Imposta risoluzione nitida per schermi HiDPI
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  const padL = 45;
  const padR = 25;
  const padT = 25;
  const padB = 40;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  ctx.clearRect(0, 0, w, h);

  if (!dist || !dist.curve_points || dist.curve_points.length === 0) {
    ctx.fillStyle = "#64748b";
    ctx.font = "14px 'Plus Jakarta Sans'";
    ctx.textAlign = "center";
    ctx.fillText("Nessun dato di distribuzione disponibile", w / 2, h / 2);
    return;
  }

  const maxY = Math.max(...dist.curve_points.map(p => p.y), 1.0) * 1.15;

  const toScreenX = (x) => padL + x * plotW;
  const toScreenY = (y) => padT + (1 - y / maxY) * plotH;

  // 1. Linee guida griglia
  ctx.strokeStyle = "rgba(51, 65, 85, 0.35)";
  ctx.lineWidth = 1;
  for (let xStep = 0; xStep <= 1.0; xStep += 0.2) {
    const sx = toScreenX(xStep);
    ctx.beginPath();
    ctx.moveTo(sx, padT);
    ctx.lineTo(sx, padT + plotH);
    ctx.stroke();

    ctx.fillStyle = "#64748b";
    ctx.font = "11px 'JetBrains Mono'";
    ctx.textAlign = "center";
    ctx.fillText(`${Math.round(xStep * 100)}%`, sx, padT + plotH + 18);
  }

  // 2. Area 95% Credible Interval ombreggiata
  const ciL = dist.ci_lower_95;
  const ciU = dist.ci_upper_95;
  ctx.fillStyle = "rgba(16, 185, 129, 0.12)";
  ctx.fillRect(toScreenX(ciL), padT, toScreenX(ciU) - toScreenX(ciL), plotH);

  ctx.strokeStyle = "rgba(16, 185, 129, 0.4)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(toScreenX(ciL), padT);
  ctx.lineTo(toScreenX(ciL), padT + plotH);
  ctx.moveTo(toScreenX(ciU), padT);
  ctx.lineTo(toScreenX(ciU), padT + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  // 3. Riempimento gradiente sotto la curva
  const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
  grad.addColorStop(0, "rgba(56, 189, 248, 0.4)");
  grad.addColorStop(1, "rgba(56, 189, 248, 0.0)");

  ctx.beginPath();
  ctx.moveTo(toScreenX(dist.curve_points[0].x), toScreenY(0));
  dist.curve_points.forEach(p => {
    ctx.lineTo(toScreenX(p.x), toScreenY(p.y));
  });
  ctx.lineTo(toScreenX(dist.curve_points[dist.curve_points.length - 1].x), toScreenY(0));
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // 4. Linea principale della curva Beta
  ctx.beginPath();
  dist.curve_points.forEach((p, idx) => {
    const sx = toScreenX(p.x);
    const sy = toScreenY(p.y);
    if (idx === 0) ctx.moveTo(sx, sy);
    else ctx.lineTo(sx, sy);
  });
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // 5. Linea verticale per la Media E[p]
  const meanX = toScreenX(dist.mean);
  ctx.strokeStyle = "#a855f7";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(meanX, padT);
  ctx.lineTo(meanX, padT + plotH);
  ctx.stroke();

  // Badge media
  ctx.fillStyle = "#a855f7";
  ctx.font = "bold 11px 'JetBrains Mono'";
  ctx.textAlign = "center";
  ctx.fillText(`E[p] = ${(dist.mean * 100).toFixed(1)}%`, meanX, padT - 8);
}

// Caricamento e rendering stato e target
async function loadDashboardData() {
  try {
    const status = await callIpc("get_engine_status");
    document.getElementById("stat-obs-count").textContent = status.total_observations;
    document.getElementById("stat-outcomes-count").textContent = status.total_outcomes;
    document.getElementById("stat-forecasts-count").textContent = status.total_forecasts;
    document.getElementById("stat-pending-label").textContent = `${status.pending_forecasts} in attesa di risoluzione`;
    document.getElementById("engine-version-label").textContent = `v${status.version} • SQLite WAL`;

    const summaries = await callIpc("get_all_target_summaries");
    document.getElementById("stat-targets-count").textContent = summaries.length;

    // Popola select dei target
    const selector = document.getElementById("target-selector");
    selector.innerHTML = "";
    summaries.forEach((s, idx) => {
      const opt = document.createElement("option");
      opt.value = s.target;
      opt.textContent = `${s.target} (${(s.probability * 100).toFixed(0)}%)`;
      if (idx === 0 && !currentTarget) currentTarget = s.target;
      selector.appendChild(opt);
    });

    if (currentTarget) {
      selector.value = currentTarget;
      await loadTargetDistribution(currentTarget);
    }

    // Popola tabella riassunti
    const tbody = document.getElementById("targets-table-body");
    if (summaries.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" class="table-empty">Nessun target registrato.</td></tr>';
    } else {
      tbody.innerHTML = summaries.map(s => `
        <tr>
          <td><strong>${s.target}</strong></td>
          <td><span class="highlight-cyan">${(s.probability * 100).toFixed(1)}%</span></td>
          <td>${s.observations_count} (${s.success_count} / ${s.failure_count})</td>
          <td>${s.uncertainty_variance.toFixed(4)}</td>
          <td>${s.entropy_bits.toFixed(2)} bit</td>
          <td>${s.estimated_daily_rate.toFixed(2)}/g</td>
        </tr>
      `).join("");
    }

    // Carica Ledger
    await loadLedgerData();
  } catch (err) {
    console.error("Errore durante il caricamento della dashboard:", err);
  }
}

async function loadTargetDistribution(target) {
  try {
    const dist = await callIpc("get_target_distribution", { target });
    currentDistribution = dist;
    renderBetaCurve(dist);

    document.getElementById("metric-mean").textContent = `${(dist.mean * 100).toFixed(1)}%`;
    document.getElementById("metric-params").textContent = `α=${dist.alpha.toFixed(1)}, β=${dist.beta.toFixed(1)}`;
    document.getElementById("metric-variance").textContent = dist.variance.toFixed(4);
    document.getElementById("metric-entropy").textContent = `${dist.entropy_bits.toFixed(2)} bit`;
    document.getElementById("metric-ci").textContent = `[${(dist.ci_lower_95 * 100).toFixed(1)}%, ${(dist.ci_upper_95 * 100).toFixed(1)}%]`;
  } catch (err) {
    console.error(`Errore nel caricamento della distribuzione per ${target}:`, err);
  }
}

async function loadLedgerData() {
  try {
    const ledger = await callIpc("get_forecast_ledger");
    const tbody = document.getElementById("ledger-table-body");
    if (!ledger || ledger.length === 0) {
      tbody.innerHTML = '<tr><td colspan="9" class="table-empty">Nessuna previsione registrata nel ledger.</td></tr>';
      return;
    }

    tbody.innerHTML = ledger.map(rec => {
      const isResolved = rec.status && rec.status.Resolved;
      const statusBadge = isResolved
        ? '<span class="badge resolved">Risolto</span>'
        : '<span class="badge pending">In Attesa</span>';
      const actual = isResolved ? (rec.status.Resolved.actual ? "Successo (1)" : "Fallimento (0)") : "—";
      const brier = isResolved ? rec.status.Resolved.brier_score.toFixed(4) : "—";

      const recId = typeof rec.id === 'object' && rec.id !== null ? (rec.id[0] ?? rec.id.toString()) : rec.id;
      return `
        <tr>
          <td>#${recId}</td>
          <td><strong>${rec.target}</strong></td>
          <td>${(rec.probability.value * 100).toFixed(1)}%</td>
          <td>${rec.window_desc}</td>
          <td>${rec.evidence_count}</td>
          <td>${rec.model_name}</td>
          <td>${statusBadge}</td>
          <td>${actual}</td>
          <td>${brier}</td>
        </tr>
      `;
    }).join("");
  } catch (err) {
    console.error("Errore nel caricamento del ledger:", err);
  }
}

async function loadBulletin() {
  const el = document.getElementById("bulletin-content");
  try {
    const text = await callIpc("generate_laplace_bulletin");
    el.textContent = text;
  } catch (e) {
    el.textContent = "Impossibile generare il bollettino: " + e;
  }
}

// Setup Form Nuova Evidenza
function setupObserveForm() {
  const form = document.getElementById("observe-form");
  const alert = document.getElementById("observe-alert");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const target = document.getElementById("obs-target-input").value.trim();
    const success = form.elements["obs-outcome"].value === "true";
    const notes = document.getElementById("obs-notes-input").value.trim() || null;

    if (!target) return;

    try {
      const summary = await callIpc("add_observation", { target, success, notes });
      alert.textContent = `Evidenza registrata con successo per "${target}". Nuova probabilità $P(p)$: ${(summary.probability * 100).toFixed(1)}%`;
      alert.classList.remove("hidden");
      setTimeout(() => alert.classList.add("hidden"), 5000);

      form.reset();
      currentTarget = target;
      await loadDashboardData();
    } catch (err) {
      window.alert("Errore registrazione: " + err);
    }
  });
}

// Setup Chat Assistente
function setupChatForm() {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const messagesBox = document.getElementById("chat-messages");

  function appendMessage(sender, text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `chat-msg ${sender}`;
    const avatarText = sender === "assistant" ? "M" : "Tu";
    
    // Formattazione base markdown (grassetto, corsivo, elenchi)
    const formatted = text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\n/g, "<br/>");

    msgDiv.innerHTML = `
      <div class="msg-avatar">${avatarText}</div>
      <div class="msg-content"><p>${formatted}</p></div>
    `;
    messagesBox.appendChild(msgDiv);
    messagesBox.scrollTop = messagesBox.scrollHeight;
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    appendMessage("user", text);
    input.value = "";

    try {
      const res = await callIpc("query_chat", { query: text });
      appendMessage("assistant", res.response);
      if (res.target) {
        currentTarget = res.target;
      }
    } catch (err) {
      appendMessage("assistant", "Si è verificato un errore durante l'elaborazione della risposta: " + err);
    }
  });
}

// Inizializzazione Principale
window.addEventListener("DOMContentLoaded", async () => {
  await initTauriIpc();
  setupNavigation();
  setupObserveForm();
  setupChatForm();

  document.getElementById("target-selector").addEventListener("change", (e) => {
    currentTarget = e.target.value;
    loadTargetDistribution(currentTarget);
  });

  document.getElementById("refresh-btn").addEventListener("click", () => {
    loadDashboardData();
  });

  document.getElementById("copy-bulletin-btn").addEventListener("click", () => {
    const content = document.getElementById("bulletin-content").textContent;
    navigator.clipboard.writeText(content).then(() => {
      window.alert("Bollettino copiato negli appunti!");
    });
  });

  window.addEventListener("resize", () => {
    if (currentDistribution) renderBetaCurve(currentDistribution);
  });

  await loadDashboardData();
});
