// Makima Desktop UI Client Script — 100% Real IPC Backend Execution
import { invoke } from "@tauri-apps/api/core";

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
    brain: { title: "Second Brain & Grafo Neurale", sub: "Mente associativa di Makima: connessioni semantiche, memorie autobiografiche e target stocastici." },
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
      if (tab === "ledger") loadLedgerData();
      if (tab === "brain" && secondBrain) secondBrain.loadAndRender();
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

// Caricamento e rendering stato e target dal DB reale
async function loadDashboardData() {
  try {
    const status = await invoke("get_engine_status");
    document.getElementById("stat-obs-count").textContent = status.total_observations;
    document.getElementById("stat-outcomes-count").textContent = status.total_outcomes;
    document.getElementById("stat-forecasts-count").textContent = status.total_forecasts;
    document.getElementById("stat-pending-label").textContent = `${status.pending_forecasts} in attesa di risoluzione`;
    document.getElementById("engine-version-label").textContent = `v${status.version} • SQLite WAL`;

    const summaries = await invoke("get_all_target_summaries");
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
      tbody.innerHTML = '<tr><td colspan="6" class="table-empty">Nessun target registrato nel database.</td></tr>';
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
    console.error("Errore durante il caricamento dei dati dal backend:", err);
  }
}

async function loadTargetDistribution(target) {
  try {
    const dist = await invoke("get_target_distribution", { target });
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
    const ledger = await invoke("get_forecast_ledger");
    const tbody = document.getElementById("ledger-table-body");
    if (!ledger || ledger.length === 0) {
      tbody.innerHTML = '<tr><td colspan="10" class="table-empty">Nessuna previsione registrata nel ledger.</td></tr>';
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
      const actionCol = isResolved
        ? '<span style="color: #64748b; font-size: 0.75rem;">Chiuso</span>'
        : `<button class="btn btn-secondary btn-sm" onclick="window.resolveTargetForecast('${rec.target}', true)" style="padding: 2px 6px; font-size: 0.75rem; margin-right: 4px;">✓ Succ</button><button class="btn btn-secondary btn-sm" onclick="window.resolveTargetForecast('${rec.target}', false)" style="padding: 2px 6px; font-size: 0.75rem;">✗ Fall</button>`;

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
          <td>${actionCol}</td>
        </tr>
      `;
    }).join("");
  } catch (err) {
    console.error("Errore nel caricamento del ledger:", err);
  }
}

// Funzione globale per risolvere una previsione con Ground Truth reale
window.resolveTargetForecast = async (target, occurred) => {
  if (window.confirm(`Vuoi registrare l'esito reale "${occurred ? 'SUCCESSO' : 'FALLIMENTO'}" per il target "${target}"?`)) {
    try {
      await invoke("resolve_forecast", { target, occurred });
      await loadDashboardData();
    } catch (e) {
      window.alert("Errore risoluzione: " + e);
    }
  }
};

async function loadBulletin() {
  const el = document.getElementById("bulletin-content");
  try {
    const text = await invoke("generate_laplace_bulletin");
    el.textContent = text;
  } catch (e) {
    el.textContent = "Impossibile generare il bollettino: " + e;
  }
}

// Setup Form Nuova Evidenza (scrive direttamente su SQLite WAL)
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
      const summary = await invoke("add_observation", { target, success, notes });
      alert.textContent = `Evidenza reale registrata per "${target}". Nuova probabilità $P(p)$: ${(summary.probability * 100).toFixed(1)}%`;
      alert.classList.remove("hidden");
      setTimeout(() => alert.classList.add("hidden"), 5000);

      form.reset();
      currentTarget = target;
      await loadDashboardData();
    } catch (err) {
      window.alert("Errore registrazione SQLite: " + err);
    }
  });
}

// Setup Chat Assistente
function setupChatForm() {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("chat-input");
  const messagesBox = document.getElementById("chat-messages");

  function appendMessage(sender, text, thoughtTrace = null) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `chat-msg ${sender}`;
    const avatarText = sender === "assistant" ? "M" : "Tu";
    
    // Formattazione base markdown
    const formatted = text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\n/g, "<br/>");

    let thoughtHtml = "";
    if (thoughtTrace) {
      const formattedThought = thoughtTrace.replace(/\n/g, "<br/>");
      thoughtHtml = `
        <details class="thought-box" open>
          <summary>🧠 <em>Flusso di Coscienza & Monologo Interiore</em></summary>
          <div class="thought-content">${formattedThought}</div>
        </details>
      `;
    }

    msgDiv.innerHTML = `
      <div class="msg-avatar">${avatarText}</div>
      <div class="msg-content">
        ${thoughtHtml}
        <p>${formatted}</p>
      </div>
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
      const res = await invoke("query_chat", { query: text });
      appendMessage("assistant", res.response, res.thought_trace);
      if (res.target) {
        currentTarget = res.target;
      }
    } catch (err) {
      appendMessage("assistant", "Errore elaborazione query: " + err);
    }
  });
}

// ============================================================================
// SECOND BRAIN KNOWLEDGE GRAPH & SYNAPTIC PHYSICS ENGINE
// ============================================================================

class SecondBrainVisualizer {
  constructor() {
    this.canvas = document.getElementById("brain-canvas");
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
    this.viewport = document.getElementById("brain-viewport");
    this.tooltip = document.getElementById("brain-tooltip");

    this.nodes = [];
    this.edges = [];
    this.nodeMap = new Map();
    this.stats = null;

    this.selectedNode = null;
    this.hoveredNode = null;
    this.currentFilter = "all";

    this.transform = { x: 0, y: 0, scale: 1.0 };
    this.drag = { isDragging: false, isNodeDrag: false, node: null, lastX: 0, lastY: 0 };
    this.animId = null;
    this.isActive = false;

    this.colors = {
      target: { fill: "#0284c7", stroke: "#38bdf8", glow: "rgba(56, 189, 248, 0.4)", label: "🎯 Target" },
      memory: { fill: "#7e22ce", stroke: "#a855f7", glow: "rgba(168, 85, 247, 0.4)", label: "🧬 Memoria" },
      reflection: { fill: "#b45309", stroke: "#f59e0b", glow: "rgba(245, 158, 11, 0.4)", label: "💡 Riflessione" },
      fact: { fill: "#047857", stroke: "#10b981", glow: "rgba(16, 185, 129, 0.4)", label: "📚 Fatto" },
      concept: { fill: "#be123c", stroke: "#f43f5e", glow: "rgba(244, 63, 94, 0.4)", label: "⚙️ Concetto" }
    };

    if (this.canvas) {
      this.setupEventListeners();
    }
  }

  async loadAndRender() {
    this.isActive = true;
    try {
      const graph = await invoke("get_second_brain_graph");
      this.stats = graph.stats;
      this.updateStatsUI();
      this.initGraph(graph.nodes, graph.edges);
      this.startLoop();
    } catch (err) {
      console.error("Errore caricamento Second Brain:", err);
    }
  }

  updateStatsUI() {
    if (!this.stats) return;
    const elNodes = document.getElementById("stat-brain-nodes");
    const elEdges = document.getElementById("stat-brain-edges");
    const elRes = document.getElementById("stat-brain-resonance");
    const elMem = document.getElementById("stat-brain-memories");

    if (elNodes) elNodes.textContent = this.stats.total_nodes;
    if (elEdges) elEdges.textContent = this.stats.total_edges;
    if (elRes) elRes.textContent = `${this.stats.resonance_score.toFixed(1)}`;
    if (elMem) elMem.textContent = this.stats.memories_count + this.stats.reflections_count + this.stats.facts_count;
  }

  initGraph(rawNodes, rawEdges) {
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width || 800;
    const h = rect.height || 500;

    const prevMap = new Map(this.nodes.map(n => [n.id, n]));

    this.nodes = rawNodes.map((n, idx) => {
      const existing = prevMap.get(n.id);
      const angle = (idx / rawNodes.length) * Math.PI * 2;
      const dist = 120 + (idx % 3) * 60;
      
      const radius = n.category === "concept" ? 22 : (n.category === "target" ? 18 : 16);

      return {
        ...n,
        x: existing ? existing.x : Math.cos(angle) * dist,
        y: existing ? existing.y : Math.sin(angle) * dist,
        vx: existing ? existing.vx * 0.5 : (Math.random() - 0.5) * 2,
        vy: existing ? existing.vy * 0.5 : (Math.random() - 0.5) * 2,
        radius,
        color: this.colors[n.category] || this.colors.concept
      };
    });

    this.nodeMap = new Map(this.nodes.map(n => [n.id, n]));

    this.edges = rawEdges.map(e => ({
      ...e,
      sourceNode: this.nodeMap.get(e.source),
      targetNode: this.nodeMap.get(e.target),
      pulse: Math.random()
    })).filter(e => e.sourceNode && e.targetNode);

    if (!this.selectedNode && this.nodes.length > 0) {
      this.selectNode(this.nodes[0]);
    }
  }

  setupEventListeners() {
    this.canvas.addEventListener("mousedown", (e) => this.onMouseDown(e));
    window.addEventListener("mousemove", (e) => this.onMouseMove(e));
    window.addEventListener("mouseup", (e) => this.onMouseUp(e));
    this.canvas.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });

    // Filter Chips
    const chips = document.querySelectorAll(".filter-chip-group .chip");
    chips.forEach(chip => {
      chip.addEventListener("click", () => {
        chips.forEach(c => c.classList.remove("active"));
        chip.classList.add("active");
        this.currentFilter = chip.getAttribute("data-filter") || "all";
      });
    });

    // Reset Zoom
    const resetBtn = document.getElementById("brain-reset-zoom-btn");
    if (resetBtn) {
      resetBtn.addEventListener("click", () => {
        this.transform = { x: 0, y: 0, scale: 1.0 };
      });
    }

    // Spontaneous Thought Pulse Trigger
    const pulseBtn = document.getElementById("brain-pulse-btn");
    if (pulseBtn) {
      pulseBtn.addEventListener("click", async () => {
        pulseBtn.disabled = true;
        pulseBtn.textContent = "⚡ Riflettendo...";
        try {
          const res = await invoke("trigger_spontaneous_thought");
          await this.loadAndRender();
          if (res.thought_trace) {
            window.alert(`✨ Nuovo Pensiero Introspettivo Formulato da Makima:\n\n"${res.response}"`);
          }
        } catch (err) {
          window.alert("Errore generazione pensiero: " + err);
        } finally {
          pulseBtn.disabled = false;
          pulseBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> ✨ Pensiero Spontaneo`;
        }
      });
    }

    // Toggle Add View
    const toggleAddBtn = document.getElementById("brain-toggle-add-btn");
    const closeAddBtn = document.getElementById("close-add-view-btn");
    const nodeView = document.getElementById("inspector-node-view");
    const addView = document.getElementById("inspector-add-view");

    if (toggleAddBtn && closeAddBtn) {
      toggleAddBtn.addEventListener("click", () => {
        nodeView.classList.add("hidden");
        addView.classList.remove("hidden");
      });
      closeAddBtn.addEventListener("click", () => {
        addView.classList.add("hidden");
        nodeView.classList.remove("hidden");
      });
    }

    // Add Memory Form Submit
    const addForm = document.getElementById("brain-add-form");
    if (addForm) {
      addForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const category = document.getElementById("brain-mem-category").value;
        const summary = document.getElementById("brain-mem-summary").value.trim();
        const content = document.getElementById("brain-mem-content").value.trim();
        const target = document.getElementById("brain-mem-target").value.trim() || null;
        const rawTags = document.getElementById("brain-mem-tags").value.trim();
        const tags = rawTags ? rawTags.split(",").map(t => t.trim()).filter(Boolean) : [];

        if (!summary || !content) return;

        try {
          const newNode = await invoke("add_brain_memory", { category, summary, content, target, tags });
          addForm.reset();
          addView.classList.add("hidden");
          nodeView.classList.remove("hidden");
          await this.loadAndRender();
          const created = this.nodeMap.get(newNode.id);
          if (created) this.selectNode(created);
        } catch (err) {
          window.alert("Errore salvataggio memoria: " + err);
        }
      });
    }
  }

  getCanvasCoords(e) {
    const rect = this.canvas.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;

    const w = rect.width;
    const h = rect.height;

    // Convert from screen pixels to graph coordinate space
    const graphX = (clientX - w / 2 - this.transform.x) / this.transform.scale;
    const graphY = (clientY - h / 2 - this.transform.y) / this.transform.scale;

    return { graphX, graphY, screenX: clientX, screenY: clientY };
  }

  findNodeAt(graphX, graphY) {
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const node = this.nodes[i];
      if (this.currentFilter !== "all" && node.category !== this.currentFilter) continue;
      const dx = node.x - graphX;
      const dy = node.y - graphY;
      if (dx * dx + dy * dy <= (node.radius + 6) * (node.radius + 6)) {
        return node;
      }
    }
    return null;
  }

  onMouseDown(e) {
    const { graphX, graphY } = this.getCanvasCoords(e);
    const hitNode = this.findNodeAt(graphX, graphY);

    if (hitNode) {
      this.drag.isDragging = true;
      this.drag.isNodeDrag = true;
      this.drag.node = hitNode;
      this.selectNode(hitNode);
    } else {
      this.drag.isDragging = true;
      this.drag.isNodeDrag = false;
      this.drag.lastX = e.clientX;
      this.drag.lastY = e.clientY;
      this.viewport.classList.add("dragging");
    }
  }

  onMouseMove(e) {
    const { graphX, graphY, screenX, screenY } = this.getCanvasCoords(e);

    if (this.drag.isDragging) {
      if (this.drag.isNodeDrag && this.drag.node) {
        this.drag.node.x = graphX;
        this.drag.node.y = graphY;
        this.drag.node.vx = 0;
        this.drag.node.vy = 0;
      } else {
        const dx = e.clientX - this.drag.lastX;
        const dy = e.clientY - this.drag.lastY;
        this.transform.x += dx;
        this.transform.y += dy;
        this.drag.lastX = e.clientX;
        this.drag.lastY = e.clientY;
      }
    } else {
      const hitNode = this.findNodeAt(graphX, graphY);
      if (hitNode !== this.hoveredNode) {
        this.hoveredNode = hitNode;
        this.updateTooltip(hitNode, screenX, screenY);
      } else if (hitNode) {
        this.updateTooltipPos(screenX, screenY);
      }
    }
  }

  onMouseUp() {
    this.drag.isDragging = false;
    this.drag.isNodeDrag = false;
    this.drag.node = null;
    this.viewport.classList.remove("dragging");
  }

  onWheel(e) {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.12 : 0.89;
    const newScale = Math.min(Math.max(this.transform.scale * zoomFactor, 0.4), 3.0);
    this.transform.scale = newScale;
  }

  updateTooltip(node, screenX, screenY) {
    if (!node || !this.tooltip) {
      if (this.tooltip) this.tooltip.classList.add("hidden");
      return;
    }
    const catLabel = this.colors[node.category]?.label || node.category;
    this.tooltip.innerHTML = `<strong>${node.label}</strong><br/><span style="color: #94a3b8; font-size: 0.7rem;">${catLabel} • Affidabilità ${(node.confidence * 100).toFixed(0)}%</span>`;
    this.tooltip.style.left = `${screenX}px`;
    this.tooltip.style.top = `${screenY - 10}px`;
    this.tooltip.classList.remove("hidden");
  }

  updateTooltipPos(screenX, screenY) {
    if (!this.tooltip) return;
    this.tooltip.style.left = `${screenX}px`;
    this.tooltip.style.top = `${screenY - 10}px`;
  }

  selectNode(node) {
    this.selectedNode = node;
    const nodeView = document.getElementById("inspector-node-view");
    const addView = document.getElementById("inspector-add-view");
    if (nodeView && addView) {
      nodeView.classList.remove("hidden");
      addView.classList.add("hidden");
    }

    const badge = document.getElementById("inspector-badge");
    const title = document.getElementById("inspector-title");
    const confFill = document.getElementById("inspector-confidence-fill");
    const confVal = document.getElementById("inspector-confidence-val");
    const summary = document.getElementById("inspector-summary");
    const content = document.getElementById("inspector-content");
    const connCount = document.getElementById("inspector-conn-count");
    const connList = document.getElementById("inspector-connections-list");
    const tagsContainer = document.getElementById("inspector-tags");

    const catInfo = this.colors[node.category] || this.colors.concept;

    if (badge) {
      badge.textContent = catInfo.label;
      badge.style.backgroundColor = catInfo.glow;
      badge.style.color = catInfo.stroke;
      badge.style.borderColor = catInfo.stroke;
    }
    if (title) title.textContent = node.label;
    if (confFill) confFill.style.width = `${(node.confidence * 100).toFixed(0)}%`;
    if (confVal) confVal.textContent = `${(node.confidence * 100).toFixed(1)}%`;
    if (summary) summary.textContent = node.summary;
    if (content) content.textContent = node.content;

    // Connessioni
    const connectedEdges = this.edges.filter(e => e.source === node.id || e.target === node.id);
    if (connCount) connCount.textContent = connectedEdges.length;

    if (connList) {
      if (connectedEdges.length === 0) {
        connList.innerHTML = '<span class="text-dim">Nessuna connessione diretta registrata.</span>';
      } else {
        connList.innerHTML = connectedEdges.map(e => {
          const isSource = e.source === node.id;
          const otherNode = isSource ? e.targetNode : e.sourceNode;
          const otherId = otherNode ? otherNode.id : (isSource ? e.target : e.source);
          const otherLabel = otherNode ? otherNode.label : otherId;

          return `
            <div class="conn-pill" onclick="window.selectBrainNode('${otherId}')">
              <span>${isSource ? '➔' : '⬅'} <strong>${otherLabel}</strong></span>
              <span class="conn-rel">${e.relation}</span>
            </div>
          `;
        }).join("");
      }
    }

    // Tag
    if (tagsContainer) {
      if (!node.tags || node.tags.length === 0) {
        tagsContainer.innerHTML = '<span class="tag-pill">conoscenza</span>';
      } else {
        tagsContainer.innerHTML = node.tags.map(t => `<span class="tag-pill">#${t}</span>`).join("");
      }
    }
  }

  stepPhysics() {
    const kRepulsion = 2200;
    const kSpring = 0.035;
    const restLength = 115;
    const kCenter = 0.006;
    const damping = 0.85;

    // 1. Repulsione Coulombiana tra nodi
    for (let i = 0; i < this.nodes.length; i++) {
      const n1 = this.nodes[i];
      for (let j = i + 1; j < this.nodes.length; j++) {
        const n2 = this.nodes[j];
        let dx = n2.x - n1.x;
        let dy = n2.y - n1.y;
        let d2 = dx * dx + dy * dy;
        if (d2 < 1.0) {
          dx = (Math.random() - 0.5) * 2;
          dy = (Math.random() - 0.5) * 2;
          d2 = dx * dx + dy * dy + 1.0;
        }
        const d = Math.sqrt(d2);
        const force = kRepulsion / d2;
        const fx = (dx / d) * force;
        const fy = (dy / d) * force;

        n1.vx -= fx;
        n1.vy -= fy;
        n2.vx += fx;
        n2.vy += fy;
      }
    }

    // 2. Molle Hookiane lungo gli archi
    for (const edge of this.edges) {
      const n1 = edge.sourceNode;
      const n2 = edge.targetNode;
      if (!n1 || !n2) continue;

      const dx = n2.x - n1.x;
      const dy = n2.y - n1.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1.0;
      const force = (d - restLength) * kSpring * edge.weight;
      const fx = (dx / d) * force;
      const fy = (dy / d) * force;

      n1.vx += fx;
      n1.vy += fy;
      n2.vx -= fx;
      n2.vy -= fy;

      // Avanzamento impulso sinaptico
      edge.pulse = (edge.pulse + 0.007) % 1.0;
    }

    // 3. Gravità verso il centro + Integrazione velocità
    for (const node of this.nodes) {
      if (this.drag.isDragging && this.drag.node === node) continue;

      node.vx -= node.x * kCenter;
      node.vy -= node.y * kCenter;

      node.vx *= damping;
      node.vy *= damping;

      node.x += node.vx;
      node.y += node.vy;
    }
  }

  render() {
    if (!this.canvas || !this.ctx) return;
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    if (this.canvas.width !== rect.width * dpr || this.canvas.height !== rect.height * dpr) {
      this.canvas.width = rect.width * dpr;
      this.canvas.height = rect.height * dpr;
    }

    const ctx = this.ctx;
    ctx.save();
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);

    // Applica trasformazione telecamera (Pan e Zoom)
    ctx.translate(w / 2 + this.transform.x, h / 2 + this.transform.y);
    ctx.scale(this.transform.scale, this.transform.scale);

    // 1. Disegna sfondo con griglia a punti
    ctx.fillStyle = "rgba(51, 65, 85, 0.25)";
    const gridSize = 40;
    const bound = 800;
    for (let gx = -bound; gx <= bound; gx += gridSize) {
      for (let gy = -bound; gy <= bound; gy += gridSize) {
        ctx.fillRect(gx - 1, gy - 1, 2, 2);
      }
    }

    // 2. Disegna archi / sinapsi
    for (const edge of this.edges) {
      const n1 = edge.sourceNode;
      const n2 = edge.targetNode;
      if (!n1 || !n2) continue;

      const isFiltered = this.currentFilter !== "all" &&
        (n1.category !== this.currentFilter && n2.category !== this.currentFilter);

      const isHighlighted = (this.selectedNode && (this.selectedNode === n1 || this.selectedNode === n2)) ||
                            (this.hoveredNode && (this.hoveredNode === n1 || this.hoveredNode === n2));

      ctx.strokeStyle = isFiltered ? "rgba(30, 41, 59, 0.2)" : (isHighlighted ? "rgba(56, 189, 248, 0.8)" : "rgba(51, 65, 85, 0.5)");
      ctx.lineWidth = isHighlighted ? 2.5 : 1.2;
      ctx.beginPath();
      ctx.moveTo(n1.x, n1.y);
      ctx.lineTo(n2.x, n2.y);
      ctx.stroke();

      // Disegna impulso di segnale sinaptico
      if (!isFiltered) {
        const px = n1.x + (n2.x - n1.x) * edge.pulse;
        const py = n1.y + (n2.y - n1.y) * edge.pulse;
        ctx.fillStyle = isHighlighted ? "#38bdf8" : "rgba(168, 85, 247, 0.85)";
        ctx.beginPath();
        ctx.arc(px, py, isHighlighted ? 3.5 : 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // 3. Disegna Nodi
    for (const node of this.nodes) {
      const isFiltered = this.currentFilter !== "all" && node.category !== this.currentFilter;
      const isSelected = this.selectedNode === node;
      const isHovered = this.hoveredNode === node;

      const alpha = isFiltered ? 0.2 : 1.0;
      const radius = node.radius + (isSelected ? 4 : (isHovered ? 2 : 0));

      ctx.save();
      ctx.globalAlpha = alpha;

      // Outer Glow Halo
      if (isSelected || isHovered) {
        ctx.fillStyle = node.color.glow;
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius + 10, 0, Math.PI * 2);
        ctx.fill();
      }

      // Main Node Body
      ctx.fillStyle = isSelected ? node.color.stroke : node.color.fill;
      ctx.strokeStyle = node.color.stroke;
      ctx.lineWidth = isSelected ? 3 : 1.8;
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Inner icon / center dot
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius * 0.35, 0, Math.PI * 2);
      ctx.fill();

      // Node Label
      ctx.fillStyle = isSelected ? "#38bdf8" : "#f8fafc";
      ctx.font = `${isSelected ? 'bold ' : ''}11px 'Plus Jakarta Sans', sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(node.label, node.x, node.y + radius + 5);

      ctx.restore();
    }

    ctx.restore();
  }

  startLoop() {
    const loop = () => {
      if (this.isActive) {
        this.stepPhysics();
        this.render();
      }
      this.animId = requestAnimationFrame(loop);
    };
    if (this.animId) cancelAnimationFrame(this.animId);
    this.animId = requestAnimationFrame(loop);
  }
}

// Istanza globale del visualizzatore Second Brain
let secondBrain = null;

window.selectBrainNode = (nodeId) => {
  if (secondBrain) {
    const node = secondBrain.nodeMap.get(nodeId);
    if (node) secondBrain.selectNode(node);
  }
};

// Inizializzazione Principale
window.addEventListener("DOMContentLoaded", async () => {
  setupNavigation();
  setupObserveForm();
  setupChatForm();

  secondBrain = new SecondBrainVisualizer();

  // Navigation hook per caricamento Second Brain
  const brainNavBtn = document.querySelector('.nav-item[data-tab="brain"]');
  if (brainNavBtn) {
    brainNavBtn.addEventListener("click", () => {
      secondBrain.loadAndRender();
    });
  }

  document.getElementById("target-selector").addEventListener("change", (e) => {
    currentTarget = e.target.value;
    loadTargetDistribution(currentTarget);
  });

  document.getElementById("refresh-btn").addEventListener("click", () => {
    loadDashboardData();
    if (secondBrain && secondBrain.isActive) secondBrain.loadAndRender();
  });

  document.getElementById("sync-git-btn").addEventListener("click", async () => {
    const btn = document.getElementById("sync-git-btn");
    btn.disabled = true;
    btn.textContent = "Sincronizzazione...";
    try {
      const count = await invoke("sync_git_telemetry");
      window.alert(`Sincronizzazione Git completata con successo! Totale osservazioni: ${count}`);
      await loadDashboardData();
      if (secondBrain) await secondBrain.loadAndRender();
    } catch (e) {
      window.alert("Errore sincronizzazione Git: " + e);
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none"><circle cx="12" cy="12" r="4"/><line x1="1.05" y1="12" x2="7" y2="12"/><line x1="17.01" y1="12" x2="22.96" y2="12"/></svg> Sincronizza Git`;
    }
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

