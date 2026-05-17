const data = window.MCA_DEMO_DATA || {
  metadata: {},
  groups: [],
  accountsByGroup: {},
  pairsByGroup: {},
  edgesByGroup: {},
  sharedTargetsByGroup: {},
  abnormalAccounts: [],
};

let selectedSeed = data.groups[0]?.seed || "";
let currentSort = "rank";

const $ = (selector) => document.querySelector(selector);
const fmt = (value, digits = 2) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "n/a";
  return n.toFixed(digits);
};
const pct = (value) => `${fmt(Number(value) * 100, 0)}%`;

const LABELS = {
  high_confidence_temporal_candidate: "High priority: group evidence",
  high_confidence_extreme_outlier: "High priority: abnormal account",
  high_mca_review_candidate: "Review: high MCA",
  temporal_only_review_candidate: "Review: timing evidence",
  low_priority_context_member: "Context member",
  strong_temporal_sync: "Strong timing sync",
  moderate_temporal_sync: "Moderate timing sync",
  weak_temporal_overlap: "Weak timing overlap",
  no_temporal_sync: "No timing evidence",
  robust: "Robust",
  moderate_review: "Reviewable",
  fragile_single_event: "Fragile: single event",
  fragile_long_median: "Fragile: long delay",
  weak_context: "Weak context",
  none: "None",
  extreme_outlier_behavior: "Extreme behavior outlier",
  high_activity_behavior: "High activity",
  low_activity_unknown: "Low activity / unknown",
};

function priorityLabel(value) {
  const raw = String(value || "");
  return LABELS[raw] || raw.replaceAll("_", " ");
}

function scoreBar(value, className = "") {
  const width = Math.max(0, Math.min(100, Number(value) * 100 || 0));
  return `<div class="bar ${className}"><span style="width:${width}%"></span></div>`;
}

function reviewPriorityClass(priority) {
  if (priority === "high_confidence_temporal_candidate") return "hot";
  if (priority === "high_confidence_extreme_outlier") return "outlier";
  if (priority === "temporal_only_review_candidate") return "sync";
  return "";
}

function renderMeta() {
  const meta = data.metadata || {};
  $("#runMeta").innerHTML = `
    <div><strong>Source</strong><br>${meta.source || "pipeline output"}</div>
    <div><strong>Groups</strong><br>${meta.groupCount || data.groups.length}</div>
    <div><strong>Candidate rows</strong><br>${meta.candidateAccountRows || 0}</div>
    <div><strong>Pair evidence</strong><br>${meta.pairEvidenceRows || 0}</div>
  `;
}

function renderMetrics() {
  const groups = data.groups;
  const p1 = groups.reduce((sum, g) => sum + Number(g.p1 || 0), 0);
  const reliable = groups.reduce((sum, g) => sum + Number(g.reliableTemporalPairs || 0), 0);
  const outliers = data.abnormalAccounts.length;
  const totalMembers = groups.reduce((sum, g) => sum + Number(g.memberCount || 0), 0);
  $("#metricGrid").innerHTML = `
    <div class="metric"><strong>${groups.length}</strong><span>ranked coordination groups</span></div>
    <div class="metric"><strong>${totalMembers}</strong><span>expanded group members</span></div>
    <div class="metric"><strong>${p1}</strong><span>P1 review candidates</span></div>
    <div class="metric"><strong>${reliable}</strong><span>reliable temporal pairs</span></div>
    <div class="metric"><strong>${outliers}</strong><span>individual abnormal accounts</span></div>
    <div class="metric"><strong>${data.groups[0]?.seed || "n/a"}</strong><span>top ranked group</span></div>
  `;
}

function sortedGroups() {
  const groups = [...data.groups];
  if (currentSort === "p1") groups.sort((a, b) => b.p1 - a.p1 || a.rank - b.rank);
  if (currentSort === "robust") groups.sort((a, b) => b.robustTemporalPairs - a.robustTemporalPairs || b.reliableTemporalPairs - a.reliableTemporalPairs);
  if (currentSort === "mca") groups.sort((a, b) => b.maxMca - a.maxMca);
  if (currentSort === "rank") groups.sort((a, b) => a.rank - b.rank);
  return groups;
}

function renderGroups() {
  $("#groupList").innerHTML = sortedGroups()
    .map((group) => {
      const active = group.seed === selectedSeed ? "active" : "";
      const evidenceLine = group.reliableTemporalPairs > 0
        ? `${group.reliableTemporalPairs} reliable timing pair${group.reliableTemporalPairs === 1 ? "" : "s"}`
        : `${group.moderateReviewTemporalPairs} reviewable timing pair${group.moderateReviewTemporalPairs === 1 ? "" : "s"}`;
      return `
        <button class="group-row ${active}" type="button" data-seed="${group.seed}">
          <div class="group-row-top">
            <span class="group-row-title">${group.seed}</span>
            <span class="rank">#${group.rank}</span>
          </div>
          <p class="group-row-summary">${evidenceLine}; ${group.p1} high-priority member${group.p1 === 1 ? "" : "s"}.</p>
          <div class="mini-metrics">
            <span class="chip hot">P1 ${group.p1}</span>
            <span class="chip">members ${group.memberCount}</span>
            <span class="chip sync">reliable ${group.reliableTemporalPairs}</span>
            <span class="chip outlier">outliers ${group.extremeOutliers}</span>
          </div>
        </button>
      `;
    })
    .join("");

  document.querySelectorAll(".group-row").forEach((button) => {
    button.addEventListener("click", () => {
      selectedSeed = button.dataset.seed;
      renderAll();
    });
  });
}

function selectedGroup() {
  return data.groups.find((group) => group.seed === selectedSeed) || data.groups[0];
}

function renderGroupHeader() {
  const group = selectedGroup();
  if (!group) return;
  $("#selectedGroupTitle").textContent = `${group.seed} group`;
  $("#groupPriorityPill").textContent = priorityLabel(group.priority);
  $("#detailStats").innerHTML = `
    <div class="stat-cell"><strong>${group.memberCount}</strong><span>members after expansion</span></div>
    <div class="stat-cell"><strong>${group.p1}</strong><span>P1 candidates</span></div>
    <div class="stat-cell"><strong>${group.reliableTemporalPairs}</strong><span>reliable temporal pairs</span></div>
    <div class="stat-cell"><strong>${group.sharedNegativeTargets}</strong><span>shared negative targets</span></div>
    <div class="stat-cell"><strong>${fmt(group.maxMca, 2)}</strong><span>max MCA score</span></div>
    <div class="stat-cell"><strong>${pct(group.automationFraction)}</strong><span>automation anomaly fraction</span></div>
  `;
  $("#groupStory").innerHTML = renderGroupStory(group);
}

function renderGroupStory(group) {
  const coordination = group.tier1CoNegative > 0
    ? `Stage 1 found ${group.tier1CoNegative} direct co-negative links around the seed.`
    : `Stage 1 found this group through weaker or indirect graph context.`;
  const temporal = group.robustTemporalPairs > 0
    ? `Stage 2 found ${group.robustTemporalPairs} robust timing pair${group.robustTemporalPairs === 1 ? "" : "s"}, so this group has stronger evidence of acting together.`
    : group.moderateReviewTemporalPairs > 0
      ? `Stage 2 found ${group.moderateReviewTemporalPairs} reviewable timing pair${group.moderateReviewTemporalPairs === 1 ? "" : "s"}, but the evidence still needs human inspection.`
      : `Stage 2 did not find reliable timing evidence, so this is mainly a graph-based candidate group.`;
  const caution = group.fragileTemporalPairs > 0
    ? `${group.fragileTemporalPairs} timing pair${group.fragileTemporalPairs === 1 ? "" : "s"} are fragile and should not be overread.`
    : `No fragile timing pair dominates the group summary.`;
  return `
    <div class="story-card primary">
      <strong>Plain-language takeaway</strong>
      <span>${coordination} ${temporal}</span>
    </div>
    <div class="story-card">
      <strong>What to be careful about</strong>
      <span>${caution} This dashboard says “review this,” not “confirmed bot network.”</span>
    </div>
  `;
}

function accountColor(account) {
  if (account.tier === 0) return "#17202a";
  if (account.reviewPriority === "high_confidence_temporal_candidate") return "#c84038";
  if (account.reviewPriority === "high_confidence_extreme_outlier") return "#c98618";
  if (account.reviewPriority === "temporal_only_review_candidate") return "#4267b2";
  if (account.reviewPriority === "high_mca_review_candidate") return "#7a5aa6";
  return "#8d9aa5";
}

function renderGraph() {
  const group = selectedGroup();
  if (!group) return;
  const svg = $("#groupGraph");
  const width = svg.clientWidth || 900;
  const height = svg.clientHeight || 430;
  const accounts = (data.accountsByGroup[group.seed] || []).slice(0, 18);
  const accountMap = new Map(accounts.map((account) => [account.account, account]));
  const rawEdges = data.edgesByGroup[group.seed] || [];
  const rawPairs = data.pairsByGroup[group.seed] || [];

  const edges = rawEdges
    .filter((edge) => accountMap.has(edge.source) && accountMap.has(edge.target))
    .slice(0, 45)
    .map((edge) => ({ ...edge, type: "co" }));

  const temporalEdges = rawPairs
    .filter((pair) => pair.within30 > 0 && accountMap.has(pair.a) && accountMap.has(pair.b))
    .slice(0, 25)
    .map((pair) => ({
      source: pair.a,
      target: pair.b,
      weight: Math.max(0.15, Math.min(1, pair.within30 / 6)),
      type: "temporal",
    }));

  const allEdges = [...edges, ...temporalEdges];
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) * 0.34;
  const seedIndex = accounts.findIndex((account) => account.tier === 0);
  const nodes = accounts.map((account, index) => {
    if (index === seedIndex || account.account === group.seed) {
      return { ...account, x: centerX, y: centerY };
    }
    const adjustedIndex = seedIndex >= 0 && index > seedIndex ? index - 1 : index;
    const count = Math.max(1, accounts.length - 1);
    const angle = (Math.PI * 2 * adjustedIndex) / count - Math.PI / 2;
    const mcaBoost = Math.min(46, Number(account.mca || 0) * 34);
    return {
      ...account,
      x: centerX + Math.cos(angle) * (radius + mcaBoost),
      y: centerY + Math.sin(angle) * (radius + mcaBoost * 0.4),
    };
  });
  const nodeMap = new Map(nodes.map((node) => [node.account, node]));

  const edgeMarkup = allEdges
    .map((edge) => {
      const source = nodeMap.get(edge.source);
      const target = nodeMap.get(edge.target);
      if (!source || !target) return "";
      const strokeWidth = Math.max(1.2, Math.min(7, Number(edge.weight || 0) * 8));
      const cls = edge.type === "temporal" ? "graph-edge temporal" : "graph-edge";
      return `<line class="${cls}" x1="${source.x}" y1="${source.y}" x2="${target.x}" y2="${target.y}" stroke-width="${strokeWidth}" />`;
    })
    .join("");

  const nodeMarkup = nodes
    .map((node) => {
      const size = node.tier === 0 ? 15 : 8 + Math.min(8, Number(node.mca || 0) * 10);
      const label = node.account.length > 18 ? `${node.account.slice(0, 16)}…` : node.account;
      return `
        <g>
          <circle class="graph-node" cx="${node.x}" cy="${node.y}" r="${size}" fill="${accountColor(node)}" />
          <text class="node-label" x="${node.x + size + 5}" y="${node.y + 4}">${label}</text>
        </g>
      `;
    })
    .join("");

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `${edgeMarkup}${nodeMarkup}`;
}

function renderMembers() {
  const group = selectedGroup();
  const accounts = data.accountsByGroup[group.seed] || [];
  $("#tab-members").innerHTML = `
    <div class="tab-intro">
      <strong>Members are sorted by review priority.</strong>
      <span>P1 means the account has stronger supporting evidence. Context members are shown so the graph remains explainable.</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Account</th>
            <th>Why it is shown</th>
            <th>Role</th>
            <th>MCA</th>
            <th>Temporal</th>
            <th>Behavior</th>
          </tr>
        </thead>
        <tbody>
          ${accounts
            .map(
              (account) => `
                <tr>
                  <td class="account-name">${account.account}<br><span class="muted">tier ${account.tier} · ${account.includeReason}</span></td>
                  <td><span class="chip ${reviewPriorityClass(account.reviewPriority)}">${priorityLabel(account.reviewPriority)}</span></td>
                  <td>${account.roleZh || account.role || "n/a"}<br><span class="muted">${account.roleReason || ""}</span></td>
                  <td>${fmt(account.mca, 3)}${scoreBar(account.mca, "red")}</td>
                  <td>${priorityLabel(account.bestTemporalLabel)}<br><span class="muted">${priorityLabel(account.bestTemporalConfidence)}</span></td>
                  <td>${priorityLabel(account.behaviorProfile || "unknown")}<br><span class="muted">anomaly ${fmt(account.anomalyScore, 2)}</span></td>
                </tr>
              `
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderPairs() {
  const group = selectedGroup();
  const pairs = data.pairsByGroup[group.seed] || [];
  const usefulPairs = pairs.filter((pair) => pair.label !== "no_temporal_sync" || pair.textDistance < 0.35).slice(0, 50);
  $("#tab-pairs").innerHTML = `
    <div class="tab-intro">
      <strong>Pair evidence explains why two accounts are connected.</strong>
      <span>Timing sync asks whether they appeared together; reliability asks whether that pattern is stable enough to trust.</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Pair</th>
            <th>Temporal</th>
            <th>Reliability</th>
            <th>Text distance</th>
            <th>Activation overlap</th>
            <th>Co-negative</th>
          </tr>
        </thead>
        <tbody>
          ${usefulPairs
            .map(
              (pair) => `
                <tr>
                  <td class="account-name">${pair.a}<br><span class="muted">↔ ${pair.b}</span></td>
                  <td>${priorityLabel(pair.label)}<br><span class="muted">${pair.samePost} same post · ${pair.within30} within 30m</span></td>
                  <td>${priorityLabel(pair.confidence)}<br><span class="muted">median ${fmt(pair.medianDelay, 1)} min</span></td>
                  <td>${fmt(pair.textDistance, 3)}${scoreBar(Math.max(0, 1 - Number(pair.textDistance || 1)), "amber")}</td>
                  <td>${pct(pair.activationOverlap)}${scoreBar(pair.activationOverlap)}</td>
                  <td>${fmt(pair.coNegative, 3)}</td>
                </tr>
              `
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderTargets() {
  const group = selectedGroup();
  const targets = data.sharedTargetsByGroup[group.seed] || [];
  $("#tab-targets").innerHTML = `
    <div class="tab-intro">
      <strong>Shared targets are the graph reason for expansion.</strong>
      <span>These accounts reacted negatively toward overlapping targets. This is useful for discovery, but not enough by itself to prove manipulation.</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Neighbor</th>
            <th>Shared targets</th>
            <th>Weight</th>
            <th>Targets</th>
          </tr>
        </thead>
        <tbody>
          ${targets
            .map(
              (target) => `
                <tr>
                  <td class="account-name">${target.neighbor}</td>
                  <td>${target.count}</td>
                  <td>${fmt(target.weight, 3)}</td>
                  <td>${target.targets}</td>
                </tr>
              `
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderAbnormalAccounts() {
  $("#abnormalTable").innerHTML = `
    <div class="tab-intro standalone">
      <strong>This is a separate result type.</strong>
      <span>An account can be abnormal or spam-like without being part of a coordinated group. Keeping this separate prevents overclaiming.</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Account</th>
            <th>Found in group</th>
            <th>Priority</th>
            <th>MCA</th>
            <th>Behavior profile</th>
            <th>Why it matters</th>
          </tr>
        </thead>
        <tbody>
          ${data.abnormalAccounts
            .map(
              (account) => `
                <tr>
                  <td class="account-name">${account.account}</td>
                  <td>${account.seedGroup}</td>
                  <td><span class="chip ${reviewPriorityClass(account.reviewPriority)}">${priorityLabel(account.reviewPriority)}</span></td>
                  <td>${fmt(account.mca, 3)}${scoreBar(account.mca, "red")}</td>
                  <td>${priorityLabel(account.behaviorProfile || "unknown")}<br><span class="muted">cluster ${account.cluster} · anomaly ${fmt(account.anomalyScore, 2)}</span></td>
                  <td>${account.behaviorReason || "High-risk individual account separated from group-level coordination claims."}</td>
                </tr>
              `
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function bindTabs() {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab-button").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      $(`#tab-${button.dataset.tab}`).classList.add("active");
    });
  });
}

function renderAll() {
  renderGroups();
  renderGroupHeader();
  renderGraph();
  renderMembers();
  renderPairs();
  renderTargets();
}

function init() {
  renderMeta();
  renderMetrics();
  renderAbnormalAccounts();
  renderAll();
  bindTabs();
  $("#groupSort").addEventListener("change", (event) => {
    currentSort = event.target.value;
    renderGroups();
  });
  $("#fitGraph").addEventListener("click", renderGraph);
  window.addEventListener("resize", renderGraph);
}

init();
