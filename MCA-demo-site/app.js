const data = window.MCA_DEMO_DATA || {
  metadata: {},
  groups: [],
  accountsByGroup: {},
  pairsByGroup: {},
  edgesByGroup: {},
  sharedTargetsByGroup: {},
  abnormalAccounts: [],
};

let currentLang = localStorage.getItem("mca-demo-lang") || "zh";
let currentMode = localStorage.getItem("mca-demo-mode") || "client";
let selectedSeed = data.groups[0]?.seed || "";
let currentSort = "rank";
const hashSeed = decodeURIComponent(window.location.hash.slice(1));
if (hashSeed && data.groups.some((group) => group.seed === hashSeed)) {
  selectedSeed = hashSeed;
}

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const page = document.body.dataset.page || "overview";

const COPY = {
  zh: {
    client: {
      brandTag: "協同風險情資",
      navOverview: "總覽",
      navGroups: "風險群組",
      navAccounts: "異常帳號",
      navMethod: "方法",
      overviewEyebrow: "企業風險營運",
      overviewTitle: "把大量帳號互動轉成可審查的風險案件",
      overviewLead:
        "MCA Sentinel 協助分析師從社群互動中找出可疑群組、同步行為與單一異常帳號，並把結果整理成可以優先處理的審查隊列。",
      reviewGroups: "檢視風險群組",
      viewMethod: "查看方法",
      priorityEyebrow: "優先處理",
      priorityTitle: "目前最值得審查的群組",
      workflowEyebrow: "作業流程",
      workflowTitle: "給分析師的下一步",
      groupsEyebrow: "風險群組審查",
      groupsTitle: "從候選群組到可解釋證據",
      groupsLead:
        "這裡呈現每個候選群的圖結構關係、短時間同步證據與帳號風險。群組排名代表審查優先級，不是自動判決。",
      queueEyebrow: "審查隊列",
      queueTitle: "群組排名",
      selectedEyebrow: "選取案件",
      accountsEyebrow: "單一帳號風險",
      accountsTitle: "不一定屬於群組，但仍值得審查",
      accountsLead:
        "有些帳號本身呈現高度異常或 spam/scam-like 行為，但不一定有群體協同證據；因此我們把它們獨立呈現。",
      abnormalEyebrow: "獨立風險輸出",
      abnormalTitle: "單一異常帳號",
      methodEyebrow: "方法摘要",
      methodTitle: "先找候選群，再用時間同步驗證",
      methodLead:
        "系統不是只靠單一分數判斷，而是先用帳號風險找入口，再用共同目標形成候選群，最後用短時間同步行為做驗證。",
      pruningEyebrow: "訊號剪枝",
      pruningTitle: "目前保留與移除的訊號",
    },
    demo: {
      brandTag: "Manipulative Coordination Analysis",
      navOverview: "總覽",
      navGroups: "群組",
      navAccounts: "異常帳號",
      navMethod: "方法",
      overviewEyebrow: "研究展示",
      overviewTitle: "MCA pipeline 輸出總覽",
      overviewLead:
        "Demo mode 保留研究語言：MCA seed selection、graph expansion、temporal verification 與 validation output。",
      reviewGroups: "查看 groups",
      viewMethod: "查看 pipeline",
      priorityEyebrow: "Case queue",
      priorityTitle: "Top candidate groups",
      workflowEyebrow: "Pipeline",
      workflowTitle: "目前流程",
      groupsEyebrow: "Stage 1 + Stage 2",
      groupsTitle: "Candidate groups and temporal evidence",
      groupsLead:
        "Stage 1 用 co-negative graph 找候選群；Stage 2 用 temporal synchrony 和 confidence 做 verification。",
      queueEyebrow: "Seed expansion",
      queueTitle: "Group ranking",
      selectedEyebrow: "Selected group",
      accountsEyebrow: "Account-level output",
      accountsTitle: "Individual abnormal accounts",
      accountsLead: "MCA/anomaly 可能找到高風險單一帳號，但這和 group-level coordination 分開處理。",
      abnormalEyebrow: "Separate output stream",
      abnormalTitle: "Individual abnormal accounts",
      methodEyebrow: "Pipeline method",
      methodTitle: "Graph discovery followed by temporal verification",
      methodLead:
        "MCA scores are used for seed selection; co-negative graph discovers candidate groups; temporal synchrony verifies coordinated action.",
      pruningEyebrow: "Signal pruning",
      pruningTitle: "Signals kept or removed",
    },
  },
  en: {
    client: {
      brandTag: "Coordination Risk Intelligence",
      navOverview: "Overview",
      navGroups: "Risk Groups",
      navAccounts: "Abnormal Accounts",
      navMethod: "Method",
      overviewEyebrow: "Enterprise risk operations",
      overviewTitle: "Turn account activity into reviewable risk cases",
      overviewLead:
        "MCA Sentinel helps analysts identify suspicious groups, synchronized behavior, and individual abnormal accounts from social-media activity, then turns them into a prioritized review queue.",
      reviewGroups: "Review risk groups",
      viewMethod: "View method",
      priorityEyebrow: "Prioritize",
      priorityTitle: "Groups that need attention first",
      workflowEyebrow: "Workflow",
      workflowTitle: "Recommended analyst flow",
      groupsEyebrow: "Risk group review",
      groupsTitle: "From candidate groups to explainable evidence",
      groupsLead:
        "Each case shows graph relationships, short-window timing evidence, and account-level risk. Ranking means review priority, not an automatic verdict.",
      queueEyebrow: "Review queue",
      queueTitle: "Group ranking",
      selectedEyebrow: "Selected case",
      accountsEyebrow: "Account-level risk",
      accountsTitle: "Not always coordinated, still worth reviewing",
      accountsLead:
        "Some accounts appear abnormal or spam-like without confirmed group coordination, so they are reported as a separate risk stream.",
      abnormalEyebrow: "Separate risk stream",
      abnormalTitle: "Individual abnormal accounts",
      methodEyebrow: "Method summary",
      methodTitle: "Discover candidate groups, then verify timing",
      methodLead:
        "The system does not rely on one score. It uses account risk to find seeds, shared targets to form candidate groups, and short-window timing behavior to verify coordination.",
      pruningEyebrow: "Signal pruning",
      pruningTitle: "Signals kept or removed",
    },
    demo: {
      brandTag: "Manipulative Coordination Analysis",
      navOverview: "Overview",
      navGroups: "Groups",
      navAccounts: "Accounts",
      navMethod: "Method",
      overviewEyebrow: "Research demo",
      overviewTitle: "MCA pipeline output overview",
      overviewLead:
        "Demo mode keeps the research vocabulary visible: MCA seed selection, graph expansion, temporal verification, and validation output.",
      reviewGroups: "View groups",
      viewMethod: "View pipeline",
      priorityEyebrow: "Case queue",
      priorityTitle: "Top candidate groups",
      workflowEyebrow: "Pipeline",
      workflowTitle: "Current flow",
      groupsEyebrow: "Stage 1 + Stage 2",
      groupsTitle: "Candidate groups and temporal evidence",
      groupsLead:
        "Stage 1 uses the co-negative graph to discover candidate groups. Stage 2 verifies them with temporal synchrony and confidence.",
      queueEyebrow: "Seed expansion",
      queueTitle: "Group ranking",
      selectedEyebrow: "Selected group",
      accountsEyebrow: "Account-level output",
      accountsTitle: "Individual abnormal accounts",
      accountsLead: "MCA/anomaly may find high-risk individual accounts, handled separately from group-level coordination.",
      abnormalEyebrow: "Separate output stream",
      abnormalTitle: "Individual abnormal accounts",
      methodEyebrow: "Pipeline method",
      methodTitle: "Graph discovery followed by temporal verification",
      methodLead:
        "MCA scores are used for seed selection; co-negative graph discovers candidate groups; temporal synchrony verifies coordinated action.",
      pruningEyebrow: "Signal pruning",
      pruningTitle: "Signals kept or removed",
    },
  },
};

const LABELS = {
  zh: {
    high_confidence_temporal_candidate: "高優先：群體證據",
    high_confidence_extreme_outlier: "高優先：單一異常",
    high_mca_review_candidate: "待審查：MCA 高",
    temporal_only_review_candidate: "待審查：時間證據",
    low_priority_context_member: "背景成員",
    strong_temporal_sync: "強時間同步",
    moderate_temporal_sync: "中等時間同步",
    weak_temporal_overlap: "弱時間重疊",
    no_temporal_sync: "無時間證據",
    robust: "穩定",
    moderate_review: "值得審查",
    fragile_single_event: "脆弱：單次事件",
    fragile_long_median: "脆弱：延遲偏長",
    weak_context: "弱背景",
    none: "無",
    extreme_outlier_behavior: "極端異常行為",
    high_activity_behavior: "高活動行為",
    low_activity_unknown: "低活動／未知",
  },
  en: {
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
  },
};

function copy() {
  return COPY[currentLang][currentMode];
}

function label(value) {
  return LABELS[currentLang][value] || String(value || "").replaceAll("_", " ");
}

function fmt(value, digits = 2) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "n/a";
}

function pct(value) {
  return `${fmt(Number(value) * 100, 0)}%`;
}

function bar(value, tone = "teal") {
  const width = Math.max(0, Math.min(100, Number(value || 0) * 100));
  return `<span class="bar ${tone}"><i style="width:${width}%"></i></span>`;
}

function priorityClass(priority) {
  if (priority === "high_confidence_temporal_candidate") return "danger";
  if (priority === "high_confidence_extreme_outlier") return "amber";
  if (priority === "temporal_only_review_candidate") return "blue";
  return "";
}

function setStaticCopy() {
  document.documentElement.lang = currentLang === "zh" ? "zh-Hant" : "en";
  document.body.dataset.lang = currentLang;
  document.body.dataset.mode = currentMode;
  $$("[data-copy]").forEach((node) => {
    const key = node.dataset.copy;
    if (copy()[key]) node.textContent = copy()[key];
  });
  $$("[data-mode]").forEach((button) => button.classList.toggle("active", button.dataset.mode === currentMode));
  $$("[data-lang]").forEach((button) => button.classList.toggle("active", button.dataset.lang === currentLang));
}

function metrics() {
  const groups = data.groups || [];
  return {
    groupCount: groups.length,
    memberCount: groups.reduce((sum, group) => sum + Number(group.memberCount || 0), 0),
    p1Count: groups.reduce((sum, group) => sum + Number(group.p1 || 0), 0),
    reliablePairs: groups.reduce((sum, group) => sum + Number(group.reliableTemporalPairs || 0), 0),
    abnormalCount: data.abnormalAccounts?.length || 0,
    topGroup: groups[0]?.seed || "n/a",
  };
}

function renderMetricGrid() {
  const m = metrics();
  const zh = currentLang === "zh";
  const items = zh
    ? [
        [m.groupCount, "風險群組"],
        [m.p1Count, "高優先帳號"],
        [m.reliablePairs, "可信時間證據"],
        [m.abnormalCount, "單一異常帳號"],
      ]
    : [
        [m.groupCount, "risk groups"],
        [m.p1Count, "high-priority accounts"],
        [m.reliablePairs, "reliable timing pairs"],
        [m.abnormalCount, "abnormal accounts"],
      ];
  const node = $("#metricGrid");
  if (!node) return;
  node.innerHTML = items.map(([value, name]) => `<div class="kpi"><strong>${value}</strong><span>${name}</span></div>`).join("");
}

function renderTopCaseCard() {
  const top = data.groups[0];
  const node = $("#topCaseCard");
  if (!node || !top) return;
  const zh = currentLang === "zh";
  node.innerHTML = `
    <div class="case-card-head">
      <span class="case-rank">#${top.rank}</span>
      <span class="pill danger">${zh ? "最高優先" : "Top priority"}</span>
    </div>
    <h2>${top.seed}</h2>
    <p>${zh ? "目前排名最高的審查案件，適合先看同步證據與群內高風險帳號。" : "The highest-ranked review case. Start with timing evidence and high-risk members."}</p>
    <dl class="mini-stats">
      <div><dt>${zh ? "成員" : "Members"}</dt><dd>${top.memberCount}</dd></div>
      <div><dt>P1</dt><dd>${top.p1}</dd></div>
      <div><dt>${zh ? "可信時間" : "Reliable timing"}</dt><dd>${top.reliableTemporalPairs}</dd></div>
    </dl>
    <a class="button primary full" href="./groups.html">${zh ? "開啟案件" : "Open case"}</a>
  `;
}

function renderPriorityGroups() {
  const node = $("#priorityGroups");
  if (!node) return;
  node.innerHTML = data.groups.slice(0, 5).map((group) => `
    <a class="priority-row" href="./groups.html#${encodeURIComponent(group.seed)}">
      <span class="case-rank">#${group.rank}</span>
      <strong>${group.seed}</strong>
      <small>${group.memberCount} ${currentLang === "zh" ? "成員" : "members"} · P1 ${group.p1} · ${group.reliableTemporalPairs} ${currentLang === "zh" ? "可信時間證據" : "reliable timing"}</small>
    </a>
  `).join("");
}

function renderWorkflow() {
  const node = $("#workflowList");
  if (!node) return;
  const steps = currentLang === "zh"
    ? [
        ["01", "先看最高優先群組", "從群組排名進入案件，確認是否有共同目標與同步到場。"],
        ["02", "檢查同步帳號對", "優先看 strong / robust temporal pairs，避免只靠關係圖下結論。"],
        ["03", "分開處理單一異常帳號", "spam/scam-like 帳號有治理價值，但不要硬說成群體協同。"],
      ]
    : [
        ["01", "Start with top groups", "Open the ranked cases and inspect shared targets plus synchronized arrivals."],
        ["02", "Review timing pairs", "Prioritize strong / robust temporal pairs instead of relying on the graph alone."],
        ["03", "Separate abnormal accounts", "Spam/scam-like accounts are useful findings, but not automatically group coordination."],
      ];
  node.innerHTML = steps.map(([num, title, body]) => `
    <div class="workflow-item"><span>${num}</span><div><strong>${title}</strong><p>${body}</p></div></div>
  `).join("");
}

function sortedGroups() {
  const groups = [...data.groups];
  if (currentSort === "p1") groups.sort((a, b) => b.p1 - a.p1 || a.rank - b.rank);
  if (currentSort === "robust") groups.sort((a, b) => b.reliableTemporalPairs - a.reliableTemporalPairs || a.rank - b.rank);
  if (currentSort === "mca") groups.sort((a, b) => b.maxMca - a.maxMca);
  if (currentSort === "rank") groups.sort((a, b) => a.rank - b.rank);
  return groups;
}

function renderGroupList() {
  const node = $("#groupList");
  if (!node) return;
  node.innerHTML = sortedGroups().map((group) => `
    <button class="group-row ${group.seed === selectedSeed ? "active" : ""}" type="button" data-seed="${group.seed}">
      <span class="case-rank">#${group.rank}</span>
      <strong>${group.seed}</strong>
      <small>${group.memberCount} ${currentLang === "zh" ? "成員" : "members"} · P1 ${group.p1} · ${group.reliableTemporalPairs} ${currentLang === "zh" ? "可信時間證據" : "reliable timing"}</small>
    </button>
  `).join("");
  $$(".group-row").forEach((button) => {
    button.addEventListener("click", () => {
      selectedSeed = button.dataset.seed;
      renderGroupsPage();
    });
  });
}

function selectedGroup() {
  return data.groups.find((group) => group.seed === selectedSeed) || data.groups[0];
}

function renderGroupSummary() {
  const group = selectedGroup();
  if (!group) return;
  const zh = currentLang === "zh";
  $("#selectedGroupTitle").textContent = `${group.seed} ${zh ? "案件" : "case"}`;
  $("#groupPriorityPill").textContent = label(group.priority);
  $("#detailStats").innerHTML = [
    [group.memberCount, zh ? "群組成員" : "Members"],
    [group.p1, zh ? "高優先帳號" : "P1 accounts"],
    [group.reliableTemporalPairs, zh ? "可信時間證據" : "Reliable timing"],
    [group.sharedNegativeTargets, zh ? "共同目標" : "Shared targets"],
    [fmt(group.maxMca, 2), "Max MCA"],
    [pct(group.automationFraction), zh ? "異常比例" : "Anomaly share"],
  ].map(([value, name]) => `<div class="stat"><strong>${value}</strong><span>${name}</span></div>`).join("");
  const hasReliable = group.reliableTemporalPairs > 0;
  $("#groupStory").innerHTML = `
    <div>
      <strong>${zh ? "審查重點" : "Review focus"}</strong>
      <p>${hasReliable
        ? (zh ? `這個群組同時有共同目標與可信時間同步，應優先檢查同步帳號對。` : `This group has both shared-target evidence and reliable timing evidence. Review synchronized pairs first.`)
        : (zh ? `這個群組主要由圖結構帶出，尚需人工檢查時間證據是否足夠。` : `This group is mostly graph-driven and needs manual inspection of timing evidence.`)
      }</p>
    </div>
    <div>
      <strong>${zh ? "注意" : "Caution"}</strong>
      <p>${zh ? "這是審查優先級，不是自動判決。只看共同目標容易把自然同溫層誤判成協同。" : "This is review priority, not an automatic verdict. Shared targets alone can confuse organic alignment with coordination."}</p>
    </div>
  `;
}

function renderGraph() {
  const group = selectedGroup();
  const svg = $("#groupGraph");
  if (!group || !svg) return;
  const width = svg.clientWidth || 900;
  const height = svg.clientHeight || 420;
  const accounts = (data.accountsByGroup[group.seed] || []).slice(0, 16);
  const accountMap = new Map(accounts.map((account) => [account.account, account]));
  const edges = (data.edgesByGroup[group.seed] || [])
    .filter((edge) => accountMap.has(edge.source) && accountMap.has(edge.target))
    .slice(0, 32)
    .map((edge) => ({ ...edge, kind: "graph" }));
  const temporalEdges = (data.pairsByGroup[group.seed] || [])
    .filter((pair) => pair.within30 > 0 && accountMap.has(pair.a) && accountMap.has(pair.b))
    .slice(0, 18)
    .map((pair) => ({ source: pair.a, target: pair.b, weight: Math.min(1, pair.within30 / 5), kind: "temporal" }));
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) * 0.32;
  const nodes = accounts.map((account, index) => {
    if (account.tier === 0 || account.account === group.seed) return { ...account, x: centerX, y: centerY };
    const angle = ((Math.PI * 2) / Math.max(1, accounts.length - 1)) * index - Math.PI / 2;
    return { ...account, x: centerX + Math.cos(angle) * radius, y: centerY + Math.sin(angle) * radius };
  });
  const nodeMap = new Map(nodes.map((node) => [node.account, node]));
  const edgeMarkup = [...edges, ...temporalEdges].map((edge) => {
    const s = nodeMap.get(edge.source);
    const t = nodeMap.get(edge.target);
    if (!s || !t) return "";
    const cls = edge.kind === "temporal" ? "graph-edge temporal" : "graph-edge";
    return `<line class="${cls}" x1="${s.x}" y1="${s.y}" x2="${t.x}" y2="${t.y}" stroke-width="${Math.max(1.2, Number(edge.weight || 0.2) * 6)}"></line>`;
  }).join("");
  const nodeMarkup = nodes.map((node) => {
    const cls = node.tier === 0 ? "seed" : node.reviewPriority === "high_confidence_temporal_candidate" ? "p1" : "context";
    const labelText = node.account.length > 17 ? `${node.account.slice(0, 15)}…` : node.account;
    return `<g><circle class="node ${cls}" cx="${node.x}" cy="${node.y}" r="${node.tier === 0 ? 14 : 10}"></circle><text x="${node.x + 15}" y="${node.y + 4}">${labelText}</text></g>`;
  }).join("");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `${edgeMarkup}${nodeMarkup}`;
}

function renderMembers() {
  const group = selectedGroup();
  const node = $("#tab-members");
  if (!node || !group) return;
  const zh = currentLang === "zh";
  const accounts = data.accountsByGroup[group.seed] || [];
  node.innerHTML = table(
    zh ? ["帳號", "優先級", "角色", "MCA", "時間證據", "行為"] : ["Account", "Priority", "Role", "MCA", "Timing", "Behavior"],
    accounts.map((account) => [
      `<strong>${account.account}</strong><small>${zh ? "層級" : "tier"} ${account.tier} · ${account.includeReason}</small>`,
      `<span class="pill ${priorityClass(account.reviewPriority)}">${label(account.reviewPriority)}</span>`,
      `${account.roleZh || account.role || "n/a"}<small>${account.roleReason || ""}</small>`,
      `${fmt(account.mca, 3)}${bar(account.mca, "red")}`,
      `${label(account.bestTemporalLabel)}<small>${label(account.bestTemporalConfidence)}</small>`,
      `${label(account.behaviorProfile)}<small>${zh ? "異常" : "anomaly"} ${fmt(account.anomalyScore, 2)}</small>`,
    ]),
  );
}

function renderPairs() {
  const group = selectedGroup();
  const node = $("#tab-pairs");
  if (!node || !group) return;
  const zh = currentLang === "zh";
  const pairs = (data.pairsByGroup[group.seed] || []).filter((pair) => pair.label !== "no_temporal_sync").slice(0, 50);
  node.innerHTML = table(
    zh ? ["帳號對", "時間同步", "可信度", "共同目標"] : ["Pair", "Timing", "Reliability", "Shared target"],
    pairs.map((pair) => [
      `<strong>${pair.a}</strong><small>↔ ${pair.b}</small>`,
      `${label(pair.label)}<small>${pair.samePost} ${zh ? "篇同文" : "same post"} · ${pair.within30} ${zh ? "次 30 分內" : "within 30m"}</small>`,
      `${label(pair.confidence)}<small>${zh ? "中位延遲" : "median"} ${fmt(pair.medianDelay, 1)} min</small>`,
      fmt(pair.coNegative, 3),
    ]),
  );
}

function renderTargets() {
  const group = selectedGroup();
  const node = $("#tab-targets");
  if (!node || !group) return;
  const zh = currentLang === "zh";
  const targets = data.sharedTargetsByGroup[group.seed] || [];
  node.innerHTML = table(
    zh ? ["鄰近帳號", "共同目標數", "權重", "目標"] : ["Neighbor", "Shared targets", "Weight", "Targets"],
    targets.map((target) => [target.neighbor, target.count, fmt(target.weight, 3), target.targets]),
  );
}

function table(headers, rows) {
  return `<div class="table-wrap"><table><thead><tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function renderGroupsPage() {
  renderGroupList();
  renderGroupSummary();
  renderGraph();
  renderMembers();
  renderPairs();
  renderTargets();
}

function renderAccountsPage() {
  const node = $("#abnormalTable");
  if (!node) return;
  const zh = currentLang === "zh";
  node.innerHTML = table(
    zh ? ["帳號", "出現群組", "優先級", "MCA", "行為類型", "理由"] : ["Account", "Group", "Priority", "MCA", "Behavior", "Reason"],
    data.abnormalAccounts.map((account) => [
      `<strong>${account.account}</strong>`,
      account.seedGroup,
      `<span class="pill ${priorityClass(account.reviewPriority)}">${label(account.reviewPriority)}</span>`,
      `${fmt(account.mca, 3)}${bar(account.mca, "red")}`,
      `${label(account.behaviorProfile)}<small>${zh ? "群集" : "cluster"} ${account.cluster} · ${zh ? "異常" : "anomaly"} ${fmt(account.anomalyScore, 2)}</small>`,
      account.behaviorReason || (zh ? "高風險單一帳號，與群體協同結論分開處理。" : "High-risk account handled separately from group-level coordination."),
    ]),
  );
}

function renderMethodologyPage() {
  const methodNode = $("#methodGrid");
  const signalNode = $("#signalTable");
  if (!methodNode || !signalNode) return;
  const zh = currentLang === "zh";
  const steps = zh
    ? [
        ["01", "MCA seed selection", "用帳號層級風險分數找出值得當入口的 seed，不把 MCA 當最終判決。"],
        ["02", "Graph candidate discovery", "用共同攻擊目標找出 seed 周圍的候選協同群。"],
        ["03", "Temporal verification", "檢查帳號是否在同一貼文下短時間同步到場。"],
        ["04", "Review output", "分成群體協同風險與單一異常帳號兩種輸出。"],
      ]
    : [
        ["01", "MCA seed selection", "Use account-level risk to select seeds, not as a final verdict."],
        ["02", "Graph candidate discovery", "Use shared negative targets to expand candidate coordination groups."],
        ["03", "Temporal verification", "Check whether accounts arrive in the same thread within short time windows."],
        ["04", "Review output", "Separate group-level coordination risk from individual abnormal accounts."],
      ];
  methodNode.innerHTML = steps.map(([num, title, body]) => `<div class="method-step"><span>${num}</span><strong>${title}</strong><p>${body}</p></div>`).join("");
  const signals = zh
    ? [
        ["保留", "co-negative target", "用來找候選群，而不是最終判決。"],
        ["保留", "temporal synchrony", "目前最能區分正反案例的核心驗證訊號。"],
        ["保留", "temporal confidence", "避免單次巧合被過度解讀。"],
        ["移除", "text fingerprint", "在單一主題社群中容易變成 topic noise。"],
        ["移除", "lifecycle / activation window", "正反樣本重疊，像時間背景噪音。"],
      ]
    : [
        ["Kept", "co-negative target", "Used for candidate discovery, not final verdict."],
        ["Kept", "temporal synchrony", "The strongest verification signal in current validation cases."],
        ["Kept", "temporal confidence", "Prevents overreading one-off coincidences."],
        ["Removed", "text fingerprint", "Behaved like topic noise in a single-topic community."],
        ["Removed", "lifecycle / activation window", "Overlapped in positive and negative cases; treated as time-window noise."],
      ];
  signalNode.innerHTML = table(zh ? ["決策", "訊號", "原因"] : ["Decision", "Signal", "Reason"], signals);
}

function bindControls() {
  $$("[data-mode]").forEach((button) => button.addEventListener("click", () => {
    currentMode = button.dataset.mode;
    localStorage.setItem("mca-demo-mode", currentMode);
    renderPage();
  }));
  $$("[data-lang]").forEach((button) => button.addEventListener("click", () => {
    currentLang = button.dataset.lang;
    localStorage.setItem("mca-demo-lang", currentLang);
    renderPage();
  }));
  const sort = $("#groupSort");
  if (sort) sort.addEventListener("change", (event) => {
    currentSort = event.target.value;
    renderGroupList();
  });
  $$(".tabs button").forEach((button) => button.addEventListener("click", () => {
    $$(".tabs button").forEach((b) => b.classList.remove("active"));
    $$(".tab-panel").forEach((p) => p.classList.remove("active"));
    button.classList.add("active");
    $(`#tab-${button.dataset.tab}`).classList.add("active");
  }));
}

function renderPage() {
  setStaticCopy();
  if (page === "overview") {
    renderMetricGrid();
    renderTopCaseCard();
    renderPriorityGroups();
    renderWorkflow();
  }
  if (page === "groups") renderGroupsPage();
  if (page === "accounts") renderAccountsPage();
  if (page === "methodology") renderMethodologyPage();
}

bindControls();
renderPage();
window.addEventListener("resize", () => {
  if (page === "groups") renderGraph();
});
