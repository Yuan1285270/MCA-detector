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
let currentLang = "zh";
let currentMode = "client";

const $ = (selector) => document.querySelector(selector);
const fmt = (value, digits = 2) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "n/a";
  return n.toFixed(digits);
};
const pct = (value) => `${fmt(Number(value) * 100, 0)}%`;

const UI = {
  en: {
    client: {
      appTitle: "Risk Dashboard",
      navGroups: "Risk Groups",
      navEvidence: "Evidence",
      navAbnormal: "Abnormal Accounts",
      navMethod: "Method",
      summaryEyebrow: "Client risk monitoring",
      summaryTitle: "Which account groups need attention first?",
      summaryLede:
        "This dashboard turns social-media activity into a review queue. It highlights risky groups, explains the evidence, and separates group coordination from individual abnormal accounts.",
      guideEyebrow: "How to read this",
      guideTitle: "Start with the business question",
      guide: [
        ["What needs review?", "The ranked groups show where coordinated activity is most likely worth investigating."],
        ["Why is it risky?", "Each group includes network links, timing evidence, and account-level risk signals."],
        ["What should we do next?", "Use the priority list to decide which groups or accounts need analyst review, monitoring, or escalation."],
      ],
      groupListEyebrow: "Prioritized review list",
      groupListTitle: "Risk Groups",
      abnormalEyebrow: "Separate risk stream",
      abnormalTitle: "Individual Abnormal Accounts",
      abnormalNote: "High-risk accounts detected even when group coordination is not confirmed.",
      methodEyebrow: "Client-facing method",
      methodTitle: "From raw activity to risk review",
      flow: [
        ["1. Detect signals", "Measure suspicious language, abnormal behavior, interaction reach, and coordination."],
        ["2. Build risk groups", "Connect accounts that repeatedly appear around the same targets or behaviors."],
        ["3. Verify evidence", "Check whether accounts act together in time and activation window."],
        ["4. Prioritize action", "Separate group-level coordination from individual abnormal accounts for review."],
      ],
    },
    demo: {
      appTitle: "Review Dashboard",
      navGroups: "Groups",
      navEvidence: "Evidence",
      navAbnormal: "Abnormal Accounts",
      navMethod: "Method",
      summaryEyebrow: "Evidence-based suspicious coordination review",
      summaryTitle: "From graph candidates to multi-evidence verification",
      summaryLede:
        "This demo is a review tool. It does not declare accounts guilty; it shows which groups deserve attention and why.",
      guideEyebrow: "How to read this",
      guideTitle: "Start with the question, then inspect the evidence",
      guide: [
        ["1. Is there a group?", "Stage 1 expands from suspicious seed accounts and finds nearby accounts in the coordination graph."],
        ["2. Is the group acting together?", "Stage 2 checks timing evidence and activation-window overlap."],
        ["3. Is it group behavior or one abnormal account?", "The site separates suspicious groups from individual spam/scam-like accounts."],
      ],
      groupListEyebrow: "Stage 1 + validation",
      groupListTitle: "Suspicious Coordination Groups",
      abnormalEyebrow: "Separate output stream",
      abnormalTitle: "Individual Abnormal Accounts",
      abnormalNote: "Detected as high-risk accounts even when group coordination is not confirmed.",
      methodEyebrow: "Pipeline story",
      methodTitle: "How the system makes a reviewable result",
      flow: [
        ["1. MCA seed selection", "Find accounts with high manipulation, coordination, reach, and automation signals."],
        ["2. Graph expansion", "Expand from seeds through co-negative and related graph evidence to form candidate groups."],
        ["3. Multi-evidence verification", "Check temporal synchrony and activation window overlap."],
        ["4. Review output", "Separate suspicious coordination groups from individual abnormal manipulation accounts."],
      ],
    },
  },
  zh: {
    client: {
      appTitle: "風險監測面板",
      navGroups: "風險群組",
      navEvidence: "證據",
      navAbnormal: "異常帳號",
      navMethod: "方法",
      summaryEyebrow: "客戶風險監測",
      summaryTitle: "哪些帳號群需要優先處理？",
      summaryLede:
        "這個面板把社群資料轉成可審查的風險清單：先看高風險群組，再看它為什麼可疑，並把群體協同和單一異常帳號分開。",
      guideEyebrow: "閱讀方式",
      guideTitle: "先看商業問題，再看證據",
      guide: [
        ["哪些對象要先看？", "左側排名列出最值得優先審查的風險群組。"],
        ["為什麼它可疑？", "每個群組會顯示帳號關係、時間同步，以及帳號本身的風險特徵。"],
        ["下一步要做什麼？", "依照優先級決定要人工審查、持續監控，或升級處理。"],
      ],
      groupListEyebrow: "優先審查清單",
      groupListTitle: "風險群組",
      abnormalEyebrow: "獨立風險輸出",
      abnormalTitle: "單一異常帳號",
      abnormalNote: "即使沒有確認群體協同，仍可能有值得處理的高風險帳號。",
      methodEyebrow: "客戶版方法摘要",
      methodTitle: "從原始互動到風險審查",
      flow: [
        ["1. 偵測風險訊號", "衡量操縱性語言、異常行為、互動觸及和協同跡象。"],
        ["2. 建立風險群組", "把反覆出現在相同目標或相似行為附近的帳號連起來。"],
        ["3. 驗證證據", "檢查帳號是否在時間同步和活躍窗口上呈現共同痕跡。"],
        ["4. 排定處理優先級", "把群體協同風險和單一異常帳號分開，方便後續處置。"],
      ],
    },
    demo: {
      appTitle: "研究展示面板",
      navGroups: "群組",
      navEvidence: "證據",
      navAbnormal: "異常帳號",
      navMethod: "方法",
      summaryEyebrow: "證據導向的協同審查",
      summaryTitle: "從圖結構候選群到多證據驗證",
      summaryLede:
        "這是研究 demo，不直接宣判帳號有罪，而是說明哪些群組值得被檢查，以及背後有哪些證據。",
      guideEyebrow: "閱讀方式",
      guideTitle: "先看問題，再檢查證據",
      guide: [
        ["1. 有沒有形成群？", "Stage 1 從高風險 seed 出發，在協同圖中找出附近帳號。"],
        ["2. 這群人有沒有一起行動？", "Stage 2 檢查時間同步和活躍窗口重疊。"],
        ["3. 是群體協同還是單一異常？", "網站把可疑群組和 spam/scam-like 的單一帳號分開呈現。"],
      ],
      groupListEyebrow: "Stage 1 + validation",
      groupListTitle: "可疑協同群組",
      abnormalEyebrow: "獨立輸出",
      abnormalTitle: "單一異常帳號",
      abnormalNote: "這些帳號本身高風險，但不一定已確認屬於協同群體。",
      methodEyebrow: "Pipeline 故事",
      methodTitle: "系統如何產生可審查結果",
      flow: [
        ["1. MCA seed selection", "找出操縱、協同、觸及和自動化訊號較高的帳號。"],
        ["2. Graph expansion", "從 seed 沿著 co-negative 等圖邊擴張成候選群。"],
        ["3. Multi-evidence verification", "檢查 temporal synchrony 和 activation window overlap。"],
        ["4. Review output", "分開輸出可疑協同群組與單一異常操縱帳號。"],
      ],
    },
  },
};

const LABELS = {
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
};

function ui() {
  return UI[currentLang][currentMode];
}

function setText(id, value) {
  const el = $(`#${id}`);
  if (el) el.textContent = value;
}

function priorityLabel(value) {
  const raw = String(value || "");
  return LABELS[currentLang][raw] || raw.replaceAll("_", " ");
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

function renderChrome() {
  const copy = ui();
  document.documentElement.lang = currentLang === "zh" ? "zh-Hant" : "en";
  document.body.dataset.mode = currentMode;
  document.body.dataset.lang = currentLang;

  [
    "appTitle",
    "navGroups",
    "navEvidence",
    "navAbnormal",
    "navMethod",
    "summaryEyebrow",
    "summaryTitle",
    "summaryLede",
    "guideEyebrow",
    "guideTitle",
    "groupListEyebrow",
    "groupListTitle",
    "abnormalEyebrow",
    "abnormalTitle",
    "abnormalNote",
    "methodEyebrow",
    "methodTitle",
  ].forEach((key) => setText(key, copy[key]));

  setText("legendSeed", currentLang === "zh" ? "Seed 帳號" : "Seed");
  setText("legendP1", currentLang === "zh" ? "高優先" : "High priority");
  setText("legendP2", currentLang === "zh" ? "待審查" : "Review");
  setText("legendGraph", currentLang === "zh" ? "共同目標" : "Shared target");
  setText("legendTiming", currentLang === "zh" ? "時間連結" : "Timing link");
  setText("tabMembersButton", currentLang === "zh" ? "帳號" : "Accounts");
  setText("tabPairsButton", currentLang === "zh" ? "行為連結" : "Behavior Links");
  setText("tabTargetsButton", currentLang === "zh" ? "共同目標" : "Shared Targets");
  setText("fitGraph", currentLang === "zh" ? "重置" : "Reset");

  const sortLabels = currentLang === "zh"
    ? ["排名", "高優先帳號", "穩定時間證據", "最高 MCA"]
    : ["Rank", "P1 accounts", "Robust timing", "Max MCA"];
  [...$("#groupSort").options].forEach((option, index) => {
    option.textContent = sortLabels[index] || option.textContent;
  });

  $("#guideGrid").innerHTML = copy.guide
    .map(([title, body]) => `
      <div class="guide-card">
        <strong>${title}</strong>
        <span>${body}</span>
      </div>
    `)
    .join("");

  $("#flowGrid").innerHTML = copy.flow
    .map(([title, body]) => `
      <div class="flow-step">
        <strong>${title}</strong>
        <span>${body}</span>
      </div>
    `)
    .join("");

  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === currentMode);
  });
  document.querySelectorAll("[data-lang]").forEach((button) => {
    button.classList.toggle("active", button.dataset.lang === currentLang);
  });
}

function renderMeta() {
  const meta = data.metadata || {};
  const labels = currentLang === "zh"
    ? ["資料來源", "群組數", "候選帳號列", "帳號對證據"]
    : ["Source", "Groups", "Candidate rows", "Pair evidence"];
  $("#runMeta").innerHTML = `
    <div><strong>${labels[0]}</strong><br>${meta.source || "pipeline output"}</div>
    <div><strong>${labels[1]}</strong><br>${meta.groupCount || data.groups.length}</div>
    <div><strong>${labels[2]}</strong><br>${meta.candidateAccountRows || 0}</div>
    <div><strong>${labels[3]}</strong><br>${meta.pairEvidenceRows || 0}</div>
  `;
}

function renderMetrics() {
  const groups = data.groups;
  const p1 = groups.reduce((sum, g) => sum + Number(g.p1 || 0), 0);
  const reliable = groups.reduce((sum, g) => sum + Number(g.reliableTemporalPairs || 0), 0);
  const outliers = data.abnormalAccounts.length;
  const totalMembers = groups.reduce((sum, g) => sum + Number(g.memberCount || 0), 0);
  const labels = currentLang === "zh"
    ? ["已排序風險群組", "擴張後帳號成員", "高優先審查帳號", "可信時間證據對", "單一異常帳號", "最高排名群組"]
    : ["ranked risk groups", "expanded group members", "high-priority accounts", "reliable timing pairs", "individual abnormal accounts", "top ranked group"];
  $("#metricGrid").innerHTML = `
    <div class="metric"><strong>${groups.length}</strong><span>${labels[0]}</span></div>
    <div class="metric"><strong>${totalMembers}</strong><span>${labels[1]}</span></div>
    <div class="metric"><strong>${p1}</strong><span>${labels[2]}</span></div>
    <div class="metric"><strong>${reliable}</strong><span>${labels[3]}</span></div>
    <div class="metric"><strong>${outliers}</strong><span>${labels[4]}</span></div>
    <div class="metric"><strong>${data.groups[0]?.seed || "n/a"}</strong><span>${labels[5]}</span></div>
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
        ? currentLang === "zh"
          ? `${group.reliableTemporalPairs} 組可信時間證據`
          : `${group.reliableTemporalPairs} reliable timing pair${group.reliableTemporalPairs === 1 ? "" : "s"}`
        : currentLang === "zh"
          ? `${group.moderateReviewTemporalPairs} 組待審時間證據`
          : `${group.moderateReviewTemporalPairs} reviewable timing pair${group.moderateReviewTemporalPairs === 1 ? "" : "s"}`;
      const highPriority = currentLang === "zh"
        ? `${group.p1} 個高優先帳號`
        : `${group.p1} high-priority member${group.p1 === 1 ? "" : "s"}`;
      return `
        <button class="group-row ${active}" type="button" data-seed="${group.seed}">
          <div class="group-row-top">
            <span class="group-row-title">${group.seed}</span>
            <span class="rank">#${group.rank}</span>
          </div>
          <p class="group-row-summary">${evidenceLine}; ${highPriority}.</p>
          <div class="mini-metrics">
            <span class="chip hot">P1 ${group.p1}</span>
            <span class="chip">${currentLang === "zh" ? "成員" : "members"} ${group.memberCount}</span>
            <span class="chip sync">${currentLang === "zh" ? "可信" : "reliable"} ${group.reliableTemporalPairs}</span>
            <span class="chip outlier">${currentLang === "zh" ? "異常" : "outliers"} ${group.extremeOutliers}</span>
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
  $("#selectedGroupTitle").textContent = `${group.seed} ${currentLang === "zh" ? "群組" : "group"}`;
  $("#groupPriorityPill").textContent = priorityLabel(group.priority);
  const statLabels = currentLang === "zh"
    ? ["擴張後成員", "高優先帳號", "可信時間證據", "共同攻擊目標", "最高 MCA", "自動化異常比例"]
    : ["members after expansion", "P1 candidates", "reliable timing pairs", "shared negative targets", "max MCA score", "automation anomaly fraction"];
  $("#detailStats").innerHTML = `
    <div class="stat-cell"><strong>${group.memberCount}</strong><span>${statLabels[0]}</span></div>
    <div class="stat-cell"><strong>${group.p1}</strong><span>${statLabels[1]}</span></div>
    <div class="stat-cell"><strong>${group.reliableTemporalPairs}</strong><span>${statLabels[2]}</span></div>
    <div class="stat-cell"><strong>${group.sharedNegativeTargets}</strong><span>${statLabels[3]}</span></div>
    <div class="stat-cell"><strong>${fmt(group.maxMca, 2)}</strong><span>${statLabels[4]}</span></div>
    <div class="stat-cell"><strong>${pct(group.automationFraction)}</strong><span>${statLabels[5]}</span></div>
  `;
  $("#groupStory").innerHTML = renderGroupStory(group);
}

function renderGroupStory(group) {
  const isClient = currentMode === "client";
  const zh = currentLang === "zh";
  const coordination = group.tier1CoNegative > 0
    ? zh
      ? `系統在 seed 附近找到 ${group.tier1CoNegative} 條直接共同目標連結。`
      : `The system found ${group.tier1CoNegative} direct shared-target links around the seed.`
    : zh
      ? `這個群組主要來自較弱或間接的圖結構線索。`
      : `This group mainly comes from weaker or indirect graph context.`;
  const temporal = group.robustTemporalPairs > 0
    ? zh
      ? `其中有 ${group.robustTemporalPairs} 組穩定時間證據，代表這群帳號更像是一起行動。`
      : `${group.robustTemporalPairs} robust timing pair${group.robustTemporalPairs === 1 ? "" : "s"} suggest the group is more likely acting together.`
    : group.moderateReviewTemporalPairs > 0
      ? zh
        ? `另有 ${group.moderateReviewTemporalPairs} 組值得審查的時間證據，需要人工確認。`
        : `${group.moderateReviewTemporalPairs} reviewable timing pair${group.moderateReviewTemporalPairs === 1 ? "" : "s"} should be inspected by an analyst.`
      : zh
        ? `目前沒有穩定時間證據，因此它比較像候選風險群。`
        : `No reliable timing evidence was found, so this remains mostly a candidate risk group.`;
  const caution = group.fragileTemporalPairs > 0
    ? zh
      ? `${group.fragileTemporalPairs} 組時間證據偏脆弱，不應直接當成確認結論。`
      : `${group.fragileTemporalPairs} timing pair${group.fragileTemporalPairs === 1 ? "" : "s"} are fragile and should not be overread.`
    : zh
      ? `目前沒有由脆弱時間證據主導的問題。`
      : `No fragile timing pair dominates the group summary.`;
  const mainTitle = isClient
    ? zh ? "客戶版重點" : "Client takeaway"
    : zh ? "研究版重點" : "Plain-language takeaway";
  const cautionTitle = zh ? "注意事項" : "What to be careful about";
  const cautionTail = isClient
    ? zh ? "建議作為優先審查對象，而不是直接判定違規。" : "Treat this as a priority for review, not an automatic enforcement decision."
    : zh ? "這個 dashboard 表示「值得審查」，不是「已確認網軍」。" : "This dashboard says “review this,” not “confirmed bot network.”";
  return `
    <div class="story-card primary">
      <strong>${mainTitle}</strong>
      <span>${coordination} ${temporal}</span>
    </div>
    <div class="story-card">
      <strong>${cautionTitle}</strong>
      <span>${caution} ${cautionTail}</span>
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
  const intro = currentMode === "client"
    ? currentLang === "zh"
      ? ["帳號依照處理優先級排序。", "高優先帳號代表同時有群體、行為或時間證據；背景成員保留在表中，方便理解整個群組。"]
      : ["Accounts are sorted by action priority.", "High-priority accounts have stronger group, behavior, or timing evidence. Context members remain visible so the group is explainable."]
    : currentLang === "zh"
      ? ["Members 依照 review priority 排序。", "P1 表示有較強的 supporting evidence；context members 用來保持圖結構可解釋。"]
      : ["Members are sorted by review priority.", "P1 means the account has stronger supporting evidence. Context members are shown so the graph remains explainable."];
  const headers = currentLang === "zh"
    ? ["帳號", "為什麼顯示", "群內角色", "MCA", "時間證據", "行為"]
    : ["Account", "Why it is shown", "Role", "MCA", "Timing", "Behavior"];
  $("#tab-members").innerHTML = `
    <div class="tab-intro">
      <strong>${intro[0]}</strong>
      <span>${intro[1]}</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>${headers[0]}</th>
            <th>${headers[1]}</th>
            <th>${headers[2]}</th>
            <th>${headers[3]}</th>
            <th>${headers[4]}</th>
            <th>${headers[5]}</th>
          </tr>
        </thead>
        <tbody>
          ${accounts
            .map(
              (account) => `
                <tr>
                  <td class="account-name">${account.account}<br><span class="muted">${currentLang === "zh" ? "層級" : "tier"} ${account.tier} · ${account.includeReason}</span></td>
                  <td><span class="chip ${reviewPriorityClass(account.reviewPriority)}">${priorityLabel(account.reviewPriority)}</span></td>
                  <td>${account.roleZh || account.role || "n/a"}<br><span class="muted">${account.roleReason || ""}</span></td>
                  <td>${fmt(account.mca, 3)}${scoreBar(account.mca, "red")}</td>
                  <td>${priorityLabel(account.bestTemporalLabel)}<br><span class="muted">${priorityLabel(account.bestTemporalConfidence)}</span></td>
                  <td>${priorityLabel(account.behaviorProfile || "unknown")}<br><span class="muted">${currentLang === "zh" ? "異常分數" : "anomaly"} ${fmt(account.anomalyScore, 2)}</span></td>
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
  const usefulPairs = pairs.filter((pair) => pair.label !== "no_temporal_sync" || pair.activationOverlap > 0.5).slice(0, 50);
  const intro = currentMode === "client"
    ? currentLang === "zh"
      ? ["行為連結說明兩個帳號為什麼被放在一起看。", "時間同步越穩定，越值得優先人工審查；活躍窗口是輔助證據。"]
      : ["Behavior links explain why two accounts should be reviewed together.", "Stable timing evidence increases review priority; activation overlap is a supporting signal."]
    : currentLang === "zh"
      ? ["Pair evidence 解釋兩個帳號為什麼被連起來。", "Timing sync 看是否一起出現；reliability 看這個同步是否穩定。"]
      : ["Pair evidence explains why two accounts are connected.", "Timing sync asks whether they appeared together; reliability asks whether that pattern is stable enough to trust."];
  const headers = currentLang === "zh"
    ? ["帳號對", "時間同步", "可信度", "活躍重疊", "共同目標"]
    : ["Pair", "Timing", "Reliability", "Activation overlap", "Shared target"];
  $("#tab-pairs").innerHTML = `
    <div class="tab-intro">
      <strong>${intro[0]}</strong>
      <span>${intro[1]}</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>${headers[0]}</th>
            <th>${headers[1]}</th>
            <th>${headers[2]}</th>
            <th>${headers[3]}</th>
            <th>${headers[4]}</th>
          </tr>
        </thead>
        <tbody>
          ${usefulPairs
            .map(
              (pair) => `
                <tr>
                  <td class="account-name">${pair.a}<br><span class="muted">↔ ${pair.b}</span></td>
                  <td>${priorityLabel(pair.label)}<br><span class="muted">${pair.samePost} ${currentLang === "zh" ? "篇同文" : "same post"} · ${pair.within30} ${currentLang === "zh" ? "次 30 分內" : "within 30m"}</span></td>
                  <td>${priorityLabel(pair.confidence)}<br><span class="muted">${currentLang === "zh" ? "中位延遲" : "median"} ${fmt(pair.medianDelay, 1)} min</span></td>
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
  const intro = currentMode === "client"
    ? currentLang === "zh"
      ? ["共同目標是形成風險群組的主要原因。", "這代表帳號反覆出現在相同爭議對象附近，但仍需要其他證據輔助判斷。"]
      : ["Shared targets are the main reason these accounts entered the same risk group.", "This shows overlap around the same targets, but it still needs supporting evidence."]
    : currentLang === "zh"
      ? ["Shared targets 是 expansion 的圖結構理由。", "這些帳號對重疊目標有負向互動；它適合 discovery，但不能單獨證明操縱。"]
      : ["Shared targets are the graph reason for expansion.", "These accounts reacted negatively toward overlapping targets. This is useful for discovery, but not enough by itself to prove manipulation."];
  const headers = currentLang === "zh"
    ? ["鄰近帳號", "共同目標數", "權重", "目標"]
    : ["Neighbor", "Shared targets", "Weight", "Targets"];
  $("#tab-targets").innerHTML = `
    <div class="tab-intro">
      <strong>${intro[0]}</strong>
      <span>${intro[1]}</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>${headers[0]}</th>
            <th>${headers[1]}</th>
            <th>${headers[2]}</th>
            <th>${headers[3]}</th>
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
  const intro = currentMode === "client"
    ? currentLang === "zh"
      ? ["這是另一種風險輸出。", "有些帳號本身像 spam、scam 或高異常行為，但不一定屬於群體協同；因此獨立呈現。"]
      : ["This is a separate risk stream.", "Some accounts look spam-like, scam-like, or behaviorally abnormal even without confirmed group coordination."]
    : currentLang === "zh"
      ? ["這是獨立結果類型。", "帳號可以是異常或垃圾推廣帳，但不一定是協同群的一部分；分開呈現能避免過度宣稱。"]
      : ["This is a separate result type.", "An account can be abnormal or spam-like without being part of a coordinated group. Keeping this separate prevents overclaiming."];
  const headers = currentLang === "zh"
    ? ["帳號", "出現在哪個群", "優先級", "MCA", "行為類型", "為什麼重要"]
    : ["Account", "Found in group", "Priority", "MCA", "Behavior profile", "Why it matters"];
  $("#abnormalTable").innerHTML = `
    <div class="tab-intro standalone">
      <strong>${intro[0]}</strong>
      <span>${intro[1]}</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>${headers[0]}</th>
            <th>${headers[1]}</th>
            <th>${headers[2]}</th>
            <th>${headers[3]}</th>
            <th>${headers[4]}</th>
            <th>${headers[5]}</th>
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
                  <td>${priorityLabel(account.behaviorProfile || "unknown")}<br><span class="muted">${currentLang === "zh" ? "群集" : "cluster"} ${account.cluster} · ${currentLang === "zh" ? "異常" : "anomaly"} ${fmt(account.anomalyScore, 2)}</span></td>
                  <td>${account.behaviorReason || (currentLang === "zh" ? "高風險單一帳號，與群體協同結論分開處理。" : "High-risk individual account separated from group-level coordination claims.")}</td>
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

function bindControls() {
  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      currentMode = button.dataset.mode;
      localStorage.setItem("mca-demo-mode", currentMode);
      renderAll();
    });
  });
  document.querySelectorAll("[data-lang]").forEach((button) => {
    button.addEventListener("click", () => {
      currentLang = button.dataset.lang;
      localStorage.setItem("mca-demo-lang", currentLang);
      renderAll();
    });
  });
}

function renderAll() {
  renderChrome();
  renderMeta();
  renderMetrics();
  renderGroups();
  renderGroupHeader();
  renderGraph();
  renderMembers();
  renderPairs();
  renderTargets();
  renderAbnormalAccounts();
}

function init() {
  currentMode = localStorage.getItem("mca-demo-mode") || "client";
  currentLang = localStorage.getItem("mca-demo-lang") || "zh";
  renderAll();
  bindTabs();
  bindControls();
  $("#groupSort").addEventListener("change", (event) => {
    currentSort = event.target.value;
    renderGroups();
  });
  $("#fitGraph").addEventListener("click", renderGraph);
  window.addEventListener("resize", renderGraph);
}

init();
