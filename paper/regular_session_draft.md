# 結合操縱性內容分析與圖擴張之社群協同行為偵測系統

學生：林琮原、張鈞浩、賴嘉安  
指導教授：李俊宏  
單位：逢甲大學資訊工程學系

## 摘要

社群平台上的操縱式資訊操作不一定表現為單一帳號的自動化行為。許多可疑帳號可能在內容、互動目標與發文時間上呈現群體協同，但若只依賴帳號層級分數，容易將一般高活躍使用者或立場相近的社群成員誤判為網軍。本文提出一套可重現且可人工審查的社群協同行為偵測流程。系統首先使用大型語言模型分析加密貨幣討論中的貼文修辭與留言立場，接著建立帳號層級特徵矩陣與多層 adjacency graph。第一階段以 Manipulative Coordination Account（MCA）分數選出高風險 seed accounts，並沿共同負向目標圖進行 seed expansion，形成候選協同群。第二階段不直接將共同攻擊目標視為網軍證據，而是以同一討論串中的短時間共同出現作為 temporal synchrony verification，降低「同立場但獨立行動」造成的誤判。初步實驗在 Reddit 加密貨幣資料上產生 20 個候選 seed groups、172 個不重複候選帳號，並辨識出 13 組 strong temporal sync pairs 與 8 組 moderate temporal sync pairs。案例分析顯示，`harvested` 群組同時具有共同攻擊目標與穩定時間同步，而 `JG87919` 類型案例雖具有共同負向目標，卻缺乏有效時間同步，說明本文方法可將候選協同群進一步分層為高優先審查目標，而非直接宣稱其為真實網軍。

**關鍵字：** 社群媒體分析、協同行為偵測、操縱式修辭、圖擴張、時間同步、Reddit

## 1. 前言

社群媒體已成為公共議題、金融討論與投資情緒形成的重要場域。然而，社群平台也可能被自動化帳號、協同帳號或操縱式內容用來放大特定敘事。既有 social bot detection 研究指出，機器人帳號不只會自動產生內容，也可能透過社交網路、時間行為、擴散模式與情緒表達影響人類使用者 [1]。然而，帳號是否「像機器人」並不等同於帳號是否屬於「協同行動」。在加密貨幣社群中，正常使用者也可能高度活躍、語氣強烈，並共同批評同一批反對者。因此，若僅依據帳號分數或共同攻擊目標，容易產生 false positive。

本文關注的問題不是直接證明某帳號為真實世界中的網軍，而是設計一個可審查的偵測流程：先找出值得調查的候選群，再提供群內關係、內容操縱性與時間同步證據，供研究者或平台審查者判斷。本文的主要貢獻如下：

1. 提出一個兩階段 suspicious coordination pipeline，將帳號層級 MCA 分數、圖擴張與 temporal verification 分開處理。
2. 將共同負向目標圖用於候選群發現，而非直接當成最終網軍判決。
3. 以短時間同串出現作為第二階段驗證訊號，區分「共同立場」與「更可疑的同步行動」。
4. 建立可輸出 group summary、pair evidence 與 account roles 的審查表格，並將 demo website 與分析 pipeline 明確分離。

## 2. 相關研究

社群機器人偵測研究最早多聚焦於帳號層級分類。Ferrara 等人指出，social bots 會嘗試模仿人類行為，並可能在政治、經濟與公共議題中產生影響 [1]。Varol 等人進一步使用帳號 metadata、內容、情緒、網路與時間序列等上千個特徵建立 Twitter bot detection framework [2]。此類方法適合判斷單一帳號是否自動化，但對「多帳號是否共同操作」的解釋力有限。

另一類研究強調群體與行為序列。Cresci 等人提出 Social Fingerprinting，透過帳號行為序列的相似性偵測 spambot groups [3]。Mazza 等人的 RTbust 則利用 retweet time series 之時間模式，從大規模 retweet 資料中找出可疑 botnets [4]。這些研究顯示，群體行為與時間模式對偵測協同帳號具有價值。

近年的 coordinated behavior detection 更直接處理「共同動作」問題。Pacheco 等人提出以 shared behavioral traces 建立 coordination networks，並在多個案例中使用 identity、image、hashtag sequence、retweet 與 temporal patterns 偵測協同帳號 [5]。Graham 等人提出 Coordination Network Toolkit，主張以 weighted directed multigraph 表示多種 coordinated behavior，並指出 coordination analysis 對理解 online influence、digital astroturfing 與 online activism 具有重要性 [6]。在 Reddit 場景中，Saeed 等人的 TROLLMAGNIFIER 以已知 troll accounts 為基礎，發現同一操控來源帳號常呈現 loose coordination、相似主題與 temporal synchronization [7]。

本文亦使用 stance detection 的概念。Mohammad 等人的 SemEval-2016 stance detection task 將文本相對於特定 target 的立場分為支持、反對或中立 [8]。本文延伸此想法，將留言對原貼文作者的支持或反對關係轉為 signed account-to-account edges，並進一步建立共同負向目標圖。

## 3. 系統架構

本文系統分為分析 pipeline 與展示層兩部分。分析 pipeline 負責從原始 Reddit 貼文與留言產生可審查證據；demo website 只讀取 pipeline output 並呈現結果，不參與分數計算或判決。

```text
raw posts/comments
  -> LLM post/comment analysis
  -> account feature matrix + adjacency graphs
  -> MCA seed ranking
  -> coordination expansion
  -> temporal verification
  -> group summary + account roles
```

### 3.1 LLM 內容與立場分析

系統先對貼文進行情緒與操縱式修辭分析，輸出 `sentiment_score`、`manipulative_rhetoric_score` 與 `rhetoric_tags`。留言分析則判斷留言對原貼文的立場，輸出 `feedback_label`、`feedback_score` 與 `edge_weight`。其中 `feedback_label` 包含 supportive、oppositional、neutral、mixed 與 unclear。這使每則留言可轉為：

```text
comment_author -> post_author
```

若留言為反對或攻擊，則形成負向互動；若留言支持或補充，則形成正向互動。

### 3.2 帳號特徵與 MCA 分數

MCA 分數不是最終判決，而是 seed selection 與 review priority。本文使用四類訊號：

| Signal | 主要特徵 | 意義 |
|---|---|---|
| Manipulative | 平均修辭分數、非中立貼文比例、反對立場比例 | 帳號是否常使用操縱性語言或攻擊性立場 |
| Coordinative | co-target、co-negative-target、trigger-response frequency | 帳號是否與其他帳號指向相似目標或有固定回應模式 |
| Interaction reach | outgoing volume、incoming attention、interaction breadth | 帳號互動範圍與影響力 |
| Automatic behavior | Isolation Forest anomaly score | 帳號活動是否偏離一般行為分布 |

目前主權重設定為 manipulative 0.30、coordinative 0.35、interaction reach 0.15、automatic behavior 0.20。由於缺乏大規模標註資料，本文不宣稱此權重為最佳化結果，而是將其作為可解釋的初始審查排序。

### 3.3 Adjacency Graph Construction

系統建立多層 account-level adjacency graphs，包括互動次數、signed interaction、positive/negative interaction、trigger-response、co-target 與 co-negative-target。其中本文的候選群發現主要使用 `A_co_negative_target`：

```text
A_co_negative_target[i,j] = cosine_similarity(negative_target_profile_i, negative_target_profile_j)
```

此圖回答的是：「兩個帳號是否常常反對或攻擊同一批作者？」然而，這只是協同的必要線索，不是充分證據。正常使用者也可能因為立場相同而批評同一批人。因此，本文將此圖用於 candidate discovery，而非 final verdict。

## 4. 候選群擴張與驗證方法

### 4.1 Stage 1: Seed Expansion

第一階段從 MCA top accounts 選出 seed accounts，並沿 graph layers 擴張候選群。擴張規則採分層制，而非黑箱加權總分：

| Tier | 納入條件 | 解釋 |
|---|---|---|
| Tier 1 | co-negative-target >= 0.20 | 與 seed 共同攻擊目標高度重疊 |
| Tier 2 | tag similarity >= 0.90 且有結構訊號支持 | 語言/修辭相近但需圖證據輔助 |
| Tier 3 | trigger-response >= 0.50 且有 co-negative 或 tag support | 有穩定回應模式且具結構支持 |
| Tier 4 | 2-hop co-negative，且連到至少兩個已納入成員 | 從已納入成員延伸外圍候選 |

`co_target` 只作輔助，不單獨納入。此設計目的在於保留每個候選人的納入理由，使輸出可人工審查。

### 4.2 Stage 2: Temporal Verification

第二階段檢查同一群內帳號是否在同一篇 post 下短時間共同出現。本文使用兩個時間窗：

```text
strong_temporal_sync   = 至少一次同串留言時間差 < 5 分鐘
moderate_temporal_sync = 至少兩次同串留言時間差 < 30 分鐘
weak_temporal_overlap  = 有同串共現，但沒有短時間同步
no_temporal_sync       = 沒有同串共現
```

另外，為避免熱門貼文造成單次巧合同步，系統加入 temporal confidence：

```text
robust          = 重複或跨多篇貼文的短時間同步
moderate_review = 有可審查同步證據，但仍需人工檢查
fragile         = 單次或典型延遲過長，只作輔助
none            = 無可用同步證據
```

早期版本曾測試 `text_fingerprint_distance` 與 `account_lifecycle_overlap`，但在本資料集上兩者難以區分同操控者與同主題獨立使用者。加密貨幣社群用詞高度相似，使 TF-IDF 類文字距離容易受主題影響；帳號生命週期也可能因操控者刻意混用新舊帳號而失效。因此本文正式版本只保留 temporal synchrony 與 temporal confidence 作為 Stage 2 evidence。

## 5. 實驗與案例分析

### 5.1 資料與輸出

本文以 Reddit 加密貨幣相關貼文與留言為分析對象。當前 pipeline 以 MCA ranking 選出 20 個 seed accounts，經 seed expansion 後形成 20 個候選群。總計有 178 筆 group membership，去除重複後為 172 個候選帳號。Stage 2 pair-level temporal verification 共找出 13 組 strong temporal sync pairs 與 8 組 moderate temporal sync pairs；其中 3 組達到 robust confidence，44 組為 moderate review confidence。

表 2 顯示前五個候選群摘要。此排序為 review priority，不代表最終網軍判決。

| Rank | Seed group | Members | P1 | P2 | Strong sync | Moderate sync | Robust pairs | Shared negative targets |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | lol_camis | 23 | 2 | 3 | 4 | 0 | 0 | 25 |
| 2 | BtcKing1111 | 11 | 2 | 3 | 2 | 1 | 0 | 13 |
| 3 | tzacPACO | 6 | 2 | 1 | 0 | 3 | 1 | 8 |
| 4 | harvested | 8 | 2 | 0 | 1 | 0 | 1 | 11 |
| 5 | iPurchaseBitcoin | 7 | 2 | 0 | 1 | 0 | 1 | 6 |

### 5.2 Case Study: harvested 群組

`harvested` 群組包含 8 名成員，在 Stage 1 中具有 7 名 Tier 1 co-negative direct members、9 條群內協同邊與 11 個 shared negative targets。Stage 2 中，`NectarineDirect936` 與 `harvested` 在 17 篇共同貼文下出現，包含 2 次 5 分鐘內同步與 4 次 30 分鐘內同步，因此被標為 `strong_temporal_sync / robust`。此案例顯示，當共同攻擊目標與重複短時間同步同時存在時，該群可被列為高優先審查對象。

### 5.3 Negative Control: JG87919 類型案例

`JG87919` 類型案例說明 Stage 1 不能被當成最終判斷。該 seed expansion 形成 15 名成員、34 條群內協同邊與 20 個 shared negative targets，表面上具有強烈共同攻擊目標。然而人工檢查與時間同步分析顯示，該群更像是同一社群內立場相近的獨立使用者，而非同操控來源。此案例支持本文核心設計：co-negative-target 適合找候選群，但必須透過 Stage 2 temporal verification 或人工審查再分級。

### 5.4 訊號刪減結果

本文實驗過文字指紋與帳號生命週期兩種補充訊號，但最後將其移出正式 evidence。文字指紋在單一主題社群中容易測到「大家都在講 Bitcoin」而非「同一操控者」；生命週期與活躍窗口則容易測到同社群使用者自然重疊。移除這些噪音後，系統主線更清楚：Stage 1 找候選協同群，Stage 2 檢查短時間同步。

## 6. 討論

本文方法的主要優點是可解釋與可審查。MCA 分數提供 seed selection，但不直接宣稱帳號有罪；co-negative-target graph 找出共同攻擊目標，但不將其視為網軍充分證據；temporal synchrony 則作為第二階段過濾器，協助區分同立場活躍用戶與更可疑的同步行動。

不過，本文仍有三項限制。第一，目前缺乏大規模 ground truth，因此結果應解讀為 review candidates，而非 supervised accuracy。第二，時間同步可能受到熱門貼文與高活躍使用者影響，因此本文加入 temporal confidence，但仍需要人工檢查。第三，資料來源集中於 Reddit 加密貨幣社群，未來若應用到其他平台，需依平台互動形式重新定義 action traces，例如轉推、分享連結、按讚或共同 hashtag。

本文的 demo website 只作為展示層，將 pipeline output 轉為 client-facing dashboard。它不重新計算 MCA score、不做 group discovery，也不作 final verdict。此分離能確保研究方法可獨立重跑，也讓展示介面未來可以重新設計而不影響核心分析。

## 7. 結論

本文提出一套結合 LLM 內容分析、MCA seed ranking、graph-based seed expansion 與 temporal synchrony verification 的社群協同行為偵測流程。相較於單純帳號分數排序，本文將偵測目標從「誰最像可疑帳號」推進到「哪些帳號像是在一起行動」。初步結果顯示，系統能從大量帳號中縮小至可審查的候選群，並透過時間同步證據進一步排序審查優先級。未來工作將擴充跨平台 action traces、建立更完整的人工標註集，並評估 temporal threshold 在不同社群與平台上的穩定性。

## 參考文獻

[1] E. Ferrara, O. Varol, C. Davis, F. Menczer, and A. Flammini, “The Rise of Social Bots,” *Communications of the ACM*, vol. 59, no. 7, pp. 96–104, 2016.

[2] O. Varol, E. Ferrara, C. Davis, F. Menczer, and A. Flammini, “Online Human-Bot Interactions: Detection, Estimation, and Characterization,” *Proceedings of the International AAAI Conference on Web and Social Media*, vol. 11, no. 1, pp. 280–289, 2017.

[3] S. Cresci, R. Di Pietro, M. Petrocchi, A. Spognardi, and M. Tesconi, “Social Fingerprinting: Detection of Spambot Groups Through DNA-Inspired Behavioral Modeling,” *IEEE Transactions on Dependable and Secure Computing*, vol. 15, no. 4, pp. 561–576, 2018.

[4] M. Mazza, S. Cresci, M. Avvenuti, W. Quattrociocchi, and M. Tesconi, “RTbust: Exploiting Temporal Patterns for Botnet Detection on Twitter,” arXiv:1902.04506, 2019.

[5] D. Pacheco, P.-M. Hui, C. Torres-Lugo, B. T. Truong, A. Flammini, and F. Menczer, “Uncovering Coordinated Networks on Social Media: Methods and Case Studies,” *Proceedings of the International AAAI Conference on Web and Social Media*, vol. 15, no. 1, pp. 455–466, 2021.

[6] T. Graham, S. Hames, and E. Alpert, “The Coordination Network Toolkit: A Framework for Detecting and Analysing Coordinated Behaviour on Social Media,” *Journal of Computational Social Science*, vol. 7, pp. 1139–1160, 2024.

[7] M. H. Saeed, S. Ali, J. Blackburn, E. De Cristofaro, S. Zannettou, and G. Stringhini, “TROLLMAGNIFIER: Detecting State-Sponsored Troll Accounts on Reddit,” arXiv:2112.00443, 2021.

[8] S. Mohammad, S. Kiritchenko, P. Sobhani, X. Zhu, and C. Cherry, “SemEval-2016 Task 6: Detecting Stance in Tweets,” *Proceedings of the 10th International Workshop on Semantic Evaluation*, pp. 31–41, 2016.
