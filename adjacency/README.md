# Adjacency Matrix Module

這個資料夾負責把 Reddit account-level 資料轉成 graph artifacts。它刻意放在 repo 根目錄，和 `behavior/`、`llm/` 平行，定位是獨立的 graph construction 模組。

目前輸出不使用 dense matrix，而是：

- edge-list CSV
- compressed sparse COO `.npz`
- node index mapping
- build summary JSON

原因是帳號節點約四萬多個，如果直接做 dense adjacency matrix，會產生數十億個 cell，不適合版本控制，也不適合一般筆電處理。

## Inputs

預設讀取：

```text
llm/Export/reddit_comments_analyzed.csv.gz
llm/Export/reddit_posts_analyzed.csv.gz
Archive/export_working_files/account_feature_matrix.csv
```

第一份提供 account-to-account comment feedback edge：

```text
source_author -> target_author
```

第二份提供 post author 的發文總數與發文時間，用來建立「A 發文後 B 是否固定來留言」的 trigger-response frequency graph。

第三份提供 account-level behavior / LLM feature matrix，主要用來建立 tag similarity graph。

## Run

在 repo 根目錄執行：

```bash
.venv/bin/python adjacency/build_adjacency_matrices.py
```

輸出會放在：

```text
adjacency/output/
```

此資料夾已被 `.gitignore` 排除，避免大型矩陣與 edge list 被推上 GitHub。

## Common Cleaning

所有 interaction graphs 共用以下清理：

- 移除 `[deleted]`、`[removed]`、空白作者與無效作者
- 預設移除 self-loop：`source_author == target_author`
- 保留 `neutral` / `unclear` 留言在 count graph
- 在 signed graph 中，`neutral` / `unclear` 通常因為 score 接近 0，對 signed weight 影響較小

## Single Graph

### `A_single`

單圖模式只產生一張 signed directed weighted graph。

```text
A_single[i,j] = mean(feedback_score_ij / 100) * log(1 + n_ij)
```

其中：

- `i` 是留言作者 `source_author`
- `j` 是文章作者 `target_author`
- `n_ij` 是 i 對 j 的留言次數
- 正值代表支持
- 負值代表反對
- 絕對值代表互動強度

使用 `log(1 + n_ij)` 是為了保留重複互動訊號，但避免少數超高頻帳號把整張圖壓扁。

輸出：

```text
adjacency/output/single-graph/edges_single_signed.csv
adjacency/output/single-graph/matrix_single_signed.npz
```

## Multi Graph

多圖模式把不同關係拆成不同 adjacency layer。

| Graph | Weight | 意義 |
| --- | --- | --- |
| `A_count` | `n_ij` | 純互動次數 baseline |
| `A_signed` | `mean(score/100) * log(1+n)` | 支持/反對方向與強度 |
| `A_positive` | `max(A_signed, 0)` | 支持、認同、補充、reinforcement |
| `A_negative` | `max(-A_signed, 0)` | 反對、批評、攻擊、削弱 |
| `A_degree_adjusted` | Zaman-inspired degree adjusted count | 強調高資訊量互動邊 |
| `A_trigger_response` | response coverage × log frequency | A 發文是否穩定觸發 B 留言 |
| `A_co_target` | cosine similarity over shared targets | 兩個帳號是否常常留言到同一批作者 |
| `A_co_negative_target` | cosine similarity over shared oppositional targets | 兩個帳號是否常常反對同一批作者 |
| `A_tag_similarity` | cosine similarity of rhetoric profiles | 內容/修辭相似度，只作 EDA / 解釋輔助 |

### `A_count`

```text
A_count[i,j] = n_ij
```

這是最接近傳統 interaction graph 的版本，只看互動頻率，不看支持或反對。

### `A_signed`

```text
A_signed[i,j] = mean(feedback_score_ij / 100) * log(1 + n_ij)
```

這張圖使用 LLM comment feedback 的結果，把 comment-to-post interaction 轉成 signed weighted directed edge。

### `A_positive` / `A_negative`

正負圖分開是為了避免把完全不同的社會關係混在一起：

- 支持圖適合看互相放大、互相認同、群體 reinforcement
- 反對圖適合看攻擊、批評、針對特定作者的行為

這也讓後續社群偵測或可視化比較容易解釋。

### `A_degree_adjusted`

Zaman et al. 的 Ising bot detection 使用 retweet graph，並讓 edge strength 受互動次數、out-degree、in-degree 影響。這裡保留它的核心想法，但不直接實作完整 Ising/min-cut model。

```text
A_degree_adjusted[i,j] =
n_ij / (1 + exp(alpha_out / out_degree_i + alpha_in / in_degree_j - 2))
```

其中：

- `out_degree_i` 是 i 的總留言互動數
- `in_degree_j` 是 j 收到的總留言互動數
- `alpha_out` 和 `alpha_in` 是對應 degree 的 99th percentile

這個設計讓低資訊量、低度數的偶然互動權重較小；高活躍留言者與高互動目標之間的邊會比較被重視。

### `A_trigger_response`

這張圖捕捉的不是「B 對 A 留言總共幾次」，而是：

```text
A 每次發文，B 是否經常出現留言？
```

方向刻意設成：

```text
post_author -> responder
```

也就是：

```text
A_trigger_response[A,B] = A 的發文觸發 B 回應的強度
```

核心欄位：

```text
response_coverage = posts_with_b_comment / a_total_posts
```

其中：

- `a_total_posts` 是 A 在 analyzed posts 裡的總發文數
- `posts_with_b_comment` 是 B 至少留言一次的 A 發文數
- `b_total_comments_on_a_posts` 是 B 在 A 發文下的總留言數

矩陣權重使用較穩定的版本：

```text
weight_trigger_response =
response_coverage
* log(1 + posts_with_b_comment)
* log(1 + b_total_comments_on_a_posts)
```

這樣可以避免「A 只發一篇、B 留一次」就得到過強訊號，同時保留穩定跟隨與重複互動的 evidence。輸出也包含 `median_response_delay_minutes` 和 `p90_response_delay_minutes`，方便判斷 B 是否總是在 A 發文後很快出現。

### `A_co_target`

這張圖捕捉的不是直接互動，而是兩個 commenter 的 target set 是否重疊：

```text
A_co_target[i,j] = cosine_similarity(target_profile_i, target_profile_j)
```

其中 `target_profile_i` 是帳號 `i` 留言過的 post authors，權重使用：

```text
log(1 + n_comments_to_target)
```

所以這層回答：

```text
i 和 j 是否常常去同一批作者底下留言？
```

這是 undirected projected graph。edge list 每對帳號只存一次，`.npz` sparse matrix 會 mirror 成雙向。

為了避免熱門 target 讓 projection 爆炸，預設限制：

- 每個 target 最多保留互動最強的 200 個 source accounts
- co-target edge 至少要有 2 個 shared targets
- cosine similarity 至少 0.15
- 每個帳號最多保留 top 25 co-target neighbors

### `A_co_negative_target`

這張圖是 `A_co_target` 的負向版本，只看 shared oppositional targets：

```text
target_negative_profile_i[target] = log(1 + oppositional_count_i_to_target)
```

因此它回答：

```text
i 和 j 是否常常反對 / 攻擊同一批作者？
```

這層很適合後續 suspicious coordination scoring，因為它比 rhetoric similarity 更接近「行動目標一致」。但它仍然不是最終判決；例如兩個正常使用者也可能共同批評同一個高爭議作者，所以應該和 trigger-response、manipulative signal、interaction reach 一起看。

### `A_tag_similarity`

這張圖不是互動圖，也不進 MCA 主分數；它是內容/修辭相似的 EDA 與解釋輔助圖。

```text
A_tag_similarity[i,j] = cosine_similarity(tag_profile_i, tag_profile_j)
```

目前的 tag profile 使用：

- 非 neutral rhetoric tag ratios
- `avg_manipulative_rhetoric_score / 100`

預設只保留：

- `analyzed_post_count >= 2`
- non-neutral tag count `>= 2`
- cosine similarity `>= 0.75`
- 每個帳號最多 top 10 neighbors

這個限制是刻意的。全體帳號中只有一部分有 post-level rhetoric tags，而且保留下來的 pair similarity 分數通常偏高，所以它不適合單獨用來排序帳號或加進 MCA score。

這層保留的用途是：

- EDA / visualization
- 解釋 top accounts 的 rhetoric-similar neighbors
- 檢查 co-target group 裡是否也有相似 rhetoric profile

換句話說，`A_tag_similarity` 回答的是：

```text
這兩個帳號的 rhetoric tag profile 像不像？
```

它不直接回答：

```text
這個帳號是否可疑？
```

因此正式 MCA scoring 不使用這層作為主分數來源。

## Output Files

```text
adjacency/output/
├── nodes.csv
├── all_interaction_edge_stats.csv
├── summary.json
├── single-graph/
│   ├── edges_single_signed.csv
│   └── matrix_single_signed.npz
└── multi-graph/
    ├── edges_count.csv
    ├── edges_signed.csv
    ├── edges_positive.csv
    ├── edges_negative.csv
    ├── edges_degree_adjusted.csv
    ├── edges_trigger_response.csv
    ├── edges_co_target.csv
    ├── edges_co_negative_target.csv
    ├── edges_tag_similarity.csv
    ├── tag_similarity_candidate_nodes.csv
    └── matrix_*.npz
```

`.npz` matrices are COO-style sparse matrices with:

```text
row
col
data
shape
```

`nodes.csv` maps `node_id` back to `user_id`.

## Design Tradeoffs

### Sparse output instead of dense matrix

Dense matrix is conceptually simple but practically wasteful. With more than 45k nodes, a full matrix would have more than 2 billion cells. Edge list and COO sparse matrix are easier to store, inspect, and feed into graph tooling.

### Directed interaction graph

Comment feedback has a clear direction:

```text
commenter -> post author
```

So interaction graphs remain directed. Direction matters because frequently criticizing an author is not the same as being criticized by that author.

### Undirected tag similarity graph

Tag similarity and co-target projection are symmetric. If two accounts use similar rhetoric profiles, or point at similar target sets, there is no natural source/target direction. The edge list stores each pair once; the sparse matrix mirrors it.

### Mean score times log frequency

Using only mean feedback ignores repeated behavior. Using raw sum overweights extremely active accounts. `mean * log(1+n)` keeps both stance direction and repeated interaction, while reducing domination by high-volume accounts.

### Positive and negative graphs are separated

Bot and coordination graphs are not always homophilous. Some suspicious accounts may mostly interact with normal users or opponents. Keeping positive and negative layers separated avoids assuming that all edges mean similarity.

### Trigger-response frequency is different from raw count

`A_count[B,A]` answers:

```text
B 對 A 留言幾次？
```

`A_trigger_response[A,B]` answers:

```text
A 的多少篇發文會引來 B 留言？
```

所以兩者方向和意義不同。前者是 commenter-to-author interaction，後者是 post-author-to-responder frequency relation。這層更接近「固定跟隨某個發文者」的 coordinated behavior evidence。

### Co-target is different from direct interaction

`A_count[i,j]` answers:

```text
i 是否直接留言給 j？
```

`A_co_target[i,j]` answers:

```text
i 和 j 是否經常留言到同一批 target？
```

所以 co-target 是 commenter-commenter relation，不是 commenter-author relation。這層能捕捉沒有直接互動、但行動目標高度一致的帳號對。

### Tag similarity is diagnostic, not scoring

The account feature matrix already contains tag ratios as node features. `A_tag_similarity` converts those node features into a pairwise rhetoric-similarity relation. This is useful for EDA and explanation, but it is not used as a primary MCA scoring input because it measures pair similarity rather than account suspiciousness.

For MCA scoring, rhetoric information enters through the manipulative signal instead:

```text
avg_rhetorical_score
non_neutral_post_ratio
oppositional_stance_ratio
```

## Literature Rationale

### Zaman et al. - interaction graph and degree-adjusted edge strength

Zaman et al. model users as nodes and retweet interactions as graph edges. Their Ising model uses interaction features such as retweet count, out-degree, and in-degree, with a link-energy function that downweights low-degree edges. `A_count` and `A_degree_adjusted` borrow this graph construction idea without claiming to reproduce the full Ising/min-cut algorithm.

### Energy Propagation - heterophily warning

Energy propagation work on bot detection emphasizes that bot-related graphs may show heterophily rather than simple homophily. This supports splitting positive and negative edges instead of forcing all relations into one similarity graph.

### BotCF - community structure as useful signal

BotCF argues that community structure features improve social bot detection when combined with semantic and account-property features. This supports producing adjacency layers that can later be used for community detection or GNN input, while keeping the current module focused only on graph construction.

### Fine-Grained Dynamic GNN - temporal graph as future work

Dynamic rumor-detection graph work shows the value of temporal edge weighting and graph snapshots. We do not implement temporal graphs here because the current task is account-level adjacency construction, but the existing edge lists can later be extended with time-windowed snapshots.

### TAXODIS / SemEval framing - rhetoric and stance signals

The rhetoric tags come from the LLM text-analysis side of the project and are grounded in fine-grained propaganda/disinformation cue research. The comment feedback labels are closer to stance detection than generic sentiment: they measure whether a comment supports or opposes the target post.

## Recommended Project Positioning

For the capstone report/demo, describe this module as:

> an account-level graph construction layer that transforms LLM-analyzed Reddit comments and rhetoric profiles into sparse single-graph and multi-graph adjacency representations for downstream suspicious-group detection.

This keeps the scope professional: the module does not itself declare bots; it produces defensible graph inputs for later clustering, visualization, anomaly review, or model training.
