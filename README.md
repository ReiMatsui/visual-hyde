# Visual HyDE 🔍📊

**Visual HyDE** は、HyDE（Hypothetical Document Embeddings）のパラダイムをチャート画像検索に応用した研究プロジェクトです。テキストクエリから「仮説チャート画像」を生成し、CLIP の同一モーダル（image→image）検索に変換することで、クロスモーダル検索のモダリティギャップを克服します。

---

## 目次

- [背景・問題設定](#背景問題設定)
- [アイデア：Visual HyDE](#アイデアvisual-hyde)
- [パイプライン](#パイプライン)
- [実装した検索手法](#実装した検索手法)
- [システムアーキテクチャ](#システムアーキテクチャ)
- [対応データセット](#対応データセット)
- [インストール](#インストール)
- [クイックスタート](#クイックスタート)
- [CLI リファレンス](#cli-リファレンス)
- [設定（環境変数）](#設定環境変数)
- [実験スクリプト](#実験スクリプト)
- [評価指標](#評価指標)
- [プロジェクト構造](#プロジェクト構造)

---

## 背景・問題設定

### CLIP のモダリティギャップ

CLIP は text encoder と image encoder を持つ vision-language モデルですが、両者のベクトル空間には構造的なギャップ（Modality Gap）が存在します。特に「形状」や「傾向」を表すテキストクエリ（例：「右肩上がりのグラフ」）と画像ベクトルの間のコサイン類似度は非常に低くなります。

```
テキストクエリ → CLIP text encoder ─┐
                                    ├─ コサイン類似度 → 0.18 ← ❌ 低い！
チャート画像   → CLIP image encoder ─┘
```

**実測値の比較（クエリ：「売上が右肩上がりのグラフ」）**

```
手法                  コサイン類似度   取得結果
──────────────────────────────────────────────────────────
Text-Direct           0.18            ❌ 無関係なチャート
TCD-HyDE              0.42            △  やや改善
Visual HyDE（本手法）  0.78            ✅  関連チャートを正確に取得
```

この「4.3倍」のスコア差が本研究の出発点です。

### HyDE（Hypothetical Document Embeddings）との関係

Gao et al. (2022) の HyDE は「クエリに対する仮説的な回答文書を生成し、その埋め込みで検索する」手法です。Visual HyDE はこれをビジュアル領域に拡張します。

```
HyDE（原論文）:  query → LLM → 仮説テキスト文書 → text embed  → 検索
Visual HyDE  :  query → VLM → 仮説チャート画像 → image embed → 検索
```

---

## アイデア：Visual HyDE

**クロスモーダル問題を同一モーダルに変換する**

```
従来（クロスモーダル）:
  "右肩上がり" ──[text enc]──→ vec_text ─────────────→ ❌ 遠い

Visual HyDE（同一モーダル）:
  "右肩上がり" ──[VLM]──→ 📊 仮説チャート画像
                                ──[image enc]──→ vec_image ──→ ✅ 近い！
```

テキストクエリを直接画像空間で比較するのではなく、VLM（Claude / GPT-4o）にそのクエリを視覚化した matplotlib コードを書かせ、生成された画像を CLIP の image encoder で埋め込み直すことでモダリティギャップを回避します。生成チャートは **値の正確さは不要**——チャートタイプとトレンドの形状さえ正しければ有効です。

---

## パイプライン

### オフライン：コーパス構築

```
データセット (ChartQA / FigureQA / ViDoRe V2)
        │
        ▼
┌──────────────────────┐
│   Dataset Loader     │  HuggingFace Hub から画像を取得・保存
└──────────────────────┘
        │  PNG 画像
        ▼
┌──────────────────────┐
│   CLIPEncoder        │  open-clip-torch ViT-L/14
│   (image encoder)    │  バッチサイズ 32 で並列処理 → 768-dim ベクトル
└──────────────────────┘
        │  float32 ベクトル群（L2正規化済み）
        ▼
┌──────────────────────┐
│   CorpusIndex        │  FAISS IndexFlatIP に格納
│   (FAISS)            │  L2正規化 → コサイン類似度 = 内積で近似
└──────────────────────┘
        │
        ▼
  data/indices/<dataset>/ に永続化（.faiss + .meta.json）
```

### オンライン：クエリ処理（Visual HyDE）

```
テキストクエリ「売上が右肩上がりのグラフ」
        │
        ▼
┌──────────────────────────────────────────┐
│  LLMClient  (Anthropic / OpenAI)         │
│                                          │
│  SYSTEM: "You are an expert data         │
│   visualization assistant. Generate      │
│   matplotlib code for the visual         │
│   pattern described by the query."       │
│                                          │
│  USER: "Query: {query}                   │
│   Focus on chart type and trend shape."  │
└──────────────────────────────────────────┘
        │  Python コード (```python ... ```)
        ▼
┌──────────────────────────────────────────┐
│  MatplotlibChartGenerator                │
│                                          │
│  • コードをサブプロセスで安全に実行       │
│  • plt.savefig(output_path, dpi=100)     │
│  • タイムアウト: 30 秒                   │
│  • 失敗時: フォールバック画像を生成       │
└──────────────────────────────────────────┘
        │  仮説チャート PNG
        ▼
┌──────────────────────────────────────────┐
│  CLIPEncoder  (image encoder)            │
│  ViT-L/14 → 768-dim ベクトル（L2正規化） │
└──────────────────────────────────────────┘
        │  クエリベクトル（image space）
        ▼
┌──────────────────────────────────────────┐
│  CorpusIndex.search()                    │
│  FAISS IndexFlatIP → top-k 候補を返却    │
│  ※ image-to-image コサイン類似度検索     │
└──────────────────────────────────────────┘
        │
        ▼
  RetrievalOutput: [rank1, rank2, ..., rank10]
```

---

## 実装した検索手法

| 手法 | カテゴリ | 説明 | モダリティ |
|------|---------|------|-----------|
| `text_direct` | Baseline | CLIP text embed → コーパス検索 | text → image |
| `tcd_hyde` | Baseline | VLM がチャートをテキスト記述 → CLIP text embed | text → image |
| `colpali` | Existing SOTA | 事前計算済み結果のロード（Late Interaction） | — |
| `visual_hyde_matplotlib` | **Proposed** | VLM → matplotlib コード → PNG → CLIP image embed | image → image |
| `visual_hyde_nano_banana` | **Proposed** | Gemini 画像生成 → PNG → CLIP image embed | image → image |
| `hybrid_rrf` | **Proposed** | Visual HyDE + Text-Direct の RRF 融合 | hybrid |

### Hybrid RRF（Reciprocal Rank Fusion）

Visual HyDE の視覚的検索とテキスト直接検索を重み付けで統合します：

```
score(doc) = α / (k + rank_visual)  +  (1-α) / (k + rank_text)

  α = 1.0 → 完全視覚（Visual HyDE のみ）
  α = 0.5 → 均等統合（デフォルト）
  α = 0.0 → 完全テキスト（Text-Direct のみ）
  k = 60  （RRF スムージング定数）
```

---

## システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────────────────┐
│  config.py  (pydantic-settings)                                     │
│  PathSettings / EmbeddingSettings / GenerationSettings /            │
│  RetrievalSettings / EvaluationSettings                             │
│  → 全設定は環境変数 or .env で上書き可能                             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
  ┌─────────────────┐ ┌──────────────────┐ ┌────────────────────┐
  │  data/          │ │  embedding/      │ │  generation/       │
  │  loaders.py     │ │  clip_encoder.py │ │  matplotlib_gen.py │
  │                 │ │  corpus_index.py │ │  image_gen.py      │
  │  ChartQA        │ │  (FAISS)         │ │  prompts.py        │
  │  FigureQA       │ │                  │ │                    │
  │  ViDoRe V2      │ │  ViT-L/14        │ │  LLMClient         │
  └────────┬────────┘ └────────┬─────────┘ └────────┬───────────┘
           │                   │                    │
           └──────────┬────────┘                    │
                      │                             │
                      ▼                             ▼
           ┌──────────────────┐          ┌──────────────────────┐
           │  retrieval/      │◄─────────│  baselines/          │
           │  base.py         │          │  tcd_hyde.py         │
           │  text_retriever  │          │  colpali.py          │
           │  visual_retriever│          └──────────────────────┘
           │  hybrid.py (RRF) │
           └────────┬─────────┘
                    │
                    ▼
           ┌──────────────────┐
           │  evaluation/     │
           │  metrics.py      │  MRR@k / Recall@k / nDCG@k
           │  runner.py       │  クエリタイプ別 / チャートタイプ別集計
           └──────────────────┘
```

---

## 対応データセット

| データセット | HF Hub ID | 規模 | 用途 |
|-------------|-----------|------|------|
| **ChartQA** | `ahmed-masry/ChartQA` | ~4,800 チャート / 2,500 クエリ (test) | Phase 1 主評価 |
| **FigureQA** | `lmms-lab/FigureQA` | ~10万チャート | Phase 2 アブレーション |
| **ViDoRe V2** | `vidore/vidore-benchmark-v2` | 複数ドメイン | Phase 1 追加評価 |

---

## インストール

[uv](https://docs.astral.sh/uv/) が必要です。

```bash
# uv のインストール（未導入の場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# リポジトリをクローン
git clone https://github.com/your-repo/visual-hyde.git
cd visual-hyde

# 依存パッケージをインストール（仮想環境も自動作成）
uv sync

# 設定ファイルを作成
cp .env.example .env
# .env を編集して API キーを設定
```

### 必須 API キー

`.env` に以下のどちらか（または両方）を設定してください：

```bash
# OpenAI を使う場合
VH_GEN_LLM_PROVIDER=openai
VH_GEN_OPENAI_API_KEY=sk-...
VH_GEN_OPENAI_MODEL=gpt-4o-mini   # コスト節約には mini 推奨

# Anthropic を使う場合
VH_GEN_LLM_PROVIDER=anthropic
VH_GEN_ANTHROPIC_API_KEY=sk-ant-...

# HuggingFace（ViDoRe V2 など認証が必要なデータセットを使う場合）
# トークン取得: https://huggingface.co/settings/tokens
HF_TOKEN=hf_...
```

---

## クイックスタート

### 1. コーパスを構築する

```bash
# ChartQA のチャート画像を FAISS インデックスに登録
uv run visual-hyde build-corpus chartqa

# → data/processed/chartqa/ に PNG を保存
# → data/indices/chartqa/ に FAISS インデックスを保存
# ※ 初回は HuggingFace からデータをダウンロード（数分）
```

### 2. 単一クエリで検索

```bash
uv run visual-hyde retrieve "売上が右肩上がりのグラフ"

# 取得件数・手法を指定
uv run visual-hyde retrieve "quarterly revenue trend" --top-k 5 --method visual_hyde
```

出力例：

```
Query: 売上が右肩上がりのグラフ
Method: visual_hyde_matplotlib

Rank 1  chartqa_multi_col_20255   score=0.7821
Rank 2  chartqa_two_col_61646     score=0.7714
Rank 3  chartqa_multi_col_18432   score=0.7689
...
```

---

## CLI リファレンス

### `build-corpus`

```
uv run visual-hyde build-corpus <dataset>

引数:
  dataset     chartqa | figureqa | vidore_v2

オプション:
  --split     HuggingFace split（デフォルト: test）
  --force     キャッシュを無視して再構築
```

### `retrieve`

```
uv run visual-hyde retrieve <query>

オプション:
  --method    text_direct | tcd_hyde | visual_hyde | hybrid
              （デフォルト: visual_hyde）
  --top-k     返す件数（デフォルト: 10）
  --dataset   使用するインデックス（デフォルト: chartqa）
```

---

## 設定（環境変数）

全設定は `.env` または環境変数で上書き可能です（pydantic-settings）。

### 埋め込みモデル（`VH_EMBED_*`）

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `VH_EMBED_MODEL` | `openai/clip-vit-large-patch14` | CLIP モデル名 |
| `VH_EMBED_BATCH_SIZE` | `32` | 埋め込みバッチサイズ |
| `VH_EMBED_DEVICE` | `cpu` | `cpu` / `cuda` / `mps` |

利用可能な CLIP モデル：

```
openai/clip-vit-base-patch32    — 軽量・高速（512-dim）
openai/clip-vit-large-patch14  — 推奨（768-dim）          ← デフォルト
google/siglip-base-patch16-224  — Google SigLIP（小型）
google/siglip-so400m-patch14-384 — SigLIP 大型版
```

### チャート生成（`VH_GEN_*`）

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `VH_GEN_METHOD` | `matplotlib` | `matplotlib` / `nano_banana` |
| `VH_GEN_LLM_PROVIDER` | `anthropic` | `anthropic` / `openai` |
| `VH_GEN_OPENAI_API_KEY` | — | OpenAI API キー |
| `VH_GEN_OPENAI_MODEL` | `gpt-4o` | OpenAI モデル名 |
| `VH_GEN_OPENAI_BASE_URL` | — | Azure / ローカルプロキシの場合に設定 |
| `VH_GEN_ANTHROPIC_API_KEY` | — | Anthropic API キー |
| `VH_GEN_ANTHROPIC_MODEL` | `claude-opus-4-6` | Anthropic モデル名 |
| `VH_GEN_MAX_CODE_TOKENS` | `1024` | コード生成の最大トークン数 |
| `VH_GEN_GENERATION_TIMEOUT_S` | `30` | コード実行タイムアウト（秒） |
| `VH_GEN_GEMINI_API_KEY` | — | Gemini API キー（nano_banana 使用時） |

### 検索・評価（`VH_RETR_*` / `VH_EVAL_*`）

| 変数 | デフォルト | 説明 |
|------|-----------|------|
| `VH_RETR_TOP_K` | `10` | 取得件数 |
| `VH_RETR_ALPHA` | `0.5` | RRF 重み（0=テキスト, 1=ビジュアル） |
| `VH_RETR_RRF_K` | `60` | RRF スムージング定数 |
| `VH_EVAL_MRR_K` | `10` | MRR の cutoff k |

---

## 実験スクリプト

### Phase 1: 主比較実験

全手法（Text-Direct / TCD-HyDE / Visual HyDE / Hybrid RRF）を同一データセットで比較します。

```bash
# 100件でクイック検証（コスト目安：gpt-4o-mini で ~$0.04）
uv run python experiments/phase1_main.py \
  --dataset chartqa \
  --max-queries 100 \
  --skip-image-gen     # Nano Banana（Gemini）をスキップ

# フルランで実行
uv run python experiments/phase1_main.py --dataset chartqa
```

出力：`results/chartqa/<YYYYMMDD_HHMMSS>/` に `results.json` と `summary_table.txt` を保存。

コンソール出力例：

```
┌────────────────────────┬─────────┬──────────┬───────────┬─────────┐
│ Retriever              │ MRR@10  │ Recall@5 │ Recall@10 │ NDCG@10 │
├────────────────────────┼─────────┼──────────┼───────────┼─────────┤
│ text_direct            │  0.082  │  0.060   │  0.091    │  0.074  │
│ tcd_hyde               │  0.231  │  0.190   │  0.260    │  0.218  │
│ visual_hyde_matplotlib │  0.445  │  0.390   │  0.510    │  0.430  │
│ hybrid_rrf             │  0.472  │  0.415   │  0.535    │  0.458  │
└────────────────────────┴─────────┴──────────┴───────────┴─────────┘
```

### Phase 2: アブレーション研究

FigureQA を使ってチャートタイプ別・クエリタイプ別の性能を分析します。

```bash
uv run python experiments/phase2_ablation.py \
  --dataset figureqa \
  --max-queries 200
```

`ChartType` × `QueryType` の 2D 性能マトリクスを出力します。

### Phase 3: Hybrid RRF の α スイープ

RRF 重み α を変化させ、最適なビジュアル/テキスト配分を探索します。

```bash
uv run python experiments/phase3_hybrid.py \
  --dataset chartqa \
  --alphas "0.0,0.3,0.5,0.7,1.0" \
  --max-queries 100
```

---

## 評価指標

純 Python で実装（scipy / sklearn 依存なし）。

| 指標 | 式 | 説明 |
|------|----|------|
| **MRR@k** | `mean(1 / rank_first_relevant)` | 最初の正解が何位かの逆数の平均 |
| **Recall@k** | `mean(|relevant ∩ top-k| / |relevant|)` | 正解を上位 k 件中に含む割合の平均 |
| **nDCG@k** | `DCG@k / IDCG@k` | ランク位置で割引いた累積ゲイン（正規化済み） |

全指標はクエリタイプ別（Trend / Comparison / Numeric）にも集計されます。

---

## プロジェクト構造

```
visual-hyde/
├── src/visual_hyde/          # メインパッケージ
│   ├── config.py             # pydantic-settings による設定管理
│   ├── llm_client.py         # Anthropic / OpenAI 抽象クライアント
│   ├── types.py              # CorpusItem / QueryItem / RetrievalOutput / GeneratedChart
│   ├── logging.py            # ログ設定
│   ├── cli.py                # typer CLI エントリポイント
│   ├── data/
│   │   ├── loaders.py        # ChartQA / FigureQA / ViDoRe V2 ローダー
│   │   └── pdf_extractor.py  # PDF からのチャート抽出
│   ├── embedding/
│   │   ├── clip_encoder.py   # open-clip-torch ラッパー（画像・テキスト両対応）
│   │   └── corpus_index.py   # FAISS インデックスの構築・保存・検索
│   ├── generation/
│   │   ├── prompts.py        # プロンプトテンプレート
│   │   ├── matplotlib_gen.py # Pattern A: コード生成 → サブプロセス実行
│   │   └── image_gen.py      # Pattern B: Gemini 画像生成
│   ├── retrieval/
│   │   ├── base.py           # BaseRetriever 抽象クラス
│   │   ├── text_retriever.py # Text-Direct 実装
│   │   ├── visual_retriever.py # Visual HyDE 実装（両 Pattern 対応）
│   │   └── hybrid.py         # RRF 融合検索
│   ├── baselines/
│   │   ├── tcd_hyde.py       # Textual Chart Description HyDE
│   │   └── colpali.py        # ColPali 事前計算結果ローダー
│   └── evaluation/
│       ├── metrics.py        # MRR / Recall / nDCG の純 Python 実装
│       └── runner.py         # 実験ランナー・結果 JSON 保存
│
├── experiments/
│   ├── phase1_main.py        # 主比較実験（6条件）
│   ├── phase2_ablation.py    # チャートタイプ別アブレーション
│   └── phase3_hybrid.py      # α スイープ（Hybrid RRF 感度分析）
│
├── data/                     # ← .gitignore 対象
│   ├── raw/                  # HuggingFace キャッシュ
│   ├── processed/            # 保存済みチャート PNG
│   ├── indices/              # FAISS インデックス（.faiss + .meta.json）
│   └── generated_charts/     # VLM 生成の仮説チャート PNG
│
├── results/                  # 実験結果 JSON・集計テキスト
├── notebooks/                # 探索的分析・可視化
├── pyproject.toml            # uv / hatchling によるパッケージ定義
├── .env.example              # 設定テンプレート
└── README.md
```

---

## 参考文献

- Gao et al. (2022). [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496). *HyDE の原論文*
- Radford et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020). *CLIP*
- Masry et al. (2022). [ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning](https://arxiv.org/abs/2203.10244). *ChartQA データセット*
- Faysse et al. (2024). [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449). *ColPali ベースライン*
- Johnson et al. (2019). [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734). *FAISS*
