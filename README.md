# 映画検索アプリ

自然文から観たい映画を探せる検索アプリです。  
**SQLite × ベクトル検索（NumPy） × Gemini API** の2段階構成により、検索精度とレスポンス速度・APIコストの両立を目指しました。

## 主な機能
- 条件絞り込み：製作国・上映時間（分）によるフィルタリング
- 自然文検索：あらすじに単語が完全一致しなくても、意味の近さから候補を抽出
- AIによる厳選＆理由生成：抽出した候補をGeminiが精査し、おすすめ理由付きで最大10本を提案

## 技術スタック
- Frontend / App: Python, Streamlit
- Database: SQLite (`movies.db`)
- Search / AI: Google Gemini API (`gemini-embedding-001`, `gemini-2.5-flash`), NumPy (`embeddings.npz`)

## 工夫した点
全件をLLMに直接渡すと高コストになるため、以下の3ステップに分離しています。

1. 属性フィルタリング（SQLite）：国・上映時間の絶対条件に合う映画IDを抽出
2. ベクトル検索（NumPy）：検索文をEmbedding化し、事前計算済みのベクトルとのコサイン類似度で上位40件を取得
3. リランキング（Gemini 2.5 Flash）：上位40件のあらすじを読み込み、リクエストに最も合う最大10本とおすすめ理由をJSON形式で生成

## セットアップ・起動方法

```bash
# 1. リポジトリをクローン
git clone　https://github.com/takashiuchida20040831-bot/movie-search-app.git
cd movie-search-app

# 2. 依存パッケージのインストール
pip install -r requirements.txt

# 3. 環境変数の設定（.env.example をコピーして GEMINI_API_KEY を記入）
cp .env.example .env

# 4. アプリ起動
streamlit run app.py
```
