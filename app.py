import json
import os
import sqlite3
from pathlib import Path
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# .envファイルからAPIキーを読み込む
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
DB_PATH = BASE_DIR / "movies.db"
NPZ_PATH = BASE_DIR / "embeddings.npz"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# ページ設定（中央寄せ1カラム、タイトル1行のみ）
st.set_page_config(page_title="映画検索", layout="centered")
st.title("映画検索")


@st.cache_data(show_spinner=False)
def load_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """事前生成した映画ID配列と正規化済みベクトル配列をメモリに読み込む"""
    data = np.load(NPZ_PATH)
    return data["ids"], data["vectors"]


def get_filter_options() -> tuple[int, list[str]]:
    """DBから最長の上映時間と、国の一覧（件数が多い順）を取得する"""
    with sqlite3.connect(DB_PATH) as conn:
        max_row = conn.execute("SELECT MAX(runtime) FROM movies").fetchone()
        max_time = int(max_row[0]) if max_row and max_row[0] else 300
        country_rows = conn.execute(
            "SELECT country FROM movies WHERE country != '不明' GROUP BY country ORDER BY COUNT(*) DESC"
        ).fetchall()
    return max_time, ["すべて"] + [r[0] for r in country_rows] + ["不明"]


@st.cache_data(show_spinner=False)
def embed_query_text(query_text: str) -> np.ndarray:
    """ユーザーの検索文をEmbeddingモデルでベクトル化（768次元）し、正規化して返す"""
    client = genai.Client(api_key=GEMINI_API_KEY)
    for model_name in ["gemini-embedding-001", "text-embedding-004"]:
        try:
            resp = client.models.embed_content(
                model=model_name,
                contents=query_text,
                config=types.EmbedContentConfig(output_dimensionality=768),
            )
            vec = np.array(resp.embeddings[0].values, dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / (norm if norm > 0 else 1.0)
        except Exception:
            if model_name == "text-embedding-004":
                raise
    raise RuntimeError("検索文のベクトル化に失敗しました。")


def fetch_vector_candidates(query_text: str, country: str, min_min: int, max_min: int) -> list:
    """【第1・2段階】SQLiteで国と上映時間を絞り込み、コサイン類似度の高い上位40件を取得する"""
    where_parts = ["runtime BETWEEN ? AND ?"]
    params: list = [min_min, max_min]
    if country != "すべて":
        where_parts.append("country = ?")
        params.append(country)

    # 1. 条件に合う映画の情報をSQLiteから取得
    sql = f"SELECT id, title, release_year, runtime, genres, country, overview FROM movies WHERE {' AND '.join(where_parts)}"
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(sql, params).fetchall()
    if not rows:
        return []

    movie_map = {r[0]: r for r in rows}
    ids_arr, vectors_arr = load_embeddings()

    # 条件に合う映画IDのインデックスだけを抽出
    mask = np.isin(ids_arr, list(movie_map.keys()))
    filtered_ids = ids_arr[mask]
    filtered_vecs = vectors_arr[mask]
    if len(filtered_ids) == 0:
        return []

    # 2. 検索文ベクトルとの内積（正規化済みのためコサイン類似度と一致）で上位40件を取得
    query_vec = embed_query_text(query_text)
    similarities = filtered_vecs @ query_vec
    top_indices = np.argsort(similarities)[::-1][:40]
    return [movie_map[int(filtered_ids[i])] for i in top_indices]


@st.cache_data(show_spinner=False)
def rerank_movies_with_gemini(user_text: str, candidates_json: str) -> list[dict]:
    """【第3段階】上位40件の候補からリクエストに最も合う最大10本と1行のおすすめ理由をJSONで返す"""
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        f"ユーザーのリクエスト: 「{user_text}」\n候補映画リスト: {candidates_json}\n"
        "上記候補の中からリクエストに最も合う順に最大10本選び、"
        '[{"id": 映画の数値ID, "reason": "1行のおすすめ理由"}] のJSON配列のみを返してください。'
    )
    models = ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-flash-lite-latest", "gemini-flash-latest"]
    for idx, model_name in enumerate(models):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            result = json.loads(resp.text)
            return result if isinstance(result, list) else []
        except Exception:
            if idx == len(models) - 1:
                raise
    return []


def search_movies_simple(country: str, min_min: int, max_min: int, sort_choice: str) -> list:
    """検索バーが空のとき用：国・上映時間・並び替えだけでSQLiteから一覧を取得する"""
    order_map = {
        "公開年が新しい順": "release_year DESC",
        "上映時間が短い順": "runtime ASC",
        "上映時間が長い順": "runtime DESC",
    }
    where_parts = ["runtime BETWEEN ? AND ?"]
    params: list = [min_min, max_min]
    if country != "すべて":
        where_parts.append("country = ?")
        params.append(country)

    sql = f"""
        SELECT id, title, release_year, runtime, genres, country, overview
        FROM movies
        WHERE {' AND '.join(where_parts)}
        ORDER BY {order_map.get(sort_choice, 'release_year DESC')}
    """
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(sql, params).fetchall()


# --- 初期チェック ---
if not DB_PATH.exists():
    st.error("movies.db が見つかりません。先に `python build_db.py` を実行してください。")
    st.stop()
if not NPZ_PATH.exists():
    st.error("embeddings.npz が見つかりません。先に `python build_embeddings.py` を実行してください。")
    st.stop()

load_embeddings()

if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
    st.warning("`.env` に `GEMINI_API_KEY` を設定してください。")

# --- 最小限の入力エリア（UI維持） ---
user_query = st.text_input("キーワード検索", placeholder="例: 泣けるSF映画、気軽に観られるコメディ")
max_db_runtime, country_options = get_filter_options()

if not user_query.strip():
    col1, col2 = st.columns(2)
    with col1:
        selected_country = st.selectbox("国", country_options)
    with col2:
        sort_option = st.selectbox("並び替え", ["公開年が新しい順", "上映時間が短い順", "上映時間が長い順"])
else:
    selected_country = st.selectbox("国", country_options)
    sort_option = "公開年が新しい順"

min_runtime, max_runtime = st.slider("上映時間（分）", 0, max_db_runtime, (1, min(180, max_db_runtime)), step=5)

# --- 検索の実行とシンプルな結果表示 ---
if user_query.strip():
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        st.error("AI検索には `.env` の `GEMINI_API_KEY` 設定が必要です。")
        st.stop()

    with st.spinner("検索中..."):
        candidates = fetch_vector_candidates(user_query.strip(), selected_country, min_runtime, max_runtime)
        selected_items = []
        if candidates:
            compact_list = [{"id": r[0], "title": r[1], "genres": r[4], "overview": r[6][:150]} for r in candidates]
            picks = rerank_movies_with_gemini(user_query.strip(), json.dumps(compact_list, ensure_ascii=False))
            candidate_map = {r[0]: r for r in candidates}
            selected_items = [(candidate_map[p["id"]], p.get("reason", "")) for p in picks if p.get("id") in candidate_map]

    st.caption(f"該当: {len(selected_items)} 件")
    lines = []
    for row, reason in selected_items:
        _, title, year, runtime, _, country, overview = row
        year_text = f"{year}年" if year > 0 else "公開年不明"
        block = f"---\n\n**{title}**（{year_text} / {runtime}分 / {country}）"
        if reason:
            block += f"\n\nおすすめ理由: {reason}"
        if overview and overview != "（あらすじ情報なし）":
            block += f"\n\n{overview}"
        lines.append(block)
    if lines:
        st.markdown("\n\n".join(lines))
else:
    movies = search_movies_simple(selected_country, min_runtime, max_runtime, sort_option)
    st.caption(f"該当: {len(movies)} 件")
    lines = []
    for _, title, year, runtime, _, country, overview in movies[:100]:
        year_text = f"{year}年" if year > 0 else "公開年不明"
        block = f"---\n\n**{title}**（{year_text} / {runtime}分 / {country}）"
        if overview and overview != "（あらすじ情報なし）":
            block += f"\n\n{overview}"
        lines.append(block)
    if lines:
        st.markdown("\n\n".join(lines))
