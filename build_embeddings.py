import os
import re
import sqlite3
import time
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

# .envからAPIキーを読み込む
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
DB_PATH = BASE_DIR / "movies.db"
NPZ_PATH = BASE_DIR / "embeddings.npz"
SOURCE_DB_PATH = Path.home() / "Desktop" / "movie" / "movies.db"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
EMBEDDING_DIM = 768
BATCH_SIZE = 100


def load_cached_embeddings() -> dict[int, np.ndarray]:
    """既存の embeddings.npz や Desktop/movie/movies.db（読み取り専用）に生成済みベクトルがあれば読み込む"""
    cached: dict[int, np.ndarray] = {}

    # 1. Desktop/movie/movies.db に保存されている生成済みベクトル（768次元 float32）を読み取り専用で取得
    if SOURCE_DB_PATH.exists():
        try:
            src_conn = sqlite3.connect(f"file:{SOURCE_DB_PATH}?mode=ro", uri=True)
            rows = src_conn.execute(
                "SELECT id, embedding FROM movies WHERE embedding IS NOT NULL AND length(embedding) = ?",
                (EMBEDDING_DIM * 4,),
            ).fetchall()
            src_conn.close()
            for movie_id, blob in rows:
                vec = np.frombuffer(blob, dtype=np.float32).copy()
                norm = np.linalg.norm(vec)
                cached[int(movie_id)] = vec / (norm if norm > 0 else 1.0)
        except Exception:
            pass

    # 2. すでに途中保存された embeddings.npz があれば上書き読み込み
    if NPZ_PATH.exists():
        try:
            data = np.load(NPZ_PATH)
            for movie_id, vec in zip(data["ids"], data["vectors"]):
                if len(vec) == EMBEDDING_DIM:
                    cached[int(movie_id)] = vec.astype(np.float32)
        except Exception:
            pass

    return cached


def save_npz(id_to_vec: dict[int, np.ndarray]):
    """現在までに生成できたID配列と正規化済みベクトル配列を embeddings.npz に保存する"""
    sorted_ids = sorted(id_to_vec.keys())
    ids_array = np.array(sorted_ids, dtype=np.int32)
    vectors_array = np.vstack([id_to_vec[i] for i in sorted_ids]).astype(np.float32)
    np.savez_compressed(NPZ_PATH, ids=ids_array, vectors=vectors_array)


def embed_batch_with_retry(client: genai.Client, texts: list[str]) -> np.ndarray:
    """100件のテキストをEmbedding APIで768次元ベクトル化する（レート制限時は自動待機してリトライ）"""
    models = ["gemini-embedding-001", "text-embedding-004"]
    for attempt in range(1, 5):
        for model_name in models:
            try:
                resp = client.models.embed_content(
                    model=model_name,
                    contents=texts,
                    config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
                )
                vectors = np.array([e.values for e in resp.embeddings], dtype=np.float32)
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                return (vectors / norms).astype(np.float32)
            except Exception as err:
                err_str = str(err)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    # 1日あたり上限（PerDay）に達した場合は例外を送出して途中保存へ移る
                    if "PerDay" in err_str or "Day" in err_str:
                        raise RuntimeError("DAILY_QUOTA_REACHED") from err
                    m_delay = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str)
                    wait_sec = int(float(m_delay.group(1))) + 3 if m_delay else 62
                    print(f"  [レート制限 100件/分] {wait_sec} 秒待機して再開します...", flush=True)
                    time.sleep(wait_sec)
                    break
                if model_name == models[-1] and attempt == 4:
                    raise
    raise RuntimeError("Embeddingの取得に失敗しました。")


def build_embeddings():
    """全映画の（タイトル / ジャンル / あらすじ）をベクトル化し、embeddings.npzに保存する"""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise ValueError(".env に有効な GEMINI_API_KEY を設定してください。")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"movies.db が見つかりません: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT id, title, genres, overview FROM movies ORDER BY id").fetchall()

    total = len(rows)
    id_to_vec = load_cached_embeddings()
    if id_to_vec:
        save_npz(id_to_vec)
        print(f"既存キャッシュから {len(id_to_vec)} / {total} 件のベクトルを読み込みました。", flush=True)

    pending_rows = [r for r in rows if r[0] not in id_to_vec]
    if not pending_rows:
        print(f"{total} / {total} 件完了（すべて生成済みです）", flush=True)
        return

    client = genai.Client(api_key=GEMINI_API_KEY)
    for start_idx in range(0, len(pending_rows), BATCH_SIZE):
        batch = pending_rows[start_idx : start_idx + BATCH_SIZE]
        batch_texts = [
            f"タイトル: {r[1]} / ジャンル: {r[2]} / あらすじ: {r[3] if r[3] != '（あらすじ情報なし）' else ''}"
            for r in batch
        ]
        try:
            vecs = embed_batch_with_retry(client, batch_texts)
            for row, vec in zip(batch, vecs):
                id_to_vec[int(row[0])] = vec
            save_npz(id_to_vec)
            print(f"{len(id_to_vec)} / {total} 件完了", flush=True)
            if start_idx + BATCH_SIZE < len(pending_rows):
                time.sleep(1.5)
        except RuntimeError as err:
            if "DAILY_QUOTA_REACHED" in str(err):
                print(
                    f"※ Gemini API無料枠の上限に達したため、生成済みの {len(id_to_vec)} / {total} 件を {NPZ_PATH.name} に保存して終了しました。",
                    flush=True,
                )
                break
            raise

    print(f"完了: {len(id_to_vec)} 件の正規化済みベクトルを {NPZ_PATH} に保存しました。", flush=True)


if __name__ == "__main__":
    build_embeddings()
