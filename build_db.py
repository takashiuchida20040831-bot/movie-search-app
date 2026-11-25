import re
import sqlite3
from pathlib import Path

# 読み取り元の既存DBと、新しく作成するDBのパス
SOURCE_DB_PATH = Path.home() / "Desktop" / "movie" / "movies.db"
TARGET_DB_PATH = Path(__file__).parent / "movies.db"


def parse_safe_int(value, default: int = 0) -> int:
    """欠損値や数値でない文字列を安全に整数へ変換する関数"""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    # 文字列から数字部分だけを抽出（例: "120 min" -> 120）
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else default


def extract_primary_country(countries_value) -> str:
    """元データの製作国（カンマ区切りの場合は先頭の主要製作国）を抽出する関数"""
    if not countries_value:
        return "不明"
    # 例: "イギリス, アメリカ" -> 先頭の "イギリス" を主要国として整形（または単一国名）
    first_country = str(countries_value).split(",")[0].strip()
    return first_country if first_country else "不明"


def build_database():
    """既存の映画データから製作国を含む必要な列を抽出し、movies.dbを再作成する"""
    if not SOURCE_DB_PATH.exists():
        raise FileNotFoundError(f"元データが見つかりません: {SOURCE_DB_PATH}")

    # 1. 既存のデータベースから読み取り専用でデータを取得
    src_conn = sqlite3.connect(f"file:{SOURCE_DB_PATH}?mode=ro", uri=True)
    src_cursor = src_conn.cursor()
    src_cursor.execute(
        """
        SELECT title, release_year, runtime, genres, countries, overview
        FROM movies
        """
    )
    raw_rows = src_cursor.fetchall()
    src_conn.close()

    # 2. 新しいデータベースに movies テーブル（country列を含む）を作成
    dst_conn = sqlite3.connect(TARGET_DB_PATH)
    dst_cursor = dst_conn.cursor()

    dst_cursor.execute("DROP TABLE IF EXISTS movies")
    dst_cursor.execute(
        """
        CREATE TABLE movies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            release_year INTEGER NOT NULL,
            runtime INTEGER NOT NULL,
            genres TEXT NOT NULL,
            country TEXT NOT NULL,
            overview TEXT NOT NULL
        )
        """
    )

    # 3. データを安全に整形して挿入
    cleaned_movies = []
    for title, release_year, runtime, genres, countries, overview in raw_rows:
        if not title:
            continue
        safe_title = str(title).strip()
        safe_year = parse_safe_int(release_year, default=0)
        safe_runtime = max(0, parse_safe_int(runtime, default=0))
        safe_genres = str(genres).strip() if genres else "未分類"
        safe_country = extract_primary_country(countries)
        safe_overview = str(overview).strip() if overview else "（あらすじ情報なし）"

        cleaned_movies.append(
            (safe_title, safe_year, safe_runtime, safe_genres, safe_country, safe_overview)
        )

    dst_cursor.executemany(
        """
        INSERT INTO movies (title, release_year, runtime, genres, country, overview)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        cleaned_movies,
    )

    dst_conn.commit()
    dst_conn.close()
    print(f"完了: {len(cleaned_movies)} 件の映画データ（country列付き）を {TARGET_DB_PATH} に保存しました。")


if __name__ == "__main__":
    build_database()
