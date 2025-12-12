from __future__ import annotations
import psycopg2
import psycopg2.extras
from pathlib import Path
from typing import Iterable
from config import PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASS

def get_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )

def apply_schema(conn, schema_path: str | Path):
    sql = Path(schema_path).read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()

def seed_sectors(conn, sectors: Iterable[str]):
    with conn.cursor() as cur:
        for code in sectors:
            cur.execute(
                "insert into sectors(code, name) values (%s, %s) on conflict(code) do nothing",
                (code, f"Setor {code}")
            )
    conn.commit()

def insert_metric(conn, row: dict):
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into sector_metrics
            (sector_code, health, stability_score, anomaly_score, anomaly_threshold,
             yolo_counts, status, cause, model_version, camera_source)
            values
            (%(sector_code)s, %(health)s, %(stability_score)s, %(anomaly_score)s, %(anomaly_threshold)s,
             %(yolo_counts)s, %(status)s, %(cause)s, %(model_version)s, %(camera_source)s)
            """,
            {
                **row,
                "yolo_counts": psycopg2.extras.Json(row["yolo_counts"])
            }
        )
    conn.commit()
