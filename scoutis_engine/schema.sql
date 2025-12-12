CREATE TABLE IF NOT EXISTS sectors (
    code VARCHAR(10) PRIMARY KEY,
    name TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sector_metrics (
    id SERIAL PRIMARY KEY,
    sector_code VARCHAR(10) REFERENCES sectors(code),
    health INTEGER NOT NULL,
    stability_score REAL NOT NULL,
    anomaly_score REAL NOT NULL,
    anomaly_threshold REAL NOT NULL,
    yolo_counts JSONB NOT NULL,
    status VARCHAR(20) NOT NULL,
    cause TEXT NOT NULL,
    model_version VARCHAR(50),
    camera_source TEXT,
    ts TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sector_metrics_sector_ts
ON sector_metrics (sector_code, ts DESC);
