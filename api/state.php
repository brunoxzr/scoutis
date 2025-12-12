<?php
require "../app/db.php";
header("Content-Type: application/json");

$sql = "
SELECT DISTINCT ON (sector_code)
  sector_code,
  status,
  health,
  stability_score,
  anomaly_score,
  anomaly_threshold,
  yolo_counts,
  cause,
  extract(epoch from ts)::int as ts
FROM sector_metrics
ORDER BY sector_code, ts DESC
";

$data = [];
foreach ($pdo->query($sql) as $r) {
  $data[$r["sector_code"]] = [
    "sector" => $r["sector_code"],
    "status" => $r["status"],
    "health" => (int)$r["health"],
    "stability" => (float)$r["stability_score"],
    "anomaly" => (float)$r["anomaly_score"],
    "threshold" => (float)$r["anomaly_threshold"],
    "objects" => $r["yolo_counts"]["objetos"] ?? 0,
    "cause" => $r["cause"],
    "ts" => $r["ts"]
  ];
}

echo json_encode($data);
