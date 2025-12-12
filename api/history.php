<?php
require "../app/db.php";
header("Content-Type: application/json");

$sector = $_GET["sector"] ?? "S11";

$stmt = $pdo->prepare("
SELECT
  extract(epoch from ts)::int as t,
  anomaly_score,
  anomaly_threshold,
  stability_score,
  (yolo_counts->>'objetos')::int as objects
FROM sector_metrics
WHERE sector_code = :s
ORDER BY ts DESC
LIMIT 120
");
$stmt->execute([":s" => $sector]);

echo json_encode(array_reverse($stmt->fetchAll()));
