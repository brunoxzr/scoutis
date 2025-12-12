<?php
require __DIR__ . "/../app/db.php";

header("Content-Type: text/event-stream");
header("Cache-Control: no-cache");
header("Connection: keep-alive");

while (true) {
  $res = $pdo->query("
    select distinct on (s.code)
      s.code as sector_id,
      m.status,
      m.health,
      m.stability_score,
      m.yolo_counts,
      m.cause,
      extract(epoch from m.ts)::int as updated_at
    from sectors s
    join sector_metrics m on m.sector_code = s.code
    order by s.code, m.ts desc
  ")->fetchAll();

  $payload = ["sectors" => []];
  foreach ($res as $r) {
    $payload["sectors"][$r["sector_id"]] = [
      "sector_id" => $r["sector_id"],
      "status" => $r["status"],
      "health" => (int)$r["health"],
      "stability_score" => (float)$r["stability_score"],
      "yolo_counts" => json_decode($r["yolo_counts"], true),
      "cause" => $r["cause"],
      "updated_at" => (int)$r["updated_at"],
    ];
  }

  echo "data: " . json_encode($payload) . "\n\n";
  ob_flush();
  flush();
  sleep(1);
}
