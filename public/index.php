<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>SCOUTIS</title>
  <link rel="stylesheet" href="assets/style.css">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>

<header>
  <h1>SCOUTIS</h1>
  <span class="subtitle">Autonomous Scout AI</span>
</header>

<main>
  <section class="map">
    <h2>Mapa da Área</h2>
    <div id="sectors"></div>
  </section>

  <section class="details">
    <h2 id="sectorTitle">Selecione um setor</h2>
    <p id="cause"></p>

    <canvas id="chartAnomaly"></canvas>
    <canvas id="chartStability"></canvas>
    <canvas id="chartObjects"></canvas>
  </section>
</main>

<script src="assets/app.js"></script>
<script src="assets/charts.js"></script>
</body>
</html>
