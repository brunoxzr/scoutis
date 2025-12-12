const cfg = { responsive: true, plugins: { legend: { display: false } } };

const anomalyChart = new Chart(document.getElementById("chartAnomaly"), {
  type: "line",
  data: { labels: [], datasets: [
    { data: [], borderWidth: 2 },
    { data: [], borderDash: [6,6], borderWidth: 2 }
  ]},
  options: cfg
});

const stabilityChart = new Chart(document.getElementById("chartStability"), {
  type: "line",
  data: { labels: [], datasets: [{ data: [], borderWidth: 2 }] },
  options: cfg
});

const objectChart = new Chart(document.getElementById("chartObjects"), {
  type: "bar",
  data: { labels: [], datasets: [{ data: [] }] },
  options: cfg
});

function updateCharts(h) {
  const labels = h.map(p => new Date(p.t * 1000).toLocaleTimeString());

  anomalyChart.data.labels = labels;
  anomalyChart.data.datasets[0].data = h.map(p => p.anomaly_score);
  anomalyChart.data.datasets[1].data = h.map(p => p.anomaly_threshold);
  anomalyChart.update();

  stabilityChart.data.labels = labels;
  stabilityChart.data.datasets[0].data = h.map(p => p.stability_score);
  stabilityChart.update();

  objectChart.data.labels = labels;
  objectChart.data.datasets[0].data = h.map(p => p.objects);
  objectChart.update();
}
