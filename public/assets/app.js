let selected = null;

async function loadState() {
  const r = await fetch("../api/state.php");
  const data = await r.json();

  const box = document.getElementById("sectors");
  box.innerHTML = "";

  Object.values(data).forEach(s => {
    const d = document.createElement("div");
    d.className = `sector ${s.status.toLowerCase()}`;
    d.innerText = s.sector;
    d.onclick = () => selectSector(s.sector);
    box.appendChild(d);
  });
}

async function selectSector(sector) {
  selected = sector;
  document.getElementById("sectorTitle").innerText = `Setor ${sector}`;

  const r = await fetch(`../api/history.php?sector=${sector}`);
  const h = await r.json();

  updateCharts(h);

  const state = await fetch("../api/state.php").then(r => r.json());
  document.getElementById("cause").innerText = state[sector].cause;
}

setInterval(loadState, 2000);
loadState();
