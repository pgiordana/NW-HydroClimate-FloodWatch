const COLORS = { GREEN: '#2f8f58', YELLOW: '#d39c1e', RED: '#c84141', GRAY: '#7d8a94' };
const STATUS_LABELS = {
  GREEN: 'Verde · segnale ordinario',
  YELLOW: 'Giallo · da approfondire',
  RED: 'Rosso · segnale elevato',
  GRAY: 'Grigio · non interpretabile'
};

const $ = id => document.getElementById(id);
const escapeHtml = value => String(value ?? '')
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;').replaceAll("'", '&#039;');
const pct = value => value == null || Number.isNaN(Number(value)) ? '—' : `${(100 * Number(value)).toFixed(1)}%`;

function formatDate(value) {
  if (!value) return '—';
  const d = new Date(`${value}T12:00:00Z`);
  return new Intl.DateTimeFormat('it-IT', { day: '2-digit', month: 'long', year: 'numeric' }).format(d);
}
function formatDateTime(value) {
  if (!value) return '—';
  const d = new Date(value);
  return new Intl.DateTimeFormat('it-IT', { dateStyle: 'medium', timeStyle: 'short' }).format(d);
}
function statusOf(r) { return COLORS[r?.overall_semaphore] ? r.overall_semaphore : 'GRAY'; }
function statusBadge(status) {
  const s = COLORS[status] ? status : 'GRAY';
  return `<span class="badge ${s}">${escapeHtml(STATUS_LABELS[s])}</span>`;
}
function technicalSuffix(data) {
  return data.scientific_beta_interpretation ? '' : 'Valore tecnico non interpretabile';
}

let map;
let receptorLayer;
let receptorById = new Map();
let layerById = new Map();
let latestData;

function fillHeader(data) {
  $('issueDate').textContent = formatDate(data.issue_date);
  $('runId').textContent = data.run_id || '—';
  $('count').textContent = (data.receptors || []).length || 20;
  $('runMode').textContent = data.bulletin_mode || '—';
  $('generatedAt').textContent = formatDateTime(data.generated_at_utc);
  $('modelInference').textContent = data.model_inference_performed ? 'ESEGUITA' : 'NON ESEGUITA';
  $('betaAllowed').textContent = data.prospective_beta_allowed ? 'AUTORIZZATO' : 'NON AUTORIZZATO';

  const interpretable = Boolean(data.scientific_beta_interpretation);
  $('science').textContent = interpretable ? 'INTERPRETABILE' : 'NON INTERPRETARE';
  $('livePill').className = `pill ${interpretable ? 'green' : 'gray'}`;
  $('livePill').textContent = interpretable ? 'INTERPRETABILE' : 'NON INTERPRETABILE';
  $('statusMessage').textContent = interpretable
    ? 'Il gate operativo consente l’interpretazione sperimentale di questo run. Resta comunque un prodotto non ufficiale.'
    : 'Il gate operativo non autorizza l’interpretazione di questo run come previsione di piena.';

  $('seasonNotice').hidden = Boolean(data.in_core_season && interpretable);

  const summary = $('summary');
  summary.innerHTML = '';
  const items = (data.summary || []).length ? data.summary : ['Nessuna nota tecnica disponibile.'];
  items.forEach(item => {
    const li = document.createElement('li');
    li.textContent = item;
    summary.appendChild(li);
  });

  if (data.bulletin_pdf) {
    ['pdf', 'heroPdf', 'navPdf'].forEach(id => {
      const el = $(id);
      el.href = data.bulletin_pdf;
      el.removeAttribute('aria-disabled');
      if (id === 'pdf') el.hidden = false;
    });
  }
}

function renderTable(data, filter = '') {
  const body = $('rows');
  const query = filter.trim().toLocaleLowerCase('it');
  body.innerHTML = '';
  (data.receptors || [])
    .filter(r => `${r.label || ''} ${r.receptor_id || ''}`.toLocaleLowerCase('it').includes(query))
    .forEach(r => {
      const status = statusOf(r);
      const tr = document.createElement('tr');
      tr.dataset.receptorId = r.receptor_id;
      tr.innerHTML = `
        <td><strong>${escapeHtml(r.label || r.receptor_id)}</strong><br><small>${escapeHtml(r.receptor_id)}</small></td>
        <td class="prob ${data.scientific_beta_interpretation ? '' : 'muted-value'}">${pct(r.probability_24h)}</td>
        <td class="prob ${data.scientific_beta_interpretation ? '' : 'muted-value'}">${pct(r.probability_48h)}</td>
        <td class="prob ${data.scientific_beta_interpretation ? '' : 'muted-value'}">${pct(r.probability_72h)}</td>
        <td>${statusBadge(status)}</td>
        <td class="note-cell">${escapeHtml(r.action_note || '—')}</td>`;
      tr.addEventListener('click', () => selectBasin(r.receptor_id, true));
      body.appendChild(tr);
    });
}

function basinPanelHtml(r, data) {
  const status = statusOf(r);
  const notInterpretable = !data.scientific_beta_interpretation;
  return `
    <div class="basin-head">
      <div><div class="basin-id">${escapeHtml(r.receptor_id)}</div><h3>${escapeHtml(r.label || r.receptor_id)}</h3></div>
      ${statusBadge(status)}
    </div>
    <div class="prob-grid">
      <div><span>24 ore</span><strong>${pct(r.probability_24h)}</strong></div>
      <div><span>48 ore</span><strong>${pct(r.probability_48h)}</strong></div>
      <div><span>72 ore</span><strong>${pct(r.probability_72h)}</strong></div>
    </div>
    <div class="panel-note">${escapeHtml(r.action_note || 'Nessuna nota disponibile.')}</div>
    ${notInterpretable ? '<div class="panel-warning"><strong>Attenzione:</strong> i valori di questo run sono prodotti tecnici ma il gate non ne autorizza l’interpretazione scientifica.</div>' : ''}
  `;
}

function selectBasin(id, zoom = false) {
  const r = receptorById.get(id);
  if (!r) return;
  $('basinPanel').innerHTML = basinPanelHtml(r, latestData);
  const layer = layerById.get(id);
  if (layer) {
    layer.openPopup();
    if (zoom) {
      const bounds = layer.getBounds?.();
      if (bounds?.isValid()) map.fitBounds(bounds.pad(0.28), { maxZoom: 9, animate: true });
    }
  }
}

function popupHtml(r, data) {
  if (!r) return '<strong>Bacino non supervisionato</strong>';
  const status = statusOf(r);
  return `
    <div class="map-popup-title">${escapeHtml(r.label || r.receptor_id)}</div>
    <div class="map-popup-status">${statusBadge(status)}</div>
    <div class="map-popup-values">
      <span>24 h<b>${pct(r.probability_24h)}</b></span>
      <span>48 h<b>${pct(r.probability_48h)}</b></span>
      <span>72 h<b>${pct(r.probability_72h)}</b></span>
    </div>
    <small>${escapeHtml(technicalSuffix(data))}</small>`;
}

async function buildMap(data) {
  map = L.map('map', { scrollWheelZoom: false, zoomControl: true, preferCanvas: true }).setView([44.7, 8.0], 7);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    attribution: '© OpenStreetMap contributors'
  }).addTo(map);

  const geojson = await fetch('data/receptors.geojson', { cache: 'no-store' }).then(r => {
    if (!r.ok) throw new Error(`GeoJSON HTTP ${r.status}`);
    return r.json();
  });

  receptorLayer = L.geoJSON(geojson, {
    style: feature => {
      const r = receptorById.get(feature.properties?.receptor_id);
      const status = statusOf(r);
      return { color: '#ffffff', weight: 1.25, opacity: .95, fillColor: COLORS[status], fillOpacity: .74 };
    },
    onEachFeature: (feature, layer) => {
      const id = feature.properties?.receptor_id;
      if (id) layerById.set(id, layer);
      const r = receptorById.get(id);
      layer.bindTooltip(escapeHtml(r?.label || feature.properties?.label || id || 'Bacino'), { sticky: true, direction: 'top' });
      layer.bindPopup(popupHtml(r, data));
      layer.on({
        click: () => id && selectBasin(id, false),
        mouseover: e => e.target.setStyle({ weight: 2.4, fillOpacity: .86 }),
        mouseout: e => receptorLayer.resetStyle(e.target)
      });
    }
  }).addTo(map);

  if (receptorLayer.getBounds().isValid()) map.fitBounds(receptorLayer.getBounds(), { padding: [18, 18] });
  setTimeout(() => map.invalidateSize(), 80);
}

async function main() {
  latestData = await fetch('data/latest.json', { cache: 'no-store' }).then(r => {
    if (!r.ok) throw new Error(`latest.json HTTP ${r.status}`);
    return r.json();
  });

  if (latestData.official_warning_use_allowed !== false || latestData.experimental !== true) {
    throw new Error('Payload safety flags are not compatible with the public experimental site.');
  }

  receptorById = new Map((latestData.receptors || []).map(r => [r.receptor_id, r]));
  fillHeader(latestData);
  renderTable(latestData);
  await buildMap(latestData);

  $('basinSearch').addEventListener('input', e => renderTable(latestData, e.target.value));
}

main().catch(error => {
  console.error(error);
  $('livePill').className = 'pill red';
  $('livePill').textContent = 'DATI NON DISPONIBILI';
  $('science').textContent = 'NON INTERPRETARE';
  $('statusMessage').textContent = 'Impossibile caricare o validare i dati del run. Non utilizzare questa pagina per alcuna interpretazione.';
  $('seasonNotice').hidden = false;
});
