const ids = ['state','epoch','epochTotal','batch','batchTotal','loss','batchSpeed','timePerBatch','elapsed','eta','epochProgress','epochBar','overallProgress','overallBar','batchSummary','lastUpdate','batchSize','imagesProcessed','coverageBar','datasetProgress','imageSpeed','currentStat','movingAverage','minimumLoss','averageLoss','lossChange','validationLoss','validationAuc','validationF1','logStatus','gaugeSpeed','currentSpeed','averageSpeed','peakSpeed','gaugeBatchSpeed','gaugeTimeBatch','gaugeBatch'];
const ui = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));
const labels = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia','Infiltration','Mass','Nodule','Pleural Thickening','Pneumonia','Pneumothorax','No Finding'];
document.getElementById('labelCloud').innerHTML = labels.map((label) => `<span>${label}</span>`).join('');

const duration = (seconds) => {
  if (seconds === null || seconds === undefined) return '--';
  const value = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const remainder = value % 60;
  return hours ? `${hours}h ${minutes}m` : minutes ? `${minutes}m ${remainder}s` : `${remainder}s`;
};
const number = (value, digits = 4) => value === null || value === undefined ? '--' : Number(value).toFixed(digits);
const percent = (value) => `${Number(value || 0).toFixed(2)}%`;
const statusClass = (state) => state === 'TRAINING' ? 'live-badge' : state === 'WAITING' ? 'waiting-badge' : state === 'COMPLETED' ? 'complete-badge' : 'unknown-badge';

const chartOptions = (xTitle, yTitle) => ({ responsive: true, maintainAspectRatio: false, animation: { duration: 250 }, interaction: { intersect: false, mode: 'index' }, plugins: { legend: { labels: { color: '#b9cbe0', usePointStyle: true } } }, scales: { x: { title: { display: true, text: xTitle, color: '#8ca1ba' }, ticks: { color: '#8ca1ba', maxTicksLimit: 12 }, grid: { color: 'rgba(153,181,214,.08)' } }, y: { title: { display: true, text: yTitle, color: '#8ca1ba' }, ticks: { color: '#8ca1ba' }, grid: { color: 'rgba(153,181,214,.08)' } } } });
const lossChart = new Chart(document.getElementById('lossChart'), { type: 'line', data: { labels: [], datasets: [{ label: 'Raw training loss', data: [], borderColor: '#ffb86b', borderWidth: 1.5, pointRadius: 0, tension: .15 }, { label: 'Moving average loss', data: [], borderColor: '#4de1d2', borderWidth: 3, pointRadius: 0, tension: .25 }] }, options: chartOptions('Global batch step', 'Loss') });
const epochChart = new Chart(document.getElementById('epochChart'), { type: 'line', data: { labels: [], datasets: [{ label: 'Epoch mean loss', data: [], borderColor: '#b995ff', backgroundColor: 'rgba(185,149,255,.15)', fill: true, borderWidth: 2, pointRadius: 4, tension: .25 }] }, options: chartOptions('Epoch', 'Mean loss') });

const gaugeCanvas = document.getElementById('speedGauge');
const gaugeContext = gaugeCanvas.getContext('2d');
let gaugeValue = 0;
let gaugeTarget = 0;
let gaugeMaximum = 150;
let gaugeMaximumTarget = 150;

function drawGauge() {
  const scale = Math.min(1, gaugeCanvas.clientWidth / gaugeCanvas.width || 1);
  const width = gaugeCanvas.width;
  const height = gaugeCanvas.height;
  gaugeContext.clearRect(0, 0, width, height);
  const centerX = width / 2;
  const centerY = height * .82;
  const radius = Math.min(width * .39, height * .75);
  const start = Math.PI;
  const end = 0;
  gaugeContext.lineCap = 'round';
  gaugeContext.lineWidth = 22;
  gaugeContext.strokeStyle = '#13243a';
  gaugeContext.beginPath(); gaugeContext.arc(centerX, centerY, radius, start, end); gaugeContext.stroke();
  gaugeContext.lineWidth = 5;
  const progressEnd = start + (end - start) * Math.min(1, gaugeValue / gaugeMaximum);
  const gradient = gaugeContext.createLinearGradient(centerX - radius, centerY, centerX + radius, centerY);
  gradient.addColorStop(0, '#ff8eae'); gradient.addColorStop(.5, '#ffb86b'); gradient.addColorStop(1, '#4de1d2');
  gaugeContext.strokeStyle = gradient;
  gaugeContext.beginPath(); gaugeContext.arc(centerX, centerY, radius, start, progressEnd); gaugeContext.stroke();
  gaugeContext.font = '600 13px ui-sans-serif'; gaugeContext.fillStyle = '#8ea4bd'; gaugeContext.textAlign = 'center';
  for (let index = 0; index <= 10; index += 1) {
    const angle = start + (end - start) * index / 10;
    const inner = radius - 30; const outer = radius - 12;
    gaugeContext.strokeStyle = index % 2 ? '#54708f' : '#c8d9ed'; gaugeContext.lineWidth = index % 2 ? 2 : 4;
    gaugeContext.beginPath(); gaugeContext.moveTo(centerX + Math.cos(angle) * inner, centerY + Math.sin(angle) * inner); gaugeContext.lineTo(centerX + Math.cos(angle) * outer, centerY + Math.sin(angle) * outer); gaugeContext.stroke();
    if (index % 2 === 0) gaugeContext.fillText(Math.round(gaugeMaximum * index / 10), centerX + Math.cos(angle) * (radius - 49), centerY + Math.sin(angle) * (radius - 49) + 4);
  }
  const gaugePercentage = Math.max(0, Math.min(1, gaugeMaximum > 0 ? gaugeValue / gaugeMaximum : 0));
  const needleAngle = start + Math.PI * gaugePercentage;
  gaugeContext.strokeStyle = '#f1f7ff'; gaugeContext.lineWidth = 4; gaugeContext.shadowColor = '#4de1d2'; gaugeContext.shadowBlur = 12;
  gaugeContext.beginPath(); gaugeContext.moveTo(centerX, centerY); gaugeContext.lineTo(centerX + Math.cos(needleAngle) * (radius - 42), centerY + Math.sin(needleAngle) * (radius - 42)); gaugeContext.stroke(); gaugeContext.shadowBlur = 0;
  gaugeContext.fillStyle = '#4de1d2'; gaugeContext.beginPath(); gaugeContext.arc(centerX, centerY, 9, 0, Math.PI * 2); gaugeContext.fill();
}
function animateGauge() {
  gaugeValue += (gaugeTarget - gaugeValue) * .08;
  gaugeMaximum += (gaugeMaximumTarget - gaugeMaximum) * .025;
  drawGauge();
  requestAnimationFrame(animateGauge);
}
animateGauge();

async function refresh() {
  try {
    const response = await fetch('/api/status', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    ui.state.textContent = data.state || 'UNKNOWN';
    document.querySelector('.live-badge').className = `live-badge ${statusClass(data.state)}`;
    ui.epoch.textContent = data.epoch ?? '--'; ui.epochTotal.textContent = `of ${data.total_epochs ?? '--'}`;
    ui.batch.textContent = data.batch ?? '--'; ui.batchTotal.textContent = `of ${data.total_batches ?? '--'}`;
    ui.loss.textContent = number(data.loss); ui.batchSpeed.textContent = number(data.batches_per_second, 3); ui.timePerBatch.textContent = `${number(data.time_per_batch, 2)} sec / batch`;
    ui.elapsed.textContent = duration(data.elapsed_seconds); ui.eta.textContent = duration(data.eta_seconds);
    ui.epochProgress.textContent = percent(data.epoch_progress); ui.epochBar.style.width = `${data.epoch_progress || 0}%`;
    ui.overallProgress.textContent = percent(data.overall_progress); ui.overallBar.style.width = `${data.overall_progress || 0}%`;
    ui.batchSummary.textContent = `Epoch ${data.epoch ?? '--'} · batch ${data.batch ?? '--'} / ${data.total_batches ?? '--'}`; ui.lastUpdate.textContent = `Last update: ${data.last_update || '--'}`;
    ui.batchSize.textContent = data.batch_size ?? '--'; ui.imagesProcessed.textContent = `${(data.images_processed ?? 0).toLocaleString()} / ~${data.training_subset_size.toLocaleString()}`;
    ui.coverageBar.style.width = `${Math.min(100, data.dataset_progress || 0)}%`; ui.datasetProgress.textContent = percent(data.dataset_progress); ui.imageSpeed.textContent = number(data.images_per_second, 2);
    ui.currentStat.textContent = number(data.loss); ui.movingAverage.textContent = number(data.moving_average_loss); ui.minimumLoss.textContent = number(data.minimum_loss); ui.averageLoss.textContent = number(data.average_loss);
    ui.lossChange.textContent = number(data.change_from_previous); ui.lossChange.style.color = data.change_from_previous > 0 ? '#ff8eae' : '#73e6a3';
    gaugeTarget = data.current_images_per_second ?? 0;

const reportedGaugeMax = Number(data.gauge_max) || 40;

// Keep the gauge responsive to the actual training throughput.
// Never fall back to 150 just because the current speed is low.
gaugeMaximumTarget = Math.max(40, reportedGaugeMax);
    ui.gaugeSpeed.textContent = number(data.current_images_per_second, 1); ui.currentSpeed.textContent = `${number(data.current_images_per_second, 1)} images/sec`; ui.averageSpeed.textContent = `${number(data.average_images_per_second, 1)} images/sec`; ui.peakSpeed.textContent = `${number(data.peak_images_per_second, 1)} images/sec`; ui.gaugeBatchSpeed.textContent = number(data.batches_per_second, 3); ui.gaugeTimeBatch.textContent = `${number(data.time_per_batch, 3)} sec`; ui.gaugeBatch.textContent = `${data.batch ?? '--'} / ${data.total_batches ?? '--'}`;
    ui.validationLoss.textContent = data.validation_loss === null ? 'Not currently logged' : number(data.validation_loss); ui.validationAuc.textContent = data.validation_auc === null ? 'Not currently logged' : number(data.validation_auc); ui.validationF1.textContent = data.validation_f1 === null ? 'Not currently logged' : number(data.validation_f1);
    ui.logStatus.textContent = data.log_status || 'Unknown log status';
    lossChart.data.labels = data.loss_history.map((point) => point.step); lossChart.data.datasets[0].data = data.loss_history.map((point) => point.loss); lossChart.data.datasets[1].data = data.smoothed_loss_history.map((point) => point.loss); lossChart.update('none');
    const epochs = Object.entries(data.epoch_history); epochChart.data.labels = epochs.map(([epoch]) => `Epoch ${epoch}`); epochChart.data.datasets[0].data = epochs.map(([, value]) => value.average_loss); epochChart.update('none');
  } catch (error) { ui.state.textContent = 'OFFLINE'; ui.logStatus.textContent = error.message; }
}
refresh(); setInterval(refresh, 1500);
