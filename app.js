/**
 * =============================================
 *  ADALINE — Logam Mulia Price Prediction
 *  Adaptive Linear Neuron Implementation
 * =============================================
 */

/* -------- Utility Functions -------- */

/** Format IDR currency */
function formatRupiah(value) {
  return new Intl.NumberFormat('id-ID', {
    style: 'currency',
    currency: 'IDR',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

/** Format compact number */
function formatCompact(value) {
  if (value >= 1_000_000) return (value / 1_000_000).toFixed(1) + ' jt';
  if (value >= 1_000) return (value / 1_000).toFixed(0) + ' rb';
  return value.toString();
}

/** Format date from timestamp */
function formatDate(timestamp) {
  const d = new Date(timestamp * 1000);
  return d.toLocaleDateString('id-ID', { day: 'numeric', month: 'short', year: 'numeric' });
}

/** Format short date */
function formatDateShort(timestamp) {
  const d = new Date(timestamp * 1000);
  return d.toLocaleDateString('id-ID', { month: 'short', year: 'numeric' });
}

/* -------- Data Processing -------- */

/** Parse CSV text into array of {time, price} objects */
function parseCSV(text) {
  const lines = text.trim().split('\n');
  const data = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(',');
    if (parts.length >= 2) {
      const time = parseInt(parts[0], 10);
      const price = parseFloat(parts[1]);
      if (!isNaN(time) && !isNaN(price)) {
        data.push({ time, price });
      }
    }
  }
  return data;
}

/** Deduplicate data — keep last entry per day */
function deduplicateDaily(data) {
  const map = new Map();
  for (const d of data) {
    const dayKey = Math.floor(d.time / 86400);
    map.set(dayKey, d);
  }
  const result = Array.from(map.values());
  result.sort((a, b) => a.time - b.time);
  return result;
}

/** Min-Max normalization */
function normalize(values) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  return {
    normalized: values.map(v => (v - min) / range),
    min,
    max,
    range,
  };
}

/** Denormalize a single value */
function denormalize(value, min, range) {
  return value * range + min;
}

/** Create sliding window dataset */
function createWindows(normalizedPrices, windowSize) {
  const X = [];
  const y = [];
  for (let i = 0; i <= normalizedPrices.length - windowSize - 1; i++) {
    X.push(normalizedPrices.slice(i, i + windowSize));
    y.push(normalizedPrices[i + windowSize]);
  }
  return { X, y };
}

/* -------- Adaline Class -------- */

class Adaline {
  constructor(inputSize) {
    this.inputSize = inputSize;
    this.weights = new Array(inputSize).fill(0).map(() => (Math.random() - 0.5) * 0.1);
    this.bias = (Math.random() - 0.5) * 0.1;
    this.mseHistory = [];
  }

  /** Compute net input (linear activation) */
  netInput(x) {
    let sum = this.bias;
    for (let i = 0; i < this.inputSize; i++) {
      sum += this.weights[i] * x[i];
    }
    return sum;
  }

  /** Predict output for a single input */
  predict(x) {
    return this.netInput(x);
  }

  /**
   * Train the Adaline network.
   * Uses Widrow-Hoff (LMS) learning rule.
   * @param {number[][]} X - Training inputs
   * @param {number[]} y - Target outputs
   * @param {number} learningRate - Alpha
   * @param {number} epochs - Number of iterations
   * @param {function} onEpoch - Callback per epoch (epoch, mse)
   * @returns {{ mseHistory: number[] }}
   */
  train(X, y, learningRate, epochs, onEpoch) {
    this.mseHistory = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      let sumSquaredError = 0;

      for (let i = 0; i < X.length; i++) {
        const output = this.netInput(X[i]);
        const error = y[i] - output;

        // Update weights: w += α * error * x
        for (let j = 0; j < this.inputSize; j++) {
          this.weights[j] += learningRate * error * X[i][j];
        }
        // Update bias: b += α * error
        this.bias += learningRate * error;

        sumSquaredError += error * error;
      }

      const mse = sumSquaredError / X.length;
      this.mseHistory.push(mse);

      if (onEpoch) {
        onEpoch(epoch, mse);
      }
    }

    return { mseHistory: this.mseHistory };
  }

  /** Calculate MSE on a test set */
  calculateMSE(X, y) {
    let sum = 0;
    for (let i = 0; i < X.length; i++) {
      const err = y[i] - this.predict(X[i]);
      sum += err * err;
    }
    return sum / X.length;
  }

  /** Calculate MAPE on denormalized values */
  calculateMAPE(X, yActual, min, range) {
    let sum = 0;
    let count = 0;
    for (let i = 0; i < X.length; i++) {
      const predicted = denormalize(this.predict(X[i]), min, range);
      const actual = denormalize(yActual[i], min, range);
      if (actual !== 0) {
        sum += Math.abs((actual - predicted) / actual);
        count++;
      }
    }
    return (sum / count) * 100;
  }

  /** Get the weights and bias */
  getParameters() {
    return {
      weights: [...this.weights],
      bias: this.bias,
    };
  }
}

/* -------- Application State -------- */

const state = {
  rawData: [],
  dailyData: [],
  normalizedPrices: [],
  normParams: null,
  model: null,
  trained: false,
  charts: {},
};

/* -------- DOM References -------- */

const DOM = {
  // Stats
  totalData: document.getElementById('totalData'),
  latestPrice: document.getElementById('latestPrice'),
  lowestPrice: document.getElementById('lowestPrice'),
  highestPrice: document.getElementById('highestPrice'),
  priceChange: document.getElementById('priceChange'),
  dateRange: document.getElementById('dateRange'),

  // Charts
  historyChart: document.getElementById('historyChart'),
  mseChart: document.getElementById('mseChart'),
  comparisonChart: document.getElementById('comparisonChart'),

  // Form
  learningRate: document.getElementById('learningRate'),
  epochs: document.getElementById('epochs'),
  windowSize: document.getElementById('windowSize'),
  trainRatio: document.getElementById('trainRatio'),

  // Buttons
  btnTrain: document.getElementById('btnTrain'),
  btnPredict: document.getElementById('btnPredict'),
  btnReset: document.getElementById('btnReset'),

  // Training results
  trainingSection: document.getElementById('trainingResults'),
  progressBar: document.getElementById('progressFill'),
  trainingStatus: document.getElementById('trainingStatus'),
  statusDot: document.getElementById('statusDot'),
  epochInfo: document.getElementById('epochInfo'),

  // Metrics
  metricMSE: document.getElementById('metricMSE'),
  metricMAPE: document.getElementById('metricMAPE'),
  metricAccuracy: document.getElementById('metricAccuracy'),

  // Weights table
  weightsBody: document.getElementById('weightsBody'),

  // Prediction
  predictionSection: document.getElementById('predictionSection'),
  predictionValue: document.getElementById('predictionValue'),
  predictionDate: document.getElementById('predictionDate'),
  predictionInput: document.getElementById('predictionInput'),
};

/* -------- Chart Configuration -------- */

const CHART_COLORS = {
  gold: 'rgba(251, 191, 36, 1)',
  goldFaded: 'rgba(251, 191, 36, 0.15)',
  amber: 'rgba(245, 158, 11, 1)',
  amberFaded: 'rgba(245, 158, 11, 0.1)',
  green: 'rgba(52, 211, 153, 1)',
  greenFaded: 'rgba(52, 211, 153, 0.1)',
  red: 'rgba(248, 113, 113, 1)',
  redFaded: 'rgba(248, 113, 113, 0.1)',
  blue: 'rgba(96, 165, 250, 1)',
  blueFaded: 'rgba(96, 165, 250, 0.1)',
  gridColor: 'rgba(255, 255, 255, 0.04)',
  tickColor: 'rgba(255, 255, 255, 0.35)',
};

const chartDefaults = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: {
        color: CHART_COLORS.tickColor,
        font: { family: 'Inter', size: 11 },
        usePointStyle: true,
        pointStyle: 'circle',
        padding: 16,
      },
    },
    tooltip: {
      backgroundColor: 'rgba(10, 10, 15, 0.9)',
      titleColor: '#fbbf24',
      bodyColor: '#f1f1f4',
      borderColor: 'rgba(251, 191, 36, 0.2)',
      borderWidth: 1,
      cornerRadius: 8,
      padding: 12,
      titleFont: { family: 'Inter', weight: '600' },
      bodyFont: { family: 'JetBrains Mono', size: 12 },
    },
  },
  scales: {
    x: {
      grid: { color: CHART_COLORS.gridColor },
      ticks: { color: CHART_COLORS.tickColor, font: { family: 'Inter', size: 10 } },
    },
    y: {
      grid: { color: CHART_COLORS.gridColor },
      ticks: { color: CHART_COLORS.tickColor, font: { family: 'JetBrains Mono', size: 10 } },
    },
  },
};

/* -------- Initialize Application -------- */

async function init() {
  try {
    const response = await fetch('data.csv');
    const text = await response.text();
    state.rawData = parseCSV(text);
    state.dailyData = deduplicateDaily(state.rawData);

    updateStats();
    renderHistoryChart();
  } catch (err) {
    console.error('Failed to load data:', err);
    alert('Gagal memuat data.csv. Pastikan file tersedia.');
  }
}

/* -------- Update Stats Display -------- */

function updateStats() {
  const data = state.dailyData;
  const prices = data.map(d => d.price);
  const latest = data[data.length - 1];
  const oldest = data[0];
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const change = ((latest.price - oldest.price) / oldest.price * 100).toFixed(1);

  DOM.totalData.textContent = data.length.toLocaleString('id-ID');
  DOM.latestPrice.textContent = formatRupiah(latest.price);
  DOM.lowestPrice.textContent = formatRupiah(minPrice);
  DOM.highestPrice.textContent = formatRupiah(maxPrice);
  DOM.priceChange.textContent = (change > 0 ? '+' : '') + change + '%';
  DOM.priceChange.className = 'stat-card__change ' + (change > 0 ? 'stat-card__change--up' : 'stat-card__change--down');
  DOM.dateRange.textContent = formatDate(oldest.time) + ' — ' + formatDate(latest.time);
}

/* -------- Render History Chart -------- */

function renderHistoryChart() {
  const data = state.dailyData;

  // Sample for performance — show max ~500 points
  const step = Math.max(1, Math.floor(data.length / 500));
  const sampled = data.filter((_, i) => i % step === 0 || i === data.length - 1);

  const labels = sampled.map(d => formatDateShort(d.time));
  const prices = sampled.map(d => d.price);

  if (state.charts.history) state.charts.history.destroy();

  const ctx = DOM.historyChart.getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 0, 350);
  gradient.addColorStop(0, CHART_COLORS.goldFaded);
  gradient.addColorStop(1, 'rgba(251, 191, 36, 0)');

  state.charts.history = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Harga LM / gram',
        data: prices,
        borderColor: CHART_COLORS.gold,
        backgroundColor: gradient,
        borderWidth: 2,
        pointRadius: 0,
        pointHitRadius: 10,
        fill: true,
        tension: 0.3,
      }],
    },
    options: {
      ...chartDefaults,
      plugins: {
        ...chartDefaults.plugins,
        tooltip: {
          ...chartDefaults.plugins.tooltip,
          callbacks: {
            label: ctx => ' ' + formatRupiah(ctx.parsed.y),
          },
        },
      },
      scales: {
        ...chartDefaults.scales,
        y: {
          ...chartDefaults.scales.y,
          ticks: {
            ...chartDefaults.scales.y.ticks,
            callback: v => formatCompact(v),
          },
        },
        x: {
          ...chartDefaults.scales.x,
          ticks: {
            ...chartDefaults.scales.x.ticks,
            maxTicksLimit: 12,
            maxRotation: 45,
          },
        },
      },
    },
  });
}

/* -------- Training -------- */

async function startTraining() {
  const learningRate = parseFloat(DOM.learningRate.value);
  const epochs = parseInt(DOM.epochs.value, 10);
  const windowSize = parseInt(DOM.windowSize.value, 10);
  const trainRatio = parseInt(DOM.trainRatio.value, 10) / 100;

  // Validation
  if (learningRate <= 0 || learningRate > 1) {
    alert('Learning rate harus antara 0.0001 dan 1');
    return;
  }
  if (epochs < 1 || epochs > 10000) {
    alert('Epoch harus antara 1 dan 10000');
    return;
  }
  if (windowSize < 2 || windowSize > 30) {
    alert('Window size harus antara 2 dan 30');
    return;
  }
  if (trainRatio < 0.5 || trainRatio > 0.95) {
    alert('Rasio training harus antara 50% dan 95%');
    return;
  }

  // Disable controls
  DOM.btnTrain.disabled = true;
  DOM.btnTrain.innerHTML = '<span class="spinner"></span> Training...';

  // Show training section
  DOM.trainingSection.classList.remove('hidden');
  DOM.predictionSection.classList.add('hidden');
  DOM.statusDot.className = 'training-status__dot training-status__dot--active';
  DOM.trainingStatus.querySelector('span').textContent = 'Training sedang berlangsung...';

  // Prepare data
  const prices = state.dailyData.map(d => d.price);
  const norm = normalize(prices);
  state.normalizedPrices = norm.normalized;
  state.normParams = { min: norm.min, max: norm.max, range: norm.range };

  const { X, y } = createWindows(norm.normalized, windowSize);

  const splitIdx = Math.floor(X.length * trainRatio);
  const trainX = X.slice(0, splitIdx);
  const trainY = y.slice(0, splitIdx);
  const testX = X.slice(splitIdx);
  const testY = y.slice(splitIdx);

  // Create model
  const model = new Adaline(windowSize);
  state.model = model;

  // Train with animation — use setTimeout to avoid blocking UI
  const mseHistory = [];
  let epoch = 0;

  const trainStep = () => {
    // Process batch of epochs per frame for speed
    const batchSize = Math.max(1, Math.floor(epochs / 100));
    const endEpoch = Math.min(epoch + batchSize, epochs);

    for (; epoch < endEpoch; epoch++) {
      let sumSquaredError = 0;
      for (let i = 0; i < trainX.length; i++) {
        const output = model.netInput(trainX[i]);
        const error = trainY[i] - output;
        for (let j = 0; j < windowSize; j++) {
          model.weights[j] += learningRate * error * trainX[i][j];
        }
        model.bias += learningRate * error;
        sumSquaredError += error * error;
      }
      const mse = sumSquaredError / trainX.length;
      mseHistory.push(mse);
    }

    // Update progress
    const progress = (epoch / epochs * 100).toFixed(0);
    DOM.progressBar.style.width = progress + '%';
    DOM.epochInfo.textContent = `Epoch ${epoch}/${epochs} — MSE: ${mseHistory[mseHistory.length - 1].toExponential(4)}`;

    if (epoch < epochs) {
      requestAnimationFrame(trainStep);
    } else {
      model.mseHistory = mseHistory;
      onTrainingComplete(model, testX, testY, trainX, trainY, windowSize);
    }
  };

  requestAnimationFrame(trainStep);
}

function onTrainingComplete(model, testX, testY, trainX, trainY, windowSize) {
  const { min, range } = state.normParams;

  // Status
  DOM.statusDot.className = 'training-status__dot training-status__dot--done';
  DOM.trainingStatus.querySelector('span').textContent = 'Training selesai!';

  // Calculate metrics
  const testMSE = model.calculateMSE(testX, testY);
  const testMAPE = model.calculateMAPE(testX, testY, min, range);
  const accuracy = Math.max(0, 100 - testMAPE);

  DOM.metricMSE.textContent = testMSE.toExponential(4);
  DOM.metricMAPE.textContent = testMAPE.toFixed(2) + '%';
  DOM.metricAccuracy.textContent = accuracy.toFixed(2) + '%';

  // Render weights table
  renderWeightsTable(model, windowSize);

  // Render MSE chart
  renderMSEChart(model.mseHistory);

  // Render comparison chart
  renderComparisonChart(model, testX, testY, min, range, windowSize);

  // Enable prediction
  DOM.predictionSection.classList.remove('hidden');
  state.trained = true;

  // Auto-predict next price
  predictNext();

  // Re-enable button
  DOM.btnTrain.disabled = false;
  DOM.btnTrain.innerHTML = '⚡ Mulai Training';
}

/* -------- Render Weights Table -------- */

function renderWeightsTable(model, windowSize) {
  const params = model.getParameters();
  let html = '';

  for (let i = 0; i < windowSize; i++) {
    html += `<tr>
      <td>w${i + 1} (t-${windowSize - i})</td>
      <td>${params.weights[i].toFixed(6)}</td>
    </tr>`;
  }
  html += `<tr>
    <td style="color: var(--gold-400);">Bias</td>
    <td style="color: var(--gold-400);">${params.bias.toFixed(6)}</td>
  </tr>`;

  DOM.weightsBody.innerHTML = html;
}

/* -------- Render MSE Chart -------- */

function renderMSEChart(mseHistory) {
  if (state.charts.mse) state.charts.mse.destroy();

  // Sample MSE for chart readability
  const step = Math.max(1, Math.floor(mseHistory.length / 200));
  const sampled = mseHistory.filter((_, i) => i % step === 0 || i === mseHistory.length - 1);
  const labels = sampled.map((_, i) => (i * step + 1).toString());

  const ctx = DOM.mseChart.getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 0, 250);
  gradient.addColorStop(0, CHART_COLORS.redFaded);
  gradient.addColorStop(1, 'transparent');

  state.charts.mse = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'MSE per Epoch',
        data: sampled,
        borderColor: CHART_COLORS.red,
        backgroundColor: gradient,
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        tension: 0.4,
      }],
    },
    options: {
      ...chartDefaults,
      scales: {
        ...chartDefaults.scales,
        x: {
          ...chartDefaults.scales.x,
          title: { display: true, text: 'Epoch', color: CHART_COLORS.tickColor, font: { family: 'Inter', size: 11 } },
          ticks: { ...chartDefaults.scales.x.ticks, maxTicksLimit: 10 },
        },
        y: {
          ...chartDefaults.scales.y,
          title: { display: true, text: 'MSE', color: CHART_COLORS.tickColor, font: { family: 'Inter', size: 11 } },
          ticks: {
            ...chartDefaults.scales.y.ticks,
            callback: v => v.toExponential(1),
          },
        },
      },
    },
  });
}

/* -------- Render Comparison Chart -------- */

function renderComparisonChart(model, testX, testY, min, range, windowSize) {
  if (state.charts.comparison) state.charts.comparison.destroy();

  const actual = testY.map(v => denormalize(v, min, range));
  const predicted = testX.map(x => denormalize(model.predict(x), min, range));

  // Get the corresponding dates from daily data
  const totalData = state.dailyData.length;
  const trainRatio = parseInt(DOM.trainRatio.value, 10) / 100;
  const allPrices = state.dailyData.map(d => d.price);
  const { X: allX } = createWindows(state.normalizedPrices, windowSize);
  const splitIdx = Math.floor(allX.length * trainRatio);

  const labels = [];
  for (let i = 0; i < testY.length; i++) {
    const dataIdx = splitIdx + windowSize + i;
    if (dataIdx < state.dailyData.length) {
      labels.push(formatDateShort(state.dailyData[dataIdx].time));
    } else {
      labels.push('Hari ' + (i + 1));
    }
  }

  // Sample if too many points
  const step = Math.max(1, Math.floor(actual.length / 300));
  const sLabels = labels.filter((_, i) => i % step === 0);
  const sActual = actual.filter((_, i) => i % step === 0);
  const sPredicted = predicted.filter((_, i) => i % step === 0);

  const ctx = DOM.comparisonChart.getContext('2d');

  state.charts.comparison = new Chart(ctx, {
    type: 'line',
    data: {
      labels: sLabels,
      datasets: [
        {
          label: 'Harga Aktual',
          data: sActual,
          borderColor: CHART_COLORS.blue,
          backgroundColor: CHART_COLORS.blueFaded,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3,
        },
        {
          label: 'Prediksi Adaline',
          data: sPredicted,
          borderColor: CHART_COLORS.green,
          backgroundColor: CHART_COLORS.greenFaded,
          borderWidth: 2,
          pointRadius: 0,
          borderDash: [5, 3],
          tension: 0.3,
        },
      ],
    },
    options: {
      ...chartDefaults,
      plugins: {
        ...chartDefaults.plugins,
        tooltip: {
          ...chartDefaults.plugins.tooltip,
          callbacks: {
            label: ctx => ' ' + ctx.dataset.label + ': ' + formatRupiah(ctx.parsed.y),
          },
        },
      },
      scales: {
        ...chartDefaults.scales,
        y: {
          ...chartDefaults.scales.y,
          ticks: {
            ...chartDefaults.scales.y.ticks,
            callback: v => formatCompact(v),
          },
        },
        x: {
          ...chartDefaults.scales.x,
          ticks: { ...chartDefaults.scales.x.ticks, maxTicksLimit: 10 },
        },
      },
    },
  });
}

/* -------- Prediction -------- */

function predictNext() {
  if (!state.trained || !state.model) return;

  const windowSize = parseInt(DOM.windowSize.value, 10);
  const { min, range } = state.normParams;

  // Use last `windowSize` normalized prices as input
  const lastWindow = state.normalizedPrices.slice(-windowSize);
  const normalizedPrediction = state.model.predict(lastWindow);
  const predictedPrice = denormalize(normalizedPrediction, min, range);

  // The prediction is for the next day after the last data point
  const lastDate = state.dailyData[state.dailyData.length - 1].time;
  const nextDate = lastDate + 86400; // +1 day

  DOM.predictionValue.textContent = formatRupiah(Math.round(predictedPrice));
  DOM.predictionDate.textContent = 'Prediksi untuk ' + formatDate(nextDate);

  // Fill prediction input with the last window prices
  const lastPrices = state.dailyData.slice(-windowSize).map(d => d.price);
  DOM.predictionInput.value = lastPrices.join(', ');
}

function predictCustom() {
  if (!state.trained || !state.model) {
    alert('Model belum di-training. Silakan training terlebih dahulu.');
    return;
  }

  const windowSize = parseInt(DOM.windowSize.value, 10);
  const { min, range } = state.normParams;

  const inputText = DOM.predictionInput.value.trim();
  if (!inputText) {
    alert('Masukkan harga ' + windowSize + ' hari terakhir, dipisahkan dengan koma.');
    return;
  }

  const inputPrices = inputText.split(',').map(s => parseFloat(s.trim())).filter(v => !isNaN(v));
  if (inputPrices.length !== windowSize) {
    alert(`Diperlukan tepat ${windowSize} nilai harga. Anda memasukkan ${inputPrices.length}.`);
    return;
  }

  // Normalize input using training params
  const normalizedInput = inputPrices.map(v => (v - min) / range);
  const normalizedPrediction = state.model.predict(normalizedInput);
  const predictedPrice = denormalize(normalizedPrediction, min, range);

  DOM.predictionValue.textContent = formatRupiah(Math.round(predictedPrice));
  DOM.predictionDate.textContent = 'Prediksi berdasarkan input kustom';
}

function resetModel() {
  state.model = null;
  state.trained = false;

  DOM.trainingSection.classList.add('hidden');
  DOM.predictionSection.classList.add('hidden');
  DOM.progressBar.style.width = '0%';
  DOM.epochInfo.textContent = '';

  if (state.charts.mse) { state.charts.mse.destroy(); state.charts.mse = null; }
  if (state.charts.comparison) { state.charts.comparison.destroy(); state.charts.comparison = null; }

  DOM.weightsBody.innerHTML = '';
}

/* -------- Event Listeners -------- */

document.addEventListener('DOMContentLoaded', () => {
  init();

  DOM.btnTrain.addEventListener('click', startTraining);
  DOM.btnPredict.addEventListener('click', predictCustom);
  DOM.btnReset.addEventListener('click', resetModel);
});
