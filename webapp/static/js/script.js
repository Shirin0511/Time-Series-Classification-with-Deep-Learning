document.addEventListener('DOMContentLoaded', function() {
  // ---------- Elements ----------
  const userGuideModal = document.getElementById('userGuideModal');
  const dashboardIcon = document.getElementById('dashboard-icon');
  const closeButton = userGuideModal ? userGuideModal.querySelector('.close-button') : null;

  const datasetSelectionModal = document.getElementById('datasetSelectionModal');
  const closeDatasetModalButton = document.getElementById('closeDatasetModalButton');
  const datasetIcon = document.getElementById('dataset-icon');
  const selectedDatasetNameSpan = document.getElementById('selected-dataset-name');
  const datasetDropdown = document.getElementById('datasetDropdown'); // source of truth for dataset list

  const hyperparamSelectionModal = document.getElementById('hyperparamSelectionModal');
  const closeHyperparamModalButton = document.getElementById('closeHyperparamModalButton');
  const hyperparamIcon = document.getElementById('hyperparam-icon');

  const learningRateDropdown = document.getElementById('learningRateDropdown'); // icon4 (menu-only)
  const epochsDropdown = document.getElementById('epochsDropdown');
  const batchSizeDropdown = document.getElementById('batchSizeDropdown');

  const resultsModal = document.getElementById('results-modal');
  const classificationReportOutput = document.getElementById('classification-report-output');
  const trainModelButton = document.getElementById('trainModelButton');
  const mainContent = document.getElementById('map');

  // Icon 5 elems
  const fifthIcon = document.getElementById('fifthIcon');
  const icon5Modal = document.getElementById('icon5Modal');
  const icon5CloseBtn = document.getElementById('icon5CloseBtn');
  const icon5DatasetDropdown = document.getElementById('icon5DatasetDropdown'); // has toggle + inner menu
  const icon5LR = document.getElementById('icon5LR');
  const icon5Epochs = document.getElementById('icon5Epochs');
  const icon5Batch = document.getElementById('icon5Batch');
  const icon5ConfirmBtn = document.getElementById('icon5ConfirmBtn');

  const icon5DatasetList = document.getElementById('icon5DatasetList');

    const modelSelectionModal = document.getElementById('modelSelectionModal');
  const closeModelModalButton = document.getElementById('closeModelModalButton');
  const modelIcon = document.getElementById('model-icon');
  const modelDropdown = document.getElementById('modelDropdown');

  const icon5Model = document.getElementById('icon5Model');




  // ---------- Single source of truth for HYPERPARAM OPTIONS ----------
  const HYPERPARAM_OPTIONS = {
    learningRate: ['0.001', '0.01', '0.1'],
    epochs:       ['100', '500', '1000'],
    batchSize:    ['8','16', '32']
  };

  // build menu items (<a data-value="...">...</a>) into the right container
  function buildMenu(dropdownEl, paramKey) {
    if (!dropdownEl) return;

    // icon4 menus are the element itself; icon5 menus are inside .dropdown
    const isIcon5 = dropdownEl.classList.contains('dropdown');
    const menuEl = isIcon5 ? dropdownEl.querySelector('.dropdown-menu') : dropdownEl;
    if (!menuEl) return;

    menuEl.innerHTML = (HYPERPARAM_OPTIONS[paramKey] || [])
      .map(v => `<a href="#" data-value="${v}">${v}</a>`)
      .join('');
  }

  // Build BOTH icon4 and icon5 hyperparam menus from one config
  buildMenu(learningRateDropdown, 'learningRate');
  buildMenu(epochsDropdown, 'epochs');
  buildMenu(batchSizeDropdown, 'batchSize');

  buildMenu(icon5LR, 'learningRate');
  buildMenu(icon5Epochs, 'epochs');
  buildMenu(icon5Batch, 'batchSize');

  // Activate dropdowns for Icon 4 hyperparameters
[learningRateDropdown, epochsDropdown, batchSizeDropdown].forEach(dd => {
  if (!dd) return;

  // make toggle button open/close menu
  bindDropdownToggle(dd);

  const menu = dd.querySelector('.dropdown-menu');
  if (!menu) return;

  menu.querySelectorAll('a').forEach(item => {
    item.addEventListener('click', e => {
      e.preventDefault();
      e.stopPropagation();

      // mark as selected in UI & save to localStorage
      setSelectedHyperparameter(dd, item.dataset.value);

      // update button text to chosen value
      const t = dd.querySelector('.dropdown-toggle');
      if (t) t.textContent = item.textContent;

      // close menu
      menu.classList.remove('show');
    });
  });
});


  // ---------- User Guide ----------
  function openUserGuideModal(){ if (userGuideModal) userGuideModal.style.display = 'flex'; }
  function closeUserGuideModal(){ if (userGuideModal) userGuideModal.style.display = 'none'; }
  dashboardIcon && dashboardIcon.addEventListener('click', e => { e.preventDefault(); openUserGuideModal(); });
  closeButton && closeButton.addEventListener('click', closeUserGuideModal);
  window.addEventListener('click', e => { if (e.target === userGuideModal) closeUserGuideModal(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && userGuideModal && userGuideModal.style.display === 'flex') closeUserGuideModal(); });

  // ---------- Dataset modal (Icon 2) ----------

function bindDropdownToggle(dropdownEl) {
  if (!dropdownEl) return;
  const btn = dropdownEl.querySelector('.dropdown-toggle');
  const menu = dropdownEl.querySelector('.dropdown-menu');
  if (!btn || !menu) return;

  btn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    // close any other open menus on the page
    document.querySelectorAll('.dropdown-menu.show').forEach(m => {
      if (m !== menu) m.classList.remove('show');
    });
    menu.classList.toggle('show');
  });
}


  function openDatasetSelectionModal(){ datasetSelectionModal && (datasetSelectionModal.style.display = 'flex'); }
  function closeDatasetSelectionModal(){ datasetSelectionModal && (datasetSelectionModal.style.display = 'none'); }
  datasetIcon && datasetIcon.addEventListener('click', e => { e.preventDefault(); openDatasetSelectionModal(); });
  closeDatasetModalButton && closeDatasetModalButton.addEventListener('click', closeDatasetSelectionModal);
  window.addEventListener('click', e => { if (e.target === datasetSelectionModal) closeDatasetSelectionModal(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && datasetSelectionModal && datasetSelectionModal.style.display === 'flex') closeDatasetSelectionModal(); });

  function setSelectedDataset(datasetId, datasetName) {
  if (datasetDropdown) {
    datasetDropdown.querySelectorAll('a').forEach(a => a.classList.remove('selected'));
    const sel = datasetDropdown.querySelector(`[data-dataset-id="${datasetId}"]`);
    sel && sel.classList.add('selected');
    // NEW: update dropdown button label
    const btn = datasetDropdown.querySelector('.dropdown-toggle');
    if (btn) btn.textContent = datasetName;
  }
  localStorage.setItem('selectedDatasetId', datasetId);
  localStorage.setItem('selectedDatasetName', datasetName);
  selectedDatasetNameSpan && (selectedDatasetNameSpan.textContent = datasetName);
}


if (datasetDropdown) {
    datasetDropdown.querySelectorAll('a').forEach(item => {
        item.addEventListener('click', e => {
            e.preventDefault();
            const id = item.dataset.datasetId;
            const name = item.textContent;
            setSelectedDataset(id, name);
            // Close dropdown menu after selection
            //const menu = datasetDropdown.querySelector('.dropdown-menu');
            //if (menu) menu.classList.remove('show');
            // Close the modal
            closeDatasetSelectionModal();
        });
    });
}

// Close any dropdown if clicked outside
document.addEventListener('click', () => {
    document.querySelectorAll('.dropdown-menu.show').forEach(m => m.classList.remove('show'));
});


  // defaults for dataset
  const storedDatasetId = localStorage.getItem('selectedDatasetId');
  const storedDatasetName = localStorage.getItem('selectedDatasetName');
  if (storedDatasetId && storedDatasetName) {
    setSelectedDataset(storedDatasetId, storedDatasetName);
  } else {
    setSelectedDataset('1', 'Italy Dataset');
  }


  function openModelSelectionModal(){ modelSelectionModal && (modelSelectionModal.style.display = 'flex'); }
  function closeModelSelectionModal(){ modelSelectionModal && (modelSelectionModal.style.display = 'none'); }

  modelIcon && modelIcon.addEventListener('click', e => { e.preventDefault(); openModelSelectionModal(); });
  closeModelModalButton && closeModelModalButton.addEventListener('click', closeModelSelectionModal);
  window.addEventListener('click', e => { if (e.target === modelSelectionModal) closeModelSelectionModal(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && modelSelectionModal && modelSelectionModal.style.display === 'flex') closeModelSelectionModal(); });


  function setSelectedModel(modelId, modelName) {
    if (modelDropdown) {
      modelDropdown.querySelectorAll('a').forEach(a => a.classList.remove('selected'));
      const sel = modelDropdown.querySelector(`[data-model-id="${modelId}"]`);
      sel && sel.classList.add('selected');
    }
    localStorage.setItem('selectedModelId', modelId);
    localStorage.setItem('selectedModelName', modelName);
  }

  if (modelDropdown) {
    modelDropdown.querySelectorAll('a').forEach(item => {
      item.addEventListener('click', e => {
        e.preventDefault();
        const id = item.dataset.modelId;        // "cnn" | "transformer"
        const name = item.textContent.trim();   // "1D CNN" | "Transformer"
        setSelectedModel(id, name);
        closeModelSelectionModal();
      });
    });
  }

  // Default model (if none chosen yet)
  const storedModelId = localStorage.getItem('selectedModelId');
  const storedModelName = localStorage.getItem('selectedModelName');
  if (storedModelId && storedModelName) {
    setSelectedModel(storedModelId, storedModelName);
  } else {
    setSelectedModel('cnn', '1D CNN'); // sensible default
  }


  // ---------- Hyperparam modal (Icon 4) ----------
  function openHyperparamSelectionModal(){
    if (!hyperparamSelectionModal) return;
    hyperparamSelectionModal.style.display = 'flex';
  }
  function closeHyperparamSelectionModal(){
    if (!hyperparamSelectionModal) return;
    hyperparamSelectionModal.style.display = 'none';
  }
  hyperparamIcon && hyperparamIcon.addEventListener('click', e => { e.preventDefault(); openHyperparamSelectionModal(); });
  closeHyperparamModalButton && closeHyperparamModalButton.addEventListener('click', closeHyperparamSelectionModal);
  window.addEventListener('click', e => { if (e.target === hyperparamSelectionModal) closeHyperparamSelectionModal(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && hyperparamSelectionModal && hyperparamSelectionModal.style.display === 'flex') closeHyperparamSelectionModal(); });

  // bind icon4 pickers
  function setSelectedHyperparameter(target, value) {
    const dropdownElement = (typeof target === 'string')
      ? document.querySelector(`.hyperparam-modal-dropdown[data-parameter-type="${target}"]`)
      : target;
    if (!dropdownElement) return;

    dropdownElement.querySelectorAll('a').forEach(a => a.classList.remove('selected'));
    const selectedItem = dropdownElement.querySelector(`[data-value="${value}"]`);
    if (selectedItem) selectedItem.classList.add('selected');

    const parameterType = dropdownElement.dataset.parameterType;
    localStorage.setItem(parameterType, value);
    updateHyperparamsSummary();
  }

  function updateHyperparamsSummary() {
    const lr = localStorage.getItem('learningRate') || 'N/A';
    const ep = localStorage.getItem('epochs') || 'N/A';
    const bs = localStorage.getItem('batchSize') || 'N/A';
    const span = document.getElementById('selected-hyperparams-summary');
    if (span) span.textContent = `LR:${lr} E:${ep} B:${bs}`;
  }


  // defaults for hyperparams (after menus exist)
  const defaultLearningRate = localStorage.getItem('learningRate') || HYPERPARAM_OPTIONS.learningRate[0];
  const defaultEpochs = localStorage.getItem('epochs') || HYPERPARAM_OPTIONS.epochs[0];
  const defaultBatchSize = localStorage.getItem('batchSize') || HYPERPARAM_OPTIONS.batchSize[0];
  learningRateDropdown && setSelectedHyperparameter(learningRateDropdown, defaultLearningRate);
  epochsDropdown && setSelectedHyperparameter(epochsDropdown, defaultEpochs);
  batchSizeDropdown && setSelectedHyperparameter(batchSizeDropdown, defaultBatchSize);

// Ensure toggle button text matches defaults
const syncBtn = (dd, key) => {
  const a = dd?.querySelector(`.dropdown-menu a[data-value="${localStorage.getItem(key)}"]`);
  const t = dd?.querySelector('.dropdown-toggle');
  if (a && t) t.textContent = a.textContent;
};
syncBtn(learningRateDropdown, 'learningRate');
syncBtn(epochsDropdown,       'epochs');
syncBtn(batchSizeDropdown,    'batchSize');


  updateHyperparamsSummary();

  // Map id -> display name (fallback if we can't read from UI/localStorage)
const DATASET_NAME_MAP = { '1': 'Italy Dataset', '2': 'CinCECGTorso', '3': 'ECG200' };

function getDatasetNameFromId(dsId, {preferIcon5=false} = {}) {
  // Try Icon 5 menu text (when Icon 5 flow is used)
  if (preferIcon5 && window.icon5DatasetDropdown) {
    const a = icon5DatasetDropdown
      ?.querySelector(`.dropdown-menu a[data-dataset-id="${dsId}"]`);
    if (a) return a.textContent.trim();
  }
  // Try saved main selection name
  const savedName = localStorage.getItem('selectedDatasetName');
  if (savedName && (localStorage.getItem('selectedDatasetId') === String(dsId))) {
    return savedName;
  }
  // Fallback
  return DATASET_NAME_MAP[String(dsId)] || `Dataset ${dsId}`;
}

function formatHP(lr, epochs, batchSize) {
  return `LR: ${lr}  |  Epochs: ${epochs}  |  Batch Size: ${batchSize}`;
}

// to retry hitting the train model button in case it fails for the first time
async function postJSONWithRetry(url, payload, retries = 1, backoffMs = 600) {
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type':'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = data?.error || `HTTP ${res.status}`;
      throw new Error(msg);
    }
    return data;
  } catch (err) {
    if (retries > 0) {
      await new Promise(r => setTimeout(r, backoffMs));
      return postJSONWithRetry(url, payload, retries - 1, Math.floor(backoffMs * 1.5));
    }
    throw err;
  }
}



  // ---------- Train model (Icon 4) ----------
  trainModelButton && trainModelButton.addEventListener('click', async function() {
    const datasetId = localStorage.getItem('selectedDatasetId');
    const learningRate = localStorage.getItem('learningRate');
    const epochs = localStorage.getItem('epochs');
    const batchSize = localStorage.getItem('batchSize');

     // NEW: grab selected model (defaults to 'cnn' if missing)
  const modelId      = localStorage.getItem('selectedModelId') || 'cnn';
  const modelName    = localStorage.getItem('selectedModelName') || '1D CNN';


    if (!datasetId || !learningRate || !epochs || !batchSize) {
      mainContent && (mainContent.innerHTML = '<h2>Please select all hyperparameters.</h2>');
      return;
    }

    mainContent && (mainContent.innerHTML = `
      <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%">
        <h2 style="color: white;">Training model... please wait.</h2>
        <span id="loading-spinner" style="font-size:3rem;">⏳</span>
      </div>
    `);
    closeHyperparamSelectionModal();

    try {
    const data = await postJSONWithRetry(
        `${window.location.origin}/train_model`,
        {
            ds_id: parseInt(datasetId),
            lr: parseFloat(learningRate),
            epochs: parseInt(epochs),
            batch_size: parseInt(batchSize),
            model: modelId                 
        },
        1 // one retry if first attempt fails
    );

    // code that updates the UI with training results
    if (classificationReportOutput) {
        const m = data.metrics || {};
        const metricsLine = (m && (m.accuracy !== undefined))
            ? `Accuracy: ${Number(m.accuracy).toFixed(3)} | Precision: ${Number(m.precision).toFixed(3)} | Recall: ${Number(m.recall).toFixed(3)} | F1: ${Number(m.f1).toFixed(3)}`
            : '';
        mainContent.innerHTML = `
            <div class="result-card">
                <h3>${getDatasetNameFromId(datasetId)}</h3>
                <div>Model: ${modelName}</div> 
                <div>${formatHP(learningRate, epochs, batchSize)}</div>
                <div style="margin-top:10px;font-weight:600;">${metricsLine}</div>
            </div>
        `;
    }
    //resultsModal && (resultsModal.style.display = 'flex');

} catch (err) {
    mainContent && (mainContent.innerHTML = `<h2>Error: ${err.message}</h2>`);
    console.error('Error:', err);
}

  });

  // ---------- Results modal ----------
  const closeResultsModalButton = document.getElementById('closeResultsModalButton');
  closeResultsModalButton && closeResultsModalButton.addEventListener('click', () => { resultsModal && (resultsModal.style.display = 'none'); });
  window.addEventListener('click', e => { if (e.target === resultsModal) resultsModal && (resultsModal.style.display='none'); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && resultsModal && resultsModal.style.display === 'flex') resultsModal.style.display='none'; });

  // ---------- Icon 5 (re-select + load saved model) ----------
  function updateIcon5ConfirmState(){
     const ok = ['ds_id_temp','model_temp','learningRate_temp','epochs_temp','batchSize_temp']
              .every(k => !!localStorage.getItem(k));
  icon5ConfirmBtn && (icon5ConfirmBtn.disabled = !ok);
  }

  function wireIcon5Dropdown(dropdownEl, storageKey){
    if (!dropdownEl) return;
  const menu = dropdownEl.querySelector('.dropdown-menu');
  if (!menu) return;

  menu.querySelectorAll('a').forEach(a => {
    a.addEventListener('click', e => {
      e.preventDefault();

      // UI highlight
      menu.querySelectorAll('a').forEach(x => x.classList.remove('selected'));
      a.classList.add('selected');

      // Update button label
      const toggle = dropdownEl.querySelector('.dropdown-toggle');
      if (toggle) toggle.textContent = a.textContent;

      // ✅ Robustly get the value for ANY dropdown:
      //   - dataset menu uses data-dataset-id
      //   - other menus use data-value
      const val =
        a.dataset.value ??
        a.dataset.datasetId ??
        a.getAttribute('data-dataset-id') ??
        a.getAttribute('data-value');

      if (val == null || val === '') {
        console.warn('No data-* value found for dropdown item:', a);
        localStorage.removeItem(storageKey);
      } else {
        localStorage.setItem(storageKey, String(val));
      }

      updateIcon5ConfirmState();
      menu.classList.remove('show');
    });
  });
  }

  function bindToggle(dd) {
    const btn = dd?.querySelector('.dropdown-toggle');
    const menu = dd?.querySelector('.dropdown-menu');
    if (!btn || !menu) return;
    btn.addEventListener('click', e => {
      e.preventDefault();
      e.stopPropagation();
      icon5Modal.querySelectorAll('.dropdown-menu.show').forEach(m => { if (m !== menu) m.classList.remove('show'); });
      menu.classList.toggle('show');
    });
  }

  function resetIcon5Temp() {
    ['ds_id_temp','model_temp','learningRate_temp','epochs_temp','batchSize_temp']
    .forEach(k => localStorage.removeItem(k));

  [icon5DatasetDropdown, icon5Model, icon5LR, icon5Epochs, icon5Batch].forEach(dd => {
    if (!dd) return;
    dd.querySelectorAll('a').forEach(x => x.classList.remove('selected'));
    const t = dd.querySelector('.dropdown-toggle');
    if (t) t.textContent = 'Choose';
    const m = dd.querySelector('.dropdown-menu');
    if (m) m.classList.remove('show');
  });
  updateIcon5ConfirmState();
  }

  // --- NEW: wire an always-visible list (no toggle) for Icon 5 dataset ---
function wireIcon5AlwaysMenu(menuEl, storageKey) {
  if (!menuEl) return;
  menuEl.querySelectorAll('a').forEach(a => {
    a.addEventListener('click', e => {
      e.preventDefault();
      // highlight current selection
      menuEl.querySelectorAll('a').forEach(x => x.classList.remove('selected'));
      a.classList.add('selected');
      // store chosen value (works with data-dataset-id or data-value)
      const val = a.dataset.datasetId || a.dataset.value;
      localStorage.setItem(storageKey, val);
      updateIcon5ConfirmState();
    });
  });
}


  // Build icon5 menus already done above (same config)
  // Wire toggles & item clicks for icon5 hyperparams:
  [icon5LR, icon5Epochs, icon5Batch].forEach(dd => {
    if (!dd) return;
    bindToggle(dd);
    // (re)bind items
    wireIcon5Dropdown(dd, dd.id === 'icon5LR' ? 'learningRate_temp' : dd.id === 'icon5Epochs' ? 'epochs_temp' : 'batchSize_temp');
  });

  // Bind dataset dropdown toggle for Icon 5
if (icon5DatasetDropdown) {
  bindToggle(icon5DatasetDropdown);
}

if (icon5Model) {
  bindToggle(icon5Model);
  wireIcon5Dropdown(icon5Model, 'model_temp');
}

  // Clone dataset list from Icon 2 into Icon 5 on open (single source for dataset list)
  function cloneDatasetIntoIcon5() {
   const srcMenu = datasetDropdown?.querySelector('.dropdown-menu');   // icon2 source
  const dstMenu = icon5DatasetDropdown?.querySelector('.dropdown-menu');
  if (!srcMenu || !dstMenu) return;
  dstMenu.innerHTML = srcMenu.innerHTML;  // copy only <a> items

  // wire item clicks for icon5 dataset dropdown (stores ds_id_temp, sets button label, closes menu)
  wireIcon5Dropdown(icon5DatasetDropdown, 'ds_id_temp');
  }
  //bindToggle(icon5DatasetDropdown);

  fifthIcon && fifthIcon.addEventListener('click', () => {
  resetIcon5Temp();          // clears *_temp + UI state
  cloneDatasetIntoIcon5();   // populate dataset menu for icon5

  // Preselect DATASET = current app selection (or default to '1')
  //const currentDs = localStorage.getItem('selectedDatasetId') || '1';
  const currentDs = localStorage.getItem('ds_id_temp') || 1;

  const dsMenu = icon5DatasetDropdown?.querySelector('.dropdown-menu');
  const dsBtn  = icon5DatasetDropdown?.querySelector('.dropdown-toggle');
  if (dsMenu) {
    const a = dsMenu.querySelector(`a[data-dataset-id="${currentDs}"]`) || dsMenu.querySelector('a');
    if (a) {
      dsMenu.querySelectorAll('a').forEach(x => x.classList.remove('selected'));
      a.classList.add('selected');
      if (dsBtn) dsBtn.textContent = a.textContent;           // show selected name on the button
      localStorage.setItem('ds_id_temp', a.dataset.datasetId); // <-- prevents NoneType on backend
    }
  }

  // Preselect hyperparams into *_temp and show the chosen values on buttons
  const lr = localStorage.getItem('learningRate');
  const ep = localStorage.getItem('epochs');
  const bs = localStorage.getItem('batchSize');

  const preselect = (dd, val, key) => {
    if (!dd || !val) return;
    const a = dd.querySelector(`.dropdown-menu a[data-value="${val}"]`);
    if (a) {
      dd.querySelectorAll('a').forEach(x => x.classList.remove('selected'));
      a.classList.add('selected');
      const t = dd.querySelector('.dropdown-toggle');
      if (t) t.textContent = a.textContent;
      localStorage.setItem(key, val);
    }
  };
  preselect(icon5LR,     lr, 'learningRate_temp');
  preselect(icon5Epochs, ep, 'epochs_temp');
  preselect(icon5Batch,  bs, 'batchSize_temp');

const currentModel = localStorage.getItem('selectedModelId') || 'cnn';
  const mMenu = icon5Model?.querySelector('.dropdown-menu');
  const mBtn  = icon5Model?.querySelector('.dropdown-toggle');

  if (mMenu) {
    const a = mMenu.querySelector(`a[data-value="${currentModel}"]`) || mMenu.querySelector('a');
    if (a) {
      mMenu.querySelectorAll('a').forEach(x => x.classList.remove('selected'));
      a.classList.add('selected');
      if (mBtn) mBtn.textContent = a.textContent;
      localStorage.setItem('model_temp', a.dataset.value);
    }
  }

  updateIcon5ConfirmState();            // enables Load Saved Model
  icon5Modal && (icon5Modal.style.display = 'flex');
});

  icon5CloseBtn && icon5CloseBtn.addEventListener('click', () => { icon5Modal && (icon5Modal.style.display = 'none'); });
  window.addEventListener('click', e => { if (e.target === icon5Modal) icon5Modal.style.display = 'none'; });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && icon5Modal && icon5Modal.style.display === 'flex') {
      icon5Modal.querySelectorAll('.dropdown-menu.show').forEach(m => m.classList.remove('show'));
      icon5Modal.style.display = 'none';
    }
  });

  icon5ConfirmBtn.addEventListener('click', function () {
 if (icon5Modal) icon5Modal.style.display = 'none';

  // Read the values the user picked in the Icon 5 modal
  const selectedDatasetId = localStorage.getItem('ds_id_temp') || localStorage.getItem('selectedDatasetId');
  const lr        = localStorage.getItem('learningRate_temp') || localStorage.getItem('learningRate');
  const epochs    = localStorage.getItem('epochs_temp')       || localStorage.getItem('epochs');
  const batchSize = localStorage.getItem('batchSize_temp')    || localStorage.getItem('batchSize');

  if (!selectedDatasetId || selectedDatasetId === 'undefined') {
    if (mainContent) mainContent.innerHTML = '<h3 style="color:red;">Please choose a dataset.</h3>';
    return;
  }

  if (!selectedDatasetId || !lr || !epochs || !batchSize) {
    alert('Please select dataset and hyperparameters first.');
    return;
  }

  // Target area for displaying output
  const target = mainContent || document.getElementById('map');
  if (target) target.innerHTML = '<h2 style="color: white;">Loading saved model and evaluating...</h2>';

  // Helper to escape HTML
  function escapeHtml(s){
    return String(s).replace(/[&<>"']/g, m => (
      {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]
    ));
  }

const modelId   = localStorage.getItem('model_temp') || localStorage.getItem('selectedModelId') || 'cnn';
//const modelName = (modelId === 'transformer') ? 'Transformer' : '1D CNN';
const modelName =
  modelId === 'transformer' ? 'Transformer' :
  modelId === 'inception' || modelId === 'inceptiontime' ? 'InceptionTime' :
  '1D CNN';

// ---- VALIDATE: avoid NaN/null being sent ----
const okLR     = lr !== null && lr !== '' && !Number.isNaN(Number(lr));
const okEpochs = epochs !== null && epochs !== '' && Number.isInteger(Number(epochs));
const okBatch  = batchSize !== null && batchSize !== '' && Number.isInteger(Number(batchSize));
const okDs     = selectedDatasetId !== null && selectedDatasetId !== '';

if (!(okLR && okEpochs && okBatch && okDs)) {
  if (target) target.innerHTML = '<h3 style="color:red;">Please select a valid dataset and hyperparameters.</h3>';
  return;
}

  // Fetch call to backend
  fetch('/model_results', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    ds_id: String(selectedDatasetId),
    lr: String(lr),
    epochs: String(epochs),
    batch_size: String(batchSize),
    model: modelId
  })
  })
  .then(res => res.json())
  .then(data => {
    console.log('model_results response:', data);

    if (!data.ok) {
      showErrorModal(data.error || 'No saved model found for these parameters.');
  return;
    }

    if (target) {
      const m = data.metrics || {};

  // Prefer *_temp (Icon 5 modal picks), fallback to main selections
  const dsId   = (localStorage.getItem('ds_id_temp') || localStorage.getItem('selectedDatasetId'));
  const lr     = (localStorage.getItem('learningRate_temp') || localStorage.getItem('learningRate'));
  const epochs = (localStorage.getItem('epochs_temp') || localStorage.getItem('epochs'));
  const batch  = (localStorage.getItem('batchSize_temp') || localStorage.getItem('batchSize'));

  const dsName = getDatasetNameFromId(dsId, { preferIcon5: true });
  const hpLine = formatHP(lr, epochs, batch);

  target.innerHTML = `
    <div class="result-card">
      <h3>${dsName}</h3>
      <div>Model: ${modelName}</div>
      <div style="opacity:.9;margin:-4px 0 12px;">${hpLine}</div>
      <div style="margin:8px 0 14px;font-weight:600;">
        Accuracy: ${Number(m.accuracy).toFixed(3)} &nbsp;|&nbsp;
        Precision: ${Number(m.precision).toFixed(3)} &nbsp;|&nbsp;
        Recall: ${Number(m.recall).toFixed(3)} &nbsp;|&nbsp;
        F1: ${Number(m.f1).toFixed(3)}
      </div>
    </div>
  `;
  target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  })
  .catch(err => {
  console.error(err);
  showErrorModal(err.message || 'Request failed. Please try again.');
});
});

}); // DOMContentLoaded end


function showErrorModal(message) {
  const modal = document.getElementById('errorModal');
  const messageEl = document.getElementById('errorModalMessage');
  const closeBtn = document.getElementById('errorModalClose');

  // Clear loading message in background
  const target = document.getElementById('map');
  if (target) target.innerHTML = '';   // <--- this clears the loading screen

  if (modal && messageEl) {
    messageEl.textContent = message;
    modal.style.display = 'flex';
  }

  if (closeBtn) {
    closeBtn.onclick = () => { modal.style.display = 'none'; };
  }

  window.onclick = (event) => {
    if (event.target === modal) {
      modal.style.display = 'none';
    }
  };

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && modal && modal.style.display === 'flex') {
      modal.style.display = 'none';
    }
  });
}

// ---------- Visualization (Icon 3) ----------
const visualizationIcon = document.getElementById('visualization-icon');
const mainArea = document.getElementById('map');
if (visualizationIcon) {
  visualizationIcon.addEventListener('click', function(e) {
  e.preventDefault();
  const selectedDatasetId = localStorage.getItem('selectedDatasetId');
  if (!selectedDatasetId) { 
    mainArea && (mainArea.innerHTML = '<h2>Please select a dataset first.</h2>'); 
    return; 
  }

  // NEW: work out the display name for the caption
  const nameFallbackMap = { '1': 'Italy Dataset', '2': 'CinCECGTorso', '3': 'ECG200' };
  const selectedDatasetName = localStorage.getItem('selectedDatasetName') 
      || nameFallbackMap[selectedDatasetId] 
      || `Dataset ${selectedDatasetId}`;

  const visualizationUrl = `/get_visualization/${selectedDatasetId}`;
  mainArea && (mainArea.innerHTML = '<h2 style="color: white;">Loading Visualisations...</h2>');

  const img = document.createElement('img');
  img.src = visualizationUrl;
  img.alt = `Time Series Visualization – ${selectedDatasetName}`;

  // NEW: a caption element for the dataset name
  const caption = document.createElement('div');
  caption.className = 'viz-caption';
  caption.textContent = `${selectedDatasetName} Visualization`;

  // Close button
  const closeBtn = document.createElement('span');
  closeBtn.innerHTML = '&times;';
  closeBtn.className = 'close-button-visualization';
  closeBtn.addEventListener('click', () => { mainArea && (mainArea.innerHTML = ''); });

  // Styling the image as before
  img.style.maxWidth = '80%';
  img.style.maxHeight = '80vh';
  img.style.width = 'auto';
  img.style.height = 'auto';
  img.style.margin = '12px auto 20px';
  img.style.objectFit = 'contain';

  // Wrap everything nicely
  const wrapper = document.createElement('div');
  wrapper.style.display = 'flex';
  wrapper.style.flexDirection = 'column';
  wrapper.style.alignItems = 'center';
  wrapper.style.padding = '8px 16px';

  img.onload = () => {
    if (!mainArea) return;
    mainArea.innerHTML = '';
    wrapper.appendChild(closeBtn);
    wrapper.appendChild(caption); // caption ABOVE the image
    wrapper.appendChild(img);
    mainArea.appendChild(wrapper);
  };
  img.onerror = () => { mainArea && (mainArea.innerHTML = '<h2>Failed to load visualization.</h2>'); };
});

}
