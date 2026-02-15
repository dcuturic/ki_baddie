// ==================== SHARED: SIMPLE FIELDS DEFINITIONS + AUDIO ====================
// Used by both index.html (create/edit modal) and instance_config.html

var audioDevices = { inputs: [], outputs: [] };
var audioModePerService = {};
var virtualAssignments = {};  // { instanceName: assignment }

function getAudioMode(serviceId) {
    return audioModePerService[serviceId] || 'all';
}

async function loadAudioDevices() {
    try {
        const result = await apiCall('/api/audio/devices');
        if (result && result.success) {
            audioDevices = { inputs: result.inputs || [], outputs: result.outputs || [] };
        }
    } catch (e) {
        console.warn('Audio devices not available:', e);
    }
}

/**
 * Assign (or retrieve existing) virtual audio cable pair for an instance.
 * Returns the assignment object or null on failure.
 */
async function handleVirtualAudioAssignment(instanceName) {
    if (!instanceName) {
        showNotification('⚠️ Bitte zuerst einen Instanznamen eingeben', 'error');
        return null;
    }
    try {
        const result = await apiCall('/api/audio/virtual/assign', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ instance_name: instanceName })
        });
        if (result && result.success && result.assignment) {
            virtualAssignments[instanceName] = result.assignment;
            if (!result.already_existed) {
                showNotification('🔌 Virtuelles Audio-Kabel zugewiesen: ' + result.assignment.slot_name, 'success');
            }
            return result.assignment;
        } else if (result && !result.success) {
            showNotification('❌ ' + (result.error || 'Fehler bei Audio-Zuweisung'), 'error');
        }
    } catch (e) {
        console.error('Virtual audio assignment error:', e);
        showNotification('❌ Virtuelles Audio konnte nicht zugewiesen werden', 'error');
    }
    return null;
}

var SIMPLE_FIELDS = {
    ollama: [
        { path: 'server.host', label: 'Host', type: 'text', hint: 'IP auf der Ollama lauscht' },
        { path: 'server.port', label: 'Port', type: 'number', hint: 'Ollama API Port (Standard: 11434)' },
        { path: 'gpu.cuda_device', label: 'GPU Device', type: 'number', hint: 'CUDA Device Index (0, 1, ...)' },
        { path: 'model.context_length', label: 'Context Length', type: 'number', hint: 'Token-Fenster (z.B. 4096, 8192)' },
        { path: 'model.threads', label: 'Threads', type: 'number', hint: 'CPU Threads für Inferenz' },
        { path: 'model.keep_alive', label: 'Keep Alive', type: 'text', hint: '"-1" = Modell immer geladen' },
        { path: 'model.max_loaded_models', label: 'Max Models', type: 'number', hint: 'Gleichzeitig geladene Modelle' },
        { path: 'model.load_timeout', label: 'Load Timeout', type: 'text', hint: 'z.B. 15m, 5m' },
    ],
    main_server: [
        { path: 'server.port', label: 'Server Port', type: 'number', hint: 'HTTP-Port des Main Servers' },
        { path: 'server.host', label: 'Host', type: 'text', hint: '0.0.0.0 = alle Interfaces' },
        { path: 'server.debug', label: 'Debug', type: 'bool', hint: 'Flask Debug-Modus' },
        { path: 'microphone._audio_mode', label: '🎧 Audio Quelle', type: 'audio_mode', hint: 'PC-Geräte oder virtuelle Kabel', target_fields: ['microphone.device_name'] },
        { path: 'microphone.device_name', label: '🎤 Mikrofon Input', type: 'audio_input', hint: 'Aufnahme-Gerät für STT' },
        { path: 'microphone.allow_partial_match', label: 'Partial Match', type: 'bool', hint: 'Teilstring-Suche beim Gerätenamen' },
        { path: 'services.ki_chat', label: '→ KI Chat URL', type: 'url', hint: 'URL zum ki_chat Service', linkedService: 'ki_chat' },
        { path: 'services.text_to_speech', label: '→ TTS URL', type: 'url', hint: 'URL zum TTS Service', linkedService: 'text_to_speech' },
        { path: 'services.vroid_poser', label: '→ Poser URL', type: 'url', hint: 'URL zum VroidPoser Service', linkedService: 'vroid_poser' },
        { path: 'services.vroid_emotion', label: '→ Emotion URL', type: 'url', hint: 'URL zum VroidEmotion Service', linkedService: 'vroid_emotion' },
    ],
    ki_chat: [
        { path: 'server.port', label: 'Server Port', type: 'number', hint: 'HTTP-Port des KI Chat' },
        { path: 'server.host', label: 'Host', type: 'text' },
        { path: 'server.debug', label: 'Debug', type: 'bool' },
        { path: 'ollama.url', label: 'Ollama URL', type: 'url', hint: 'Ollama API Endpoint' },
        { path: 'ollama.default_model', label: 'LLM Model', type: 'text', hint: 'z.B. llama3:latest' },
        { path: 'default_character', label: 'Default Character', type: 'text', hint: 'z.B. dilara, alex, luna' },
    ],
    text_to_speech: [
        { path: 'server.port', label: 'Server Port', type: 'number', hint: 'HTTP-Port des TTS' },
        { path: 'server.host', label: 'Host', type: 'text' },
        { path: 'server.debug', label: 'Debug', type: 'bool' },
        { path: 'osc.enabled', label: 'OSC aktiviert', type: 'bool', hint: 'Lipsync via OSC' },
        { path: 'osc.port', label: 'OSC Port', type: 'number', hint: 'VSeeFace OSC Port' },
        { path: 'tts.language', label: 'Sprache', type: 'text', hint: 'z.B. de, en' },
        { path: 'voicemod._audio_mode', label: '🎧 Audio Ausgabe', type: 'audio_mode', hint: 'PC-Geräte oder virtuelle Kabel', target_fields: ['voicemod.output_name_substring'] },
        { path: 'voicemod.output_name_substring', label: '🔊 Audio Output', type: 'audio_output', hint: 'Primäres Ausgabegerät (z.B. virtuelles Kabel)' },
        { path: 'voicemod.additional_outputs', label: '🔊+ Zusätzliche Ausgaben', type: 'audio_output_multi', hint: 'Parallele Ausgabe auf weitere Geräte' },
        { path: 'services.vroid_emotion', label: '→ Emotion URL', type: 'url', hint: 'URL zum VroidEmotion Service', linkedService: 'vroid_emotion' },
        { path: 'services.main_server', label: '→ Main Server URL', type: 'url', hint: 'URL zum Main Server', linkedService: 'main_server' },
    ],
    vroid_poser: [
        { path: 'server.port', label: 'Server Port', type: 'number', hint: 'HTTP-Port des Posers' },
        { path: 'server.host', label: 'Host', type: 'text' },
        { path: 'server.debug', label: 'Debug', type: 'bool' },
        { path: 'osc.enabled', label: 'OSC aktiviert', type: 'bool' },
        { path: 'osc.port', label: 'OSC Port', type: 'number' },
    ],
    vroid_emotion: [
        { path: 'server.port', label: 'Server Port', type: 'number', hint: 'HTTP-Port der Emotion' },
        { path: 'server.host', label: 'Host', type: 'text' },
        { path: 'server.debug', label: 'Debug', type: 'bool' },
        { path: 'osc.enabled', label: 'OSC aktiviert', type: 'bool' },
        { path: 'osc.port', label: 'OSC Port', type: 'number' },
    ],
};

var SIMPLE_FIELDS_DEFAULT = [
    { path: 'server.port', label: 'Server Port', type: 'number' },
    { path: 'server.host', label: 'Host', type: 'text' },
    { path: 'server.debug', label: 'Debug', type: 'bool' },
];

function getNestedValue(obj, path) {
    return path.split('.').reduce((o, k) => (o && o[k] !== undefined) ? o[k] : undefined, obj);
}

function setNestedValue(obj, path, value) {
    const keys = path.split('.');
    let cur = obj;
    for (let i = 0; i < keys.length - 1; i++) {
        if (cur[keys[i]] === undefined || typeof cur[keys[i]] !== 'object') cur[keys[i]] = {};
        cur = cur[keys[i]];
    }
    cur[keys[keys.length - 1]] = value;
}

/**
 * Collect all checked values from a multi-select checkbox group.
 * @param {string} fieldId - base field ID (checkboxes are in #fieldId-list)
 * @returns {string[]}
 */
function collectMultiSelectValues(fieldId) {
    const container = document.getElementById(fieldId + '-list');
    if (!container) return [];
    const cbs = container.querySelectorAll('input[type="checkbox"]');
    const selected = [];
    cbs.forEach(cb => { if (cb.checked) selected.push(cb.value); });
    return selected;
}

/**
 * Build the HTML for simple-mode fields for one service.
 * @param {string} serviceId
 * @param {object} cfg - the config object for this service
 * @param {object} options - { usedAudioRegistry, allConfigs, onChangeCallback, instanceName }
 *   onChangeCallback: string - JS function name to call on change, e.g. 'simpleFieldChanged'
 *   instanceName: string - current instance name (for virtual audio assignment display)
 */
function buildSimpleFieldsHtml(serviceId, cfg, options) {
    const fields = SIMPLE_FIELDS[serviceId] || SIMPLE_FIELDS_DEFAULT;
    const usedAudio = (options && options.usedAudioRegistry) || {};
    const allConfigs = (options && options.allConfigs) || {};
    const onChangeFn = (options && options.onChangeCallback) || 'simpleFieldChanged';
    const getConflicts = (options && options.getConflicts) || null;
    const instanceName = (options && options.instanceName) || '';

    const port = getNestedValue(cfg, 'server.port');
    const conflicts = (port && getConflicts) ? getConflicts(serviceId, port) : [];

    let fieldsHtml = '';
    fields.forEach(field => {
        const val = getNestedValue(cfg, field.path);
        const fieldId = `simple-${serviceId}-${field.path.replace(/\./g, '-')}`;
        const hintHtml = field.hint ? `<small style="color:var(--text-muted);">${field.hint}</small>` : '';

        // Check if this is a URL pointing to another service with a port conflict
        let linkedWarning = '';
        if (field.linkedService && typeof val === 'string') {
            const linkedCfg = allConfigs[field.linkedService];
            if (linkedCfg) {
                const linkedPort = getNestedValue(linkedCfg, 'server.port');
                if (linkedPort && !val.includes(':' + linkedPort)) {
                    linkedWarning = `<div style="color:#e67e22;font-size:0.8em;">⚠️ ${field.linkedService} läuft auf Port ${linkedPort}, URL zeigt woanders hin</div>`;
                }
            }
        }

        if (field.type === 'bool') {
            fieldsHtml += `
                <div style="display:flex;align-items:center;gap:0.7rem;padding:0.35rem 0;">
                    <input type="checkbox" id="${fieldId}" ${val ? 'checked' : ''}
                           onchange="${onChangeFn}('${serviceId}','${field.path}',this.checked,'bool')">
                    <label for="${fieldId}" style="margin:0;cursor:pointer;">${field.label}</label>
                    ${hintHtml}
                </div>`;
        } else if (field.type === 'number') {
            const portConflictHtml = field.path === 'server.port' && conflicts.length > 0
                ? `<span style="color:#ff8f8f;font-size:0.8em;">⚠️ belegt von ${conflicts.map(c=>c.instance+'/'+c.service).join(', ')}</span>`
                : '';
            fieldsHtml += `
                <div style="display:flex;align-items:center;gap:0.7rem;padding:0.35rem 0;flex-wrap:wrap;">
                    <label for="${fieldId}" style="min-width:120px;margin:0;">${field.label}</label>
                    <input type="number" id="${fieldId}" value="${val ?? ''}" class="form-input" style="width:120px;"
                           onchange="${onChangeFn}('${serviceId}','${field.path}',Number(this.value),'number')">
                    ${hintHtml} ${portConflictHtml}
                </div>`;
        } else if (field.type === 'audio_mode') {
            const currentMode = getAudioMode(serviceId);
            const assignment = instanceName ? (virtualAssignments[instanceName] || null) : null;
            const assignmentBanner = (currentMode === 'virtual' && assignment)
                ? `<div style="width:100%;margin-top:0.4rem;padding:0.4rem 0.7rem;background:rgba(46,204,113,0.12);border:1px solid rgba(46,204,113,0.25);border-radius:6px;font-size:0.85em;color:#2ecc71;">
                     🔌 <strong>${assignment.slot_name}</strong> &mdash; ${assignment.cable_label}
                     <br><small style="color:var(--text-muted);">Lesen: ${assignment.read_from} | Schreiben: ${assignment.write_to}</small>
                   </div>`
                : (currentMode === 'virtual' && !assignment)
                    ? `<div style="width:100%;margin-top:0.4rem;padding:0.4rem 0.7rem;background:rgba(231,76,60,0.1);border:1px solid rgba(231,76,60,0.25);border-radius:6px;font-size:0.85em;color:#e74c3c;">
                         ⏳ Klicke auf "🔌 Virtuell" um ein Audio-Kabel zuzuweisen
                       </div>`
                    : '';
            fieldsHtml += `
                <div style="display:flex;align-items:center;gap:0.7rem;padding:0.55rem 0;flex-wrap:wrap;border-top:1px solid var(--border-color);margin-top:0.4rem;padding-top:0.6rem;">
                    <label style="min-width:120px;margin:0;font-weight:600;">${field.label}</label>
                    <div style="display:flex;gap:0.3rem;">
                        <button type="button" class="btn ${currentMode === 'all' ? 'btn-primary' : 'btn-secondary'}" style="padding:0.25rem 0.7rem;font-size:0.85em;"
                                onclick="setAudioMode('${serviceId}','all')">Alle</button>
                        <button type="button" class="btn ${currentMode === 'physical' ? 'btn-primary' : 'btn-secondary'}" style="padding:0.25rem 0.7rem;font-size:0.85em;"
                                onclick="setAudioMode('${serviceId}','physical')">🖥️ PC-Geräte</button>
                        <button type="button" class="btn ${currentMode === 'virtual' ? 'btn-primary' : 'btn-secondary'}" style="padding:0.25rem 0.7rem;font-size:0.85em;"
                                onclick="setAudioMode('${serviceId}','virtual')">🔌 Virtuell</button>
                    </div>
                    ${hintHtml}
                    ${assignmentBanner}
                </div>`;
        } else if (field.type === 'audio_input' || field.type === 'audio_output') {
            const allDevices = field.type === 'audio_input' ? audioDevices.inputs : audioDevices.outputs;
            const mode = getAudioMode(serviceId);
            const devices = mode === 'all' ? allDevices
                : mode === 'virtual' ? allDevices.filter(d => d.virtual)
                : allDevices.filter(d => !d.virtual);
            const currentVal = val ?? '';
            let optionsHtml = `<option value="">(nicht gesetzt)</option>`;
            let matched = false;
            devices.forEach(dev => {
                const isSelected = currentVal && (dev.name === currentVal || dev.name.toLowerCase().includes(currentVal.toLowerCase()) || currentVal.toLowerCase().includes(dev.name.toLowerCase()));
                if (isSelected) matched = true;
                const devLower = dev.name.toLowerCase();
                let audioConflicts = [];
                Object.entries(usedAudio).forEach(([key, owners]) => {
                    if (devLower.includes(key) || key.includes(devLower)) {
                        audioConflicts = audioConflicts.concat(owners);
                    }
                });
                const conflictTag = audioConflicts.length > 0
                    ? ` ⚠️ (${audioConflicts.map(c => c.instance).join(', ')})`
                    : '';
                optionsHtml += `<option value="${dev.name}" ${isSelected ? 'selected' : ''}>${dev.name}${conflictTag}</option>`;
            });
            if (currentVal && !matched) {
                optionsHtml += `<option value="${currentVal}" selected>⚠️ ${currentVal} (nicht gefunden)</option>`;
            }
            let selectedConflicts = [];
            if (currentVal) {
                const cvLower = currentVal.toLowerCase();
                Object.entries(usedAudio).forEach(([key, owners]) => {
                    if (cvLower.includes(key) || key.includes(cvLower)) {
                        selectedConflicts = selectedConflicts.concat(owners);
                    }
                });
            }
            const audioConflictHtml = selectedConflicts.length > 0
                ? `<span style="color:#ff8f8f;font-size:0.8em;">⚠️ Bereits genutzt von: ${selectedConflicts.map(c => c.instance + '/' + c.service).join(', ')}</span>`
                : '';
            fieldsHtml += `
                <div style="display:flex;align-items:center;gap:0.7rem;padding:0.35rem 0;flex-wrap:wrap;">
                    <label for="${fieldId}" style="min-width:120px;margin:0;">${field.label}</label>
                    <select id="${fieldId}" class="form-input" style="flex:1;min-width:200px;"
                            onchange="${onChangeFn}('${serviceId}','${field.path}',this.value,'text')">
                        ${optionsHtml}
                    </select>
                    ${hintHtml} ${audioConflictHtml}
                </div>`;
        } else if (field.type === 'audio_output_multi') {
            // Multi-select for additional parallel audio outputs — filtered by audio_mode
            const allDevices = audioDevices.outputs || [];
            const mode = getAudioMode(serviceId);
            const devices = mode === 'all' ? allDevices
                : mode === 'virtual' ? allDevices.filter(d => d.virtual)
                : allDevices.filter(d => !d.virtual);
            const currentArr = Array.isArray(val) ? val : [];
            // Build checkboxes for each device
            let checkboxesHtml = '';
            if (devices.length === 0) {
                checkboxesHtml = '<span style="color:var(--text-muted);font-size:0.85em;">Keine Geräte in diesem Modus verfügbar</span>';
            } else {
                devices.forEach((dev, di) => {
                    const cbId = `${fieldId}-${di}`;
                    const isChecked = currentArr.some(v => v === dev.name || dev.name.toLowerCase().includes(v.toLowerCase()) || v.toLowerCase().includes(dev.name.toLowerCase()));
                    const vTag = dev.virtual ? ' 🔌' : ' 🖥️';
                    checkboxesHtml += `
                        <label style="display:flex;align-items:center;gap:0.4rem;padding:0.2rem 0;cursor:pointer;font-size:0.9em;">
                            <input type="checkbox" id="${cbId}" value="${dev.name}" ${isChecked ? 'checked' : ''}
                                   onchange="${onChangeFn}_multi('${serviceId}','${field.path}','${fieldId}')">
                            ${dev.name}${vTag}
                        </label>`;
                });
            }
            const selectedCount = currentArr.length;
            const countBadge = selectedCount > 0
                ? `<span style="background:#2ecc71;color:#fff;border-radius:10px;padding:0.1rem 0.5rem;font-size:0.8em;">${selectedCount} aktiv</span>`
                : '';
            fieldsHtml += `
                <div style="padding:0.35rem 0;">
                    <div style="display:flex;align-items:center;gap:0.7rem;flex-wrap:wrap;">
                        <label style="min-width:120px;margin:0;">${field.label}</label>
                        ${countBadge}
                        ${hintHtml}
                    </div>
                    <div id="${fieldId}-list" style="max-height:150px;overflow-y:auto;margin-top:0.3rem;padding:0.3rem 0.5rem;border:1px solid var(--border-color);border-radius:6px;background:var(--card-bg,#1a1a2e);">
                        ${checkboxesHtml}
                    </div>
                </div>`;
        } else {
            fieldsHtml += `
                <div style="display:flex;align-items:center;gap:0.7rem;padding:0.35rem 0;flex-wrap:wrap;">
                    <label for="${fieldId}" style="min-width:120px;margin:0;">${field.label}</label>
                    <input type="text" id="${fieldId}" value="${val ?? ''}" class="form-input" style="flex:1;min-width:200px;"
                           onchange="${onChangeFn}('${serviceId}','${field.path}',this.value,'text')">
                    ${hintHtml}
                </div>
                ${linkedWarning}`;
        }
    });

    return fieldsHtml;
}
