// ==================== SHARED: SIMPLE FIELDS DEFINITIONS + AUDIO ====================
// Used by both index.html (create/edit modal) and instance_config.html

var audioDevices = { inputs: [], outputs: [] };
var audioModePerService = {};
var virtualAssignments = {};  // { instanceName: assignment }
var ollamaModels = [];  // cached list of installed Ollama models
var ollamaModelsLoaded = false;
var charactersList = [];  // cached character list for dropdowns
var charactersLoaded = false;

// ===========================================================================
// GLOBAL FIELDS – editable once, propagated to all services
// ===========================================================================
var GLOBAL_FIELDS = [
    {
        id: 'character',
        label: '🎭 Character',
        type: 'character_select',
        hint: 'Setzt automatisch Chat-Character, Voice & 3D-Model',
        read: (cfgs) => getNestedValue(cfgs['ki_chat'] || {}, 'default_character') || '',
        write: (cfgs, val) => {
            // handled by applyCharacterToConfigs + ki_chat default_character
        }
    },
    {
        id: 'llm_model',
        label: '🤖 LLM Model',
        type: 'ollama_model',
        hint: 'Ollama Modell für KI Chat',
        read: (cfgs) => getNestedValue(cfgs['ki_chat'] || {}, 'ollama.default_model') || '',
        write: (cfgs, val) => {
            if (cfgs['ki_chat']) setNestedValue(cfgs['ki_chat'], 'ollama.default_model', val);
        }
    },
    {
        id: 'port_main_server',
        label: '🎯 Main Server Port',
        type: 'number',
        hint: '',
        service: 'main_server',
        read: (cfgs) => getNestedValue(cfgs['main_server'] || {}, 'server.port') ?? '',
        write: (cfgs, val) => {
            if (cfgs['main_server']) setNestedValue(cfgs['main_server'], 'server.port', val);
            _propagatePortToUrls(cfgs, 'main_server', val);
        }
    },
    {
        id: 'port_ki_chat',
        label: '💬 KI Chat Port',
        type: 'number',
        hint: '',
        service: 'ki_chat',
        read: (cfgs) => getNestedValue(cfgs['ki_chat'] || {}, 'server.port') ?? '',
        write: (cfgs, val) => {
            if (cfgs['ki_chat']) setNestedValue(cfgs['ki_chat'], 'server.port', val);
            _propagatePortToUrls(cfgs, 'ki_chat', val);
        }
    },
    {
        id: 'port_text_to_speech',
        label: '🔊 TTS Port',
        type: 'number',
        hint: '',
        service: 'text_to_speech',
        read: (cfgs) => getNestedValue(cfgs['text_to_speech'] || {}, 'server.port') ?? '',
        write: (cfgs, val) => {
            if (cfgs['text_to_speech']) setNestedValue(cfgs['text_to_speech'], 'server.port', val);
            _propagatePortToUrls(cfgs, 'text_to_speech', val);
        }
    },
    {
        id: 'port_web_avatar',
        label: '🌐 Web Avatar Port',
        type: 'number',
        hint: '',
        service: 'web_avatar',
        read: (cfgs) => getNestedValue(cfgs['web_avatar'] || {}, 'server.port') ?? '',
        write: (cfgs, val) => {
            if (cfgs['web_avatar']) setNestedValue(cfgs['web_avatar'], 'server.port', val);
            _propagatePortToUrls(cfgs, 'web_avatar', val);
        }
    },
    {
        id: 'port_vroid_emotion',
        label: '😊 VroidEmotion Port',
        type: 'number',
        hint: '',
        service: 'vroid_emotion',
        read: (cfgs) => getNestedValue(cfgs['vroid_emotion'] || {}, 'server.port') ?? '',
        write: (cfgs, val) => {
            if (cfgs['vroid_emotion']) setNestedValue(cfgs['vroid_emotion'], 'server.port', val);
            _propagatePortToUrls(cfgs, 'vroid_emotion', val);
        }
    },
    {
        id: 'port_vroid_poser',
        label: '🎭 VroidPoser Port',
        type: 'number',
        hint: '',
        service: 'vroid_poser',
        read: (cfgs) => getNestedValue(cfgs['vroid_poser'] || {}, 'server.port') ?? '',
        write: (cfgs, val) => {
            if (cfgs['vroid_poser']) setNestedValue(cfgs['vroid_poser'], 'server.port', val);
            _propagatePortToUrls(cfgs, 'vroid_poser', val);
        }
    },
    {
        id: 'port_ollama',
        label: '🤖 Ollama Port',
        type: 'number',
        hint: '',
        service: 'ollama',
        read: (cfgs) => getNestedValue(cfgs['ollama'] || {}, 'server.port') ?? '',
        write: (cfgs, val) => {
            if (cfgs['ollama']) setNestedValue(cfgs['ollama'], 'server.port', val);
            // Ollama URL in ki_chat
            if (cfgs['ki_chat'] && cfgs['ki_chat'].ollama && cfgs['ki_chat'].ollama.url) {
                const oHost = getNestedValue(cfgs['ollama'] || {}, 'server.host') || '127.0.0.1';
                cfgs['ki_chat'].ollama.url = cfgs['ki_chat'].ollama.url.replace(/\/\/[^/]+/, '//' + oHost + ':' + val);
            }
        }
    },
    {
        id: 'osc_port',
        label: '🔌 OSC Port',
        type: 'number',
        hint: 'Gemeinsamer OSC Port für alle Services',
        read: (cfgs) => getNestedValue(cfgs['text_to_speech'] || cfgs['vroid_emotion'] || cfgs['web_avatar'] || {}, 'osc.port') ||
                         getNestedValue(cfgs['web_avatar'] || {}, 'osc.listen_port') || '',
        write: (cfgs, val) => {
            if (cfgs['text_to_speech']) setNestedValue(cfgs['text_to_speech'], 'osc.port', val);
            if (cfgs['vroid_emotion']) setNestedValue(cfgs['vroid_emotion'], 'osc.port', val);
            if (cfgs['vroid_poser']) setNestedValue(cfgs['vroid_poser'], 'osc.port', val);
            if (cfgs['web_avatar']) setNestedValue(cfgs['web_avatar'], 'osc.listen_port', val);
        }
    },
];

/**
 * Map of which services reference which other services' ports in their `services.*` URLs.
 * Key = target service whose port changed. Values = [{service, path}] where the URL lives.
 */
var PORT_URL_MAP = {
    ki_chat:        [{svc: 'main_server', path: 'services.ki_chat'}],
    text_to_speech: [{svc: 'main_server', path: 'services.text_to_speech'}],
    vroid_poser:    [{svc: 'main_server', path: 'services.vroid_poser'}],
    vroid_emotion:  [
        {svc: 'main_server',    path: 'services.vroid_emotion'},
        {svc: 'text_to_speech', path: 'services.vroid_emotion'},
    ],
    web_avatar: [
        {svc: 'vroid_emotion', path: 'services.web_avatar'},
        {svc: 'vroid_poser',   path: 'services.web_avatar'},
    ],
    main_server: [
        {svc: 'text_to_speech', path: 'services.main_server'},
    ],
};

function _propagatePortToUrls(cfgs, changedService, newPort) {
    const refs = PORT_URL_MAP[changedService] || [];
    refs.forEach(ref => {
        const cfg = cfgs[ref.svc];
        if (!cfg) return;
        const currentUrl = getNestedValue(cfg, ref.path);
        if (typeof currentUrl === 'string') {
            setNestedValue(cfg, ref.path, currentUrl.replace(/:\d+$/, ':' + newPort));
        }
    });
}

/**
 * Build the HTML for the Global Settings card.
 * @param {object} allConfigs - all service configs (pendingServiceConfigs)
 * @param {string} onChangeFn - JS callback name for field changes
 * @returns {string} HTML
 */
function buildGlobalSettingsHtml(allConfigs, onChangeFn) {
    let html = '';

    GLOBAL_FIELDS.forEach(field => {
        // Skip fields whose service is not in the config set
        if (field.service && !allConfigs[field.service]) return;

        const val = field.read(allConfigs);
        const fieldId = `global-${field.id}`;
        const hintHtml = field.hint ? `<small style="color:var(--text-muted);">${field.hint}</small>` : '';

        if (field.type === 'character_select') {
            const currentVal = val || '';
            let optionsHtml = `<option value="">(kein Character)</option>`;
            let matched = false;
            charactersList.forEach(c => {
                const isSelected = currentVal === c.id;
                if (isSelected) matched = true;
                const voiceTag = c.has_voice ? '🎙️' : '';
                const modelTag = c.has_model ? '🧊' : '';
                optionsHtml += `<option value="${c.id}" ${isSelected ? 'selected' : ''}>${c.name} (${c.id}) ${voiceTag}${modelTag}</option>`;
            });
            if (currentVal && !matched) {
                optionsHtml += `<option value="${currentVal}" selected>⚠️ ${currentVal}</option>`;
            }
            html += `
                <div style="display:flex;align-items:center;gap:0.7rem;padding:0.4rem 0;flex-wrap:wrap;">
                    <label for="${fieldId}" style="min-width:150px;margin:0;font-weight:600;">${field.label}</label>
                    <select id="${fieldId}" class="form-input" style="flex:1;min-width:200px;"
                            onchange="globalFieldChanged('${field.id}', this.value, '${onChangeFn}')">
                        ${optionsHtml}
                    </select>
                    <button type="button" class="btn btn-secondary" style="padding:0.2rem 0.5rem;font-size:0.8em;"
                            onclick="refreshCharactersList()" title="Neu laden">🔄</button>
                    ${hintHtml}
                </div>`;
            if (!charactersLoaded) { loadCharactersList().then(() => { if (typeof renderSimpleEditors === 'function') renderSimpleEditors(); }); }
        } else if (field.type === 'ollama_model') {
            const currentVal = val || '';
            let optionsHtml = `<option value="">(nicht gesetzt)</option>`;
            let matched = false;
            ollamaModels.forEach(m => {
                const isSelected = currentVal === m.name;
                if (isSelected) matched = true;
                const sizeTag = m.size_gb ? ` (${m.size_gb} GB)` : '';
                optionsHtml += `<option value="${m.name}" ${isSelected ? 'selected' : ''}>${m.name}${sizeTag}</option>`;
            });
            if (currentVal && !matched) {
                optionsHtml += `<option value="${currentVal}" selected>⚠️ ${currentVal}</option>`;
            }
            html += `
                <div style="display:flex;align-items:center;gap:0.7rem;padding:0.4rem 0;flex-wrap:wrap;">
                    <label for="${fieldId}" style="min-width:150px;margin:0;font-weight:600;">${field.label}</label>
                    <select id="${fieldId}" class="form-input" style="flex:1;min-width:200px;"
                            onchange="globalFieldChanged('${field.id}', this.value, '${onChangeFn}')">
                        ${optionsHtml}
                    </select>
                    <button type="button" class="btn btn-secondary" style="padding:0.2rem 0.5rem;font-size:0.8em;"
                            onclick="refreshOllamaModels()" title="Neu laden">🔄</button>
                    ${hintHtml}
                </div>`;
            if (!ollamaModelsLoaded) { loadOllamaModels().then(() => { if (typeof renderSimpleEditors === 'function') renderSimpleEditors(); }); }
        } else if (field.type === 'number') {
            html += `
                <div style="display:flex;align-items:center;gap:0.7rem;padding:0.35rem 0;flex-wrap:wrap;">
                    <label for="${fieldId}" style="min-width:150px;margin:0;">${field.label}</label>
                    <input type="number" id="${fieldId}" value="${val}" class="form-input" style="width:100px;"
                           onchange="globalFieldChanged('${field.id}', Number(this.value), '${onChangeFn}')">
                    ${hintHtml}
                </div>`;
        } else {
            html += `
                <div style="display:flex;align-items:center;gap:0.7rem;padding:0.35rem 0;flex-wrap:wrap;">
                    <label for="${fieldId}" style="min-width:150px;margin:0;">${field.label}</label>
                    <input type="text" id="${fieldId}" value="${val}" class="form-input" style="flex:1;min-width:200px;"
                           onchange="globalFieldChanged('${field.id}', this.value, '${onChangeFn}')">
                    ${hintHtml}
                </div>`;
        }
    });

    return html;
}

/**
 * Handle a change from the Global Settings panel.
 * Propagates the value to all affected service configs and re-renders.
 * Works in two contexts:
 *  1. index.html create/edit modal (pendingServiceConfigs in memory)
 *  2. instance_config.html (delegates to icGlobalFieldChanged via onChangeFn)
 */
function globalFieldChanged(fieldId, value, onChangeFn) {
    // If the onChangeFn is 'icGlobalFieldChanged', delegate to the instance_config handler
    if (onChangeFn === 'icGlobalFieldChanged' && typeof window['icGlobalFieldChanged'] === 'function') {
        window['icGlobalFieldChanged'](fieldId, value, onChangeFn);
        return;
    }

    // Only works in the create/edit modal context (pendingServiceConfigs)
    if (typeof pendingServiceConfigs === 'undefined' || !pendingServiceConfigs) return;

    const field = GLOBAL_FIELDS.find(f => f.id === fieldId);
    if (!field) return;

    // Character is special — use the existing applyCharacterToConfigs
    if (field.type === 'character_select') {
        if (pendingServiceConfigs['ki_chat']) {
            setNestedValue(pendingServiceConfigs['ki_chat'], 'default_character', value);
        }
        if (value) {
            applyCharacterToConfigs(value, pendingServiceConfigs, null);
        }
    } else {
        field.write(pendingServiceConfigs, value);
    }

    if (typeof renderPortConflictSummary === 'function') renderPortConflictSummary();
    if (typeof refreshAllServiceConflictLabels === 'function') refreshAllServiceConflictLabels();
    if (typeof renderSimpleEditors === 'function') renderSimpleEditors();
}

async function loadOllamaModels() {
    if (ollamaModelsLoaded) return;
    try {
        const result = await apiCall('/api/ollama/models');
        if (result && result.success) {
            ollamaModels = result.models || [];
        } else {
            console.warn('Ollama models not available:', result ? result.error : 'unknown');
            ollamaModels = [];
        }
    } catch (e) {
        console.warn('Could not load Ollama models:', e);
        ollamaModels = [];
    }
    ollamaModelsLoaded = true;
}

async function refreshOllamaModels(serviceId, path, onChangeFn) {
    ollamaModelsLoaded = false;
    await loadOllamaModels();
    if (typeof renderICSimple === 'function') renderICSimple();
}

async function loadCharactersList() {
    if (charactersLoaded) return;
    try {
        const result = await apiCall('/api/characters/list-simple');
        if (result && result.success) {
            charactersList = result.characters || [];
        } else {
            console.warn('Characters not available:', result ? result.error : 'unknown');
            charactersList = [];
        }
    } catch (e) {
        console.warn('Could not load characters:', e);
        charactersList = [];
    }
    charactersLoaded = true;
}

async function refreshCharactersList() {
    charactersLoaded = false;
    await loadCharactersList();
    if (typeof renderICSimple === 'function') renderICSimple();
    if (typeof renderSimpleEditors === 'function') renderSimpleEditors();
}

/**
 * Apply a selected character's assets across all service configs.
 * Updates ki_chat default_character, text_to_speech emotions, and web_avatar vrm.model_path.
 * @param {string} charId - character ID
 * @param {object} allConfigs - reference to pendingServiceConfigs or similar
 * @param {function} rerenderFn - function to call after applying
 */
function applyCharacterToConfigs(charId, allConfigs, rerenderFn) {
    const char = charactersList.find(c => c.id === charId);
    if (!char) return;

    // 1. ki_chat: set default_character
    if (allConfigs['ki_chat']) {
        setNestedValue(allConfigs['ki_chat'], 'default_character', charId);
    }

    // 2. text_to_speech: set voice file for all emotions
    if (allConfigs['text_to_speech'] && char.voice_file) {
        const voicePath = 'voices/' + char.voice_file;
        const ttsCfg = allConfigs['text_to_speech'];
        if (ttsCfg.emotions && typeof ttsCfg.emotions === 'object') {
            Object.keys(ttsCfg.emotions).forEach(emotion => {
                ttsCfg.emotions[emotion] = voicePath;
            });
        }
    }

    // 3. web_avatar: set vrm model path
    if (allConfigs['web_avatar'] && char.model_file) {
        const modelPath = 'models/' + char.model_file;
        setNestedValue(allConfigs['web_avatar'], 'vrm.model_path', modelPath);
    }

    if (rerenderFn) rerenderFn();
}

/**
 * Handler called when a character_select dropdown changes.
 * Detects the context (index.html create modal vs instance_config.html) and applies accordingly.
 */
function handleCharacterSelect(charId, serviceId, onChangeFn) {
    // 1. Set ki_chat default_character via the normal field change callback
    if (typeof window[onChangeFn] === 'function') {
        window[onChangeFn](serviceId, 'default_character', charId, 'text');
    }

    // 2. Apply character assets across all services
    if (typeof pendingServiceConfigs !== 'undefined' && pendingServiceConfigs) {
        // index.html create/edit modal context — apply in-memory
        if (charId) {
            applyCharacterToConfigs(charId, pendingServiceConfigs, 
                typeof renderSimpleEditors === 'function' ? renderSimpleEditors : null);
        }
    } else if (typeof currentInstanceName !== 'undefined' && currentInstanceName && charId) {
        // instance_config.html context — call API to apply across all service configs
        fetch(`/api/instance/${currentInstanceName}/apply-character`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ character_id: charId })
        })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                const parts = [];
                if (data.voice) parts.push('🎙️ ' + data.voice);
                if (data.model) parts.push('🧊 ' + data.model);
                const msg = `Character "${charId}" angewendet` + (parts.length ? ': ' + parts.join(', ') : '');
                if (typeof showNotification === 'function') showNotification('✅ ' + msg, 'success');
                // Reload current service config to show updated values
                if (typeof loadServiceConfig === 'function' && currentServiceId) {
                    loadServiceConfig(currentServiceId);
                }
            } else {
                if (typeof showNotification === 'function') showNotification('❌ ' + (data.error || 'Fehler'), 'error');
            }
        })
        .catch(e => {
            if (typeof showNotification === 'function') showNotification('❌ Netzwerkfehler: ' + e.message, 'error');
        });
    }
}

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
        { path: 'ollama.default_model', label: 'LLM Model', type: 'ollama_model', hint: 'Installiertes Ollama-Modell auswählen' },
        { path: 'default_character', label: '🎭 Character', type: 'character_select', hint: 'Character für diese Instanz – setzt auch Voice & Model automatisch' },
    ],
    text_to_speech: [
        { path: '_character_voice', label: '🎭 Character Voice', type: 'character_select_voice', hint: 'Voice wird automatisch vom Character gesetzt' },
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
    web_avatar: [
        { path: 'server.port', label: 'Server Port', type: 'number', hint: 'HTTP-Port des Web Avatar (Standard: 5006)' },
        { path: 'server.host', label: 'Host', type: 'text' },
        { path: '_character_model', label: '🎭 Character Model', type: 'character_select_model', hint: 'VRM wird automatisch vom Character gesetzt' },
        { path: 'vrm.model_path', label: 'VRM Modell', type: 'text', hint: 'Pfad zur .vrm Datei (relativ zum Service-Ordner)' },
        { path: 'camera.fov', label: '📷 FOV', type: 'number', hint: 'Sichtfeld der Kamera in Grad (Standard: 28)' },
        { path: 'camera.position.0', label: '📷 Position X', type: 'number_float', hint: 'Kamera X-Position' },
        { path: 'camera.position.1', label: '📷 Position Y', type: 'number_float', hint: 'Kamera Y (Höhe)' },
        { path: 'camera.position.2', label: '📷 Position Z', type: 'number_float', hint: 'Kamera Z (Entfernung)' },
        { path: 'camera.target.0', label: '🎯 Ziel X', type: 'number_float', hint: 'Blickpunkt X' },
        { path: 'camera.target.1', label: '🎯 Ziel Y', type: 'number_float', hint: 'Blickpunkt Y (Höhe)' },
        { path: 'camera.target.2', label: '🎯 Ziel Z', type: 'number_float', hint: 'Blickpunkt Z' },
        { path: 'osc.enabled', label: 'OSC Empfang', type: 'bool', hint: 'OSC-Daten von VSeeFace empfangen' },
        { path: 'osc.listen_port', label: 'OSC Port', type: 'number', hint: 'OSC Listen Port' },
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
        const k = keys[i];
        const nextKey = keys[i + 1];
        if (cur[k] === undefined || typeof cur[k] !== 'object') {
            // Create array if next key is numeric, else object
            cur[k] = /^\d+$/.test(nextKey) ? [] : {};
        }
        cur = cur[k];
    }
    const lastKey = keys[keys.length - 1];
    cur[/^\d+$/.test(lastKey) ? parseInt(lastKey) : lastKey] = value;
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
        } else if (field.type === 'number_float') {
            fieldsHtml += `
                <div style="display:flex;align-items:center;gap:0.7rem;padding:0.35rem 0;flex-wrap:wrap;">
                    <label for="${fieldId}" style="min-width:120px;margin:0;">${field.label}</label>
                    <input type="number" step="0.01" id="${fieldId}" value="${val ?? ''}" class="form-input" style="width:120px;"
                           onchange="${onChangeFn}('${serviceId}','${field.path}',parseFloat(this.value),'number')">
                    ${hintHtml}
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
        } else if (field.type === 'ollama_model') {
            // Dropdown populated from Ollama API /api/tags
            const currentVal = val ?? '';
            let optionsHtml = `<option value="">(nicht gesetzt)</option>`;
            let matched = false;
            if (ollamaModels.length > 0) {
                ollamaModels.forEach(m => {
                    const isSelected = currentVal === m.name;
                    if (isSelected) matched = true;
                    const sizeTag = m.size_gb ? ` (${m.size_gb} GB` + (m.parameter_size ? `, ${m.parameter_size}` : '') + (m.quantization ? `, ${m.quantization}` : '') + ')' : '';
                    optionsHtml += `<option value="${m.name}" ${isSelected ? 'selected' : ''}>${m.name}${sizeTag}</option>`;
                });
            }
            if (currentVal && !matched) {
                optionsHtml += `<option value="${currentVal}" selected>⚠️ ${currentVal} (nicht in Ollama gefunden)</option>`;
            }
            const refreshBtnId = `${fieldId}-refresh`;
            fieldsHtml += `
                <div style="display:flex;align-items:center;gap:0.7rem;padding:0.35rem 0;flex-wrap:wrap;">
                    <label for="${fieldId}" style="min-width:120px;margin:0;">${field.label}</label>
                    <select id="${fieldId}" class="form-input" style="flex:1;min-width:200px;"
                            onchange="${onChangeFn}('${serviceId}','${field.path}',this.value,'text')">
                        ${optionsHtml}
                    </select>
                    <button type="button" id="${refreshBtnId}" class="btn btn-secondary" style="padding:0.25rem 0.6rem;font-size:0.85em;"
                            onclick="refreshOllamaModels('${serviceId}','${field.path}','${onChangeFn}')"
                            title="Modell-Liste neu laden">🔄</button>
                    ${hintHtml}
                </div>`;
            // Trigger lazy load if not yet loaded
            if (!ollamaModelsLoaded) { loadOllamaModels().then(() => { if (typeof renderICSimple === 'function') renderICSimple(); }); }
        } else if (field.type === 'character_select') {
            // Full character dropdown – sets default_character AND applies voice/model across services
            const currentVal = val ?? '';
            let optionsHtml = `<option value="">(kein Character)</option>`;
            let matched = false;
            charactersList.forEach(c => {
                const isSelected = currentVal === c.id;
                if (isSelected) matched = true;
                const voiceTag = c.has_voice ? '🎙️' : '';
                const modelTag = c.has_model ? '🧊' : '';
                optionsHtml += `<option value="${c.id}" ${isSelected ? 'selected' : ''}>${c.name} (${c.id}) ${voiceTag}${modelTag}</option>`;
            });
            if (currentVal && !matched) {
                optionsHtml += `<option value="${currentVal}" selected>⚠️ ${currentVal} (nicht gefunden)</option>`;
            }
            fieldsHtml += `
                <div style="display:flex;align-items:center;gap:0.7rem;padding:0.55rem 0;flex-wrap:wrap;border-top:1px solid var(--border-color);margin-top:0.4rem;padding-top:0.6rem;">
                    <label for="${fieldId}" style="min-width:120px;margin:0;font-weight:600;">${field.label}</label>
                    <select id="${fieldId}" class="form-input" style="flex:1;min-width:200px;"
                            onchange="handleCharacterSelect(this.value, '${serviceId}', '${onChangeFn}')">
                        ${optionsHtml}
                    </select>
                    <button type="button" class="btn btn-secondary" style="padding:0.25rem 0.6rem;font-size:0.85em;"
                            onclick="refreshCharactersList()"
                            title="Character-Liste neu laden">🔄</button>
                    ${hintHtml}
                </div>`;
            if (!charactersLoaded) { loadCharactersList().then(() => { if (typeof renderICSimple === 'function') renderICSimple(); if (typeof renderSimpleEditors === 'function') renderSimpleEditors(); }); }
        } else if (field.type === 'character_select_voice') {
            // Read-only display showing which voice is assigned by the character
            const charId = getNestedValue(cfg, 'default_character') || getNestedValue(allConfigs['ki_chat'] || {}, 'default_character') || '';
            const char = charactersList.find(c => c.id === charId);
            const voiceInfo = char && char.has_voice ? `🎙️ ${char.voice_file}` : '<span style="color:var(--text-muted);">Kein Character / keine Voice</span>';
            fieldsHtml += `
                <div style="display:flex;align-items:center;gap:0.7rem;padding:0.35rem 0;flex-wrap:wrap;">
                    <label style="min-width:120px;margin:0;">${field.label}</label>
                    <span style="font-size:0.9em;">${voiceInfo}</span>
                    ${hintHtml}
                </div>`;
        } else if (field.type === 'character_select_model') {
            // Read-only display showing which model is assigned by the character
            const charId = getNestedValue(cfg, 'default_character') || getNestedValue(allConfigs['ki_chat'] || {}, 'default_character') || '';
            const char = charactersList.find(c => c.id === charId);
            const modelInfo = char && char.has_model ? `🧊 ${char.model_file}` : '<span style="color:var(--text-muted);">Kein Character / kein Model</span>';
            fieldsHtml += `
                <div style="display:flex;align-items:center;gap:0.7rem;padding:0.35rem 0;flex-wrap:wrap;">
                    <label style="min-width:120px;margin:0;">${field.label}</label>
                    <span style="font-size:0.9em;">${modelInfo}</span>
                    ${hintHtml}
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
