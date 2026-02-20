// ==================== UTILITY FUNCTIONS ====================

/**
 * Format uptime from seconds
 */
function formatUptime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

/**
 * Format bytes to human readable
 */
function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Show notification/toast
 */
function showNotification(message, type = 'info', duration = 4000) {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        document.body.appendChild(container);
    }

    const icons = { success: '✓', error: '✕', warning: '⚠', info: 'ℹ' };
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || icons.info}</span>
        <span class="toast-message">${message}</span>
        <button class="toast-close" onclick="this.parentElement.remove()">×</button>
    `;

    container.appendChild(toast);
    // Trigger entrance animation
    requestAnimationFrame(() => toast.classList.add('toast-visible'));

    // Auto-dismiss
    const timer = setTimeout(() => {
        toast.classList.remove('toast-visible');
        toast.addEventListener('transitionend', () => toast.remove(), { once: true });
        // Fallback removal
        setTimeout(() => { if (toast.parentElement) toast.remove(); }, 400);
    }, duration);

    // Pause timer on hover
    toast.addEventListener('mouseenter', () => clearTimeout(timer));
    toast.addEventListener('mouseleave', () => {
        setTimeout(() => {
            toast.classList.remove('toast-visible');
            toast.addEventListener('transitionend', () => toast.remove(), { once: true });
            setTimeout(() => { if (toast.parentElement) toast.remove(); }, 400);
        }, 1500);
    });
}

/**
 * Format timestamp
 */
function formatTime() {
    const now = new Date();
    return now.toLocaleTimeString('de-DE');
}

// ==================== API CALLS ====================

/**
 * Fetch wrapper with error handling
 */
async function apiCall(url, options = {}) {
    try {
        const response = await fetch(url, options);
        if (!response.ok) {
            let detailedError = `HTTP ${response.status}: ${response.statusText}`;
            try {
                const errorBody = await response.json();
                if (errorBody && errorBody.error) {
                    detailedError = errorBody.error;
                }
            } catch (parseError) {
                // keep default HTTP error text
            }
            throw new Error(detailedError);
        }
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

/**
 * Get all services status
 */
async function getServicesStatus(instanceName) {
    try {
        const inst = instanceName || window.currentInstanceName;
        const url = inst ? `/api/status/all?instance=${encodeURIComponent(inst)}` : '/api/status/all';
        const result = await apiCall(url);
        return result.services || {};
    } catch (error) {
        console.error('Error fetching statuses:', error);
        return {};
    }
}

/**
 * Get all instances
 */
async function getInstances() {
    try {
        const result = await apiCall('/api/instance/list');
        return result.instances || [];
    } catch (error) {
        console.error('Error fetching instances:', error);
        return [];
    }
}

/**
 * Get specific instance
 */
async function getInstance(instanceName) {
    try {
        const result = await apiCall(`/api/instance/get/${instanceName}`);
        return result.instance || null;
    } catch (error) {
        console.error('Error fetching instance:', error);
        return null;
    }
}

/**
 * Start service
 */
async function startService(serviceId, instanceName) {
    try {
        const inst = instanceName || window.currentInstanceName;
        if (!inst) { showNotification('Keine Instanz ausgewählt', 'error'); return { success: false }; }
        const result = await apiCall(`/api/service/${encodeURIComponent(inst)}/${serviceId}/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        if (result.success) {
            showNotification(`Service "${serviceId}" gestartet`, 'success');
            updateServiceStatuses();
        } else {
            showNotification(result.error || 'Fehler beim Starten', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error starting service:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Stop service
 */
async function stopService(serviceId, instanceName) {
    try {
        const inst = instanceName || window.currentInstanceName;
        if (!inst) { showNotification('Keine Instanz ausgewählt', 'error'); return { success: false }; }
        const result = await apiCall(`/api/service/${encodeURIComponent(inst)}/${serviceId}/stop`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        if (result.success) {
            showNotification(`Service "${serviceId}" gestoppt`, 'success');
            updateServiceStatuses();
        } else {
            showNotification(result.error || 'Fehler beim Stoppen', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error stopping service:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Restart service
 */
async function restartService(serviceId, instanceName) {
    try {
        const inst = instanceName || window.currentInstanceName;
        if (!inst) { showNotification('Keine Instanz ausgewählt', 'error'); return { success: false }; }
        const result = await apiCall(`/api/service/${encodeURIComponent(inst)}/${serviceId}/restart`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        if (result.success) {
            showNotification(`Service "${serviceId}" neu gestartet`, 'success');
            updateServiceStatuses();
        } else {
            showNotification(result.error || 'Fehler beim Neustarten', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error restarting service:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Start all services
 */
async function startAllServices(instanceName) {
    const inst = instanceName || window.currentInstanceName;
    const statuses = await getServicesStatus(inst);
    for (const serviceId of Object.keys(statuses)) {
        await startService(serviceId, inst);
        await new Promise(r => setTimeout(r, 500)); // Delay between starts
    }
}

/**
 * Stop all services
 */
async function stopAllServices(instanceName) {
    const inst = instanceName || window.currentInstanceName;
    const statuses = await getServicesStatus(inst);
    for (const serviceId of Object.keys(statuses)) {
        await stopService(serviceId, inst);
        await new Promise(r => setTimeout(r, 500)); // Delay between stops
    }
}

/**
 * Get service logs
 */
async function getServiceLogs(serviceId, lines = 100, instanceName) {
    try {
        const inst = instanceName || window.currentInstanceName;
        let url = `/api/service/logs/${serviceId}?lines=${lines}`;
        if (inst) url += `&instance=${encodeURIComponent(inst)}`;
        const result = await apiCall(url);
        return result.logs || '';
    } catch (error) {
        console.error('Error fetching logs:', error);
        return '';
    }
}

/**
 * Set current instance
 */
async function setCurrentInstance(instanceName) {
    try {
        const result = await apiCall(`/api/instance/set-current/${instanceName}`, {
            method: 'POST'
        });
        if (result.success) {
            showNotification(`Instanz zu "${instanceName}" gewechselt`, 'success');
            location.reload();
        } else {
            showNotification(result.error || 'Fehler beim Wechsel', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error switching instance:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Switch instance from selector
 */
async function switchInstance() {
    const selector = document.getElementById('current-instance');
    if (selector) {
        await setCurrentInstance(selector.value);
    }
}

/**
 * Save instance
 */
async function saveInstance(instanceData) {
    try {
        const result = await apiCall('/api/instance/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(instanceData)
        });
        if (result.success) {
            showNotification('Instanz gespeichert', 'success');
            location.reload();
        } else {
            showNotification(result.error || 'Fehler beim Speichern', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error saving instance:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Delete instance
 */
async function deleteInstance(instanceName) {
    if (!confirm(`Instanz "${instanceName}" wirklich löschen?`)) return;
    try {
        const result = await apiCall(`/api/instance/delete/${instanceName}`, {
            method: 'DELETE'
        });
        if (result.success) {
            showNotification('Instanz gelöscht', 'success');
            location.reload();
        } else {
            showNotification(result.error || 'Fehler beim Löschen', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error deleting instance:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Get configuration
 */
async function getConfig(serviceId, instanceName) {
    try {
        const inst = instanceName || window.currentInstanceName;
        if (serviceId === 'manager') {
            const result = await apiCall('/api/config/get');
            return result.config || {};
        } else if (inst) {
            // Instance-aware: load from per-instance service config
            const result = await apiCall(`/api/instance/${encodeURIComponent(inst)}/config/service/${encodeURIComponent(serviceId)}`);
            return result.config || {};
        } else {
            const result = await apiCall(`/api/config/service/${serviceId}`);
            return result.config || {};
        }
    } catch (error) {
        console.error('Error fetching config:', error);
        return {};
    }
}

/**
 * Save configuration
 */
async function saveConfiguration(serviceId, config, instanceName) {
    try {
        const inst = instanceName || window.currentInstanceName;
        let url;
        if (serviceId === 'manager') {
            url = '/api/config/service/manager/save';
        } else if (inst) {
            // Instance-aware: save to per-instance service config
            url = `/api/instance/${encodeURIComponent(inst)}/config/service/${encodeURIComponent(serviceId)}/save`;
        } else {
            url = `/api/config/service/${serviceId}/save`;
        }
        const result = await apiCall(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        if (result.success) {
            showNotification('Config gespeichert', 'success');
        } else {
            showNotification(result.error || 'Fehler beim Speichern', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error saving config:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Update service statuses in dashboard
 */
async function updateServiceStatuses() {
    const statuses = await getServicesStatus();
    for (const [serviceId, status] of Object.entries(statuses)) {
        updateServiceCard(serviceId, status);
    }
}

// ==================== UI UPDATES ====================

/**
 * Update service card status
 */
function updateServiceCard(serviceId, status) {
    const card = document.querySelector(`[data-service="${serviceId}"]`);
    if (!card) return;
    
    const badge = card.querySelector('.status-badge');
    if (status && status.status === 'running') {
        badge.className = 'status-badge status-running';
        badge.textContent = 'Läuft';
    } else {
        badge.className = 'status-badge status-stopped';
        badge.textContent = 'Gestoppt';
    }
}

// ==================== CONFIG MANAGEMENT ====================

let currentEditingConfig = 'manager';

/**
 * Select and load a configuration
 */
async function selectConfig(configName) {
    currentEditingConfig = configName;
    
    // Update active config item
    document.querySelectorAll('.config-item').forEach(item => {
        item.classList.remove('active');
    });
    event.target.closest('.config-item').classList.add('active');
    
    // Update title
    const names = {
        'manager': 'Manager Config',
        'ki_chat': 'KI Chat Config',
        'main_server': 'Main Server Config',
        'text_to_speech': 'Text-to-Speech Config',
        'vroid_poser': 'VRoid Poser Config',
        'vroid_emotion': 'VRoid Emotion Config'
    };
    
    document.getElementById('editor-title').textContent = names[configName] || configName;
    
    // Load config
    const editor = document.getElementById('config-editor');
    editor.value = 'Wird geladen...';
    
    const config = await getConfig(configName);
    editor.value = JSON.stringify(config, null, 2);
    
    document.getElementById('line-count').textContent = editor.value.split('\n').length;
    document.getElementById('file-size').textContent = (editor.value.length / 1024).toFixed(1) + ' KB';
}

/**
 * Reload configuration from server
 */
async function reloadConfig() {
    const inst = window.currentInstanceName;
    const config = await getConfig(currentEditingConfig, inst);
    const editor = document.getElementById('config-editor');
    editor.value = JSON.stringify(config, null, 2);
    showNotification('Config neu geladen', 'info');
}

/**
 * Save configuration to server
 */
async function saveConfig() {
    try {
        const editor = document.getElementById('config-editor');
        const config = JSON.parse(editor.value);
        const inst = window.currentInstanceName;
        const result = await saveConfiguration(currentEditingConfig, config, inst);
        
        if (result.success) {
            showNotification('Config gespeichert!', 'success');
        }
    } catch (error) {
        if (error instanceof SyntaxError) {
            showNotification('JSON Fehler: ' + error.message, 'error');
        } else {
            showNotification('Fehler beim Speichern: ' + error.message, 'error');
        }
    }
}

// ==================== INSTANCE MANAGEMENT ====================

var allServices = {};
var currentEditingInstanceName = null;

/**
 * Create a new instance
 */
async function createInstance() {
    currentEditingInstanceName = null;
    
    // Load services first
    allServices = await getServicesStatus();
    
    // Populate modal with services
    const modal = document.getElementById('create-instance-modal');
    const modalTitle = document.getElementById('modalTitle');
    const deleteBtn = document.getElementById('modal-delete-btn');
    const checkboxesContainer = document.querySelector('.service-checkboxes');
    if (checkboxesContainer) {
        checkboxesContainer.innerHTML = '';
        Object.keys(allServices).forEach(serviceId => {
            const label = document.createElement('label');
            label.className = 'service-checkbox';
            label.innerHTML = `
                <div style="display: flex; gap: 1rem; align-items: center;">
                    <div>
                        <input type="checkbox" name="service_${serviceId}" value="${serviceId}" checked>
                        <span>${serviceId}</span>
                    </div>
                    <div style="margin-left: auto;">
                        <input type="checkbox" name="autostart_${serviceId}" value="${serviceId}" class="autostart-checkbox" title="Auto-Start">
                        <small>Auto-Start</small>
                    </div>
                </div>
            `;
            checkboxesContainer.appendChild(label);
        });
    }
    
    // Clear form
    const form = document.getElementById('create-instance-form');
    if (form) {
        form.reset();
    }

    if (modalTitle) {
        modalTitle.textContent = 'Neue Instanz erstellen';
    }

    if (deleteBtn) {
        deleteBtn.style.display = 'none';
    }
    
    // Show modal
    if (modal) {
        modal.style.display = 'flex';
    }
}

/**
 * Edit an instance
 */
async function editInstance(instanceName) {
    currentEditingInstanceName = instanceName;
    const instance = await getInstance(instanceName);
    if (!instance) {
        showNotification('Instanz nicht gefunden', 'error');
        return;
    }
    
    // Load services
    allServices = await getServicesStatus();
    
    // Populate modal
    const modal = document.getElementById('create-instance-modal');
    const modalTitle = document.getElementById('modalTitle');
    const deleteBtn = document.getElementById('modal-delete-btn');
    const form = document.getElementById('create-instance-form');
    const checkboxesContainer = document.querySelector('.service-checkboxes');
    
    if (form) {
        form.querySelector('input[name="name"]').value = instance.name || '';
        form.querySelector('textarea[name="description"]').value = instance.description || '';
    }
    
    // Populate service checkboxes with current state
    if (checkboxesContainer) {
        checkboxesContainer.innerHTML = '';
        Object.keys(allServices).forEach(serviceId => {
            const serviceConfig = instance.services && instance.services[serviceId];
            const isEnabled = serviceConfig ? serviceConfig.enabled : true;
            const isAutoStart = serviceConfig ? serviceConfig.auto_start : false;
            
            const label = document.createElement('label');
            label.className = 'service-checkbox';
            label.innerHTML = `
                <div style="display: flex; gap: 1rem; align-items: center;">
                    <div>
                        <input type="checkbox" name="service_${serviceId}" value="${serviceId}" ${isEnabled ? 'checked' : ''}>
                        <span>${serviceId}</span>
                    </div>
                    <div style="margin-left: auto;">
                        <input type="checkbox" name="autostart_${serviceId}" value="${serviceId}" class="autostart-checkbox" ${isAutoStart ? 'checked' : ''} title="Auto-Start">
                        <small>Auto-Start</small>
                    </div>
                </div>
            `;
            checkboxesContainer.appendChild(label);
        });
    }
    
    if (modal) {
        modal.style.display = 'flex';
    }

    if (modalTitle) {
        modalTitle.textContent = `Bearbeite: ${instance.name || instanceName}`;
    }

    if (deleteBtn) {
        deleteBtn.style.display = 'inline-flex';
        deleteBtn.onclick = () => {
            closeModal();
            deleteInstance(instanceName);
        };
    }
}

/**
 * Save instance (create or update)
 */
async function saveInstanceFromModal() {
    const form = document.getElementById('create-instance-form');
    if (!form) return;
    
    const name = form.querySelector('input[name="name"]').value.trim();
    const description = form.querySelector('textarea[name="description"]').value.trim();
    
    if (!name) {
        showNotification('Name ist erforderlich', 'error');
        return;
    }
    
    // Collect selected services with their enabled and auto_start status
    const services = {};
    const allCheckboxes = document.querySelectorAll('.service-checkboxes input[type="checkbox"]');
    
    // First pass: collect all services
    allCheckboxes.forEach(checkbox => {
        if (checkbox.name.startsWith('service_')) {
            const serviceId = checkbox.value;
            if (!services[serviceId]) {
                services[serviceId] = {
                    enabled: false,
                    auto_start: false
                };
            }
            services[serviceId].enabled = checkbox.checked;
        }
    });
    
    // Second pass: set auto_start flags
    allCheckboxes.forEach(checkbox => {
        if (checkbox.name.startsWith('autostart_')) {
            const serviceId = checkbox.value;
            if (!services[serviceId]) {
                services[serviceId] = {
                    enabled: false,
                    auto_start: false
                };
            }
            services[serviceId].auto_start = checkbox.checked;
        }
    });
    
    const instanceData = {
        name: name,
        description: description,
        services: services,
        config_overrides: {}
    };
    
    const result = await saveInstance(instanceData);
    if (result && result.success) {
        closeModal();
        location.reload();
    }
}

/**
 * Toggle service in instance configuration
 */
function toggleServiceInInstance(instanceName, serviceId) {
    // This updates local UI state (handled by checkbox)
    console.log(`Toggled ${serviceId} in ${instanceName}`);
}

/**
 * Close modal
 */
function closeModal() {
    const modal = document.getElementById('create-instance-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

/**
 * Start instance (launches auto-start services)
 */
async function startInstance(instanceName) {
    try {
        const result = await apiCall(`/api/instance/start/${instanceName}`, {
            method: 'POST'
        });
        if (result.success) {
            const msg = `Instanz gestartet! ${result.started.length} Services gestartet`;
            showNotification(msg, 'success');
            setTimeout(() => location.reload(), 2000);
        } else {
            showNotification(result.error || 'Fehler beim Starten', 'error');
        }
        return result;
    } catch (error) {
        console.error('Error starting instance:', error);
        return { success: false, error: error.message };
    }
}

/**
 * Stop all services for an instance
 */
async function stopInstance(instanceName) {
    try {
        // Get instance to find enabled services
        const instance = await getInstance(instanceName);
        if (!instance) {
            showNotification('Instanz nicht gefunden', 'error');
            return;
        }
        
        // Stop all enabled services
        const services = instance.services || {};
        let stoppedCount = 0;
        
        for (const [serviceId, settings] of Object.entries(services)) {
            if (settings.enabled) {
                await stopService(serviceId, instanceName);
                stoppedCount++;
            }
        }
        
        showNotification(`${stoppedCount} Services gestoppt`, 'success');
        setTimeout(() => location.reload(), 1000);
    } catch (error) {
        console.error('Error stopping instance:', error);
        showNotification('Fehler beim Stoppen', 'error');
    }
}

/**
 * Add log entry to container
 */
function addLogEntry(containerId, message, level = 'info', service = null) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const time = formatTime();
    const entry = document.createElement('div');
    entry.className = `log-entry log-${level}`;
    
    let html = `<span class="log-time">${time}</span>`;
    if (service) {
        html += `<span class="log-service">${service}</span>`;
    }
    html += `<span class="log-message">${message}</span>`;
    
    entry.innerHTML = html;
    container.insertBefore(entry, container.firstChild);
    
    // Keep only last 100 entries
    while (container.children.length > 100) {
        container.removeChild(container.lastChild);
    }
}

// ==================== INITIALIZATION ====================

document.addEventListener('DOMContentLoaded', () => {
    console.log('🎭 KI Girl Manager initialized');
    
    // Start uptime counter if element exists
    const uptimeEl = document.getElementById('uptime');
    if (uptimeEl) {
        let uptime = 0;
        setInterval(() => {
            uptime++;
            uptimeEl.textContent = formatUptime(uptime);
        }, 1000);
    }
});

// ==================== GLOBAL ERROR HANDLER ====================

window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    showNotification('Ein Fehler ist aufgetreten', 'error');
});

// ==================== KEYBOARD SHORTCUTS ====================

const _shortcuts = [
    { keys: 'Ctrl + S', description: 'Config speichern (im Editor)' },
    { keys: 'Ctrl + R', description: 'Config neu laden (im Editor)' },
    { keys: '?', description: 'Shortcuts Übersicht anzeigen' },
    { keys: 'Escape', description: 'Modal / Overlay schließen' },
];

function showShortcutsOverlay() {
    let overlay = document.getElementById('shortcuts-overlay');
    if (overlay) {
        overlay.style.display = overlay.style.display === 'flex' ? 'none' : 'flex';
        return;
    }
    overlay = document.createElement('div');
    overlay.id = 'shortcuts-overlay';
    overlay.style.cssText = 'position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,0.6);display:flex;align-items:center;justify-content:center;';
    overlay.onclick = (e) => { if (e.target === overlay) overlay.style.display = 'none'; };
    const rows = _shortcuts.map(s =>
        `<tr><td style="padding:0.5rem 1rem;"><kbd style="background:var(--bg-tertiary);padding:0.2rem 0.6rem;border-radius:4px;font-size:0.85rem;border:1px solid var(--border-color);">${s.keys}</kbd></td><td style="padding:0.5rem 1rem;color:var(--text-secondary);">${s.description}</td></tr>`
    ).join('');
    overlay.innerHTML = `
        <div style="background:var(--bg-secondary);border:1px solid var(--border-color);border-radius:12px;padding:1.5rem 2rem;min-width:360px;box-shadow:var(--shadow-lg);">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                <h3 style="color:var(--text-primary);margin:0;">⌨️ Keyboard Shortcuts</h3>
                <button onclick="document.getElementById('shortcuts-overlay').style.display='none'" style="background:none;border:none;color:var(--text-muted);font-size:1.3rem;cursor:pointer;">×</button>
            </div>
            <table style="width:100%;">${rows}</table>
        </div>`;
    document.body.appendChild(overlay);
}

document.addEventListener('keydown', (event) => {
    // Ctrl/Cmd + S to save config
    if ((event.ctrlKey || event.metaKey) && event.key === 's') {
        const editor = document.getElementById('config-editor');
        if (editor && document.activeElement === editor) {
            event.preventDefault();
            if (typeof saveConfig === 'function') {
                saveConfig();
            }
        }
    }
    
    // Ctrl/Cmd + R to reload config
    if ((event.ctrlKey || event.metaKey) && event.key === 'r') {
        const editor = document.getElementById('config-editor');
        if (editor && document.activeElement === editor) {
            event.preventDefault();
            if (typeof reloadConfig === 'function') {
                reloadConfig();
            }
        }
    }

    // ? to show shortcuts overlay (when not in an input)
    if (event.key === '?' && !event.ctrlKey && !event.metaKey) {
        const tag = document.activeElement?.tagName;
        if (tag !== 'INPUT' && tag !== 'TEXTAREA' && tag !== 'SELECT') {
            event.preventDefault();
            showShortcutsOverlay();
        }
    }

    // Escape to close overlays
    if (event.key === 'Escape') {
        const shortcutsOv = document.getElementById('shortcuts-overlay');
        if (shortcutsOv && shortcutsOv.style.display === 'flex') {
            shortcutsOv.style.display = 'none';
            return;
        }
        if (typeof closeModal === 'function') closeModal();
    }
});

// Export functions to window for onclick handlers
window.apiCall = apiCall;
window.getServicesStatus = getServicesStatus;
window.getInstances = getInstances;
window.getInstance = getInstance;
window.startService = startService;
window.stopService = stopService;
window.restartService = restartService;
window.startAllServices = startAllServices;
window.stopAllServices = stopAllServices;
window.getServiceLogs = getServiceLogs;
window.setCurrentInstance = setCurrentInstance;
window.switchInstance = switchInstance;
window.saveInstance = saveInstance;
window.deleteInstance = deleteInstance;
window.getConfig = getConfig;
window.saveConfiguration = saveConfiguration;
window.updateServiceStatuses = updateServiceStatuses;
window.updateServiceCard = updateServiceCard;
window.selectConfig = selectConfig;
window.reloadConfig = reloadConfig;
window.saveConfig = saveConfig;
window.createInstance = createInstance;
window.editInstance = editInstance;
window.saveInstanceFromModal = saveInstanceFromModal;
window.deleteInstance = deleteInstance;
window.toggleServiceInInstance = toggleServiceInInstance;
window.closeModal = closeModal;
window.goToInstance = window.goToInstance || function(instanceName) {
    location.href = `/instance/${instanceName}/dashboard`;
};
window.createInstanceFromIndex = window.createInstanceFromIndex || createInstance;
window.editInstanceModal = window.editInstanceModal || function(event, instanceName) {
    if (event && typeof event.stopPropagation === 'function') {
        event.stopPropagation();
    }
    return editInstance(instanceName);
};
window.saveInstanceFromIndex = window.saveInstanceFromIndex || saveInstanceFromModal;
window.launchInstance = window.launchInstance || function(event, instanceName) {
    if (event && typeof event.stopPropagation === 'function') {
        event.stopPropagation();
    }
    return startInstance(instanceName);
};
window.deleteInstanceFromIndex = window.deleteInstanceFromIndex || deleteInstance;
window.showNotification = showNotification;
window.formatUptime = formatUptime;
window.formatBytes = formatBytes;
window.formatTime = formatTime;
window.addLogEntry = addLogEntry;
