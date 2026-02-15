
// ==================== SERVICE DEFINITIONS CACHE ====================
let serviceDefinitions = {}; // from config.json
let currentEditingInstanceName = null;

// Load service definitions from config
async function loadServiceDefinitions() {
    try {
        const result = await apiCall('/api/config/get');
        if (result && result.config && result.config.services) {
            serviceDefinitions = result.config.services;
        }
    } catch (error) {
        console.error('Error loading service definitions:', error);
    }
    return serviceDefinitions;
}

// ==================== NAVIGATION ====================

// Navigate to instance dashboard
function goToInstance(instanceName) {
    location.href = `/instance/${instanceName}/dashboard`;
}

// ==================== INSTANCE ACTIONS ====================

// Launch instance (start auto-start services) then navigate
async function launchInstance(event, instanceName) {
    event.stopPropagation();
    
    const btn = event.target.closest('button');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = 'â³ Startet...';
    }
    
    showNotification('Instanz wird gestartet...', 'info');
    try {
        const result = await apiCall(`/api/instance/start/${instanceName}`, {
            method: 'POST'
        });
        if (result && result.success) {
            showNotification(`âœ… Instanz gestartet!`, 'success');
            setTimeout(() => goToInstance(instanceName), 800);
        } else {
            showNotification(`âŒ Fehler: ${result.error || 'Unbekannter Fehler'}`, 'error');
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = 'â–¶ï¸ Starten';
            }
        }
    } catch (error) {
        showNotification(`âŒ Fehler: ${error.message}`, 'error');
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = 'â–¶ï¸ Starten';
        }
    }
}

// Delete instance
async function deleteInstanceFromIndex(instanceName) {
    if (!confirm(`Instanz "${instanceName}" wirklich lÃ¶schen?`)) return;
    try {
        const result = await apiCall(`/api/instance/delete/${instanceName}`, {
            method: 'DELETE'
        });
        if (result && result.success) {
            showNotification('âœ… Instanz gelÃ¶scht!', 'success');
            setTimeout(() => location.reload(), 500);
        } else {
            showNotification(`âŒ Fehler: ${result.error || 'Unbekannter Fehler'}`, 'error');
        }
    } catch (error) {
        showNotification(`âŒ Fehler: ${error.message}`, 'error');
    }
}

// ==================== MODAL: SERVICE CHECKBOXES ====================

function renderServiceCheckboxes(container, instanceServices) {
    container.innerHTML = '';
    
    if (Object.keys(serviceDefinitions).length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted);">Keine Services verfÃ¼gbar</p>';
        return;
    }
    
    Object.entries(serviceDefinitions).forEach(([serviceId, serviceDef]) => {
        const serviceConfig = instanceServices ? instanceServices[serviceId] : null;
        const isEnabled = serviceConfig ? serviceConfig.enabled : true;
        const isAutoStart = serviceConfig ? serviceConfig.auto_start : false;
        const displayName = serviceDef.name || serviceId;
        const icon = serviceDef.icon || 'âš™ï¸';
        const color = serviceDef.color || '#888';
        
        const row = document.createElement('label');
        row.className = 'service-checkbox';
        row.innerHTML = `
            <div style="display: flex; gap: 1rem; align-items: center; padding: 0.5rem 0;">
                <div style="display: flex; align-items: center; gap: 0.5rem; flex: 1;">
                    <input type="checkbox" name="service_${serviceId}" value="${serviceId}" ${isEnabled ? 'checked' : ''}>
                    <span style="color: ${color}; font-size: 1.1em;">${icon}</span>
                    <span>${displayName}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.3rem;">
                    <input type="checkbox" name="autostart_${serviceId}" value="${serviceId}" class="autostart-checkbox" ${isAutoStart ? 'checked' : ''} title="Auto-Start">
                    <small style="color: var(--text-muted);">Auto-Start</small>
                </div>
            </div>
        `;
        container.appendChild(row);
    });
}

// ==================== MODAL: CREATE / EDIT ====================

async function createInstanceFromIndex() {
    try {
        currentEditingInstanceName = null;
        
        // Load service definitions
        await loadServiceDefinitions();
        
        // Clear form
        const form = document.getElementById('create-instance-form');
        if (form) {
            form.reset();
        }
        const title = document.getElementById('modalTitle');
        if (title) title.textContent = 'Neue Instanz erstellen';
        
        // Populate service checkboxes (all enabled by default, no auto-start)
        const checkboxesContainer = document.querySelector('.service-checkboxes');
        if (checkboxesContainer) {
            renderServiceCheckboxes(checkboxesContainer, null);
        }
        
        // Hide delete button for new instances
        const deleteBtn = document.getElementById('modal-delete-btn');
        if (deleteBtn) deleteBtn.style.display = 'none';
    } catch (error) {
        console.error('Error preparing modal:', error);
    }
    
    // ALWAYS show modal, even if something above failed
    const modal = document.getElementById('create-instance-modal');
    if (modal) {
        modal.style.display = 'flex';
    } else {
        alert('Modal element nicht gefunden!');
    }
}

async function editInstanceModal(event, instanceName) {
    event.stopPropagation();
    currentEditingInstanceName = instanceName;
    
    try {
        // Load service definitions + instance data in parallel
        const [_, instanceResult] = await Promise.all([
            loadServiceDefinitions(),
            apiCall(`/api/instance/get/${instanceName}`)
        ]);
        
        const instance = instanceResult || {};
        
        // Set form values
        const form = document.getElementById('create-instance-form');
        if (form) {
            const nameInput = form.querySelector('[name="name"]');
            const descInput = form.querySelector('[name="description"]');
            if (nameInput) nameInput.value = instance.name || '';
            if (descInput) descInput.value = instance.description || '';
        }
        const title = document.getElementById('modalTitle');
        if (title) title.textContent = `Bearbeite: ${instance.name || instanceName}`;
        
        // Populate service checkboxes with instance-specific settings
        const checkboxesContainer = document.querySelector('.service-checkboxes');
        if (checkboxesContainer) {
            renderServiceCheckboxes(checkboxesContainer, instance.services || {});
        }
        
        // Show delete button for existing instances
        const deleteBtn = document.getElementById('modal-delete-btn');
        if (deleteBtn) {
            deleteBtn.style.display = 'inline-flex';
            deleteBtn.onclick = () => { closeModal(); deleteInstanceFromIndex(instanceName); };
        }
    } catch (error) {
        console.error('Error loading instance:', error);
    }
    
    // ALWAYS show modal
    const modal = document.getElementById('create-instance-modal');
    if (modal) {
        modal.style.display = 'flex';
    }
}

// ==================== MODAL: SAVE ====================

async function saveInstanceFromIndex() {
    const form = document.getElementById('create-instance-form');
    if (!form) return;
    
    const name = form.querySelector('input[name="name"]').value.trim();
    const description = form.querySelector('textarea[name="description"]').value.trim();
    
    if (!name) {
        showNotification('âŒ Name ist erforderlich', 'error');
        return;
    }
    
    // Filename: use existing or generate from name
    const filename = currentEditingInstanceName || name.toLowerCase().replace(/[^a-z0-9_-]/g, '_');
    
    // Collect services
    const services = {};
    const checkboxes = document.querySelectorAll('.service-checkboxes input[type="checkbox"]');
    
    checkboxes.forEach(cb => {
        if (cb.name.startsWith('service_')) {
            const serviceId = cb.value;
            services[serviceId] = services[serviceId] || { enabled: false, auto_start: false };
            services[serviceId].enabled = cb.checked;
        }
        if (cb.name.startsWith('autostart_')) {
            const serviceId = cb.value;
            services[serviceId] = services[serviceId] || { enabled: false, auto_start: false };
            services[serviceId].auto_start = cb.checked;
        }
    });
    
    const instanceData = {
        filename: filename,
        name: name,
        description: description,
        services: services,
        config_overrides: {}
    };
    
    try {
        // Direct API call (don't use saveInstance from main.js to avoid double-reload)
        const result = await apiCall('/api/instance/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(instanceData)
        });
        
        if (result && result.success) {
            closeModal();
            showNotification('âœ… Instanz gespeichert!', 'success');
            setTimeout(() => location.reload(), 500);
        } else {
            showNotification(`âŒ Fehler: ${result.error || 'Unbekannter Fehler'}`, 'error');
        }
    } catch (error) {
        showNotification(`âŒ Fehler: ${error.message}`, 'error');
    }
}

// ==================== MODAL: CLOSE ====================

function closeModal() {
    const modal = document.getElementById('create-instance-modal');
    if (modal) modal.style.display = 'none';
    currentEditingInstanceName = null;
}

// Close modal on backdrop click
document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('create-instance-modal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeModal();
        });
    }
});

// ==================== LIVE STATUS UPDATES ====================

async function updateInstanceStatuses() {
    try {
        const statuses = await getServicesStatus();
        
        // Count running services per instance
        document.querySelectorAll('.instance-selector-card').forEach(card => {
            const activeBadge = card.querySelector('[id^="active-"]');
            if (!activeBadge) return;
            const instanceName = activeBadge.id.replace('active-', '');
            
            // We need instance data to know which services belong to it
            // For now show total running count
            let runningCount = 0;
            Object.values(statuses).forEach(s => {
                if (s.status === 'running') runningCount++;
            });
            activeBadge.textContent = runningCount;
            
            // Update status indicator
            const statusSpan = card.querySelector('.instance-status');
            if (statusSpan) {
                statusSpan.textContent = runningCount > 0 ? 'ðŸŸ¢' : 'âšª';
                statusSpan.title = runningCount > 0 ? `${runningCount} Services laufen` : 'Gestoppt';
            }
        });
    } catch (error) {
        console.error('Status update error:', error);
    }
}

// Poll status every 3 seconds
setInterval(updateInstanceStatuses, 3000);
updateInstanceStatuses();

