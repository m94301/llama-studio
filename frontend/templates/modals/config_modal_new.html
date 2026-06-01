<div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
     id="modal"
     onclick="if(event.target.id === 'modal') document.getElementById('modal').remove();">
    <div class="bg-gray-800 border border-gray-600 rounded-lg p-6 w-full max-w-3xl max-h-[90vh] flex flex-col"
         onclick="event.stopPropagation()">

        <div class="flex justify-between items-center mb-4">
            <h2 class="text-xl font-bold">Edit Model Configuration</h2>
            <button onclick="document.getElementById('modal').remove();"
                    class="text-gray-400 hover:text-white text-2xl">×</button>
        </div>

        <form hx-post="/api/save-model-config-new"
              hx-target="#modal-container"
              hx-swap="innerHTML"
              class="flex flex-col flex-1 gap-6 overflow-y-auto pr-2">

            <input type="hidden" name="model_name" value="{{ model_name }}">

            <!-- CORE FIELDS SECTION -->
            <fieldset>
                <!-- Display Name + Health-check Timeout -->
                <div class="mb-4 flex gap-3 items-end">
                    <div class="flex-1">
                        <label class="text-xs font-bold text-gray-400">Display Name</label>
                        <input type="text" name="display_name"
                               value="{{ display_name }}"
                               class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded text-sm text-gray-300">
                    </div>
                    <div class="w-44">
                        <label class="text-xs font-bold text-gray-400"
                               title="Per-model startup health-check timeout. Leave blank to use the default.">
                            Health timeout (s)
                        </label>
                        <input type="number" id="health-timeout-input" name="health_timeout"
                               value="{{ health_timeout if health_timeout is not none else '' }}"
                               min="1" max="1800" step="1"
                               placeholder="{{ default_health_timeout }} (default)"
                               class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded text-sm text-gray-300">
                    </div>
                </div>

                <!-- llama-server Binary Path -->
                <div class="mb-4">
                    <label class="text-xs font-bold text-gray-400">llama-server Binary</label>
                    <div class="flex gap-2 items-center">
                        <input type="text" id="llama-path-input" name="llama_path"
                               value="{{ llama_path }}"
                               placeholder="default (use global setting)"
                               class="flex-1 px-3 py-2 bg-gray-900 border border-gray-700 rounded text-sm text-gray-300"
                               onblur="validateLlamaPath()">
                        <button type="button" onclick="openConfigFileBrowser('llama_server')"
                                class="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm font-bold transition-colors">
                            Browse...
                        </button>
                        <span id="llama-version-badge" class="text-xs font-bold px-2 py-1 rounded hidden" style="min-width: 80px; text-align: center;"></span>
                    </div>
                    <p id="llama-error-text" class="text-xs text-red-400 mt-1 hidden"></p>
                </div>

                <!-- Required for Load (Host, Port, VRAM Calculator) -->
                <div class="mb-4 p-4 bg-gray-800 border border-gray-700 rounded">
                    <p class="text-xs font-bold text-gray-300 mb-4">Required for Load</p>

                    <!-- Host & Port (single line, compact) -->
                    <div class="flex items-center gap-3 mb-4">
                        <div class="flex items-center gap-1">
                            <label class="text-xs font-bold text-gray-400 whitespace-nowrap">Host</label>
                            <input type="text" id="host-input" name="host"
                                   value="{{ host }}"
                                   class="w-24 px-2 py-1 bg-gray-900 border border-gray-700 rounded text-xs text-gray-300"
                                   placeholder="0.0.0.0"
                                   oninput="updateHostInTable(this.value)"
                                   required>
                        </div>
                        <div class="flex items-center gap-1">
                            <label class="text-xs font-bold text-gray-400 whitespace-nowrap">Port</label>
                            <input type="number" id="port-input" name="port"
                                   value="{{ port }}"
                                   class="w-20 px-2 py-1 bg-gray-900 border border-gray-700 rounded text-xs text-gray-300"
                                   oninput="updatePortInTable(this.value)"
                                   required>
                        </div>
                    </div>

                    <!-- Context size slider -->
                    <div class="flex items-center gap-3 mb-4">
                        <label class="text-xs font-bold text-gray-400 w-24">Context Size</label>
                        <input type="range" id="ctx-slider"
                               min="512" max="{{ max_context or 32768 }}" step="512"
                               value="{{ current_ctx }}"
                               oninput="updateCtx(this.value)"
                               class="flex-1 h-2 bg-gray-700 rounded appearance-none cursor-pointer">
                        <span id="ctx-display" class="text-sm text-white w-20 text-right font-bold">{{ current_ctx }}</span>
                    </div>

                    <!-- KV quantization dropdowns (K and V separate) -->
                    {% set KV_QUANTS = [
                        ("f16",  "F16  — full precision"),
                        ("q8_0", "Q8_0 — 8-bit"),
                        ("q6_k", "Q6_K — 6-bit"),
                        ("q5_k", "Q5_K — 5-bit"),
                        ("q5_0", "Q5_0 — 5-bit (alt)"),
                        ("q4_k", "Q4_K — 4-bit"),
                        ("q4_0", "Q4_0 — 4-bit (alt)"),
                        ("q3_k", "Q3_K — 3-bit"),
                    ] %}
                    <div class="flex gap-6 mb-4">
                        <div class="flex items-center gap-2 flex-1">
                            <label class="text-xs font-bold text-gray-400 w-16">K Cache</label>
                            <select id="kv-quant-k" onchange="updateKvCacheType('k', this.value)" class="flex-1 bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-300">
                                {% for val, label in KV_QUANTS %}
                                <option value="{{ val }}" {% if val == current_ctk %}selected{% endif %}>{{ label }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        <div class="flex items-center gap-2 flex-1">
                            <label class="text-xs font-bold text-gray-400 w-16">V Cache</label>
                            <select id="kv-quant-v" onchange="updateKvCacheType('v', this.value)" class="flex-1 bg-gray-900 border border-gray-700 rounded px-2 py-1 text-sm text-gray-300">
                                {% for val, label in KV_QUANTS %}
                                <option value="{{ val }}" {% if val == current_ctv %}selected{% endif %}>{{ label }}</option>
                                {% endfor %}
                            </select>
                        </div>
                    </div>

                    <!-- Model metadata (single line, compact key:value) -->
                    <div class="text-xs text-gray-400 mb-4 flex gap-4">
                        <div><span class="text-gray-500">File Size:</span> <span class="text-white font-bold">{{ size_gb|round(2) ~ ' GB' if size_gb else '—' }}</span></div>
                        <div><span class="text-gray-500">Layers:</span> <span class="text-white font-bold">{{ block_count or '—' }}</span></div>
                    </div>

                    <!-- VRAM estimate bar -->
                    <div class="bg-gray-900 border border-gray-700 rounded p-3 text-sm">
                        <div class="flex items-center justify-between gap-6">
                            <div class="flex items-center gap-3">
                                <span class="text-gray-400 font-bold">VRAM preview:</span>
                                <span id="vram-total" class="text-yellow-300 font-bold">—</span>
                                <span class="text-gray-600 text-xs">(updates on save)</span>
                            </div>
                            <div class="flex items-center gap-1 text-xs">
                                <span class="text-gray-400">Weights:</span>
                                <span id="vram-weights" class="text-white font-bold">—</span>
                                <span class="text-gray-400">+ KV:</span>
                                <span id="vram-kv" class="text-white font-bold">—</span>
                            </div>
                        </div>
                    </div>
                </div>
            </fieldset>

            <!-- LAUNCH ARGS SECTION -->
            <fieldset>
                <legend class="text-sm font-bold text-gray-300 mb-4">Launch Args</legend>

                <!-- Validation errors display -->
                <div id="validation-errors" class="mb-4 hidden">
                    <div class="p-3 bg-red-900 border border-red-700 rounded text-sm text-red-200">
                        <p class="font-bold mb-2">⚠ Invalid options detected:</p>
                        <ul id="error-list" class="list-disc list-inside space-y-1"></ul>
                    </div>
                </div>

                <!-- Current launch args (fully editable table) -->
                <div id="launch-args-table" class="border border-gray-700 rounded mb-4 overflow-hidden">
                    <div class="bg-gray-900 px-4 py-2 grid gap-2 font-bold text-sm text-gray-400" style="grid-template-columns: 1fr 1fr auto;">
                        <span>Argument</span>
                        <span>Value</span>
                        <span></span>
                    </div>

                    <!-- Permanent required args (read-only, tied to master controls) -->
                    <!-- Host -->
                    <div class="px-4 py-2 border-t border-gray-700 grid gap-2 items-center bg-gray-800"
                         style="grid-template-columns: 1fr 1fr auto;">
                        <input type="text"
                               class="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-500 cursor-not-allowed"
                               value="--host"
                               disabled>
                        <input type="text" id="table-host-value"
                               class="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-500 cursor-not-allowed"
                               value="{{ host }}"
                               disabled>
                        <span class="text-gray-600 text-xs text-center">auto</span>
                    </div>

                    <!-- Port -->
                    <div class="px-4 py-2 border-t border-gray-700 grid gap-2 items-center bg-gray-800"
                         style="grid-template-columns: 1fr 1fr auto;">
                        <input type="text"
                               class="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-500 cursor-not-allowed"
                               value="--port"
                               disabled>
                        <input type="text" id="table-port-value"
                               class="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-500 cursor-not-allowed"
                               value="{{ port }}"
                               disabled>
                        <span class="text-gray-600 text-xs text-center">auto</span>
                    </div>

                    <!-- Context Size -->
                    <div class="px-4 py-2 border-t border-gray-700 grid gap-2 items-center bg-gray-800"
                         style="grid-template-columns: 1fr 1fr auto;">
                        <input type="text"
                               class="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-500 cursor-not-allowed"
                               value="--ctx-size"
                               disabled>
                        <input type="text" id="table-ctx-size-value"
                               class="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-500 cursor-not-allowed"
                               value="{{ current_ctx }}"
                               disabled>
                        <span class="text-gray-600 text-xs text-center">auto</span>
                    </div>

                    <!-- Cache Type K -->
                    <div class="px-4 py-2 border-t border-gray-700 grid gap-2 items-center bg-gray-800"
                         style="grid-template-columns: 1fr 1fr auto;">
                        <input type="text"
                               class="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-500 cursor-not-allowed"
                               value="--cache-type-k"
                               disabled>
                        <input type="text" id="table-cache-type-k-value"
                               class="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-500 cursor-not-allowed"
                               value="{{ current_ctk }}"
                               disabled>
                        <span class="text-gray-600 text-xs text-center">auto</span>
                    </div>

                    <!-- Cache Type V -->
                    <div class="px-4 py-2 border-t border-gray-700 grid gap-2 items-center bg-gray-800"
                         style="grid-template-columns: 1fr 1fr auto;">
                        <input type="text"
                               class="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-500 cursor-not-allowed"
                               value="--cache-type-v"
                               disabled>
                        <input type="text" id="table-cache-type-v-value"
                               class="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-500 cursor-not-allowed"
                               value="{{ current_ctv }}"
                               disabled>
                        <span class="text-gray-600 text-xs text-center">auto</span>
                    </div>

                    {% for key, value in advanced_options.items() %}
                    <div class="px-4 py-2 border-t border-gray-700 grid gap-2 items-center"
                         data-option-key="{{ key }}"
                         data-pasteable-row="1"
                         {% if value %}style="grid-template-columns: 1fr 1fr auto;"{% else %}style="grid-template-columns: 1fr auto;"{% endif %}>
                        <input type="text"
                               class="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-300"
                               data-arg-name
                               value="{{ key }}">
                        {% if value %}
                        <input type="text"
                               class="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-300"
                               data-arg-value
                               value="{{ value }}">
                        {% endif %}
                        <button type="button"
                                onclick="this.closest('div').remove()"
                                class="text-red-500 hover:text-red-400 text-xl leading-none w-6 h-6 flex items-center justify-center">×</button>
                    </div>
                    {% endfor %}
                </div>

                <!-- Add new row and all options buttons -->
                <div class="flex gap-2">
                    <button type="button"
                            onclick="addLaunchArg()"
                            class="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-sm font-bold">
                        + Add Argument
                    </button>
                    <button type="button"
                            onclick="showOptionBrowser()"
                            class="flex-1 px-4 py-2 bg-purple-600 hover:bg-purple-500 rounded text-sm font-bold">
                        🔍 Option Browser
                    </button>
                </div>
            </fieldset>

            <!-- Hidden field: serialized core + advanced options as JSON -->
            <textarea name="core_json"
                      id="core-json-field"
                      style="display: none;"></textarea>
            <textarea name="advanced_json"
                      id="advanced-json-field"
                      style="display: none;"></textarea>

            <!-- Script preview / actions -->
            <fieldset class="border border-gray-700 rounded p-3 mt-2">
                <legend class="text-xs font-bold text-gray-400 px-2">Launch Script</legend>
                <p class="text-xs text-gray-500 mb-2">
                    This config is stored as <code>config/models/{{ model_name }}.sh</code>. Copy or download it to share with other machines.
                </p>
                <div class="flex gap-2 mb-2">
                    <button type="button"
                            onclick="openPasteScriptModal()"
                            class="px-3 py-1 bg-purple-700 hover:bg-purple-600 rounded text-xs font-bold">
                        Edit
                    </button>
                    <button type="button"
                            onclick="copyScriptToClipboard(event)"
                            class="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs font-bold">
                        Copy to Clipboard
                    </button>
                    <button type="button"
                            onclick="loadScriptPreview()"
                            class="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs font-bold">
                        Refresh
                    </button>
                </div>
                <textarea id="script-preview"
                          readonly
                          placeholder="Loading script…"
                          class="w-full h-40 px-2 py-1 bg-gray-900 border border-gray-700 rounded text-xs font-mono text-gray-300"></textarea>
            </fieldset>

            <!-- Buttons -->
            <div class="flex gap-3 mt-6">
                <button type="button"
                        onclick="document.getElementById('modal').remove();"
                        class="flex-1 px-4 py-2 rounded bg-gray-700 hover:bg-gray-600 text-white font-bold">
                    Cancel
                </button>
                <button type="submit"
                        onclick="serializeFormData(); return true;"
                        class="flex-1 px-4 py-2 rounded bg-green-600 hover:bg-green-500 text-white font-bold">
                    Save Configuration
                </button>
            </div>
        </form>
    </div>
</div>

<!-- Paste Script nested modal -->
<div id="paste-script-modal"
     class="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-[70] hidden"
     onclick="if(event.target.id === 'paste-script-modal') document.getElementById('paste-script-modal').classList.add('hidden');">
    <div class="bg-gray-800 border border-purple-600 rounded-lg p-6 w-full max-w-2xl max-h-[85vh] flex flex-col"
         onclick="event.stopPropagation()">
        <div class="flex justify-between items-center mb-3">
            <h2 class="text-lg font-bold text-purple-300">Edit Launch Script</h2>
            <button onclick="document.getElementById('paste-script-modal').classList.add('hidden');"
                    class="text-gray-400 hover:text-white text-2xl">×</button>
        </div>
        <p class="text-xs text-gray-400 mb-3">
            Edit the script directly, or paste a llama-server script from elsewhere (e.g. Unsloth). Parse &amp; Apply will replace the form fields above with the values parsed from the script.
        </p>
        <textarea id="paste-script-input"
                  placeholder="./llama.cpp/llama-server \&#10;    --model unsloth/.../model.gguf \&#10;    --port 8001 \&#10;    ..."
                  class="w-full h-56 px-2 py-2 bg-gray-900 border border-gray-700 rounded text-xs font-mono text-gray-200"></textarea>
        <div id="paste-script-warnings" class="text-xs text-yellow-400 mt-2"></div>
        <div class="flex gap-3 mt-4">
            <button type="button"
                    onclick="document.getElementById('paste-script-modal').classList.add('hidden');"
                    class="flex-1 px-4 py-2 rounded bg-gray-700 hover:bg-gray-600 text-white font-bold">
                Cancel
            </button>
            <button type="button"
                    onclick="applyPastedScript()"
                    class="flex-1 px-4 py-2 rounded bg-purple-600 hover:bg-purple-500 text-white font-bold">
                Parse & Apply
            </button>
        </div>
    </div>
</div>

<!-- Cheat Sheet Modal (shown conditionally) -->
<div id="option-browser-modal" class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-[60] hidden"
     onclick="if(event.target.id === 'option-browser-modal') document.getElementById('option-browser-modal').classList.add('hidden');">
    <div class="bg-gray-800 border border-gray-600 rounded-lg p-6 w-full max-w-2xl max-h-[80vh] flex flex-col"
         onclick="event.stopPropagation()">

        <div class="flex justify-between items-center mb-4">
            <h2 class="text-xl font-bold">Option Browser — llama-server {{ llama_version }}</h2>
            <button onclick="document.getElementById('option-browser-modal').classList.add('hidden');"
                    class="text-gray-400 hover:text-white text-2xl">×</button>
        </div>

        <!-- Search box -->
        <div class="mb-4">
            <input type="text"
                   id="option-search-box"
                   placeholder="Search options (e.g., 'rope', 'gpu', 'thread')..."
                   class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded text-sm text-gray-300 focus:outline-none focus:border-blue-500"
                   oninput="filterOptions(this.value)">
        </div>

        <div id="option-browser-content" class="flex-1 overflow-y-auto space-y-4">
            <!-- Populated by JavaScript -->
        </div>

        <!-- Empty state message -->
        <div id="option-search-empty" class="hidden text-center text-gray-400 py-8">
            No options found. Try a different search term.
        </div>
    </div>
</div>

<script>
// Get all option schema from backend (injected in template)
window.OPTION_SCHEMA = {{ option_schema | tojson }};

function addLaunchArg() {
    const table = document.getElementById('launch-args-table');
    const row = document.createElement('div');
    row.className = 'px-4 py-2 border-t border-gray-700 grid gap-2 items-center';
    row.style.gridTemplateColumns = '1fr 1fr auto';
    row.innerHTML = `
        <input type="text"
               class="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-300"
               data-arg-name
               placeholder="e.g., -c">
        <input type="text"
               class="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-300"
               data-arg-value
               placeholder="e.g., 32768">
        <button type="button"
                onclick="this.closest('div').remove()"
                class="text-red-500 hover:text-red-400 text-xl leading-none w-6 h-6 flex items-center justify-center">×</button>
    `;
    table.appendChild(row);
}

function showOptionBrowser() {
    const modal = document.getElementById('option-browser-modal');
    const content = document.getElementById('option-browser-content');
    const searchBox = document.getElementById('option-search-box');
    const emptyState = document.getElementById('option-search-empty');

    // Clear search box and empty state
    searchBox.value = '';
    emptyState.classList.add('hidden');

    // Group by category
    const byCategory = {};
    for (const [key, spec] of Object.entries(window.OPTION_SCHEMA)) {
        const cat = spec.category || 'other';
        if (!byCategory[cat]) byCategory[cat] = [];
        byCategory[cat].push({ key, ...spec });
    }

    // Define display order: gpu, network, context, threading, then rest
    const categoryOrder = ['gpu', 'networking', 'context', 'threading'];
    const orderedCategories = [];

    // Add categories in specified order
    for (const cat of categoryOrder) {
        if (byCategory[cat]) {
            orderedCategories.push([cat, byCategory[cat]]);
            delete byCategory[cat];
        }
    }

    // Add remaining categories in alphabetical order
    for (const [cat, options] of Object.entries(byCategory).sort()) {
        orderedCategories.push([cat, options]);
    }

    content.innerHTML = '';
    for (const [category, options] of orderedCategories) {
        const catDiv = document.createElement('div');
        catDiv.setAttribute('data-category', category);
        const displayName = category === 'networking' ? 'Network' : category.charAt(0).toUpperCase() + category.slice(1);
        catDiv.innerHTML = `<h3 class="text-sm font-bold text-gray-300 mb-2">${displayName}</h3>`;

        const optList = document.createElement('div');
        optList.className = 'space-y-1';

        for (const opt of options) {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.setAttribute('data-option-key', opt.key);
            btn.className = 'w-full text-left px-3 py-2 rounded bg-gray-700 hover:bg-gray-600 text-sm text-gray-300 transition-colors';
            btn.innerHTML = `<span class="font-mono font-bold">${opt.key}</span> — ${opt.description}`;
            btn.onclick = () => addOptionFromBrowser(opt.key);
            optList.appendChild(btn);
        }

        catDiv.appendChild(optList);
        content.appendChild(catDiv);
    }

    modal.classList.remove('hidden');
}

function addOptionFromBrowser(key) {
    const table = document.getElementById('launch-args-table');
    const row = document.createElement('div');
    row.className = 'px-4 py-2 border-t border-gray-700 grid gap-2 items-center';
    row.style.gridTemplateColumns = '1fr 1fr auto';
    row.innerHTML = `
        <input type="text"
               class="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-300"
               data-arg-name
               value="${key}"
               readonly>
        <input type="text"
               class="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-300"
               data-arg-value
               placeholder="Enter value..."
               autofocus>
        <button type="button"
                onclick="this.closest('div').remove()"
                class="text-red-500 hover:text-red-400 text-xl leading-none w-6 h-6 flex items-center justify-center">×</button>
    `;
    table.appendChild(row);
    document.getElementById('option-browser-modal').classList.add('hidden');

    // Focus the value input
    row.querySelector('[data-arg-value]').focus();
}

function filterOptions(query) {
    const normalized = query.toLowerCase().trim();
    const categories = document.querySelectorAll('[data-category]');
    const emptyState = document.getElementById('option-search-empty');
    let hasVisibleOptions = false;

    for (const cat of categories) {
        const buttons = cat.querySelectorAll('[data-option-key]');
        let visibleInCategory = 0;

        for (const btn of buttons) {
            const key = btn.getAttribute('data-option-key').toLowerCase();
            const desc = btn.textContent.toLowerCase();
            const matches = !normalized || key.includes(normalized) || desc.includes(normalized);

            btn.style.display = matches ? 'block' : 'none';
            if (matches) visibleInCategory++;
        }

        // Hide empty categories
        cat.style.display = visibleInCategory > 0 ? 'block' : 'none';
        if (visibleInCategory > 0) hasVisibleOptions = true;
    }

    // Show empty state message
    emptyState.classList.toggle('hidden', hasVisibleOptions);
}

function serializeFormData() {
    // Core fields
    const htRaw = document.getElementById('health-timeout-input').value.trim();
    const core = {
        port: parseInt(document.querySelector('input[name="port"]').value) || null,
        display_name: document.querySelector('input[name="display_name"]').value.trim() || null,
        host: document.querySelector('input[name="host"]').value.trim() || "0.0.0.0",
        llama_path: document.getElementById('llama-path-input').value.trim() || "default",
        // Empty → use default (server treats null/empty as "omit the meta line")
        health_timeout: htRaw === '' ? null : parseInt(htRaw),
    };

    // Launch args: collect all rows (permanent + user-added)
    const advanced = {};

    // Collect permanent required args from table
    advanced['--host'] = document.getElementById('table-host-value').value.trim();
    advanced['--port'] = document.getElementById('table-port-value').value.trim();
    advanced['--ctx-size'] = document.getElementById('table-ctx-size-value').value.trim();
    advanced['--cache-type-k'] = document.getElementById('table-cache-type-k-value').value.trim();
    advanced['--cache-type-v'] = document.getElementById('table-cache-type-v-value').value.trim();

    // Collect user-added rows from launch-args-table (those with inputs having data-arg-name attribute)
    document.querySelectorAll('#launch-args-table input[data-arg-name]').forEach(input => {
        const row = input.closest('div');
        const argName = row.querySelector('[data-arg-name]')?.value.trim();
        const argValue = row.querySelector('[data-arg-value]')?.value.trim();
        // Allow arguments with or without values (e.g., --no-mmap has no value)
        if (argName) {
            advanced[argName] = argValue || null;
        }
    });

    document.getElementById('core-json-field').value = JSON.stringify(core);
    document.getElementById('advanced-json-field').value = JSON.stringify(advanced);

    // === NEW: Validate fields and highlight invalid ones ===
    validateAndHighlightFields(advanced);
}

function validateAndHighlightFields(advanced) {
    const errorList = document.getElementById('error-list');
    const errorsDiv = document.getElementById('validation-errors');
    const errors = [];

    // Only validate user-added rows (those with data-arg-name inputs), skip permanent rows
    document.querySelectorAll('#launch-args-table input[data-arg-name]').forEach(input => {
        const row = input.closest('div');
        const argName = row.querySelector('[data-arg-name]')?.value.trim();
        const argValue = row.querySelector('[data-arg-value]')?.value.trim();

        if (!argName || !argValue) return;

        // Check if option is valid
        if (!(argName in window.OPTION_SCHEMA)) {
            errors.push(argName);
            row.style.backgroundColor = '#7f1d1d';  // red-900
        } else {
            row.style.backgroundColor = '';  // reset
        }
    });

    if (errors.length > 0) {
        errorList.innerHTML = errors.map(e => `<li>${e}</li>`).join('');
        errorsDiv.classList.remove('hidden');
    } else {
        errorsDiv.classList.add('hidden');
    }
}

// ============================================================================
// VRAM Calculator
// ============================================================================

window.KV_BYTES = {
    'f16': 2.0,
    'q8_0': 1.0,
    'q6_k': 0.75,
    'q5_k': 0.625,
    'q5_0': 0.625,
    'q4_k': 0.5,
    'q4_0': 0.5,
    'q3_k': 0.375
};

// Always overwrite — stale values from a previously-opened modal cause cross-model
// VRAM miscalcs (the new modal's slider/dropdowns are fresh, but block_count etc.
// would silently inherit from the prior model).
window.VRAM_CONFIG = {
    kv_cache_multiplier: {{ kv_cache_multiplier or 0 }},
    block_count: {{ block_count or 0 }},
    size_gb: {{ size_gb or 0 }}
};

async function updateVramEstimate() {
    const kvMultiplier = VRAM_CONFIG.kv_cache_multiplier;
    const blockCount = VRAM_CONFIG.block_count;
    const sizeGb = VRAM_CONFIG.size_gb;

    if (kvMultiplier === 0 || blockCount === 0) {
        document.getElementById('vram-total').textContent = '—';
        document.getElementById('vram-weights').textContent = '—';
        document.getElementById('vram-kv').textContent = '—';
        return;
    }

    const ctx = parseInt(document.getElementById('ctx-slider').value || 0);
    const kvK = document.getElementById('kv-quant-k').value;
    const kvV = document.getElementById('kv-quant-v').value;

    try {
        // Call backend API for calculation (single source of truth)
        const response = await fetch('/api/calculate-vram', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({
                block_count: blockCount,
                kv_cache_multiplier: kvMultiplier,
                size_gb: sizeGb,
                ctx_size: ctx,
                cache_type_k: kvK,
                cache_type_v: kvV
            })
        });

        const result = await response.json();

        if (result.error) {
            document.getElementById('vram-total').textContent = '—';
            document.getElementById('vram-weights').textContent = '—';
            document.getElementById('vram-kv').textContent = '—';
            return;
        }

        document.getElementById('vram-weights').textContent = result.weights + ' GB';
        document.getElementById('vram-kv').textContent = result.kv_cache + ' GB';
        document.getElementById('vram-total').textContent = result.total + ' GB';
    } catch (error) {
        console.error('Error calculating VRAM:', error);
        document.getElementById('vram-total').textContent = '—';
        document.getElementById('vram-weights').textContent = '—';
        document.getElementById('vram-kv').textContent = '—';
    }
}

window.updateCtx = function(value) {
    document.getElementById('ctx-display').textContent = value;
    document.getElementById('table-ctx-size-value').value = value;
    updateVramEstimate();
};

window.updateHostInTable = function(value) {
    document.getElementById('table-host-value').value = value;
};

window.updatePortInTable = function(value) {
    document.getElementById('table-port-value').value = value;
};

window.updateKvCacheType = function(type, value) {
    const fieldId = type === 'k' ? 'table-cache-type-k-value' : 'table-cache-type-v-value';
    document.getElementById(fieldId).value = value;
    updateVramEstimate();
};

// ============================================================================
// llama-server Binary Path Management
// ============================================================================

window.validateLlamaPath = async function() {
    const pathInput = document.getElementById('llama-path-input');
    const badge = document.getElementById('llama-version-badge');
    const errorText = document.getElementById('llama-error-text');
    const path = pathInput.value.trim();

    // Clear previous states
    badge.classList.add('hidden');
    errorText.classList.add('hidden');

    // Empty or "default" is valid
    if (!path || path === 'default') {
        pathInput.value = 'default';
        return;
    }

    // Validate the path
    try {
        const response = await fetch('/api/validate-binary-path', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({ path: path })
        });
        const data = await response.json();

        if (data.error) {
            errorText.textContent = data.error;
            errorText.classList.remove('hidden');
            pathInput.classList.add('border-red-600');
            pathInput.classList.remove('border-gray-700');
        } else if (data.version) {
            // Update input with resolved path (may have /llama-server appended)
            if (data.resolved_path) {
                pathInput.value = data.resolved_path;
            }
            badge.textContent = `✓ v${data.version}`;
            badge.classList.add('bg-green-600', 'text-white');
            badge.classList.remove('bg-red-600', 'hidden');
            pathInput.classList.remove('border-red-600');
            pathInput.classList.add('border-gray-700');
        }
    } catch (error) {
        errorText.textContent = 'Error validating binary: ' + error.message;
        errorText.classList.remove('hidden');
        pathInput.classList.add('border-red-600');
        pathInput.classList.remove('border-gray-700');
    }
};

window.openConfigFileBrowser = function(pathType) {
    const currentPath = document.getElementById('llama-path-input').value || '';
    const url = `/api/file-browser?path_type=${pathType}&current_path=${encodeURIComponent(currentPath)}&modal=settings`;

    htmx.ajax('GET', url, {
        target: '#file-browser-content',
        swap: 'innerHTML'
    });
    document.getElementById('file-browser-modal').classList.remove('hidden');
};

window.selectPath = function(path) {
    document.getElementById('llama-path-input').value = path;
    document.getElementById('file-browser-modal').classList.add('hidden');
    validateLlamaPath();
};

// Run calculation on modal load
updateVramEstimate();
// Auto-load the script preview so the user doesn't have to click Refresh
loadScriptPreview();

// ---------- Script preview / download / copy ----------

async function loadScriptPreview() {
    try {
        const resp = await fetch('/api/model-script-raw?model={{ model_name }}');
        const text = await resp.text();
        document.getElementById('script-preview').value = text;
    } catch (e) {
        document.getElementById('script-preview').value = '# error loading script: ' + e;
    }
}

async function copyScriptToClipboard(ev) {
    const btn = ev && ev.currentTarget;
    const preview = document.getElementById('script-preview');
    if (!preview.value) await loadScriptPreview();
    try {
        await navigator.clipboard.writeText(preview.value);
        if (btn) {
            const orig = btn.textContent;
            btn.textContent = 'Copied!';
            setTimeout(() => { btn.textContent = orig; }, 1500);
        }
    } catch (e) {
        alert('Copy failed: ' + e);
    }
}

// ---------- Paste Script ----------

async function openPasteScriptModal() {
    const input = document.getElementById('paste-script-input');
    document.getElementById('paste-script-warnings').innerHTML = '';
    // Pre-populate with the current script so the user can edit in place.
    // Fall back to the preview textarea if the live fetch fails.
    try {
        const resp = await fetch('/api/model-script-raw?model={{ model_name }}');
        input.value = resp.ok ? await resp.text() : (document.getElementById('script-preview').value || '');
    } catch (e) {
        input.value = document.getElementById('script-preview').value || '';
    }
    document.getElementById('paste-script-modal').classList.remove('hidden');
}

async function applyPastedScript() {
    const text = document.getElementById('paste-script-input').value;
    if (!text.trim()) {
        alert('Paste a script first.');
        return;
    }
    const warningsBox = document.getElementById('paste-script-warnings');
    warningsBox.innerHTML = '<span class="text-gray-400">Parsing…</span>';

    const form = new FormData();
    form.append('script_text', text);
    let result;
    try {
        const resp = await fetch('/api/parse-script', { method: 'POST', body: form });
        result = await resp.json();
    } catch (e) {
        warningsBox.innerHTML = `<span class="text-red-400">✗ Request failed: ${e}</span>`;
        return;
    }
    if (result.error) {
        warningsBox.innerHTML = `<span class="text-red-400">✗ ${result.error}</span>`;
        return;
    }

    // Populate fields (overwrite always — resolved decision #1 applies to display_name; we apply same rule to other fields when present in paste)
    if (result.display_name) {
        document.querySelector('input[name="display_name"]').value = result.display_name;
    }
    // Health timeout from meta fence (null = leave blank; user keeps their previous value)
    if (result.health_timeout !== null && result.health_timeout !== undefined) {
        document.getElementById('health-timeout-input').value = result.health_timeout;
    }
    if (result.port) {
        document.getElementById('port-input').value = result.port;
        const tp = document.getElementById('table-port-value');
        if (tp) tp.value = result.port;
    }
    if (result.host) {
        document.getElementById('host-input').value = result.host;
        const th = document.getElementById('table-host-value');
        if (th) th.value = result.host;
    }
    if (result.ctx_size) {
        const slider = document.getElementById('ctx-slider');
        if (slider) { slider.value = result.ctx_size; if (typeof updateCtx === 'function') updateCtx(result.ctx_size); }
        const tc = document.getElementById('table-ctx-size-value');
        if (tc) tc.value = result.ctx_size;
    }
    if (result.cache_type_k) {
        const tck = document.getElementById('table-cache-type-k-value');
        if (tck) tck.value = result.cache_type_k;
    }
    if (result.cache_type_v) {
        const tcv = document.getElementById('table-cache-type-v-value');
        if (tcv) tcv.value = result.cache_type_v;
    }
    if (result.llama_path) {
        document.getElementById('llama-path-input').value = result.llama_path;
        if (typeof validateLlamaPath === 'function') validateLlamaPath();
    }

    // Clear existing user-added rows and append the parsed 'advanced' dict
    const table = document.getElementById('launch-args-table');
    table.querySelectorAll('[data-option-key]').forEach(row => row.remove());
    for (const [key, value] of Object.entries(result.advanced || {})) {
        addAdvancedRow(key, value);
    }

    // Render any warnings
    const warnings = result.warnings || [];
    if (warnings.length === 0) {
        warningsBox.innerHTML = '<span class="text-green-400">✓ Applied to form.</span>';
    } else {
        warningsBox.innerHTML = '<p class="text-yellow-400 font-bold mb-1">✓ Applied with warnings:</p><ul class="list-disc ml-4">' +
            warnings.map(w => `<li>${w}</li>`).join('') + '</ul>';
    }
    setTimeout(() => {
        document.getElementById('paste-script-modal').classList.add('hidden');
        if (typeof updateVramEstimate === 'function') updateVramEstimate();
    }, warnings.length === 0 ? 800 : 2500);
}

// Helper to append an advanced-args row (used by paste-script flow)
function addAdvancedRow(key, value) {
    const table = document.getElementById('launch-args-table');
    const row = document.createElement('div');
    row.className = 'px-4 py-2 border-t border-gray-700 grid gap-2 items-center';
    row.setAttribute('data-option-key', key);
    if (value !== null && value !== undefined && value !== '') {
        row.style.gridTemplateColumns = '1fr 1fr auto';
        row.innerHTML = `
            <input type="text" data-arg-name value="${key}"
                   class="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-300">
            <input type="text" data-arg-value value="${String(value).replace(/"/g, '&quot;')}"
                   class="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-300">
            <button type="button" onclick="this.closest('div').remove()"
                    class="text-red-500 hover:text-red-400 text-xl leading-none w-6 h-6 flex items-center justify-center">×</button>
        `;
    } else {
        row.style.gridTemplateColumns = '1fr auto';
        row.innerHTML = `
            <input type="text" data-arg-name value="${key}"
                   class="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-gray-300">
            <button type="button" onclick="this.closest('div').remove()"
                    class="text-red-500 hover:text-red-400 text-xl leading-none w-6 h-6 flex items-center justify-center">×</button>
        `;
    }
    table.appendChild(row);
}
</script>
