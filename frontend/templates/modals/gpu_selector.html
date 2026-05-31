<!-- frontend/templates/modals/gpu_selector.html -->
{% extends "base_modal.html" %}
{% set title = "Load to which GPU?" %}
{% set width = "md" %}

{% block content %}
<p class="text-sm text-gray-300 mb-6">
    Loading: <span class="font-bold text-blue-400">{{ display_name }}</span>
</p>
<div class="space-y-2 mb-6">
    {% for gpu_id, gpu in gpu_data.items() | sort %}
        {% set available = gpu.memory - gpu.allocated %}
        {% set size_needed = model_config.total_vram or model_config.size_gb or 0 %}
        {% set can_fit = available >= size_needed and size_needed > 0 %}
        {% set btn_class = "bg-green-600 hover:bg-green-500 cursor-pointer" if can_fit else "bg-red-600 cursor-not-allowed opacity-50" %}
        <button hx-post="/api/trigger-load"
                hx-vals='{"model_name": "{{ model_name }}", "gpu_ids": [{{ gpu_id }}]}'
                hx-target="#modal-container" hx-swap="innerHTML"
                {% if not can_fit %}disabled{% endif %}
                class="w-full p-3 rounded {{ btn_class }} text-white font-bold transition-colors text-left">
            <div class="flex justify-between items-center">
                <span>GPU {{ gpu_id }}: {{ gpu.name }}</span>
                <span class="text-xs">{{ "%.1f"|format(available) }}/{{ "%.1f"|format(gpu.memory) }} GB</span>
            </div>
            <div class="text-xs text-gray-200 mt-1">
                {{ "%.0f"|format(gpu.memory_pct) }}% used &bull; {{ "%.1f"|format(size_needed) }} GB needed
                {{ '✓' if can_fit else '✗ Not enough VRAM' }}
            </div>
        </button>
    {% endfor %}
</div>
<div class="flex gap-3 mt-6">
    <button onclick="document.getElementById('modal').remove();"
            class="flex-1 px-4 py-2 rounded bg-gray-700 hover:bg-gray-600 text-white font-bold transition-colors">
        Cancel
    </button>
</div>
{% endblock %}
