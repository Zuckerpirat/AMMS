// Theme switcher: persists choice across sessions.
(function () {
    const root = document.documentElement;

    function markActive(theme) {
        document.querySelectorAll('.swatch').forEach(function (el) {
            el.classList.toggle('is-active', el.dataset.themeSet === theme);
        });
    }

    function applyTheme(theme) {
        root.setAttribute('data-theme', theme);
        try { localStorage.setItem('amms-theme', theme); } catch (e) { /* ignore */ }
        markActive(theme);
    }

    document.querySelectorAll('[data-theme-set]').forEach(function (el) {
        el.addEventListener('click', function () {
            applyTheme(el.dataset.themeSet);
        });
    });

    markActive(root.getAttribute('data-theme') || 'claude');
})();

// Equity chart view toggle. Choice persists in localStorage and is
// re-applied after every grid refresh, so the active view survives the
// auto-refresh innerHTML swap.
function applyChartView() {
    let view = 'equity';
    try { view = localStorage.getItem('amms-chart-view') || 'equity'; } catch (e) {}
    document.querySelectorAll('[data-equity-chart]').forEach(function (chart) {
        chart.setAttribute('data-view', view);
        chart.querySelectorAll('[data-chart-view]').forEach(function (btn) {
            btn.classList.toggle('is-active', btn.dataset.chartView === view);
        });
    });
}

document.addEventListener('click', function (e) {
    const btn = e.target.closest('[data-chart-view]');
    if (!btn) return;
    try { localStorage.setItem('amms-chart-view', btn.dataset.chartView); } catch (e2) {}
    applyChartView();
});

applyChartView();

// Real-time refresh: polls /api/grid and swaps the grid contents.
// Paused while in edit mode and while the tab is hidden.
(function () {
    const grid = document.getElementById('grid');
    const indicator = document.getElementById('live-indicator');
    if (!grid) return;

    const editMode = grid.dataset.edit === '1';
    const interval = parseInt(grid.dataset.refreshMs || '5000', 10);

    if (editMode) {
        if (indicator) {
            indicator.classList.add('live--paused');
            indicator.querySelector('.live-text').textContent = 'Pausiert (Bearbeiten)';
        }
        return;
    }

    if (!interval || interval < 500) {
        if (indicator) {
            indicator.classList.add('live--paused');
            indicator.querySelector('.live-text').textContent = 'Aus';
        }
        return;
    }

    let timer = null;
    let inFlight = false;

    function setState(state) {
        if (!indicator) return;
        indicator.classList.remove('live--ok', 'live--error', 'live--syncing');
        if (state) indicator.classList.add('live--' + state);
    }

    async function refresh() {
        if (inFlight || document.hidden || window._ammsDragging) return;
        inFlight = true;
        // Only show the 'syncing' visual if the fetch is actually slow,
        // otherwise it would flicker on every fast tick.
        const slowTimer = window.setTimeout(function () { setState('syncing'); }, 300);
        try {
            const r = await fetch('/api/grid', { headers: { 'Accept': 'text/html' } });
            if (!r.ok) throw new Error('HTTP ' + r.status);
            const html = await r.text();
            grid.innerHTML = html;
            applyChartView();
            setState('ok');
        } catch (e) {
            setState('error');
            if (indicator) indicator.title = 'Fehler beim Aktualisieren: ' + e.message;
        } finally {
            window.clearTimeout(slowTimer);
            inFlight = false;
        }
    }

    function start() {
        if (timer) return;
        timer = window.setInterval(refresh, interval);
    }
    function stop() {
        if (!timer) return;
        window.clearInterval(timer);
        timer = null;
    }

    document.addEventListener('visibilitychange', function () {
        if (document.hidden) { stop(); } else { start(); refresh(); }
    });

    setState('ok');
    start();
})();

// Drag-to-reorder. Picking up a widget is initiated by mousedown on the
// corner grip; only then does the parent become draggable, so all other
// interactions inside the widget keep working normally. Polling pauses
// while a drag is in progress so the live refresh doesn't yank the DOM
// out from under the user.
(function () {
    const grid = document.getElementById('grid');
    if (!grid) return;

    let dragState = null;

    function clearDropHints() {
        grid.querySelectorAll('.is-drag-over-before, .is-drag-over-after').forEach(function (el) {
            el.classList.remove('is-drag-over-before', 'is-drag-over-after');
        });
    }

    grid.addEventListener('mousedown', function (e) {
        const handle = e.target.closest('[data-drag-handle]');
        if (!handle) return;
        const widget = handle.closest('.widget[data-widget-id]');
        if (!widget) return;
        widget.setAttribute('draggable', 'true');
    });

    grid.addEventListener('mouseup', function () {
        // If a drag never started, drop the draggable bit again.
        grid.querySelectorAll('.widget[draggable="true"]').forEach(function (w) {
            if (!dragState || dragState.element !== w) w.removeAttribute('draggable');
        });
    });

    grid.addEventListener('dragstart', function (e) {
        const widget = e.target.closest('.widget[data-widget-id]');
        if (!widget) { e.preventDefault(); return; }
        dragState = { id: widget.dataset.widgetId, element: widget };
        widget.classList.add('is-dragging');
        window._ammsDragging = true;
        e.dataTransfer.effectAllowed = 'move';
        try { e.dataTransfer.setData('text/plain', widget.dataset.widgetId); } catch (err) {}
    });

    grid.addEventListener('dragover', function (e) {
        if (!dragState) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        const target = e.target.closest('.widget[data-widget-id]');
        if (!target || target === dragState.element) return;
        const rect = target.getBoundingClientRect();
        const before = (e.clientX - rect.left) < rect.width / 2;
        clearDropHints();
        target.classList.add(before ? 'is-drag-over-before' : 'is-drag-over-after');
    });

    grid.addEventListener('dragleave', function (e) {
        const target = e.target.closest('.widget[data-widget-id]');
        if (target) target.classList.remove('is-drag-over-before', 'is-drag-over-after');
    });

    grid.addEventListener('drop', function (e) {
        if (!dragState) return;
        e.preventDefault();
        const target = e.target.closest('.widget[data-widget-id]');
        if (target && target !== dragState.element) {
            const rect = target.getBoundingClientRect();
            const before = (e.clientX - rect.left) < rect.width / 2;
            if (before) {
                target.parentNode.insertBefore(dragState.element, target);
            } else {
                target.parentNode.insertBefore(dragState.element, target.nextSibling);
            }
        }
        // Determine new index amongst all widget tiles
        const tiles = Array.from(grid.querySelectorAll('.widget[data-widget-id]'));
        const newIndex = tiles.findIndex(function (t) { return t.dataset.widgetId === dragState.id; });
        const body = new URLSearchParams({
            widget_id: dragState.id,
            new_index: String(newIndex),
        });
        fetch('/layout/reorder', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: body,
        }).catch(function () { /* next refresh will reconcile */ });
    });

    grid.addEventListener('dragend', function () {
        if (dragState) {
            dragState.element.classList.remove('is-dragging');
            dragState.element.removeAttribute('draggable');
        }
        clearDropHints();
        dragState = null;
        window._ammsDragging = false;
    });
})();
