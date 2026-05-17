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
        if (inFlight || document.hidden) return;
        inFlight = true;
        setState('syncing');
        try {
            const r = await fetch('/api/grid', { headers: { 'Accept': 'text/html' } });
            if (!r.ok) throw new Error('HTTP ' + r.status);
            const html = await r.text();
            // Swap with a brief opacity fade for smoothness.
            grid.style.transition = 'opacity .15s ease';
            grid.style.opacity = '0.6';
            window.setTimeout(function () {
                grid.innerHTML = html;
                grid.style.opacity = '1';
            }, 80);
            setState('ok');
        } catch (e) {
            setState('error');
            if (indicator) indicator.title = 'Fehler beim Aktualisieren: ' + e.message;
        } finally {
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
