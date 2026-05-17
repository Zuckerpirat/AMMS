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
