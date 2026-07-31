/*
 * GISclaw — an LLM agent for geospatial analysis.
 * Copyright (C) 2026 Han Jinzhen
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 *
 * This file is part of GISclaw. GISclaw is free software: you can redistribute
 * it and/or modify it under the terms of the GNU Affero General Public License
 * as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version. It is distributed in the hope
 * that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Affero General Public License in the LICENSE file, or
 * <https://www.gnu.org/licenses/>, for more details.
 */

/* ==========================================================================
   GISclaw product frontend.
   Talks to the FastAPI backend: projects, server-side file browser, and a live
   SSE run stream. Map + result images are real files; the trace is the real
   agent's Thought/Action/Observation as it happens.
   ========================================================================== */
(() => {
  'use strict';
  console.log('%cGISclaw app.js v12 loaded (records in tree)', 'color:#2b8a3e;font-weight:bold');
  const $  = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const esc = s => (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  // Inline Lucide (ISC) line-icons — monochrome, inherit currentColor, CSP-safe.
  const LUCIDE = {
    wrench: '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2 2 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
    globe: '<circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>',
    box: '<path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/>',
    layers: '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="M2 12l8.58 3.91a2 2 0 0 0 1.66 0L22 12"/><path d="M2 17l8.58 3.91a2 2 0 0 0 1.66 0L22 17"/>',
    bars: '<line x1="12" x2="12" y1="20" y2="10"/><line x1="18" x2="18" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/>',
    grid: '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v18"/><path d="M15 3v18"/>',
    target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none"/>',
    palette: '<circle cx="13.5" cy="6.5" r="1.2"/><circle cx="17" cy="10.5" r="1.2"/><circle cx="8.5" cy="7.5" r="1.2"/><circle cx="6.5" cy="12.5" r="1.2"/><path d="M12 2A10 10 0 0 0 2 12a10 10 0 0 0 10 10 2.4 2.4 0 0 0 2.4-2.4c0-.62-.24-1.18-.63-1.6a2.4 2.4 0 0 1 1.75-4.06H18a4 4 0 0 0 4-4A9.9 9.9 0 0 0 12 2Z"/>',
    table: '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M12 3v18"/>',
    trash: '<path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
    eye: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>',
    fit: '<path d="M3 9V3h6M21 9V3h-6M3 15v6h6M21 15v6h-6"/>',
  };
  const svgIcon = (name, cls = 'ic') =>
    `<svg class="${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${LUCIDE[name] || ''}</svg>`;
  const CAT_ICON = { crs: 'globe', geometry: 'box', overlay: 'layers', analysis: 'bars', raster: 'grid' };

  const GEO_EXT = ['geojson', 'json', 'shp', 'gpkg', 'gml', 'kml'];
  const IMG_EXT = ['png', 'jpg', 'jpeg'];
  const extOf = p => (p.toLowerCase().split('.').pop() || '');

  async function jget(url) { const r = await fetch(url); return r.json(); }
  async function jpost(url, body) {
    const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    return r.json();
  }

  const state = {
    project: null,          // {id, name}
    tree: null,             // {data:[], outputs:[], runs:[]}
    models: [],
    running: false, runId: 0, codeStarted: false,
    startedAt: 0, steps: 0, selfCorr: 0, timer: null,
  };

  // ==================================================================
  // Map (Leaflet)
  // ==================================================================
  const map = L.map('mapLeaflet', { zoomControl: false, attributionControl: true }).setView([20, 0], 2);
  // Leaflet 1.9's default attribution prefix embeds a Ukrainian flag SVG — replace it with a plain link.
  map.attributionControl.setPrefix('<a href="https://leafletjs.com" title="A JavaScript library for interactive maps">Leaflet</a>');
  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OSM &copy; CARTO', subdomains: 'abcd', maxZoom: 19,
  }).addTo(map);

  const LAYER_COLORS = ['#2f6f9f', '#B14B26', '#3f7d58', '#8a5fb0', '#b0892f', '#4a7a8c'];
  const shownLayers = {};   // name -> { layer, color }
  let colorIdx = 0;

  function buildGeoLayer(geojson, color) {
    return L.geoJSON(geojson, {
      style: () => ({ color, weight: 1.3, fillColor: color, fillOpacity: 0.22 }),
      pointToLayer: (f, latlng) => L.circleMarker(latlng, { radius: 5, color, weight: 1.5, fillColor: color, fillOpacity: 0.8 }),
      onEachFeature: (f, lyr) => {
        const p = f.properties || {};
        const rows = Object.entries(p).slice(0, 8)
          .map(([k, v]) => `<div><b>${esc(String(k))}</b>: ${esc(String(v))}</div>`).join('');
        if (rows) lyr.bindPopup(`<div class="gp-popup">${rows}</div>`, { maxWidth: 260 });
        lyr.on('mouseover', () => { if (lyr.setStyle) lyr.setStyle({ weight: 2.6 }); });
      },
    });
  }

  async function showFileOnMap(path, where) {
    const name = path.split('/').pop();
    if (shownLayers[name]) { map.removeLayer(shownLayers[name].layer); delete shownLayers[name]; renderLegend(); return; }
    const url = `/api/projects/${state.project.id}/file?where=${where}&path=${encodeURIComponent(path)}`;
    let gj;
    try { gj = await jget(url); } catch (e) { addMsg({ kind: 'error', text: `Could not load ${name}` }); return; }
    if (gj.error) { addMsg({ kind: 'error', text: `${name}: ${gj.error}` }); return; }
    if (gj._notice) addMsg({ kind: 'system', text: `${name}: ${gj._notice}` });
    const color = LAYER_COLORS[colorIdx++ % LAYER_COLORS.length];
    const layer = buildGeoLayer(gj, color);
    layer.addTo(map);
    shownLayers[name] = { layer, color, gj, kind: 'vector', fillOpacity: 0.22, visible: true };
    try { map.fitBounds(layer.getBounds(), { padding: [24, 24] }); } catch (e) {}
    renderLegend();
    switchTab('map');
  }

  async function addRasterOverlay(name, overlayUrl) {
    if (shownLayers[name]) { map.removeLayer(shownLayers[name].layer); delete shownLayers[name]; }
    let pl;
    try { pl = await jget(overlayUrl); } catch (e) { return; }
    if (!pl || pl.error || !pl.bounds) { addMsg({ kind: 'error', text: `${name}: ${pl && pl.error || 'overlay failed'}` }); return; }
    const b = pl.bounds;
    const layer = L.imageOverlay(pl.image, [[b.south, b.west], [b.north, b.east]], { opacity: 0.85 });
    layer.addTo(map);
    shownLayers[name] = { layer, color: '#3f7d58', isRaster: true, kind: 'raster', opacity: 0.85, visible: true };
    try { map.fitBounds(layer.getBounds(), { padding: [24, 24] }); } catch (e) {}
    renderLegend();
    switchTab('map');
  }

  function addResultGeoToMap(url, name) {
    jget(url).then(gj => {
      if (!gj || gj.error) return;
      const color = LAYER_COLORS[colorIdx++ % LAYER_COLORS.length];
      const layer = buildGeoLayer(gj, color);
      layer.addTo(map);
      shownLayers[name] = { layer, color, gj, kind: 'vector', fillOpacity: 0.22, visible: true };
      try { map.fitBounds(layer.getBounds(), { padding: [24, 24] }); } catch (e) {}
      renderLegend();
    }).catch(() => {});
  }

  function clearMap() {
    Object.values(shownLayers).forEach(o => { if (map.hasLayer(o.layer)) map.removeLayer(o.layer); });
    Object.keys(shownLayers).forEach(k => delete shownLayers[k]);
    colorIdx = 0;
    renderLegend();
  }

  function renderLegend() {
    const host = $('#legendLayers');
    host.innerHTML = '';
    const base = document.createElement('div');
    base.className = 'legend-layer';
    base.innerHTML = `<div class="legend-layer-head"><span class="legend-layer-name">Basemap</span><span class="legend-layer-meta">OSM</span></div>`;
    host.appendChild(base);

    const entries = Object.entries(shownLayers);
    entries.forEach(([name, o]) => {
      const kind = o.kind === 'raster' ? 'raster' : 'vector';
      const div = document.createElement('div');
      div.className = 'legend-layer interactive' + (o.visible === false ? ' hidden-layer' : '');
      div.innerHTML =
        `<div class="legend-layer-head">`
        + `<span class="lyr-vis" title="Toggle visibility">${svgIcon('eye', 'ctx-ic')}</span>`
        + `<span class="legend-layer-name">${esc(name)}</span>`
        + `<span class="legend-layer-meta">${kind}</span></div>`
        + `<div class="legend-chips"><span class="chip"><i style="background:${o.color}"></i>${kind}</span></div>`;
      div.querySelector('.lyr-vis').addEventListener('click', ev => { ev.stopPropagation(); toggleLayerVisibility(name); });
      div.addEventListener('contextmenu', ev => { ev.preventDefault(); ev.stopPropagation(); openLayerMenu(name, ev.clientX, ev.clientY); });
      host.appendChild(div);
    });
    if (entries.length) {
      const hint = document.createElement('div');
      hint.className = 'legend-hint-rc';
      hint.textContent = 'Right-click a layer for symbology, attribute table…';
      host.appendChild(hint);
    }
  }

  // ---- Layer operations (desktop-GIS style) --------------------------------
  function toggleLayerVisibility(name) {
    const o = shownLayers[name]; if (!o) return;
    o.visible = !(o.visible !== false);
    if (o.visible) o.layer.addTo(map); else map.removeLayer(o.layer);
    renderLegend();
  }
  function zoomToLayer(o) { try { map.fitBounds(o.layer.getBounds(), { padding: [24, 24] }); } catch (e) {} }
  function removeLayer(name) {
    const o = shownLayers[name]; if (!o) return;
    if (map.hasLayer(o.layer)) map.removeLayer(o.layer);
    delete shownLayers[name];
    renderLegend();
  }
  function restyleVector(o) {
    try { o.layer.setStyle({ color: o.color, fillColor: o.color, fillOpacity: o.fillOpacity ?? 0.22, weight: 1.3 }); } catch (e) {}
  }

  function openLayerMenu(name, x, y) {
    const o = shownLayers[name]; if (!o) return;
    showContextMenu(x, y, [
      { label: o.visible === false ? 'Show layer' : 'Hide layer', icon: 'eye', action: () => toggleLayerVisibility(name) },
      { label: 'Zoom to layer', icon: 'target', action: () => zoomToLayer(o) },
      // "Fit to data" lives here rather than in a View menu — it is a layer
      // action, and you are already pointing at the layer panel.
      { label: 'Fit to all layers', icon: 'fit', action: () => $('#btnFit').click() },
      { label: 'Symbology…', icon: 'palette', action: () => openSymbology(name, x, y) },
      { label: 'Attribute table', icon: 'table', disabled: o.kind !== 'vector' || !o.gj, action: () => openAttributeTable(name) },
      { sep: true },
      { label: 'Remove layer', icon: 'trash', danger: true, action: () => removeLayer(name) },
    ]);
  }

  // ---- Floating menus / popovers -------------------------------------------
  function closeFloaters() { ['ctxMenu', 'symbPop'].forEach(id => { const el = document.getElementById(id); if (el) el.remove(); }); }
  function positionFloater(el, x, y) {
    const r = el.getBoundingClientRect();
    el.style.left = Math.max(6, Math.min(x, window.innerWidth - r.width - 8)) + 'px';
    el.style.top = Math.max(6, Math.min(y, window.innerHeight - r.height - 8)) + 'px';
  }
  function showContextMenu(x, y, items) {
    closeFloaters();
    const menu = document.createElement('div');
    menu.className = 'ctx-menu'; menu.id = 'ctxMenu';
    items.forEach(it => {
      if (it.sep) { const s = document.createElement('div'); s.className = 'ctx-sep'; menu.appendChild(s); return; }
      const el = document.createElement('div');
      el.className = 'ctx-item' + (it.disabled ? ' disabled' : '') + (it.danger ? ' danger' : '');
      el.innerHTML = (it.icon ? svgIcon(it.icon, 'ctx-ic') : '') + `<span>${esc(it.label)}</span>`;
      if (!it.disabled) el.addEventListener('click', e => { e.stopPropagation(); closeFloaters(); it.action(); });
      menu.appendChild(el);
    });
    menu.addEventListener('click', e => e.stopPropagation());
    document.body.appendChild(menu);
    positionFloater(menu, x, y);
  }
  document.addEventListener('click', closeFloaters);

  function openSymbology(name, x, y) {
    const o = shownLayers[name]; if (!o) return;
    closeFloaters();
    const pop = document.createElement('div');
    pop.className = 'symb-pop'; pop.id = 'symbPop';
    if (o.kind === 'raster') {
      pop.innerHTML = `<div class="symb-h">${esc(name)}</div>`
        + `<div class="symb-row"><span>Opacity</span><input type="range" min="0" max="100" value="${Math.round((o.opacity ?? 0.85) * 100)}" id="symbOpacity"></div>`;
    } else {
      pop.innerHTML = `<div class="symb-h">${esc(name)}</div>`
        + `<div class="symb-row"><span>Color</span><input type="color" value="${o.color}" id="symbColor"></div>`
        + `<div class="symb-row"><span>Fill opacity</span><input type="range" min="0" max="100" value="${Math.round((o.fillOpacity ?? 0.22) * 100)}" id="symbFill"></div>`;
    }
    pop.addEventListener('click', e => e.stopPropagation());
    document.body.appendChild(pop);
    positionFloater(pop, x, y);
    if (o.kind === 'raster') {
      $('#symbOpacity').addEventListener('input', e => { o.opacity = e.target.value / 100; if (o.layer.setOpacity) o.layer.setOpacity(o.opacity); });
    } else {
      $('#symbColor').addEventListener('input', e => { o.color = e.target.value; restyleVector(o); renderLegend(); });
      $('#symbFill').addEventListener('input', e => { o.fillOpacity = e.target.value / 100; restyleVector(o); });
    }
  }

  // ---- Attribute table ------------------------------------------------------
  let attrRows = [], attrCols = [];
  function openAttributeTable(name) {
    const o = shownLayers[name]; if (!o || !o.gj) return;
    const feats = o.gj.features || [];
    const CAP = 2000;
    attrCols = [...new Set(feats.slice(0, 200).flatMap(f => Object.keys(f.properties || {})))];
    attrRows = feats.slice(0, CAP).map(f => f.properties || {});
    $('#attrTitle').textContent = `${name} — ${feats.length} feature${feats.length === 1 ? '' : 's'}`
      + (feats.length > CAP ? ` (showing first ${CAP})` : '');
    $('#attrFilter').value = '';
    renderAttrTable('');
    $('#attrModal').classList.remove('hidden');
    $('#attrFilter').focus();
  }
  function renderAttrTable(q) {
    q = (q || '').toLowerCase();
    const rows = q ? attrRows.filter(r => Object.values(r).some(v => String(v).toLowerCase().includes(q))) : attrRows;
    const thead = `<thead><tr><th class="attr-idx">#</th>${attrCols.map(c => `<th>${esc(c)}</th>`).join('')}</tr></thead>`;
    const tbody = `<tbody>${rows.map((r, i) =>
      `<tr><td class="attr-idx">${i + 1}</td>${attrCols.map(c => {
        const v = esc(String(r[c] ?? ''));
        return `<td title="${v}">${v}</td>`;
      }).join('')}</tr>`).join('')}</tbody>`;
    $('#attrTable').innerHTML = thead + tbody;
    $('#attrCount').textContent = `${rows.length} row${rows.length === 1 ? '' : 's'}`;
  }
  $('#attrClose').addEventListener('click', () => $('#attrModal').classList.add('hidden'));
  $('#attrModal').addEventListener('click', e => { if (e.target === $('#attrModal')) $('#attrModal').classList.add('hidden'); });
  $('#attrFilter').addEventListener('input', e => renderAttrTable(e.target.value));

  $('#btnZoomIn').addEventListener('click', () => map.zoomIn());
  $('#btnZoomOut').addEventListener('click', () => map.zoomOut());
  $('#btnFit').addEventListener('click', () => {
    const layers = Object.values(shownLayers);
    if (!layers.length) return;
    const group = L.featureGroup(layers.map(o => o.layer));
    try { map.fitBounds(group.getBounds(), { padding: [24, 24] }); } catch (e) {}
  });

  // ==================================================================
  // Menu bar (top) — routes to the same handlers as the old buttons
  // ==================================================================
  const MENU_TRIG = { new: '#btnNewProject', adddata: '#btnAddData', toolbox: '#btnToolbox' };
  const NEEDS_PROJECT = ['journal', 'log', 'note', 'newthread'];
  function syncMenuState(menu) {
    menu.querySelectorAll('.menu-item').forEach(it => {
      const trig = MENU_TRIG[it.dataset.act];
      if (trig) it.classList.toggle('disabled', $(trig).disabled);
      else if (NEEDS_PROJECT.includes(it.dataset.act)) it.classList.toggle('disabled', !state.project);
    });
  }
  function menuAct(act) {
    if (MENU_TRIG[act]) { const b = $(MENU_TRIG[act]); if (!b.disabled) b.click(); return; }
    // Settings is its own menu; each item opens its own pane directly.
    if (act.startsWith('set-')) { openSettings(act.slice(4)); return; }
    if (act === 'journal') openJournal();
    else if (act === 'log') openLog();
    else if (act === 'note') { openJournal().then(addJournalNote); }
    else if (act === 'newthread') newThread();
    else if (act === 'about') openAbout();
  }
  $$('#menubar .menu').forEach(menu => {
    const btn = menu.querySelector('.menu-btn');
    const openThis = () => { $$('#menubar .menu').forEach(m => m.classList.remove('open')); syncMenuState(menu); menu.classList.add('open'); };
    btn.addEventListener('click', e => {
      e.stopPropagation();
      if (menu.classList.contains('open')) menu.classList.remove('open'); else openThis();
    });
    btn.addEventListener('mouseenter', () => { if ($('#menubar .menu.open') && !menu.classList.contains('open')) openThis(); });
    menu.querySelectorAll('.menu-item').forEach(it => {
      it.addEventListener('click', e => {
        e.stopPropagation();
        if (it.classList.contains('disabled')) return;
        menu.classList.remove('open');
        menuAct(it.dataset.act);
      });
    });
  });
  document.addEventListener('click', () => $$('#menubar .menu').forEach(m => m.classList.remove('open')));

  // ==================================================================
  // Catalog (projects + files)
  // ==================================================================
  const ICONS = {
    folder: `<svg class="tree-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z"/></svg>`,
    poly:   `<svg class="tree-icon icon-pg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 5h6v6H5z"/><path d="M13 7h6v6h-6z"/><path d="M7 13h6v6H7z"/></svg>`,
    raster: `<svg class="tree-icon icon-rs" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="1"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/></svg>`,
    record: `<svg class="tree-icon" viewBox="0 0 24 24" fill="none" stroke="#6b7b8c" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4h11l5 5v11H4Z"/><path d="M8 10h7M8 14h7M8 18h4"/></svg>`,
    result: `<svg class="tree-icon" viewBox="0 0 24 24" fill="none" stroke="#B14B26" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4h12l4 4v12H4Z"/><path d="M16 4v4h4"/><circle cx="9" cy="13" r="1.5"/><path d="m7 19 4-5 3 3 3-4 2 6Z"/></svg>`,
  };
  const iconFor = fn => (IMG_EXT.includes(extOf(fn)) ? 'result' : (extOf(fn) === 'tif' || extOf(fn) === 'tiff' ? 'raster' : 'poly'));

  let allProjects = [];
  const collapsed = new Set();      // projects the user folded shut

  function toggleProjectOpen(id) {
    if (collapsed.has(id)) collapsed.delete(id); else collapsed.add(id);
    renderCatalog();
  }

  function openProjectMenu(pj, x, y) {
    const isActive = state.project && state.project.id === pj.id;
    showContextMenu(x, y, [
      { label: 'Add data…', icon: 'box', action: async () => {
          if (!isActive) await selectProject(pj.id);
          openBrowse();
        } },
      { label: 'Rename\u2026', icon: 'table',
        disabled: state.running && isActive,
        action: () => openRename(pj) },
      { label: (isActive && !collapsed.has(pj.id)) ? 'Collapse' : 'Open', icon: 'layers',
        action: () => { isActive ? toggleProjectOpen(pj.id) : selectProject(pj.id); } },
      { sep: true },
      { label: 'Journal…', icon: 'table', action: async () => {
          if (!isActive) await selectProject(pj.id);
          openJournal();
        } },
    ]);
  }

  async function loadProjects() {
    allProjects = await jget('/api/projects');
    renderCatalog();
  }

  function fileItem(fn, where) {
    const it = document.createElement('div');
    it.className = 'tree-item';
    it.innerHTML = `${ICONS[iconFor(fn)]}<span class="tree-label">${esc(fn)}</span><span class="tree-meta">${extOf(fn)}</span>`;
    it.addEventListener('click', e => {
      e.stopPropagation();
      $$('#catalog .tree-item').forEach(x => x.classList.remove('active'));
      it.classList.add('active');
      const ex = extOf(fn);
      if (IMG_EXT.includes(ex)) openImageView(fn, `/api/projects/${state.project.id}/file?where=${where}&path=${encodeURIComponent(fn)}`);
      else if (ex === 'tif' || ex === 'tiff') addRasterOverlay(fn, `/api/projects/${state.project.id}/overlay?where=${where}&path=${encodeURIComponent(fn)}`);
      else if (GEO_EXT.includes(ex)) showFileOnMap(fn, where);
      else openTextFile(fn, where);
    });
    return it;
  }

  function recordItem(fn) {
    const it = document.createElement('div');
    it.className = 'tree-item record-item';
    const label = { 'JOURNAL.md': 'Journal (full record)', 'LOG.md': 'Log (compacted)',
                    'chat.jsonl': 'Conversation' }[fn] || fn;
    it.innerHTML = `${ICONS.record}<span class="tree-label">${esc(label)}</span>`
      + `<span class="tree-meta">${esc(fn.split('.').pop())}</span>`;
    it.addEventListener('click', e => {
      e.stopPropagation();
      $$('#catalog .tree-item').forEach(x => x.classList.remove('active'));
      it.classList.add('active');
      openTextFile(fn, 'records');
    });
    return it;
  }

  function renderCatalog() {
    const host = $('#catalog');
    host.innerHTML = '';
    if (!allProjects.length) {
      host.innerHTML = `<div class="tree-empty">No projects yet.<br/>Press <b>＋ New</b> to create one.</div>`;
      return;
    }
    allProjects.forEach(pj => {
      const isActive = state.project && state.project.id === pj.id;
      const isOpen = isActive && !collapsed.has(pj.id);
      const sec = document.createElement('div');
      sec.className = 'tree-section';
      sec.dataset.project = pj.id;

      const grp = document.createElement('div');
      grp.className = 'tree-group task-folder' + (isOpen ? ' expanded' : '') + (isActive ? ' active-folder' : '');
      grp.innerHTML = `<span class="tree-caret"></span>${ICONS.folder}<span class="tree-label">${esc(pj.name)}</span>`
        + `<span class="tree-meta">${pj.data_count || 0}</span>`;
      // Clicking the open project collapses it again; clicking another opens it.
      grp.addEventListener('click', () => {
        if (state.project && state.project.id === pj.id) toggleProjectOpen(pj.id);
        else selectProject(pj.id);
      });
      // Right-click a project folder → Add data (and the other project actions).
      grp.addEventListener('contextmenu', e => {
        e.preventDefault();
        e.stopPropagation();
        openProjectMenu(pj, e.clientX, e.clientY);
      });
      sec.appendChild(grp);

      const children = document.createElement('div');
      children.className = 'tree-children';
      children.style.display = (isActive && !collapsed.has(pj.id)) ? '' : 'none';

      if (isOpen && state.tree) {
        // data
        const dHead = document.createElement('div');
        dHead.className = 'tree-subhead'; dHead.textContent = 'data';
        children.appendChild(dHead);
        if (state.tree.data.length) state.tree.data.forEach(fn => children.appendChild(fileItem(fn, 'data')));
        else { const e = document.createElement('div'); e.className = 'tree-hint-item'; e.textContent = '(empty — add data)'; children.appendChild(e); }
        // records — the project's own journal / log / conversation
        if ((state.tree.records || []).length) {
          const rHead = document.createElement('div');
          rHead.className = 'tree-subhead'; rHead.textContent = 'records';
          children.appendChild(rHead);
          state.tree.records.forEach(fn => children.appendChild(recordItem(fn)));
        }
        // outputs
        if (state.tree.outputs.length) {
          const oHead = document.createElement('div');
          oHead.className = 'tree-subhead'; oHead.textContent = 'outputs';
          children.appendChild(oHead);
          state.tree.outputs.forEach(fn => children.appendChild(fileItem(fn, 'outputs')));
        }
      }
      sec.appendChild(children);
      host.appendChild(sec);
    });
  }

  async function selectProject(id) {
    if (state.running) return;
    const pj = allProjects.find(p => p.id === id);
    if (!pj) return;
    state.project = { id: pj.id, name: pj.name };
    state.tree = await jget(`/api/projects/${id}/tree`);
    $('#chatTaskTitle').textContent = pj.name;
    $('#regionVal').textContent = pj.name;
    $('#btnAddData').disabled = false;
    $('#btnToolbox').disabled = false;
    $('#startBtn').disabled = false;
    $('#footHint').textContent = state.tree.data.length ? 'Describe an analysis and press Run.' : 'Add data to this project to begin.';
    renderCatalog();
    clearMap();
    resetCounters(); resetCode(); resetImageView();
    await loadHistory();          // the conversation survives reloads and restarts
    setTimeout(() => map.invalidateSize(), 30);
  }

  async function refreshTree() {
    if (!state.project) return;
    state.tree = await jget(`/api/projects/${state.project.id}/tree`);
    const pj = allProjects.find(p => p.id === state.project.id);
    if (pj) pj.data_count = state.tree.data.length;
    renderCatalog();
  }

  // ==================================================================
  // Chat + counters
  // ==================================================================
  const chatScroll = $('#chatScroll');
  const KIND_LABELS = { user: 'You', thought: 'Thought', action: 'Action', observe: 'Observation', error: 'Issue', finish: 'Done', system: 'Info' };
  function nowTime() { const d = new Date(); return d.toTimeString().slice(0, 8); }
  function addMsg({ kind, text, html }) {
    const m = document.createElement('div');
    m.className = 'msg msg-' + kind;
    m.innerHTML = `<div class="msg-meta"><span class="agent-tag">${KIND_LABELS[kind] || kind}</span><span class="ts">${nowTime()}</span></div>`
      + `<div class="msg-actions"><button class="msg-act" title="Save this as a standing preference in your global memory">Remember</button></div>`
      + `<div class="msg-body">${html || esc(text || '')}</div>`;
    m.querySelector('.msg-act').addEventListener('click', () => {
      rememberText(text || m.querySelector('.msg-body').textContent);
    });
    chatScroll.appendChild(m);
    chatScroll.scrollTop = chatScroll.scrollHeight;
    return m;
  }
  function setStatus(text, klass) {
    $('#statusText').textContent = text;
    $('.status').classList.remove('running', 'done');
    if (klass) $('.status').classList.add(klass);
  }
  function startTimer() {
    state.startedAt = Date.now();
    state.timer = setInterval(() => {
      const s = Math.floor((Date.now() - state.startedAt) / 1000);
      $('#timerCount').textContent = `Elapsed · ${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
    }, 250);
  }
  function stopTimer() { if (state.timer) { clearInterval(state.timer); state.timer = null; } }
  function bumpStep() { state.steps++; $('#stepCount').textContent = `Steps · ${state.steps}`; }
  function setSelfCorr(n) { state.selfCorr = n; $('#selfCorrCount').textContent = `Self-corrections · ${n}`; }
  function resetCounters() {
    state.steps = 0; state.selfCorr = 0;
    $('#stepCount').textContent = 'Steps · 0';
    $('#selfCorrCount').textContent = 'Self-corrections · 0';
    $('#timerCount').textContent = 'Elapsed · 00:00';
  }

  // Tabs
  function switchTab(name) {
    $$('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
    $('#mapView').classList.toggle('hidden', name !== 'map');
    $('#codeView').classList.toggle('hidden', name !== 'code');
    $('#imageView').classList.toggle('hidden', name !== 'image');
    $('#dataView').classList.toggle('hidden', name !== 'data');
    if (name === 'map') setTimeout(() => map.invalidateSize(), 20);
  }
  $$('.tab').forEach(t => t.addEventListener('click', () => { if (!t.hasAttribute('hidden')) switchTab(t.dataset.tab); }));

  // ---- Table / text viewer: csv, txt, json, md open here, not in the chat ----
  let dataDoc = { name: '', text: '', rows: null, raw: false };

  function parseCsv(text) {
    // Small, dependency-free CSV reader: quotes, escaped quotes, embedded commas.
    const rows = [];
    let row = [], field = '', inQ = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (inQ) {
        if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else inQ = false; }
        else field += c;
      } else if (c === '"') inQ = true;
      else if (c === ',') { row.push(field); field = ''; }
      else if (c === '\n') { row.push(field); rows.push(row); row = []; field = ''; }
      else if (c !== '\r') field += c;
    }
    if (field.length || row.length) { row.push(field); rows.push(row); }
    return rows.filter(r => r.length && !(r.length === 1 && r[0] === ''));
  }

  function openDataView(filename, text) {
    const ext = extOf(filename);
    dataDoc = { name: filename, text, raw: false, rows: null };
    if (ext === 'csv' || ext === 'tsv') {
      const rows = parseCsv(ext === 'tsv' ? text.replace(/\t/g, ',') : text);
      if (rows.length > 1) dataDoc.rows = rows;
    }
    $('#dataFilename').textContent = filename;
    $('#dataTab').removeAttribute('hidden');
    $('#dataFilter').value = '';
    renderDataView();
    switchTab('data');
  }

  function renderDataView() {
    const stage = $('#dataStage');
    const q = ($('#dataFilter').value || '').toLowerCase();
    $('#dataFilter').style.display = (dataDoc.rows && !dataDoc.raw) ? '' : 'none';
    $('#dataRaw').textContent = dataDoc.raw ? 'Table' : 'Raw text';
    $('#dataRaw').style.display = dataDoc.rows ? '' : 'none';

    if (!dataDoc.rows || dataDoc.raw) {
      stage.innerHTML = `<pre class="data-text">${esc(dataDoc.text.slice(0, 200000))}</pre>`;
      $('#dataMeta').textContent = `${dataDoc.text.split('\n').length} lines`;
      return;
    }
    const [head, ...body] = dataDoc.rows;
    const shown = q ? body.filter(r => r.some(c => (c || '').toLowerCase().includes(q))) : body;
    const capped = shown.slice(0, 2000);
    stage.innerHTML =
      `<table class="attr-table"><thead><tr><th class="attr-idx">#</th>`
      + head.map(h => `<th>${esc(h)}</th>`).join('') + `</tr></thead><tbody>`
      + capped.map((r, i) => `<tr><td class="attr-idx">${i + 1}</td>`
          + head.map((_, c) => `<td>${esc(r[c] === undefined ? '' : r[c])}</td>`).join('') + `</tr>`).join('')
      + `</tbody></table>`;
    $('#dataMeta').textContent = `${shown.length} of ${body.length} rows × ${head.length} cols`
      + (capped.length < shown.length ? ` · showing first ${capped.length}` : '');
  }
  $('#dataFilter').addEventListener('input', renderDataView);
  $('#dataRaw').addEventListener('click', () => { dataDoc.raw = !dataDoc.raw; renderDataView(); });

  function openImageView(filename, src) {
    const img = $('#imageEl');
    img.src = src + (src.includes('?') ? '&' : '?') + 't=' + Date.now();
    $('#imageFilename').textContent = filename;
    $('#imageMeta').textContent = 'Agent output';
    $('#imageTab').removeAttribute('hidden');
    switchTab('image');
  }
  function resetImageView() {
    $('#imageTab').setAttribute('hidden', '');
    $('#imageEl').removeAttribute('src');
    $('#imageFilename').textContent = '—';
    $('#imageMeta').textContent = '';
  }
  function openTextFile(fn, where) {
    fetch(`/api/projects/${state.project.id}/file?where=${where}&path=${encodeURIComponent(fn)}`)
      .then(r => r.json()).then(j => {
        openDataView(fn, j.content || '');
      }).catch(() => {});
  }

  // ==================================================================
  // Code viewer (real streamed code + stdout)
  // ==================================================================
  function syntaxHighlight(line) {
    let s = line.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    if (/^\s*#/.test(s)) return '<span class="cmt">' + s + '</span>';
    const tok = [];
    const stash = (cls, text) => { tok.push([cls, text]); return 'qXk' + (tok.length - 1) + 'Xq'; };
    s = s.replace(/('[^']*'|"[^"]*")/g, m => stash('str', m));
    s = s.replace(/\b(\d+\.?\d*)\b/g, m => stash('num', m));
    s = s.replace(/\b(import|from|as|def|return|for|in|if|else|True|False|None|not|and|or|with|lambda|try|except|print)\b/g, m => stash('kw', m));
    s = s.replace(/\b(gpd|np|pd|plt|rasterio|shape|zonal_stats|read_file|open|plot|savefig|to_crs|sjoin|apply|groupby|to_file|to_csv)\b/g, m => stash('fn', m));
    s = s.replace(/qXk(\d+)Xq/g, (_, i) => '<span class="' + tok[+i][0] + '">' + tok[+i][1] + '</span>');
    return s;
  }
  function resetCode() {
    $('#codeGutter').textContent = '';
    $('#codeLines').innerHTML = '<span class="code-placeholder">// Generated code appears here as the agent runs…</span>';
    $('#codeOutputBody').innerHTML = '<span class="out-muted">Output appears here as the agent runs code.</span>';
    $('#codeRunState').textContent = 'Ready'; $('#codeRunState').className = 'code-run-state';
    state.codeStarted = false;
  }
  function appendCode(codeText) {
    const gutter = $('#codeGutter'), lines = $('#codeLines');
    if (!state.codeStarted) { gutter.textContent = ''; lines.innerHTML = ''; state.codeStarted = true; }
    const start = gutter.children.length;
    const arr = codeText.replace(/\n$/, '').split('\n');
    arr.forEach((raw, i) => {
      const g = document.createElement('span');
      g.textContent = (start + i + 1).toString().padStart(3, ' ') + '\n';
      gutter.appendChild(g);
      const el = document.createElement('span');
      el.className = 'code-line';
      el.innerHTML = (syntaxHighlight(raw) || '&nbsp;') + '\n';
      lines.appendChild(el);
    });
    lines.scrollTop = lines.scrollHeight;
  }
  function showStdout(text, ok) {
    const out = $('#codeOutputBody');
    if (out.querySelector('.out-muted')) out.innerHTML = '';
    const pre = document.createElement('pre');
    pre.className = ok ? 'out-ok' : 'out-err';
    pre.textContent = text;
    out.appendChild(pre);
    out.scrollTop = out.scrollHeight;
    const rs = $('#codeRunState');
    rs.textContent = ok ? 'Executed' : 'Error';
    rs.className = 'code-run-state ' + (ok ? 'success' : 'failed');
  }

  // ==================================================================
  // Run (SSE over POST)
  // ==================================================================
  function resetRunUI() {
    // Counters/code/image reset per run — but the conversation is a record now,
    // so it is never wiped here; new turns are appended below the history.
    resetCounters(); resetCode(); resetImageView();
    setStatus('Idle', '');
  }

  async function newThread() {
    if (!state.project || state.running) return;
    await fetch(`/api/projects/${state.project.id}/chat`, { method: 'DELETE' });
    await loadHistory();
    addMsg({ kind: 'system', text: 'Started a new conversation. The previous one is archived in the project folder, and JOURNAL.md still holds every run.' });
  }

  async function runAnalysis() {
    if (state.running || !state.project) return;
    const instruction = $('#promptInput').value.trim();
    if (!instruction) { addMsg({ kind: 'error', text: 'Please describe the analysis first.' }); return; }
    if (!state.tree || !state.tree.data.length) { addMsg({ kind: 'error', text: 'This project has no data yet. Add data first.' }); return; }
    if (!$('#modelSelect').value) {
      addMsg({ kind: 'error', html: 'No model configured — open <b>Tools → Settings</b> and add an API key first.' });
      return;
    }

    const rid = ++state.runId;
    state.running = true;
    resetRunUI();
    // Echo the prompt as the user's own turn, then clear the box — otherwise the
    // text sits there and the next run silently re-sends it.
    addMsg({ kind: 'user', text: instruction });
    $('#promptInput').value = '';
    $('#startBtn').classList.add('hidden'); $('#stopBtn').classList.remove('hidden');
    setStatus('Running', 'running'); startTimer();

    let resp;
    try {
      resp = await fetch('/api/run', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_id: state.project.id, model: $('#modelSelect').value, instruction }),
      });
    } catch (e) { finishRun(rid, false, 'Network error'); return; }

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      if (rid !== state.runId) { try { reader.cancel(); } catch (e) {} return; }
      let chunk;
      try { chunk = await reader.read(); } catch (e) { break; }
      if (chunk.done) break;
      buf += dec.decode(chunk.value, { stream: true });
      const blocks = buf.split(/\r?\n\r?\n/);
      buf = blocks.pop();
      for (const block of blocks) {
        if (rid !== state.runId) return;
        handleSSE(block, rid);
      }
    }
  }

  function handleSSE(block, rid) {
    let ev = 'message', data = '';
    block.split('\n').forEach(line => {
      if (line.startsWith('event:')) ev = line.slice(6).trim();
      else if (line.startsWith('data:')) data += line.slice(5).trim();
    });
    if (!data) return;
    let msg; try { msg = JSON.parse(data); } catch (e) { return; }

    if (ev === 'heartbeat') return;
    if (ev === 'status') { addMsg({ kind: 'system', text: msg.content }); return; }
    if (ev === 'answer') {
      // Answered from the project record — no analysis run was needed.
      addMsg({ kind: msg.mode === 'offtopic' ? 'system' : 'finish', text: msg.content });
      return;
    }
    if (ev === 'log') {
      const m = addMsg({ kind: 'system', html: `<b>Log entry written</b><div class="log-digest">${esc(msg.content)}</div>` });
      m.classList.add('msg-history');
      return;
    }
    if (ev === 'error') { addMsg({ kind: 'error', text: msg.content }); finishRun(rid, false); return; }
    if (ev === 'result') {
      addResultFile(msg.filename, msg.url);
      const ex = extOf(msg.filename);
      if (ex === 'tif' || ex === 'tiff') {
        addRasterOverlay(msg.filename, `/api/projects/${state.project.id}/overlay?run=${encodeURIComponent(msg.run_id)}&path=${encodeURIComponent(msg.filename)}`);
      } else if (IMG_EXT.includes(ex)) openImageView(msg.filename, msg.url);
      else if (GEO_EXT.includes(ex)) addResultGeoToMap(msg.url, msg.filename);
      return;
    }
    if (ev === 'done') { finishRun(rid, msg.success, null, msg); return; }
    if (ev === 'step') {
      if (msg.thought) addMsg({ kind: 'thought', text: msg.thought });
      if (msg.action) {
        bumpStep();
        addMsg({ kind: 'action', html: `<code>${esc(msg.action)}</code>` });
      }
      if (msg.code) { switchTab('code'); appendCode(msg.code); }
      if (msg.observation) {
        const obs = msg.observation.slice(0, 600);
        addMsg({ kind: msg.success ? 'observe' : 'error', text: obs });
        if (msg.code) showStdout(obs, msg.success);
      }
      return;
    }
  }

  function addResultFile(filename, url) {
    // reflect into the outputs tree without a full refetch
    if (state.tree && !state.tree.outputs.includes(filename)) state.tree.outputs.push(filename);
    renderCatalog();
  }

  function finishRun(rid, success, errText, done) {
    if (rid !== state.runId) return;
    stopTimer();
    state.running = false;
    $('#startBtn').classList.remove('hidden'); $('#stopBtn').classList.add('hidden');
    setStatus(success ? 'Done' : 'Stopped', success ? 'done' : '');
    if (done && done.answered) {
      setStatus('Idle', '');
      return;                       // answered from the record; nothing was run
    }
    if (done) {
      setSelfCorr(done.self_corrections || 0);
      addMsg({ kind: 'finish', html: `Finished in <b>${done.elapsed_s}s</b> · ${done.rounds} rounds · ${(done.output_files || []).length} output(s)` });
    } else if (errText) {
      addMsg({ kind: 'error', text: errText });
    }
    refreshTree();
  }

  $('#startBtn').addEventListener('click', runAnalysis);
  // Hotkey: Enter submits the run, Shift+Enter inserts a newline.
  $('#promptInput').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      runAnalysis();
    }
  });
  $('#stopBtn').addEventListener('click', () => {
    state.runId++; state.running = false; stopTimer();
    setStatus('Stopped', '');
    $('#startBtn').classList.remove('hidden'); $('#stopBtn').classList.add('hidden');
  });
  // "Clear" resets the view, then restores the conversation from disk — the
  // record is the source of truth, not whatever this tab happened to render.
  $('#resetBtn').addEventListener('click', async () => {
    state.runId++; state.running = false; stopTimer(); resetRunUI(); clearMap();
    if (state.project) await loadHistory(); else chatScroll.innerHTML = '';
  });

  // ==================================================================
  // New project modal
  // ==================================================================
  const newModal = $('#newModal');
  $('#btnNewProject').addEventListener('click', () => { $('#newName').value = ''; newModal.classList.remove('hidden'); $('#newName').focus(); });
  const closeNew = () => newModal.classList.add('hidden');
  $('#newClose').addEventListener('click', closeNew);
  $('#newCancel').addEventListener('click', closeNew);
  $('#newCreate').addEventListener('click', async () => {
    const name = $('#newName').value.trim();
    if (!name) return;
    const res = await jpost('/api/projects', { name });
    closeNew();
    await loadProjects();
    if (res.id) selectProject(res.id);
  });
  $('#newName').addEventListener('keydown', e => { if (e.key === 'Enter') $('#newCreate').click(); });

  // ==================================================================
  // File browser modal
  // ==================================================================
  const browseModal = $('#browseModal');
  let browseSelected = new Set();
  let browseHere = '';

  function openBrowse() {
    if (!state.project) return;
    browseSelected = new Set();
    browseHere = '';
    browseModal.classList.remove('hidden');
    loadBrowse('');
  }
  const closeBrowse = () => browseModal.classList.add('hidden');

  // The container only sees the mounted workspace, so data kept anywhere else
  // has to come in through the browser. One raw-body request per file.
  async function uploadLocalFiles(fileList) {
    if (!state.project || !fileList || !fileList.length) return;
    const files = Array.from(fileList);
    const count = $('#browseCount');
    let ok = 0, bad = 0;
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      count.textContent = `uploading ${i + 1}/${files.length} — ${f.name}`;
      const q = `?name=${encodeURIComponent(f.name)}`
              + `&rel=${encodeURIComponent(f.webkitRelativePath || f.name)}`;
      try {
        const r = await fetch(`/api/projects/${state.project.id}/upload${q}`,
                              { method: 'POST', body: f });
        if (!r.ok) throw new Error(r.statusText);
        ok++;
      } catch (e) { bad++; }
    }
    count.textContent = `${ok} uploaded${bad ? `, ${bad} failed` : ''}`;
    closeBrowse();
    await refreshTree();
    addMsg({ kind: bad ? 'error' : 'system',
             text: `Uploaded ${ok} file(s) to ${state.project.name}.`
                   + (bad ? ` ${bad} failed.` : '') });
    try {
      const chk = await jget(`/api/projects/${state.project.id}/data_check`);
      (chk.notices || []).forEach(n => addMsg({ kind: 'system', text: n }));
    } catch (e) {}
  }

  $('#btnUploadFiles').addEventListener('click', () => $('#uploadFiles').click());
  $('#btnUploadFolder').addEventListener('click', () => $('#uploadFolder').click());
  ['#uploadFiles', '#uploadFolder'].forEach(sel =>
    $(sel).addEventListener('change', e => {
      // Copy the list out before resetting the input: input.files is live, so
      // clearing value would empty the very FileList we are about to send.
      const files = Array.from(e.target.files || []);
      e.target.value = '';
      uploadLocalFiles(files);
    }));
  ['dragenter', 'dragover'].forEach(ev => browseModal.addEventListener(ev, e => {
    e.preventDefault(); browseModal.classList.add('dragging');
  }));
  browseModal.addEventListener('dragleave', e => {
    if (e.target === browseModal) browseModal.classList.remove('dragging');
  });
  browseModal.addEventListener('drop', e => {
    e.preventDefault();
    browseModal.classList.remove('dragging');
    uploadLocalFiles(e.dataTransfer.files);
  });

  // ---- rename a project -------------------------------------------------
  const renameModal = $('#renameModal');
  let renameTarget = null;
  function openRename(pj) {
    renameTarget = pj.id;
    const inp = $('#renameName');
    inp.value = pj.name || '';
    renameModal.classList.remove('hidden');
    setTimeout(() => { inp.focus(); inp.select(); }, 30);
  }
  async function doRename() {
    const name = $('#renameName').value.trim();
    if (!name || !renameTarget) return;
    const res = await jpost(`/api/projects/${renameTarget}/rename`, { name });
    renameModal.classList.add('hidden');
    if (res.error) { addMsg({ kind: 'error', text: res.error }); return; }
    if (res.notice) addMsg({ kind: 'system', text: res.notice });
    renameTarget = null;
    await loadProjects();
    await selectProject(res.id);
  }
  $('#renameSave').addEventListener('click', doRename);
  $('#renameCancel').addEventListener('click', () => renameModal.classList.add('hidden'));
  $('#renameClose').addEventListener('click', () => renameModal.classList.add('hidden'));
  $('#renameName').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); doRename(); }
  });
  $('#btnAddData').addEventListener('click', openBrowse);
  $('#browseClose').addEventListener('click', closeBrowse);
  $('#browseCancel').addEventListener('click', closeBrowse);

  async function loadBrowse(path) {
    const data = await jget('/api/browse?path=' + encodeURIComponent(path));
    if (data.error) return;
    browseHere = data.here || '';
    renderBreadcrumb(browseHere);
    const list = $('#browseList');
    list.innerHTML = '';
    if (browseHere) {
      const up = document.createElement('div');
      up.className = 'browse-row browse-up';
      up.innerHTML = `<span class="browse-name">⬑ ..</span>`;
      up.addEventListener('click', () => { const parts = browseHere.split('/'); parts.pop(); loadBrowse(parts.join('/')); });
      list.appendChild(up);
    }
    data.entries.forEach(en => {
      const row = document.createElement('div');
      row.className = 'browse-row' + (en.is_dir ? ' is-dir' : '');
      const checked = browseSelected.has(en.rel) ? 'checked' : '';
      row.innerHTML = `<input type="checkbox" class="browse-cb" ${checked} />`
        + `<span class="browse-ic">${en.is_dir ? ICONS.folder : ICONS.poly}</span>`
        + `<span class="browse-name">${esc(en.name)}</span>`
        + `<span class="browse-size">${en.is_dir ? '' : humanSize(en.size)}</span>`;
      const cb = row.querySelector('.browse-cb');
      cb.addEventListener('click', e => {
        e.stopPropagation();
        if (cb.checked) browseSelected.add(en.rel); else browseSelected.delete(en.rel);
        updateBrowseCount();
      });
      // Clicking anywhere on a file row picks it. Only the checkbox used to
      // work, which reads as "the file cannot be selected".
      row.addEventListener('click', () => {
        if (en.is_dir) { loadBrowse(en.rel); return; }
        cb.checked = !cb.checked;
        if (cb.checked) browseSelected.add(en.rel); else browseSelected.delete(en.rel);
        row.classList.toggle('picked', cb.checked);
        updateBrowseCount();
      });
      row.classList.toggle('picked', browseSelected.has(en.rel));
      list.appendChild(row);
    });
    updateBrowseCount();
  }
  function renderBreadcrumb(here) {
    const el = $('#browsePath');
    const parts = here ? here.split('/') : [];
    let acc = '';
    const crumbs = [`<a data-p="">workspace</a>`];
    parts.forEach(p => { acc = acc ? acc + '/' + p : p; crumbs.push(`<a data-p="${esc(acc)}">${esc(p)}</a>`); });
    el.innerHTML = crumbs.join(' / ');
    $$('#browsePath a').forEach(a => a.addEventListener('click', () => loadBrowse(a.dataset.p)));
  }
  function humanSize(n) { if (!n) return ''; const u = ['B', 'KB', 'MB', 'GB']; let i = 0; while (n >= 1024 && i < 3) { n /= 1024; i++; } return n.toFixed(i ? 1 : 0) + u[i]; }
  function updateBrowseCount() { $('#browseCount').textContent = `${browseSelected.size} selected`; }

  $('#browseAttach').addEventListener('click', async () => {
    if (!browseSelected.size) { closeBrowse(); return; }
    const res = await jpost(`/api/projects/${state.project.id}/attach`, { paths: [...browseSelected] });
    closeBrowse();
    await refreshTree();
    if (res.attached && res.attached.length) {
      $('#footHint').textContent = 'Describe an analysis and press Run.';
      addMsg({ kind: 'system', text: `Added ${res.attached.length} file(s) to ${state.project.name}.` });
    }
    if (res.notices && res.notices.length) {
      res.notices.forEach(n => addMsg({ kind: 'system', text: n }));
    }
  });

  // ==================================================================
  // Toolbox: run an operation directly, or insert it into the chat
  // ==================================================================
  const toolModal = $('#toolModal');
  let toolCatalog = [];
  const RASTER_EXT = ['tif', 'tiff'];
  const isRasterFile = fn => RASTER_EXT.includes(extOf(fn));
  const isVectorFile = fn => GEO_EXT.includes(extOf(fn));

  function projectLayers(kind) {
    if (!state.tree) return [];
    const all = [...state.tree.data, ...state.tree.outputs];
    return all.filter(kind === 'raster' ? isRasterFile : isVectorFile);
  }

  async function openToolbox() {
    if (!state.project) return;
    if (!toolCatalog.length) toolCatalog = await jget('/api/tools');
    renderToolList();
    $('#toolForm').innerHTML = `<div class="tool-empty">Select an operation on the left.</div>`;
    toolModal.classList.remove('hidden');
  }
  const closeToolbox = () => toolModal.classList.add('hidden');
  $('#btnToolbox').addEventListener('click', openToolbox);
  $('#toolClose').addEventListener('click', closeToolbox);

  function renderToolList() {
    const host = $('#toolList');
    host.innerHTML = '';
    toolCatalog.forEach(group => {
      const h = document.createElement('div');
      h.className = 'tool-cat';
      h.innerHTML = svgIcon(CAT_ICON[group.category] || 'box', 'ic-cat') + `<span>${esc(group.category)}</span>`;
      host.appendChild(h);
      group.ops.forEach(op => {
        const it = document.createElement('div');
        it.className = 'tool-op';
        it.innerHTML = `<span class="tool-op-name">${esc(op.op)}</span>`;
        it.title = op.desc;
        it.addEventListener('click', () => {
          $$('#toolList .tool-op').forEach(x => x.classList.remove('active'));
          it.classList.add('active');
          renderToolForm(op);
        });
        host.appendChild(it);
      });
    });
  }

  function renderToolForm(op) {
    const form = $('#toolForm');
    let html = `<div class="tf-title">${esc(op.op)}</div><div class="tf-desc">${esc(op.desc)}</div>`;
    // input layers
    op.inputs.forEach(inp => {
      const opts = projectLayers(inp.kind);
      const optHtml = opts.length
        ? opts.map(f => `<option value="${esc(f)}">${esc(f)}</option>`).join('')
        : `<option value="">(no ${inp.kind} layers — add data)</option>`;
      html += `<div class="tf-row"><label>${esc(inp.role)} <span class="tf-kind">${inp.kind}</span></label>`
        + `<select class="tf-input" data-role="${esc(inp.role)}">${optHtml}</select></div>`;
    });
    // params
    op.args.forEach(a => {
      const req = a.required ? ' *' : '';
      let field;
      if (a.type === 'bool') field = `<input type="checkbox" class="tf-arg" data-name="${esc(a.name)}" data-type="bool" ${a.default ? 'checked' : ''}/>`;
      else if (a.type === 'select') field = `<select class="tf-arg" data-name="${esc(a.name)}" data-type="select">${(a.choices || []).map(c => `<option ${c === a.default ? 'selected' : ''}>${esc(c)}</option>`).join('')}</select>`;
      else field = `<input type="${a.type === 'number' || a.type === 'crs' ? 'text' : 'text'}" class="tf-arg" data-name="${esc(a.name)}" data-type="${a.type}" value="${a.default !== null && a.default !== '' ? esc(String(a.default)) : ''}" placeholder="${esc(a.type)}"/>`;
      html += `<div class="tf-row"><label>${esc(a.name)}${req}</label>${field}</div>`;
    });
    html += `<div class="tf-row"><label>output name</label><input class="tf-output" value="${esc(op.op)}_out"/></div>`;
    html += `<div class="tf-actions"><button class="reset-btn" id="tfInsert"><span>Insert into chat</span></button>`
      + `<span class="ca-spacer"></span><button class="primary-btn" id="tfRun"><span>Run</span></button></div>`
      + `<div class="tf-status" id="tfStatus"></div>`;
    form.innerHTML = html;
    $('#tfRun').addEventListener('click', () => runTool(op));
    $('#tfInsert').addEventListener('click', () => insertTool(op));
  }

  function collectTool(op) {
    const inputs = {};
    $$('#toolForm .tf-input').forEach(s => { inputs[s.dataset.role] = s.value; });
    const params = {};
    $$('#toolForm .tf-arg').forEach(el => {
      const name = el.dataset.name, t = el.dataset.type;
      if (t === 'bool') params[name] = el.checked;
      else if (t === 'number' || t === 'crs') { const v = el.value.trim(); if (v !== '') params[name] = t === 'crs' && /^\d+$/.test(v) ? parseInt(v) : (t === 'number' ? parseFloat(v) : (/^\d+$/.test(v) ? parseInt(v) : v)); }
      else { if (el.value.trim() !== '') params[name] = el.value; }
    });
    const output = ($('#toolForm .tf-output').value || (op.op + '_out')).trim();
    return { op: op.op, inputs, params, output };
  }

  async function runTool(op) {
    const payload = collectTool(op);
    const st = $('#tfStatus'); st.textContent = 'Running…'; st.className = 'tf-status running';
    let res;
    try { res = await jpost(`/api/projects/${state.project.id}/geoprocess`, payload); }
    catch (e) { st.textContent = 'Network error'; st.className = 'tf-status err'; return; }
    if (res.error || res.ok === false) {
      st.textContent = res.error || 'Failed — check inputs.'; st.className = 'tf-status err';
      return;
    }
    st.textContent = `Done → ${(res.outputs || []).map(o => o.filename).join(', ')}`; st.className = 'tf-status ok';
    await refreshTree();
    (res.outputs || []).forEach(o => {
      const url = o.kind === 'raster'
        ? `/api/projects/${state.project.id}/overlay?where=outputs&path=${encodeURIComponent(o.filename)}`
        : `/api/projects/${state.project.id}/file?where=outputs&path=${encodeURIComponent(o.filename)}`;
      if (o.kind === 'raster') addRasterOverlay(o.filename, url);
      else addResultGeoToMap(url, o.filename);
    });
    addMsg({ kind: 'system', html: `${svgIcon('wrench', 'ic')} <b>${esc(op.op)}</b> → ${esc((res.outputs || []).map(o => o.filename).join(', '))}` });
    closeToolbox();
  }

  function insertTool(op) {
    const p = collectTool(op);
    const ins = Object.entries(p.inputs).map(([r, f]) => `${r}=${f}`).join(', ');
    const args = Object.entries(p.params).map(([k, v]) => `${k}=${v}`).join(', ');
    const sentence = `Use the ${op.op} operation on ${ins}${args ? ' with ' + args : ''}.`;
    const ta = $('#promptInput');
    ta.value = (ta.value ? ta.value.trim() + '\n' : '') + sentence;
    closeToolbox();
    ta.focus();
  }

  // ==================================================================
  // Resizable panes — drag the hairlines, double-click to reset
  // ==================================================================
  const COL_DEFAULTS = { left: 264, right: 390 };
  const COL_MIN = { left: 170, right: 260 };
  const COL_MAX_FRAC = 0.55;          // never let one pane eat the map

  function applyCols(cols) {
    const root = document.documentElement;
    root.style.setProperty('--col-left', cols.left + 'px');
    root.style.setProperty('--col-right', cols.right + 'px');
    try { localStorage.setItem('gisclaw.cols', JSON.stringify(cols)); } catch (e) {}
    map.invalidateSize();
  }

  const cols = (() => {
    try {
      const s = JSON.parse(localStorage.getItem('gisclaw.cols') || 'null');
      if (s && s.left && s.right) return s;
    } catch (e) {}
    return { ...COL_DEFAULTS };
  })();
  applyCols(cols);

  function wireSplitter(el, side) {
    let startX = 0, startW = 0, dragging = false;

    const onMove = e => {
      if (!dragging) return;
      const dx = e.clientX - startX;
      const raw = side === 'left' ? startW + dx : startW - dx;
      const max = Math.round(window.innerWidth * COL_MAX_FRAC);
      cols[side] = Math.max(COL_MIN[side], Math.min(max, Math.round(raw)));
      applyCols(cols);
    };
    const onUp = () => {
      if (!dragging) return;
      dragging = false;
      el.classList.remove('dragging');
      document.body.classList.remove('col-resizing');
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      setTimeout(() => map.invalidateSize(), 50);
    };

    el.addEventListener('mousedown', e => {
      e.preventDefault();
      dragging = true;
      startX = e.clientX;
      startW = cols[side];
      el.classList.add('dragging');
      document.body.classList.add('col-resizing');
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);
    });
    el.addEventListener('dblclick', () => {
      cols[side] = COL_DEFAULTS[side];
      applyCols(cols);
    });
  }
  wireSplitter($('#splitLeft'), 'left');
  wireSplitter($('#splitRight'), 'right');

  // ==================================================================
  // Settings — API keys, model registry, memory
  // ==================================================================
  const setModal = $('#setModal');
  let settings = { providers: [], models: [] };

  async function jsend(url, body, method) {
    const r = await fetch(url, {
      method: method || 'POST', headers: { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    return r.json();
  }

  async function openSettings(pane) {
    settings = await jget('/api/settings');
    $('#setPath').textContent = settings.settings_path || '';
    $('#memPath').textContent = settings.memory_path || '';
    renderProviders();
    renderModels();
    await loadSkills();
    await loadMemory();
    switchSetPane(pane || 'keys');
    setModal.classList.remove('hidden');
  }
  const closeSettings = () => setModal.classList.add('hidden');
  $('#setClose').addEventListener('click', closeSettings);

  function switchSetPane(name) {
    $$('.set-tab').forEach(t => t.classList.toggle('active', t.dataset.pane === name));
    $('#paneKeys').classList.toggle('hidden', name !== 'keys');
    $('#paneModels').classList.toggle('hidden', name !== 'models');
    $('#paneSkills').classList.toggle('hidden', name !== 'skills');
    $('#paneMemory').classList.toggle('hidden', name !== 'memory');
  }
  $$('.set-tab').forEach(t => t.addEventListener('click', () => switchSetPane(t.dataset.pane)));

  function renderProviders() {
    const host = $('#provList');
    host.innerHTML = '';
    settings.providers.forEach(p => {
      const card = document.createElement('div');
      card.className = 'prov';
      const badge = p.configured
        ? `<span class="prov-badge ${p.from_env ? 'env' : 'ok'}">${p.from_env ? 'from ' + esc(p.env_var) : 'key saved'}</span>`
        : `<span class="prov-badge">no key</span>`;
      card.innerHTML =
        `<div class="prov-head">
           <span class="prov-name">${esc(p.display)}</span>${badge}
           ${p.docs ? `<a class="prov-docs" href="${esc(p.docs)}" target="_blank" rel="noopener">get a key ↗</a>` : ''}
         </div>
         <div class="prov-row">
           <input type="password" class="pv-key" autocomplete="off" spellcheck="false"
                  placeholder="${p.configured ? esc(p.masked_key) + '  (stored — type to replace)' : esc(p.key_hint || 'paste your API key')}" />
           <button class="mini-btn primary pv-save">Save</button>
           <button class="mini-btn pv-test">Test</button>
           ${p.configured && !p.from_env ? `<button class="mini-btn danger pv-clear">Remove</button>` : ''}
         </div>
         ${p.needs_base_url ? `<div class="prov-row">
           <input type="text" class="pv-url" spellcheck="false" value="${esc(p.base_url || '')}"
                  placeholder="https://your-endpoint/v1  (OpenAI-compatible base URL)" />
         </div>` : ''}
         <div class="prov-note"></div>`;

      const note = card.querySelector('.prov-note');
      const keyIn = card.querySelector('.pv-key');
      const urlIn = card.querySelector('.pv-url');
      const say = (t, cls) => { note.textContent = t; note.className = 'prov-note ' + (cls || ''); };

      card.querySelector('.pv-save').addEventListener('click', async () => {
        const body = {};
        if (keyIn.value.trim()) body.api_key = keyIn.value.trim();
        if (urlIn) body.base_url = urlIn.value.trim();
        if (!Object.keys(body).length) { say('Nothing to save — paste a key first.', 'err'); return; }
        say('Saving…');
        const res = await jsend(`/api/settings/providers/${p.id}`, body);
        settings.providers = res.providers; settings.models = res.models;
        keyIn.value = '';
        renderProviders(); renderModels(); refreshModelSelect();
        say('Saved.', 'ok');
      });

      card.querySelector('.pv-test').addEventListener('click', async () => {
        say('Calling the API…');
        const res = await jsend(`/api/settings/providers/${p.id}/test`, {});
        if (res.ok) say(`Works — ${esc(res.model_name)} replied "${esc(res.reply)}".`, 'ok');
        else say(res.error || 'Failed.', 'err');
      });

      const clr = card.querySelector('.pv-clear');
      if (clr) clr.addEventListener('click', async () => {
        const res = await jsend(`/api/settings/providers/${p.id}`, { clear: true });
        settings.providers = res.providers; settings.models = res.models;
        renderProviders(); renderModels(); refreshModelSelect();
      });

      host.appendChild(card);
    });
  }

  function renderModels() {
    const host = $('#modelList');
    host.innerHTML = '';
    (settings.models || []).forEach(m => {
      const row = document.createElement('div');
      row.className = 'model-row' + (m.ready ? '' : ' not-ready');
      row.innerHTML =
        `<input type="checkbox" ${m.enabled ? 'checked' : ''} title="Show in the run selector" />
         <span class="model-name">${esc(m.display)}</span>
         <span class="model-meta">${esc(m.provider_display)} · ${esc(m.model_name)}</span>
         <span class="model-spacer"></span>
         <span class="model-flag ${m.ready ? 'ready' : 'nokey'}">${m.ready ? 'ready' : 'no key'}</span>
         ${m.custom ? '<span class="model-flag">custom</span>' : ''}
         <button class="mini-btn md-edit">Edit</button>
         <button class="mini-btn danger md-del">${m.custom ? 'Delete' : 'Disable'}</button>`;
      row.querySelector('input').addEventListener('change', async e => {
        const res = await jsend(`/api/settings/models/${m.id}/toggle`, { enabled: e.target.checked });
        settings.models = res.models; renderModels(); refreshModelSelect();
      });
      row.querySelector('.md-edit').addEventListener('click', () => fillModelForm(m));
      row.querySelector('.md-del').addEventListener('click', async () => {
        const res = await jsend(`/api/settings/models/${m.id}`, undefined, 'DELETE');
        settings.models = res.models; renderModels(); refreshModelSelect();
      });
      host.appendChild(row);
    });

    [$('#maProvider'), $('#discProvider')].forEach(sel => {
      if (sel.options.length) return;
      sel.innerHTML = settings.providers
        .map(p => `<option value="${esc(p.id)}">${esc(p.display)}</option>`).join('');
    });
  }

  // ---- live discovery: ask the provider what it is actually serving ----
  async function fetchAvailable() {
    const pid = $('#discProvider').value;
    const st = $('#discStatus'), host = $('#discList');
    st.textContent = 'Asking the provider…'; st.className = 'tf-status running';
    host.innerHTML = '';
    const res = await jget(`/api/settings/providers/${pid}/available`);
    if (!res.ok) {
      st.textContent = res.error || 'Could not list models.'; st.className = 'tf-status err';
      return;
    }
    st.textContent = `${res.models.length} chat model(s)`
      + (res.filtered_out ? ` · ${res.filtered_out} non-chat hidden` : '');
    st.className = 'tf-status ok';
    if (!res.models.length) { host.innerHTML = `<span class="disc-empty">Nothing returned.</span>`; return; }
    res.models.forEach(m => {
      const el = document.createElement('span');
      el.className = 'disc-item' + (m.already_added ? ' added' : '');
      el.innerHTML = `<span>${esc(m.id)}</span>`
        + (m.already_added ? '<span class="model-flag">added</span>'
                           : `<button class="disc-add" title="Add this model">＋</button>`);
      const add = el.querySelector('.disc-add');
      if (add) add.addEventListener('click', () => addDiscovered(pid, m.id));
      host.appendChild(el);
    });
  }
  $('#discFetch').addEventListener('click', fetchAvailable);

  async function addDiscovered(pid, modelName) {
    const st = $('#discStatus');
    const res = await jsend('/api/settings/models', {
      id: modelName, display: modelName, provider: pid, model_name: modelName,
      max_rounds: 50, max_tokens: 4096,
    });
    if (res.error) { st.textContent = res.error; st.className = 'tf-status err'; return; }
    settings.models = res.models;
    renderModels(); refreshModelSelect();
    st.textContent = `Added ${modelName} — it is now in the run selector.`;
    st.className = 'tf-status ok';
    fetchAvailable();
  }

  // ==================================================================
  // Skills — SKILL.md files injected into the agent's system prompt
  // ==================================================================
  let skillsInfo = { skills: [] };

  async function loadSkills() {
    skillsInfo = await jget('/api/skills');
    $('#skillAuto').checked = !!skillsInfo.auto;
    $('#skillRoots').innerHTML = (skillsInfo.roots || []).map(r =>
      `<span class="skill-root ${r.exists ? '' : 'missing'}">
         <b>${esc(r.source)}</b> <code>${esc(r.path)}</code>${r.exists ? '' : ' — not mounted'}
       </span>`).join('');
    renderSkills();
  }
  $('#skillAuto').addEventListener('change', async e => {
    skillsInfo = await jsend('/api/skills/auto', { enabled: e.target.checked });
    renderSkills();
  });

  // ---- import / export: bundles move as folders or .zip, same as the ecosystem ----
  $('#skillImportZip').addEventListener('click', () => $('#skillZipFile').click());
  $('#skillZipFile').addEventListener('change', async e => {
    const file = e.target.files[0];
    if (!file) return;
    const st = $('#skillStatus');
    st.textContent = `Importing ${file.name}…`; st.className = 'tf-status running';
    const r = await fetch('/api/skills/import', {
      method: 'POST', headers: { 'Content-Type': 'application/zip' }, body: file,
    });
    const res = await r.json();
    e.target.value = '';
    if (res.error) { st.textContent = res.error; st.className = 'tf-status err'; return; }
    skillsInfo = res; renderSkills();
    st.textContent = `Installed “${res.name}”.`; st.className = 'tf-status ok';
  });
  $('#skillImportGo').addEventListener('click', async () => {
    const path = $('#skillImportPath').value.trim();
    if (!path) return;
    const st = $('#skillStatus');
    const res = await jsend('/api/skills/import', { path });
    if (res.error) { st.textContent = res.error; st.className = 'tf-status err'; return; }
    skillsInfo = res; $('#skillImportPath').value = ''; renderSkills();
    st.textContent = `Installed “${res.name}”.`; st.className = 'tf-status ok';
  });

  function renderSkills() {
    const host = $('#skillList');
    host.innerHTML = '';
    if (!skillsInfo.skills.length) {
      host.innerHTML = `<div class="disc-empty">No skills found.</div>`;
    }
    skillsInfo.skills.forEach(sk => {
      const row = document.createElement('div');
      row.className = 'skill-row';
      row.innerHTML =
        `<input type="checkbox" ${sk.enabled ? 'checked' : ''} title="Make available to the agent" />
         <span class="skill-main">
           <span class="skill-name">${esc(sk.name)}</span>
           <div class="skill-desc">${esc(sk.description || 'no description')}</div>
         </span>
         <span class="skill-tags">
           ${sk.always ? '<span class="skill-tag default">always on</span>' : ''}
           <span class="skill-tag ${sk.source === 'user' ? 'user' : ''}">${sk.source}</span>
           ${sk.resources ? `<span class="skill-tag">${sk.resources} files</span>` : ''}
           <span class="skill-tag" title="Router ~${sk.router_tokens_est} tok, loaded on demand">
             ${sk.always_tokens_est} tok always</span>
         </span>
         <button class="mini-btn sk-edit">${sk.source === 'user' ? 'Edit' : 'View'}</button>
         <a class="mini-btn" href="/api/skills/${encodeURIComponent(sk.name)}/export" download>Export</a>
         ${sk.source === 'user' ? '<button class="mini-btn danger sk-del">Delete</button>'
                                : '<button class="mini-btn sk-fork">Fork</button>'}`;
      const fork = row.querySelector('.sk-fork');
      if (fork) fork.addEventListener('click', async () => {
        skillsInfo = await jsend(`/api/skills/${encodeURIComponent(sk.name)}/fork`, {});
        renderSkills();
        $('#skillStatus').textContent = `Copied into your workspace — your version now wins.`;
        $('#skillStatus').className = 'tf-status ok';
      });
      row.querySelector('input').addEventListener('change', async e => {
        skillsInfo = await jsend(`/api/skills/${encodeURIComponent(sk.name)}/toggle`, { enabled: e.target.checked });
        renderSkills();
      });
      row.querySelector('.sk-edit').addEventListener('click', () => openSkillEditor(sk.name));
      const del = row.querySelector('.sk-del');
      if (del) del.addEventListener('click', async () => {
        skillsInfo = await jsend(`/api/skills/${encodeURIComponent(sk.name)}`, undefined, 'DELETE');
        $('#skillEditor').classList.add('hidden');
        renderSkills();
      });
      host.appendChild(row);
    });
    const total = skillsInfo.enabled_tokens_est || 0;
    $('#skillStatus').textContent = total
      ? `Enabled skills cost ~${total} tokens on every call (routers and references load on demand, on top).`
      : '';
    $('#skillStatus').className = 'tf-status';
  }

  let editingSkill = null;
  async function openSkillEditor(name) {
    const sk = await jget(`/api/skills/${encodeURIComponent(name)}`);
    if (sk.error) return;
    editingSkill = name;
    $('#seTitle').textContent = sk.name;
    $('#seText').value = sk.raw || '';
    $('#sePreview').classList.add('hidden');
    $('#seText').classList.remove('hidden');
    $('#seNote').textContent = sk.source === 'user'
      ? `Your bundle at ${sk.path}. Changes apply on the next run — no restart.`
      : `Read from ${sk.path} (${sk.source}). Saving forks the whole bundle into your workspace, where your copy wins.`;

    // The bundle's other files — this is where the depth lives.
    const files = sk.files || [];
    $('#seFiles').innerHTML = files.length
      ? `<span class="sef-label">bundle:</span> ` + files.slice(0, 40).map(f =>
          `<a class="sef ${f.readable ? '' : 'bin'}" data-p="${esc(f.path)}">${esc(f.path)}</a>`).join('')
        + (files.length > 40 ? `<span class="sef-label">+${files.length - 40} more</span>` : '')
        + ` <a class="sef sef-back" data-p="">SKILL.md</a>`
      : `<span class="sef-label">no other files in this bundle</span>`;
    $$('#seFiles .sef').forEach(a => a.addEventListener('click', async () => {
      const p = a.dataset.p;
      if (!p) {   // back to the router
        $('#sePreview').classList.add('hidden');
        $('#seText').classList.remove('hidden');
        return;
      }
      if (a.classList.contains('bin')) return;
      const res = await jget(`/api/skills/${encodeURIComponent(name)}/file?path=${encodeURIComponent(p)}`);
      $('#sePreview').textContent = res.error ? res.error : res.text;
      $('#sePreview').classList.remove('hidden');
      $('#seText').classList.add('hidden');
    }));

    $('#skillEditor').classList.remove('hidden');
  }
  $('#seClose').addEventListener('click', () => { $('#skillEditor').classList.add('hidden'); editingSkill = null; });
  $('#seSave').addEventListener('click', async () => {
    if (!editingSkill) return;
    const res = await jsend(`/api/skills/${encodeURIComponent(editingSkill)}`, { raw: $('#seText').value }, 'PUT');
    if (res.error) { $('#skillStatus').textContent = res.error; $('#skillStatus').className = 'tf-status err'; return; }
    skillsInfo = res;
    renderSkills();
    $('#skillStatus').textContent = `Saved to ${res.path}`;
    $('#skillStatus').className = 'tf-status ok';
  });
  $('#skillNew').addEventListener('click', async () => {
    const name = $('#skillNewName').value.trim();
    if (!name) return;
    const res = await jsend('/api/skills', { name });
    if (res.error) { $('#skillStatus').textContent = res.error; $('#skillStatus').className = 'tf-status err'; return; }
    skillsInfo = res;
    $('#skillNewName').value = '';
    renderSkills();
    openSkillEditor(res.name);
  });

  function fillModelForm(m) {
    $('#maId').value = m.id;
    $('#maDisplay').value = m.display;
    $('#maProvider').value = m.provider;
    $('#maModelName').value = m.model_name;
    $('#maBaseUrl').value = m.base_url || '';
    $('#maRounds').value = m.max_rounds;
    $('#maTokens').value = m.max_tokens;
    $('#maCostIn').value = (m.cost_per_m || [0, 0])[0];
    $('#maCostOut').value = (m.cost_per_m || [0, 0])[1];
    $('#maStatus').textContent = `Editing “${m.display}”.`;
    $('#maStatus').className = 'tf-status';
  }
  $('#maReset').addEventListener('click', () => {
    ['maId', 'maDisplay', 'maModelName', 'maBaseUrl'].forEach(id => { $('#' + id).value = ''; });
    $('#maRounds').value = 50; $('#maTokens').value = 4096;
    $('#maCostIn').value = 0; $('#maCostOut').value = 0;
    $('#maStatus').textContent = '';
  });
  $('#maSave').addEventListener('click', async () => {
    const st = $('#maStatus');
    const payload = {
      id: $('#maId').value.trim(), display: $('#maDisplay').value.trim(),
      provider: $('#maProvider').value, model_name: $('#maModelName').value.trim(),
      base_url: $('#maBaseUrl').value.trim(), max_rounds: +$('#maRounds').value || 50,
      max_tokens: +$('#maTokens').value || 4096,
      cost_in: +$('#maCostIn').value || 0, cost_out: +$('#maCostOut').value || 0,
    };
    if (!payload.id || !payload.model_name) {
      st.textContent = 'Id and API model name are both required.'; st.className = 'tf-status err'; return;
    }
    const res = await jsend('/api/settings/models', payload);
    if (res.error) { st.textContent = res.error; st.className = 'tf-status err'; return; }
    settings.models = res.models;
    renderModels(); refreshModelSelect();
    st.textContent = `Saved “${payload.display || payload.id}”.`; st.className = 'tf-status ok';
  });

  // ---- memory ----
  async function loadMemory() {
    const mem = await jget('/api/memory');
    $('#memText').value = mem.text || '';
    $('#memEnabled').checked = !!mem.enabled;
  }
  $('#memSave').addEventListener('click', async () => {
    const st = $('#memStatus');
    await jsend('/api/memory', { text: $('#memText').value, enabled: $('#memEnabled').checked }, 'PUT');
    st.textContent = 'Saved — this applies from the next run onwards.'; st.className = 'tf-status ok';
  });

  async function rememberText(text) {
    const line = (text || '').trim();
    if (!line) return;
    await jsend('/api/memory/append', { text: line.slice(0, 500), section: 'Notes' });
    addMsg({ kind: 'system', html: `Remembered — added to your global memory: <i>${esc(line.slice(0, 120))}</i>` });
  }

  // ==================================================================
  // Project journal (durable markdown record)
  // ==================================================================
  const journalModal = $('#journalModal');
  const closeJournal = () => journalModal.classList.add('hidden');
  $('#journalClose').addEventListener('click', closeJournal);

  function mdToHtml(md) {
    const inline = s => esc(s)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
    const out = [];
    let list = null;
    const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
    (md || '').split('\n').forEach(raw => {
      const line = raw.replace(/\s+$/, '');
      if (/^---+$/.test(line)) { closeList(); out.push('<hr/>'); return; }
      if (/^#\s+/.test(line)) { closeList(); out.push(`<h1>${inline(line.slice(2))}</h1>`); return; }
      if (/^##\s+/.test(line)) { closeList(); out.push(`<h2>${inline(line.slice(3))}</h2>`); return; }
      if (/^>\s?/.test(line)) { closeList(); out.push(`<blockquote>${inline(line.replace(/^>\s?/, ''))}</blockquote>`); return; }
      const ol = line.match(/^(\d+)\.\s+(.*)$/);
      if (ol) {
        if (list !== 'ol') { closeList(); out.push('<ol>'); list = 'ol'; }
        out.push(`<li>${inline(ol[2])}</li>`); return;
      }
      if (/^[-*]\s+/.test(line)) {
        if (list !== 'ul') { closeList(); out.push('<ul>'); list = 'ul'; }
        out.push(`<li>${inline(line.replace(/^[-*]\s+/, ''))}</li>`); return;
      }
      closeList();
      if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    });
    closeList();
    return out.join('\n');
  }

  async function openLog() {
    if (!state.project) return;
    const res = await jget(`/api/projects/${state.project.id}/log`);
    $('#journalTitle').textContent = `${state.project.name} — running log`;
    $('#journalPath').textContent = res.path || '';
    const body = $('#journalBody');
    body.innerHTML = res.markdown
      ? mdToHtml(res.markdown)
      : `<div class="journal-empty">No compacted entries yet — one is written after each analysis run.${res.enabled ? '' : ' (Compaction is currently switched off.)'}</div>`;
    journalModal.classList.remove('hidden');
    body.scrollTop = body.scrollHeight;
  }

  // Where the source lives, shown in Help → About. If you fork GISclaw and run
  // it as a service, point this at your own repository — that is what AGPL §13
  // asks for, and it saves your users hunting for it.
  const SOURCE_URL = 'https://github.com/geumjin99/GISclaw';

  function openAbout() {
    $('#journalTitle').textContent = 'About GISclaw';
    $('#journalPath').textContent = 'AGPL-3.0-or-later';
    $('#journalBody').innerHTML = mdToHtml([
      '**GISclaw** — an LLM agent for geospatial analysis.',
      '',
      'Copyright (C) 2026 Han Jinzhen',
      '',
      'This program is free software under the **GNU Affero General Public',
      'License, version 3 or later**. It comes with ABSOLUTELY NO WARRANTY.',
      '',
      '### Source code',
      '',
      `The complete corresponding source is at <${SOURCE_URL}>.`,
      '',
      'If you have modified GISclaw and are running it as a service that other',
      'people use, AGPL section 13 requires you to offer *those* users the source',
      'of your modified version — replace the link above with yours.',
      '',
      '### Before you rely on a result',
      '',
      'GISclaw plans and writes its own analysis code, and that planning comes',
      'from a language model. It can be wrong, and it can be wrong while sounding',
      'confident. Every run leaves the executed code and the full trace on disk so',
      'the work can be checked \u2014 check it, and treat the output as a draft for',
      'expert review rather than a finding. The author accepts no responsibility',
      'for results the model produces or for decisions based on them, and this',
      'software comes with no warranty. The reasoning runs on whichever model',
      'provider you configure, so a description of your data leaves this machine.',
      'See `DISCLAIMER.md`.',
      '',
      '### Commercial licence',
      '',
      'Embedding GISclaw in a proprietary product, or running a closed-source',
      'hosted service, needs a separate commercial licence from the copyright',
      'holder. See `COMMERCIAL-LICENSE.md`, or contact hanjinzhen9@gmail.com.',
      '',
      '### Third-party material',
      '',
      'Bundled example data is from GeoAnalystBench (Apache-2.0, Zhang et al.',
      '2025). Map tiles © CARTO, data © OpenStreetMap contributors (ODbL).',
      'Leaflet (BSD-2-Clause) and Lucide icons (ISC) are bundled. Full notices',
      'are in `THIRD_PARTY_NOTICES.md`.',
    ].join('\n'));
    journalModal.classList.remove('hidden');
    $('#journalBody').scrollTop = 0;
  }

  async function openJournal() {
    if (!state.project) return;
    const res = await jget(`/api/projects/${state.project.id}/journal`);
    $('#journalTitle').textContent = `${state.project.name} — journal`;
    $('#journalPath').textContent = res.path || '';
    const body = $('#journalBody');
    body.innerHTML = res.markdown
      ? mdToHtml(res.markdown)
      : `<div class="journal-empty">Nothing recorded yet — the journal is written when a run finishes.</div>`;
    journalModal.classList.remove('hidden');
    body.scrollTop = body.scrollHeight;
  }

  async function addJournalNote() {
    if (!state.project) return;
    const body = $('#journalBody');
    if (body.querySelector('.note-compose')) return;
    const box = document.createElement('div');
    box.className = 'note-compose';
    box.innerHTML = `<textarea class="mem-text" style="min-height:90px;margin:12px 0 8px;width:100%"
        placeholder="A decision, a client requirement, why an approach was dropped…"></textarea>`;
    const save = document.createElement('button');
    save.className = 'mini-btn primary';
    save.textContent = 'Save note';
    box.appendChild(save);
    body.appendChild(box);
    const ta = box.querySelector('textarea');
    ta.focus();
    save.addEventListener('click', async () => {
      const text = ta.value.trim();
      if (!text) return;
      const res = await jsend(`/api/projects/${state.project.id}/journal/note`, { text });
      body.innerHTML = mdToHtml(res.markdown || '');
      body.scrollTop = body.scrollHeight;
    });
  }
  $('#journalNote').addEventListener('click', addJournalNote);

  // ==================================================================
  // Conversation history — rebuilt from the project's chat.jsonl
  // ==================================================================
  function addDivider(label) {
    const d = document.createElement('div');
    d.className = 'chat-divider';
    d.textContent = label;
    chatScroll.appendChild(d);
  }

  function renderHistoryEntry(e) {
    if (e.role === 'user') {
      const m = addMsg({ kind: 'system', html: `<b>You ·</b> ${esc(e.text || '')}` });
      m.classList.add('msg-history');
      return;
    }
    if (e.role === 'note') {
      const m = addMsg({ kind: 'system', html: `<b>Note ·</b> ${esc(e.text || '')}` });
      m.classList.add('msg-history');
      return;
    }
    const outs = e.outputs || [];
    const chip = `<span class="run-chip ${e.success ? 'ok' : 'bad'}" data-run="${esc(e.run_id || '')}">${esc(e.run_id || 'failed')}</span>`;
    const stats = e.success
      ? `${e.rounds || 0} rounds · ${e.self_corrections || 0} self-corr · ${e.elapsed_s || 0}s`
      : esc((e.error || 'run failed').slice(0, 140));
    const files = outs.length
      ? `<div class="msg-files">${outs.map(f => `<a class="run-file" data-f="${esc(f)}">${esc(f)}</a>`).join(' · ')}</div>`
      : '';
    const m = addMsg({
      kind: e.success ? 'finish' : 'error',
      html: `<div class="msg-run-head">${chip}<span>${stats}</span></div>${files}`,
    });
    m.classList.add('msg-history');
    const c = m.querySelector('.run-chip');
    if (c && e.run_id) c.addEventListener('click', () => replayRun(e.run_id));
    m.querySelectorAll('.run-file').forEach(a => a.addEventListener('click', () => {
      const fn = a.dataset.f, ex = extOf(fn);
      const base = `/api/projects/${state.project.id}`;
      if (IMG_EXT.includes(ex)) openImageView(fn, `${base}/file?where=outputs&path=${encodeURIComponent(fn)}`);
      else if (ex === 'tif' || ex === 'tiff') addRasterOverlay(fn, `${base}/overlay?where=outputs&path=${encodeURIComponent(fn)}`);
      else if (GEO_EXT.includes(ex)) showFileOnMap(fn, 'outputs');
      else openTextFile(fn, 'outputs');
    }));
  }

  async function loadHistory() {
    if (!state.project) return;
    chatScroll.innerHTML = '';
    const res = await jget(`/api/projects/${state.project.id}/chat`);
    const entries = res.entries || [];
    if (!entries.length) {
      addMsg({
        kind: 'system',
        html: `Project <b>${esc(state.project.name)}</b> is open. Describe an analysis and press <b>Run</b>.`,
      });
      return;
    }
    const runs = entries.filter(e => e.role === 'agent').length;
    addDivider(`earlier · ${runs} run${runs === 1 ? '' : 's'}`);
    entries.forEach(renderHistoryEntry);
    addDivider('now');
    chatScroll.scrollTop = chatScroll.scrollHeight;
  }

  async function replayRun(runId) {
    if (!state.project || !runId) return;
    const res = await jget(`/api/projects/${state.project.id}/trace?run=${encodeURIComponent(runId)}`);
    if (res.error) { addMsg({ kind: 'error', text: `No trace stored for ${runId}.` }); return; }
    addDivider(`replay · ${runId}`);
    (res.events || []).forEach(ev => {
      if (ev.thought) addMsg({ kind: 'thought', text: ev.thought }).classList.add('msg-history');
      if (ev.action) addMsg({ kind: 'action', html: `<code>${esc(ev.action)}</code>` }).classList.add('msg-history');
      const obs = ev.observation_full || ev.observation;
      if (obs) addMsg({ kind: ev.success === false ? 'error' : 'observe', text: String(obs).slice(0, 600) }).classList.add('msg-history');
    });
    if (res.code) { resetCode(); appendCode(res.code); switchTab('code'); }
    addDivider('end of replay');
    chatScroll.scrollTop = chatScroll.scrollHeight;
  }

  // ==================================================================
  // Init
  // ==================================================================
  async function refreshModelSelect() {
    state.models = await jget('/api/models');
    const sel = $('#modelSelect');
    if (!state.models.length) {
      sel.innerHTML = `<option value="">No model configured</option>`;
      sel.disabled = true;
      return false;
    }
    const prev = sel.value;
    sel.disabled = false;
    sel.innerHTML = state.models.map(m => `<option value="${m.id}">${esc(m.display)}</option>`).join('');
    if (prev && state.models.some(m => m.id === prev)) sel.value = prev;
    return true;
  }

  async function init() {
    const haveModel = await refreshModelSelect();
    await loadProjects();
    renderLegend();
    if (!haveModel) {
      addMsg({
        kind: 'error',
        html: 'No model is available yet. Open <b>Tools → Settings</b> and paste an API key '
            + '(the key is stored on the server, not in this browser).',
      });
    }
    setTimeout(() => map.invalidateSize(), 60);
  }
  init();
})();
