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
  const $  = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const t = (k, v) => window.I18N.t(k, v);
  const esc = s => String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

  // Every call to the API carries a header only this page sends; the server
  // refuses state changes without it (see server.py). Done once, here, so no
  // individual request can forget.
  const _fetch = window.fetch.bind(window);
  window.fetch = (url, opts) => {
    if (typeof url === 'string' && url.startsWith('/api/')) {
      opts = Object.assign({}, opts || {});
      opts.headers = Object.assign({}, opts.headers || {}, { 'X-GISclaw': '1' });
    }
    return _fetch(url, opts);
  };

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

  // Agent prose is markdown-ish. Printing it literally is what made a summary
  // arrive as one unbroken wall of text with ** and - still sitting in it. This
  // is deliberately small: headings, lists, tables, quotes, code, emphasis.
  function prose(md) {
    const inline = s => esc(s)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s.,;:)!?]|$)/g, '$1<em>$2</em>')
      .replace(/(^|[\s(])_([^_\n]+)_(?=[\s.,;:)!?]|$)/g, '$1<em>$2</em>');
    const lines = String(md || '').replace(/\r/g, '').split('\n');
    const out = [];
    let para = [], list = null, fence = null;
    const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join(' '))}</p>`); para = []; } };
    const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
    const block = () => { flushPara(); closeList(); };
    const cellsOf = r => r.trim().replace(/^\||\|$/g, '').split('|').map(c => inline(c.trim()));

    for (let i = 0; i < lines.length; i++) {
      const raw = lines[i].replace(/\s+$/, '');
      const line = raw.trim();

      if (fence !== null) {                       // inside a code fence
        if (/^```/.test(line)) { out.push(`<pre class="prose-pre"><code>${esc(fence.join('\n'))}</code></pre>`); fence = null; }
        else fence.push(raw);
        continue;
      }
      if (/^```/.test(line)) { block(); fence = []; continue; }
      if (!line) { block(); continue; }           // blank line ends a paragraph
      if (/^([-*_]\s*){3,}$/.test(line)) { block(); out.push('<hr/>'); continue; }

      const h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) { block(); out.push(`<h${Math.min(h[1].length + 2, 6)}>${inline(h[2])}</h${Math.min(h[1].length + 2, 6)}>`); continue; }

      if (/^>\s?/.test(line)) { block(); out.push(`<blockquote>${inline(line.replace(/^>\s?/, ''))}</blockquote>`); continue; }

      // a table needs its dashed separator on the next line to count as one
      if (line.includes('|') && /^\|?[\s:|-]*-{2,}[\s:|-]*$/.test((lines[i + 1] || '').trim())) {
        block();
        const head = cellsOf(line);
        const rows = [];
        i += 2;
        while (i < lines.length && lines[i].trim() && lines[i].includes('|')) { rows.push(cellsOf(lines[i])); i++; }
        i--;
        out.push('<table class="prose-table"><thead><tr>' + head.map(c => `<th>${c}</th>`).join('')
          + '</tr></thead><tbody>' + rows.map(r => '<tr>' + r.map(c => `<td>${c}</td>`).join('') + '</tr>').join('')
          + '</tbody></table>');
        continue;
      }

      const ol = line.match(/^(\d+)[.)]\s+(.*)$/);
      const ul = line.match(/^[-*•]\s+(.*)$/);
      if (ol || ul) {
        const want = ol ? 'ol' : 'ul';
        flushPara();
        if (list !== want) { closeList(); out.push(`<${want}>`); list = want; }
        out.push(`<li${/^\s{2,}/.test(raw) ? ' class="sub"' : ''}>${inline(ol ? ol[2] : ul[1])}</li>`);
        continue;
      }
      closeList();
      para.push(line);                            // wrapped lines join one paragraph
    }
    if (fence && fence.length) out.push(`<pre class="prose-pre"><code>${esc(fence.join('\n'))}</code></pre>`);
    block();
    return out.join('\n');
  }

  // The finish tool wraps the agent's words in "Task complete / Summary: /
  // Output files". Only the middle was written for a person. Mirrors the same
  // strip on the server, for replayed runs that never went through it.
  const cleanFinish = obs => String(obs || '')
    .replace(/^\s*(?:\u{1F4CB}\s*)?Task complete\s*\n?/u, '')
    .replace(/^\s*Summary:\s*/, '')
    .split(/\n(?:Output files \(\d+\):|\u26A0\uFE0F? ?No output files)/)[0]
    .trim();

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
  // Underneath everything: land, lakes and borders from a built-in file, so
  // the map is never blank — not offline, not behind a blocked provider.
  map.createPane('offline').style.zIndex = 150;
  const NE = 'vendor/naturalearth/';
  const neStyle = {
    ne_110m_land: { color: '#cfd4d8', weight: 0.6, fillColor: '#eeece6', fillOpacity: 1 },
    ne_110m_lakes: { color: '#cfd9e2', weight: 0.5, fillColor: '#dde7ee', fillOpacity: 1 },
    ne_110m_admin_0_boundary_lines_land: { color: '#b9bfc6', weight: 0.6, dashArray: '2 3' },
  };
  Object.entries(neStyle).forEach(([name, style]) => {
    fetch(`${NE}${name}.geojson`).then(r => r.json())
      .then(gj => L.geoJSON(gj, { pane: 'offline', style: () => style, interactive: false }).addTo(map))
      .catch(() => {});
  });

  // The basemap: tiles served by GISclaw (see app/basemap.py).
  let baseLayer = null;
  let basemapInfo = null;
  function applyBasemap(cfg) {
    basemapInfo = cfg;
    if (baseLayer) { map.removeLayer(baseLayer); baseLayer = null; }
    if (cfg && cfg.tiles && cfg.ready) {
      // The map may zoom past what the source draws; beyond maxNativeZoom
      // Leaflet scales the last native level instead of showing nothing.
      baseLayer = L.tileLayer(`/api/basemap/tile/{z}/{x}/{y}?r={r}&v=${cfg.version}`, {
        attribution: esc(cfg.attribution || ''), maxZoom: 22, maxNativeZoom: cfg.max_zoom || 19,
        errorTileUrl: 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
      }).addTo(map);
    }
    renderLegend();
  }
  fetch('/api/settings/basemap').then(r => r.json()).then(applyBasemap).catch(() => applyBasemap(null));

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

  // Layers are keyed by their location (data/x.geojson, outputs/x.geojson)
  // and shown by file name — the same name in both folders is two layers.
  const layerKey = (where, path) => `${where}/${path}`;
  const layerLabel = key => key.split('/').pop();

  async function showFileOnMap(path, where) {
    const key = layerKey(where, path), name = layerLabel(key);
    if (shownLayers[key]) { map.removeLayer(shownLayers[key].layer); delete shownLayers[key]; renderLegend(); return; }
    const url = `/api/projects/${state.project.id}/file?where=${where}&path=${encodeURIComponent(path)}`;
    let gj;
    try { gj = await jget(url); } catch (e) { addMsg({ kind: 'error', text: t('Could not load {name}', { name }) }); return; }
    if (gj.error) { addMsg({ kind: 'error', text: `${name}: ${gj.error}` }); return; }
    if (!gj || gj.type !== 'FeatureCollection' && gj.type !== 'Feature') {
      // Plain JSON that only looked like a layer by its extension.
      openDataView(name, JSON.stringify(gj, null, 2));
      return;
    }
    if (gj._notice) addMsg({ kind: 'system', text: `${name}: ${gj._notice}` });
    const color = LAYER_COLORS[colorIdx++ % LAYER_COLORS.length];
    let layer;
    try { layer = buildGeoLayer(gj, color); layer.addTo(map); }
    catch (e) { addMsg({ kind: 'error', text: t('{name}: not a valid GeoJSON layer.', { name }) }); return; }
    shownLayers[key] = { layer, color, gj, kind: 'vector', fillOpacity: 0.22, visible: true };
    try { map.fitBounds(layer.getBounds(), { padding: [24, 24] }); } catch (e) {}
    renderLegend();
    switchTab('map');
  }

  async function addRasterOverlay(name, overlayUrl, where = 'outputs') {
    const key = layerKey(where, name);
    if (shownLayers[key]) { map.removeLayer(shownLayers[key].layer); delete shownLayers[key]; }
    let pl;
    try { pl = await jget(overlayUrl); } catch (e) { return; }
    if (!pl || pl.error || !pl.bounds) { addMsg({ kind: 'error', text: `${name}: ${pl && pl.error || t('overlay failed')}` }); return; }
    const b = pl.bounds;
    const layer = L.imageOverlay(pl.image, [[b.south, b.west], [b.north, b.east]], { opacity: 0.85 });
    layer.addTo(map);
    shownLayers[key] = { layer, color: '#3f7d58', isRaster: true, kind: 'raster', opacity: 0.85, visible: true };
    try { map.fitBounds(layer.getBounds(), { padding: [24, 24] }); } catch (e) {}
    renderLegend();
    switchTab('map');
  }

  function addResultGeoToMap(url, name, where = 'outputs') {
    const key = layerKey(where, name);
    jget(url).then(gj => {
      if (!gj || gj.error || !gj.type) return;
      if (shownLayers[key]) { map.removeLayer(shownLayers[key].layer); delete shownLayers[key]; }
      const color = LAYER_COLORS[colorIdx++ % LAYER_COLORS.length];
      let layer;
      try { layer = buildGeoLayer(gj, color); layer.addTo(map); } catch (e) { return; }
      shownLayers[key] = { layer, color, gj, kind: 'vector', fillOpacity: 0.22, visible: true };
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
    const bmName = basemapInfo ? (basemapInfo.tiles && basemapInfo.ready ? t(basemapInfo.display) : t('offline reference')) : '…';
    base.innerHTML = `<div class="legend-layer-head"><span class="legend-layer-name">${t('Basemap')}</span><span class="legend-layer-meta">${esc(bmName)}</span></div>`;
    host.appendChild(base);

    const entries = Object.entries(shownLayers);
    entries.forEach(([name, o]) => {
      const kind = o.kind === 'raster' ? 'raster' : 'vector';
      const div = document.createElement('div');
      div.className = 'legend-layer interactive' + (o.visible === false ? ' hidden-layer' : '');
      div.title = name;
      div.innerHTML =
        `<div class="legend-layer-head">`
        + `<span class="lyr-vis" title="${esc(t('Toggle visibility'))}">${svgIcon('eye', 'ctx-ic')}</span>`
        + `<span class="legend-layer-name">${esc(layerLabel(name))}</span>`
        + `<span class="legend-layer-meta">${t(kind)}</span></div>`
        + `<div class="legend-chips"><span class="chip"><i style="background:${o.color}"></i>${t(kind)}</span></div>`;
      div.querySelector('.lyr-vis').addEventListener('click', ev => { ev.stopPropagation(); toggleLayerVisibility(name); });
      div.addEventListener('contextmenu', ev => { ev.preventDefault(); ev.stopPropagation(); openLayerMenu(name, ev.clientX, ev.clientY); });
      host.appendChild(div);
    });
    if (entries.length) {
      const hint = document.createElement('div');
      hint.className = 'legend-hint-rc';
      hint.textContent = t('Right-click a layer for symbology, attribute table…');
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
      { label: t(o.visible === false ? 'Show layer' : 'Hide layer'), icon: 'eye', action: () => toggleLayerVisibility(name) },
      { label: t('Zoom to layer'), icon: 'target', action: () => zoomToLayer(o) },
      // "Fit to data" lives here rather than in a View menu — it is a layer
      // action, and you are already pointing at the layer panel.
      { label: t('Fit to all layers'), icon: 'fit', action: () => $('#btnFit').click() },
      { label: t('Symbology…'), icon: 'palette', action: () => openSymbology(name, x, y) },
      { label: t('Attribute table'), icon: 'table', disabled: o.kind !== 'vector' || !o.gj, action: () => openAttributeTable(name) },
      { sep: true },
      { label: t('Remove layer'), icon: 'trash', danger: true, action: () => removeLayer(name) },
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
      pop.innerHTML = `<div class="symb-h">${esc(layerLabel(name))}</div>`
        + `<div class="symb-row"><span>${t('Opacity')}</span><input type="range" min="0" max="100" value="${Math.round((o.opacity ?? 0.85) * 100)}" id="symbOpacity"></div>`;
    } else {
      pop.innerHTML = `<div class="symb-h">${esc(layerLabel(name))}</div>`
        + `<div class="symb-row"><span>${t('Color')}</span><input type="color" value="${o.color}" id="symbColor"></div>`
        + `<div class="symb-row"><span>${t('Fill opacity')}</span><input type="range" min="0" max="100" value="${Math.round((o.fillOpacity ?? 0.22) * 100)}" id="symbFill"></div>`;
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
    $('#attrTitle').textContent = t('{name} — {n} features', { name: layerLabel(name), n: feats.length })
      + (feats.length > CAP ? ' ' + t('(showing first {n})', { n: CAP }) : '');
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
    $('#attrCount').textContent = t('{n} rows', { n: rows.length });
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
    if (!act) return;
    if (MENU_TRIG[act]) { const b = $(MENU_TRIG[act]); if (!b.disabled) b.click(); return; }
    // Settings is its own menu; each item opens its own pane directly.
    if (act.startsWith('set-')) { openSettings(act.slice(4)); return; }
    if (act === 'journal') openJournal();
    else if (act === 'log') openLog();
    else if (act === 'note') { openJournal().then(addJournalNote); }
    else if (act === 'newthread') newThread();
    else if (act === 'about') openAbout();
    else if (act === 'archived') openArchived();
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
      { label: t('＋ Add data…').replace(/^＋\s*/, ''), icon: 'box', action: async () => {
          if (!isActive) await selectProject(pj.id);
          openBrowse();
        } },
      { label: t('Rename…'), icon: 'table',
        disabled: state.running && isActive,
        action: () => openRename(pj) },
      { sep: true },
      { label: t('Export as zip'), icon: 'box',
        action: () => { window.location = `/api/projects/${pj.id}/export`; } },
      { label: t('Archive…'), icon: 'layers',
        disabled: state.running && isActive,
        action: () => archiveProject(pj) },
      { label: t((isActive && !collapsed.has(pj.id)) ? 'Collapse' : 'Open'), icon: 'layers',
        action: () => { isActive ? toggleProjectOpen(pj.id) : selectProject(pj.id); } },
      { sep: true },
      { label: t('Journal…'), icon: 'table', action: async () => {
          if (!isActive) await selectProject(pj.id);
          openJournal();
        } },
      { sep: true },
      { label: t('Delete project…'), icon: 'trash', danger: true,
        disabled: state.running && isActive,
        action: () => deleteProject(pj) },
    ]);
  }

  async function loadViewerFollow() {
    let lang = '';
    try {
      const s = await jget('/api/settings');
      state.viewerFollow = s.viewer_follow !== false;
      lang = s.language || '';
    } catch (e) { state.viewerFollow = true; }
    setLanguage(window.I18N.detect(lang), false);
  }

  async function loadProjects() {
    allProjects = await jget('/api/projects');
    renderCatalog();
  }

  function fileItem(fn, where) {
    const it = document.createElement('div');
    it.className = 'tree-item';
    it.innerHTML = `${ICONS[iconFor(fn)]}<span class="tree-label">${esc(fn)}</span><span class="tree-meta">${extOf(fn)}</span>`;
    it.addEventListener('contextmenu', e => {
      e.preventDefault();
      e.stopPropagation();
      showContextMenu(e.clientX, e.clientY, [
        { label: t('Open'), icon: 'layers', action: () => it.click() },
        { sep: true },
        { label: t('Delete file…'), icon: 'trash', danger: true,
          disabled: state.running,
          action: () => deleteFile(fn, where) },
      ]);
    });
    it.addEventListener('click', e => {
      e.stopPropagation();
      $$('#catalog .tree-item').forEach(x => x.classList.remove('active'));
      it.classList.add('active');
      const ex = extOf(fn);
      if (IMG_EXT.includes(ex)) openImageView(fn, `/api/projects/${state.project.id}/file?where=${where}&path=${encodeURIComponent(fn)}`);
      else if (ex === 'tif' || ex === 'tiff') addRasterOverlay(fn, `/api/projects/${state.project.id}/overlay?where=${where}&path=${encodeURIComponent(fn)}`, where);
      else if (GEO_EXT.includes(ex)) showFileOnMap(fn, where);
      else openTextFile(fn, where);
    });
    return it;
  }

  function recordItem(fn) {
    const it = document.createElement('div');
    it.className = 'tree-item record-item';
    const label = t({ 'JOURNAL.md': 'Journal (full record)', 'LOG.md': 'Log (compacted)',
                      'chat.jsonl': 'Conversation' }[fn] || fn);
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
      host.innerHTML = `<div class="tree-empty">${t('No projects yet.<br/>Press <b>＋ New</b> to create one.')}</div>`;
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
        dHead.className = 'tree-subhead'; dHead.textContent = t('data');
        children.appendChild(dHead);
        if (state.tree.data.length) state.tree.data.forEach(fn => children.appendChild(fileItem(fn, 'data')));
        else { const e = document.createElement('div'); e.className = 'tree-hint-item'; e.textContent = t('(empty — add data)'); children.appendChild(e); }
        // records — the project's own journal / log / conversation
        if ((state.tree.records || []).length) {
          const rHead = document.createElement('div');
          rHead.className = 'tree-subhead'; rHead.textContent = t('records');
          children.appendChild(rHead);
          state.tree.records.forEach(fn => children.appendChild(recordItem(fn)));
        }
        // outputs
        if (state.tree.outputs.length) {
          const oHead = document.createElement('div');
          oHead.className = 'tree-subhead'; oHead.textContent = t('outputs');
          children.appendChild(oHead);
          state.tree.outputs.forEach(fn => children.appendChild(fileItem(fn, 'outputs')));
        }
      }
      sec.appendChild(children);
      host.appendChild(sec);
    });
  }

  // Opening another project while one runs is allowed: the run lives on the
  // server, not in this view. The stream is dropped here and rejoined by
  // attachActiveRun() on the way back — it replays every event from the start,
  // so nothing of the reasoning is lost by looking away.
  async function selectProject(id) {
    const pj = allProjects.find(p => p.id === id);
    if (!pj) return;
    if (state.project && state.project.id === id) return;
    if (state.running) {
      state.runId++;                    // let the stream reader stand down
      state.running = false; state.stopping = false; stopTimer();
    }
    state.project = { id: pj.id, name: pj.name };
    state.tree = await jget(`/api/projects/${id}/tree`);
    $('#chatTaskTitle').textContent = pj.name;
    $('#regionVal').textContent = pj.name;
    $('#btnAddData').disabled = false;
    $('#btnToolbox').disabled = false;
    $('#startBtn').disabled = false;
    $('#footHint').textContent = t(state.tree.data.length ? 'Describe an analysis and press Run.' : 'Add data to this project to begin.');
    renderCatalog();
    clearMap();
    resetCounters(); resetCode(); resetImageView();
    await loadHistory();          // the conversation survives reloads and restarts
    setTimeout(() => map.invalidateSize(), 30);
    await attachActiveRun();      // and so does a run that was in progress
    refreshRunBanner();           // …or says where it is, if it is elsewhere
  }

  // One run at a time, and it belongs to a project. When that project is not
  // the one on screen, say where it is — otherwise Run is simply dead and the
  // interface never explains why.
  let bannerTimer = null;
  async function refreshRunBanner() {
    const banner = $('#runBanner');
    let active = null;
    try { active = (await jget('/api/run/active')).active; } catch (e) {}
    const elsewhere = active && !active.done && state.project
                      && active.project_id !== state.project.id;
    $('#startBtn').disabled = !state.project || state.running || !!elsewhere;
    if (!elsewhere) {
      banner.classList.add('hidden');
      banner.innerHTML = '';
      if (bannerTimer) { clearInterval(bannerTimer); bannerTimer = null; }
      return;
    }
    const pj = allProjects.find(p => p.id === active.project_id);
    const name = pj ? pj.name : active.project_id;
    banner.classList.remove('hidden');
    banner.innerHTML = '<span class="run-dot"></span>'
      + `<span>${t('Running in <b>{p}</b>', { p: esc(name) })}${active.stopping ? ' · ' + t('Stopping…') : ''}</span>`
      + `<button class="banner-link" type="button">${t('Open it')}</button>`;
    banner.querySelector('.banner-link').addEventListener('click', () => selectProject(active.project_id));
    // That run ends without this page hearing about it, so check back.
    if (!bannerTimer) bannerTimer = setInterval(refreshRunBanner, 4000);
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
  const KIND_SRC = { user: 'You', thought: 'Thought', action: 'Action', observe: 'Observation', error: 'Issue', finish: 'Done', answer: 'Answer', system: 'Info' };
  const kindLabel = k => t(KIND_SRC[k] || k);
  function nowTime() { const d = new Date(); return d.toTimeString().slice(0, 8); }
  function addMsg({ kind, text, html, md }) {
    const m = document.createElement('div');
    m.className = 'msg msg-' + kind;
    // `md` is anything the model wrote: it gets laid out as prose, not dumped.
    const body = html || (md ? prose(md) : esc(text || ''));
    m.innerHTML = `<div class="msg-meta"><span class="agent-tag">${kindLabel(kind)}</span><span class="ts">${nowTime()}</span></div>`
      + `<div class="msg-actions"><button class="msg-act" title="${esc(t('Save this as a standing preference in your global memory'))}">${t('Remember')}</button></div>`
      + `<div class="msg-body${md ? ' prose' : ''}">${body}</div>`;
    m.querySelector('.msg-act').addEventListener('click', () => {
      rememberText(text || m.querySelector('.msg-body').textContent);
    });
    chatScroll.appendChild(m);
    chatScroll.scrollTop = chatScroll.scrollHeight;
    return m;
  }
  // ---- reasoning bubble ---------------------------------------------------
  // Thought / Action / Observation used to be posted as ordinary messages, so a
  // twenty-round run buried its own answer under sixty entries. They go into
  // one live bubble instead: open and streaming while the agent works, folded
  // away when it finishes so the answer is what you see. Click to reopen.
  function openTrace() {
    closeTraceEl(state.traceEl);        // never leave a previous one live
    const wrap = document.createElement('div');
    wrap.className = 'trace open';
    wrap.innerHTML =
      '<div class="trace-head">'
      + '<span class="trace-caret"></span>'
      + '<span class="trace-title">' + t('Working…') + '</span>'
      + '<span class="trace-sub"></span>'
      + '</div><div class="trace-body"></div>';
    wrap.querySelector('.trace-head').addEventListener('click', () => {
      wrap.classList.toggle('open');
    });
    chatScroll.appendChild(wrap);
    chatScroll.scrollTop = chatScroll.scrollHeight;
    state.traceEl = wrap;
    state.traceCount = 0;
    state.traceRounds = 0;
    // Status lines that arrived before the first step belong in here too.
    (state.pendingStatus || []).forEach(t => traceAdd('system', t));
    state.pendingStatus = [];
    return wrap;
  }

  // Progress chatter ("Initializing…", "Loaded skill: x") is not conversation.
  // It goes to the status bar, and into the bubble as part of the record —
  // never as its own message in the thread. Held back until the bubble exists,
  // so a question answered without a run still leaves no empty bubble behind.
  function traceStatus(text) {
    if (!text) return;
    setStatus(text, 'running');
    if (state.traceEl) traceAdd('system', text);
    else if (!state.traceDone) (state.pendingStatus = state.pendingStatus || []).push(text);
  }

  function traceAdd(kind, text, html) {
    const wrap = state.traceEl || openTrace();
    const row = document.createElement('div');
    row.className = 'trace-row trace-' + kind;
    row.innerHTML = `<span class="trace-tag">${kindLabel(kind)}</span>`
      + `<span class="trace-text">${html || esc(text || '')}</span>`;
    wrap.querySelector('.trace-body').appendChild(row);
    state.traceCount++;
    // Count rounds, not rows: the header should agree with "5 rounds" below it,
    // not report the seventeen lines those five rounds happened to produce.
    if (kind === 'action') state.traceRounds = (state.traceRounds || 0) + 1;
    // The one-line summary in the header is what you read while it runs.
    const first = (text || '').split('\n')[0].slice(0, 90);
    if (first) wrap.querySelector('.trace-sub').textContent = first;
    if (wrap.classList.contains('open')) chatScroll.scrollTop = chatScroll.scrollHeight;
    return row;
  }

  function closeTraceEl(wrap, label) {
    if (!wrap) return;
    wrap.classList.remove('open');
    wrap.querySelector('.trace-sub').textContent = '';
    if (label) wrap.querySelector('.trace-title').textContent = label;
  }

  function closeTrace(done) {
    const wrap = state.traceEl;
    state.traceDone = true;
    if (!wrap) return;
    const n = state.traceRounds || state.traceCount || 0;
    // The run's own figure if we have it; otherwise the clock on this page,
    // since the bubble is folded the moment the answer lands, before `done`.
    const el = (done && done.elapsed_s)
      || (state.startedAt ? ((Date.now() - state.startedAt) / 1000).toFixed(1) : 0);
    closeTraceEl(wrap, el ? t('Reasoning · {n} steps · {t}s', { n, t: el }) : t('Reasoning · {n} steps', { n }));
    state.traceEl = null;
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
      $('#timerCount').textContent = t('Elapsed · {t}', { t: `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}` });
    }, 250);
  }
  function stopTimer() { if (state.timer) { clearInterval(state.timer); state.timer = null; } }
  function bumpStep() { state.steps++; $('#stepCount').textContent = t('Steps · {n}', { n: state.steps }); }
  function setSelfCorr(n) { state.selfCorr = n; $('#selfCorrCount').textContent = t('Self-corrections · {n}', { n }); }
  function resetCounters() {
    state.steps = 0; state.selfCorr = 0;
    $('#stepCount').textContent = t('Steps · 0');
    $('#selfCorrCount').textContent = t('Self-corrections · 0');
    $('#timerCount').textContent = t('Elapsed · 00:00');
  }

  // Tabs
  // A tab change the agent caused, as opposed to one the user asked for.
  // Clicking a layer or a file should always take you there; the run pulling
  // the view around while you are reading something else should be optional.
  function followTab(name) {
    if (state.viewerFollow === false) return;
    switchTab(name);
  }

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
    $('#dataRaw').textContent = t(dataDoc.raw ? 'Table' : 'Raw text');
    $('#dataRaw').style.display = dataDoc.rows ? '' : 'none';

    if (!dataDoc.rows || dataDoc.raw) {
      stage.innerHTML = `<pre class="data-text">${esc(dataDoc.text.slice(0, 200000))}</pre>`;
      $('#dataMeta').textContent = t('{n} lines', { n: dataDoc.text.split('\n').length });
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
    $('#dataMeta').textContent = t('{a} of {b} rows × {c} cols', { a: shown.length, b: body.length, c: head.length })
      + (capped.length < shown.length ? ' ' + t('· showing first {n}', { n: capped.length }) : '');
  }
  $('#dataFilter').addEventListener('input', renderDataView);
  $('#dataRaw').addEventListener('click', () => { dataDoc.raw = !dataDoc.raw; renderDataView(); });

  function openImageView(filename, src) {
    state.imageName = filename;     // so a delete can clear the view it is showing
    const img = $('#imageEl');
    img.src = src + (src.includes('?') ? '&' : '?') + 't=' + Date.now();
    $('#imageFilename').textContent = filename;
    $('#imageMeta').textContent = t('Agent output');
    $('#imageTab').removeAttribute('hidden');
    switchTab('image');
  }
  function resetImageView() {
    state.imageName = null;
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
    $('#codeLines').innerHTML = '<span class="code-placeholder">// …</span>';
    $('#codeOutputBody').innerHTML = `<span class="out-muted">${t('Output appears here as the agent runs code.')}</span>`;
    $('#codeRunState').textContent = t('Ready'); $('#codeRunState').className = 'code-run-state';
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
    rs.textContent = t(ok ? 'Executed' : 'Error');
    rs.className = 'code-run-state ' + (ok ? 'success' : 'failed');
  }

  // ==================================================================
  // Run (SSE over POST)
  // ==================================================================
  // Stop / Stopping…, and the second press that stops watching.
  function setStopLabel(stopping) {
    const b = $('#stopBtn');
    b.disabled = false;
    b.querySelector('span').textContent = stopping ? t('Stopping…') : t('Stop');
    b.title = stopping ? t('Already stopping — press again to stop watching this run')
                       : t('Stop after the current step');
  }

  function resetRunUI() {
    // Counters/code/image reset per run — but the conversation is a record now,
    // so it is never wiped here; new turns are appended below the history.
    resetCounters(); resetCode(); resetImageView();
    setStatus(t('Idle'), '');
  }

  async function newThread() {
    if (!state.project || state.running) return;
    await fetch(`/api/projects/${state.project.id}/chat`, { method: 'DELETE' });
    await loadHistory();
    addMsg({ kind: 'system', text: t('Started a new conversation. The previous one is archived in the project folder, and JOURNAL.md still holds every run.') });
  }

  // Per-run UI state, shared by a fresh start and by rejoining a run after a
  // reload. `rid` is this page's own counter; the server's id arrives in the
  // first event and is what Stop is addressed to.
  function beginRunUI(instruction) {
    const rid = ++state.runId;
    state.running = true;
    state.stopping = false;
    state.serverRun = null;
    state.traceEl = null; state.traceCount = 0;
    // Status lines that arrive after a bubble has been folded belong to no
    // bubble at all — without this reset they would surface inside the *next*
    // run's reasoning.
    state.traceDone = false; state.pendingStatus = []; state.traceRounds = 0;
    state.gotSummary = false; state.finishObs = '';
    resetRunUI();
    if (instruction) addMsg({ kind: 'user', text: instruction });
    $('#startBtn').classList.add('hidden'); $('#stopBtn').classList.remove('hidden');
    setStopLabel(false);
    setStatus(t('Running'), 'running'); startTimer();
    return rid;
  }

  async function runAnalysis() {
    if (state.running || !state.project) return;
    const instruction = $('#promptInput').value.trim();
    if (!instruction) { addMsg({ kind: 'error', text: t('Please describe the analysis first.') }); return; }
    if (!state.tree || !state.tree.data.length) { addMsg({ kind: 'error', text: t('This project has no data yet. Add data first.') }); return; }
    if (!$('#modelSelect').value) {
      addMsg({ kind: 'error', html: t('No model configured — open <b>Settings → API keys</b> and add one first.') });
      return;
    }
    const rid = beginRunUI(instruction);
    // Clear the box now — otherwise the text sits there and the next run
    // silently re-sends it.
    $('#promptInput').value = '';

    let resp;
    try {
      resp = await fetch('/api/run', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_id: state.project.id, model: $('#modelSelect').value, instruction, language: window.I18N.lang }),
      });
    } catch (e) { finishRun(rid, false, t('Network error')); return; }
    if (!resp.ok || (resp.headers.get('content-type') || '').includes('application/json')) {
      // Refused before it started: another run is active, or the request was bad.
      let err = t('Could not start ({code}).', { code: resp.status });
      try { const j = await resp.json(); err = j.error || err; } catch (e) {}
      finishRun(rid, false, err);
      if (resp.status === 409) attachActiveRun();
      return;
    }
    await consumeStream(resp, rid);
  }

  // Read one SSE response to its end. Frames are separated by a blank line;
  // the server writes CRLF, so both line endings are accepted.
  async function consumeStream(resp, rid) {
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
    // The stream ended without a `done` event: the server went away mid-run.
    if (rid === state.runId && state.running) finishRun(rid, false, t('The connection to the server was lost.'));
  }

  // A run keeps going on the server when this page reloads. If one is active
  // for this project, rejoin it: the server replays every event from the start.
  async function attachActiveRun() {
    if (!state.project || state.running) return;
    let info;
    try { info = await jget(`/api/run/active?project=${encodeURIComponent(state.project.id)}`); }
    catch (e) { return; }
    const run = info && info.active;
    if (!run || run.done) return;
    const rid = beginRunUI(null);
    state.serverRun = run.run_id;
    state.startedAt = Math.round(run.started * 1000);
    addMsg({ kind: 'system', text: t('Rejoined the run in progress ({id}).', { id: run.run_id }) });
    if (run.stopping) { state.stopping = true; setStopLabel(true); setStatus(t('Stopping…'), 'running'); }
    let resp;
    try { resp = await fetch(`/api/run/${encodeURIComponent(run.run_id)}/stream`); }
    catch (e) { finishRun(rid, false, t('Network error')); return; }
    if (!resp.ok) { finishRun(rid, false, t('Could not rejoin the run ({code}).', { code: resp.status })); return; }
    await consumeStream(resp, rid);
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
    if (ev === 'run') { state.serverRun = msg.run_id; return; }
    if (ev === 'status') { traceStatus(msg.code ? t(msg.code, msg.params || {}) : msg.content); return; }
    if (ev === 'answer') {
      // Answered from the project record — no analysis run was needed.
      addMsg({ kind: msg.mode === 'offtopic' ? 'system' : 'answer', md: msg.content });
      return;
    }
    if (ev === 'summary') {
      // Every run ends in words. This is them.
      state.gotSummary = true;
      closeTrace();                       // fold the reasoning before the answer
      renderSummary(msg.code ? t(msg.code, msg.params || {}) + (msg.code2 ? t(msg.code2, msg.params || {}) : '') : msg.content, msg.outputs);
      return;
    }
    if (ev === 'log') {
      // The durable digest. It repeats the summary above, so it starts folded.
      const m = addMsg({
        kind: 'system',
        html: `<details class="log-fold"><summary>${t('Log entry written to LOG.md')}</summary>`
            + `<div class="log-digest prose">${prose(msg.content)}</div></details>`,
      });
      m.classList.add('msg-history');
      return;
    }
    if (ev === 'error') { addMsg({ kind: 'error', text: msg.content }); finishRun(rid, false); return; }
    if (ev === 'result') {
      addResultFile(msg.filename, msg.url);
      const ex = extOf(msg.filename);
      if (ex === 'tif' || ex === 'tiff') {
        addRasterOverlay(msg.filename, `/api/projects/${state.project.id}/overlay?run=${encodeURIComponent(msg.run_id)}&path=${encodeURIComponent(msg.filename)}`);
      } else if (IMG_EXT.includes(ex)) { if (state.viewerFollow !== false) openImageView(msg.filename, msg.url); }
      else if (GEO_EXT.includes(ex)) addResultGeoToMap(msg.url, msg.filename);
      return;
    }
    if (ev === 'done') { finishRun(rid, msg.success, null, msg); return; }
    if (ev === 'step') {
      // Everything the agent does goes in the bubble, the closing words
      // included — those come back as their own `summary` event, cleaned of the
      // tool's packaging, and that is what gets shown as the answer.
      if (/^finish\b/.test(msg.action || '')) state.finishObs = msg.observation || '';
      if (msg.thought) traceAdd('thought', msg.thought);
      if (msg.action) {
        bumpStep();
        traceAdd('action', null, `<code>${esc(msg.action)}</code>`);
      }
      if (msg.code) { followTab('code'); appendCode(msg.code); }
      if (msg.observation) {
        const obs = msg.observation.slice(0, 600);
        traceAdd(msg.success ? 'observe' : 'error', obs);
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

  // The answer: the agent's own closing words, laid out as prose, with whatever
  // it produced listed underneath as things you can click.
  function renderSummary(text, outputs) {
    const body = (text || '').trim();
    if (!body && !(outputs || []).length) return null;
    const m = addMsg({ kind: 'answer', md: body || t('_No closing summary was written for this run._') });
    if ((outputs || []).length) {
      const row = document.createElement('div');
      row.className = 'ans-files';
      row.innerHTML = `<span class="ans-files-label">${t('Produced')}</span>`
        + outputs.map(f => `<a class="run-file" data-f="${esc(f)}">${esc(f)}</a>`).join('');
      row.querySelectorAll('.run-file').forEach(a => a.addEventListener('click', () => openOutput(a.dataset.f)));
      m.querySelector('.msg-body').appendChild(row);
    }
    chatScroll.scrollTop = chatScroll.scrollHeight;
    return m;
  }

  function openOutput(fn) {
    const ex = extOf(fn), base = `/api/projects/${state.project.id}`;
    if (IMG_EXT.includes(ex)) openImageView(fn, `${base}/file?where=outputs&path=${encodeURIComponent(fn)}`);
    else if (ex === 'tif' || ex === 'tiff') addRasterOverlay(fn, `${base}/overlay?where=outputs&path=${encodeURIComponent(fn)}`);
    else if (GEO_EXT.includes(ex)) showFileOnMap(fn, 'outputs');
    else openTextFile(fn, 'outputs');
  }

  function finishRun(rid, success, errText, done) {
    if (rid !== state.runId) return;
    stopTimer();
    state.running = false;
    state.stopping = false;
    state.serverRun = null;
    $('#startBtn').classList.remove('hidden'); $('#stopBtn').classList.add('hidden');
    setStopLabel(false);
    setStatus(t(success ? 'Done' : (done && done.stopped ? 'Stopped' : 'Failed')), success ? 'done' : '');
    if (done && done.answered) {
      setStatus(t('Idle'), '');
      return;                       // answered from the record; nothing was run
    }
    closeTrace(done);
    // Belt and braces: if no summary arrived, say something rather than nothing.
    if (done && !state.gotSummary) renderSummary(cleanFinish(state.finishObs), done.output_files);
    state.gotSummary = false; state.finishObs = '';
    if (done) {
      setSelfCorr(done.self_corrections || 0);
      addMsg({ kind: 'finish', html: t('Finished in <b>{t}s</b> · {n} rounds · {k} output(s)', { t: done.elapsed_s, n: done.rounds, k: (done.output_files || []).length }) });
    } else if (errText) {
      addMsg({ kind: 'error', text: errText });
    }
    refreshTree();
    refreshRunBanner();
  }

  $('#startBtn').addEventListener('click', runAnalysis);
  // Hotkey: Enter submits the run, Shift+Enter inserts a newline.
  $('#promptInput').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      runAnalysis();
    }
  });
  // The run ends on the server, not here: the model call is cut short, code
  // executing in the sandbox is interrupted, and the run records itself as
  // stopped — which arrives back as `done`. Between the press and that event
  // there is the current step, so the button says so instead of going quiet.
  $('#stopBtn').addEventListener('click', async () => {
    if (!state.running) return;
    if (state.stopping) {
      // Pressed again: stop watching. The server was already told, and the run
      // writes its record either way.
      state.runId++; state.running = false; state.stopping = false; stopTimer();
      $('#startBtn').classList.remove('hidden'); $('#stopBtn').classList.add('hidden');
      setStopLabel(false);
      setStatus(t('Stopped'), '');
      addMsg({ kind: 'system', text: t('Stopped watching this run. It ends on the server after the current step, and the result is kept in the project.') });
      refreshRunBanner();
      return;
    }
    if (!state.serverRun) {
      // No server id yet — nothing to address. Drop the stream.
      finishRun(state.runId, false, t('Stopped.'));
      return;
    }
    state.stopping = true;
    setStopLabel(true);
    setStatus(t('Stopping — finishing the current step…'), 'running');
    try {
      const r = await jpost(`/api/run/${encodeURIComponent(state.serverRun)}/cancel`, {});
      if (r && r.ok === false) finishRun(state.runId, false, t('The run had already ended.'));
    } catch (e) {
      state.stopping = false;
      setStopLabel(false);
      addMsg({ kind: 'error', text: t('Could not reach the server to stop the run.') });
    }
  });
  // "Clear" resets the view, then restores the conversation from disk — the
  // record is the source of truth, not whatever this tab happened to render.
  // A run in progress is not affected; the view simply rejoins it.
  $('#resetBtn').addEventListener('click', async () => {
    state.runId++; state.running = false; stopTimer(); resetRunUI(); clearMap();
    if (state.project) { await loadHistory(); attachActiveRun(); } else chatScroll.innerHTML = '';
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
      count.textContent = t('uploading {i}/{n} — {name}', { i: i + 1, n: files.length, name: f.name });
      const q = `?name=${encodeURIComponent(f.name)}`
              + `&rel=${encodeURIComponent(f.webkitRelativePath || f.name)}`;
      try {
        const r = await fetch(`/api/projects/${state.project.id}/upload${q}`,
                              { method: 'POST', body: f });
        if (!r.ok) throw new Error(r.statusText);
        ok++;
      } catch (e) { bad++; }
    }
    count.textContent = t('{n} uploaded', { n: ok }) + (bad ? t(', {n} failed', { n: bad }) : '');
    closeBrowse();
    await refreshTree();
    addMsg({ kind: bad ? 'error' : 'system',
             text: t('Uploaded {n} file(s) to {p}.', { n: ok, p: state.project.name })
                   + (bad ? t(' {n} failed.', { n: bad }) : '') });
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

  // ---- archive / restore -------------------------------------------------
  // Archiving is a move, not a delete: the folder goes under _archived/ so it
  // leaves the list and the agent's reach with every file intact.
  // ---- confirm dialog, for the things that cannot be undone ----
  const confirmModal = $('#confirmModal');
  let confirmResolve = null;
  function closeConfirm(answer) {
    confirmModal.classList.add('hidden');
    if (confirmResolve) { confirmResolve(answer); confirmResolve = null; }
  }
  $('#confirmCancel').addEventListener('click', () => closeConfirm(false));
  $('#confirmClose').addEventListener('click', () => closeConfirm(false));
  $('#confirmOk').addEventListener('click', () => closeConfirm(true));
  function askConfirm({ title, html, okLabel = t('Delete') }) {
    $('#confirmTitle').textContent = title;
    $('#confirmBody').innerHTML = html;
    $('#confirmOk').querySelector('span').textContent = okLabel;
    confirmModal.classList.remove('hidden');
    return new Promise(res => { confirmResolve = res; });
  }

  async function deleteFile(fn, where) {
    const ok = await askConfirm({
      title: t('Delete file'),
      html: t('<b>{f}</b> will be deleted from <code>{w}/</code>.<br/>This cannot be undone. Earlier runs keep their own copies under <code>runs/</code>.', { f: esc(fn), w: esc(where) })
          + (extOf(fn) === 'shp'
             ? t('<br/><br/>Its <code>.shx</code>/<code>.dbf</code>/<code>.prj</code> siblings go with it — a lone <code>.shp</code> is unreadable.')
             : ''),
    });
    if (!ok) return;
    const res = await jsend(
      `/api/projects/${state.project.id}/file?where=${where}&path=${encodeURIComponent(fn)}`,
      undefined, 'DELETE');
    if (res.error) { addMsg({ kind: 'error', text: res.error }); return; }
    // Drop it from the map and the viewer if it happened to be open.
    const key = layerKey(where, fn);
    if (shownLayers[key]) { map.removeLayer(shownLayers[key].layer); delete shownLayers[key]; renderLegend(); }
    if (state.imageName === fn) resetImageView();
    addMsg({ kind: 'system', text: t('Deleted {list} from {w}/.', { list: res.removed.join(', '), w: where }) });
    await refreshTree();
  }

  async function deleteProject(pj) {
    const tr = (state.project && state.project.id === pj.id && state.tree) || null;
    const counts = tr
      ? t('{n} data file(s), {m} output(s), {r} run(s)', { n: tr.data.length, m: tr.outputs.length, r: (tr.runs || []).length })
      : t('{n} data file(s) and its whole run history', { n: pj.data_count || 0 });
    const ok = await askConfirm({
      title: t('Delete project'),
      html: t('<b>{p}</b> and everything in it — {counts} — will be deleted from disk. This cannot be undone.<br/><br/>Prefer <b>Archive</b> if you might want it back, or <b>Export as zip</b> first.', { p: esc(pj.name), counts }),
      okLabel: t('Delete project'),
    });
    if (!ok) return;
    const res = await jsend(`/api/projects/${pj.id}?confirm=${encodeURIComponent(pj.id)}`,
                            undefined, 'DELETE');
    if (res.error) { addMsg({ kind: 'error', text: res.error }); return; }
    if (state.project && state.project.id === pj.id) {
      state.project = null; state.tree = null;
      clearMap(); resetCode(); resetImageView();
      $('#btnAddData').disabled = true; $('#btnToolbox').disabled = true; $('#startBtn').disabled = true;
    }
    addMsg({ kind: 'system', text: t('Deleted project "{p}".', { p: pj.name }) });
    await loadProjects();
  }

  async function archiveProject(pj) {
    const res = await jpost(`/api/projects/${pj.id}/archive`, {});
    if (res.error) { addMsg({ kind: 'error', text: res.error }); return; }
    addMsg({ kind: 'system',
             text: t('Archived "{p}". Nothing was deleted — bring it back from Project → Archived projects.', { p: pj.name }) });
    if (state.project && state.project.id === pj.id) { state.project = null; clearMap(); }
    await loadProjects();
  }

  const archivedModal = $('#archivedModal');
  const closeArchived = () => archivedModal.classList.add('hidden');
  $('#archivedClose').addEventListener('click', closeArchived);
  $('#archivedDone').addEventListener('click', closeArchived);

  async function openArchived() {
    const list = $('#archivedList');
    list.innerHTML = `<div class="tree-hint-item">${t('Loading…')}</div>`;
    archivedModal.classList.remove('hidden');
    let rows = [];
    try { rows = await jget('/api/archived'); } catch (e) { rows = []; }
    if (!rows.length) {
      list.innerHTML = `<div class="tree-hint-item">${t('Nothing archived yet.')}</div>`;
      return;
    }
    list.innerHTML = '';
    rows.forEach(r => {
      const row = document.createElement('div');
      row.className = 'browse-row';
      row.innerHTML = `<span class="browse-ic">${ICONS.folder}</span>`
        + `<span class="browse-name">${esc(r.name)}</span>`
        + `<span class="browse-size">${t('{n} file(s) · {m} run(s)', { n: r.data_count, m: r.run_count })}</span>`;
      const btn = document.createElement('button');
      btn.className = 'mini-btn'; btn.textContent = t('Restore');
      btn.addEventListener('click', async e => {
        e.stopPropagation();
        const res = await jpost(`/api/archived/${r.id}/restore`, {});
        if (res.error) { addMsg({ kind: 'error', text: res.error }); return; }
        addMsg({ kind: 'system', text: t('Restored "{p}".', { p: r.name }) });
        await loadProjects();
        openArchived();
      });
      row.appendChild(btn);
      const del = document.createElement('button');
      del.className = 'mini-btn danger'; del.textContent = t('Delete');
      del.addEventListener('click', async e => {
        e.stopPropagation();
        const ok = await askConfirm({
          title: t('Delete archived project'),
          html: t('<b>{p}</b> — {n} file(s), {m} run(s) — will be deleted from disk. This cannot be undone.', { p: esc(r.name), n: r.data_count, m: r.run_count }),
          okLabel: t('Delete project'),
        });
        if (!ok) { openArchived(); return; }
        const res = await jsend(`/api/archived/${r.id}?confirm=${encodeURIComponent(r.id)}`,
                                undefined, 'DELETE');
        if (res.error) { addMsg({ kind: 'error', text: res.error }); return; }
        addMsg({ kind: 'system', text: t('Deleted archived project "{p}".', { p: r.name }) });
        openArchived();
      });
      row.appendChild(del);
      list.appendChild(row);
    });
  }

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
    const crumbs = [`<a data-p="">${t('workspace')}</a>`];
    parts.forEach(p => { acc = acc ? acc + '/' + p : p; crumbs.push(`<a data-p="${esc(acc)}">${esc(p)}</a>`); });
    el.innerHTML = crumbs.join(' / ');
    $$('#browsePath a').forEach(a => a.addEventListener('click', () => loadBrowse(a.dataset.p)));
  }
  function humanSize(n) { if (!n) return ''; const u = ['B', 'KB', 'MB', 'GB']; let i = 0; while (n >= 1024 && i < 3) { n /= 1024; i++; } return n.toFixed(i ? 1 : 0) + u[i]; }
  function updateBrowseCount() { $('#browseCount').textContent = t('{n} selected', { n: browseSelected.size }); }

  $('#browseAttach').addEventListener('click', async () => {
    if (!browseSelected.size) { closeBrowse(); return; }
    const res = await jpost(`/api/projects/${state.project.id}/attach`, { paths: [...browseSelected] });
    closeBrowse();
    await refreshTree();
    if (res.attached && res.attached.length) {
      $('#footHint').textContent = t('Describe an analysis and press Run.');
      addMsg({ kind: 'system', text: t('Added {n} file(s) to {p}.', { n: res.attached.length, p: state.project.name }) });
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
    $('#toolForm').innerHTML = `<div class="tool-empty">${t('Select an operation on the left.')}</div>`;
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
        : `<option value="">${t('(no {kind} layers — add data)', { kind: t(inp.kind) })}</option>`;
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
    html += `<div class="tf-row"><label>${t('output name')}</label><input class="tf-output" value="${esc(op.op)}_out"/></div>`;
    html += `<div class="tf-actions"><button class="reset-btn" id="tfInsert"><span>${t('Insert into chat')}</span></button>`
      + `<span class="ca-spacer"></span><button class="primary-btn" id="tfRun"><span>${t('Run')}</span></button></div>`
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
    const st = $('#tfStatus'); st.textContent = t('Running…'); st.className = 'tf-status running';
    let res;
    try { res = await jpost(`/api/projects/${state.project.id}/geoprocess`, payload); }
    catch (e) { st.textContent = t('Network error'); st.className = 'tf-status err'; return; }
    if (res.error || res.ok === false) {
      st.textContent = res.error || t('Failed — check inputs.'); st.className = 'tf-status err';
      return;
    }
    st.textContent = t('Done → {files}', { files: (res.outputs || []).map(o => o.filename).join(', ') }); st.className = 'tf-status ok';
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
    $('#paneLocal').classList.toggle('hidden', name !== 'local');
    $('#paneMap').classList.toggle('hidden', name !== 'map');
    if (name === 'local') loadLocal();
    if (name === 'map') loadBasemapPane();
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
      // A model on your own machine bills nobody, so having no key is its
      // normal state — calling that "no key" would read as something missing.
      const badge = p.key_optional && !p.masked_key
        ? `<span class="prov-badge ${p.configured ? 'ok' : ''}">${t('no key needed')}</span>`
        : p.configured
          ? `<span class="prov-badge ${p.from_env ? 'env' : 'ok'}">${p.from_env ? t('from {env}', { env: esc(p.env_var) }) : t('key saved')}</span>`
          : `<span class="prov-badge">${t('no key')}</span>`;
      card.innerHTML =
        `<div class="prov-head">
           <span class="prov-name">${esc(t(p.display))}</span>${badge}
           ${p.key_optional ? `<a class="prov-docs pv-setup" href="#">${t('set up →')}</a>` : ''}
           ${p.docs && !p.key_optional ? `<a class="prov-docs" href="${esc(p.docs)}" target="_blank" rel="noopener">${t('get a key ↗')}</a>` : ''}
         </div>
         <div class="prov-row">
           <input type="password" class="pv-key" autocomplete="off" spellcheck="false"
                  placeholder="${p.masked_key ? esc(t('{mask}  (stored — type to replace)', { mask: p.masked_key })) : esc(p.key_hint || t('paste your API key'))}" />
           <button class="mini-btn primary pv-save">${t('Save')}</button>
           <button class="mini-btn pv-test">${t('Test')}</button>
           ${p.masked_key && !p.from_env ? `<button class="mini-btn danger pv-clear">${t('Remove')}</button>` : ''}
         </div>
         ${p.needs_base_url ? `<div class="prov-row">
           <input type="text" class="pv-url" spellcheck="false" value="${esc(p.base_url || '')}"
                  placeholder="https://your-endpoint/v1  (OpenAI-compatible base URL)" />
         </div>` : ''}
         ${p.hint ? `<div class="prov-hint">${esc(t(p.hint))}</div>` : ''}
         <div class="prov-note"></div>`;

      const note = card.querySelector('.prov-note');
      const keyIn = card.querySelector('.pv-key');
      const urlIn = card.querySelector('.pv-url');
      const say = (t, cls) => { note.textContent = t; note.className = 'prov-note ' + (cls || ''); };

      card.querySelector('.pv-save').addEventListener('click', async () => {
        const body = {};
        if (keyIn.value.trim()) body.api_key = keyIn.value.trim();
        if (urlIn) body.base_url = urlIn.value.trim();
        if (!Object.keys(body).length) {
          say(t(p.key_optional ? 'Nothing to save — fill in the address first.'
                               : 'Nothing to save — paste a key first.'), 'err');
          return;
        }
        say(t('Saving…'));
        const res = await jsend(`/api/settings/providers/${p.id}`, body);
        settings.providers = res.providers; settings.models = res.models;
        keyIn.value = '';
        renderProviders(); renderModels(); refreshModelSelect();
        say(t('Saved.'), 'ok');
      });

      card.querySelector('.pv-test').addEventListener('click', async () => {
        say(t('Calling the API…'));
        const res = await jsend(`/api/settings/providers/${p.id}/test`, {});
        if (res.ok) say(t('Works — {model} replied "{reply}".', { model: esc(res.model_name), reply: esc(res.reply) }), 'ok');
        else say(res.error || t('Failed.'), 'err');
      });

      const setup = card.querySelector('.pv-setup');
      if (setup) setup.addEventListener('click', e => { e.preventDefault(); switchSetPane('local'); });
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
        `<input type="checkbox" ${m.enabled ? 'checked' : ''} title="${esc(t('Show in the run selector'))}" />
         <span class="model-name">${esc(m.display)}</span>
         <span class="model-meta">${esc(m.provider_display)} · ${esc(m.model_name)}</span>
         <span class="model-spacer"></span>
         <span class="model-flag ${m.ready ? 'ready' : 'nokey'}">${m.ready ? t('ready') : esc(t(m.blocked || 'no key'))}</span>
         ${m.custom ? `<span class="model-flag">${t('custom')}</span>` : ''}
         <button class="mini-btn md-edit">${t('Edit')}</button>
         <button class="mini-btn danger md-del">${t(m.custom ? 'Delete' : 'Disable')}</button>`;
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
    st.textContent = t('Asking the provider…'); st.className = 'tf-status running';
    host.innerHTML = '';
    const res = await jget(`/api/settings/providers/${pid}/available`);
    if (!res.ok) {
      st.textContent = res.error || t('Could not list models.'); st.className = 'tf-status err';
      return;
    }
    st.textContent = t('{n} chat model(s)', { n: res.models.length })
      + (res.filtered_out ? t(' · {n} non-chat hidden', { n: res.filtered_out }) : '');
    st.className = 'tf-status ok';
    if (!res.models.length) { host.innerHTML = `<span class="disc-empty">${t('Nothing returned.')}</span>`; return; }
    res.models.forEach(m => {
      const el = document.createElement('span');
      el.className = 'disc-item' + (m.already_added ? ' added' : '');
      el.innerHTML = `<span>${esc(m.id)}</span>`
        + (m.already_added ? `<span class="model-flag">${t('added')}</span>`
                           : `<button class="disc-add" title="${esc(t('Add this model'))}">＋</button>`);
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
    st.textContent = t('Added {m} — it is now in the run selector.', { m: modelName });
    st.className = 'tf-status ok';
    fetchAvailable();
  }

  // ==================================================================
  // Local models — a server on this machine: find it, see what it serves,
  // add what fits, and learn the one number that decides whether it works.
  // ==================================================================
  let localInfo = null;
  let localProbe = null;

  async function loadLocal() {
    localInfo = await jget('/api/settings/local');
    $('#localMinCtx').textContent = (localInfo.min_context || 8192).toLocaleString();
    const kind = $('#localKind');
    if (!kind.options.length) {
      kind.innerHTML = Object.entries(localInfo.presets)
        .map(([k, p]) => `<option value="${esc(k)}">${esc(t(p.display))}</option>`).join('');
      kind.addEventListener('change', () => {
        const p = localInfo.presets[kind.value] || {};
        if (p.base_url) $('#localUrl').value = p.base_url;
        const docs = $('#localDocs');
        docs.hidden = !p.docs; if (p.docs) docs.href = p.docs;
      });
    }
    // Guess the server from the saved address, so the form opens on it.
    const saved = localInfo.base_url || '';
    const match = Object.entries(localInfo.presets).find(([k, p]) => p.base_url && saved.startsWith(p.base_url.replace(/\/v1\/?$/, '')));
    kind.value = match ? match[0] : (saved ? 'other' : 'ollama');
    $('#localUrl').value = saved || (localInfo.presets[kind.value] || {}).base_url || '';
    kind.dispatchEvent(new Event('change'));
    if (saved) $('#localUrl').value = saved;
    renderLocalModels();
    renderLocalReco();
  }

  function fmtCtx(n) { return n ? n.toLocaleString() : '?'; }

  function renderLocalModels() {
    const host = $('#localModels');
    const added = new Map((localInfo.models || []).map(m => [m.model_name, m]));
    if (!localProbe) {
      host.innerHTML = added.size
        ? [...added.values()].map(m => `<div class="lm-row"><span class="lm-name">${esc(m.model_name)}</span>`
            + `<span class="lm-meta">${t('added · context {n} chars per round', { n: fmtCtx(m.context_chars) })}</span>`
            + `<button class="mini-btn lm-test" data-m="${esc(m.model_name)}">${t('Test')}</button></div>`).join('')
        : `<div class="lm-empty">${t('Press <b>Connect</b> to see what the server is serving.')}</div>`;
    } else {
      const rows = localProbe.models || [];
      const running = new Map((localProbe.running || []).map(r => [r.id, r.context]));
      host.innerHTML = rows.length ? rows.map(m => {
        const ctx = running.get(m.id) || m.context_set;
        const ctxLabel = ctx
          ? t('context {n}', { n: fmtCtx(ctx) }) + (running.has(m.id) ? t(' (loaded)') : '')
          : (m.context_max ? t('context up to {n} · server default applies', { n: fmtCtx(m.context_max) }) : t('context unknown'));
        const bad = ctx && ctx < (localInfo.min_context || 8192);
        const meta = [m.params, m.quant, m.size_gb ? `${m.size_gb} GB` : ''].filter(Boolean).join(' · ');
        const isAdded = added.has(m.id) || m.already_added;
        return `<div class="lm-row">`
          + `<span class="lm-name">${esc(m.id)}</span>`
          + `<span class="lm-meta">${esc(meta)}</span>`
          + `<span class="lm-ctx ${bad ? 'bad' : (ctx ? 'ok' : '')}">${esc(ctxLabel)}</span>`
          + (isAdded ? `<span class="model-flag">${t('added')}</span><button class="mini-btn lm-test" data-m="${esc(m.id)}">${t('Test')}</button>`
                     : `<button class="mini-btn primary lm-add" data-m="${esc(m.id)}" data-ctx="${m.context_chars || ''}">${t('Add this model').split(' ')[0]}</button>`)
          + `</div>`;
      }).join('') : `<div class="lm-empty">${t('The server answered but has no models. Pull one first — see below.')}</div>`;
    }
    host.querySelectorAll('.lm-add').forEach(b => b.addEventListener('click', () => addLocalModel(b.dataset.m, +b.dataset.ctx || 0)));
    host.querySelectorAll('.lm-test').forEach(b => b.addEventListener('click', () => testLocalModel(b.dataset.m, b)));
  }

  function renderLocalReco() {
    $('#localReco').innerHTML = (localInfo.recommended || []).map(r =>
      `<div class="reco-row"><code>ollama pull ${esc(r.name)}</code>`
      + `<span class="reco-needs">${esc(t(r.needs))}</span><span class="reco-note">${esc(t(r.note))}</span>`
      + `<button class="mini-btn reco-copy" data-cmd="ollama pull ${esc(r.name)}">${t('Copy')}</button></div>`).join('');
    $$('#localReco .reco-copy').forEach(b => b.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(b.dataset.cmd); b.textContent = t('Copied'); setTimeout(() => { b.textContent = t('Copy'); }, 1500); } catch (e) {}
    }));
  }

  async function connectLocal() {
    const st = $('#localStatus'), url = $('#localUrl').value.trim();
    if (!url) { st.textContent = t('Enter the server address first.'); st.className = 'tf-status err'; return; }
    st.textContent = t('Connecting…'); st.className = 'tf-status running';
    $('#localAdvice').classList.add('hidden');
    const res = await jget(`/api/settings/local/probe?base_url=${encodeURIComponent(url)}`);
    if (!res.ok) { localProbe = null; st.textContent = res.error || t('No answer.'); st.className = 'tf-status err'; renderLocalModels(); return; }
    localProbe = res;
    const kindName = res.kind === 'ollama' ? `Ollama${res.version ? ' ' + res.version : ''}` : t('an OpenAI-compatible server');
    st.textContent = t('Connected to {kind} — {n} model(s).', { kind: kindName, n: res.models.length }); st.className = 'tf-status ok';
    localInfo = await jget('/api/settings/local');
    renderLocalModels();
    // One warning for the whole listing, when the server's window is known and small.
    const low = (res.running || []).find(r => r.context && r.context < (localInfo.min_context || 8192));
    if (low) showLocalAdvice(t('{m} is loaded with a {ctx}-token context. Raise it in the Ollama app (Settings → Context length) or start the server with OLLAMA_CONTEXT_LENGTH={rec}.', { m: low.id, ctx: low.context.toLocaleString(), rec: localInfo.recommended_context || 16384 }));
  }
  function showLocalAdvice(text) { const a = $('#localAdvice'); a.textContent = text; a.classList.remove('hidden'); }

  async function addLocalModel(name, contextChars) {
    const st = $('#localStatus');
    const res = await jsend('/api/settings/models', {
      id: name, display: name, provider: 'local', model_name: name,
      max_rounds: 35, max_tokens: 2048, timeout: 600, context_chars: contextChars || 0,
    });
    if (res.error) { st.textContent = res.error; st.className = 'tf-status err'; return; }
    settings.models = res.models; renderModels(); refreshModelSelect();
    localInfo = await jget('/api/settings/local');
    renderLocalModels();
    st.textContent = t('Added {m} — it is in the run selector. Press Test to load it once.', { m: name }); st.className = 'tf-status ok';
  }

  async function testLocalModel(name, btn) {
    const st = $('#localStatus');
    btn.disabled = true; st.textContent = t('Calling {m}… (the first call loads the model; this can take a minute)', { m: name }); st.className = 'tf-status running';
    const res = await jsend('/api/settings/providers/local/test', { model_name: name });
    btn.disabled = false;
    if (!res.ok) { st.textContent = res.error || t('Failed.'); st.className = 'tf-status err'; return; }
    let msg = t('Works — {m} replied "{reply}".', { m: name, reply: res.reply });
    if (res.context_length) msg += t(' Loaded with a {ctx}-token context.', { ctx: res.context_length.toLocaleString() });
    st.textContent = msg; st.className = 'tf-status ok';
    if (res.context_advice) showLocalAdvice(res.context_advice); else $('#localAdvice').classList.add('hidden');
    if (localProbe) { localProbe.running = [{ id: name, context: res.context_length }]; renderLocalModels(); }
  }
  $('#localConnect').addEventListener('click', connectLocal);
  $('#localUrl').addEventListener('keydown', e => { if (e.key === 'Enter') connectLocal(); });

  // ==================================================================
  // Map pane — which tiles, with what key, from where
  // ==================================================================
  let bmCfg = null;
  function bmRows() {
    const sel = $('#bmProvider').value;
    const p = (bmCfg.providers || []).find(x => x.id === sel) || {};
    $('#bmKeyRow').style.display = (p.needs_key || sel === 'custom') ? '' : 'none';
    $('#bmUrlRow').style.display = sel === 'custom' ? '' : 'none';
    $('#bmAttrRow').style.display = (sel === 'custom' || sel === 'mbtiles') ? '' : 'none';
    $('#bmFileRow').style.display = sel === 'mbtiles' ? '' : 'none';
    $('#bmHint').textContent = p.hint ? t(p.hint) : '';
    const docs = $('#bmDocs'); docs.hidden = !p.docs; if (p.docs) docs.href = p.docs;
  }
  async function loadBasemapPane() {
    bmCfg = await jget('/api/settings/basemap');
    const sel = $('#bmProvider');
    sel.innerHTML = bmCfg.providers.map(p => `<option value="${esc(p.id)}">${esc(t(p.display))}</option>`).join('');
    sel.value = bmCfg.provider;
    $('#bmKey').value = ''; $('#bmKey').placeholder = bmCfg.masked_key ? t('{mask}  (stored — type to replace)', { mask: bmCfg.masked_key }) : t("paste the provider's key");
    $('#bmUrl').value = bmCfg.url || ''; $('#bmAttr').value = bmCfg.attribution || ''; $('#bmFile').value = bmCfg.mbtiles || '';
    $('#bmCache').checked = bmCfg.cache !== false;
    bmRows();
    const mb = (bmCfg.cache_bytes || 0) / 1048576;
    $('#bmStatus').textContent = (bmCfg.ready ? '' : bmCfg.problem + ' ') + t('Cache: {mb} MB.', { mb: mb.toFixed(1) });
    $('#bmStatus').className = 'tf-status' + (bmCfg.ready ? '' : ' err');
    if (bmCfg.ready) checkBasemap();
  }
  // Fetch one real tile through the server and say what happened — the
  // question "why is my map blank" answered in one line.
  async function checkBasemap() {
    const st = $('#bmStatus');
    let res;
    try { res = await jget('/api/settings/basemap/check'); } catch (e) { return; }
    const cache = st.textContent;
    st.textContent = (res.ok ? t('Source reachable ({ms} ms).', { ms: res.ms || 0 }) : t('Source failed: {detail}', { detail: res.detail || '' })) + ' ' + cache;
    st.className = 'tf-status ' + (res.ok ? 'ok' : 'err');
  }
  $('#bmProvider').addEventListener('change', bmRows);
  $('#bmSave').addEventListener('click', async () => {
    const st = $('#bmStatus');
    const body = { provider: $('#bmProvider').value, url: $('#bmUrl').value, attribution: $('#bmAttr').value,
                   mbtiles: $('#bmFile').value, cache: $('#bmCache').checked };
    if ($('#bmKey').value.trim()) body.key = $('#bmKey').value.trim();
    const res = await jsend('/api/settings/basemap', body);
    if (res.error) { st.textContent = res.error; st.className = 'tf-status err'; return; }
    bmCfg = res; $('#bmKey').value = '';
    applyBasemap(res);
    st.textContent = res.ready ? t('Applied — {name}.', { name: t(res.display) }) : res.problem;
    st.className = 'tf-status ' + (res.ready ? 'ok' : 'err');
    if (res.ready) checkBasemap();
  });
  $('#bmClear').addEventListener('click', async () => {
    await jpost('/api/settings/basemap/clear_cache', {});
    $('#bmStatus').textContent = t('Tile cache cleared.'); $('#bmStatus').className = 'tf-status ok';
  });

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
  $('#viewerFollow').addEventListener('change', async e => {
    state.viewerFollow = e.target.checked;
    await jpost('/api/settings/viewer_follow', { enabled: e.target.checked });
  });

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
    st.textContent = t('Importing {f}…', { f: file.name }); st.className = 'tf-status running';
    const r = await fetch('/api/skills/import', {
      method: 'POST', headers: { 'Content-Type': 'application/zip' }, body: file,
    });
    const res = await r.json();
    e.target.value = '';
    if (res.error) { st.textContent = res.error; st.className = 'tf-status err'; return; }
    skillsInfo = res; renderSkills();
    st.textContent = t('Installed “{n}”.', { n: res.name }); st.className = 'tf-status ok';
  });
  $('#skillImportGo').addEventListener('click', async () => {
    const path = $('#skillImportPath').value.trim();
    if (!path) return;
    const st = $('#skillStatus');
    const res = await jsend('/api/skills/import', { path });
    if (res.error) { st.textContent = res.error; st.className = 'tf-status err'; return; }
    skillsInfo = res; $('#skillImportPath').value = ''; renderSkills();
    st.textContent = t('Installed “{n}”.', { n: res.name }); st.className = 'tf-status ok';
  });

  function renderSkills() {
    const host = $('#skillList');
    host.innerHTML = '';
    if (!skillsInfo.skills.length) {
      host.innerHTML = `<div class="disc-empty">${t('No skills found.')}</div>`;
    }
    skillsInfo.skills.forEach(sk => {
      const row = document.createElement('div');
      row.className = 'skill-row';
      row.innerHTML =
        `<input type="checkbox" ${sk.enabled ? 'checked' : ''} title="${esc(t('Make available to the agent'))}" />
         <span class="skill-main">
           <span class="skill-name">${esc(sk.name)}</span>
           <div class="skill-desc">${esc(sk.description || 'no description')}</div>
         </span>
         <span class="skill-tags">
           ${sk.always ? `<span class="skill-tag default">${t('always on')}</span>` : ''}
           <span class="skill-tag ${sk.source === 'user' ? 'user' : ''}">${t(sk.source)}</span>
           ${sk.resources ? `<span class="skill-tag">${t('{n} files', { n: sk.resources })}</span>` : ''}
           <span class="skill-tag" title="${esc(t('Router ~{n} tok, loaded on demand', { n: sk.router_tokens_est }))}">
             ${t('{n} tok always', { n: sk.always_tokens_est })}</span>
         </span>
         <button class="mini-btn sk-edit">${t(sk.source === 'user' ? 'Edit' : 'View')}</button>
         <a class="mini-btn" href="/api/skills/${encodeURIComponent(sk.name)}/export" download>${t('Export')}</a>
         ${sk.source === 'user' ? `<button class="mini-btn danger sk-del">${t('Delete')}</button>`
                                : `<button class="mini-btn sk-fork">${t('Fork')}</button>`}`;
      const fork = row.querySelector('.sk-fork');
      if (fork) fork.addEventListener('click', async () => {
        skillsInfo = await jsend(`/api/skills/${encodeURIComponent(sk.name)}/fork`, {});
        renderSkills();
        $('#skillStatus').textContent = t('Copied into your workspace — your version now wins.');
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
      ? t('Enabled skills cost ~{n} tokens on every call (routers and references load on demand, on top).', { n: total })
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
      ? t('Your bundle at {p}. Changes apply on the next run — no restart.', { p: sk.path })
      : t('Read from {p} ({s}). Saving forks the whole bundle into your workspace, where your copy wins.', { p: sk.path, s: sk.source });

    // The bundle's other files — this is where the depth lives.
    const files = sk.files || [];
    $('#seFiles').innerHTML = files.length
      ? `<span class="sef-label">${t('bundle:')}</span> ` + files.slice(0, 40).map(f =>
          `<a class="sef ${f.readable ? '' : 'bin'}" data-p="${esc(f.path)}">${esc(f.path)}</a>`).join('')
        + (files.length > 40 ? `<span class="sef-label">${t('+{n} more', { n: files.length - 40 })}</span>` : '')
        + ` <a class="sef sef-back" data-p="">SKILL.md</a>`
      : `<span class="sef-label">${t('no other files in this bundle')}</span>`;
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
    $('#skillStatus').textContent = t('Saved to {p}', { p: res.path });
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
    $('#maStatus').textContent = t('Editing “{n}”.', { n: m.display });
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
      st.textContent = t('Id and API model name are both required.'); st.className = 'tf-status err'; return;
    }
    const res = await jsend('/api/settings/models', payload);
    if (res.error) { st.textContent = res.error; st.className = 'tf-status err'; return; }
    settings.models = res.models;
    renderModels(); refreshModelSelect();
    st.textContent = t('Saved “{n}”.', { n: payload.display || payload.id }); st.className = 'tf-status ok';
  });

  // ---- memory ----
  async function loadMemory() {
    const mem = await jget('/api/memory');
    $('#memText').value = mem.text || '';
    $('#memEnabled').checked = !!mem.enabled;
    $('#viewerFollow').checked = state.viewerFollow !== false;
  }
  $('#memSave').addEventListener('click', async () => {
    const st = $('#memStatus');
    await jsend('/api/memory', { text: $('#memText').value, enabled: $('#memEnabled').checked }, 'PUT');
    st.textContent = t('Saved — this applies from the next run onwards.'); st.className = 'tf-status ok';
  });

  async function rememberText(text) {
    const line = (text || '').trim();
    if (!line) return;
    await jsend('/api/memory/append', { text: line.slice(0, 500), section: 'Notes' });
    addMsg({ kind: 'system', html: t('Remembered — added to your global memory: <i>{t}</i>', { t: esc(line.slice(0, 120)) }) });
  }

  // ==================================================================
  // Project journal (durable markdown record)
  // ==================================================================
  const journalModal = $('#journalModal');
  const closeJournal = () => journalModal.classList.add('hidden');
  $('#journalClose').addEventListener('click', closeJournal);

  const mdToHtml = md => prose(md);

  async function openLog() {
    if (!state.project) return;
    const res = await jget(`/api/projects/${state.project.id}/log`);
    $('#journalTitle').textContent = t('{p} — running log', { p: state.project.name });
    $('#journalPath').textContent = res.path || '';
    const body = $('#journalBody');
    body.innerHTML = res.markdown
      ? mdToHtml(res.markdown)
      : `<div class="journal-empty">${t('No compacted entries yet — one is written after each analysis run.')}${res.enabled ? '' : t(' (Compaction is currently switched off.)')}</div>`;
    journalModal.classList.remove('hidden');
    body.scrollTop = body.scrollHeight;
  }

  // Where the source lives, shown in Help → About. If you fork GISclaw and run
  // it as a service, point this at your own repository — that is what AGPL §13
  // asks for, and it saves your users hunting for it.
  const SOURCE_URL = 'https://github.com/geumjin99/GISclaw';
  let appVersion = '';

  function openAbout() {
    $('#journalTitle').textContent = 'About GISclaw';
    $('#journalPath').textContent = 'AGPL-3.0-or-later';
    $('#journalBody').innerHTML = mdToHtml([
      `**GISclaw**${appVersion ? ' ' + appVersion : ''} — an LLM agent for geospatial analysis.`,
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
    $('#journalTitle').textContent = t('{p} — journal', { p: state.project.name });
    $('#journalPath').textContent = res.path || '';
    const body = $('#journalBody');
    body.innerHTML = res.markdown
      ? mdToHtml(res.markdown)
      : `<div class="journal-empty">${t('Nothing recorded yet — the journal is written when a run finishes.')}</div>`;
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
        placeholder="${esc(t('A decision, a client requirement, why an approach was dropped…'))}"></textarea>`;
    const save = document.createElement('button');
    save.className = 'mini-btn primary';
    save.textContent = t('Save note');
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
      const m = addMsg({ kind: 'user', text: e.text || '' });
      m.classList.add('msg-history');
      return;
    }
    if (e.role === 'note') {
      const m = addMsg({ kind: 'system', html: `<b>${t('Note ·')}</b> ${esc(e.text || '')}` });
      m.classList.add('msg-history');
      return;
    }
    const outs = e.outputs || [];
    const chip = `<span class="run-chip ${e.success ? 'ok' : 'bad'}" data-run="${esc(e.run_id || '')}">${esc(e.run_id || t('failed'))}</span>`;
    const stats = e.kind === 'tool'
      ? esc(e.ask || t('Toolbox'))
      : e.stopped
        ? t('stopped after {n} rounds · {t}s', { n: e.rounds || 0, t: e.elapsed_s || 0 })
        : e.success
          ? t('{n} rounds · {c} self-corr · {t}s', { n: e.rounds || 0, c: e.self_corrections || 0, t: e.elapsed_s || 0 })
          : esc((e.error || t('run failed')).slice(0, 140));
    const files = outs.length
      ? `<div class="msg-files">${outs.map(f => `<a class="run-file" data-f="${esc(f)}">${esc(f)}</a>`).join(' · ')}</div>`
      : '';
    // What was actually concluded, not just how long it took. Without this the
    // reasoning was foldable but the answer disappeared on the next page load.
    const said = (e.final_summary || '').trim();
    const m = addMsg({
      kind: e.success ? 'finish' : 'error',
      html: `<div class="msg-run-head">${chip}<span>${stats}</span></div>`
          + (said ? `<div class="msg-summary prose">${prose(said)}</div>` : '')
          + files,
    });
    m.classList.add('msg-history');
    const c = m.querySelector('.run-chip');
    if (c && e.run_id && e.kind !== 'tool') c.addEventListener('click', () => replayRun(e.run_id));
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
        html: t('Project <b>{p}</b> is open. Describe an analysis and press <b>Run</b>.', { p: esc(state.project.name) }),
      });
      return;
    }
    const runs = entries.filter(e => e.role === 'agent').length;
    addDivider(t('earlier · {n} runs', { n: runs }));
    entries.forEach(renderHistoryEntry);
    addDivider(t('now'));
    chatScroll.scrollTop = chatScroll.scrollHeight;
  }

  async function replayRun(runId) {
    if (!state.project || !runId) return;
    const res = await jget(`/api/projects/${state.project.id}/trace?run=${encodeURIComponent(runId)}`);
    if (res.error) { addMsg({ kind: 'error', text: t('No trace stored for {id}.', { id: runId }) }); return; }
    addDivider(t('replay · {id}', { id: runId }));
    (res.events || []).forEach(ev => {
      if (ev.thought) traceAdd('thought', ev.thought);
      if (ev.action) traceAdd('action', null, `<code>${esc(ev.action)}</code>`);
      const obs = ev.observation_full || ev.observation;
      if (obs) {
        if (/^finish\b/.test(ev.action || '')) renderSummary(cleanFinish(String(obs)), null);
        else traceAdd(ev.success === false ? 'error' : 'observe', String(obs).slice(0, 600));
      }
    });
    if (state.traceEl) {
      closeTraceEl(state.traceEl, t('Reasoning · {n} steps', { n: state.traceCount || 0 }));
      state.traceEl = null;
    }
    if (res.code) { resetCode(); appendCode(res.code); switchTab('code'); }
    addDivider(t('end of replay'));
    chatScroll.scrollTop = chatScroll.scrollHeight;
  }

  // ==================================================================
  // Init
  // ==================================================================
  async function refreshModelSelect() {
    state.models = await jget('/api/models');
    const sel = $('#modelSelect');
    if (!state.models.length) {
      sel.innerHTML = `<option value="">${t('No model configured')}</option>`;
      sel.disabled = true;
      return false;
    }
    const prev = sel.value;
    sel.disabled = false;
    sel.innerHTML = state.models.map(m => `<option value="${m.id}">${esc(m.display)}</option>`).join('');
    if (prev && state.models.some(m => m.id === prev)) sel.value = prev;
    return true;
  }

  // Language: the server remembers the choice (shared by every browser on
  // this machine); localStorage and the browser's own language fill in first.
  function setLanguage(code, persist) {
    window.I18N.set(code);
    $$('#menubar [data-lang]').forEach(it => it.classList.toggle('active', it.dataset.lang === window.I18N.lang));
    if (persist) jpost('/api/settings/ui', { language: window.I18N.lang }).catch(() => {});
  }
  $$('#menubar [data-lang]').forEach(it => it.addEventListener('click', e => {
    e.stopPropagation();
    $$('#menubar .menu').forEach(m => m.classList.remove('open'));
    setLanguage(it.dataset.lang, true);
  }));
  document.addEventListener('gisclaw:lang', () => {
    // Everything drawn by script, redrawn in the new language.
    renderLegend(); renderCatalog(); resetCounters();
    if (!state.running) setStatus(t('Idle'), '');
    if (state.project) {
      $('#footHint').textContent = t(state.tree && state.tree.data.length ? 'Describe an analysis and press Run.' : 'Add data to this project to begin.');
      // The title carries data-i18n, so the language switch had just replaced
      // the open project's name with "No project selected".
      $('#chatTaskTitle').textContent = state.project.name;
    }
    if (!state.models.length) $('#modelSelect').innerHTML = `<option value="">${t('No model configured')}</option>`;
    if (state.project && !state.running) loadHistory();
    if (state.running) setStopLabel(state.stopping);
    refreshRunBanner();
  });

  async function init() {
    fetch('/api/version').then(r => r.json()).then(j => { appVersion = j.version || ''; }).catch(() => {});
    await loadViewerFollow();
    const haveModel = await refreshModelSelect();
    await loadProjects();
    renderLegend();
    if (!haveModel) {
      addMsg({
        kind: 'error',
        html: t('No model is available yet. Open <b>Settings → API keys</b> and paste an API key (the key is stored on the server, not in this browser), or point the <b>Local model</b> entry at a server of your own and fetch what it is serving.'),
      });
    }
    refreshRunBanner();
    setTimeout(() => map.invalidateSize(), 60);
  }
  init();
})();
