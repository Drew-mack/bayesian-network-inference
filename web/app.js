const $ = (id) => document.getElementById(id);
let network = null;
let graph = null;
let loadVersion = 0;
let inputVersion = 0;

function options(select, selected) {
  select.replaceChildren(...network.nodes.map(({ id }) => {
    const option = document.createElement('option');
    option.value = option.textContent = id;
    option.selected = id === selected;
    return option;
  }));
}

function describeNode(id) {
  const node = network.nodes.find(item => item.id === id);
  if (!node) return;
  $('selected-node').value = id;
  $('details-title').textContent = `Variable ${id}`;
  $('parents').textContent = node.parents.length ? node.parents.join(', ') : 'None · prior variable';
  $('table-label').textContent = `P(${id} = true)`;
  const header = document.createElement('tr');
  [...node.parents, `P(${id} = true)`].forEach(text => {
    const th = document.createElement('th'); th.textContent = text; header.append(th);
  });
  $('cpt-head').replaceChildren(header);
  $('cpt-body').replaceChildren(...node.rows.map(row => {
    const tr = document.createElement('tr');
    [...node.parents.map(parent => row.values[parent] ? 'True' : 'False'), row.probability.toFixed(3)].forEach(text => {
      const td = document.createElement('td'); td.textContent = text; tr.append(td);
    });
    return tr;
  }));
  if (graph) {
    graph.nodes().removeClass('selected');
    graph.getElementById(id).addClass('selected');
  }
}

function updateHighlights() {
  if (!graph) return;
  graph.nodes().removeClass('query evidence');
  graph.getElementById($('query').value).addClass('query');
  document.querySelectorAll('.evidence-variable').forEach(select => graph.getElementById(select.value).addClass('evidence'));
}

function invalidateResult() {
  inputVersion++;
  $('result').hidden = true;
  $('form-message').textContent = '';
  updateHighlights();
}

function addEvidence() {
  const row = document.createElement('div'); row.className = 'evidence-row';
  const variable = document.createElement('select'); variable.className = 'evidence-variable'; variable.setAttribute('aria-label', 'Evidence variable');
  const used = new Set([...document.querySelectorAll('.evidence-variable')].map(el => el.value));
  const available = network.nodes.find(node => node.id !== $('query').value && !used.has(node.id));
  if (!available) { $('form-message').textContent = 'Every other variable is already evidence.'; return; }
  options(variable, available.id);
  const value = document.createElement('select'); value.className = 'evidence-value'; value.setAttribute('aria-label', 'Evidence value');
  for (const v of ['true', 'false']) { const option = document.createElement('option'); option.value = v; option.textContent = v === 'true' ? 'True' : 'False'; value.append(option); }
  const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'remove-button'; remove.textContent = 'Remove';
  remove.addEventListener('click', () => { row.remove(); invalidateResult(); });
  variable.addEventListener('change', invalidateResult); value.addEventListener('change', invalidateResult);
  row.append(variable, value, remove); $('evidence-rows').append(row); invalidateResult();
}

function graphPositions(nodes) {
  const byId = Object.fromEntries(nodes.map(node => [node.id, node]));
  const levels = {};
  function depth(id) { return levels[id] ?? (levels[id] = byId[id].parents.length ? 1 + Math.max(...byId[id].parents.map(depth)) : 0); }
  nodes.forEach(node => depth(node.id));
  const groups = {};
  nodes.forEach(node => { (groups[levels[node.id]] ??= []).push(node.id); });
  const maxLevel = Math.max(...Object.keys(groups).map(Number));
  const positions = {};
  Object.entries(groups).forEach(([level, ids]) => ids.forEach((id, index) => {
    positions[id] = { x: 130 + (maxLevel ? Number(level) / maxLevel : .5) * 420, y: 100 + (index - (ids.length - 1) / 2) * 125 + 120 };
  }));
  return positions;
}

async function loadNetwork() {
  const version = ++loadVersion;
  $('form-message').textContent = 'Loading network…';
  try {
    const response = await fetch(`/api/model/${$('model').value}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not load network.');
    if (version !== loadVersion) return;
    network = data;
    $('network-key').textContent = $('model').value === 'burglary' ? 'B Burglary  ·  E Earthquake  ·  A Alarm  ·  J John calls  ·  M Mary calls' : 'Each letter represents a variable in the custom network.';
    if (!window.cytoscape) throw new Error('The graph library could not load. Check your internet connection and refresh.');
    if (graph) graph.destroy();
    const positions = graphPositions(network.nodes);
    const elements = network.nodes.flatMap(node => [
      { data: { id: node.id, label: node.id }, position: positions[node.id] },
      ...node.parents.map(parent => ({ data: { id: `${parent}->${node.id}`, source: parent, target: node.id } }))
    ]);
    graph = cytoscape({ container: $('graph'), elements, layout: { name: 'preset', fit: true, padding: 70 }, minZoom: 0.4, maxZoom: 2.5,
      style: [
        { selector: 'node', style: { 'background-color': '#e4edfa', 'border-color': '#8caed7', 'border-width': 1.5, 'width': 47, 'height': 47, 'label': 'data(label)', 'color': '#263b55', 'font-family': 'Inter, sans-serif', 'font-size': 15, 'font-weight': 650, 'text-valign': 'center', 'text-halign': 'center' } },
        { selector: 'node.query', style: { 'background-color': '#d6e8ff', 'border-color': '#3f7fce', 'border-width': 2.5 } },
        { selector: 'node.evidence', style: { 'background-color': '#fae8d5', 'border-color': '#c98743', 'border-width': 2.5 } },
        { selector: 'node.selected', style: { 'overlay-color': '#92b2da', 'overlay-opacity': .16, 'overlay-padding': 10 } },
        { selector: 'edge', style: { 'width': 1.5, 'line-color': '#93a8c1', 'target-arrow-color': '#93a8c1', 'target-arrow-shape': 'triangle', 'arrow-scale': .85, 'curve-style': 'bezier' } }
      ] });
    graph.on('tap', 'node', event => describeNode(event.target.id()));
    $('evidence-rows').replaceChildren();
    const initial = network.nodes.find(node => node.id === ( $('model').value === 'burglary' ? 'B' : 'D'))?.id || network.nodes[0].id;
    options($('query'), initial); options($('selected-node'), initial); describeNode(initial); invalidateResult();
  } catch (error) {
    if (version === loadVersion) $('form-message').textContent = error.message;
  }
}

$('model').addEventListener('change', loadNetwork);
$('selected-node').addEventListener('change', event => describeNode(event.target.value));
$('query').addEventListener('change', invalidateResult);
$('query-value').addEventListener('change', invalidateResult);
$('method').addEventListener('change', invalidateResult);
$('add-evidence').addEventListener('click', addEvidence);
$('reset-view').addEventListener('click', () => { if (graph) graph.fit(undefined, 70); });
$('query-form').addEventListener('submit', async event => {
  event.preventDefault();
  const evidence = {};
  for (const row of document.querySelectorAll('.evidence-row')) {
    const variable = row.querySelector('.evidence-variable').value;
    if (variable === $('query').value) { $('form-message').textContent = 'The query variable cannot also be evidence.'; return; }
    if (Object.hasOwn(evidence, variable)) { $('form-message').textContent = 'Each evidence variable can only appear once.'; return; }
    evidence[variable] = row.querySelector('.evidence-value').value === 'true';
  }
  const request = { model: $('model').value, query: $('query').value, value: $('query-value').value === 'true', method: $('method').value, evidence };
  const version = loadVersion;
  const inputAtRequest = inputVersion;
  $('form-message').textContent = 'Calculating…'; $('result').hidden = true;
  try {
    const response = await fetch('/api/infer', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Unable to calculate.');
    if (version !== loadVersion || inputAtRequest !== inputVersion) return;
    const ev = Object.entries(evidence).map(([key, value]) => `${key} = ${value ? 'True' : 'False'}`).join(', ');
    $('result-expression').textContent = `P(${request.query} = ${request.value ? 'True' : 'False'}${ev ? ' | ' + ev : ''})`;
    $('result-probability').textContent = `${(data.probability * 100).toFixed(2)}%`;
    $('result').hidden = false; $('form-message').textContent = '';
  } catch (error) { $('form-message').textContent = error.message; }
});
loadNetwork();
