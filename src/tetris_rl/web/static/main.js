const boardEl = document.getElementById('board');
const startBtn = document.getElementById('startBtn');
const episodesInput = document.getElementById('episodesInput');
const statEpisode = document.getElementById('statEpisode');
const statGlobalStep = document.getElementById('statGlobalStep');
const statReward = document.getElementById('statReward');
const statLoss = document.getElementById('statLoss');
const statLines = document.getElementById('statLines');
const statLinesDelta = document.getElementById('statLinesDelta');
const episodeRewards = document.getElementById('episodeRewards');
const colorBoard = document.getElementById('colorBoard');
const configForm = document.getElementById('rewardConfigForm');
const configStatus = document.getElementById('configStatus');
const epsilonForm = document.getElementById('epsilonForm');
const epsilonStatus = document.getElementById('epsilonStatus');
// Model config controls
const modelConfigForm = document.getElementById('modelConfigForm');
const modelConfigStatus = document.getElementById('modelConfigStatus');
// Resume controls
const resumeForm = document.getElementById('resumeForm');
const resumeStatus = document.getElementById('resumeStatus');
const resumeMeta = document.getElementById('resumeMeta');
// Live exploration runtime elements
const liveEpsilon = document.getElementById('liveEpsilon');
const liveTemperature = document.getElementById('liveTemperature');
const liveTemperatureWrap = document.getElementById('liveTemperatureWrap');
const liveRandomPct = document.getElementById('liveRandomPct');
const liveGreedyPct = document.getElementById('liveGreedyPct');
const liveBoltzPct = document.getElementById('liveBoltzPct');
let epActionCounts = { random:0, greedy:0, boltzmann:0 };

// Line clears chart data
let lcChart; let lcLabels = []; let lcData = [];

function initLineClearsChart(){
  const el = document.getElementById('lineClearsChart');
  if(!el) return;
  lcChart = new Chart(el.getContext('2d'), {
    type: 'bar',
    data: { labels: lcLabels, datasets: [{ label:'Lines', data: lcData, backgroundColor:'#198754' }]},
    options: { responsive:true, animation:false, scales:{ x:{ ticks:{ color:'#888'}}, y:{ ticks:{ color:'#888'}, beginAtZero:true }}, plugins:{ legend:{ labels:{ color:'#ccc'} } } }
  });
}

async function loadConfig(){
  if(!configForm) return;
  const resp = await fetch('/api/reward-config');
  if(!resp.ok) return;
  const cfg = await resp.json();
  [...configForm.elements].forEach(el=>{
    if(el.name && cfg.hasOwnProperty(el.name)){
      el.value = cfg[el.name];
    }
  });
}
loadConfig();

if(configForm){
  configForm.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const data = {};
    [...configForm.elements].forEach(el=>{
      if(el.name){ data[el.name] = el.value; }
    });
    configStatus.textContent = 'Updating...';
    try {
      const resp = await fetch('/api/reward-config', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)});
      const js = await resp.json();
      configStatus.textContent = `Updated ${js.updated} fields`;
      setTimeout(()=>{ configStatus.textContent=''; }, 3000);
    } catch(err){
      configStatus.textContent = 'Error';
    }
  });
}

let rewardChart; let rewardData = []; let rewardLabels = [];
function initChart(){
  const ctx = document.getElementById('rewardChart').getContext('2d');
  rewardChart = new Chart(ctx, {
    type: 'line',
    data: { labels: rewardLabels, datasets: [{ label: 'Step Reward', data: rewardData, borderColor:'#0d6efd', tension:0.2, pointRadius:0 }]},
    options: { responsive:true, animation:false, scales:{ x:{ ticks:{ color:'#888'}}, y:{ ticks:{ color:'#888'}}}, plugins:{ legend:{ labels:{ color:'#ccc'} } } }
  });
}
initChart();
initLineClearsChart();

function updateBoard(snapshot){
  if(!snapshot) return;
  const rows = snapshot.board || [];
  boardEl.textContent = rows.map(r=>r.replace(/\./g,' ').replace(/#/g,'█').replace(/\*/g,'*')).join('\n');
  if(!colorBoard) return;
  if(snapshot.matrix){
    // Clear existing
    while(colorBoard.firstChild) colorBoard.removeChild(colorBoard.firstChild);
    const activeMap = new Set();
    (snapshot.active_cells||[]).forEach(c=>activeMap.add(`${c.x},${c.y}`));
    snapshot.matrix.forEach((row,y)=>{
      row.forEach((color,x)=>{
        const div = document.createElement('div');
        div.className = 'cell' + (color? ' filled':'') + (activeMap.has(`${x},${y}`)? ' active':'');
        if(color){ div.style.background = color; }
        colorBoard.appendChild(div);
      });
    });
  }
}

async function startTraining(){
  const ep = parseInt(episodesInput.value)||5; 
  // pass device if desired later (placeholder)
  await fetch(`/api/train?episodes=${ep}`, { method:'POST'});
}
startBtn.addEventListener('click', startTraining);

function addEpisodeReward(ep, r, lines, avgStruct){
  const li = document.createElement('li');
  const parts = [`Ep ${ep}`, `R=${r.toFixed(2)}`];
  if(typeof lines === 'number') parts.push(`L=${lines}`);
  if(avgStruct){
    const { holes=0, height=0, bumpiness=0 } = avgStruct;
    parts.push(`S(h:${holes.toFixed(3)},H:${height.toFixed(3)},B:${bumpiness.toFixed(3)})`);
  }
  li.textContent = parts.join(' | ');
  episodeRewards.prepend(li);
  while(episodeRewards.children.length>100){ episodeRewards.removeChild(episodeRewards.lastChild); }
}

async function loadTrainingConfig(){
  const resp = await fetch('/api/training-config');
  if(!resp.ok) return;
  const data = await resp.json();
  if(epsilonForm){
    [...epsilonForm.elements].forEach(el=>{
      if(el.name){
        if(data.overrides && data.overrides.hasOwnProperty(el.name)){
          el.value = data.overrides[el.name];
        } else if(data.defaults && data.defaults.hasOwnProperty(el.name)){
          el.value = data.defaults[el.name];
        }
      }
    });
  }
}
loadTrainingConfig();

async function loadModelConfig(){
  if(!modelConfigForm) return;
  try {
    const resp = await fetch('/api/model-config');
    if(!resp.ok) return;
    const data = await resp.json();
    const overrides = data.overrides || {};
    const defaults = data.defaults || {};
    [...modelConfigForm.elements].forEach(el=>{
      if(!el.name) return;
      if(el.name === 'hidden_layers'){
        if(overrides.hidden_layers){
          el.value = overrides.hidden_layers.join(',');
        } else if(data.hidden_layers_csv){
          el.value = data.hidden_layers_csv;
        } else if(defaults.hidden_layers){
          el.value = defaults.hidden_layers.join(',');
        }
      } else if(el.name === 'dueling' || el.name === 'use_layer_norm'){
        const val = (overrides[el.name] !== undefined)? overrides[el.name] : defaults[el.name];
        if(val !== undefined){ el.value = String(val); }
      } else if(el.name === 'dropout'){
        const val = (overrides.dropout !== undefined)? overrides.dropout : defaults.dropout;
        if(val !== undefined){ el.value = val; }
      }
    });
  } catch(err){ /* ignore */ }
}
loadModelConfig();

if(modelConfigForm){
  modelConfigForm.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const payload = {};
    [...modelConfigForm.elements].forEach(el=>{ if(el.name){ payload[el.name] = el.value; }});
    // normalize booleans
    ['dueling','use_layer_norm'].forEach(k=>{ if(payload[k] != null){ payload[k] = (payload[k] === 'true'); }});
    modelConfigStatus.textContent = 'Updating...';
    try {
      const resp = await fetch('/api/model-config', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
      const js = await resp.json();
      modelConfigStatus.textContent = 'Updated';
      setTimeout(()=>{ if(modelConfigStatus.textContent==='Updated') modelConfigStatus.textContent=''; }, 2500);
    } catch(err){ modelConfigStatus.textContent = 'Error'; }
  });
}

if(epsilonForm){
  epsilonForm.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const payload = {};
    [...epsilonForm.elements].forEach(el=>{ if(el.name) payload[el.name] = el.value; });
    epsilonStatus.textContent = 'Updating...';
    try {
      const resp = await fetch('/api/training-config', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
      await resp.json();
      epsilonStatus.textContent = 'Updated';
      setTimeout(()=>{ epsilonStatus.textContent=''; }, 2500);
    } catch(err){ epsilonStatus.textContent='Error'; }
  });
}

if(resumeForm){
  resumeForm.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const fd = new FormData(resumeForm);
    const checkpoint = (fd.get('checkpoint')||'').toString().trim();
    const episodes = parseInt(fd.get('episodes'))||5;
    const device = (fd.get('device')||'auto').toString();
    if(!checkpoint){
      resumeStatus.textContent = 'Checkpoint required';
      return;
    }
    resumeStatus.textContent = 'Resuming...';
    try {
      const qs = new URLSearchParams({ checkpoint, episodes: String(episodes), device });
      const resp = await fetch(`/api/resume?${qs.toString()}`, { method:'POST' });
      const js = await resp.json();
      if(js.status === 'resuming'){
        resumeStatus.textContent = 'Started';
      } else {
        resumeStatus.textContent = js.error ? ('Error: '+js.error) : (js.status||'Unknown');
      }
    } catch(err){
      resumeStatus.textContent = 'Error';
    }
    setTimeout(()=>{ if(resumeStatus.textContent==='Started') resumeStatus.textContent=''; }, 4000);
  });
}

const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onmessage = (event)=>{
  const msg = JSON.parse(event.data);
  if(msg.type === 'step'){
    statEpisode.textContent = msg.episode;
    statGlobalStep.textContent = msg.global_step;
    statReward.textContent = msg.reward.toFixed(2);
    statLoss.textContent = (msg.loss||0).toFixed(4);
    statLines.textContent = msg.lines_cleared_total;
    statLinesDelta.textContent = msg.lines_delta;
  updateBoard(msg.board);
    rewardLabels.push('');
    rewardData.push(msg.reward);
    if(rewardData.length>500){ rewardData.shift(); rewardLabels.shift(); }
    rewardChart.update('none');
    // Live exploration meta
    if(msg.epsilon != null && liveEpsilon){ liveEpsilon.textContent = msg.epsilon.toFixed(3); }
    if(msg.temperature != null && liveTemperature && liveTemperatureWrap){
      liveTemperatureWrap.style.display = 'inline';
      liveTemperature.textContent = msg.temperature.toFixed(3);
    }
    if(msg.action_source){
      if(epActionCounts.hasOwnProperty(msg.action_source)){
        epActionCounts[msg.action_source] += 1;
        const total = Object.values(epActionCounts).reduce((a,b)=>a+b,0) || 1;
        liveRandomPct.textContent = ((epActionCounts.random/total)*100).toFixed(1);
        liveGreedyPct.textContent = ((epActionCounts.greedy/total)*100).toFixed(1);
        liveBoltzPct.textContent = ((epActionCounts.boltzmann/total)*100).toFixed(1);
      }
    }
  } else if(msg.type === 'episode_end'){
    addEpisodeReward(msg.episode, msg.reward, msg.line_clears, msg.avg_structural);
    if(lcChart){
      lcLabels.push(String(msg.episode));
      lcData.push(msg.line_clears||0);
      if(lcData.length>120){ lcData.shift(); lcLabels.shift(); }
      lcChart.update('none');
    }
    // Reset per-episode counts & optionally display final distribution
    epActionCounts = { random:0, greedy:0, boltzmann:0 };
    if(msg.last_epsilon != null && liveEpsilon){ liveEpsilon.textContent = msg.last_epsilon.toFixed(3); }
    if(msg.last_temperature != null && liveTemperature){ liveTemperature.textContent = msg.last_temperature.toFixed(3); }
  } else if(msg.type === 'session_end'){
    console.log('Session ended', msg);
  } else if(msg.type === 'snapshot'){
  updateBoard(msg.board);
  } else if(msg.type === 'config_update'){
    // Refresh form values to reflect authoritative config
    if(configForm){
      const cfg = msg.config || {};
      [...configForm.elements].forEach(el=>{
        if(el.name && cfg.hasOwnProperty(el.name)){
          el.value = cfg[el.name];
        }
      });
      configStatus.textContent = 'Config synced';
      setTimeout(()=>{ if(configStatus.textContent==='Config synced') configStatus.textContent=''; }, 1500);
    }
  } else if(msg.type === 'training_config_update'){
    loadTrainingConfig();
  } else if(msg.type === 'resume_info'){
    if(resumeMeta){
      resumeMeta.innerHTML = `Loaded <strong>${msg.checkpoint}</strong> | prior episodes: ${msg.prior_episodes} | prior reward: ${(+msg.prior_reward).toFixed(2)} | prior lines: ${msg.prior_lines} | global step: ${msg.global_step}`;
    }
    if(resumeStatus){ resumeStatus.textContent = 'Loaded'; setTimeout(()=>{ if(resumeStatus.textContent==='Loaded') resumeStatus.textContent=''; }, 3000); }
  } else if(msg.type === 'resume_error'){
    if(resumeStatus){ resumeStatus.textContent = 'Error: '+msg.error; }
  } else if(msg.type === 'model_config_update'){
    loadModelConfig();
  }
};
ws.onopen = ()=> console.log('WebSocket connected');
ws.onclose = ()=> console.log('WebSocket closed');
