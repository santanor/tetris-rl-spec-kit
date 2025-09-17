const boardEl = document.getElementById('board');
const startBtn = document.getElementById('startBtn');
const episodesInput = document.getElementById('episodesInput');
const broadcastEveryInput = document.getElementById('broadcastEveryInput');
const statEpisode = document.getElementById('statEpisode');
const statGlobalStep = document.getElementById('statGlobalStep');
const statReward = document.getElementById('statReward');
const statLoss = document.getElementById('statLoss');
const statEpsilon = document.getElementById('statEpsilon');
const statLines = document.getElementById('statLines');
const statLinesDelta = document.getElementById('statLinesDelta');
// Recent episodes list (new minimal ticker)
const recentEpisodesEl = document.getElementById('recentEpisodes');
const recentEpisodes = [];
const colorBoard = document.getElementById('colorBoard');
const configForm = document.getElementById('rewardConfigForm');
const configStatus = document.getElementById('configStatus');
// Removed exploration, model config, resume, charts for simplified UI

// (Charts removed)
// --- Charts (reintroduced lightweight observability) ---
let rewardChart, lossChart, linesChart, componentsChart, actionChart, epsilonChart;
let rewardMAChartOverlay = null; // overlay data for reward moving average
let linesMAChartOverlay = null;  // overlay for lines moving average
const PERSIST_KEY = 'tetris_rl_dashboard_state_v1';
const MAX_POINTS = 200; // sliding window for episode-level charts
let lossEma = null;
function initCharts(){
  if(typeof Chart === 'undefined') return; // CDN failed
  const rewardCtx = document.getElementById('rewardChart');
  const lossCtx = document.getElementById('lossChart');
  const linesCtx = document.getElementById('linesChart');
  const compCtx = document.getElementById('componentsChart');
  // Reuse lossChart canvas overlay or create hidden canvas for epsilon? We'll append a tiny canvas under loss.
  let epsilonCanvas = document.getElementById('epsilonChart');
  if(!epsilonCanvas){
    const lossParent = lossCtx.parentElement;
    if(lossParent){
      epsilonCanvas = document.createElement('canvas');
      epsilonCanvas.id = 'epsilonChart';
      epsilonCanvas.style.width = '100%';
      epsilonCanvas.style.height = '40px';
      epsilonCanvas.style.marginTop = '4px';
      lossParent.appendChild(epsilonCanvas);
    }
  }
  const actionCtx = document.getElementById('actionChart');
  const commonScales = { x: { ticks: { display:false }, grid: { display:false } }, y: { ticks: { font:{ size:9 } }, grid: { color:'rgba(255,255,255,0.05)' } } };
  const commonOpts = { responsive:true, animation:false, spanGaps:true, scales: commonScales, plugins:{ legend:{ display:false } } };
  rewardChart = new Chart(rewardCtx,{ type:'line', data:{ labels:[], datasets:[{ label:'Reward', data:[], borderColor:'#4ade80', tension:0.15, pointRadius:0, borderWidth:1.4 }, { label:'Reward MA', data:[], borderColor:'#22c55e', tension:0.15, pointRadius:0, borderWidth:1, borderDash:[3,3], hidden:true }] }, options: Object.assign({}, commonOpts, { plugins:{ legend:{ display:false } } }) });
  lossChart = new Chart(lossCtx,{ type:'line', data:{ labels:[], datasets:[ { label:'Loss', data:[], borderColor:'#0ea5e9', tension:0.15, pointRadius:0, borderWidth:1.2 }, { label:'EMA', data:[], borderColor:'#f59e0b', tension:0.15, pointRadius:0, borderWidth:1.2, borderDash:[4,3] } ] }, options: JSON.parse(JSON.stringify(commonOpts)) });
  linesChart = new Chart(linesCtx,{ type:'bar', data:{ labels:[], datasets:[{ label:'Lines', data:[], backgroundColor:'#6366f1' }, { label:'Lines MA', type:'line', data:[], borderColor:'#818cf8', tension:0.15, pointRadius:0, borderWidth:1, borderDash:[3,3], hidden:true }] }, options: JSON.parse(JSON.stringify(commonOpts)) });
  // Reward components horizontal bar chart with rich tooltip explanations
  componentsChart = new Chart(compCtx,{
    type:'bar',
    data:{
      labels:['line_reward','survival','placement','lock','skyline','top_out'],
      datasets:[{
        label:'Value',
        data:[0,0,0,0,0,0],
        backgroundColor:(ctx)=>{ const v = ctx.raw; return (typeof v === 'number' && v >= 0) ? '#10b981' : '#ef4444'; }
      }]
    },
    options:{
      indexAxis:'y',
      responsive:true,
      animation:false,
      scales:{
        x:{ grid:{ display:false } },
        y:{ grid:{ display:false }, ticks:{ font:{ size:8 } } }
      },
      plugins:{
        legend:{ display:false },
        tooltip:{
          callbacks:{
            label:(ctx)=>{
              try {
                const label = ctx.label;
                const rawVal = ctx.raw ?? 0;
                const raw = (typeof rawVal === 'number') ? rawVal : parseFloat(rawVal) || 0;
                const ds = ctx.chart.data.datasets[0].data;
                const sum = ds.reduce((a,b)=> a + (typeof b==='number'? b : (parseFloat(b)||0)), 0);
                const pct = sum ? (raw / (sum||1) * 100).toFixed(1) : '0.0';
                let expl = '';
                if(label==='line_reward') expl = 'Lines cleared this step (multi-line bonus).';
                else if(label==='survival') expl = 'Alive step bonus (non-terminal).';
                else if(label==='placement') expl = 'Hole avoidance / creation shaping on lock.';
                else if(label==='lock') expl = 'Flat reward each piece lock.';
                else if(label==='skyline') expl = 'Flattening reward or spike penalty based on tallest column spread.';
                else if(label==='top_out') expl = 'Penalty when topping out.';
                return `${label}: ${raw.toFixed(3)} (${pct}% of step total)\n${expl}`;
              } catch(err){
                return ctx.label + ': ' + ctx.raw;
              }
            }
          }
        }
      }
    }
  });
  actionChart = new Chart(actionCtx,{
    type:'bar',
    data:{
      labels:['L','R','CW','CCW','Drop'],
      datasets:[{ label:'% last 500 steps', data:[0,0,0,0,0], backgroundColor:'#3b82f6' }]
    },
    options:{
      responsive:true,
      animation:false,
      scales:{
        x:{ grid:{ display:false }, ticks:{ font:{ size:9 } } },
        y:{ grid:{ color:'rgba(255,255,255,0.05)' }, ticks:{ font:{ size:8 }, callback:(v)=> v + '%' }, suggestedMax:100 }
      },
      plugins:{ legend:{ display:false } }
    }
  });
  if(epsilonCanvas){
    epsilonChart = new Chart(epsilonCanvas,{
      type:'line',
      data:{ labels:[], datasets:[{ label:'Eps', data:[], borderColor:'#f472b6', tension:0.15, pointRadius:0, borderWidth:1 }] },
      options:{ responsive:true, animation:false, scales:{ x:{ display:false }, y:{ display:false, min:0, max:1 } }, plugins:{ legend:{ display:false } } }
    });
  }
}
initCharts();

// Maintain sliding action counts (client-side approx until backend supplies explicit counts)
const actionWindow = [];
const ACTION_WINDOW_SIZE = 500; // steps
function pushAction(a){
  actionWindow.push(a);
  if(actionWindow.length > ACTION_WINDOW_SIZE) actionWindow.shift();
  if(actionChart){
    const counts = [0,0,0,0,0];
    actionWindow.forEach(x=>{ if(x>=0 && x<counts.length) counts[x]++; });
    const total = actionWindow.length || 1;
    actionChart.data.datasets[0].data = counts.map(c=> (c/total*100).toFixed(1));
    actionChart.update('none');
  }
}

// --- Real Network View ---
const netCanvas = document.getElementById('networkCanvas');
const netLegendEl = document.getElementById('netLegend');
const freezeBtn = document.getElementById('freezeNetBtn');
let freezeNetwork = false;
let nodeHitRegions = [];
let netMeta = null; // {input_dim, hidden_layers[], num_actions, dueling}
let netActs = null; // {layers: [...], advantages, value}
let lastQVals = [];
const tooltipEl = (()=>{ let el=document.getElementById('networkTooltip'); if(!el){ el=document.createElement('div'); el.id='networkTooltip'; document.body.appendChild(el);} return el; })();

function setNetData(meta, acts, qvals){
  if(freezeNetwork) return;
  if(meta) netMeta = meta;
  if(acts) netActs = acts;
  if(qvals) lastQVals = qvals.slice();
  // Update legend dynamically
  if(netLegendEl && netMeta){
    const idim = netMeta.input_dim ?? '—';
    const actsCount = netMeta.num_actions ?? (lastQVals? lastQVals.length : '—');
    netLegendEl.textContent = `Input dims: ${idim} | Actions: ${actsCount} (L,R,CW,CCW,Drop) | Color = relative Q (min→max)`;
  }
  drawRealNetwork();
}

function drawRealNetwork(){
  if(!netCanvas || !netCanvas.getContext || !netMeta) return;
  const ctx = netCanvas.getContext('2d');
  const W = netCanvas.width; const H = netCanvas.height;
  ctx.clearRect(0,0,W,H);
  nodeHitRegions = [];
  const padX = 60; const padY = 30;
  const layers = [];
  // Input layer (no activations per component; we just number them)
  const inputDim = netMeta.input_dim || 0;
  layers.push({ type:'input', size: inputDim, activations: null });
  // Hidden layers and their activations (post-ReLU)
  const hiddenSizes = netMeta.hidden_layers || [];
  if(netActs && Array.isArray(netActs.layers)){
    for(let i=0;i<hiddenSizes.length;i++){
      const actVec = netActs.layers[i] || [];
      layers.push({ type:'hidden', index:i, size: hiddenSizes[i], activations: actVec });
    }
  } else {
    hiddenSizes.forEach((sz,i)=> layers.push({ type:'hidden', index:i, size:sz, activations:null }));
  }
  // Dueling heads (if any) logically come after final hidden (value + advantage) then combine to Q outputs
  const dueling = !!netMeta.dueling;
  if(dueling){
    layers.push({ type:'value', size:1, activations: netActs && netActs.value!=null ? [netActs.value] : null });
    layers.push({ type:'advantage', size: netMeta.num_actions, activations: netActs && netActs.advantages ? netActs.advantages : null });
  }
  // Output Q layer
  layers.push({ type:'output', size: netMeta.num_actions, activations: lastQVals });

  const L = layers.length;
  const layerSpacing = (W - 2*padX) / Math.max(1, L-1);
  const maxNodesInAny = Math.max(...layers.map(l=> l.size));
  const nodeRadiusBase = 7;

  function nodeColor(layer){
    if(layer.type==='output') return '#222';
    if(layer.type==='value') return '#845ef7';
    if(layer.type==='advantage') return '#0ea5e9';
    if(layer.type==='hidden') return '#334155';
    if(layer.type==='input') return '#2563eb';
    return '#555';
  }
  function scaleActivation(a){ // simple scaling 0..1 for ReLU outputs
    if(a == null) return 0;
    return 1 - Math.exp(-Math.abs(a)); // smooth squash
  }
  // Precompute output Q range for coloring
  let qMin=0,qMax=0;
  if(lastQVals.length){ qMin = Math.min(...lastQVals); qMax = Math.max(...lastQVals); }
  const actionLabels=['L','R','CW','CCW','Drop'];

  // Layout + draw
  const layerPositions = [];
  for(let li=0; li<L; li++){
    const layer = layers[li];
    const x = padX + li * layerSpacing;
    const count = layer.size;
    const verticalSpace = H - 2*padY;
    const step = count>1 ? verticalSpace/(count-1) : 0;
    const nodes = [];
    for(let ni=0; ni<count; ni++){
      const y = padY + (count>1? ni*step : verticalSpace/2);
      nodes.push({x,y});
    }
    layerPositions.push({ layer, nodes, x });
  }
  // Connections (stylized). We connect sequential logical layers; if dueling, value & advantage both connect to output.
  ctx.lineWidth=1;
  for(let i=0;i<layerPositions.length-1;i++){
    const a = layerPositions[i];
    const b = layerPositions[i+1];
    // Skip connection from value or advantage to each other (they are siblings) – handle separately
    if(a.layer.type==='value' && b.layer.type==='advantage') continue;
    if(a.layer.type==='advantage' && b.layer.type==='output' && dueling){
      // Draw from advantage directly to output
    }
    ctx.strokeStyle='rgba(150,150,150,0.15)';
    a.nodes.forEach(n1=> b.nodes.forEach(n2=>{ ctx.beginPath(); ctx.moveTo(n1.x,n1.y); ctx.lineTo(n2.x,n2.y); ctx.stroke(); }));
  }
  // If dueling: also connect value layer to output explicitly
  if(dueling){
    const valueLayerPos = layerPositions.find(lp=> lp.layer.type==='value');
    const advLayerPos = layerPositions.find(lp=> lp.layer.type==='advantage');
    const outLayerPos = layerPositions.find(lp=> lp.layer.type==='output');
    if(valueLayerPos && outLayerPos){
      ctx.strokeStyle='rgba(200,140,255,0.25)';
      valueLayerPos.nodes.forEach(n1=> outLayerPos.nodes.forEach(n2=>{ ctx.beginPath(); ctx.moveTo(n1.x,n1.y); ctx.lineTo(n2.x,n2.y); ctx.stroke(); }));
    }
    if(advLayerPos && outLayerPos){
      ctx.strokeStyle='rgba(14,165,233,0.2)';
      advLayerPos.nodes.forEach(n1=> outLayerPos.nodes.forEach(n2=>{ ctx.beginPath(); ctx.moveTo(n1.x,n1.y); ctx.lineTo(n2.x,n2.y); ctx.stroke(); }));
    }
  }
  // Draw nodes
  layerPositions.forEach(lp=>{
    const { layer, nodes } = lp;
    const baseColor = nodeColor(layer);
    nodes.forEach((p, idx)=>{
      let r = nodeRadiusBase;
      let fill = baseColor;
      let meta = { type:layer.type, index:idx };
      if(layer.type==='hidden' && layer.activations){
        const act = layer.activations[idx];
        const s = scaleActivation(act);
        r = nodeRadiusBase * (0.6 + 0.6*s);
        const g = Math.min(255, Math.floor(80 + 140*s));
        fill = `rgba(${40+60*s},${g},${120+80*s},1)`;
        meta.activation = act;
        meta.scaled = s;
      } else if(layer.type==='value' && layer.activations){
        const val = layer.activations[0];
        meta.value = val;
      } else if(layer.type==='advantage' && layer.activations){
        const adv = layer.activations[idx];
        meta.advantage = adv;
      } else if(layer.type==='output' && layer.activations){
        const q = layer.activations[idx];
        meta.q = q;
        const t = (q - qMin) / ((qMax - qMin) || 1);
        fill = `hsl(${120*t},70%,40%)`;
        meta.norm = t;
        if(idx === layer.activations.indexOf(qMax)) meta.best = true;
      }
      // Draw circle
      ctx.beginPath();
      ctx.fillStyle=fill;
      ctx.arc(p.x,p.y,r,0,Math.PI*2);
      ctx.fill();
      if(layer.type==='output'){
        ctx.font='9px monospace'; ctx.fillStyle='#ddd'; ctx.textAlign='left'; ctx.textBaseline='middle';
        const label = actionLabels[idx] || ('A'+idx);
        ctx.fillText(`${label} ${(meta.q!==undefined? meta.q.toFixed(2):'')}`, p.x+10, p.y);
      }
      nodeHitRegions.push({x:p.x,y:p.y,r, meta});
      if(meta.best){
        ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.beginPath(); ctx.arc(p.x,p.y,r+2,0,Math.PI*2); ctx.stroke();
      }
    });
  });
  // Gradient labels
  const minSpan = document.getElementById('qMin');
  const maxSpan = document.getElementById('qMax');
  if(minSpan && maxSpan && lastQVals.length){ minSpan.textContent = qMin.toFixed(2); maxSpan.textContent = qMax.toFixed(2); }
}

function handleNetTooltip(e){
  if(!netCanvas) return;
  const rect = netCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left; const y = e.clientY - rect.top;
  const hit = nodeHitRegions.find(n=> ((x-n.x)**2 + (y-n.y)**2) <= n.r*n.r);
  if(hit){
    const m = hit.meta;
    let html='';
    if(m.type==='input') html = `Input feature #${m.index}`;
    else if(m.type==='hidden') html = `Hidden L${m.index+1} node #${m.index}<br>Act: ${(m.activation??0).toFixed? m.activation.toFixed(4):m.activation}`;
    else if(m.type==='value') html = `Value V(s): ${(m.value??0).toFixed(4)}`;
    else if(m.type==='advantage') html = `Advantage A(s,a${m.index}): ${(m.advantage??0).toFixed(4)}`;
    else if(m.type==='output') html = `Action ${m.index}<br>Q: ${m.q.toFixed(4)}<br>Norm: ${(m.norm*100).toFixed(1)}%`;
    tooltipEl.innerHTML = html;
    tooltipEl.style.left = (e.clientX+12)+'px';
    tooltipEl.style.top = (e.clientY+12)+'px';
    tooltipEl.style.display='block';
  } else {
    tooltipEl.style.display='none';
  }
}
if(netCanvas){
  netCanvas.addEventListener('mousemove', handleNetTooltip);
  netCanvas.addEventListener('mouseleave', ()=> tooltipEl.style.display='none');
}
if(freezeBtn){
  freezeBtn.addEventListener('click',()=>{ freezeNetwork=!freezeNetwork; freezeBtn.classList.toggle('active', freezeNetwork); freezeBtn.textContent= freezeNetwork? 'Unfreeze':'Freeze'; if(!freezeNetwork) drawRealNetwork(); });
}

// Remove legacy duplicate freeze listener and old illustrative tooltip code
window.addEventListener('resize', ()=>{ if(netCanvas){ netCanvas.width = netCanvas.clientWidth; netCanvas.height = netCanvas.clientHeight; drawRealNetwork(); }});

// Initial sync of canvas logical size to styled size (so 2d coords match CSS height)
if(netCanvas){
  const resizeToCSS=()=>{
    // Use current rendered size to set drawing buffer
    netCanvas.width = netCanvas.clientWidth;
    netCanvas.height = netCanvas.clientHeight;
    drawRealNetwork();
  };
  setTimeout(resizeToCSS, 50);
  window.addEventListener('resize', resizeToCSS);
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

// (Reward timeline removed)

function updateBoard(snapshot){
  if(!snapshot) return;
  const rows = snapshot.board || snapshot.rows || [];
  if(boardEl){
    boardEl.textContent = rows.map(r=>r.replace(/\./g,' ').replace(/#/g,'█').replace(/\*/g,'*')).join('\n');
  }
  if(!colorBoard) return;
  if(snapshot.matrix){
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
  const qs = new URLSearchParams({ episodes:String(ep)});
  await fetch(`/api/train?${qs.toString()}`, { method:'POST'});
}
startBtn.addEventListener('click', startTraining);
// Dynamic broadcast cadence update
if(broadcastEveryInput){
  broadcastEveryInput.addEventListener('change', async ()=>{
    const v = parseInt(broadcastEveryInput.value)||1;
    broadcastEveryInput.value = String(Math.max(1,v));
    try {
      await fetch('/api/broadcast-config', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ broadcast_every: parseInt(broadcastEveryInput.value) }) });
    } catch(e){ console.warn('Failed to update broadcast cadence'); }
  });
  // Initialize from server
  (async ()=>{ try { const r = await fetch('/api/broadcast-config'); if(r.ok){ const js = await r.json(); if(js.broadcast_every){ broadcastEveryInput.value = js.broadcast_every; } } } catch(e){} })();
}
// Keyboard shortcut: press 's' to start training
window.addEventListener('keydown', (e)=>{
  if(e.key === 's' || e.key === 'S'){
    startTraining();
  }
});

// (Episode reward list removed)

// (Training config exploration removed)

// (Model config removed)

// (Model config form listener removed)

// (Exploration form listener removed)

// (Resume form removed)

const ws = new WebSocket(`ws://${location.host}/ws`);
ws.onmessage = (event)=>{
  const msg = JSON.parse(event.data);
  if(msg.type === 'step'){
    // Multi-env aggregated step
    statGlobalStep.textContent = msg.global_step;
  statLoss.textContent = (msg.loss||0).toFixed(4);
  if(typeof msg.epsilon === 'number' && statEpsilon){ statEpsilon.textContent = msg.epsilon.toFixed(3); }
    // Determine best env board
  if(msg.board){ updateBoard(msg.board); }
  else if(msg.board_snapshot){ updateBoard(msg.board_snapshot); }
    // If server later adds last_action index: msg.last_action
    if(typeof msg.last_action === 'number'){ pushAction(msg.last_action); }
    if(msg.q_values){
      // New real network view uses meta + activations
      setNetData(msg.net_meta, msg.net_activations, msg.q_values);
    }
    // Heights visualization
    if(Array.isArray(msg.heights)){
      const hb = document.getElementById('heightsBar');
      const infoBox = document.getElementById('imbalanceInfo');
      if(hb){
        hb.innerHTML='';
        const heightsArr = msg.heights;
        const maxH = Math.max(1, ...heightsArr);
        const tallest = Math.max(...heightsArr);
        const tallestIdxs = heightsArr.map((h,i)=>h===tallest?i:null).filter(v=>v!==null);
        heightsArr.forEach((h,i)=>{
          const bar = document.createElement('div');
          bar.style.flex='1';
          bar.style.background = tallestIdxs.includes(i)? '#f39c12' : '#0d6efd';
          bar.style.height = ((h / maxH) * 100).toFixed(1) + '%';
          bar.style.position='relative';
          bar.style.borderRadius='2px 2px 0 0';
          bar.title = `col ${i}: ${h}`;
          const lab = document.createElement('span');
          lab.textContent = h;
          lab.style.position='absolute';
          lab.style.bottom='-1.1rem';
          lab.style.left='50%';
          lab.style.transform='translateX(-50%)';
          lab.style.fontSize='0.55rem';
          lab.style.color='#888';
          bar.appendChild(lab);
          hb.appendChild(bar);
        });
        if(infoBox){
          // We only have heights in the step message; penalty components come via
          // reward components stored inside env step info (not directly broadcast per env step here).
          // For now show tallest and columns count; server broadcasts hole column count within reward_components on env step info
          const holeColPenalty = msg.reward_components && msg.reward_components.hole_column_penalty;
          const holeCols = msg.hole_columns != null ? msg.hole_columns : '—';
          infoBox.textContent = `Tallest: ${tallest} | Hole cols: ${holeCols}` + (holeColPenalty ? ` | Hole penalty: ${holeColPenalty.toFixed(3)}` : '');
        }
      }
    }
    // (Scoreboard removed)
  } else if(msg.type === 'episode_end'){
    if(recentEpisodesEl){
      recentEpisodes.push({ ep: msg.episode, reward: msg.reward, lines: msg.line_clears });
      while(recentEpisodes.length > 5) recentEpisodes.shift();
      recentEpisodesEl.innerHTML = recentEpisodes.map(r=>`<li><span>#${r.ep}</span><span>R:${r.reward.toFixed(2)}</span><span>L:${r.lines}</span></li>`).join('');
      statEpisode.textContent = msg.episode;
      statReward.textContent = msg.reward.toFixed(2);
      statLines.textContent = msg.line_clears;
      // If backend supplies aggregate action_dist (random/greedy/boltzmann) we can annotate chart subtitle
      if(msg.action_dist && actionChart){
        // Convert distribution to pseudo last-window percent labels if strategy purely epsilon greedy
        // We cannot map directly to L,R,CW,CCW,Drop, so just modify chart title attribute as quick reference.
        const ad = msg.action_dist;
        const titleEl = actionChart.canvas.parentElement.querySelector('.chart-title');
        if(titleEl){
          titleEl.setAttribute('title', `Action source mix: random ${(ad.random*100).toFixed(1)}%, greedy ${(ad.greedy*100).toFixed(1)}%, boltz ${(ad.boltzmann*100).toFixed(1)}%`);
        }
      }
      // Update episode-level charts
      if(rewardChart){
          rewardChart.data.labels.push(msg.episode);
          const dsReward = rewardChart.data.datasets[0];
          dsReward.data.push(msg.reward);
          // moving average (window 20)
          const maDs = rewardChart.data.datasets[1];
          const win = 20;
          if(dsReward.data.length >= 1){
            const start = Math.max(0, dsReward.data.length - win);
            const slice = dsReward.data.slice(start);
            const avg = slice.reduce((a,b)=>a+b,0)/slice.length;
            maDs.data.push(avg);
          } else { maDs.data.push(msg.reward); }
          if(rewardChart.data.labels.length > MAX_POINTS){
            rewardChart.data.labels.shift();
            rewardChart.data.datasets.forEach(d=>d.data.shift());
          }
          rewardChart.update('none');
        }
        if(linesChart){
          linesChart.data.labels.push(msg.episode);
          const dsLines = linesChart.data.datasets[0];
          dsLines.data.push(msg.line_clears);
          const maDs = linesChart.data.datasets[1];
          const winL = 20;
          if(dsLines.data.length >= 1){
            const start = Math.max(0, dsLines.data.length - winL);
            const slice = dsLines.data.slice(start);
            const avg = slice.reduce((a,b)=>a+b,0)/slice.length;
            maDs.data.push(avg);
          } else { maDs.data.push(msg.line_clears); }
          if(linesChart.data.labels.length > MAX_POINTS){
            linesChart.data.labels.shift();
            linesChart.data.datasets.forEach(d=>d.data.shift());
          }
          linesChart.update('none');
        }
    }
  } else if(msg.type === 'session_end'){
    console.log('Session ended', msg);
  } else if(msg.type === 'snapshot'){
  updateBoard(msg.board || msg.board_snapshot);
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
  } else if(msg.type === 'broadcast_config_update'){
    if(broadcastEveryInput){ broadcastEveryInput.value = msg.broadcast_every; }
  }
  // Per-step streaming updates for loss + reward components
  if(msg.type === 'step'){
    if(lossChart){
      const raw = (msg.loss || 0);
      if(lossEma == null) lossEma = raw; else lossEma = 0.9 * lossEma + 0.1 * raw;
      lossChart.data.labels.push(msg.global_step);
      lossChart.data.datasets[0].data.push(raw);
      lossChart.data.datasets[1].data.push(lossEma);
      if(lossChart.data.labels.length > MAX_POINTS){
        lossChart.data.labels.shift();
        lossChart.data.datasets.forEach(d=>d.data.shift());
      }
      lossChart.update('none');
    }
    if(epsilonChart && typeof msg.epsilon === 'number'){
      epsilonChart.data.labels.push(msg.global_step);
      epsilonChart.data.datasets[0].data.push(msg.epsilon);
      if(epsilonChart.data.labels.length > MAX_POINTS){
        epsilonChart.data.labels.shift();
        epsilonChart.data.datasets[0].data.shift();
      }
      epsilonChart.update('none');
    }
    if(componentsChart && msg.reward_components){
      const order = ['line_reward','survival','top_out'];
      componentsChart.data.datasets[0].data = order.map(k=> msg.reward_components[k] ?? 0);
      componentsChart.update('none');
    }
    persistStateThrottled();
  }
};
ws.onopen = ()=> console.log('WebSocket connected');
ws.onclose = ()=> console.log('WebSocket closed');

// --- Persistence ---
function persistState(){
  const state = {
    rewardEpisodes: rewardChart? rewardChart.data.labels : [],
    rewardValues: rewardChart? rewardChart.data.datasets[0].data : [],
    rewardMA: rewardChart? rewardChart.data.datasets[1].data : [],
    lineEpisodes: linesChart? linesChart.data.labels : [],
    lineValues: linesChart? linesChart.data.datasets[0].data : [],
    lineMA: linesChart? linesChart.data.datasets[1].data : [],
    actions: actionWindow
  };
  try { localStorage.setItem(PERSIST_KEY, JSON.stringify(state)); } catch(e){}
}
let persistTimer = null;
function persistStateThrottled(){ if(persistTimer) return; persistTimer = setTimeout(()=>{ persistState(); persistTimer=null; }, 1000); }
function restoreState(){
  try { const raw = localStorage.getItem(PERSIST_KEY); if(!raw) return; const st = JSON.parse(raw);
    if(rewardChart && st.rewardEpisodes && st.rewardValues){
      rewardChart.data.labels = st.rewardEpisodes; rewardChart.data.datasets[0].data = st.rewardValues; rewardChart.data.datasets[1].data = st.rewardMA || []; rewardChart.update('none');
    }
    if(linesChart && st.lineEpisodes && st.lineValues){
      linesChart.data.labels = st.lineEpisodes; linesChart.data.datasets[0].data = st.lineValues; linesChart.data.datasets[1].data = st.lineMA || []; linesChart.update('none');
    }
    if(Array.isArray(st.actions)){ st.actions.forEach(a=>actionWindow.push(a)); pushAction(-1); } // refresh chart
  } catch(e){}
}
window.addEventListener('beforeunload', persistState);
setTimeout(()=> restoreState(), 500);

// Toggle moving averages via keyboard (m) for reward and lines
window.addEventListener('keydown', (e)=>{
  if(e.key==='m'){ if(rewardChart){ rewardChart.data.datasets[1].hidden = !rewardChart.data.datasets[1].hidden; rewardChart.update('none'); } if(linesChart){ linesChart.data.datasets[1].hidden = !linesChart.data.datasets[1].hidden; linesChart.update('none'); } }
});

// Provide explanation tooltip trigger (press '?')
window.addEventListener('keydown', (e)=>{
  if(e.key==='?'){
    alert('Policy Network card:\n\nHidden nodes show scaled ReLU activations (size & color intensity). Output nodes reflect live Q-values for actions: Left, Right, Rotate CW, Rotate CCW, Drop. Gradient spans min→max Q in current step. Hover any node for details. Freeze to pause updates.');
  }
});
