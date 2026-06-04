import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

let DATA = 'data/dog';   // current dataset path (switchable)
const COLORS = { q:0x30d158, k:0xff9f0a, v:0xbf5af2 };   // Q green · K orange · V purple
const DIM = 0x2b2f3a;

const state = { view:'attention', layer:0, head:0, selQ:null };
let meta, attn, N, H, L, T0, NPT;
// label a token index: prefix tokens are CLS + registers, the rest are patches
function tokLabel(i){ return i<NPT ? (i===0 ? 'CLS' : 'REG'+i) : 'patch '+(i-NPT); }
const BG = new THREE.Color(0x12141d);   // dim-toward colour (keeps hue, just fades)

// ---------- load a dataset ----------
async function fetchDataset(name){
  DATA = 'data/' + name;
  meta = await (await fetch(`${DATA}/meta.json`)).json();
  const buf = await (await fetch(`${DATA}/${meta.attn.file}`)).arrayBuffer();
  attn = new Uint8Array(buf);
  N = meta.n_tokens; H = meta.n_heads; L = meta.n_layers;
  T0 = 0;            // all tokens (prefix + patches) are normal nodes
  NPT = meta.n_prefix ?? (meta.has_cls ? 1 : 0);   // # prefix tokens before patches
}
const attnRow = (layer,head,q) => {
  const base = ((layer*H+head)*N + q)*N;
  const row = new Float32Array(N);
  for(let k=0;k<N;k++) row[k] = attn[base+k]/255;
  return row;
};

// ---------- three setup ----------
const canvas = document.getElementById('scene');
const labelLayer = document.getElementById('labels');
const renderer = new THREE.WebGLRenderer({ canvas, antialias:true, alpha:true });
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 2000);
camera.position.set(0,2,46);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.08;
controls.enableZoom = true;            // scroll wheel zoom
controls.enablePan = true;             // right-drag (or two-finger) pan up/down/left/right
controls.screenSpacePanning = true;    // pan in screen plane, not along ground
controls.zoomSpeed = 0.9; controls.panSpeed = 0.8;
scene.add(new THREE.AmbientLight(0xffffff,0.55));
const key = new THREE.DirectionalLight(0xffffff,1.1); key.position.set(8,14,12); scene.add(key);
const rim = new THREE.DirectionalLight(0x88aaff,0.5); rim.position.set(-10,-4,-8); scene.add(rim);

const sphere = new THREE.SphereGeometry(0.32, 20, 16);
function mkInstanced(color){
  const mat = new THREE.MeshStandardMaterial({ color:0xffffff, metalness:0.15, roughness:0.5 });
  const m = new THREE.InstancedMesh(sphere, mat, N);
  m.frustumCulled = false;            // instances spread far beyond geo bounds
  m.userData.base = color; scene.add(m); return m;
}
let meshes = null;   // created after load() once N is known (InstancedMesh needs count)
const dummy = new THREE.Object3D();
let pos = { q:[], k:[], v:[] };               // current Nx3 (scaled)
const linkGroup = new THREE.Group(); scene.add(linkGroup);  // Q->K tubes + K->V dashed

// ---------- positioning ----------
function buildNodes(){
  const raw = { q:meta.attn_view.q[state.head][state.layer],
                k:meta.attn_view.k[state.head][state.layer],
                v:meta.attn_view.v[state.head][state.layer] };
  // center on median; scale by the 95th-percentile radius. CLS (index 0) is
  // excluded from centering/scaling and hidden entirely.
  let cx=0,cy=0,cz=0,n=0;
  for(const t of ['q','k','v']) for(let i=T0;i<N;i++){ const p=raw[t][i]; cx+=p[0];cy+=p[1];cz+=p[2];n++; }
  cx/=n; cy/=n; cz/=n;
  const dists=[];
  for(const t of ['q','k','v']) for(let i=T0;i<N;i++){ const p=raw[t][i];
    dists.push(Math.hypot(p[0]-cx,p[1]-cy,p[2]-cz)); }
  dists.sort((a,b)=>a-b);
  const p95 = dists[Math.floor(dists.length*0.95)] || 1e-6;
  const s = 15/p95;
  for(const t of ['q','k','v']){
    pos[t] = raw[t].map(p => [ (p[0]-cx)*s, (p[1]-cy)*s, (p[2]-cz)*s ]);
    const mesh = meshes[t];
    for(let i=0;i<N;i++){
      dummy.position.set(pos[t][i][0],pos[t][i][1],pos[t][i][2]);
      dummy.scale.setScalar(i<T0 ? 0 : 1.0);    // hide CLS
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  }
  colorNodes(null);
}
function colorNodes(involvedK){
  const c = new THREE.Color();
  const set = involvedK ? new Set(involvedK) : null;
  for(const t of ['q','k','v']){
    const mesh = meshes[t];
    for(let i=0;i<N;i++){
      let lit = true;
      if(state.selQ!=null && set) lit = (t==='q') ? (i===state.selQ) : set.has(i);
      c.set(mesh.userData.base);
      if(!lit) c.lerp(BG, 0.8);   // fade but keep hue, so other Q nodes stay findable
      mesh.setColorAt(i,c);
    }
    mesh.instanceColor.needsUpdate = true;
  }
}

// ---------- links on selection ----------
function clearLinks(){
  while(linkGroup.children.length){
    const o=linkGroup.children.pop(); o.geometry.dispose(); o.material.dispose();
  }
}
function tube(a,b,radius,color,opacity){
  const va=new THREE.Vector3(...a), vb=new THREE.Vector3(...b);
  const len=va.distanceTo(vb);
  const g=new THREE.CylinderGeometry(radius,radius,len,8,1,true);
  const m=new THREE.MeshStandardMaterial({color,emissive:color,emissiveIntensity:.4,
    metalness:.3,roughness:.4,transparent:true,opacity});
  const mesh=new THREE.Mesh(g,m);
  mesh.position.copy(va).add(vb).multiplyScalar(.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),
    vb.clone().sub(va).normalize());
  linkGroup.add(mesh); return mesh;
}
function dashed(a,b,color){
  const g=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a),new THREE.Vector3(...b)]);
  const m=new THREE.LineDashedMaterial({color,dashSize:.9,gapSize:.6,transparent:true,opacity:.7});
  const l=new THREE.Line(g,m); l.computeLineDistances(); linkGroup.add(l); return l;
}
let topK = [];
function select(q){
  state.selQ = q; clearLinks(); topK=[];
  const row = attnRow(state.layer,state.head,q);
  const order = [...row.keys()].filter(k=>k>=T0).sort((a,b)=>row[b]-row[a]);
  const keep = order.filter(k=>row[k]>=0.01).slice(0,30);
  for(const k of keep){
    const w=row[k];
    // uniform thickness (weight is shown by the label + opacity + the map, not radius)
    tube(pos.q[q], pos.k[k], 0.11, COLORS.q, Math.min(.35+0.6*w,1));
    dashed(pos.k[k], pos.v[k], COLORS.v);
  }
  topK = keep.slice(0,8).map(k=>({k,w:row[k]}));
  colorNodes(keep);
  updateSelInfo(q,row,order);
  const qs=document.getElementById('query');           // keep the Query slider in sync
  qs.value=q; document.getElementById('query-val').textContent = q<NPT ? tokLabel(q) : (q-NPT);
  drawMap();
  drawQueryHeat(q);
  drawCurImg();
}

function resetSelUI(){
  const si=document.getElementById('selinfo'); si.className='selinfo dim';
  si.innerHTML='Drag the Query slider or click a Q node to see its attention.';
  document.getElementById('query-val').textContent='—';
}
function deselect(){
  state.selQ=null; clearLinks(); topK=[];
  colorNodes(null); resetSelUI(); drawMap();
  qhctx.clearRect(0,0,qheat.width,qheat.height);
  document.getElementById('qheat-cap').textContent='query → patch attention';
  drawCurImg();
}

// ---------- labels (top-k weights) ----------
const v3 = new THREE.Vector3();
function updateLabels(){
  const w=canvas.clientWidth, h=canvas.clientHeight;
  const html=[];
  if(state.view==='attention' && state.selQ!=null){
    for(const {k,w:wt} of topK){
      const mid=[ (pos.q[state.selQ][0]+pos.k[k][0])/2,
                  (pos.q[state.selQ][1]+pos.k[k][1])/2,
                  (pos.q[state.selQ][2]+pos.k[k][2])/2 ];
      v3.set(...mid).project(camera);
      if(v3.z>1) continue;
      const x=(v3.x*.5+.5)*w, y=(-v3.y*.5+.5)*h;
      html.push(`<div class="wlabel" style="left:${x}px;top:${y}px">${wt.toFixed(2)}</div>`);
    }
  }
  labelLayer.innerHTML = html.join('');
}

// ---------- attention map ----------
const map = document.getElementById('map');
const mctx = map.getContext('2d');
function cmap(t){ // orange(low)->green(high), dark near 0
  const o=[255,159,10], g=[48,209,88];
  const f=Math.pow(t,0.5);
  return [ (o[0]+(g[0]-o[0])*t)*f, (o[1]+(g[1]-o[1])*t)*f, (o[2]+(g[2]-o[2])*t)*f ];
}
function drawMap(){
  // attention map keeps the full 197x197 incl. CLS (only the 3D nodes drop CLS)
  if(map.width!==N){ map.width=N; map.height=N; }
  const img=mctx.createImageData(N,N);
  for(let q=0;q<N;q++){
    const base=((state.layer*H+state.head)*N+q)*N;
    for(let k=0;k<N;k++){
      const w=attn[base+k]/255; const [r,g,b]=cmap(w);
      const o=(q*N+k)*4; img.data[o]=r; img.data[o+1]=g; img.data[o+2]=b; img.data[o+3]=255;
    }
  }
  if(state.selQ!=null){
    const q=state.selQ;
    for(let k=0;k<N;k++){ const o=(q*N+k)*4;
      img.data[o]=Math.min(255,img.data[o]+60); img.data[o+1]=Math.min(255,img.data[o+1]+60);
      img.data[o+2]=Math.min(255,img.data[o+2]+90); }
  }
  mctx.putImageData(img,0,0);
  document.getElementById('map-sub').textContent = `L${state.layer} · head ${state.head}`;
}
function drawColorbar(){
  const cb=document.getElementById('cbar'); if(!cb) return;
  const ctx=cb.getContext('2d'); const W=cb.width,Hh=cb.height;
  const img=ctx.createImageData(W,Hh);
  for(let x=0;x<W;x++){
    const [r,g,b]=cmap(x/(W-1));
    for(let y=0;y<Hh;y++){ const o=(y*W+x)*4; img.data[o]=r;img.data[o+1]=g;img.data[o+2]=b;img.data[o+3]=255; }
  }
  ctx.putImageData(img,0,0);
}
// per-query patch heatmap: selected query's attention to each patch, one flat
// colour per patch (no upsampling), laid out on the 14x14 patch grid.
const qheat = document.getElementById('qheat');
const qhctx = qheat.getContext('2d');
// input image (canvas) with the current query's patch marked by a red box
const curImg = document.getElementById('cur-img');
const cictx = curImg.getContext('2d');
const inputImg = new Image();
inputImg.onload = drawCurImg;
function drawCurImg(){
  const S=224; if(curImg.width!==S){ curImg.width=S; curImg.height=S; }
  cictx.clearRect(0,0,S,S);
  if(inputImg.complete && inputImg.naturalWidth) cictx.drawImage(inputImg,0,0,S,S);
  const q=state.selQ;
  if(q!=null && q>=NPT){                      // prefix tokens have no patch location
    const grid=meta.patch_grid, cell=S/grid, p=q-NPT;
    const x=(p%grid)*cell, y=Math.floor(p/grid)*cell;
    cictx.lineWidth=3; cictx.strokeStyle='#ff3b30';
    cictx.strokeRect(x+1.5,y+1.5,cell-3,cell-3);
  }
}
function drawQueryHeat(q){
  const grid = meta.patch_grid;                 // 14
  if(qheat.width!==grid){ qheat.width=grid; qheat.height=grid; }
  const base=((state.layer*H+state.head)*N+q)*N;
  let mx=1e-6;                                   // normalize per query for visibility
  for(let p=0;p<grid*grid;p++){ const w=attn[base+NPT+p]/255; if(w>mx) mx=w; }
  const img=qhctx.createImageData(grid,grid);
  for(let p=0;p<grid*grid;p++){                  // patches are tokens NPT..NPT+grid^2-1
    const [r,g,b]=cmap((attn[base+NPT+p]/255)/mx);
    const o=p*4; img.data[o]=r; img.data[o+1]=g; img.data[o+2]=b; img.data[o+3]=255;
  }
  qhctx.putImageData(img,0,0);
  document.getElementById('qheat-cap').textContent = `${tokLabel(q)} → patches`;
}
map.addEventListener('click', e=>{
  const r=map.getBoundingClientRect();
  const q=Math.floor((e.clientY-r.top)/r.height*N);
  if(q>=T0&&q<N){ if(q===state.selQ) deselect(); else select(q); }  // CLS row not selectable
});

function updateSelInfo(q,row,order){
  const top=order.slice(0,6);
  let html=`<div style="margin-bottom:6px">query <b>${tokLabel(q)}</b> · top keys:</div>`;
  for(const k of top){
    html+=`<div class="row"><span>${tokLabel(k)}</span><b>${row[k].toFixed(3)}</b></div>`;
  }
  document.getElementById('selinfo').className='selinfo';
  document.getElementById('selinfo').innerHTML=html;
}

// ---------- picking ----------
const ray = new THREE.Raycaster(); const ptr = new THREE.Vector2();
canvas.addEventListener('pointerdown', e=>{
  const r=canvas.getBoundingClientRect();
  ptr.x=((e.clientX-r.left)/r.width)*2-1; ptr.y=-((e.clientY-r.top)/r.height)*2+1;
  ray.setFromCamera(ptr,camera);
  const hit=ray.intersectObject(meshes.q);
  if(hit.length && hit[0].instanceId>=T0){
    const id=hit[0].instanceId;
    if(id===state.selQ) deselect(); else select(id);   // re-click toggles off
  }
});

// ---------- controls UI ----------
const layerEl=document.getElementById('layer'), headEl=document.getElementById('head'),
      queryEl=document.getElementById('query');
layerEl.addEventListener('input',e=>{ state.layer=+e.target.value;
  document.getElementById('layer-val').textContent=state.layer; refresh(); });
headEl.addEventListener('input',e=>{ state.head=+e.target.value;
  document.getElementById('head-val').textContent=state.head; refresh(); });
queryEl.addEventListener('input',e=>{ select(+e.target.value); });   // live query selection
function refresh(){ setDbg();
  // layer/head change keeps the selected query but re-renders it cleanly (no afterimage)
  clearLinks(); buildNodes();
  if(state.selQ!=null) select(state.selQ); else drawMap();
}

// ---------- resize + loop ----------
function resize(){
  const w=canvas.clientWidth,h=canvas.clientHeight;
  if(canvas.width!==w*devicePixelRatio||canvas.height!==h*devicePixelRatio){
    renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix();
  }
}
function loop(){
  resize(); controls.update(); renderer.render(scene,camera); updateLabels();
  requestAnimationFrame(loop);
}

// ---------- on-screen debug (no console needed) ----------
const dbg = document.createElement('div');
dbg.style.cssText='position:fixed;left:8px;top:8px;z-index:99;font:11px/1.4 monospace;'
 +'color:#9fe;background:#000b;padding:6px 9px;border-radius:6px;white-space:pre;pointer-events:none';
const DEBUG = new URLSearchParams(location.search).has('debug');
if(DEBUG) document.body.appendChild(dbg);
function setDbg(){
  if(!DEBUG) return;
  const q0=pos.q[0], qm=pos.q[98]||[0,0,0];
  dbg.textContent =
    `canvas ${canvas.clientWidth}x${canvas.clientHeight}\n`+
    `nodes ${N}  layer ${state.layer} head ${state.head}\n`+
    `Q[0]   ${q0?q0.map(x=>x.toFixed(1)).join(','):'-'}\n`+
    `Q[98]  ${qm.map(x=>x.toFixed(1)).join(',')}\n`+
    `cam ${camera.position.x.toFixed(0)},${camera.position.y.toFixed(0)},${camera.position.z.toFixed(0)}`;
}

// apply the currently-loaded meta/attn to the scene (also used on dataset switch)
function applyDataset(initialQ){
  if(!meshes) meshes = { q:mkInstanced(COLORS.q), k:mkInstanced(COLORS.k), v:mkInstanced(COLORS.v) };
  state.layer=0; state.head=0; state.selQ=null;
  layerEl.max=L-1; layerEl.value=0; document.getElementById('layer-val').textContent=0;
  headEl.max=H-1;  headEl.value=0; document.getElementById('head-val').textContent=0;
  queryEl.max=N-1;
  inputImg.src = `${DATA}/image.jpg`;        // onload → drawCurImg (image + query box)
  buildNodes(); drawColorbar(); setDbg();
  select(initialQ!=null ? Math.max(0,Math.min(N-1,initialQ)) : 0);  // default CLS
}
async function switchDataset(name){
  await fetchDataset(name);
  applyDataset();
  document.querySelectorAll('.sample').forEach(el=>
    el.classList.toggle('active', el.dataset.name===name));
}
function buildSampleStrip(samples){
  const box=document.getElementById('samples'); box.innerHTML='';
  for(const s of samples){
    const d=document.createElement('button'); d.className='sample'; d.dataset.name=s.name;
    d.innerHTML=`<img src="data/${s.name}/image.jpg" alt="${s.label}"><span>${s.label}</span>`;
    d.onclick=()=>switchDataset(s.name);
    box.appendChild(d);
  }
}
function buildLayerViz(samples, nLayers){
  const sel=document.getElementById('lv-samples'); sel.innerHTML='';
  for(const s of samples){
    const b=document.createElement('button'); b.className='sample'; b.dataset.lv=s.name;
    b.innerHTML=`<img src="data/${s.name}/image.jpg" alt="${s.label}"><span>${s.label}</span>`;
    b.onclick=()=>renderLayerViz(s.name, nLayers);
    sel.appendChild(b);
  }
  renderLayerViz(samples[0].name, nLayers);
}
function renderLayerViz(name, nLayers){
  document.querySelectorAll('#lv-samples .sample').forEach(el=>
    el.classList.toggle('active', el.dataset.lv===name));
  const box=document.getElementById('lv-grid'); box.innerHTML='';
  const card=(src,tag,pca)=>{
    const c=document.createElement('div'); c.className='lv-card'+(pca?' pca':'');
    c.innerHTML=`<img src="${src}" alt="${tag}"><span class="lv-tag">${tag}</span>`;
    box.appendChild(c);
  };
  card(`data/${name}/image.jpg`, 'input', false);   // original first
  for(let Lr=0;Lr<nLayers;Lr++)
    card(`data/${name}/layers/L${String(Lr).padStart(2,'0')}.png`, 'L'+Lr, true);
  loadClusters(name);                                // sync the 3D cluster view
}

(async ()=>{
  const u=new URLSearchParams(location.search);
  const samples = await (await fetch('data/index.json')).json();
  buildSampleStrip(samples);
  const first = u.get('img') || samples[0].name;
  await fetchDataset(first);
  applyDataset(u.has('q') ? +u.get('q') : 0);
  document.querySelectorAll('.sample').forEach(el=>
    el.classList.toggle('active', el.dataset.name===first));
  initClusters();
  buildLayerViz(samples, L);     // L = meta.n_layers (24); also triggers loadClusters
  loop();
})().catch(err=>{
  if(!dbg.isConnected) document.body.appendChild(dbg);
  dbg.style.color='#f88'; dbg.textContent='ERROR: '+err.message+'\n'+(err.stack||'');
  console.error(err);
});

// ================= Layer clusters: all layers' MLP outputs in one 3D PCA space =================
let clRenderer, clScene, clCamera, clControls, clMat, clPts, clGeo, clCol, clLayerOf, clNB, clBtns;
let clSel = null;                                  // null = all layers, else layer index
const CLBG = new THREE.Color(0x0e1018);
function clCmap(t){ return new THREE.Color(0xff9f0a).lerp(new THREE.Color(0x30d158), t); }
function initClusters(){
  const cv = document.getElementById('cl-scene');
  clRenderer = new THREE.WebGLRenderer({ canvas:cv, antialias:true, alpha:true });
  clRenderer.setPixelRatio(Math.min(devicePixelRatio,2));
  clScene = new THREE.Scene();
  clCamera = new THREE.PerspectiveCamera(45,1,0.1,2000); clCamera.position.set(0,2,30);
  clControls = new OrbitControls(clCamera, cv); clControls.enableDamping=true; clControls.dampingFactor=0.08;
  const tcv=document.createElement('canvas'); tcv.width=tcv.height=64; const tg=tcv.getContext('2d');
  tg.beginPath(); tg.arc(32,32,28,0,7); tg.fillStyle='#fff'; tg.fill();
  clMat = new THREE.PointsMaterial({ size:0.6, sizeAttenuation:true, vertexColors:true,
    map:new THREE.CanvasTexture(tcv), alphaTest:0.4, transparent:true });
  (function loopC(){
    if(clRenderer){
      const w=cv.clientWidth,h=cv.clientHeight;
      if(cv.width!==w*devicePixelRatio||cv.height!==h*devicePixelRatio){
        clRenderer.setSize(w,h,false); clCamera.aspect=w/h; clCamera.updateProjectionMatrix(); }
      clControls.update(); clRenderer.render(clScene,clCamera);
    }
    requestAnimationFrame(loopC);
  })();
}
async function loadClusters(name){
  const cl = await (await fetch(`data/${name}/cluster.json`)).json();
  clNB = cl.n_layers; const P=cl.n_patches, coords=cl.coords;
  let cx=0,cy=0,cz=0,nn=0; const ds=[];
  for(let l=0;l<clNB;l++) for(let p=0;p<P;p++){ const q=coords[l][p]; cx+=q[0];cy+=q[1];cz+=q[2];nn++; }
  cx/=nn;cy/=nn;cz/=nn;
  for(let l=0;l<clNB;l++) for(let p=0;p<P;p++){ const q=coords[l][p];
    ds.push(Math.hypot(q[0]-cx,q[1]-cy,q[2]-cz)); }
  ds.sort((a,b)=>a-b); const s=15/(ds[Math.floor(ds.length*0.95)]||1);
  const pos=new Float32Array(clNB*P*3); clLayerOf=new Int16Array(clNB*P); let k=0;
  for(let l=0;l<clNB;l++) for(let p=0;p<P;p++){ const q=coords[l][p];
    pos[k*3]=(q[0]-cx)*s; pos[k*3+1]=(q[1]-cy)*s; pos[k*3+2]=(q[2]-cz)*s; clLayerOf[k]=l; k++; }
  if(clPts){ clScene.remove(clPts); clGeo.dispose(); }
  clGeo=new THREE.BufferGeometry();
  clGeo.setAttribute('position', new THREE.BufferAttribute(pos,3));
  clCol=new Float32Array(clNB*P*3); clGeo.setAttribute('color', new THREE.BufferAttribute(clCol,3));
  clPts=new THREE.Points(clGeo, clMat); clScene.add(clPts);
  if(!clBtns) buildClButtons();
  clRecolor();
}
function clRecolor(){
  const c=new THREE.Color();
  for(let i=0;i<clLayerOf.length;i++){
    const l=clLayerOf[i]; c.copy(clCmap(l/(clNB-1)));
    if(clSel!==null && l!==clSel) c.lerp(CLBG,0.9);
    clCol[i*3]=c.r; clCol[i*3+1]=c.g; clCol[i*3+2]=c.b;
  }
  clGeo.attributes.color.needsUpdate=true;
  document.getElementById('cl-hint').textContent =
    clSel===null ? `all ${clNB} layers` : `layer ${clSel} highlighted`;
}
function buildClButtons(){
  const bar=document.getElementById('cl-bar');
  bar.querySelectorAll('button').forEach(b=>b.remove());
  const mk=(txt,fn,cls)=>{ const b=document.createElement('button'); if(cls)b.className=cls;
    b.textContent=txt; b.onclick=fn; bar.appendChild(b); return b; };
  clBtns=[ mk('All',()=>{clSel=null;clSync();},'all') ];
  for(let l=0;l<clNB;l++) clBtns.push(mk(l,()=>{clSel=l;clSync();}));
  clSync();
}
function clSync(){
  clBtns.forEach((b,i)=>b.classList.toggle('active',(i===0&&clSel===null)||(i-1===clSel)));
  clRecolor();
}

