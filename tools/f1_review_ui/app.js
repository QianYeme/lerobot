const $ = s => document.querySelector(s);
const video = $('#video');
const state = {episodes: [], current: 0, rows: [], revision: '', frame: 0, dirty: false, saving: false, dragging: false, frameRequest: 0, autosave: false, saveTimer: null};
const labels = {
  PRECONTACT: '尚未接触', CONTACT_SUPPORTED: '接触有视频支持', CLOSING: '闭爪过程',
  POST_CLOSE: '闭爪后至释放前', RELEASE: '张爪释放', UNKNOWN: '已审但不可判', OCCLUDED: '遮挡不可判'
};
const fields = [['first_reliable_frame','可靠开始'],['last_reliable_frame','可靠结束'],['uncertain_start','不确定开始'],['uncertain_end','不确定结束']];
const esc = v => String(v ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const episode = () => state.episodes.find(e => e.episode === state.current);
const draftKey = () => `f1-rgb:${state.csvPath}:ep${state.current}`;
function message(text, kind = '') { $('#saveStatus').textContent = text; $('#saveStatus').className = kind; }
function updateFrame(frame) {
  state.frame = Math.max(0, Math.min(episode().frames - 1, frame));
  if (!state.dragging) $('#timeline').value = state.frame;
  $('#frameText').textContent = `原始第 ${state.frame} / ${episode().frames-1} 帧 · ${(state.frame / episode().fps).toFixed(3)} 秒`;
}
function showExactFrame() {
  if (!episode() || !video.paused || state.dragging) return;
  const id = ++state.frameRequest, ep = state.current, frame = state.frame;
  const img = new Image();
  img.onload = () => { if (id === state.frameRequest && video.paused && ep === state.current) { $('#still').src = img.src; $('#still').hidden = false; } };
  img.onerror = () => message('指定帧解码失败，请重试跳转；不要据此取帧', 'error');
  img.src = `/api/frame?episode=${ep}&frame=${frame}`;
}
function seek(frame) {
  video.pause();
  updateFrame(Math.round(frame));
  video.currentTime = (state.frame + .1) / episode().fps;
  showExactFrame();
}
function dirty() {
  state.dirty = true;
  message('有未保存更改 · 草稿保留在本浏览器，请点击保存到 CSV');
  try { localStorage.setItem(draftKey(), JSON.stringify({rows: state.rows, revision: state.revision, reviewer: $('#reviewer').value})); }
  catch { message('浏览器草稿存储不可用，请及时保存或下载草稿', 'error'); }
  clearTimeout(state.saveTimer);
  if (state.autosave) state.saveTimer = setTimeout(() => {
    const ready = state.rows.length > 0 && state.rows.every(r => {
      const unknown = ['UNKNOWN','OCCLUDED'].includes(r.label);
      return (r.reviewer || $('#reviewer').value.trim()) && r.reason?.trim() &&
        (unknown ? r.uncertain_start !== '' && r.uncertain_end !== '' && r.first_reliable_frame === '' && r.last_reliable_frame === '' :
          r.first_reliable_frame !== '' && r.last_reliable_frame !== '');
    });
    if (ready && !state.saving) save();
    else message('草稿已保留；填写完整审核者、区间及证据说明后自动写入 CSV');
  }, 1000);
}
function renderRows() {
  $('#intervals').innerHTML = state.rows.map((r,i) => `<article class="interval" data-row="${i}">
    <div class="row-head"><select aria-label="区间 ${i+1} 标签" data-field="label">${Object.entries(labels).filter(([k])=>state.allowedLabels.includes(k)).map(([k,v])=>`<option value="${k}" ${r.label===k?'selected':''}>${k} · ${v}</option>`).join('')}</select><button class="remove" data-remove="${i}">删除</button></div>
    ${fields.map(([k,title])=>`<div class="boundary"><label for="r${i}_${k}">${title}</label><input id="r${i}_${k}" aria-label="${title}" type="number" min="0" max="${episode().frames-1}" value="${esc(r[k])}" data-field="${k}"><div><button data-capture="${k}">取当前帧</button> <button data-seek="${k}">跳转</button></div></div>`).join('')}
    <textarea class="reason" data-field="reason" aria-label="可见证据或不可判原因" placeholder="必填：看到了什么？为何可支持此标签？若遮挡，说明遮挡部位。">${esc(r.reason)}</textarea>
    <p class="muted">本段审核者：${esc(r.reviewer || $('#reviewer').value || '请填写顶部审核者')} · 视频证据路径保存时自动生成</p>
  </article>`).join('') || '<p class="muted">尚无区间。播放视频后添加；不会预填信号推导的标签。</p>';
}
function loadEpisode(ep) {
  if (state.dirty && !confirm('当前修改尚未写入 CSV，仍切换吗？本集草稿保留，可用“恢复本集草稿”找回。')) return;
  clearTimeout(state.saveTimer);
  video.pause(); ++state.frameRequest;
  state.current = ep;
  state.rows = structuredClone(episode().rows);
  state.dirty = false;
  $('#episodeTitle').textContent = `Train ${String(ep).padStart(3,'0')} · 区间标注`;
  $('#timeline').max = episode().frames - 1;
  $('#jump').max = episode().frames - 1;
  $('#still').hidden = true;
  video.src = episode().video;
  video.playbackRate = Number($('#speed').value);
  updateFrame(0); renderRows();
  $('#episodes').innerHTML = state.episodes.map(e=>`<button data-episode="${e.episode}" class="${e.episode===ep?'active':''}">Train ${e.episode} · ${e.rows.length} 段</button>`).join('');
  let draft = false;
  try { draft = !!localStorage.getItem(draftKey()); } catch {}
  message(draft ? '本集存在浏览器草稿，可点击“恢复本集草稿”；当前显示磁盘 CSV' : '已读取磁盘 CSV · 保存不会自动判定整集完成');
}
async function save() {
  if (state.saving) return;
  clearTimeout(state.saveTimer);
  const ep = state.current, rows = structuredClone(state.rows);
  for (const r of rows) if (!r.reviewer) r.reviewer = $('#reviewer').value.trim();
  state.saving = true;
  const controls = [...document.querySelectorAll('button,input,textarea,select')];
  controls.forEach(e=>e.disabled=true);
  message('正在写入、备份并回读 CSV…');
  try {
    const response = await fetch('/api/save', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({episode:ep,rows,revision:state.revision})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '保存失败');
    state.revision = result.revision;
    state.rows = result.rows;
    episode().rows = structuredClone(result.rows);
    state.dirty = false;
    try { localStorage.removeItem(draftKey()); } catch {}
    loadEpisodeAfterSave(ep);
    message(`已写入并回读确认：Train ${ep}，${result.rows.length} 段；旧 CSV 已备份`, 'saved');
  } catch(error) { message(`保存失败：${error.message}。当前填写未清空，可下载草稿。`, 'error'); }
  finally { state.saving = false; controls.forEach(e=>e.disabled=false); }
}
function loadEpisodeAfterSave(ep) {
  renderRows();
  const button = document.querySelector(`[data-episode="${ep}"]`);
  if (button) button.textContent = `Train ${ep} · ${state.rows.length} 段`;
}
$('#episodes').addEventListener('click', e=>{ const b=e.target.closest('[data-episode]'); if(b) loadEpisode(Number(b.dataset.episode)); });
$('#intervals').addEventListener('input', e=>{
  const card=e.target.closest('[data-row]'); if(!card || !e.target.dataset.field) return;
  state.rows[Number(card.dataset.row)][e.target.dataset.field]=e.target.value; dirty();
});
$('#intervals').addEventListener('click', e=>{
  const card=e.target.closest('[data-row]'); if(!card) return;
  const i=Number(card.dataset.row), b=e.target.closest('button'); if(!b) return;
  if(b.dataset.remove!==undefined) { if(confirm('删除这段区间？保存后才会覆盖磁盘。')){ state.rows.splice(i,1);renderRows();dirty(); } }
  if(b.dataset.capture) { video.pause(); state.rows[i][b.dataset.capture]=String(state.frame);renderRows();dirty();showExactFrame(); }
  if(b.dataset.seek && state.rows[i][b.dataset.seek] !== '') seek(Number(state.rows[i][b.dataset.seek]));
});
$('#reviewer').addEventListener('input', ()=>{ for(const r of state.rows) r.reviewer=$('#reviewer').value.trim(); dirty(); });
$('#add').onclick=()=>{ state.rows.push({label:'PRECONTACT',first_reliable_frame:'',last_reliable_frame:'',uncertain_start:'',uncertain_end:'',reviewer:$('#reviewer').value.trim(),reason:''});renderRows();dirty(); };
$('#save').onclick=save;
$('#recover').onclick=()=>{
  try { const draft=JSON.parse(localStorage.getItem(draftKey())); if(!draft) return message('本集没有浏览器草稿');
    if(!confirm('用浏览器草稿替换当前表单？磁盘 CSV 此时不会改变。')) return;
    state.rows=draft.rows;$('#reviewer').value=draft.reviewer||'';
    // Preserve draft's old revision: cannot overwrite a newer CSV silently.
    state.revision=draft.revision;renderRows();dirty();
  }catch(error){message(`恢复失败：${error.message}`,'error');}
};
$('#download').onclick=()=>{ const url=URL.createObjectURL(new Blob([JSON.stringify({episode:state.current,rows:state.rows,revision:state.revision},null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`f1_train_${state.current}_draft.json`;a.click();URL.revokeObjectURL(url); };
$('#play').onclick=()=>{ if(video.paused){++state.frameRequest;$('#still').hidden=true;video.play().catch(e=>message(`播放失败：${e.message}`,'error'));}else video.pause(); };
document.querySelectorAll('[data-step]').forEach(b=>b.onclick=()=>seek(state.frame+Number(b.dataset.step)));
$('#speed').onchange=()=>video.playbackRate=Number($('#speed').value);
$('#go').onclick=()=>{ if($('#jump').value!=='') seek(Number($('#jump').value)); };
$('#timeline').addEventListener('pointerdown',()=>{state.dragging=true;video.pause();++state.frameRequest;$('#still').hidden=true;});
$('#timeline').addEventListener('input',()=>{state.dragging=true;video.pause();updateFrame(Number($('#timeline').value));video.currentTime=(state.frame+.1)/episode().fps;});
$('#timeline').addEventListener('change',()=>{state.dragging=false;seek(Number($('#timeline').value));});
$('#timeline').addEventListener('pointercancel',()=>{state.dragging=false;showExactFrame();});
video.addEventListener('loadedmetadata',()=>{video.playbackRate=Number($('#speed').value);showExactFrame();});
video.addEventListener('pause',()=>{if(!state.dragging) showExactFrame();});
video.addEventListener('error',()=>message('视频加载失败，请检查本地服务终端；标注草稿仍保留','error'));
function progress(time) { if(!state.dragging && !video.paused && episode()) updateFrame(Math.floor(time*episode().fps+1e-5)); }
if(video.requestVideoFrameCallback) { const tick=(_,m)=>{progress(m.mediaTime);video.requestVideoFrameCallback(tick);};video.requestVideoFrameCallback(tick); }
else video.addEventListener('timeupdate',()=>progress(video.currentTime));
document.addEventListener('keydown',e=>{ if(e.target.closest('input,textarea,select') || state.saving) return;
  if(e.code==='Space'){e.preventDefault();$('#play').click();}
  if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();seek(state.frame+(e.key==='ArrowLeft'?-1:1)*(e.shiftKey?10:1));}
});
window.addEventListener('beforeunload',e=>{if(state.dirty){e.preventDefault();e.returnValue='';}});
fetch('/api/bootstrap').then(async r=>{if(!r.ok)throw new Error('CSV读取失败');return r.json();}).then(data=>{
  state.episodes=data.episodes;state.revision=data.revision;state.csvPath=data.csv_path;state.allowedLabels=data.labels;state.autosave=!!data.autosave;
  if(data.pilot) {
    document.title='F1P · 新增六集 RGB 独立盲审';
    $('#pageTitle').textContent='F1P · 新增六集接触盲审';
    $('#contextLabel').textContent='EXPLORATORY ONLY / 原 TRAIN 子集 · PILOT-EVAL';
    $('#save').textContent='立即保存到 CSV';
    document.querySelectorAll('[data-phase-help]').forEach(e=>e.hidden=true);
    $('#axisHelp').textContent='只标尚未接触 / 接触有支持；两类可靠帧不能重叠。不确定边界不放进可靠区间，可用 UNKNOWN / OCCLUDED 单列。无需覆盖全视频，可为每类添加多段。';
    $('.editor > .muted').textContent='只标尚未接触 / 接触有支持两类；不确定可留空隙或另标。完整区间停输1秒后自动保存、备份及回读。无需再标闭爪、保持、释放阶段。';
  }
  $('#csvPath').textContent=`直接写入：${data.csv_path}`;loadEpisode(state.episodes[0].episode);
}).catch(e=>message(`初始化失败：${e.message}`,'error'));
