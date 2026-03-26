"""
qsar_viewer.py  v5
==================
Python 端用 marching cubes 預先計算等值面網格，
前端用 3Dmol.js addCustomMesh 直接渲染三角形，
完全不依賴 addVolumetricData，穩定可靠。

架構：
  1. Python：_marching_cubes() → 頂點+面索引 JSON
  2. JS：addCustomMesh() 渲染各等值面層
  3. 立體場（綠/黃）+ 靜電場（藍/紅）分開計算，同時疊加

用法：
  python qsar_viewer.py --overlay -i qsar_grid_maps/ -s all_aligned.sdf
  python qsar_viewer.py -i qsar_grid_maps/
"""

import os, sys, gc, json, base64, argparse, traceback, time
import numpy as np

_C={"reset":"\033[0m","bold":"\033[1m","cyan":"\033[96m",
    "green":"\033[92m","yellow":"\033[93m","red":"\033[91m","dim":"\033[2m"}
def _c(t,*k):
    if not sys.stdout.isatty(): return str(t)
    return "".join(_C.get(x,"") for x in k)+str(t)+_C["reset"]
def _hr(c="─",w=62): return _c(c*w,"dim")

ANGSTROM_TO_BOHR = 1.0/0.529177
_ELEMENT={1:"H",5:"B",6:"C",7:"N",8:"O",9:"F",14:"Si",
          15:"P",16:"S",17:"Cl",35:"Br",53:"I"}

# CoMFA 標準配色
FIELDS_CFG = {
    "perturb_steric":  {"label":"立體場", "pos_color":"#00cc44","neg_color":"#ffcc00"},
    "perturb_electro": {"label":"靜電場", "pos_color":"#2255ff","neg_color":"#ff2222"},
}

# ══════════════════════════════════════════════════════════════════════════════
# 核心：marching cubes 在 Python 端預計算等值面
# ══════════════════════════════════════════════════════════════════════════════

def _auto_isoval(arr_raw):
    """
    自動計算合適的正負等值面閾值。
    基準：至少要有 3 個格點超過閾值才能形成等值面。
    優先 ±1.0，不夠再降到 ±0.5、±0.3、±0.1。
    """
    arr  = arr_raw.astype(np.float64)
    flat = arr.ravel()

    iso_p, iso_n = float(arr.max() * 0.5), float(arr.min() * 0.5)

    for iso in [1.0, 0.5, 0.3, 0.1]:
        n_pos = int(np.sum(flat >  iso))
        n_neg = int(np.sum(flat < -iso))
        if n_pos >= 3 and n_neg >= 3:
            return float(iso), float(-iso)
        if n_pos >= 3 and iso_n == float(arr.min() * 0.5):
            iso_p = float(iso)
        if n_neg >= 3 and iso_p == float(arr.max() * 0.5):
            iso_n = float(-iso)

    return iso_p, iso_n


def _marching_cubes(arr_raw, origin, axes_diag,
                    isoval_pos=None, isoval_neg=None,
                    step_size=1):
    """
    輸入原始場資料，輸出等值面網格資料。
    step_size=1：不跳格，對小格點網格（891個）必須如此才有足夠頂點。
    """
    from skimage.measure import marching_cubes as mc

    arr = arr_raw.astype(np.float64)

    if isoval_pos is None or isoval_neg is None:
        auto_pos, auto_neg = _auto_isoval(arr)
        if isoval_pos is None: isoval_pos = auto_pos
        if isoval_neg is None: isoval_neg = auto_neg

    print(f"\n    {arr.shape} max={arr.max():.3f} min={arr.min():.3f} "
          f"iso+={isoval_pos:.2f} iso-={isoval_neg:.2f}",
          end="", flush=True)

    res = float(axes_diag[0][0]) if hasattr(axes_diag[0], '__len__') else float(axes_diag[0])
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])

    def _extract(level):
        if level > arr.max() or level < arr.min():
            return {"verts": [], "faces": []}
        try:
            verts, faces, _, _ = mc(arr, level=level, step_size=step_size)
        except Exception:
            return {"verts": [], "faces": []}
        coords = verts * res + np.array([ox, oy, oz])
        return {
            "verts": coords.astype(np.float32).tolist(),
            "faces": faces.astype(np.int32).tolist(),
        }

    pos_mesh = _extract(isoval_pos)
    neg_mesh  = _extract(isoval_neg)
    print(f" → +{len(pos_mesh['verts'])}/-{len(neg_mesh['verts'])}頂點",
          end="", flush=True)

    return {
        "pos":         pos_mesh,
        "neg":         neg_mesh,
        "isoval_pos":  isoval_pos,
        "isoval_neg":  isoval_neg,
        "zscore_mu":   0.0,
        "zscore_sig":  1.0,
        "n_verts_pos": len(pos_mesh["verts"]),
        "n_verts_neg": len(neg_mesh["verts"]),
    }


def _verts_to_fake_sdf(pos_verts, neg_verts, mol_name="field"):
    """
    把等值面頂點轉換成假原子 SDF。
    正等值面頂點 → He 原子（原子序 2）
    負等值面頂點 → Li 原子（原子序 3）
    JS 端用 addSurface(SES, {elem:'He'}) 渲染，得到光滑實心面。
    """
    n_pos = len(pos_verts)
    n_neg = len(neg_verts)
    n_total = n_pos + n_neg

    if n_total == 0:
        return None, 0, 0

    lines = [mol_name, "  qsar_field", "",
             f"{n_total:3d}  0  0  0  0  0  0  0  0  0999 V2000"]

    for v in pos_verts:
        x, y, z = float(v[0]), float(v[1]), float(v[2])
        lines.append(f"{x:10.4f}{y:10.4f}{z:10.4f} He  0  0  0  0  0  0  0  0  0  0  0  0")

    for v in neg_verts:
        x, y, z = float(v[0]), float(v[1]), float(v[2])
        lines.append(f"{x:10.4f}{y:10.4f}{z:10.4f} Li  0  0  0  0  0  0  0  0  0  0  0  0")

    lines.append("M  END")
    return "\n".join(lines), n_pos, n_neg


def _b64(s):
    return base64.b64encode(s.encode()).decode()

# ══════════════════════════════════════════════════════════════════════════════
# 資料讀取
# ══════════════════════════════════════════════════════════════════════════════

def _load_npz(path):
    f = np.load(path, allow_pickle=True, mmap_mode=None)
    try:
        meta   = json.loads(str(f["_meta"][0]))
        fields = {}
        for k in FIELDS_CFG:
            if k in f:
                fields[k] = f[k].astype(np.float32)
        nums   = f["atomic_nums"].astype(np.int32)
        coords = f["atomic_coords"].astype(np.float32)
    finally:
        f.close(); del f; gc.collect()
    return meta, nums, coords, fields


def _mol_sdf(nums, coords, name="mol"):
    n = len(nums)
    ls = [name, "  qsar", "",
          f"{n:3d}  0  0  0  0  0  0  0  0  0999 V2000"]
    for i in range(n):
        x,y,z = coords[i]
        s = _ELEMENT.get(int(nums[i]),"C")
        ls.append(f"{x:10.4f}{y:10.4f}{z:10.4f} {s:<3s} 0  0  0  0  0  0  0  0  0  0  0  0")
    ls.append("M  END")
    return "\n".join(ls)


def _read_sdf(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        txt = f.read()
    mols = []
    for rec in txt.split("$$$$"):
        rec = rec.strip()
        if not rec: continue
        lines = rec.splitlines()
        name = lines[0].strip() if lines else f"mol_{len(mols)+1}"
        if not name: name = f"mol_{len(mols)+1}"
        mols.append((name, rec+"\n"))
    return mols

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

_CSS = """
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#1a1a2e;color:#e0e0e0;font-family:"Segoe UI",sans-serif;
     display:flex;height:100vh;overflow:hidden;}
#sidebar{width:270px;min-width:240px;background:#16213e;
  border-right:1px solid #0f3460;overflow-y:auto;padding:14px;}
.title{font-size:14px;font-weight:bold;color:#00d4ff;margin-bottom:2px;}
.sub{font-size:11px;color:#888;margin-bottom:12px;line-height:1.5;}
.grp{background:#0f3460;border-radius:7px;padding:11px;margin-bottom:10px;}
.grp-t{font-size:11px;color:#777;text-transform:uppercase;
  letter-spacing:.5px;margin-bottom:8px;}
.fb{display:flex;align-items:center;gap:6px;background:#1a1a2e;
  border:1px solid #0f3460;border-radius:5px;padding:6px 9px;
  cursor:pointer;margin-bottom:5px;transition:border-color .2s;width:100%;}
.fb:hover{border-color:#e94560;} .fb.active{border-color:#00d4ff;background:#162447;}
.dot{width:9px;height:9px;border-radius:50%;flex-shrink:0;}
.fn{font-size:13px;flex:1;text-align:left;}
.row{display:flex;align-items:center;gap:7px;margin-bottom:7px;}
.row label{font-size:11px;width:62px;flex-shrink:0;}
.row input[type=range]{flex:1;accent-color:#00d4ff;}
.rv{font-size:11px;color:#00d4ff;width:36px;text-align:right;}
.sb,.bb,.fb2{background:#1a1a2e;border:1px solid #0f3460;border-radius:4px;
  padding:4px 8px;font-size:11px;cursor:pointer;color:#ccc;transition:all .2s;}
.sb:hover,.bb:hover,.fb2:hover{border-color:#e94560;}
.sb.active,.bb.active,.fb2.active{border-color:#00d4ff;color:#00d4ff;}
.btn-row{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:6px;}
#stats{font-size:11px;color:#aaa;line-height:1.8;}
#viewer-wrap{flex:1;position:relative;}
#viewer{width:100%;height:100%;}
#hint{position:absolute;bottom:8px;left:50%;transform:translateX(-50%);
  background:rgba(0,0,0,.6);color:#777;font-size:11px;
  padding:3px 10px;border-radius:10px;pointer-events:none;}
#loading{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  color:#00d4ff;font-size:16px;text-align:center;z-index:10;}
.color-swatch{display:inline-block;width:10px;height:10px;
  border-radius:2px;margin-right:3px;vertical-align:middle;}
.pic50-bar{height:7px;border-radius:3px;margin:4px 0;
  background:linear-gradient(to right,#2255ff,#00cc44,#ffcc00,#ff2222);}
.pic50-lbl{display:flex;justify-content:space-between;font-size:10px;color:#666;}
"""

# ══════════════════════════════════════════════════════════════════════════════
# JS 核心
# ══════════════════════════════════════════════════════════════════════════════

_JS_CORE = """
function b64dec(s){try{return decodeURIComponent(escape(atob(s)));}catch(e){return atob(s);}}

// FIELD_SDFS[key] = {sdf, n_pos, n_neg, isoval_pos, isoval_neg}
const FIELD_SDFS = {};
for(const [k,v] of Object.entries(FIELD_SDFS_B64)){
  FIELD_SDFS[k] = JSON.parse(b64dec(v));
}

let molSDF=null, _molCache=null;
function getMolCache(){
  if(_molCache) return _molCache;
  if(Array.isArray(molSDF))
    _molCache=molSDF.map(function(e){return [e[0],e[1],b64dec(e[2])];});
  return _molCache;
}

let viewer=null;
let activeFields=new Set();
let isoOpacity=0.75, molOpacity=1.0, molStyle="stick", bgColor="black";
let COLORING="element", FILTER="all";
let showPos={perturb_steric:true,perturb_electro:true};
let showNeg={perturb_steric:true,perturb_electro:true};

let _timer=null;
function _schedule(){if(_timer)clearTimeout(_timer);_timer=setTimeout(_doRebuild,80);}

function _doRebuild(){
  _timer=null;
  const el=document.getElementById("viewer");
  el.innerHTML="";
  viewer=$3Dmol.createViewer(el,{backgroundColor:bgColor,antialias:true});

  // 加入分子
  if(Array.isArray(molSDF)){
    getMolCache().forEach(function(e){
      const mdl=viewer.addModel(e[2],"mol");
      _styleModel(mdl,e[0]===REF_NAME,e[1]);
    });
  } else if(molSDF){
    _styleModel(viewer.addModel(molSDF,"mol"),true,null);
  }
  viewer.zoomTo();

  // 疊加場：假原子 SDF + addSurface(SES)
  activeFields.forEach(function(key){
    const fd=FIELD_SDFS[key], cfg=FIELDS_CFG[key];
    if(!fd||!fd.sdf) return;
    const fMdl=viewer.addModel(fd.sdf,"mol");
    viewer.setStyle({model:fMdl.getID()},{});  // 隱藏假原子本身
    if(showPos[key]&&fd.n_pos>0){
      viewer.addSurface($3Dmol.SurfaceType.SES,
        {color:cfg.pos_color,opacity:isoOpacity,smoothness:10},
        {model:fMdl.getID(),elem:"He"});
    }
    if(showNeg[key]&&fd.n_neg>0){
      viewer.addSurface($3Dmol.SurfaceType.SES,
        {color:cfg.neg_color,opacity:isoOpacity,smoothness:10},
        {model:fMdl.getID(),elem:"Li"});
    }
  });

  if(showAxes) _drawAxes();
  viewer.render();
  _updateStats();
}

// ── XYZ 座標軸 ────────────────────────────────────────────────────────────────
let showAxes = true;

function _drawAxes(){
  const len = _axisLen();
  const r   = Math.max(len*0.018, 0.12);

  // 白色箭頭本體，軸端顯示彩色標籤
  // X 軸：白色箭頭 + 紅色 X 標籤
  viewer.addArrow({
    start:{x:0,y:0,z:0}, end:{x:len,y:0,z:0},
    color:"#ffffff", radius:r, radiusRatio:2.5
  });
  viewer.addLabel("X", {
    position:{x:len*1.1,y:0,z:0},
    fontSize:16, fontColor:"#ff4444",
    backgroundColor:"rgba(0,0,0,0)", borderThickness:0,
    fontOpacity:1.0
  });

  // Y 軸：白色箭頭 + 綠色 Y 標籤
  viewer.addArrow({
    start:{x:0,y:0,z:0}, end:{x:0,y:len,z:0},
    color:"#ffffff", radius:r, radiusRatio:2.5
  });
  viewer.addLabel("Y", {
    position:{x:0,y:len*1.1,z:0},
    fontSize:16, fontColor:"#44ff44",
    backgroundColor:"rgba(0,0,0,0)", borderThickness:0,
    fontOpacity:1.0
  });

  // Z 軸：白色箭頭 + 藍色 Z 標籤
  viewer.addArrow({
    start:{x:0,y:0,z:0}, end:{x:0,y:0,z:len},
    color:"#ffffff", radius:r, radiusRatio:2.5
  });
  viewer.addLabel("Z", {
    position:{x:0,y:0,z:len*1.1},
    fontSize:16, fontColor:"#4488ff",
    backgroundColor:"rgba(0,0,0,0)", borderThickness:0,
    fontOpacity:1.0
  });

  // 原點小球（白色）
  viewer.addSphere({center:{x:0,y:0,z:0}, radius:r*1.8,
    color:"#ffffff", opacity:0.9});
}

function _axisLen(){
  // 取目前所有 model 的原子座標，計算包圍盒，回傳合適的軸長
  try{
    let xmin=1e9,xmax=-1e9,ymin=1e9,ymax=-1e9,zmin=1e9,zmax=-1e9;
    let count=0;
    viewer.getModel().selectedAtoms({}).forEach(function(a){
      if(a.x<xmin)xmin=a.x; if(a.x>xmax)xmax=a.x;
      if(a.y<ymin)ymin=a.y; if(a.y>ymax)ymax=a.y;
      if(a.z<zmin)zmin=a.z; if(a.z>zmax)zmax=a.z;
      count++;
    });
    if(count===0) return 10;
    const span=Math.max(xmax-xmin, ymax-ymin, zmax-zmin);
    return Math.max(span*0.35, 5);
  }catch(e){ return 10; }
}

function toggleAxes(btn){
  showAxes=!showAxes;
  if(btn) btn.classList.toggle("active", showAxes);
  _schedule();
}


function toggleField(key){
  if(activeFields.has(key)){
    activeFields.delete(key);
    document.getElementById("btn-"+key).classList.remove("active");
    document.getElementById("ind-"+key).textContent="○";
  }else{
    activeFields.add(key);
    document.getElementById("btn-"+key).classList.add("active");
    document.getElementById("ind-"+key).textContent="●";
  }
  _schedule();_updateStats();
}
function showField(key){
  activeFields.add(key);
  const b=document.getElementById("btn-"+key);
  const i=document.getElementById("ind-"+key);
  if(b)b.classList.add("active");
  if(i)i.textContent="●";
  _schedule();
}
function togglePosNeg(key,which,btn){
  if(which==="pos")showPos[key]=!showPos[key];
  else             showNeg[key]=!showNeg[key];
  btn.classList.toggle("active",which==="pos"?showPos[key]:showNeg[key]);
  _schedule();
}

document.addEventListener("DOMContentLoaded",function(){
  function bind(id,setter,dispId){
    const el=document.getElementById(id);
    if(!el)return;
    el.addEventListener("input",function(){
      setter(parseFloat(this.value));
      document.getElementById(dispId).textContent=parseFloat(this.value).toFixed(2);
      _schedule();
    });
  }
  bind("sl-opa", function(v){isoOpacity=v;},"vl-opa");
  bind("sl-mopa",function(v){molOpacity=v;},"vl-mopa");
});

function _styleModel(mdl,isRef,pic50){
  const sel={model:mdl.getID()};
  const op=isRef?1.0:molOpacity;
  const r=isRef?(molStyle==="stick"?0.15:0.28):(molStyle==="stick"?0.08:0.16);
  let s;
  if(COLORING==="pic50"&&pic50!=null){
    const c=_pic50Color(pic50);
    s=molStyle==="stick"?{stick:{radius:r,color:c,opacity:op}}
     :molStyle==="sphere"?{sphere:{scale:r,color:c,opacity:op}}
     :{line:{color:c,opacity:op}};
  }else{
    s=molStyle==="stick"?{stick:{radius:r,opacity:op}}
     :molStyle==="sphere"?{sphere:{scale:r,opacity:op}}
     :{line:{opacity:op}};
  }
  viewer.setStyle(sel,s);
}
function setStyle(s){
  molStyle=s;
  document.querySelectorAll(".sb").forEach(b=>b.classList.toggle("active",b.dataset.style===s));
  _schedule();
}
function setBg(c,btn){
  bgColor=c;
  document.querySelectorAll(".bb").forEach(b=>b.classList.remove("active"));
  if(btn)btn.classList.add("active");
  _schedule();
}
function setColoring(c,btn){
  COLORING=c;
  document.querySelectorAll(".color-btn").forEach(b=>b.classList.remove("active"));
  if(btn)btn.classList.add("active");
  const leg=document.getElementById("legend");
  if(leg)leg.style.display=(c==="pic50")?"block":"none";
  _schedule();
}
function filterMols(f,btn){
  FILTER=f;
  document.querySelectorAll(".fb2").forEach(b=>b.classList.remove("active"));
  if(btn)btn.classList.add("active");
  if(!window.ALL_MOLS)return;
  molSDF=window.ALL_MOLS.filter(function(m){
    if(f==="all")return true;
    if(f==="high")return m[1]!=null&&m[1]>=PIC50.p75;
    if(f==="low") return m[1]!=null&&m[1]<=PIC50.p25;
    if(f==="ref") return m[0]===REF_NAME;
    return true;
  });
  _molCache=null;
  const cnt=document.getElementById("mol-count");
  if(cnt)cnt.textContent="顯示 "+molSDF.length+" / "+window.ALL_MOLS.length+" 個";
  _schedule();
}
function _updateStats(){
  if(activeFields.size===0){document.getElementById("stats").textContent="點擊場名開啟";return;}
  let h="";
  activeFields.forEach(function(key){
    const fd=FIELD_SDFS[key],cfg=FIELDS_CFG[key];
    if(!fd||!cfg)return;
    h+="<b style='color:"+cfg.pos_color+"'>"+cfg.label+"</b><br>";
    h+="<span style='color:"+cfg.pos_color+"'>正 iso="+fd.isoval_pos.toFixed(2)+" ("+fd.n_pos+"頂點)</span><br>";
    h+="<span style='color:"+cfg.neg_color+"'>負 iso="+fd.isoval_neg.toFixed(2)+" ("+fd.n_neg+"頂點)</span><br><br>";
  });
  document.getElementById("stats").innerHTML=h;
}
function _pic50Color(v){
  if(v==null)return"#888";
  const t=Math.max(0,Math.min(1,(v-PIC50.min)/(PIC50.max-PIC50.min+1e-9)));
  return"rgb("+Math.round(t<.5?34+t*400:234-(t-.5)*400)+","+
              Math.round(t<.5?85+t*300:235-(t-.5)*200)+","+
              Math.round(t<.5?255-t*400:55)+")";
}
"""

# ══════════════════════════════════════════════════════════════════════════════
# HTML 片段
# ══════════════════════════════════════════════════════════════════════════════

def _sidebar_fields(fields, isoval_pos=None, isoval_neg=None):
    pos_label = f"iso={isoval_pos:.3f}" if isoval_pos is not None else "iso=自動"
    neg_label = f"iso={isoval_neg:.3f}" if isoval_neg is not None else "iso=自動"
    h = '  <div class="grp"><div class="grp-t">場顯示（立體/靜電各自開關）</div>\n'
    for k in fields:
        if k not in FIELDS_CFG: continue
        cfg = FIELDS_CFG[k]
        h += (
            f'  <div class="fb" id="btn-{k}" onclick="toggleField(\'{k}\')">\n'
            f'    <div class="dot" style="background:{cfg["pos_color"]}"></div>\n'
            f'    <div class="dot" style="background:{cfg["neg_color"]}"></div>\n'
            f'    <span class="fn">{cfg["label"]}</span>\n'
            f'    <span id="ind-{k}" style="font-size:10px;color:#555">○</span></div>\n'
            f'  <div style="display:flex;gap:5px;margin-bottom:6px;margin-left:8px">\n'
            f'    <button class="sb active" id="pos-btn-{k}" '
            f'onclick="togglePosNeg(\'{k}\',\'pos\',this)" '
            f'style="border-color:{cfg["pos_color"]};color:{cfg["pos_color"]}">'
            f'正({pos_label})</button>\n'
            f'    <button class="sb active" id="neg-btn-{k}" '
            f'onclick="togglePosNeg(\'{k}\',\'neg\',this)" '
            f'style="border-color:{cfg["neg_color"]};color:{cfg["neg_color"]}">'
            f'負({neg_label})</button>\n'
            f'  </div>\n'
        )
    h += '  </div>\n'
    return h


def _sidebar_opacity():
    return (
        '  <div class="grp"><div class="grp-t">等值面透明度</div>\n'
        '  <div class="row"><label>等值面</label>'
        '  <input type="range" id="sl-opa" min="0.1" max="1.0" step="0.05" value="0.75">'
        '  <span class="rv" id="vl-opa">0.75</span></div>\n'
        '  </div>\n'
    )


def _sidebar_mol(is_overlay=False):
    mol_opacity_default = "0.7" if is_overlay else "1.0"
    h = (
        '  <div class="grp"><div class="grp-t">分子樣式</div>\n'
        '  <div class="btn-row">\n'
        '    <button class="sb active" data-style="stick"  onclick="setStyle(\'stick\')">Stick</button>\n'
        '    <button class="sb"        data-style="sphere" onclick="setStyle(\'sphere\')">Sphere</button>\n'
        '    <button class="sb"        data-style="line"   onclick="setStyle(\'line\')">Line</button>\n'
        '  </div>\n'
    )
    if is_overlay:
        h += (
            '  <div class="btn-row">\n'
            '    <button class="sb active" onclick="setColoring(\'element\',this)">元素色</button>\n'
            '    <button class="sb"        onclick="setColoring(\'pic50\',this)">pIC50色</button>\n'
            '  </div>\n'
            '  <div id="legend" style="display:none">'
            '  <div class="pic50-bar"></div>'
            '  <div class="pic50-lbl"><span>低活性</span><span>高活性</span></div></div>\n'
        )
    h += (
        f'  <div class="row"><label>{"非參考" if is_overlay else ""}透明度</label>'
        f'  <input type="range" id="sl-mopa" min="0.05" max="1.0" step="0.05" value="{mol_opacity_default}">'
        f'  <span class="rv" id="vl-mopa">{mol_opacity_default}</span></div>\n'
        '  </div>\n'
    )
    return h


def _sidebar_bg():
    return (
        '  <div class="grp"><div class="grp-t">背景</div>\n'
        '  <div class="btn-row">\n'
        '    <button class="bb active" onclick="setBg(\'black\',this)">黑</button>\n'
        '    <button class="bb"        onclick="setBg(\'white\',this)">白</button>\n'
        '    <button class="bb"        onclick="setBg(\'#1a1a2e\',this)">深藍</button>\n'
        '  </div></div>\n'
        '  <div class="grp"><div class="grp-t">顯示選項</div>\n'
        '  <div class="btn-row">\n'
        '    <button class="sb active" onclick="toggleAxes(this)" '
        'style="border-color:#aaa;color:#aaa">\n'
        '      <span style="color:#ff4444">X</span>'
        '<span style="color:#44ff44">Y</span>'
        '<span style="color:#4488ff">Z</span> 座標軸</button>\n'
        '  </div></div>\n'
    )


def _sidebar_filter(n_mols):
    return (
        '  <div class="grp"><div class="grp-t">分子篩選</div>\n'
        '  <div class="btn-row">\n'
        '    <button class="fb2 active" onclick="filterMols(\'all\',this)">全部</button>\n'
        '    <button class="fb2" onclick="filterMols(\'high\',this)">高活性↑Q75</button>\n'
        '    <button class="fb2" onclick="filterMols(\'low\',this)">低活性↓Q25</button>\n'
        '    <button class="fb2" onclick="filterMols(\'ref\',this)">僅參考</button>\n'
        '  </div>\n'
        f'  <div id="mol-count" style="font-size:11px;color:#888;margin-top:4px">顯示 {n_mols} / {n_mols} 個</div>\n'
        '  </div>\n'
    )


def _sidebar_filter_overlay(n_mols):
    return (
        '  <div class="grp"><div class="grp-t">分子篩選</div>\n'
        '  <div class="btn-row">\n'
        '    <button class="fb2 active" onclick="filterOverlay(\'all\',this)">全部</button>\n'
        '    <button class="fb2" onclick="filterOverlay(\'high\',this)">高活性↑Q75</button>\n'
        '    <button class="fb2" onclick="filterOverlay(\'low\',this)">低活性↓Q25</button>\n'
        '    <button class="fb2" onclick="filterOverlay(\'ref\',this)">僅參考</button>\n'
        '  </div>\n'
        f'  <div id="mol-count" style="font-size:11px;color:#888;margin-top:4px">顯示 {n_mols} / {n_mols} 個</div>\n'
        '  </div>\n'
    )


# overlay 模式專用 JS（追加在 _JS_CORE 之後）
_JS_OVERLAY = """
// ══════════════════════════════════════════════════════════════════════════════
// 疊合視圖專用 JS
// 核心發現：3Dmol.js 2.1.0 的 addModel('sdf') 只讀第一個分子
// 解法：每個分子各自 addModel('mol')，用 model ID 控制樣式
// 效能：requestAnimationFrame 分批加入，避免瀏覽器凍結
// ══════════════════════════════════════════════════════════════════════════════

// 每批加入的分子數（太大會讓瀏覽器卡頓）
const BATCH_ADD = 50;

// 儲存每個分子對應的 model ID
// molModelIDs[i] = model ID of MOL_META[i]
window.molModelIDs = [];

function _initOverlay(){
  _rebuildAll();
}

function _rebuildAll(){
  const el=document.getElementById("viewer");
  el.innerHTML="";
  viewer=$3Dmol.createViewer(el,{backgroundColor:bgColor,antialias:true});
  window.molModelIDs=[];

  // 顯示載入中提示
  const loading=document.getElementById("loading");
  if(loading) loading.style.display="block";

  // 分批加入分子（requestAnimationFrame 避免 UI 凍結）
  let idx=0;
  const filter=window._currentFilter||"all";
  const coloring=window._currentColoring||"element";

  function addBatch(){
    const end=Math.min(idx+BATCH_ADD, MOL_META.length);
    for(let i=idx;i<end;i++){
      const m=MOL_META[i];
      const show=
        filter==="all"  ? true :
        filter==="high" ? (m.pic50!==null&&m.pic50>=PIC50.p75) :
        filter==="low"  ? (m.pic50!==null&&m.pic50<=PIC50.p25) :
        filter==="ref"  ? m.is_ref : true;

      if(!show){
        window.molModelIDs.push(null);
        continue;
      }

      // 從 MOL_SDFS_B64 陣列取出該分子的 SDF
      const sdf=b64dec(MOL_SDFS_B64[i]);
      const mdl=viewer.addModel(sdf,"mol");
      window.molModelIDs.push(mdl.getID());

      const op=m.is_ref?1.0:molOpacity;
      const r =m.is_ref?(molStyle==="stick"?0.15:0.28)
                       :(molStyle==="stick"?0.08:0.16);
      let s;
      if(coloring==="pic50"&&m.pic50!==null){
        const c=_pic50Color(m.pic50);
        s=molStyle==="stick"  ?{stick:{radius:r,color:c,opacity:op}}
         :molStyle==="sphere" ?{sphere:{scale:r,color:c,opacity:op}}
         :                    {line:{color:c,opacity:op}};
      }else{
        s=molStyle==="stick"  ?{stick:{radius:r,opacity:op}}
         :molStyle==="sphere" ?{sphere:{scale:r,opacity:op}}
         :                    {line:{opacity:op}};
      }
      viewer.setStyle({model:mdl.getID()},s);
    }
    idx=end;

    // 更新計數
    const shown=window.molModelIDs.filter(id=>id!==null).length;
    const cnt=document.getElementById("mol-count");
    if(cnt) cnt.textContent="載入中 "+shown+" / "+MOL_META.length+" 個";

    if(idx<MOL_META.length){
      requestAnimationFrame(addBatch);
    } else {
      // 全部加完，加場並渲染
      _addFieldSurfaces();
      if(showAxes) _drawAxes();
      viewer.zoomTo();
      viewer.render();
      if(loading) loading.style.display="none";
      if(cnt) cnt.textContent="顯示 "+shown+" / "+MOL_META.length+" 個";

      // 開啟所有場按鈕
      Object.keys(FIELD_SDFS).forEach(function(k){
        activeFields.add(k);
        const b=document.getElementById("btn-"+k);
        const i=document.getElementById("ind-"+k);
        if(b) b.classList.add("active");
        if(i) i.textContent="●";
      });
      _updateStats();
    }
  }
  requestAnimationFrame(addBatch);
}

function _addFieldSurfaces(){
  activeFields.forEach(function(key){
    const fd=FIELD_SDFS[key],cfg=FIELDS_CFG[key];
    if(!fd||!fd.sdf) return;
    const fMdl=viewer.addModel(fd.sdf,"mol");
    viewer.setStyle({model:fMdl.getID()},{});
    if(showPos[key]&&fd.n_pos>0)
      viewer.addSurface($3Dmol.SurfaceType.SES,
        {color:cfg.pos_color,opacity:isoOpacity,smoothness:10},
        {model:fMdl.getID(),elem:"He"});
    if(showNeg[key]&&fd.n_neg>0)
      viewer.addSurface($3Dmol.SurfaceType.SES,
        {color:cfg.neg_color,opacity:isoOpacity,smoothness:10},
        {model:fMdl.getID(),elem:"Li"});
  });
}

// ── _doRebuild：切換場時只重建場，不重建分子 ─────────────────────────────────
function _doRebuild(){
  _timer=null;
  if(!viewer||window.molModelIDs.length===0){
    _rebuildAll(); return;
  }
  // 移除所有場的假原子 model
  // 找出哪些 model ID 不在 molModelIDs 裡的就是場的 model
  const molIDs=new Set(window.molModelIDs.filter(id=>id!==null));
  // 最簡單可靠：完整重建
  // 只要分子不多，速度可接受
  _rebuildAll();
}

// ── 篩選：完整重建（分子顯示條件改變）──────────────────────────────────────
function filterOverlay(f,btn){
  window._currentFilter=f;
  document.querySelectorAll(".fb2").forEach(b=>b.classList.remove("active"));
  if(btn) btn.classList.add("active");
  _rebuildAll();
}

// ── 樣式切換 ──────────────────────────────────────────────────────────────────
function setStyle(s){
  molStyle=s;
  document.querySelectorAll(".sb").forEach(b=>
    b.classList.toggle("active",b.dataset.style===s));
  _rebuildAll();
}
function setMolOpacity(val){
  molOpacity=parseFloat(val);
  _rebuildAll();
}
function setColoring(c,btn){
  COLORING=c;
  window._currentColoring=c;
  document.querySelectorAll(".color-btn").forEach(b=>b.classList.remove("active"));
  if(btn) btn.classList.add("active");
  const leg=document.getElementById("legend");
  if(leg) leg.style.display=(c==="pic50")?"block":"none";
  _rebuildAll();
}

// ── 場開關 ────────────────────────────────────────────────────────────────────
function toggleField(key){
  if(activeFields.has(key)){
    activeFields.delete(key);
    const b=document.getElementById("btn-"+key);
    const i=document.getElementById("ind-"+key);
    if(b) b.classList.remove("active");
    if(i) i.textContent="○";
  }else{
    activeFields.add(key);
    const b=document.getElementById("btn-"+key);
    const i=document.getElementById("ind-"+key);
    if(b) b.classList.add("active");
    if(i) i.textContent="●";
  }
  _schedule();
  _updateStats();
}
function showField(key){
  activeFields.add(key);
  const b=document.getElementById("btn-"+key);
  const i=document.getElementById("ind-"+key);
  if(b) b.classList.add("active");
  if(i) i.textContent="●";
}

// ── 統計 ──────────────────────────────────────────────────────────────────────
function _updateStats(){
  if(activeFields.size===0){
    document.getElementById("stats").textContent="點擊場名開啟";return;
  }
  let h="";
  activeFields.forEach(function(key){
    const fd=FIELD_SDFS[key],cfg=FIELDS_CFG[key];
    if(!fd||!cfg)return;
    h+="<b style='color:"+cfg.pos_color+"'>"+cfg.label+"</b><br>";
    h+="<span style='color:"+cfg.pos_color+"'>正 iso="+fd.isoval_pos.toFixed(2)+" ("+fd.n_pos+"頂點)</span><br>";
    h+="<span style='color:"+cfg.neg_color+"'>負 iso="+fd.isoval_neg.toFixed(2)+" ("+fd.n_neg+"頂點)</span><br><br>";
  });
  document.getElementById("stats").innerHTML=h;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# 生成 HTML（共用組裝函式）
# ══════════════════════════════════════════════════════════════════════════════

def _assemble_html(title, subtitle, sidebar_html, js_data, init_js, extra_js=""):
    """組裝完整 HTML 頁面。"""
    return (
        '<!DOCTYPE html><html lang="zh-TW"><head><meta charset="UTF-8">\n'
        f'<title>{title}</title>\n'
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.1.0/3Dmol-min.js"></script>\n'
        '<style>' + _CSS + '</style></head><body>\n'
        '<div id="sidebar">\n'
        f'  <div class="title">{title}</div>\n'
        f'  <div class="sub">{subtitle}</div>\n'
        + sidebar_html +
        '  <div class="grp"><div class="grp-t">場資訊</div>'
        '  <div id="stats">點擊場名開啟</div></div>\n'
        '</div>\n'
        '<div id="viewer-wrap">'
        '<div id="viewer"></div>'
        '<div id="loading">載入中…</div>'
        '<div id="hint">左鍵旋轉 · 右鍵平移 · 滾輪縮放</div>'
        '</div>\n'
        '<script>\n'
        + js_data +
        _JS_CORE +
        extra_js +
        init_js +
        '</script></body></html>\n'
    )

# ══════════════════════════════════════════════════════════════════════════════
# 生成單分子 HTML
# ══════════════════════════════════════════════════════════════════════════════

def generate_html(npz_path, output_path, isoval_pos=None, isoval_neg=None):
    try:
        meta, nums, coords, fields = _load_npz(npz_path)
        mol_name   = meta["mol_name"]
        pred_pic50 = meta.get("pred_pic50", 0.0)
        resolution = meta.get("resolution", 1.0)
        origin     = meta["origin"]
        axes       = meta["axes"]

        if not fields:
            print(_c(f"  [跳過] {mol_name}：無立體/靜電場資料", "yellow"))
            return False

        # 計算等值面網格
        print(f"  {_c(mol_name,'yellow')} 計算等值面…", end="", flush=True)
        field_sdfs_b64 = {}
        for k, arr in fields.items():
            mesh = _marching_cubes(arr, origin, axes, isoval_pos, isoval_neg)
            fake_sdf, n_pos, n_neg = _verts_to_fake_sdf(
                mesh['pos']['verts'], mesh['neg']['verts'], mol_name)
            fsd = {'sdf': fake_sdf, 'n_pos': n_pos, 'n_neg': n_neg,
                   'isoval_pos': mesh['isoval_pos'], 'isoval_neg': mesh['isoval_neg']}
            field_sdfs_b64[k] = _b64(json.dumps(fsd, separators=(',',':')))
            print(f" {k}(+{n_pos}/-{n_neg}頂點)", end="", flush=True)
        print()

        sdf_js  = json.dumps(_mol_sdf(nums, coords, mol_name))
        cfg_js  = json.dumps({k: FIELDS_CFG[k] for k in fields if k in FIELDS_CFG})

        js_data = (
            f'const FIELD_SDFS_B64 = {json.dumps(field_sdfs_b64)};\n'
            f'const FIELDS_CFG  = {cfg_js};\n'
            f'const REF_NAME    = null;\n'
            f'const PIC50       = {{min:0,max:1,p25:0,p75:1}};\n'
        )
        init_js = (
            'document.getElementById("loading").style.display="none";\n'
            f'molSDF = {sdf_js};\n'
            '_schedule();\n'
            'setTimeout(function(){\n'
            '  Object.keys(FIELD_SDFS).forEach(function(k){ '
            '    activeFields.add(k);'
            '    document.getElementById("btn-"+k).classList.add("active");'
            '    document.getElementById("ind-"+k).textContent="●";'
            '  });\n'
            '  _doRebuild();\n'
            '  _updateStats();\n'
            '}, 100);\n'
        )

        sidebar = (
            _sidebar_fields(fields, isoval_pos, isoval_neg) +
            _sidebar_opacity() +
            _sidebar_mol(False) +
            _sidebar_bg()
        )

        html = _assemble_html(
            mol_name,
            f'pIC50: {pred_pic50:.4f}  ·  解析度 {resolution} Å',
            sidebar, js_data, init_js)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        kb = os.path.getsize(output_path) // 1024
        print(f"  → {_c(os.path.basename(output_path),'cyan')}  {kb} KB")
        return True

    except Exception:
        print(_c(f"\n  ✗ {traceback.format_exc(limit=3)}", "red"))
        return False

# ══════════════════════════════════════════════════════════════════════════════
# 生成疊合視圖 HTML
# ══════════════════════════════════════════════════════════════════════════════

def generate_overlay_html(npz_dir, sdf_path, output_path,
                           ref_mol=None, max_mols=500,
                           isoval_pos=None, isoval_neg=None):
    try:
        # 讀取所有 npz
        pic50_map  = {}
        fields_map = {}
        for fn in sorted(os.listdir(npz_dir)):
            if not fn.endswith(".npz") or fn.startswith("_"): continue
            try:
                meta, nums, coords, fields = _load_npz(os.path.join(npz_dir,fn))
                pic50_map[meta["mol_name"]] = meta.get("pred_pic50", 0.0)
                if fields:
                    fields_map[meta["mol_name"]] = (meta, nums, coords, fields)
            except: pass

        if not pic50_map:
            print(_c("  找不到分子資料","red")); return False
        print(f"  讀取 {len(pic50_map)} 個分子")

        # 選參考分子
        if ref_mol and ref_mol in fields_map:
            ref_name = ref_mol
        else:
            ref_name = max((k for k in pic50_map if k in fields_map),
                           key=lambda k: pic50_map[k], default=None)
        if not ref_name:
            print(_c("  無有效參考分子","red")); return False
        print(f"  參考分子：{_c(ref_name,'yellow')} (pIC50={pic50_map[ref_name]:.4f})")

        ref_meta, ref_nums, ref_coords, ref_fields = fields_map[ref_name]
        origin = ref_meta["origin"]
        axes   = ref_meta["axes"]

        # 計算等值面（只用參考分子的場）
        print(f"  計算等值面…", end="", flush=True)
        field_sdfs_b64 = {}
        for k, arr in ref_fields.items():
            mesh = _marching_cubes(arr, origin, axes, isoval_pos, isoval_neg)
            field_sdfs_b64[k] = _b64(json.dumps(fsd, separators=(',',':')))
            print(f" {k}(+{mesh['n_verts_pos']}/-{mesh['n_verts_neg']}頂點)",
                  end="", flush=True)
        print()

        # 讀取 SDF
        if not os.path.isfile(sdf_path):
            print(_c(f"  找不到 SDF：{sdf_path}","red")); return False
        all_sdf = _read_sdf(sdf_path)
        n_total = len(all_sdf)
        if n_total > max_mols:
            print(_c(f"  ⚠ 截取前 {max_mols} 個（共 {n_total} 個）","yellow"))
            all_sdf = all_sdf[:max_mols]
        n = len(all_sdf)
        print(f"  載入 {n} 個分子（SDF）")

        # ── 合併成單一 SDF，一次 addModel 載入全部 ───────────────────────────
        mol_meta    = []
        mol_sdfs_b64 = []   # 每個分子各自的 base64 SDF
        for mol_name_i, sdf_block in all_sdf:
            p50 = pic50_map.get(mol_name_i, None)
            mol_meta.append({
                "name":   mol_name_i,
                "pic50":  round(p50,4) if p50 is not None else None,
                "is_ref": mol_name_i == ref_name,
            })
            mol_sdfs_b64.append(_b64(sdf_block.strip()))

        total_kb = sum(len(b) for b in mol_sdfs_b64) // 1024
        print(f"  {len(mol_sdfs_b64)} 個分子 SDF，總計 {total_kb} KB (base64)")

        vals = [m["pic50"] for m in mol_meta if m["pic50"] is not None]
        pic50_range = {
            "min": round(float(min(vals)),4) if vals else 0,
            "max": round(float(max(vals)),4) if vals else 1,
            "p25": round(float(np.percentile(vals,25)),4) if vals else 0,
            "p75": round(float(np.percentile(vals,75)),4) if vals else 1,
        }

        cfg_js   = json.dumps({k: FIELDS_CFG[k] for k in ref_fields if k in FIELDS_CFG})
        meta_js  = json.dumps(mol_meta)

        js_data = (
            f'const FIELD_SDFS_B64 = {json.dumps(field_sdfs_b64)};\n'
            f'const FIELDS_CFG     = {cfg_js};\n'
            f'const REF_NAME       = {json.dumps(ref_name)};\n'
            f'const PIC50          = {json.dumps(pic50_range)};\n'
            f'const MOL_META       = {json.dumps(mol_meta)};\n'
            f'const MOL_SDFS_B64   = {json.dumps(mol_sdfs_b64)};\n'
        )
        init_js = (
            f'{_ISO_OPACITY_INIT}'
            'document.getElementById("loading").style.display="none";\n'
            'window._currentFilter   = "all";\n'
            'window._currentColoring = "element";\n'
            '_initOverlay();\n'
        )

        ref_pic50 = pic50_map[ref_name]
        sidebar = (
            _sidebar_fields(ref_fields, isoval_pos, isoval_neg) +
            _sidebar_opacity() +
            _sidebar_filter_overlay(n) +
            _sidebar_mol(True) +
            _sidebar_bg()
        )

        html = _assemble_html(
            "QSAR 疊合視圖",
            f'{n} 個分子 · 場圖：<span style="color:#e94560">{ref_name}</span>'
            f' (pIC50={ref_pic50:.4f})',
            sidebar, js_data, init_js,
            extra_js=_JS_OVERLAY)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        mb = os.path.getsize(output_path)/1024/1024
        print(f"  → {_c(output_path,'cyan')}  {mb:.1f} MB")
        return True

    except Exception:
        print(_c(f"\n  ✗ {traceback.format_exc(limit=4)}","red"))
        return False

# ══════════════════════════════════════════════════════════════════════════════
# 分批疊合視圖：每500個分子一個HTML + 總場圖
# ══════════════════════════════════════════════════════════════════════════════

def generate_batch_overlay(npz_dir, sdf_path, output_dir,
                            batch_size=500, ref_mol=None,
                            isoval_pos=None, isoval_neg=None):
    """
    分批生成疊合視圖：
      - 每 batch_size 個分子生成一個 HTML
      - 額外生成一個「總場圖」HTML（體素平均後的場，不含分子）
      - 全局統一使用 pIC50 最高的分子作為場圖來源

    輸出：
      output_dir/batch_001.html  ... batch_NNN.html
      output_dir/total_field.html  （所有批次場的體素平均）
      output_dir/index.html
    """
    t0 = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. 讀取所有 npz ──────────────────────────────────────────────────────
    print(f"\n  {_c('Step 1','bold')} 讀取 npz 資料…")
    pic50_map  = {}
    fields_map = {}
    for fn in sorted(os.listdir(npz_dir)):
        if not fn.endswith(".npz") or fn.startswith("_"): continue
        try:
            meta, nums, coords, fields = _load_npz(os.path.join(npz_dir, fn))
            name = meta["mol_name"]
            pic50_map[name] = meta.get("pred_pic50", 0.0)
            if fields:
                fields_map[name] = (meta, nums, coords, fields)
        except: pass

    if not pic50_map:
        print(_c("  找不到分子資料","red")); return False
    print(f"  {len(pic50_map)} 個分子，{len(fields_map)} 個有場資料")

    # ── 2. 全局參考分子（pIC50 最高）────────────────────────────────────────
    if ref_mol and ref_mol in fields_map:
        ref_name = ref_mol
    else:
        ref_name = max((k for k in pic50_map if k in fields_map),
                       key=lambda k: pic50_map[k], default=None)
    if not ref_name:
        print(_c("  無有效參考分子","red")); return False
    ref_meta, ref_nums, ref_coords, ref_fields = fields_map[ref_name]
    print(f"  全局參考分子：{_c(ref_name,'yellow')} (pIC50={pic50_map[ref_name]:.4f})")

    origin = ref_meta["origin"]
    axes   = ref_meta["axes"]

    # ── 3. 計算全局參考場（用於所有批次 HTML）───────────────────────────────
    print(f"\n  {_c('Step 2','bold')} 計算全局參考場…", end="", flush=True)
    field_sdfs_b64 = {}
    ref_meshes     = {}   # key → mesh dict（用於總場圖計算）
    for k, arr in ref_fields.items():
        mesh = _marching_cubes(arr, origin, axes, isoval_pos, isoval_neg)
        fake_sdf, n_pos, n_neg = _verts_to_fake_sdf(
            mesh['pos']['verts'], mesh['neg']['verts'], ref_name)
        fsd = {'sdf': fake_sdf, 'n_pos': n_pos, 'n_neg': n_neg,
               'isoval_pos': mesh['isoval_pos'], 'isoval_neg': mesh['isoval_neg']}
        field_sdfs_b64[k] = _b64(json.dumps(fsd, separators=(',',':')))
        ref_meshes[k]     = {'arr': ref_fields[k], 'mesh': mesh}
        print(f" {k}(+{n_pos}/-{n_neg})", end="", flush=True)
    print()

    # ── 4. 讀取 SDF，分批 ────────────────────────────────────────────────────
    print(f"\n  {_c('Step 3','bold')} 讀取 SDF 並分批…")
    if not os.path.isfile(sdf_path):
        print(_c(f"  找不到 SDF：{sdf_path}","red")); return False
    all_sdf = _read_sdf(sdf_path)
    n_total  = len(all_sdf)
    batches  = [all_sdf[i:i+batch_size]
                for i in range(0, n_total, batch_size)]
    print(f"  {n_total} 個分子 → {len(batches)} 批（每批 {batch_size} 個）")

    vals_all = [pic50_map.get(n, 0.0) for n, _ in all_sdf if n in pic50_map]
    pic50_range = {
        "min": round(float(min(vals_all)),4) if vals_all else 0,
        "max": round(float(max(vals_all)),4) if vals_all else 1,
        "p25": round(float(np.percentile(vals_all,25)),4) if vals_all else 0,
        "p75": round(float(np.percentile(vals_all,75)),4) if vals_all else 1,
    }

    # ── 5. 生成每批 HTML ──────────────────────────────────────────────────────
    print(f"\n  {_c('Step 4','bold')} 生成批次 HTML…")
    cfg_js = json.dumps({k: FIELDS_CFG[k] for k in ref_fields if k in FIELDS_CFG})
    batch_files = []

    for bi, batch in enumerate(batches):
        batch_num  = bi + 1
        html_name  = f"batch_{batch_num:03d}.html"
        html_path  = os.path.join(output_dir, html_name)
        n_batch    = len(batch)
        start_mol  = bi * batch_size + 1
        end_mol    = start_mol + n_batch - 1

        print(f"  [{batch_num}/{len(batches)}] mol {start_mol}~{end_mol}  ", end="", flush=True)

        # 合併 SDF
        mol_meta     = []
        mol_sdfs_b64 = []
        for mol_name_i, sdf_block in batch:
            p50 = pic50_map.get(mol_name_i, None)
            mol_meta.append({
                "name":   mol_name_i,
                "pic50":  round(p50,4) if p50 is not None else None,
                "is_ref": mol_name_i == ref_name,
            })
            mol_sdfs_b64.append(_b64(sdf_block.strip()))

        batch_pic50 = [m["pic50"] for m in mol_meta if m["pic50"] is not None]
        bpr = {
            "min": round(float(min(batch_pic50)),4) if batch_pic50 else pic50_range["min"],
            "max": round(float(max(batch_pic50)),4) if batch_pic50 else pic50_range["max"],
            "p25": round(float(np.percentile(batch_pic50,25)),4) if batch_pic50 else pic50_range["p25"],
            "p75": round(float(np.percentile(batch_pic50,75)),4) if batch_pic50 else pic50_range["p75"],
        }

        js_data = (
            f'const FIELD_SDFS_B64   = {json.dumps(field_sdfs_b64)};\n'
            f'const FIELDS_CFG       = {cfg_js};\n'
            f'const REF_NAME         = {json.dumps(ref_name)};\n'
            f'const PIC50            = {json.dumps(bpr)};\n'
            f'const MOL_META       = {json.dumps(mol_meta)};\n'
            f'const MOL_SDFS_B64   = {json.dumps(mol_sdfs_b64)};\n'
        )
        init_js = (
            'document.getElementById("loading").style.display="none";\n'
            'window._currentFilter   = "all";\n'
            'window._currentColoring = "element";\n'
            '_initOverlay();\n'
        )
        subtitle = (
            f'批次 {batch_num}/{len(batches)}  ·  分子 {start_mol}~{end_mol}'
            f'  ·  場圖：<span style="color:#e94560">{ref_name}</span>'
        )
        sidebar = (
            _sidebar_fields(ref_fields, isoval_pos, isoval_neg) +
            _sidebar_opacity() +
            _sidebar_filter_overlay(n_batch) +
            _sidebar_mol(True) +
            _sidebar_bg()
        )
        html = _assemble_html(
            f"QSAR 疊合視圖 批次{batch_num}",
            subtitle, sidebar, js_data, init_js,
            extra_js=_JS_OVERLAY)

        with open(html_path, "w", encoding="utf-8") as fout:
            fout.write(html)
        kb = os.path.getsize(html_path) // 1024
        print(f"→ {html_name} ({kb} KB)")
        batch_files.append((html_name, batch_num, start_mol, end_mol, n_batch))

    # ── 6. 計算總場圖（體素平均）────────────────────────────────────────────
    print(f"\n  {_c('Step 5','bold')} 計算總場圖（所有批次代表分子場的體素平均）…")

    # 收集所有有場資料的分子（每批取 pIC50 最高的，全局只用一個場）
    # 這裡採用「全局最高 pIC50 的前 N 個分子場平均」策略
    # 取前 min(len(batches)*2, 20) 個高活性分子做平均
    n_avg = min(len(batches) * 2, 20)
    top_mols = sorted(
        (k for k in pic50_map if k in fields_map),
        key=lambda k: pic50_map[k], reverse=True
    )[:n_avg]
    print(f"  使用前 {len(top_mols)} 個高活性分子的場做體素平均")

    # 確認所有場的 shape 一致（若不一致則跳過形狀不符的）
    ref_shape = {k: v.shape for k, v in ref_fields.items()}
    avg_fields = {k: np.zeros(ref_shape[k], dtype=np.float64)
                  for k in ref_fields if k in FIELDS_CFG}
    n_contributed = {k: 0 for k in avg_fields}

    for mol_name_i in top_mols:
        m_meta, m_nums, m_coords, m_fields = fields_map[mol_name_i]
        for k in avg_fields:
            if k in m_fields and m_fields[k].shape == ref_shape.get(k):
                avg_fields[k] += m_fields[k].astype(np.float64)
                n_contributed[k] += 1

    # 正規化為平均
    for k in avg_fields:
        if n_contributed[k] > 0:
            avg_fields[k] /= n_contributed[k]
            print(f"  {k}: {n_contributed[k]} 個分子平均  "
                  f"max={avg_fields[k].max():.4f} min={avg_fields[k].min():.4f}")

    # 計算平均場的等值面
    print(f"  計算平均場等值面…", end="", flush=True)
    avg_field_sdfs_b64 = {}
    for k, arr in avg_fields.items():
        if n_contributed[k] == 0: continue
        mesh = _marching_cubes(arr.astype(np.float32), origin, axes,
                               isoval_pos, isoval_neg)
        fake_sdf, n_pos, n_neg = _verts_to_fake_sdf(
            mesh['pos']['verts'], mesh['neg']['verts'], "avg_field")
        fsd = {'sdf': fake_sdf, 'n_pos': n_pos, 'n_neg': n_neg,
               'isoval_pos': mesh['isoval_pos'], 'isoval_neg': mesh['isoval_neg']}
        avg_field_sdfs_b64[k] = _b64(json.dumps(fsd, separators=(',',':')))
        print(f" {k}(+{n_pos}/-{n_neg})", end="", flush=True)
    print()

    # ── 7. 總場圖 HTML（含參考分子 + 平均場）────────────────────────────────
    total_html_name = "total_field.html"
    total_html_path = os.path.join(output_dir, total_html_name)

    # 參考分子的 SDF
    ref_sdf_str = _mol_sdf(ref_nums, ref_coords, ref_name)
    for mol_name_i, sdf_block in all_sdf:
        if mol_name_i == ref_name:
            ref_sdf_str = sdf_block.strip()
            break
    ref_sdf_b64 = _b64(ref_sdf_str)

    # top_mols 的 SDF（優先從 fields_map 座標取，確保全部都有）
    top_mol_sdfs_b64 = []
    all_sdf_names = {n for n, _ in all_sdf}
    all_sdf_dict  = {n: s for n, s in all_sdf}
    for mn in top_mols:
        if mn in all_sdf_dict:
            top_mol_sdfs_b64.append(_b64(all_sdf_dict[mn].strip()))
        elif mn in fields_map:
            m_meta2, m_nums2, m_coords2, _ = fields_map[mn]
            top_mol_sdfs_b64.append(_b64(_mol_sdf(m_nums2, m_coords2, mn)))

    print(f"  總場圖顯示分子：{len(top_mol_sdfs_b64)} 個（含參考分子）")

    # top_mols 的 MOL_META（讓 _initOverlay 能正確處理）
    top_mol_meta = []
    for mn in top_mols:
        p50 = pic50_map.get(mn, None)
        top_mol_meta.append({
            "name":   mn,
            "pic50":  round(p50, 4) if p50 is not None else None,
            "is_ref": mn == ref_name,
        })

    # 總場圖的 pIC50 範圍（用 top_mols 的值）
    top_vals = [m["pic50"] for m in top_mol_meta if m["pic50"] is not None]
    top_pic50_range = {
        "min": round(float(min(top_vals)), 4) if top_vals else pic50_range["min"],
        "max": round(float(max(top_vals)), 4) if top_vals else pic50_range["max"],
        "p25": round(float(np.percentile(top_vals, 25)), 4) if top_vals else pic50_range["p25"],
        "p75": round(float(np.percentile(top_vals, 75)), 4) if top_vals else pic50_range["p75"],
    }

    # 完全使用與批次 HTML 相同的架構，走 _initOverlay() 流程
    total_js_data = (
        f'const FIELD_SDFS_B64 = {json.dumps(avg_field_sdfs_b64)};\n'
        f'const FIELDS_CFG     = {cfg_js};\n'
        f'const REF_NAME       = {json.dumps(ref_name)};\n'
        f'const PIC50          = {json.dumps(top_pic50_range)};\n'
        f'const MOL_META       = {json.dumps(top_mol_meta)};\n'
        f'const MOL_SDFS_B64   = {json.dumps(top_mol_sdfs_b64)};\n'
    )
    total_init_js = (
        'document.getElementById("loading").style.display="none";\n'
        'window._currentFilter   = "all";\n'
        'window._currentColoring = "element";\n'
        '_initOverlay();\n'
    )
    total_subtitle = (
        f'前 {len(top_mols)} 個高活性分子場的體素平均  ·  '
        f'共 {n_total} 個分子 · {len(batches)} 個批次  ·  '
        f'參考分子：<span style="color:#e94560">{ref_name}</span>'
    )
    total_sidebar = (
        _sidebar_fields(avg_fields, isoval_pos, isoval_neg) +
        _sidebar_opacity() +
        _sidebar_filter_overlay(len(top_mols)) +
        _sidebar_mol(True) +
        _sidebar_bg()
    )
    total_html = _assemble_html(
        "QSAR 總場圖（體素平均）",
        total_subtitle, total_sidebar,
        total_js_data, total_init_js,
        extra_js=_JS_OVERLAY)

    with open(total_html_path, "w", encoding="utf-8") as fout:
        fout.write(total_html)
    kb = os.path.getsize(total_html_path) // 1024
    print(f"  總場圖 → {_c(total_html_name,'green')} ({kb} KB)")

    # ── 8. 索引頁 ─────────────────────────────────────────────────────────────
    index_path = os.path.join(output_dir, "index.html")
    batch_cards = "".join(
        f'<a class="card" href="{hf}" target="_blank">'
        f'<div class="cn">批次 {bn}</div>'
        f'<div class="cp">分子 {sm}~{em}（共 {nm} 個）</div></a>'
        for hf, bn, sm, em, nm in batch_files
    )
    index_html = (
        '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>QSAR 批次疊合索引</title>'
        '<style>*{box-sizing:border-box;margin:0;padding:0;}'
        'body{background:#1a1a2e;color:#e0e0e0;font-family:"Segoe UI",sans-serif;padding:20px;}'
        'h1{color:#00d4ff;margin-bottom:6px;} .sub{color:#888;font-size:13px;margin-bottom:20px;}'
        '.total-btn{display:block;width:100%;padding:14px 20px;margin-bottom:16px;'
        'background:linear-gradient(135deg,#0f3460,#16213e);'
        'border:2px solid #00d4ff;border-radius:10px;text-decoration:none;'
        'color:#00d4ff;font-size:15px;font-weight:bold;text-align:center;transition:all .2s;}'
        '.total-btn:hover{background:#0f3460;transform:translateY(-2px);}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;}'
        '.card{background:#16213e;border:1px solid #0f3460;border-radius:8px;padding:12px;'
        'cursor:pointer;transition:all .2s;text-decoration:none;display:block;}'
        '.card:hover{border-color:#00d4ff;transform:translateY(-2px);}'
        '.cn{color:#00d4ff;font-size:14px;font-weight:bold;margin-bottom:4px;}'
        '.cp{color:#888;font-size:12px;}'
        '</style></head><body>'
        f'<h1>QSAR 批次疊合視圖</h1>'
        f'<div class="sub">共 {n_total} 個分子 · {len(batches)} 個批次 · '
        f'全局參考分子：{ref_name} (pIC50={pic50_map[ref_name]:.4f})</div>'
        f'<a class="total-btn" href="{total_html_name}" target="_blank">'
        f'★ 總場圖（前 {len(top_mols)} 個高活性分子體素平均）</a>'
        f'<div class="grid">{batch_cards}</div>'
        '</body></html>'
    )
    with open(index_path, "w", encoding="utf-8") as fout:
        fout.write(index_html)

    elapsed = time.perf_counter() - t0
    print(f"\n  {_c('完成','bold','green')}  耗時 {elapsed:.1f}s")
    print(f"  批次 HTML  : {len(batches)} 個")
    print(f"  總場圖     : {_c(total_html_name,'cyan')}")
    print(f"  索引頁     : {_c('index.html','cyan')}")
    print(f"  輸出目錄   : {_c(output_dir,'yellow')}")
    return True



def batch_generate(input_dir, output_dir, isoval_pos=None, isoval_neg=None, resume=True):
    npzs = sorted(f for f in os.listdir(input_dir)
                  if f.endswith(".npz") and not f.startswith("_"))
    if not npzs:
        print(_c(f"  找不到 .npz：{input_dir}","red")); return
    os.makedirs(output_dir, exist_ok=True)
    ok=skip=fail=0; t0=time.perf_counter(); entries=[]
    for i,fn in enumerate(npzs):
        name=fn[:-4]; out=os.path.join(output_dir,name+".html")
        if resume and os.path.exists(out):
            skip+=1
            try:
                meta,_,_,_=_load_npz(os.path.join(input_dir,fn))
                entries.append((name,name+".html",meta.get("pred_pic50",0)))
            except: entries.append((name,name+".html",0))
            continue
        print(f"  [{i+1}/{len(npzs)}] ",end="")
        if generate_html(os.path.join(input_dir,fn), out, isoval_pos, isoval_neg):
            ok+=1
            try:
                meta,_,_,_=_load_npz(os.path.join(input_dir,fn))
                entries.append((name,name+".html",meta.get("pred_pic50",0)))
            except: entries.append((name,name+".html",0))
        else: fail+=1

    if entries:
        cards="".join(
            f'<a class="card" href="{hf}" target="_blank" data-n="{n.lower()}">'
            f'<div class="cn">{n}</div><div class="cp">pIC50:{p:.4f}</div></a>'
            for n,hf,p in sorted(entries))
        idx=(
            '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>QSAR索引</title>'
            '<style>*{box-sizing:border-box;margin:0;padding:0;}'
            'body{background:#1a1a2e;color:#e0e0e0;font-family:"Segoe UI",sans-serif;padding:20px;}'
            'h1{color:#00d4ff;margin-bottom:14px;}'
            '.search{width:380px;padding:7px 12px;background:#0f3460;border:1px solid #0f3460;'
            'border-radius:7px;color:#fff;font-size:13px;margin-bottom:16px;outline:none;display:block;}'
            '.search:focus{border-color:#00d4ff;}'
            '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;}'
            '.card{background:#16213e;border:1px solid #0f3460;border-radius:8px;padding:12px;'
            'cursor:pointer;transition:all .2s;text-decoration:none;display:block;}'
            '.card:hover{border-color:#00d4ff;transform:translateY(-2px);}'
            '.cn{color:#00d4ff;font-size:14px;font-weight:bold;margin-bottom:3px;word-break:break-all;}'
            '.cp{color:#e94560;font-size:12px;}'
            '</style></head><body>'
            f'<h1>QSAR 場圖索引 ({len(entries)} 個分子)</h1>'
            '<input class="search" placeholder="搜尋分子名稱…" oninput="'
            'document.querySelectorAll(\'.card\').forEach(c=>'
            'c.style.display=c.dataset.n.includes(this.value.toLowerCase())?\'\':\'none\')">'
            f'<div class="grid">{cards}</div>'
            '</body></html>')
        with open(os.path.join(output_dir,"index.html"),"w",encoding="utf-8") as f:
            f.write(idx)
        print(f"\n  索引 → index.html")

    print(f"\n  ✓{ok} ⊘{skip} ✗{fail}  {time.perf_counter()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# 互動式 CLI 工具函式
# ══════════════════════════════════════════════════════════════════════════════

def _banner():
    print("\n" + _hr("═"))
    print(_c("  QSAR 場圖檢視器 v5", "bold", "cyan") +
          _c("  ·  Marching Cubes 預計算等值面", "dim"))
    print(_hr("═") + "\n")

def _ask(label, default=None, validator=None, allow_empty=False):
    hint = f" [{_c(str(default), 'dim')}]" if default is not None else ""
    while True:
        v = input(f"  {_c('▸','cyan')} {label}{hint}: ").strip()
        v = v or (str(default) if default is not None else "")
        if not v and not allow_empty:
            print(_c("    請輸入值", "red")); continue
        if validator and v:
            ok, msg = validator(v)
            if not ok:
                print(_c(f"    {msg}", "red")); continue
        return v

def _ask_float(label, default, lo=None, hi=None):
    def _v(v):
        try:
            f = float(v)
        except ValueError:
            return False, "請輸入數字"
        if lo is not None and f < lo: return False, f"最小值 {lo}"
        if hi is not None and f > hi: return False, f"最大值 {hi}"
        return True, ""
    return float(_ask(label, default, _v))

def _ask_int(label, default, lo=None, hi=None):
    def _v(v):
        try:
            i = int(v)
        except ValueError:
            return False, "請輸入整數"
        if lo is not None and i < lo: return False, f"最小值 {lo}"
        if hi is not None and i > hi: return False, f"最大值 {hi}"
        return True, ""
    return int(_ask(label, default, _v))

def _ask_path(label, must_exist=True, exts=None, default=None):
    def _v(v):
        if must_exist and not os.path.exists(v):
            return False, f"找不到路徑：{v}"
        if exts and not any(v.lower().endswith(e) for e in exts):
            return False, f"需為 {'/'.join(exts)} 格式"
        return True, ""
    return _ask(label, default, _v)

def _ask_choice(label, choices, default=None):
    """choices: list of (key, description)"""
    print(f"\n  {_c(label, 'bold')}")
    for i, (k, desc) in enumerate(choices, 1):
        marker = _c("●", "cyan") if default and k == default else _c("○", "dim")
        print(f"    {marker} {_c(str(i), 'cyan')}.  {_c(k, 'white')}  {_c(desc, 'dim')}")
    def _v(v):
        if v.isdigit() and 1 <= int(v) <= len(choices): return True, ""
        if v in [c[0] for c in choices]: return True, ""
        return False, f"請輸入 1~{len(choices)} 或選項名稱"
    raw = _ask("選擇", default=default, validator=_v)
    if raw.isdigit():
        return choices[int(raw)-1][0]
    return raw

def _confirm(label, default=True):
    hint = "Y/n" if default else "y/N"
    v = input(f"  {_c('▸','cyan')} {label} [{hint}]: ").strip().lower()
    return default if not v else v in ("y","yes","是")

def _hr2(): return _c("─"*62, "dim")

def _section(title):
    print(f"\n  {_c(title, 'bold')}")
    print(_hr2())

# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if sys.platform == "win32": os.system("")
    _banner()

    # ── 模式選擇 ──────────────────────────────────────────────────────────────
    _section("Step 1 / 4  ·  選擇模式")
    mode = _ask_choice("輸出模式", [
        ("overlay",       "疊合視圖 — 所有分子疊在一起 + 場圖（≤500個）"),
        ("batch_overlay", "分批疊合 — 自動分批 + 總場圖（推薦 >500個）"),
        ("single",        "單一分子 — 單個 .npz 生成一個 HTML"),
        ("batch",         "批次模式 — 整個目錄的所有 .npz 逐一生成 HTML"),
    ], default="batch_overlay")

    # ── 輸入路徑 ──────────────────────────────────────────────────────────────
    _section("Step 2 / 4  ·  輸入路徑")

    if mode in ("overlay", "batch_overlay"):
        npz_dir = _ask_path("npz 目錄（含 .npz 的資料夾）",
                             must_exist=True, default="qsar_grid_maps")
        while not os.path.isdir(npz_dir):
            print(_c("    需要目錄路徑", "red"))
            npz_dir = _ask_path("npz 目錄", must_exist=True)

        default_sdf = os.path.join(npz_dir, "all_aligned.sdf")
        sdf_path = _ask_path("all_aligned.sdf 路徑",
                              must_exist=True, exts=[".sdf"],
                              default=default_sdf if os.path.isfile(default_sdf) else None)

    elif mode == "single":
        npz_path = _ask_path("單一 .npz 路徑", must_exist=True, exts=[".npz"])

    else:  # batch
        npz_dir = _ask_path("npz 目錄", must_exist=True, default="qsar_grid_maps")
        while not os.path.isdir(npz_dir):
            print(_c("    需要目錄路徑", "red"))
            npz_dir = _ask_path("npz 目錄", must_exist=True)

    # ── 輸出路徑 ──────────────────────────────────────────────────────────────
    if mode == "overlay":
        default_out = os.path.join(npz_dir, "overlay.html")
        output = _ask("輸出 HTML 路徑", default=default_out)
    elif mode == "batch_overlay":
        default_out = os.path.join(npz_dir, "batch_overlay")
        output = _ask("輸出目錄（自動建立）", default=default_out)
    elif mode == "single":
        name = os.path.splitext(os.path.basename(npz_path))[0]
        default_out = os.path.join(os.path.dirname(os.path.abspath(npz_path)), name+".html")
        output = _ask("輸出 HTML 路徑", default=default_out)
    else:
        default_out = npz_dir
        output = _ask("輸出目錄", default=default_out)

    # ── 等值面與場圖設定 ──────────────────────────────────────────────────────
    _section("Step 3 / 4  ·  等值面設定")

    print(_c("  等值面閾值（isoval）說明：", "dim"))
    print(_c("  · 資料已是 Z-score 標準化（mean=0, std=1）", "dim"))
    print(_c("  · isoval=1.0 → 顯示前 16% 最顯著格點的邊界（建議起點）", "dim"))
    print(_c("  · isoval=1.5 → 前 7%  /  isoval=0.5 → 前 31%", "dim"))
    print(_c("  · 填 0 = 自動決定（程式從資料計算最佳值）", "dim"))
    print()

    raw_pos = _ask_float("立體場正等值面 / 靜電場正等值面（綠/藍，0=自動）",
                          default=1.0, lo=0.0, hi=10.0)
    raw_neg = _ask_float("立體場負等值面 / 靜電場負等值面（黃/紅，0=自動，輸入正數）",
                          default=1.0, lo=0.0, hi=10.0)
    isoval_pos = None if raw_pos == 0 else raw_pos
    isoval_neg = None if raw_neg == 0 else -raw_neg   # 負等值面取負號

    iso_opacity = _ask_float("等值面透明度（0.1=全透明 ~ 1.0=不透明）",
                              default=0.75, lo=0.1, hi=1.0)

    # 疊合視圖專屬設定
    ref_mol  = None
    max_mols = 500
    resume   = True

    if mode == "overlay":
        print()
        print(_c("  疊合視圖設定：", "dim"))
        max_mols = _ask_int("最多載入幾個分子（HTML 檔案大小考量）",
                             default=500, lo=1, hi=5000)
        ref_mol_raw = _ask("場圖來源分子（直接 Enter = 自動選 pIC50 最高）",
                            default="", allow_empty=True)
        ref_mol = ref_mol_raw.strip() if ref_mol_raw.strip() else None

    elif mode == "batch_overlay":
        print()
        print(_c("  分批疊合設定：", "dim"))
        batch_size = _ask_int("每批分子數量（每個 HTML 的分子數）",
                               default=500, lo=50, hi=2000)
        ref_mol_raw = _ask("全局場圖來源分子（直接 Enter = 自動選 pIC50 最高）",
                            default="", allow_empty=True)
        ref_mol = ref_mol_raw.strip() if ref_mol_raw.strip() else None

    elif mode == "batch":
        resume = _confirm("跳過已存在的 HTML（斷點續轉）？", default=True)

    # ── 確認 ──────────────────────────────────────────────────────────────────
    _section("Step 4 / 4  ·  確認執行")

    print(f"  模式       : {_c(mode, 'cyan')}")
    if mode == "overlay":
        print(f"  npz 目錄   : {_c(npz_dir, 'cyan')}")
        print(f"  SDF        : {_c(sdf_path, 'cyan')}")
        print(f"  參考分子   : {_c(ref_mol or '自動（pIC50 最高）', 'green')}")
        print(f"  最多分子   : {max_mols}")
    elif mode == "batch_overlay":
        print(f"  npz 目錄   : {_c(npz_dir, 'cyan')}")
        print(f"  SDF        : {_c(sdf_path, 'cyan')}")
        print(f"  每批分子數 : {batch_size}")
        print(f"  參考分子   : {_c(ref_mol or '自動（全局 pIC50 最高）', 'green')}")
        print(f"  輸出目錄   : {_c(output, 'cyan')}")
        print(_c("  輸出內容：batch_001.html … batch_NNN.html + total_field.html + index.html", "dim"))
    elif mode == "single":
        print(f"  輸入       : {_c(npz_path, 'cyan')}")
    else:
        print(f"  npz 目錄   : {_c(npz_dir, 'cyan')}")
        print(f"  斷點續轉   : {'是' if resume else '否'}")
    print(f"  輸出       : {_c(output, 'cyan')}")
    print(f"  等值面 pos : {_c(str(isoval_pos) if isoval_pos else '自動', 'green')}")
    print(f"  等值面 neg : {_c(str(isoval_neg) if isoval_neg else '自動', 'green')}")
    print(f"  等值面透明 : {iso_opacity}")
    print(_hr2())

    if not _confirm("開始生成？"):
        print(_c("  已取消。", "dim")); sys.exit(0)

    print()

    # ── 執行 ──────────────────────────────────────────────────────────────────

    # 把 iso_opacity 注入到 JS init（透過覆寫全域初始值）
    # 這裡用一個簡單的方式：在 init_js 前插入 isoOpacity 設定
    _ISO_OPACITY_INIT = f'isoOpacity={iso_opacity};\n'

    def _patched_generate_html(npz, out):
        """單分子版：注入 isoOpacity 初始值。"""
        try:
            meta, nums, coords, fields = _load_npz(npz)
            mol_name   = meta["mol_name"]
            pred_pic50 = meta.get("pred_pic50", 0.0)
            resolution = meta.get("resolution", 1.0)
            origin     = meta["origin"]
            axes       = meta["axes"]
            if not fields:
                print(_c(f"  [跳過] {mol_name}：無立體/靜電場資料", "yellow"))
                return False
            print(f"  {_c(mol_name,'yellow')} 計算等值面…", end="", flush=True)
            field_sdfs_b64 = {}
            for k, arr in fields.items():
                mesh = _marching_cubes(arr, origin, axes, isoval_pos, isoval_neg)
                fake_sdf, n_pos, n_neg = _verts_to_fake_sdf(
                    mesh['pos']['verts'], mesh['neg']['verts'], ref_name)
                fsd = {'sdf': fake_sdf, 'n_pos': n_pos, 'n_neg': n_neg,
                       'isoval_pos': mesh['isoval_pos'], 'isoval_neg': mesh['isoval_neg']}
                field_sdfs_b64[k] = _b64(json.dumps(fsd, separators=(',',':')))
            print()
            sdf_js = json.dumps(_mol_sdf(nums, coords, mol_name))
            cfg_js = json.dumps({k: FIELDS_CFG[k] for k in fields if k in FIELDS_CFG})
            js_data = (
                f'const FIELD_SDFS_B64 = {json.dumps(field_sdfs_b64)};\n'
                f'const FIELDS_CFG  = {cfg_js};\n'
                f'const REF_NAME    = null;\n'
                f'const PIC50       = {{min:0,max:1,p25:0,p75:1}};\n'
            )
            init_js = (
                _ISO_OPACITY_INIT +
                'document.getElementById("loading").style.display="none";\n'
                f'molSDF = {sdf_js};\n'
                '_schedule();\n'
                'setTimeout(function(){\n'
                '  Object.keys(FIELD_SDFS).forEach(function(k){'
                '    activeFields.add(k);'
                '    document.getElementById("btn-"+k).classList.add("active");'
                '    document.getElementById("ind-"+k).textContent="●";'
                '  });\n'
                '  _doRebuild();\n'
                '  _updateStats();\n'
                '}, 100);\n'
            )
            sidebar = (
                _sidebar_fields(fields, isoval_pos, isoval_neg) +
                _sidebar_opacity() +
                _sidebar_mol(False) +
                _sidebar_bg()
            )
            html = _assemble_html(
                mol_name,
                f'pIC50: {pred_pic50:.4f}  ·  解析度 {resolution} Å',
                sidebar, js_data, init_js)
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                f.write(html)
            kb = os.path.getsize(out) // 1024
            print(f"  → {_c(os.path.basename(out),'cyan')}  {kb} KB")
            return True
        except Exception:
            print(_c(f"\n  ✗ {traceback.format_exc(limit=3)}", "red"))
            return False

    def _patched_generate_overlay():
        """疊合視圖版：注入 isoOpacity。"""
        try:
            pic50_map = {}; fields_map = {}
            for fn in sorted(os.listdir(npz_dir)):
                if not fn.endswith(".npz") or fn.startswith("_"): continue
                try:
                    meta, nums, coords, fields = _load_npz(os.path.join(npz_dir,fn))
                    pic50_map[meta["mol_name"]] = meta.get("pred_pic50",0.0)
                    if fields: fields_map[meta["mol_name"]] = (meta,nums,coords,fields)
                except: pass
            if not pic50_map:
                print(_c("  找不到分子資料","red")); return False
            print(f"  讀取 {len(pic50_map)} 個分子")
            rn = ref_mol if ref_mol and ref_mol in fields_map else \
                 max((k for k in pic50_map if k in fields_map),
                     key=lambda k:pic50_map[k], default=None)
            if not rn:
                print(_c("  無有效參考分子","red")); return False
            print(f"  參考分子：{_c(rn,'yellow')} (pIC50={pic50_map[rn]:.4f})")
            rm, rnums, rcoords, rf = fields_map[rn]
            print(f"  計算等值面…", end="", flush=True)
            field_sdfs_b64 = {}
            for k, arr in rf.items():
                mesh = _marching_cubes(arr, rm["origin"], rm["axes"], isoval_pos, isoval_neg)
                fake_sdf, n_pos, n_neg = _verts_to_fake_sdf(
                    mesh['pos']['verts'], mesh['neg']['verts'], rn)
                fsd = {'sdf': fake_sdf, 'n_pos': n_pos, 'n_neg': n_neg,
                       'isoval_pos': mesh['isoval_pos'], 'isoval_neg': mesh['isoval_neg']}
                field_sdfs_b64[k] = _b64(json.dumps(fsd, separators=(',',':')))
            print()
            if not os.path.isfile(sdf_path):
                print(_c(f"  找不到 SDF：{sdf_path}","red")); return False
            all_sdf = _read_sdf(sdf_path)
            if len(all_sdf) > max_mols:
                print(_c(f"  ⚠ 截取前 {max_mols} 個","yellow"))
                all_sdf = all_sdf[:max_mols]
            print(f"  載入 {len(all_sdf)} 個分子（SDF）")
            b64e = lambda s: base64.b64encode(s.encode()).decode()
            mols_data = [[n, round(pic50_map.get(n,0),4) if n in pic50_map else None,
                          b64e(s)] for n,s in all_sdf]
            vals = [m[1] for m in mols_data if m[1] is not None]
            pr = {"min":round(float(min(vals)),4) if vals else 0,
                  "max":round(float(max(vals)),4) if vals else 1,
                  "p25":round(float(np.percentile(vals,25)),4) if vals else 0,
                  "p75":round(float(np.percentile(vals,75)),4) if vals else 1}
            cfg_js = json.dumps({k:FIELDS_CFG[k] for k in rf if k in FIELDS_CFG})
            n = len(mols_data)
            js_data = (
                f'const FIELD_SDFS_B64 = {json.dumps(field_sdfs_b64)};\n'
                f'const FIELDS_CFG = {cfg_js};\n'
                f'const REF_NAME   = {json.dumps(rn)};\n'
                f'const PIC50      = {json.dumps(pr)};\n'
                f'const ALL_MOLS   = {json.dumps(mols_data)};\n'
            )
            init_js = (
                _ISO_OPACITY_INIT +
                'document.getElementById("loading").style.display="none";\n'
                'window.ALL_MOLS=ALL_MOLS; molSDF=ALL_MOLS;\n'
                '_schedule();\n'
                'setTimeout(function(){\n'
                '  Object.keys(FIELD_SDFS).forEach(function(k){'
                '    activeFields.add(k);'
                '    document.getElementById("btn-"+k).classList.add("active");'
                '    document.getElementById("ind-"+k).textContent="●";'
                '  });\n'
                '  _doRebuild();\n'
                '  _updateStats();\n'
                '}, 150);\n'
            )
            sidebar = (
                _sidebar_fields(rf, isoval_pos, isoval_neg) +
                _sidebar_opacity() +
                _sidebar_filter(n) +
                _sidebar_mol(True) +
                _sidebar_bg()
            )
            html = _assemble_html(
                "QSAR 疊合視圖",
                f'{n} 個分子 · 場圖：<span style="color:#e94560">{rn}</span>'
                f' (pIC50={pic50_map[rn]:.4f})',
                sidebar, js_data, init_js)
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
            with open(output,"w",encoding="utf-8") as f:
                f.write(html)
            mb = os.path.getsize(output)/1024/1024
            print(f"  → {_c(output,'cyan')}  {mb:.1f} MB")
            return True
        except Exception:
            print(_c(f"\n  ✗ {traceback.format_exc(limit=4)}","red"))
            return False

    # ── 執行對應模式 ──────────────────────────────────────────────────────────
    if mode == "overlay":
        ok = _patched_generate_overlay()
        if ok:
            print(f"\n  {_c('完成！','bold','green')} 雙擊開啟：{_c(output,'yellow')}")

    elif mode == "batch_overlay":
        ok = generate_batch_overlay(
            npz_dir    = npz_dir,
            sdf_path   = sdf_path,
            output_dir = output,
            batch_size = batch_size,
            ref_mol    = ref_mol,
            isoval_pos = isoval_pos,
            isoval_neg = isoval_neg,
        )
        if ok:
            print(f"\n  {_c('完成！','bold','green')} 開啟索引頁：{_c(os.path.join(output,'index.html'),'yellow')}")

    elif mode == "single":
        ok = _patched_generate_html(npz_path, output)
        if ok:
            print(f"\n  {_c('完成！','bold','green')} 雙擊開啟：{_c(output,'yellow')}")

    else:  # batch
        npzs = sorted(f for f in os.listdir(npz_dir)
                      if f.endswith(".npz") and not f.startswith("_"))
        ok_n = skip_n = fail_n = 0
        t0 = time.perf_counter()
        entries = []
        os.makedirs(output, exist_ok=True)
        for i, fn in enumerate(npzs):
            name = fn[:-4]
            out_path = os.path.join(output, name+".html")
            if resume and os.path.exists(out_path):
                skip_n += 1
                try:
                    meta,_,_,_ = _load_npz(os.path.join(npz_dir,fn))
                    entries.append((name, name+".html", meta.get("pred_pic50",0)))
                except: entries.append((name,name+".html",0))
                continue
            print(f"  [{i+1}/{len(npzs)}] ", end="")
            if _patched_generate_html(os.path.join(npz_dir,fn), out_path):
                ok_n += 1
                try:
                    meta,_,_,_ = _load_npz(os.path.join(npz_dir,fn))
                    entries.append((name,name+".html",meta.get("pred_pic50",0)))
                except: entries.append((name,name+".html",0))
            else: fail_n += 1

        if entries:
            cards = "".join(
                f'<a class="card" href="{hf}" target="_blank" data-n="{n.lower()}">'
                f'<div class="cn">{n}</div><div class="cp">pIC50:{p:.4f}</div></a>'
                for n,hf,p in sorted(entries))
            idx = (
                '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>QSAR索引</title>'
                '<style>*{box-sizing:border-box;margin:0;padding:0;}'
                'body{background:#1a1a2e;color:#e0e0e0;font-family:"Segoe UI",sans-serif;padding:20px;}'
                'h1{color:#00d4ff;margin-bottom:14px;}'
                '.search{width:380px;padding:7px 12px;background:#0f3460;border:1px solid #0f3460;'
                'border-radius:7px;color:#fff;font-size:13px;margin-bottom:16px;outline:none;display:block;}'
                '.search:focus{border-color:#00d4ff;}'
                '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;}'
                '.card{background:#16213e;border:1px solid #0f3460;border-radius:8px;padding:12px;'
                'cursor:pointer;transition:all .2s;text-decoration:none;display:block;}'
                '.card:hover{border-color:#00d4ff;transform:translateY(-2px);}'
                '.cn{color:#00d4ff;font-size:14px;font-weight:bold;margin-bottom:3px;word-break:break-all;}'
                '.cp{color:#e94560;font-size:12px;}'
                '</style></head><body>'
                f'<h1>QSAR 場圖索引 ({len(entries)} 個分子)</h1>'
                '<input class="search" placeholder="搜尋…" oninput="'
                'document.querySelectorAll(\'.card\').forEach(c=>'
                'c.style.display=c.dataset.n.includes(this.value.toLowerCase())?\'\':\'none\')">'
                f'<div class="grid">{cards}</div></body></html>')
            idx_path = os.path.join(output, "index.html")
            with open(idx_path, "w", encoding="utf-8") as f:
                f.write(idx)

        elapsed = time.perf_counter()-t0
        print(f"\n  {_c('批次完成','bold','green')}")
        print(f"  成功 {ok_n}  跳過 {skip_n}  失敗 {fail_n}  耗時 {elapsed:.1f}s")
        print(f"  索引頁：{_c(os.path.join(output,'index.html'),'yellow')}")

    print()

if __name__ == "__main__":
    main()
