"""
qsar_benchmark.py
=================
GNN-QSAR 計算參數 Benchmark 工具
針對 qsar_grid_map.py 的所有可調參數進行系統性測速，
輸出建議參數設定（JSON）可直接匯入 qsar_grid_map.py 使用。

測量項目：
  A. MMFF 前處理速度     → 建議 CPU worker 數
  B. GPU forward 速度    → 驗證 AMP BF16 效益
  C. Perturbation batch  → 找 VRAM 安全上限與最佳 batch_size
  D. 解析度 vs 時間      → 建議 resolution
  E. IG 步數效益         → 建議 n_ig_steps
  F. NPZ 壓縮 I/O 速度   → 確認磁碟不成瓶頸
  G. CPU worker 擴展性   → 確認最佳 worker 數

用法：
  python qsar_benchmark.py               ← 互動式
  python qsar_benchmark.py --model model.pt --sdf input.sdf
  python qsar_benchmark.py --model model.pt --sdf input.sdf --quick
"""

import os, sys, time, json, math, argparse, traceback, tempfile, gc
import multiprocessing
import threading
import numpy as np

# ── ANSI 顏色（與 qsar_grid_map.py 一致）──────────────────────────────────
_C = {
    "reset": "\033[0m", "bold":   "\033[1m",
    "cyan":  "\033[96m","green":  "\033[92m",
    "yellow":"\033[93m","red":    "\033[91m",
    "dim":   "\033[2m", "white":  "\033[97m",
    "blue":  "\033[94m","magenta":"\033[95m",
}
def _c(text, *keys):
    if not sys.stdout.isatty(): return str(text)
    return "".join(_C.get(k,"") for k in keys) + str(text) + _C["reset"]

def _hr(char="─", width=66):
    return _c(char * width, "dim")

def _banner():
    print("\n" + _hr("═"))
    print(_c("  GNN-QSAR Benchmark", "bold", "cyan") +
          _c("  ·  參數自動調校工具", "dim"))
    print(_c("  qsar_benchmark.py  ·  RTX 5080 / CUDA 12.8 最佳化", "dim"))
    print(_hr("═") + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# 計時工具
# ══════════════════════════════════════════════════════════════════════════════

class _Timer:
    """CUDA-aware 高精度計時器。"""
    def __init__(self, device, label=""):
        self.device = device
        self.label  = label
        self._t0    = None

    def __enter__(self):
        if self.device.type == "cuda":
            import torch; torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        if self.device.type == "cuda":
            import torch; torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._t0

    @property
    def ms(self):
        return self.elapsed * 1000


def _vram_used_mb():
    """回傳目前 VRAM 佔用（MB），不支援 CUDA 時回傳 0。"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
    except Exception:
        pass
    return 0.0


def _vram_peak_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
    except Exception:
        pass
    return 0.0


def _vram_reset_peak():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 合成測試分子產生器
#   若使用者沒有 SDF，用內建 SMILES 產生代表性測試分子。
#   覆蓋小/中/大三種尺寸，反映真實藥物分子範圍。
# ══════════════════════════════════════════════════════════════════════════════

_TEST_SMILES = {
    "small_20at":  "c1ccc(cc1)C(=O)O",                              # 苯甲酸 ~15 重原子
    "medium_35at": "CC(=O)Nc1ccc(cc1)O",                            # 對乙醯胺酚 ~20 重原子
    "medium_45at": "Cc1ccc(cc1)S(=O)(=O)Nc2ccc(cc2)N",             # 磺胺類 ~28 重原子
    "large_60at":  "CC1=C2CC(CC(=O)c3ccc4c(c3)OCO4)(C2=O)C1",      # 黃樟素衍生物 ~38 重原子
    "large_80at":  "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                   # 布洛芬 ~26 重原子，加氫後~50+
}


def _make_test_mol(smiles: str, mol_name: str):
    """從 SMILES 產生 3D 分子（ETKDGv3 + MMFF）。"""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"無效 SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    ret = AllChem.EmbedMolecule(mol, params)
    if ret == -1:
        ret = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if ret == -1:
        raise RuntimeError(f"{mol_name}: EmbedMolecule 失敗")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    mol.SetProp("_Name", mol_name)
    return mol


# ══════════════════════════════════════════════════════════════════════════════
# 個別 benchmark 函式
# ══════════════════════════════════════════════════════════════════════════════

def bench_mmff(mols_raw: list, n_repeat: int = 3):
    """
    A. MMFF 前處理速度測試
    ─────────────────────
    測量 engine._minimize_mmff() 的每分子耗時，
    並測試不同 worker 數下的整體吞吐量。
    回傳 {single_s, workers_tput: {n: mols/s}, recommended_workers}
    """
    print(f"\n  {_c('A. MMFF 前處理速度', 'bold')}")
    print(_hr())

    # 單核基準
    times = []
    for mol_raw, mol_name in mols_raw[:min(5, len(mols_raw))]:
        from rdkit.Chem import AllChem
        from rdkit import Chem
        mol = Chem.AddHs(mol_raw) if mol_raw.GetNumConformers() == 0 else mol_raw
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        times.append((time.perf_counter() - t0) / n_repeat)

    single_s = float(np.median(times))
    print(f"  單核 MMFF  : {single_s*1000:.1f} ms / 分子")

    # 多核吞吐量測試
    cpu_count = multiprocessing.cpu_count()
    test_workers = sorted(set([1, 2, 4, 8, min(12, cpu_count//2),
                                min(16, cpu_count)] ))
    test_workers = [w for w in test_workers if w <= cpu_count]

    # 建立測試 mol pool（重複用前幾個分子湊成 20 個任務）
    test_mols = []
    for i in range(20):
        mol_raw, _ = mols_raw[i % len(mols_raw)]
        from rdkit import Chem
        test_mols.append(Chem.AddHs(mol_raw) if mol_raw.GetNumConformers() == 0 else mol_raw)

    def _mmff_worker(mol):
        from rdkit.Chem import AllChem
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        return True

    workers_tput = {}
    for nw in test_workers:
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        # 用 ThreadPoolExecutor（RDKit 釋放 GIL）
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=nw) as pool:
            list(pool.map(_mmff_worker, test_mols))
        elapsed = time.perf_counter() - t0
        tput = len(test_mols) / elapsed
        workers_tput[nw] = tput
        print(f"  {nw:2d} workers : {tput:.1f} mol/s  ({elapsed:.2f}s / 20 mol)")

    # 找效益遞減拐點：吞吐量增幅 < 15% 時停止增加 worker
    rec_workers = 1
    sorted_w = sorted(workers_tput.keys())
    for i in range(1, len(sorted_w)):
        w_prev = sorted_w[i-1]
        w_curr = sorted_w[i]
        gain   = (workers_tput[w_curr] - workers_tput[w_prev]) / workers_tput[w_prev]
        if gain >= 0.15:
            rec_workers = w_curr
        else:
            break

    print(f"\n  {_c('→ 建議 worker 數', 'green')} : {_c(str(rec_workers), 'bold', 'green')}"
          f"  （效益拐點）")

    return {
        "single_ms":          single_s * 1000,
        "workers_tput":       {str(k): round(v, 2) for k, v in workers_tput.items()},
        "recommended_workers": rec_workers,
    }


def bench_gpu_forward(model, graph, device, n_warmup=5, n_repeat=20):
    """
    B. GPU forward 速度 + AMP 效益
    ─────────────────────────────
    比較 FP32 vs BF16 autocast 的 forward 速度。
    回傳 {fp32_ms, bf16_ms, speedup, amp_recommended}
    """
    import torch
    print(f"\n  {_c('B. GPU Forward 速度 (FP32 vs BF16)', 'bold')}")
    print(_hr())

    model.eval()
    data  = graph.to(device)
    ea    = getattr(data, "edge_attr", None)
    batch = data.batch if data.batch is not None else \
        torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    def _run(use_amp, label):
        # warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type,
                                        dtype=torch.bfloat16, enabled=use_amp):
                    o = model(data.x, data.pos, data.edge_index, batch,
                              x=data.x, edge_attr=ea)
        if device.type == "cuda": torch.cuda.synchronize()

        times = []
        for _ in range(n_repeat):
            _vram_reset_peak()
            with _Timer(device) as t:
                with torch.no_grad():
                    with torch.amp.autocast(device_type=device.type,
                                            dtype=torch.bfloat16, enabled=use_amp):
                        o = model(data.x, data.pos, data.edge_index, batch,
                                  x=data.x, edge_attr=ea)
            times.append(t.elapsed * 1000)
        peak_mb = _vram_peak_mb()
        med = float(np.median(times))
        p95 = float(np.percentile(times, 95))
        print(f"  {label:<12}: median={med:6.2f}ms  p95={p95:6.2f}ms  "
              f"VRAM={peak_mb:.0f}MB")
        return med, peak_mb

    fp32_ms, fp32_vram = _run(False, "FP32")
    bf16_ms, bf16_vram = _run(True,  "BF16 AMP")

    speedup = fp32_ms / bf16_ms if bf16_ms > 0 else 1.0
    amp_rec = speedup >= 1.10   # 10% 以上才值得開

    print(f"\n  加速比    : {_c(f'{speedup:.2f}×', 'green' if speedup>=1.1 else 'yellow')}")
    print(f"  {_c('→ AMP BF16', 'green')} : {_c('建議開啟', 'bold','green') if amp_rec else _c('效益不明顯', 'yellow')}")

    return {
        "fp32_ms":         round(fp32_ms, 2),
        "bf16_ms":         round(bf16_ms, 2),
        "fp32_vram_mb":    round(fp32_vram, 1),
        "bf16_vram_mb":    round(bf16_vram, 1),
        "speedup":         round(speedup, 3),
        "amp_recommended": amp_rec,
    }


def bench_perturbation_batch(model, graph, device, n_pts_target=50_000,
                              batch_sizes=None, vram_limit_mb=14_000):
    """
    C. Perturbation batch_size 掃描
    ────────────────────────────────
    固定格點數，測試不同 batch_size 的 GPU 吞吐量與 VRAM 峰值。
    自動找出不超過 vram_limit_mb 的最快設定。
    回傳 {results: [{batch, tput, vram_mb}], recommended_batch, safe_max_batch}
    """
    import torch
    from torch_geometric.data import Batch, Data

    if batch_sizes is None:
        batch_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    print(f"\n  {_c('C. Perturbation Batch Size 掃描', 'bold')}")
    print(_hr())
    print(f"  測試格點數 : {n_pts_target:,}  VRAM 上限 : {vram_limit_mb:,} MB")

    model.eval()
    data0  = graph.to(device)
    ea0    = getattr(data0, "edge_attr", None)

    # 產生虛擬格點（不需要真實值，只測速）
    coords_np = data0.pos.cpu().numpy()
    lo  = coords_np.min(axis=0) - 4.0
    hi  = coords_np.max(axis=0) + 4.0
    pts = np.random.uniform(lo, hi, (n_pts_target, 3)).astype(np.float32)

    x_cpu  = data0.x.cpu()
    pos_cpu= data0.pos.cpu()
    ei_cpu = data0.edge_index.cpu()
    ea_cpu = data0.edge_attr.cpu() if ea0 is not None else None

    results = []
    safe_max = 64
    recommended = 256

    for bs in batch_sizes:
        # 嘗試跑一個 batch，捕捉 OOM
        try:
            _vram_reset_peak()
            # 建立 bs 個複製圖（代表 bs 個擾動格點）
            graphs = []
            for _ in range(min(bs, n_pts_target)):
                x_mod = x_cpu.clone()
                # 隨機 mask 幾個原子（模擬擾動）
                mask_idx = np.random.randint(0, len(x_cpu))
                x_mod[mask_idx] = 0.0
                graphs.append(Data(x=x_mod, pos=pos_cpu,
                                   edge_index=ei_cpu, edge_attr=ea_cpu))

            # warmup 1 次
            batch_obj = Batch.from_data_list(graphs).to(device)
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type,
                                        dtype=torch.bfloat16,
                                        enabled=(device.type=="cuda")):
                    o = model(batch_obj.x, batch_obj.pos,
                              batch_obj.edge_index, batch_obj.batch,
                              x=batch_obj.x,
                              edge_attr=getattr(batch_obj,"edge_attr",None))

            # 正式計時（3 次取中位）
            times = []
            for _ in range(3):
                _vram_reset_peak()
                with _Timer(device) as t:
                    with torch.no_grad():
                        with torch.amp.autocast(device_type=device.type,
                                                dtype=torch.bfloat16,
                                                enabled=(device.type=="cuda")):
                            o = model(batch_obj.x, batch_obj.pos,
                                      batch_obj.edge_index, batch_obj.batch,
                                      x=batch_obj.x,
                                      edge_attr=getattr(batch_obj,"edge_attr",None))
                times.append(t.elapsed)

            vram_peak = _vram_peak_mb()
            med_s     = float(np.median(times))
            tput      = bs / med_s   # graphs/s

            # 推算單一 valid_pts 全跑完的時間
            n_full_batches = math.ceil(n_pts_target / bs)
            total_est_s    = n_full_batches * med_s

            status = ""
            if vram_peak > vram_limit_mb:
                status = _c(" ⚠ VRAM 超限", "red")
            else:
                safe_max = bs
                status = _c(" ✓", "green")

            print(f"  batch={bs:5d} : {tput:7.0f} graphs/s  "
                  f"VRAM={vram_peak:6.0f}MB  "
                  f"50k pts≈{total_est_s:.2f}s{status}")

            results.append({
                "batch_size":   bs,
                "throughput":   round(tput, 1),
                "vram_peak_mb": round(vram_peak, 1),
                "est_50k_s":    round(total_est_s, 3),
                "oom":          False,
                "vram_over":    vram_peak > vram_limit_mb,
            })

            del batch_obj
            if device.type == "cuda":
                import torch; torch.cuda.empty_cache()
            gc.collect()

        except Exception as exc:
            if "out of memory" in str(exc).lower() or "OutOfMemory" in type(exc).__name__:
                print(f"  batch={bs:5d} : {_c('OOM', 'red')}")
                results.append({"batch_size": bs, "oom": True})
                if device.type == "cuda":
                    import torch; torch.cuda.empty_cache()
                gc.collect()
                break
            else:
                print(f"  batch={bs:5d} : {_c(f'錯誤: {exc}', 'red')}")
                break

    # 找最佳 batch：在 VRAM 安全範圍內吞吐量最高
    valid = [r for r in results if not r.get("oom") and not r.get("vram_over")]
    if valid:
        best = max(valid, key=lambda r: r["throughput"])
        recommended = best["batch_size"]
        print(f"\n  {_c('→ 建議 batch_size', 'green')} : "
              f"{_c(str(recommended), 'bold','green')}  "
              f"（{best['throughput']:.0f} graphs/s，"
              f"VRAM {best['vram_peak_mb']:.0f} MB）")
        print(f"  安全最大值   : {_c(str(safe_max), 'cyan')}")
    else:
        recommended = 256
        print(f"  {_c('→ 無有效結果，使用保守預設 256', 'yellow')}")

    return {
        "results":           results,
        "recommended_batch": recommended,
        "safe_max_batch":    safe_max,
    }


def bench_resolution(coords_np, resolutions=None, padding=4.0):
    """
    D. 解析度 vs 格點數 vs 估計時間
    ──────────────────────────────────
    純 CPU 計算（build_grid），測量格點數與 smearing 速度。
    回傳 {results, recommended_resolution}
    """
    from qsar_grid_map import build_grid, _smear_vectorised, VDW_RADII, DEFAULT_VDW

    if resolutions is None:
        resolutions = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0]

    print(f"\n  {_c('D. 解析度 vs 格點數 / Smearing 速度', 'bold')}")
    print(_hr())

    # 合成虛擬 contrib（模擬 saliency smearing）
    n_atoms  = len(coords_np)
    contrib  = np.random.rand(n_atoms).astype(np.float32)
    atomic_n = np.full(n_atoms, 6, dtype=np.int32)   # 全碳

    results = []
    recommended = 1.0

    for res in resolutions:
        lo, (nx,ny,nz), axes, gx, gy, gz, _ = build_grid(
            coords_np, resolution=res, padding=padding)
        n_pts = nx * ny * nz

        # smearing 計時（3 次取中位）
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = _smear_vectorised(contrib, coords_np, atomic_n, gx, gy, gz)
            times.append(time.perf_counter() - t0)
        smear_ms = float(np.median(times)) * 1000

        # 估計 Perturbation 所有 forward（假設 10k valid pts，batch=512，5ms/batch）
        perturb_fwd_est_s = math.ceil(n_pts * 0.7 / 512) * 0.005 * 3   # ×3 for S/+/-

        size_mb = n_pts * 4 / 1e6
        print(f"  {res:.2f} Å : {nx:4d}×{ny:4d}×{nz:4d} = {n_pts:>10,}  "
              f"({size_mb:5.1f} MB/場)  "
              f"smear={smear_ms:6.1f}ms  "
              f"perturb≈{perturb_fwd_est_s:.1f}s")

        results.append({
            "resolution":       res,
            "grid_pts":         n_pts,
            "size_mb_per_field":round(size_mb, 1),
            "smear_ms":         round(smear_ms, 1),
            "perturb_est_s":    round(perturb_fwd_est_s, 2),
        })

    # 建議：smearing < 500ms 且 grid < 5M 的最細解析度
    for r in sorted(results, key=lambda x: x["resolution"]):
        if r["smear_ms"] < 500 and r["grid_pts"] < 5_000_000:
            recommended = r["resolution"]

    print(f"\n  {_c('→ 建議解析度', 'green')} : "
          f"{_c(str(recommended), 'bold','green')} Å")

    return {"results": results, "recommended_resolution": recommended}


def bench_ig_steps(model, graph, device,
                   step_counts=None, tolerance=0.01):
    """
    E. IG 積分步數 vs 精度 vs 時間
    ──────────────────────────────
    以 n=500 步的 3 次平均為穩定參考基準（「真值」），
    測試較少步數的近似誤差。
    找出誤差 < tolerance 的最少步數。

    修正說明：
      - 原本參考值只跑一次 n=200，但 BF16 autocast 的數值噪聲
        加上梯度累積誤差，使得 n=200 本身對自身的誤差就有 0.2%，
        造成所有步數的誤差偏高且 n=200 無法通過自身檢查。
      - 新版以 n=500 跑 3 次取平均作為穩定參考，
        並把 n=200 納入 step_counts 一起比較。
    """
    import torch

    if step_counts is None:
        step_counts = [10, 20, 30, 50, 100, 200]

    # 確保 step_counts 不含基準步數（500），避免誤導
    REF_STEPS   = 500
    REF_REPEATS = 3

    print(f"\n  {_c('E. IG 積分步數 vs 精度', 'bold')}")
    print(_hr())
    print(f"  基準      : n={REF_STEPS} 步 × {REF_REPEATS} 次平均（穩定參考真值）")
    print(f"  誤差容忍  : {tolerance*100:.0f}%  (歸一化 L2 誤差)")

    model.eval()
    data  = graph.to(device)
    ea    = getattr(data, "edge_attr", None)
    batch = data.batch if data.batch is not None else \
        torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    use_amp = (device.type == "cuda")

    # x_orig / x_base 在整個 bench 期間固定不變
    x_orig = data.x.detach().float()
    x_base = torch.zeros_like(x_orig)

    def _run_ig(n_steps):
        """執行一次 IG，回傳 per-atom 屬性向量 (numpy float64)。"""
        integ = torch.zeros_like(x_orig)
        for step in range(n_steps):
            alpha  = (step + 0.5) / n_steps
            x_step = (x_base + alpha * (x_orig - x_base)).detach().requires_grad_(True)
            with torch.amp.autocast(device_type=device.type,
                                    dtype=torch.bfloat16, enabled=use_amp):
                out = model(x_step, data.pos, data.edge_index, batch,
                            x=x_step, edge_attr=ea)
            if isinstance(out, dict): out = out["pic50"]
            out.sum().backward()
            with torch.no_grad():
                integ = integ + x_step.grad.detach()
        integ /= n_steps
        # 轉 float64 累加，降低多次平均的精度損失
        return ((x_orig - x_base) * integ).norm(p=2, dim=1).detach().cpu().numpy().astype(np.float64)

    # ── 穩定參考值：多次平均 ───────────────────────────────────────────────
    print(f"  計算參考真值 (n={REF_STEPS} × {REF_REPEATS})…", end="", flush=True)
    ref_runs = []
    with _Timer(device) as t_ref:
        for _ in range(REF_REPEATS):
            ref_runs.append(_run_ig(REF_STEPS))
    ref      = np.mean(ref_runs, axis=0)          # 平均後作為真值
    ref_norm = np.linalg.norm(ref) + 1e-12
    print(f" {t_ref.elapsed:.1f}s  (per-run: {t_ref.elapsed/REF_REPEATS:.1f}s)")

    # 自我一致性驗證（三次結果之間的差異，顯示參考值的穩定程度）
    self_errs = [np.linalg.norm(r - ref) / ref_norm for r in ref_runs]
    print(f"  參考值穩定性: {[f'{e*100:.3f}%' for e in self_errs]}  "
          f"(max {max(self_errs)*100:.3f}%)")

    results    = []
    recommended = step_counts[-1]   # 預設推薦最大值，若有更少步數通過則更新

    for n in step_counts:
        with _Timer(device) as t:
            attr = _run_ig(n)
        err         = np.linalg.norm(attr.astype(np.float64) - ref) / ref_norm
        ms_per_step = t.elapsed * 1000 / n
        ok          = bool(err < tolerance)
        status      = _c("✓", "green") if ok else _c("✗", "red")

        print(f"  n={n:4d} : {t.elapsed:6.2f}s  "
              f"err={err*100:5.2f}%  "
              f"{ms_per_step:.1f}ms/step  {status}")

        results.append({
            "n_steps":     n,
            "elapsed_s":   round(t.elapsed, 3),
            "error_pct":   round(err * 100, 4),
            "ms_per_step": round(ms_per_step, 2),
            "ok":          ok,
        })
        # 取第一個通過的步數作為建議（最少步數）
        if ok and recommended == step_counts[-1]:
            recommended = n

    print(f"\n  {_c('→ 建議 IG 步數', 'green')} : "
          f"{_c(str(recommended), 'bold', 'green')}")

    return {
        "results":              results,
        "recommended_ig_steps": recommended,
        "ref_steps":            REF_STEPS,
        "ref_self_err_max_pct": round(max(self_errs) * 100, 4),
    }


def bench_npz_io(output_dir: str, grid_shape=(60, 55, 50), n_trials=5):
    """
    F. NPZ 壓縮 I/O 速度
    ──────────────────
    測試寫入 + 讀取速度，確認磁碟 I/O 不成為瓶頸。
    回傳 {write_ms, read_ms, size_kb, bottleneck}

    Windows 修正：np.load() 預設使用 memory-map，檔案 handle 不會
    在讀取完成後立即釋放，導致 os.remove() 失敗（WinError 32）。
    修正方式：
      1. 讀取時傳入 mmap_mode=None 強制載入到記憶體
      2. 明確呼叫 npzfile.close() 釋放 handle
      3. del + gc.collect() 確保物件釋放
      4. 最後用 try/finally 保證暫存檔一定被清理
    """
    print(f"\n  {_c('F. NPZ 壓縮 I/O 速度', 'bold')}")
    print(_hr())

    nx, ny, nz = grid_shape
    n_fields   = 6   # saliency×3 + perturb×2 + ig×1
    arrays = {f"field_{i}": np.random.rand(nx, ny, nz).astype(np.float32)
              for i in range(n_fields)}
    arrays["atomic_nums"]   = np.array([6] * 30, dtype=np.int16)
    arrays["atomic_coords"] = np.random.rand(30, 3).astype(np.float32)
    arrays["_meta"]         = np.array(['{"mol_name":"bench"}'], dtype=object)

    tmp = os.path.join(output_dir, "_bench_io.npz")

    write_times, read_times = [], []
    try:
        for _ in range(n_trials):
            # 寫入計時
            t0 = time.perf_counter()
            np.savez_compressed(tmp, **arrays)
            write_times.append(time.perf_counter() - t0)

            # 讀取計時：mmap_mode=None 強制載入記憶體，不保留檔案 handle
            t0 = time.perf_counter()
            npzfile = np.load(tmp, allow_pickle=True, mmap_mode=None)
            loaded  = {k: npzfile[k] for k in npzfile.files}   # 強制讀進記憶體
            npzfile.close()    # 明確釋放 handle（Windows 必須）
            del npzfile, loaded
            read_times.append(time.perf_counter() - t0)

        size_kb  = os.path.getsize(tmp) / 1024
        write_ms = float(np.median(write_times)) * 1000
        read_ms  = float(np.median(read_times))  * 1000

    finally:
        # 確保暫存檔一定被清理（即使上面發生例外）
        if os.path.exists(tmp):
            try:
                gc.collect()          # 觸發 GC，確保所有 mmap handle 釋放
                os.remove(tmp)
            except OSError as e:
                print(_c(f"  [警告] 暫存檔無法刪除：{e}", "yellow"))

    # 相對於典型 GPU 計算時間（30s/mol），I/O 佔比
    io_pct     = (write_ms + read_ms) / 30_000 * 100
    bottleneck = io_pct > 5.0

    print(f"  格點尺寸  : {nx}×{ny}×{nz} × {n_fields} 場")
    print(f"  檔案大小  : {size_kb:.0f} KB")
    print(f"  寫入      : {write_ms:.0f} ms  (中位數，n={n_trials})")
    print(f"  讀取      : {read_ms:.0f} ms  (中位數，n={n_trials})")
    print(f"  I/O 佔比  : {io_pct:.1f}% （相對 30s/mol）"
          + (_c("  ⚠ 磁碟可能成瓶頸", "yellow") if bottleneck else
             _c("  ✓ 無瓶頸", "green")))

    return {
        "write_ms":        round(write_ms, 1),
        "read_ms":         round(read_ms, 1),
        "size_kb":         round(size_kb, 1),
        "io_pct":          round(io_pct, 2),
        "disk_bottleneck": bottleneck,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 建議參數彙整與 JSON 輸出
# ══════════════════════════════════════════════════════════════════════════════

def _build_recommendations(bench_results: dict) -> dict:
    """
    根據各 benchmark 結果彙整建議參數，
    回傳可直接傳給 run_qsar_grid_map / main() 的參數 dict。
    """
    r = {}

    if "mmff" in bench_results:
        r["n_workers"] = bench_results["mmff"]["recommended_workers"]

    if "gpu_forward" in bench_results:
        r["amp_bf16"] = bench_results["gpu_forward"]["amp_recommended"]

    if "perturbation_batch" in bench_results:
        r["perturb_batch"] = bench_results["perturbation_batch"]["recommended_batch"]

    if "resolution" in bench_results:
        r["resolution"] = bench_results["resolution"]["recommended_resolution"]

    if "ig_steps" in bench_results:
        r["n_ig_steps"] = bench_results["ig_steps"]["recommended_ig_steps"]

    # 保守預設補丁
    r.setdefault("n_workers",    4)
    r.setdefault("amp_bf16",     True)
    r.setdefault("perturb_batch",512)
    r.setdefault("resolution",   1.0)
    r.setdefault("n_ig_steps",   50)
    r["padding"]      = 4.0
    r["sigma_scale"]  = 1.0
    r["output_mode"]  = "npz"

    return r


def _print_recommendation_table(rec: dict, bench_results: dict):
    """印出最終建議參數表。"""
    print("\n" + _hr("═"))
    print(_c("  建議參數設定", "bold", "green"))
    print(_hr())

    rows = [
        ("解析度",          f"{rec['resolution']} Å",
         "格點數 / Perturbation 速度的最大影響因子"),
        ("Perturb batch",   str(rec["perturb_batch"]),
         "GPU 吞吐量飽和點，VRAM 安全範圍內最大值"),
        ("IG 步數",         str(rec["n_ig_steps"]),
         f"誤差 < {bench_results.get('ig_steps',{}).get('results',[{}])[-1].get('error_pct','?')}% 的最少步數"),
        ("CPU workers",     str(rec["n_workers"]),
         "MMFF 前處理吞吐量拐點"),
        ("AMP BF16",        "開啟" if rec["amp_bf16"] else "關閉",
         f"加速比 {bench_results.get('gpu_forward',{}).get('speedup',1.0):.2f}×"),
        ("Padding",         f"{rec['padding']} Å",    "建議維持預設"),
        ("Sigma scale",     str(rec["sigma_scale"]),  "建議維持預設"),
        ("輸出格式",        rec["output_mode"],        "npz 體積最小"),
    ]

    for name, val, note in rows:
        print(f"  {_c(name, 'cyan'):<20}  {_c(val, 'bold', 'white'):<10}  "
              f"{_c(note, 'dim')}")

    print(_hr())

    # 時間估算
    if "resolution" in bench_results and "perturbation_batch" in bench_results:
        res_row  = next((r for r in bench_results["resolution"]["results"]
                         if r["resolution"] == rec["resolution"]), None)
        pt_res   = bench_results["perturbation_batch"]
        best_pt  = next((r for r in pt_res["results"]
                         if r["batch_size"] == rec["perturb_batch"]), None)
        if res_row and best_pt:
            perturb_s = best_pt.get("est_50k_s", 30)
            smear_s   = res_row["smear_ms"] / 1000 * 3
            ig_s      = bench_results.get("ig_steps",{}).get(
                "results",[{"elapsed_s":10}])[-2]["elapsed_s"] \
                if "ig_steps" in bench_results else 10
            saliency_s = bench_results.get("gpu_forward",{}).get("bf16_ms",5)/1000
            mol_est_s  = perturb_s + smear_s + ig_s + saliency_s
            total_h    = mol_est_s * 5000 / 3600

            print(f"  {'預估單分子耗時':<20}  "
                  f"{_c(f'~{mol_est_s:.0f}s', 'bold','yellow')}")
            print(f"  {'5000 分子估計':<20}  "
                  f"{_c(f'~{total_h:.1f}h', 'bold','yellow')}")
            print(_hr())


# ══════════════════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if sys.platform == "win32":
        os.system("")

    parser = argparse.ArgumentParser(
        description="GNN-QSAR 參數 Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", "-m", help="模型 .pt 路徑")
    parser.add_argument("--sdf",   "-s", help="SDF 路徑（取前 5 個分子）")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="快速模式：跳過 IG 步數測試（節省 ~5 分鐘）")
    parser.add_argument("--out",   "-o", default="benchmark_result.json",
                        help="輸出 JSON 路徑（預設 benchmark_result.json）")
    parser.add_argument("--vram-limit", type=int, default=14000,
                        help="VRAM 安全上限 MB（預設 14000）")
    args = parser.parse_args()

    _banner()

    # ── 互動式補全缺少的路徑 ────────────────────────────────────────────────
    def _ask(prompt, check_ext=None):
        while True:
            v = input(f"  {_c('▸', 'cyan')} {prompt}: ").strip()
            if not v: continue
            if not os.path.isfile(v):
                print(_c(f"    找不到檔案：{v}", "red")); continue
            if check_ext and not any(v.lower().endswith(e) for e in check_ext):
                print(_c(f"    需為 {check_ext} 格式", "red")); continue
            return v

    if not args.model:
        args.model = _ask("模型 .pt 路徑", [".pt",".pth"])
    if not args.sdf:
        use_builtin = input(
            f"  {_c('▸','cyan')} SDF 路徑（留空使用內建測試分子）: ").strip()
        args.sdf = use_builtin if use_builtin else None

    # ── 載入模型 ────────────────────────────────────────────────────────────
    print(f"\n  {_c('載入模型…', 'bold')}")
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(args.model)))
    try:
        from gpu_qsar_engine import SchNetQSAR, GpuQsarEngine, DataConfig, TrainConfig
        import dataclasses
    except ImportError:
        print(_c("  錯誤：找不到 gpu_qsar_engine.py", "red")); sys.exit(1)

    cuda_ok  = torch.cuda.is_available()
    device   = torch.device("cuda" if cuda_ok else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU only"
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2 \
                    if cuda_ok else 0

    print(f"  裝置    : {_c(str(device).upper(), 'cyan')}  {_c(gpu_name, 'dim')}")
    if cuda_ok:
        print(f"  VRAM    : {total_vram_mb:.0f} MB  "
              f"（安全上限設為 {args.vram_limit} MB）")
        # 自動調整 vram_limit（不超過 90% 總 VRAM）
        args.vram_limit = min(args.vram_limit, int(total_vram_mb * 0.90))

    ckpt      = torch.load(args.model, map_location=device, weights_only=False)
    train_cfg = TrainConfig(**{k:v for k,v in ckpt["train_cfg"].items()
                               if k in {f.name for f in dataclasses.fields(TrainConfig)}})
    model     = SchNetQSAR(train_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  模型    : {_c('載入成功', 'green')}")

    data_cfg = DataConfig(**{k:v for k,v in ckpt["data_cfg"].items()
                             if k in {f.name for f in dataclasses.fields(DataConfig)}})
    engine   = GpuQsarEngine(data_cfg)

    # CUDA warmup
    if cuda_ok:
        _ = torch.zeros(1, device=device)
        torch.cuda.synchronize()

    # ── 載入 / 建立測試分子 ─────────────────────────────────────────────────
    print(f"\n  {_c('準備測試分子…', 'bold')}")
    mols_raw   = []   # list of (mol, mol_name)
    test_graphs= []

    if args.sdf:
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(args.sdf, removeHs=False)
        for i, m in enumerate(suppl):
            if m is None or i >= 5: break
            name = m.GetProp("_Name") if m.HasProp("_Name") else f"mol_{i+1}"
            mols_raw.append((m, name))
        print(f"  從 SDF 載入 {len(mols_raw)} 個分子")
    else:
        from rdkit import Chem
        print("  使用內建測試分子…")
        for name, smi in _TEST_SMILES.items():
            try:
                m = _make_test_mol(smi, name)
                mols_raw.append((m, name))
                print(f"    {name} : {m.GetNumAtoms()} 個原子（含H）")
            except Exception as e:
                print(_c(f"    {name} 建構失敗：{e}", "yellow"))

    if not mols_raw:
        print(_c("  無可用測試分子，退出。", "red")); sys.exit(1)

    # 前處理第一個分子取得 graph 和 coords
    print("  前處理分子中…", end="", flush=True)
    for mol_raw, mol_name in mols_raw:
        try:
            mol_out, _ = engine._minimize_mmff(mol_raw)
            if mol_out is None: continue
            g = engine.mol_to_graph(mol_out, label=0.0, smiles="")
            test_graphs.append((mol_name, g, mol_out))
        except Exception:
            pass
    print(f" {len(test_graphs)} 個分子就緒")

    if not test_graphs:
        print(_c("  前處理全部失敗，退出。", "red")); sys.exit(1)

    _, ref_graph, ref_mol = test_graphs[0]
    coords_np = ref_graph.pos.cpu().numpy()

    # ── 暫存目錄（用於 I/O bench）───────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="qsar_bench_")

    # ══════════════════════════════════════════════════════════════════════════
    # 執行 benchmark
    # ══════════════════════════════════════════════════════════════════════════
    bench_results = {}
    t_total = time.perf_counter()

    # A. MMFF
    bench_results["mmff"] = bench_mmff(mols_raw)

    # B. GPU forward
    bench_results["gpu_forward"] = bench_gpu_forward(model, ref_graph, device)

    # C. Perturbation batch
    bench_results["perturbation_batch"] = bench_perturbation_batch(
        model, ref_graph, device,
        vram_limit_mb=args.vram_limit)

    # D. 解析度
    bench_results["resolution"] = bench_resolution(coords_np)

    # E. IG 步數（quick 模式跳過）
    if not args.quick:
        bench_results["ig_steps"] = bench_ig_steps(model, ref_graph, device)
    else:
        print(f"\n  {_c('E. IG 步數測試', 'dim')} — {_c('已跳過（quick 模式）', 'yellow')}")
        bench_results["ig_steps"] = {"recommended_ig_steps": 50, "results": []}

    # F. NPZ I/O
    bench_results["npz_io"] = bench_npz_io(tmp_dir)

    total_elapsed = time.perf_counter() - t_total

    # ── 彙整建議 ────────────────────────────────────────────────────────────
    rec = _build_recommendations(bench_results)
    _print_recommendation_table(rec, bench_results)

    # ── 儲存 JSON ────────────────────────────────────────────────────────────
    output = {
        "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
        "model":          os.path.basename(args.model),
        "device":         gpu_name,
        "total_vram_mb":  round(total_vram_mb, 0),
        "cpu_count":      multiprocessing.cpu_count(),
        "benchmark_s":    round(total_elapsed, 1),
        "recommended":    rec,
        "details":        bench_results,
    }

    # numpy 數值序列化
    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):    return obj.tolist()
            return super().default(obj)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, cls=_NpEncoder, ensure_ascii=False, indent=2)

    print(f"\n  {_c('結果已儲存', 'green')} → {_c(args.out, 'cyan')}")
    print(f"  Benchmark 耗時 : {total_elapsed:.0f}s\n")

    # ── 使用說明 ────────────────────────────────────────────────────────────
    print(_hr())
    print(_c("  如何使用 Benchmark 結果", "bold"))
    print(_hr("─"))
    print(f"  在 qsar_grid_map.py 的進階參數提示時，直接輸入以下建議值：")
    print()
    print(f"    解析度       : {_c(str(rec['resolution']), 'yellow')} Å")
    print(f"    Perturb batch: {_c(str(rec['perturb_batch']), 'yellow')}")
    print(f"    IG 步數      : {_c(str(rec['n_ig_steps']), 'yellow')}")
    print(f"    CPU workers  : {_c(str(rec['n_workers']), 'yellow')}")
    print()
    print(_c("  或直接程式匯入（進階用法）：", "dim"))
    print(_c(f"    import json", "dim"))
    print(_c(f"    rec = json.load(open('{args.out}'))['recommended']", "dim"))
    print(_hr("═") + "\n")

    # 清理暫存
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
