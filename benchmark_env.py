"""
benchmark_env.py — GpuQsarEngine 環境基準測試與參數建議工具
=============================================================
功能：
  1. 偵測 CPU / RAM / GPU 規格
  2. 執行實際計算基準測試（矩陣乘法、分子模擬、GNN 前向傳播）
  3. 根據測試結果輸出三種配置建議：
       【穩定模式】  正常速度，資源留有餘裕
       【效能模式】  極限的 80%，長時間穩定訓練建議
       【極限模式】  榨盡硬體，短時間最大吞吐
  4. 輸出純文字報告 + JSON 設定檔（可直接匯入 GpuQsarEngine）

使用方式：
  python benchmark_env.py             # 完整測試（約 2–3 分鐘）
  python benchmark_env.py --quick     # 快速測試（約 30 秒）
  python benchmark_env.py --no-rdkit  # 跳過分子模擬測試
  python benchmark_env.py --output-dir results/
"""

import os
import sys
import time
import json
import math
import platform
import argparse
import datetime
import traceback
import multiprocessing
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# 彩色終端輸出
# ═══════════════════════════════════════════════════════════════════════════════

_USE_COLOR = sys.stdout.isatty()

def _c(text, code):
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def green(s):  return _c(s, "32")
def yellow(s): return _c(s, "33")
def red(s):    return _c(s, "31")
def cyan(s):   return _c(s, "36")
def bold(s):   return _c(s, "1")
def dim(s):    return _c(s, "2")

def _sep(char="═", width=64):
    print(char * width)

def _section(title):
    print()
    _sep()
    print(f"  {bold(title)}")
    _sep()

def _ok(label, value):
    print(f"  {green('OK'):6s}  {label:<30s} {value}")

def _warn(label, value):
    print(f"  {yellow('WARN'):6s}  {label:<30s} {value}")

def _err(label, value):
    print(f"  {red('ERR'):6s}  {label:<30s} {value}")

def _info(label, value):
    print(f"  {dim('INFO'):6s}  {label:<30s} {value}")


# ═══════════════════════════════════════════════════════════════════════════════
# 結果容器
# ═══════════════════════════════════════════════════════════════════════════════

class BenchResult:
    def __init__(self):
        self.cpu_logical    = os.cpu_count() or 1
        self.cpu_physical   = 0
        self.cpu_brand      = "Unknown"
        self.cpu_freq_mhz   = 0.0
        self.ram_total_gb   = 0.0
        self.ram_avail_gb   = 0.0
        self.gpu_available  = False
        self.gpu_name       = "None"
        self.gpu_vram_gb    = 0.0
        self.gpu_vram_free_gb = 0.0
        self.gpu_cuda_ver   = ""
        self.gpu_sm         = ""
        self.gpu_bf16       = False
        self.gpu_tf32       = False
        self.cpu_matrix_gflops = 0.0
        self.gpu_matrix_gflops = 0.0
        self.gpu_matrix_gflops_bf16 = 0.0
        self.mol_per_sec    = 0.0
        self.gnn_batch_per_sec = 0.0
        self.dataloader_workers_optimal = 0
        self.python_ver     = sys.version.split()[0]
        self.torch_ver      = "N/A"
        self.rdkit_ver      = "N/A"
        self.pyg_ver        = "N/A"
        self.test_time      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.elapsed_sec    = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 硬體偵測
# ═══════════════════════════════════════════════════════════════════════════════

def detect_hardware(result):
    _section("1 / 5  硬體規格偵測")
    result.cpu_logical = os.cpu_count() or 1

    try:
        import psutil
        result.cpu_physical = psutil.cpu_count(logical=False) or result.cpu_logical
        mem = psutil.virtual_memory()
        result.ram_total_gb = mem.total / 1024**3
        result.ram_avail_gb = mem.available / 1024**3
        freq = psutil.cpu_freq()
        if freq:
            result.cpu_freq_mhz = freq.max or freq.current
    except ImportError:
        result.cpu_physical = result.cpu_logical
        _warn("psutil", "未安裝（pip install psutil），RAM 資訊有限")

    # CPU 型號
    try:
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            result.cpu_brand = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        result.cpu_brand = line.split(":")[1].strip()
                        break
        elif platform.system() == "Darwin":
            import subprocess
            result.cpu_brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        pass

    _ok("CPU 型號", result.cpu_brand[:55])
    _ok("CPU 核心數", f"邏輯={result.cpu_logical}  實體={result.cpu_physical}")
    if result.cpu_freq_mhz:
        _ok("CPU 最大頻率", f"{result.cpu_freq_mhz:.0f} MHz")
    _ok("RAM 總量", f"{result.ram_total_gb:.1f} GB")
    _ok("RAM 可用", f"{result.ram_avail_gb:.1f} GB")

    # GPU
    try:
        import torch
        result.torch_ver = torch.__version__
        if torch.cuda.is_available():
            result.gpu_available = True
            props = torch.cuda.get_device_properties(0)
            result.gpu_name      = props.name
            result.gpu_vram_gb   = props.total_memory / 1024**3
            result.gpu_vram_free_gb = (props.total_memory -
                torch.cuda.memory_allocated(0)) / 1024**3
            result.gpu_cuda_ver  = torch.version.cuda or ""
            result.gpu_sm        = f"{props.major}.{props.minor}"
            result.gpu_bf16      = torch.cuda.is_bf16_supported()
            result.gpu_tf32      = (props.major >= 8)
            _ok("GPU 型號",   result.gpu_name)
            _ok("VRAM 總量",  f"{result.gpu_vram_gb:.1f} GB")
            _ok("VRAM 可用",  f"{result.gpu_vram_free_gb:.1f} GB")
            _ok("CUDA 版本",  result.gpu_cuda_ver)
            _ok("SM 版本",    result.gpu_sm)
            _ok("bfloat16",  green("支援 ✓") if result.gpu_bf16 else yellow("不支援"))
            _ok("TF32",      green("支援 ✓") if result.gpu_tf32 else yellow("不支援"))
        else:
            _warn("GPU", "CUDA 不可用，將使用 CPU 模式")
    except ImportError:
        _err("PyTorch", "未安裝")

    print()
    _info("Python",  result.python_ver)
    _info("PyTorch", result.torch_ver)
    try:
        from rdkit import __version__ as rv
        result.rdkit_ver = rv
        _info("RDKit", rv)
    except ImportError:
        _warn("RDKit", "未安裝（分子測試將跳過）")
    try:
        import torch_geometric
        result.pyg_ver = torch_geometric.__version__
        _info("PyG", result.pyg_ver)
    except ImportError:
        _warn("PyG", "未安裝（GNN 測試將跳過）")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CPU 計算基準
# ═══════════════════════════════════════════════════════════════════════════════

def bench_cpu(result, quick=False):
    _section("2 / 5  CPU 計算基準")
    try:
        import torch
        size   = 2048 if not quick else 1024
        warmup = 3    if not quick else 1
        runs   = 10   if not quick else 3

        a = torch.randn(size, size)
        b = torch.randn(size, size)
        for _ in range(warmup):
            torch.mm(a, b)

        t0 = time.perf_counter()
        for _ in range(runs):
            torch.mm(a, b)
        elapsed = time.perf_counter() - t0

        gflops = 2 * size**3 * runs / elapsed / 1e9
        result.cpu_matrix_gflops = gflops
        _ok("矩陣乘法（fp32）", f"{gflops:.1f} GFLOPS  ({size}x{size}x{runs}次)")

        # DataLoader workers 建議
        if platform.system() == "Windows":
            opt = min(int(result.cpu_physical * 0.75), 8)
        else:
            opt = min(int(result.cpu_physical * 0.75), 16)
        result.dataloader_workers_optimal = max(1, opt)
        _ok("DataLoader workers 建議", str(result.dataloader_workers_optimal))

    except Exception as e:
        _err("CPU 基準", str(e)[:60])


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GPU 計算基準
# ═══════════════════════════════════════════════════════════════════════════════

def bench_gpu(result, quick=False):
    _section("3 / 5  GPU 計算基準")
    if not result.gpu_available:
        _warn("GPU 基準", "跳過（無 CUDA）")
        return
    try:
        import torch
        device = torch.device("cuda")
        size   = 4096 if not quick else 2048
        warmup = 5    if not quick else 2
        runs   = 20   if not quick else 5

        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        torch.cuda.synchronize()

        for _ in range(warmup):
            torch.mm(a, b)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(runs):
            torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        gflops = 2 * size**3 * runs / elapsed / 1e9
        result.gpu_matrix_gflops = gflops
        _ok("矩陣乘法（fp32）", f"{gflops:.1f} GFLOPS  ({size}x{size}x{runs}次)")

        if result.gpu_bf16:
            a16 = a.to(torch.bfloat16)
            b16 = b.to(torch.bfloat16)
            torch.cuda.synchronize()
            for _ in range(warmup):
                torch.mm(a16, b16)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                torch.mm(a16, b16)
            torch.cuda.synchronize()
            elapsed16 = time.perf_counter() - t0
            gflops16 = 2 * size**3 * runs / elapsed16 / 1e9
            result.gpu_matrix_gflops_bf16 = gflops16
            speedup = gflops16 / max(gflops, 1e-9)
            _ok("矩陣乘法（bf16）", f"{gflops16:.1f} GFLOPS  ({speedup:.1f}x fp32 加速)")

        # 記憶體頻寬
        n_elem = 256 * 1024 * 1024 // 4
        src = torch.randn(n_elem, device=device)
        dst = torch.empty(n_elem, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            dst.copy_(src)
        torch.cuda.synchronize()
        bw_gbps = 256 * 10 / 1024 / (time.perf_counter() - t0)
        _ok("記憶體頻寬（估計）", f"{bw_gbps:.1f} GB/s")

        del a, b, src, dst
        torch.cuda.empty_cache()
    except Exception as e:
        _err("GPU 基準", str(e)[:80])


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 分子計算基準
# ═══════════════════════════════════════════════════════════════════════════════

_BENCH_SMILES = [
    "CC1=CC=CC=C1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "CC1=C2CC(CC2(C)CCC1=O)C(C)(C)O",
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
    "C1CC2=CC=CC=C2N1",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
]

def _minimize_one(smi):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        mol = Chem.AddHs(mol)
        p = AllChem.ETKDGv3()
        p.randomSeed = 42
        if AllChem.EmbedMolecule(mol, p) != 0:
            return False
        ff = AllChem.MMFFGetMoleculeForceField(
            mol, AllChem.MMFFGetMoleculeProperties(mol))
        if ff:
            ff.Minimize(maxIts=200)
        return True
    except Exception:
        return False


def bench_rdkit(result, quick=False):
    _section("4 / 5  分子計算基準（RDKit MMFF）")
    try:
        from rdkit import __version__ as rv
        result.rdkit_ver = rv
    except ImportError:
        _warn("RDKit", "未安裝，跳過分子基準")
        return

    n_repeat   = 2 if quick else 8
    smiles_pool = _BENCH_SMILES * n_repeat

    # 單線程
    t0 = time.perf_counter()
    ok1 = sum(_minimize_one(s) for s in smiles_pool)
    t1  = time.perf_counter() - t0
    mps_single = ok1 / t1
    _ok("MMFF 最小化（單線程）",
        f"{mps_single:.1f} mol/s  ({ok1}/{len(smiles_pool)} 成功)")

    # 多線程
    n_workers = min(result.cpu_logical, 8)
    try:
        from concurrent.futures import ThreadPoolExecutor
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            ok_mt = sum(ex.map(_minimize_one, smiles_pool * 2))
        t1 = time.perf_counter() - t0
        mps_mt  = ok_mt / t1
        speedup = mps_mt / max(mps_single, 1e-9)
        result.mol_per_sec = mps_mt
        _ok(f"MMFF 最小化（{n_workers} 線程）",
            f"{mps_mt:.1f} mol/s  ({speedup:.1f}x 加速)")
    except Exception as e:
        result.mol_per_sec = mps_single
        _warn("多線程", str(e)[:60])


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GNN 前向傳播基準
# ═══════════════════════════════════════════════════════════════════════════════

def bench_gnn(result, quick=False):
    _section("5 / 5  GNN 前向傳播基準")
    try:
        import torch, torch.nn as nn
    except ImportError:
        _warn("GNN 基準", "PyTorch 未安裝，跳過")
        return
    try:
        from torch_geometric.data import Data, Batch
        from torch_geometric.loader import DataLoader
    except ImportError:
        _warn("GNN 基準", "PyG 未安裝，跳過")
        return

    device = torch.device("cuda" if result.gpu_available else "cpu")

    def _fake_graph():
        n, e = 32, 60
        return Data(
            x=torch.randn(n, 8), pos=torch.randn(n, 3),
            edge_index=torch.randint(0, n, (2, e)),
            edge_attr=torch.randn(e, 6), y=torch.randn(1))

    class _Mini(nn.Module):
        """
        簡化版 SchNet 用於基準測試。
        訊息輸入：[h_source(hc) || dist(1) || edge_attr(6)] = hc + 7 維
        """
        def __init__(self, hc=128):
            super().__init__()
            edge_dim   = 7          # dist(1) + edge_attr(6)
            msg_in_dim = hc + edge_dim
            self.hc    = hc
            self.proj  = nn.Linear(8, hc)
            self.layers= nn.ModuleList([
                nn.Sequential(nn.Linear(msg_in_dim, hc), nn.SiLU(),
                              nn.Linear(hc, hc))
                for _ in range(6)])
            self.out   = nn.Sequential(nn.Linear(hc, hc//2), nn.SiLU(),
                                       nn.Linear(hc//2, 1))

        def forward(self, batch):
            h   = self.proj(batch.x.float())
            row, col = batch.edge_index
            dist = (batch.pos[row] - batch.pos[col]).norm(dim=-1, keepdim=True)
            # 訊息輸入：source 節點特徵 + 距離 + 邊特徵
            ef   = torch.cat([h[col],
                               dist,
                               batch.edge_attr.float()], dim=-1)  # [E, hc+7]
            for layer in self.layers:
                msg = layer(ef)                                    # [E, hc]
                agg = torch.zeros(h.shape, dtype=msg.dtype, device=h.device)
                agg.index_add_(0, row, msg)
                h   = h + agg.to(h.dtype)
                # 更新 ef 中的 h[col] 部分（讓後續層看到更新後的特徵）
                ef  = torch.cat([h[col], dist,
                                 batch.edge_attr.float()], dim=-1)
            out = torch.zeros(batch.batch.max()+1, h.size(-1), device=h.device)
            out.index_add_(0, batch.batch, h)
            cnt = torch.bincount(batch.batch).float().unsqueeze(-1)
            return self.out(out / cnt.clamp(min=1)).squeeze(-1)

    model = _Mini().to(device)
    model.eval()

    batch_sizes = [8, 16, 32] if not quick else [16]
    best_tp, best_bs = 0.0, 16

    for bs in batch_sizes:
        graphs  = [_fake_graph() for _ in range(bs * 4)]
        loader  = DataLoader(graphs, batch_size=bs)
        batches = list(loader)
        with torch.no_grad():
            for bat in batches[:2]:
                model(bat.to(device))
        if result.gpu_available:
            torch.cuda.synchronize()

        runs = 20 if not quick else 5
        t0   = time.perf_counter()
        with torch.no_grad():
            for _ in range(runs):
                for bat in batches:
                    model(bat.to(device))
        if result.gpu_available:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        bps = runs * len(batches) / elapsed
        tp  = bps * bs
        _ok(f"GNN 推論（batch={bs:2d}）",
            f"{bps:.1f} batch/s  ~{tp:.0f} mol/s")
        if tp > best_tp:
            best_tp, best_bs = tp, bs
            result.gnn_batch_per_sec = bps

    _ok("最佳 batch_size（推論）", str(best_bs))
    del model
    if result.gpu_available:
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 參數建議生成
# ═══════════════════════════════════════════════════════════════════════════════

def generate_configs(result):
    vram   = result.gpu_vram_gb
    vcores = result.cpu_logical
    gflops = result.gpu_matrix_gflops or result.cpu_matrix_gflops

    # hidden_channels 上限
    if vram >= 16:      hc_max = 256
    elif vram >= 8:     hc_max = 128
    elif vram >= 4:     hc_max = 64
    else:               hc_max = 64

    # batch_size 估計（每樣本 ~4 MB）
    vram_for_train = max(vram * 0.75, 1.0)
    est_bs_max = min(128, max(4, int(vram_for_train * 1024 / 4)))

    # num_gaussians / num_interactions
    if gflops >= 50000:   ng_max, ni_max = 100, 8
    elif gflops >= 20000: ng_max, ni_max = 75,  6
    elif gflops >= 5000:  ng_max, ni_max = 50,  6
    else:                 ng_max, ni_max = 50,  4

    # parallel_workers
    if platform.system() == "Windows":
        workers_max = min(vcores, 32)
    else:
        workers_max = min(vcores, 80)

    def _profile(label, ratio, desc):
        r   = ratio
        hc  = max(32, 32 * round(int(hc_max * r) / 32))
        bs  = max(4,  4  * round(int(est_bs_max * r) / 4))
        ni  = max(3,  int(ni_max * r))
        ng  = max(25, 5  * round(int(ng_max * r) / 5))
        ml  = max(1,  int(2 * r + 0.5))
        wp  = max(1,  int(workers_max * r))
        wd  = max(0,  int(result.dataloader_workers_optimal * r))
        return {
            "_label": label, "_description": desc, "_ratio": ratio,
            # TrainConfig
            "hidden_channels":   hc,
            "num_interactions":  ni,
            "num_gaussians":     ng,
            "num_filters":       hc,
            "epochs":            int(200 * (1 + r * 0.5)),
            "batch_size":        bs,
            "lr":                round(1e-3 * (1 + r * 0.5), 6),
            "weight_decay":      1e-5,
            "scheduler":         "cosine",
            "patience":          max(10, int(30 * (1 - r * 0.3))),
            "dropout":           round(max(0.05, 0.3 - r * 0.25), 2),
            "activation":        "silu",
            "mlp_layers":        ml,
            "scaling_factor":    1.0,
            "cutoff":            round(5.0 + r * 5.0, 1),
            "sigma_factor":      round(1.0 - r * 0.3, 2),
            "device":            "cuda" if result.gpu_available else "cpu",
            # PerformanceConfig
            "parallel_workers":   wp,
            "dataloader_workers": wd,
            "pin_memory":         result.gpu_available,
            "persistent_workers": wd > 0,
            "cudnn_benchmark":    result.gpu_available,
            "amp_inference":      result.gpu_available,
            "monitor_interval":   10 if r >= 0.8 else 0,
            "chunk_size":         8 if wp >= 16 else 4,
        }

    return {
        "stable":   _profile("穩定模式", 0.5,
                             "正常速度，資源留有餘裕，適合長時間穩定訓練和除錯"),
        "balanced": _profile("效能模式", 0.8,
                             "極限的 80%，兼顧速度與穩定性，日常訓練推薦"),
        "maximum":  _profile("極限模式", 1.0,
                             "榨盡硬體，適合短時間快速實驗，OOM 風險較高"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 報告輸出
# ═══════════════════════════════════════════════════════════════════════════════

def print_config_table(cfg):
    label = cfg["_label"]
    desc  = cfg["_description"]
    r     = cfg["_ratio"]
    stars = "★" * int(r * 5) + "☆" * (5 - int(r * 5))

    print(f"\n  +-{'-'*58}-+")
    print(f"  |  {bold(label):<30s} {stars}  ({r*100:.0f}% 資源使用)  |")
    print(f"  |  {dim(desc):<58s}  |")
    print(f"  +-{'-'*58}-+")

    rows = [
        ("hidden_channels",    cfg["hidden_channels"],    "GNN 特徵維度"),
        ("num_interactions",   cfg["num_interactions"],   "消息傳遞層數"),
        ("num_gaussians",      cfg["num_gaussians"],      "高斯基函數數量"),
        ("num_filters",        cfg["num_filters"],        "濾波器通道數"),
        ("cutoff",             f"{cfg['cutoff']} A",      "截斷距離"),
        ("sigma_factor",       cfg["sigma_factor"],       "高斯寬度縮放"),
        ("epochs",             cfg["epochs"],             "訓練輪數"),
        ("batch_size",         cfg["batch_size"],         "批次大小"),
        ("lr",                 cfg["lr"],                 "學習率"),
        ("dropout",            cfg["dropout"],            "Dropout 率"),
        ("mlp_layers",         cfg["mlp_layers"],         "MLP 層數"),
        ("activation",         cfg["activation"],         "激活函數"),
        ("parallel_workers",   cfg["parallel_workers"],   "分子前處理線程"),
        ("dataloader_workers", cfg["dataloader_workers"], "DataLoader 線程"),
        ("pin_memory",         cfg["pin_memory"],         "記憶體鎖頁"),
        ("amp_inference",      cfg["amp_inference"],      "AMP 推論加速"),
    ]
    for key, val, comment in rows:
        vs = str(val)
        if val is True:  vs = green("True")
        if val is False: vs = dim("False")
        print(f"  |  {key:<24s} {vs:<14s} {dim(comment):<20s}  |")
    print(f"  +-{'-'*58}-+")


def save_outputs(result, configs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 純文字報告
    rp = os.path.join(output_dir, f"benchmark_report_{ts}.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write("GpuQsarEngine 環境基準測試報告\n")
        f.write(f"測試時間：{result.test_time}\n")
        f.write(f"測試耗時：{result.elapsed_sec:.1f} 秒\n")
        f.write("=" * 64 + "\n\n")
        f.write(f"CPU : {result.cpu_brand}\n")
        f.write(f"CPU 邏輯核心: {result.cpu_logical}\n")
        f.write(f"RAM 總量    : {result.ram_total_gb:.1f} GB\n")
        if result.gpu_available:
            f.write(f"GPU : {result.gpu_name}\n")
            f.write(f"VRAM: {result.gpu_vram_gb:.1f} GB\n")
            f.write(f"CUDA: {result.gpu_cuda_ver}  SM={result.gpu_sm}\n")
            f.write(f"bf16: {'支援' if result.gpu_bf16 else '不支援'}\n")
        f.write(f"\nCPU GFLOPS : {result.cpu_matrix_gflops:.1f}\n")
        if result.gpu_available:
            f.write(f"GPU GFLOPS : {result.gpu_matrix_gflops:.1f}")
            if result.gpu_bf16:
                f.write(f"  (bf16: {result.gpu_matrix_gflops_bf16:.1f})")
            f.write("\n")
        f.write(f"mol/s      : {result.mol_per_sec:.1f}\n")
        f.write(f"GNN batch/s: {result.gnn_batch_per_sec:.1f}\n")
        f.write("\n" + "=" * 64 + "\n")
        for key, cfg in configs.items():
            f.write(f"\n[{cfg['_label']}] ({cfg['_ratio']*100:.0f}%)\n")
            f.write(f"  {cfg['_description']}\n")
            for k, v in cfg.items():
                if not k.startswith("_"):
                    f.write(f"  {k:<26s}: {v}\n")

    # JSON 設定檔
    json_paths = {}
    train_keys = {"hidden_channels","num_interactions","num_gaussians",
                  "num_filters","epochs","batch_size","lr","weight_decay",
                  "scheduler","patience","dropout","activation","mlp_layers",
                  "scaling_factor","cutoff","sigma_factor","device"}
    perf_keys  = {"parallel_workers","dataloader_workers","pin_memory",
                  "persistent_workers","cudnn_benchmark","amp_inference",
                  "monitor_interval","chunk_size"}
    for key, cfg in configs.items():
        payload = {
            "_version": "1.0",
            "_created": result.test_time,
            "_label":   cfg["_label"],
            "_source":  "benchmark_env.py",
            "data":  {"output_dir": "qsar_output", "minimizer": "mmff",
                      "mmff_variant": "MMFF94s", "train_size": 0.8,
                      "random_seed": 42},
            "train": {k: v for k, v in cfg.items() if k in train_keys},
            "perf":  {k: v for k, v in cfg.items() if k in perf_keys},
        }
        jp = os.path.join(output_dir, f"config_{key}_{ts}.json")
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        json_paths[key] = jp

    return rp, json_paths


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GpuQsarEngine 環境基準測試工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick",      action="store_true",
                        help="快速測試模式（約 30 秒）")
    parser.add_argument("--no-rdkit",   action="store_true",
                        help="跳過 RDKit 分子計算基準")
    parser.add_argument("--no-gnn",     action="store_true",
                        help="跳過 GNN 前向傳播基準")
    parser.add_argument("--output-dir", default="benchmark_results",
                        help="輸出目錄（預設 benchmark_results/）")
    args = parser.parse_args()

    t_start = time.perf_counter()

    print()
    _sep("=")
    print(bold("  GpuQsarEngine -- 環境基準測試工具"))
    print(dim(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
    if args.quick:
        print(yellow("  [快速模式] 精度較低，建議正式調參前執行完整測試"))
    _sep("=")

    result = BenchResult()
    detect_hardware(result)
    bench_cpu(result, quick=args.quick)
    bench_gpu(result, quick=args.quick)

    if not args.no_rdkit:
        bench_rdkit(result, quick=args.quick)
    else:
        _section("4 / 5  分子計算基準")
        _warn("RDKit 基準", "已跳過（--no-rdkit）")

    if not args.no_gnn:
        bench_gnn(result, quick=args.quick)
    else:
        _section("5 / 5  GNN 基準")
        _warn("GNN 基準", "已跳過（--no-gnn）")

    result.elapsed_sec = time.perf_counter() - t_start

    _section("配置建議")
    configs = generate_configs(result)
    for key in ["stable", "balanced", "maximum"]:
        print_config_table(configs[key])

    report_path, json_paths = save_outputs(result, configs, args.output_dir)

    _section("測試完成")
    mm, ss = divmod(int(result.elapsed_sec), 60)
    _ok("總耗時",   f"{mm:02d}:{ss:02d}")
    _ok("文字報告", report_path)
    for key, path in json_paths.items():
        _ok(f"JSON（{configs[key]['_label']}）", path)

    print()
    print(bold("  如何使用 JSON 設定檔："))
    best_path = json_paths.get("balanced", list(json_paths.values())[0])
    print(f"  python gpu_qsar_engine.py --config {best_path}")
    print()
    _sep("=")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
