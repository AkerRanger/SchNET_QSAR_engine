"""
prepare_protein.py — 蛋白質口袋資料預處理腳本
===============================================
用途：將原始蛋白質 PDB 檔清洗並提取口袋殘基座標，
      供 GpuQsarEngine 的 Pocket-Aware Cross-Attention 模組使用。

功能：
  1. PDB 讀取與結構驗證：
       - 偵測多鏈（自動選最大鏈或使用者指定）
       - 解析度檢查（REMARK 2 / REMARK 3）
       - 缺失殘基警告
  2. 蛋白質清洗：
       - 去除 HETATM 水分子 / 輔因子（可選保留）
       - 去除 B-factor 異常原子（可選）
       - 補充缺失氫原子（BioPython）
  3. 口袋定義（三種策略）：
       - 策略 A：配體中心法（自動偵測 HETATM 配體，最常用）
       - 策略 B：使用者輸入座標 + 半徑
       - 策略 C：fpocket（需另外安裝）
  4. 口袋殘基特徵化：
       - 殘基類型 one-hot（20 種標準氨基酸）
       - Cα 座標
       - 疏水性 / 電荷 / 極性標籤
  5. 輸出：
       - pocket_residues.csv       ← 殘基座標 + 特徵
       - pocket_coords.npy         ← [N, 3] numpy 陣列（供模型直接載入）
       - pocket_features.npy       ← [N, F] 殘基特徵矩陣
       - cleaned_protein.pdb       ← 清洗後的蛋白質結構
       - pocket_report.txt         ← 口袋定義報告

使用方式：
  # 自動偵測配體口袋（最常用）
  python prepare_protein.py --pdb structure.pdb

  # 指定配體名稱 + 自訂半徑
  python prepare_protein.py --pdb structure.pdb --ligand ATP --cutoff 12.0

  # 手動指定口袋中心座標
  python prepare_protein.py --pdb structure.pdb --center "12.5 34.2 -8.7" --cutoff 10.0

  # 指定鏈
  python prepare_protein.py --pdb structure.pdb --chain A

搭配 GpuQsarEngine 使用：
  訓練時啟用 use_pocket=True，系統會呼叫互動式口袋偵測（_interactive_pocket_loader），
  或直接在 DataConfig 中指定此腳本輸出的 pocket_coords.npy。
"""

import os
import sys
import argparse
import csv
import warnings
from typing import Optional, List, Tuple, Dict

import numpy as np
warnings.filterwarnings("ignore")


# ── 依賴檢查 ──────────────────────────────────────────────────────────────────
try:
    from Bio import PDB as BioPDB
    from Bio.PDB.PDBIO import PDBIO
    from Bio.PDB.Polypeptide import is_aa
    _HAS_BIOPYTHON = True
except ImportError:
    print("[錯誤] 需要 BioPython：pip install biopython")
    sys.exit(1)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False
    print("[警告] RDKit 未安裝，部分功能受限。")


# =============================================================================
# 氨基酸常數
# =============================================================================

AA_LIST = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL"
]
AA_INDEX = {aa: i for i, aa in enumerate(AA_LIST)}

# 氨基酸特性（疏水/帶電/極性）
AA_HYDROPHOBIC  = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO"}
AA_POSITIVE     = {"ARG", "LYS", "HIS"}
AA_NEGATIVE     = {"ASP", "GLU"}
AA_POLAR        = {"SER", "THR", "ASN", "GLN", "CYS", "TYR", "GLY"}

# 標準殘基 MW（近似值）
AA_MW = {
    "ALA": 89, "ARG": 174, "ASN": 132, "ASP": 133, "CYS": 121,
    "GLN": 146, "GLU": 147, "GLY": 75, "HIS": 155, "ILE": 131,
    "LEU": 131, "LYS": 146, "MET": 149, "PHE": 165, "PRO": 115,
    "SER": 105, "THR": 119, "TRP": 204, "TYR": 181, "VAL": 117,
}


# =============================================================================
# 工具函式
# =============================================================================

def get_residue_features(resname: str) -> np.ndarray:
    """
    殘基特徵向量（25 維）：
      - 0–19  : 氨基酸類型 one-hot（20 維，未知=全 0）
      - 20    : 疏水性（1/0）
      - 21    : 帶正電（1/0）
      - 22    : 帶負電（1/0）
      - 23    : 極性（1/0）
      - 24    : 歸一化 MW（/ 200）
    """
    feat = np.zeros(25, dtype=np.float32)
    resname = resname.strip().upper()
    if resname in AA_INDEX:
        feat[AA_INDEX[resname]] = 1.0
    feat[20] = float(resname in AA_HYDROPHOBIC)
    feat[21] = float(resname in AA_POSITIVE)
    feat[22] = float(resname in AA_NEGATIVE)
    feat[23] = float(resname in AA_POLAR)
    feat[24] = AA_MW.get(resname, 0) / 200.0
    return feat


def get_ca_coord(residue) -> Optional[np.ndarray]:
    """取 Cα 座標，失敗時取第一個原子。"""
    try:
        return residue["CA"].get_vector().get_array()
    except KeyError:
        try:
            return next(residue.get_atoms()).get_vector().get_array()
        except StopIteration:
            return None


def parse_center_string(s: str) -> Optional[np.ndarray]:
    """把 '12.5 34.2 -8.7' 或 '12.5,34.2,-8.7' 轉為 numpy array。"""
    try:
        parts = s.replace(",", " ").split()
        if len(parts) != 3:
            return None
        return np.array([float(p) for p in parts])
    except ValueError:
        return None


def check_resolution(structure_file: str) -> Optional[float]:
    """
    從 PDB REMARK 中讀取解析度（Å）。
    """
    try:
        with open(structure_file, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("REMARK   2 RESOLUTION."):
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "ANGSTROMS.":
                            return float(parts[i-1])
    except Exception:
        pass
    return None


def find_ligands(chain) -> List[Dict]:
    """
    找出鏈中的 HETATM 配體（排除水分子和常見輔因子標記）。
    回傳 [{"name": str, "res": residue, "center": ndarray, "n_atoms": int}, ...]
    """
    EXCLUDE = {"HOH", "WAT", "H2O", "SO4", "PO4", "GOL", "EDO",
               "MPD", "MES", "ACT", "DMS", "FMT", "TRS", "PEG"}
    ligands = []
    for res in chain.get_residues():
        het_flag = res.get_id()[0]
        if not het_flag.startswith("H_"):
            continue
        resname = res.get_resname().strip()
        if resname in EXCLUDE:
            continue
        atoms = list(res.get_atoms())
        if not atoms:
            continue
        center = np.mean([a.get_vector().get_array() for a in atoms], axis=0)
        ligands.append({
            "name":    resname,
            "res":     res,
            "center":  center,
            "n_atoms": len(atoms),
        })
    return ligands


def fpocket_detect(pdb_path: str) -> Optional[np.ndarray]:
    """
    呼叫 fpocket 偵測口袋，回傳最高得分口袋的中心座標。
    需要系統安裝 fpocket（https://github.com/Discngine/fpocket）。
    """
    import subprocess, tempfile, re
    try:
        result = subprocess.run(
            ["fpocket", "-f", pdb_path],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"  [fpocket] 執行失敗：{result.stderr[:200]}")
            return None
        # 找 _out/pockets/pocket1_atm.pdb（得分最高）
        out_dir = os.path.splitext(pdb_path)[0] + "_out"
        pocket1 = os.path.join(out_dir, "pockets", "pocket1_atm.pdb")
        if not os.path.isfile(pocket1):
            return None
        # 讀取口袋原子座標取中心
        coords = []
        with open(pocket1) as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    try:
                        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                        coords.append([x, y, z])
                    except ValueError:
                        pass
        if coords:
            return np.mean(coords, axis=0)
    except FileNotFoundError:
        print("  [fpocket] 找不到 fpocket 執行檔，請確認已安裝。")
    except Exception as e:
        print(f"  [fpocket] 錯誤：{e}")
    return None


# =============================================================================
# 主流程
# =============================================================================

def prepare_pocket(
    pdb_path:      str,
    output_dir:    str       = "prepared_protein",
    chain_id:      str       = "",
    ligand_name:   str       = "",
    center_xyz:    Optional[np.ndarray] = None,
    cutoff:        float     = 10.0,
    use_fpocket:   bool      = False,
    min_residues:  int       = 5,
    max_residues:  int       = 150,
    keep_waters:   bool      = False,
    bfactor_cutoff:float     = 0.0,
) -> dict:
    """
    主清洗 + 口袋提取流程。

    Returns:
        dict 含 "coords" [N,3], "features" [N,25],
              "residues" list, "ligand" str, "cutoff" float
    """
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "coords": None, "features": None,
        "residues": [], "ligand": "", "cutoff": cutoff, "ok": False
    }

    print(f"\n{'='*60}")
    print(f"  蛋白質口袋預處理")
    print(f"  PDB：{pdb_path}")
    print(f"{'='*60}")

    # ── 解析度檢查 ────────────────────────────────────────────────────────
    resolution = check_resolution(pdb_path)
    if resolution:
        res_warn = "  ⚠ 解析度偏低，座標可靠性下降" if resolution > 3.0 else "  ✓"
        print(f"  解析度：{resolution:.2f} Å{res_warn}")
    else:
        print("  解析度：未找到（NMR 或資訊缺失）")

    # ── 讀取 PDB ──────────────────────────────────────────────────────────
    parser    = BioPDB.PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    model     = structure[0]

    # ── 鏈選擇 ────────────────────────────────────────────────────────────
    chains = list(model.get_chains())
    print(f"\n  鏈資訊（共 {len(chains)} 條）：")
    chain_info = []
    for ch in chains:
        residues = list(ch.get_residues())
        n_std = sum(1 for r in residues if r.get_id()[0] == " ")
        n_lig = sum(1 for r in residues if r.get_id()[0].startswith("H_"))
        chain_info.append({"id": ch.get_id(), "std": n_std, "lig": n_lig})
        print(f"    鏈 {ch.get_id():2s}：標準殘基={n_std:4d}  HETATM配體={n_lig:3d}")

    if chain_id:
        sel_chain_id = chain_id
    else:
        sel_chain_id = max(chain_info, key=lambda x: x["std"])["id"]
        print(f"  → 自動選擇鏈：{sel_chain_id}（殘基最多）")

    if sel_chain_id not in model:
        print(f"  [錯誤] 找不到鏈 {sel_chain_id}")
        return result
    sel_chain = model[sel_chain_id]

    # ── 缺失殘基偵測 ──────────────────────────────────────────────────────
    std_residues = [r for r in sel_chain.get_residues() if r.get_id()[0] == " "]
    resids = sorted([r.get_id()[1] for r in std_residues])
    gaps   = [resids[i+1] - resids[i] - 1
              for i in range(len(resids)-1) if resids[i+1] - resids[i] > 1]
    if gaps:
        print(f"  ⚠ 偵測到 {len(gaps)} 個缺失殘基片段"
              f"（最大空缺：{max(gaps)} 個殘基）")
    else:
        print(f"  ✓ 無缺失殘基（{len(std_residues)} 個標準殘基）")

    # ── 口袋中心定義 ──────────────────────────────────────────────────────
    pocket_center = center_xyz   # 可能已由 CLI 指定

    if pocket_center is None:
        # 策略 A：配體中心法
        ligands = find_ligands(sel_chain)

        if ligands:
            print(f"\n  偵測到 {len(ligands)} 個 HETATM 配體：")
            for i, lig in enumerate(ligands, 1):
                c = lig["center"]
                print(f"    {i}. {lig['name']:6s}  原子數={lig['n_atoms']:3d}"
                      f"  中心=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f})")

            if ligand_name:
                # 按名稱選
                match = next((l for l in ligands
                               if l["name"].upper() == ligand_name.upper()), None)
                if match:
                    pocket_center = match["center"]
                    result["ligand"] = match["name"]
                    print(f"  → 使用指定配體：{ligand_name}")
                else:
                    print(f"  [警告] 找不到配體 {ligand_name}，改用原子數最多的")
                    best = max(ligands, key=lambda x: x["n_atoms"])
                    pocket_center = best["center"]
                    result["ligand"] = best["name"]
            else:
                # 自動選原子數最多的
                best = max(ligands, key=lambda x: x["n_atoms"])
                pocket_center = best["center"]
                result["ligand"] = best["name"]
                print(f"  → 自動選擇配體：{best['name']}（原子數={best['n_atoms']}）")

        elif use_fpocket:
            # 策略 C：fpocket
            print("  → 無配體，嘗試 fpocket 偵測...")
            pocket_center = fpocket_detect(pdb_path)
            if pocket_center is not None:
                result["ligand"] = "fpocket"
                print(f"  → fpocket 偵測成功："
                      f"({pocket_center[0]:.1f},{pocket_center[1]:.1f},{pocket_center[2]:.1f})")
            else:
                print("  [錯誤] fpocket 偵測失敗。")
                return result
        else:
            print("  [錯誤] 未找到 HETATM 配體，且未啟用 fpocket。")
            print("  提示：使用 --center 'X Y Z' 手動指定口袋中心，"
                  "或 --use-fpocket。")
            return result

    else:
        result["ligand"] = "manual"
        print(f"  → 使用手動指定中心："
              f"({pocket_center[0]:.1f},{pocket_center[1]:.1f},{pocket_center[2]:.1f})")

    # ── 口袋殘基篩選 ──────────────────────────────────────────────────────
    def _collect_residues(cutoff_val):
        pocket_res = []
        for res in sel_chain.get_residues():
            if res.get_id()[0] != " ":
                continue   # 只要標準殘基
            ca = get_ca_coord(res)
            if ca is None:
                continue
            dist = float(np.linalg.norm(ca - pocket_center))
            if dist <= cutoff_val:
                # B-factor 過濾
                if bfactor_cutoff > 0:
                    try:
                        bfac = res["CA"].get_bfactor()
                        if bfac > bfactor_cutoff:
                            continue
                    except KeyError:
                        pass
                pocket_res.append({
                    "resname": res.get_resname().strip(),
                    "resid":   res.get_id()[1],
                    "ca":      ca,
                    "dist":    round(dist, 3),
                    "res_obj": res,
                })
        return sorted(pocket_res, key=lambda x: x["dist"])

    pocket_residues = _collect_residues(cutoff)
    print(f"\n  口袋殘基（{cutoff:.1f} Å 內）：{len(pocket_residues)} 個")

    # 殘基數警告 + 自動調整
    if len(pocket_residues) < min_residues:
        print(f"  ⚠ 口袋殘基過少（< {min_residues}），嘗試擴大截斷距離...")
        for trial_cutoff in [12.0, 15.0, 20.0]:
            pocket_residues = _collect_residues(trial_cutoff)
            if len(pocket_residues) >= min_residues:
                cutoff = trial_cutoff
                result["cutoff"] = cutoff
                print(f"  → 調整截斷距離至 {cutoff:.1f} Å，"
                      f"口袋殘基={len(pocket_residues)}")
                break
    elif len(pocket_residues) > max_residues:
        print(f"  ⚠ 口袋殘基過多（> {max_residues}），"
              f"建議縮小 cutoff 以減少記憶體消耗。")

    if len(pocket_residues) < min_residues:
        print(f"  [錯誤] 口袋殘基仍然過少（{len(pocket_residues)}），"
              f"無法繼續。")
        return result

    # ── 印出口袋摘要 ──────────────────────────────────────────────────────
    print(f"\n  前 10 個口袋殘基（依距離排序）：")
    aa_count = {}
    for r in pocket_residues[:10]:
        aa_count[r["resname"]] = aa_count.get(r["resname"], 0) + 1
        print(f"    {r['resname']:4s} {r['resid']:5d}  "
              f"dist={r['dist']:.2f} Å  "
              f"Cα=({r['ca'][0]:.1f},{r['ca'][1]:.1f},{r['ca'][2]:.1f})")

    # 氨基酸組成統計
    full_aa_count = {}
    for r in pocket_residues:
        full_aa_count[r["resname"]] = full_aa_count.get(r["resname"], 0) + 1
    top5_aa = sorted(full_aa_count.items(), key=lambda x: -x[1])[:5]
    print(f"\n  口袋氨基酸組成（前 5）：")
    for aa, cnt in top5_aa:
        print(f"    {aa}: {cnt} 個 ({cnt/len(pocket_residues)*100:.1f}%)")

    # ── 特徵化 ────────────────────────────────────────────────────────────
    coords   = np.array([r["ca"] for r in pocket_residues], dtype=np.float32)
    features = np.array([get_residue_features(r["resname"])
                         for r in pocket_residues], dtype=np.float32)

    print(f"\n  特徵矩陣：coords={coords.shape}  features={features.shape}")

    # ── 輸出 ──────────────────────────────────────────────────────────────
    coords_path  = os.path.join(output_dir, "pocket_coords.npy")
    feats_path   = os.path.join(output_dir, "pocket_features.npy")
    csv_path     = os.path.join(output_dir, "pocket_residues.csv")
    report_path  = os.path.join(output_dir, "pocket_report.txt")
    cleaned_pdb  = os.path.join(output_dir, "cleaned_protein.pdb")

    np.save(coords_path, coords)
    np.save(feats_path,  features)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["resname","resid","dist","ca_x","ca_y","ca_z"])
        w.writeheader()
        for r in pocket_residues:
            w.writerow({
                "resname": r["resname"], "resid": r["resid"],
                "dist":    r["dist"],
                "ca_x":    round(float(r["ca"][0]),3),
                "ca_y":    round(float(r["ca"][1]),3),
                "ca_z":    round(float(r["ca"][2]),3),
            })

    # 輸出清洗後 PDB（僅保留蛋白質主鏈）
    class ProteinSelect(BioPDB.Select):
        def accept_residue(self, residue):
            het = residue.get_id()[0]
            if het == " ":
                return True   # 標準殘基
            if keep_waters and het == "W":
                return True
            return False

    io = PDBIO()
    io.set_structure(structure)
    io.save(cleaned_pdb, ProteinSelect())

    # 報告
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  蛋白質口袋預處理報告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"PDB：{pdb_path}\n")
        f.write(f"鏈：{sel_chain_id}\n")
        if resolution:
            f.write(f"解析度：{resolution:.2f} Å\n")
        f.write(f"配體中心依據：{result['ligand']}\n")
        f.write(f"截斷距離：{cutoff:.1f} Å\n")
        f.write(f"口袋殘基數：{len(pocket_residues)}\n\n")
        f.write("[座標矩陣]\n")
        f.write(f"  形狀：{coords.shape}\n")
        f.write(f"  範圍 X：{coords[:,0].min():.2f} – {coords[:,0].max():.2f}\n")
        f.write(f"  範圍 Y：{coords[:,1].min():.2f} – {coords[:,1].max():.2f}\n")
        f.write(f"  範圍 Z：{coords[:,2].min():.2f} – {coords[:,2].max():.2f}\n\n")
        f.write("[氨基酸組成]\n")
        for aa, cnt in sorted(full_aa_count.items(), key=lambda x: -x[1]):
            f.write(f"  {aa}: {cnt}\n")
        f.write(f"\n[輸出]\n")
        f.write(f"  座標        : {coords_path}\n")
        f.write(f"  特徵矩陣    : {feats_path}\n")
        f.write(f"  殘基 CSV    : {csv_path}\n")
        f.write(f"  清洗後 PDB  : {cleaned_pdb}\n")

    print(f"\n{'='*60}")
    print(f"  口袋提取完成！殘基數：{len(pocket_residues)}")
    print(f"  座標   → {coords_path}")
    print(f"  特徵   → {feats_path}")
    print(f"  殘基   → {csv_path}")
    print(f"  報告   → {report_path}")
    print(f"{'='*60}")
    print(f"\n[下一步] 在 GpuQsarEngine 中啟用 Pocket-Aware 模式：")
    print(f"  1. 互動式選單選擇「啟用蛋白質口袋 Cross-Attention」")
    print(f"  2. 輸入 PDB 路徑，程式自動呼叫口袋偵測")
    print(f"  或在程式碼中直接載入：")
    print(f"     import numpy as np")
    print(f"     pocket_coords = np.load('{coords_path}')")
    print(f"     pocket_feats  = np.load('{feats_path}')")

    result.update({
        "coords":   coords,
        "features": features,
        "residues": pocket_residues,
        "ok": True,
    })
    return result


# =============================================================================
# CLI 入口
# =============================================================================


# =============================================================================
# 功能 B：口袋品質評分（Drug-likeness of Binding Site）
# =============================================================================

def score_pocket_quality(pocket_residues: list, coords: np.ndarray) -> dict:
    """
    對已提取的口袋進行品質評估：

    評分維度：
      1. 殘基多樣性  — 氨基酸種類多樣性（越多越好）
      2. 疏水核心    — 疏水殘基比例（藥物結合通常需要 20-50%）
      3. 口袋緊密度  — Cα 座標的標準差（太鬆散不適合小分子結合）
      4. 帶電殘基    — 正/負電比例（影響靜電相互作用）
      5. 口袋體積估算 — 用 Cα 包絡球估算（Å³）

    Returns:
        dict 含各維度評分（0-100）和總分
    """
    if not pocket_residues or coords is None or len(coords) == 0:
        return {"total": 0, "ok": False}

    n = len(pocket_residues)
    resnames = [r["resname"] for r in pocket_residues]

    # 1. 殘基多樣性（不同氨基酸種類 / 20）
    unique_aa = len(set(resnames))
    diversity = min(unique_aa / 20 * 100, 100)

    # 2. 疏水性（疏水殘基比例，目標 25-45%）
    HYDROPHOBIC = {"ALA","VAL","ILE","LEU","MET","PHE","TRP","PRO"}
    hydro_ratio = sum(1 for r in resnames if r in HYDROPHOBIC) / n
    hydro_score = 100 - abs(hydro_ratio - 0.35) / 0.35 * 100
    hydro_score = max(0, min(100, hydro_score))

    # 3. 口袋緊密度（Cα 標準差，越小越緊密）
    centroid  = coords.mean(axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)
    compactness = max(0, 100 - distances.std() * 5)

    # 4. 帶電殘基平衡（正負電接近時分數高）
    POS = {"ARG","LYS","HIS"}
    NEG = {"ASP","GLU"}
    pos_r = sum(1 for r in resnames if r in POS) / n
    neg_r = sum(1 for r in resnames if r in NEG) / n
    charge_score = 100 - abs(pos_r - neg_r) * 200

    # 5. 口袋體積估算（用凸包估算，Å³）
    try:
        from scipy.spatial import ConvexHull
        hull    = ConvexHull(coords)
        volume  = hull.volume
        # 理想口袋體積 300-1000 Å³
        vol_score = 100 if 300 <= volume <= 1000 else                     max(0, 100 - abs(volume - 650) / 6.5)
    except Exception:
        volume, vol_score = 0, 50

    # 加權總分
    total = (diversity   * 0.20 +
             hydro_score * 0.25 +
             compactness * 0.20 +
             charge_score* 0.15 +
             vol_score   * 0.20)

    return {
        "total":         round(total, 1),
        "diversity":     round(diversity, 1),
        "hydrophobicity":round(hydro_score, 1),
        "compactness":   round(compactness, 1),
        "charge_balance":round(max(0, charge_score), 1),
        "volume_score":  round(vol_score, 1),
        "n_residues":    n,
        "estimated_vol": round(volume, 1),
        "unique_aa":     unique_aa,
        "ok":            True,
    }


# =============================================================================
# 功能 C：批次多配體口袋提取
# =============================================================================

def batch_pocket_extraction(
    pdb_path:   str,
    output_dir: str  = "prepared_protein_batch",
    chain_id:   str  = "",
    cutoff:     float= 10.0,
) -> list:
    """
    自動提取 PDB 中所有 HETATM 配體的口袋，適用於：
      - 含多個結合位點的蛋白質（如二聚體）
      - 共晶結構含多個配體的情況
      - 需要比較不同口袋的研究

    Returns:
        list of dict，每個元素含口袋資訊和品質評分
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        from Bio import PDB as _BioPDB
        parser    = _BioPDB.PDBParser(QUIET=True)
        structure = parser.get_structure("prot", pdb_path)
        model     = structure[0]
    except Exception as e:
        print(f"  [錯誤] PDB 讀取失敗：{e}")
        return []

    chains = list(model.get_chains())
    if chain_id:
        chains = [ch for ch in chains if ch.get_id() == chain_id]

    all_ligands = []
    for ch in chains:
        ligs = find_ligands(ch)
        for lig in ligs:
            lig["chain"] = ch.get_id()
        all_ligands.extend(ligs)

    if not all_ligands:
        print("  [批次提取] 未找到任何 HETATM 配體。")
        return []

    print(f"\n  [批次提取] 共偵測到 {len(all_ligands)} 個配體")
    results = []

    for i, lig in enumerate(all_ligands, 1):
        lig_name   = lig["name"]
        lig_chain  = lig["chain"]
        lig_center = lig["center"]
        sub_dir    = os.path.join(output_dir, f"pocket_{i:02d}_{lig_name}")

        print(f"\n  [{i}/{len(all_ligands)}] 配體 {lig_name} (鏈 {lig_chain})")
        pocket_result = prepare_pocket(
            pdb_path    = pdb_path,
            output_dir  = sub_dir,
            chain_id    = lig_chain,
            center_xyz  = lig_center,
            cutoff      = cutoff,
            min_residues= 3,
        )
        if pocket_result.get("ok"):
            score = score_pocket_quality(
                pocket_result["residues"],
                pocket_result["coords"]
            )
            pocket_result["ligand_name"] = lig_name
            pocket_result["chain"]       = lig_chain
            pocket_result["quality"]     = score
            results.append(pocket_result)
            print(f"     品質評分：{score['total']:.1f}/100  "
                  f"殘基={score['n_residues']}  "
                  f"體積估算={score['estimated_vol']:.0f} Å³")

    # 按品質評分排序
    results.sort(key=lambda x: x.get("quality", {}).get("total", 0), reverse=True)

    # 輸出比較報告
    report_path = os.path.join(output_dir, "pocket_comparison.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("批次口袋提取報告\n")
        f.write("=" * 60 + "\n\n")
        for i, r in enumerate(results, 1):
            q = r.get("quality", {})
            f.write(f"#{i} 配體 {r['ligand_name']} (鏈 {r['chain']})\n")
            f.write(f"  總分    : {q.get('total', 0):.1f}/100\n")
            f.write(f"  殘基數  : {q.get('n_residues', 0)}\n")
            f.write(f"  多樣性  : {q.get('diversity', 0):.1f}\n")
            f.write(f"  疏水性  : {q.get('hydrophobicity', 0):.1f}\n")
            f.write(f"  緊密度  : {q.get('compactness', 0):.1f}\n")
            f.write(f"  體積估算: {q.get('estimated_vol', 0):.0f} Å³\n")
            f.write(f"  輸出狀態: {'已輸出 ✓' if r.get('ok', False) else '失敗 ✗'}\n\n")

    print(f"\n  比較報告 → {report_path}")
    if results:
        best = results[0]
        print(f"  最佳口袋：{best['ligand_name']}（鏈 {best['chain']}）"
              f"  得分 {best['quality']['total']:.1f}/100")
    return results

# =============================================================================
# 互動式輔助函式
# =============================================================================

def _iask_prot(prompt: str, default=None, cast=str,
               choices: list = None, is_path: bool = False):
    """互動式單行輸入，非互動環境直接回傳預設值。"""
    import sys
    is_tty = sys.stdin.isatty()
    default_str = str(default) if default is not None else ""
    if choices:
        hint = f"[{'/'.join(choices)}]（預設 {default_str}）"
    else:
        hint = f"（預設 {default_str}，Enter 接受）" if default_str else ""

    if not is_tty:
        return default

    while True:
        raw = input(f"  {prompt} {hint}: ").strip()
        if not raw:
            return default
        if is_path:
            return raw.strip('"').strip("'")
        if choices and raw not in choices:
            print(f"  ✗ 請輸入以下之一：{choices}")
            continue
        try:
            return cast(raw)
        except (ValueError, TypeError):
            print(f"  ✗ 格式錯誤，預期 {cast.__name__}")


def _section_prot(title: str):
    print(f"\n  ╔{'═'*(len(title)+4)}╗")
    print(f"  ║  {title}  ║")
    print(f"  ╚{'═'*(len(title)+4)}╝")


def _quick_pdb_scan(pdb_path: str):
    """
    快速掃描 PDB，印出鏈資訊 + 配體清單 + 解析度，
    幫助使用者在後續問答中做出正確選擇。
    """
    print(f"\n  ── PDB 快速掃描 ──")
    try:
        from Bio import PDB as _BioPDB
        parser    = _BioPDB.PDBParser(QUIET=True)
        structure = parser.get_structure("scan", pdb_path)
        model     = structure[0]

        chains = list(model.get_chains())
        print(f"  鏈數：{len(chains)}")
        for ch in chains:
            residues = list(ch.get_residues())
            n_std = sum(1 for r in residues if r.get_id()[0] == " ")
            ligs  = [r.get_resname().strip() for r in residues
                     if r.get_id()[0].startswith("H_")
                     and r.get_resname().strip() not in
                     {"HOH","WAT","SO4","PO4","GOL","EDO","MPD"}]
            print(f"    鏈 {ch.get_id():2s}：標準殘基={n_std:4d}"
                  f"  HETATM配體={ligs[:5]}")

        resolution = check_resolution(pdb_path)
        if resolution:
            warn = " ⚠ 偏低" if resolution > 3.0 else " ✓"
            print(f"  解析度：{resolution:.2f} Å{warn}")
        else:
            print("  解析度：未找到（NMR 或 CRYO-EM）")
    except Exception as e:
        print(f"  (快速掃描失敗：{e})")


# =============================================================================
# 互動式主程式
# =============================================================================

def interactive_main_protein(args):
    """
    互動式流程。CLI 已提供的旗標直接使用，未提供的進入問答。
    """
    import sys

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║       蛋白質口袋預處理 — 互動式設定                       ║")
    print("║  直接 Enter 套用預設值；輸入新值後按 Enter 確認           ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ════════════════════════════════════════════════════════════════
    # 步驟 0：選擇準備資料的用途
    # ════════════════════════════════════════════════════════════════
    _section_prot("步驟 0：選擇處理用途（可多選，空格分隔）")
    print("  A. 單一口袋提取（最常用，Pocket-Aware 訓練用）")
    print("  B. 批次提取所有配體口袋（含品質評分與比較）")
    print("  C. 僅進行口袋品質評估（已有 pocket_coords.npy）")
    print()
    _prot_mode_raw = _iask_prot("用途選項", default="A").upper().split()
    _prot_modes    = set(_prot_mode_raw) if _prot_mode_raw else {"A"}
    print(f"  → 已選：{' '.join(sorted(_prot_modes))}")

    # ── 步驟 1：PDB 檔案 ─────────────────────────────────────────────────
    _section_prot("步驟 1／6：輸入 PDB 檔案")
    if args.pdb:
        pdb_path = args.pdb
        print(f"  [CLI] PDB：{pdb_path}")
    else:
        while True:
            pdb_path = _iask_prot(
                "請輸入 PDB 檔案路徑", default="", is_path=True)
            if pdb_path and os.path.isfile(pdb_path):
                break
            print(f"  ✗ 找不到檔案：{pdb_path!r}，請重新輸入。")

    # 快速掃描，印出鏈 + 配體清單讓使用者參考
    _quick_pdb_scan(pdb_path)

    # ── 步驟 2：鏈選擇 ───────────────────────────────────────────────────
    _section_prot("步驟 2／6：鏈選擇")
    if args.chain:
        chain_id = args.chain
        print(f"  [CLI] 鏈：{chain_id}")
    else:
        chain_id = _iask_prot(
            "指定蛋白質鏈 ID（留空=自動選最大鏈）",
            default="")

    # ── 步驟 3：口袋定義策略 ─────────────────────────────────────────────
    _section_prot("步驟 3／6：口袋定義策略")
    print("  可選策略：")
    print("    A. 配體中心法（自動偵測 HETATM 配體，最常用）")
    print("    B. 手動輸入口袋中心座標（X Y Z）")
    print("    C. fpocket 自動偵測（需另外安裝 fpocket）")

    center_xyz  = None
    ligand_name = ""
    use_fpocket = False

    # 若 CLI 已指定 center，直接使用
    if args.center:
        center_xyz = parse_center_string(args.center)
        if center_xyz is None:
            print(f"  ✗ 無法解析座標：{args.center!r}")
            sys.exit(1)
        print(f"  [CLI] 手動座標：{center_xyz}")
        strategy = "B"
    elif args.use_fpocket:
        strategy = "C"
        use_fpocket = True
        print("  [CLI] 使用 fpocket")
    else:
        strategy = _iask_prot(
            "選擇口袋定義策略",
            default="A", choices=["A", "B", "C"])

    if strategy == "A":
        # 配體中心法：可選擇指定配體名稱
        if args.ligand:
            ligand_name = args.ligand
            print(f"  [CLI] 指定配體：{ligand_name}")
        else:
            ligand_name = _iask_prot(
                "指定配體名稱（三字母代碼，例如 ATP；留空=自動選原子數最多者）",
                default="")
        print("  → 將自動偵測 HETATM 配體並以其為口袋中心")

    elif strategy == "B":
        if center_xyz is None:
            print("  請輸入口袋中心座標（Å）")
            while True:
                xyz_raw = _iask_prot(
                    "座標（格式：X Y Z，例如 12.5 34.2 -8.7）",
                    default="")
                center_xyz = parse_center_string(xyz_raw)
                if center_xyz is not None:
                    break
                print("  ✗ 格式錯誤，請重新輸入（三個數字以空格分隔）")
        print(f"  → 口袋中心：({center_xyz[0]:.2f}, {center_xyz[1]:.2f}, {center_xyz[2]:.2f})")

    elif strategy == "C":
        use_fpocket = True
        print("  → 將呼叫 fpocket 偵測口袋（需確認已安裝）")

    # ── 步驟 4：口袋參數 ─────────────────────────────────────────────────
    _section_prot("步驟 4／6：口袋參數")
    cutoff = _iask_prot(
        "口袋截斷距離（Å，Cα 到配體中心的距離上限）",
        default=10.0, cast=float)
    min_res = _iask_prot(
        "口袋最少殘基數（不足時自動擴大截斷距離）",
        default=5, cast=int)
    max_res = _iask_prot(
        "口袋最多殘基數警告閾值（超過時提示縮小 cutoff）",
        default=150, cast=int)

    # ── 步驟 5：進階選項 ─────────────────────────────────────────────────
    _section_prot("步驟 5／6：進階選項")
    keep_waters = args.keep_waters
    if not keep_waters:
        kw_raw = _iask_prot(
            "是否保留結構中的水分子",
            default="n", choices=["y", "n"])
        keep_waters = kw_raw.lower() == "y"

    bfactor_cutoff = args.bfactor_cutoff
    if bfactor_cutoff == 0.0:
        bf_raw = _iask_prot(
            "B-factor 上限（0=不過濾，建議 80.0 過濾高可動性殘基）",
            default=0.0, cast=float)
        bfactor_cutoff = bf_raw

    # ── 步驟 6：輸出目錄 + 確認 ──────────────────────────────────────────
    _section_prot("步驟 6／6：輸出設定與確認")
    output_dir = args.output_dir or _iask_prot(
        "輸出目錄路徑",
        default="prepared_protein", is_path=True)

    print(f"\n  ── 設定摘要 ──")
    print(f"  PDB 檔案    : {pdb_path}")
    print(f"  鏈選擇      : {'自動' if not chain_id else chain_id}")
    print(f"  口袋策略    : {strategy}"
          + (f"（配體={ligand_name or '自動'}）" if strategy == "A" else
             f"（座標={center_xyz}）"            if strategy == "B" else
             "（fpocket）"))
    print(f"  截斷距離    : {cutoff:.1f} Å")
    print(f"  殘基數範圍  : {min_res} – {max_res}")
    print(f"  保留水分子  : {'是' if keep_waters else '否'}")
    print(f"  B-factor 上限: {bfactor_cutoff if bfactor_cutoff > 0 else '不過濾'}")
    print(f"  輸出目錄    : {output_dir}")

    confirm = _iask_prot("\n以上設定確認，開始處理？",
                         default="y", choices=["y", "n"])
    if confirm.lower() != "y":
        print("  已取消。")
        sys.exit(0)

    # ── 執行 ─────────────────────────────────────────────────────────────
    if "A" in _prot_modes or strategy in ("A","B","C"):
        result = prepare_pocket(
            pdb_path       = pdb_path,
            output_dir     = output_dir,
            chain_id       = chain_id,
            ligand_name    = ligand_name,
            center_xyz     = center_xyz,
            cutoff         = cutoff,
            use_fpocket    = use_fpocket,
            min_residues   = min_res,
            max_residues   = max_res,
            keep_waters    = keep_waters,
            bfactor_cutoff = bfactor_cutoff,
        )

        # 口袋品質評分（自動執行）
        if result.get("ok") and result.get("residues") and result.get("coords") is not None:
            _section_prot("口袋品質評估")
            quality = score_pocket_quality(result["residues"], result["coords"])
            print(f"  ┌─── 口袋品質評分 ─────────────────────────────────┐")
            print(f"  │  總分        : {quality['total']:5.1f} / 100                    │")
            print(f"  │  殘基多樣性  : {quality['diversity']:5.1f}  （氨基酸種類豐富度）    │")
            print(f"  │  疏水性      : {quality['hydrophobicity']:5.1f}  （理想 25-45% 疏水殘基）│")
            print(f"  │  緊密度      : {quality['compactness']:5.1f}  （口袋形狀緊密程度）    │")
            print(f"  │  電荷平衡    : {quality['charge_balance']:5.1f}  （正負電殘基比例）    │")
            print(f"  │  體積評分    : {quality['volume_score']:5.1f}  （估算 {quality['estimated_vol']:.0f} Å³）         │")
            print(f"  └──────────────────────────────────────────────────┘")
            if quality["total"] < 50:
                print("  ⚠ 品質評分偏低，建議：")
                print("    - 嘗試不同的截斷距離（cutoff 8–12 Å）")
                print("    - 確認配體選擇是否正確")
                print("    - 考慮使用 fpocket 重新偵測口袋")
            elif quality["total"] >= 75:
                print("  ✓ 口袋品質良好，適合 Pocket-Aware QSAR 訓練")

    # ── 批次模式 B ──────────────────────────────────────────────────────
    if "B" in _prot_modes:
        _section_prot("批次口袋提取（模式 B）")
        _batch_out = _iask_prot("批次輸出目錄",
                                default=output_dir + "_batch", is_path=True)
        _batch_cutoff = _iask_prot("每個口袋的截斷距離 Å",
                                   default=cutoff, cast=float)
        print()
        batch_pocket_extraction(
            pdb_path   = pdb_path,
            output_dir = _batch_out,
            chain_id   = chain_id,
            cutoff     = _batch_cutoff,
        )


def main():
    parser = argparse.ArgumentParser(
        description="蛋白質口袋預處理腳本（GpuQsarEngine Pocket-Aware 模式前置處理）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="不帶任何參數直接執行進入互動模式；帶 CLI 旗標則跳過對應問答。",
    )
    parser.add_argument("--pdb",            default="",
                        help="PDB 檔案路徑（留空進入互動輸入）")
    parser.add_argument("--chain",          default="",
                        help="指定鏈 ID（預設自動選最大鏈）")
    parser.add_argument("--ligand",         default="",
                        help="指定配體名稱（三字母代碼，例如 ATP）")
    parser.add_argument("--center",         default="",
                        help="手動口袋中心座標（格式：'X Y Z'）")
    parser.add_argument("--cutoff",         default=0.0, type=float,
                        help="口袋截斷距離 Å（0=互動詢問，預設 10.0）")
    parser.add_argument("--output-dir",     default="",
                        help="輸出目錄（空=互動詢問）")
    parser.add_argument("--use-fpocket",    action="store_true",
                        help="使用 fpocket 偵測口袋")
    parser.add_argument("--keep-waters",    action="store_true",
                        help="保留水分子")
    parser.add_argument("--bfactor-cutoff", default=0.0, type=float,
                        help="B-factor 上限（0=不過濾）")
    parser.add_argument("--min-residues",   default=5, type=int,
                        help="口袋最少殘基數（預設 5）")
    parser.add_argument("--max-residues",   default=150, type=int,
                        help="口袋最多殘基數警告閾值（預設 150）")
    args = parser.parse_args()

    # cutoff 為 0 時改為互動詢問預設值
    if args.cutoff == 0.0:
        args.cutoff = 0.0   # 讓 interactive_main_protein 處理

    interactive_main_protein(args)


if __name__ == "__main__":
    main()
