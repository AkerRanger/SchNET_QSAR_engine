"""
prepare_ligands.py — 小分子資料預處理腳本
==========================================
用途：將原始小分子資料（SDF / CSV / SMILES）轉換成 GpuQsarEngine 可直接使用的格式。

功能：
  1. 多格式讀取：SDF / CSV（含 SMILES 欄）/ 純 SMILES 文字檔
  2. IC50 → pIC50 單位換算（nM / uM / M）
  3. 資料品質驗證：
       - SMILES 合法性（RDKit 解析）
       - 活性值範圍過濾（pIC50 < 3 或 > 12 視為異常）
       - 重複 SMILES 去除（保留活性值最高者）
       - 有機小分子過濾（排除無機鹽、金屬配合物）
  4. 3D 構象生成：ETKDGv3 + MMFF94s 最小化（多線程加速）
  5. 藥物相似性計算：MW / cLogP / HBA / HBD / TPSA / QED / SA Score
  6. 化學多樣性報告：Tanimoto 相似度矩陣 + Scaffold 統計
  7. 輸出：
       - cleaned_ligands.sdf   ← 可直接餵入 GpuQsarEngine
       - cleaned_ligands.csv   ← 含 SMILES + pIC50 + ADMET 描述符
       - ligand_report.txt     ← 資料品質報告

使用方式：
  # 從 SDF（IC50 欄位為 nM）
  python prepare_ligands.py --input raw_data.sdf --label-field IC50 --unit nM

  # 從 CSV（pIC50 直接使用）
  python prepare_ligands.py --input data.csv --smiles-col SMILES --label-col pIC50 --no-convert

  # 指定輸出目錄 + 線程數
  python prepare_ligands.py --input data.sdf --output-dir prepared/ --workers 16
"""

import os
import sys
import argparse
import csv
import math
import warnings
from typing import Optional, List, Tuple

warnings.filterwarnings("ignore")

# ── 依賴檢查 ──────────────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import (
        AllChem, Descriptors, rdMolDescriptors,
        MolStandardize, QED, RDConfig, ChemicalFeatures,
    )
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    print("[錯誤] 需要 RDKit：conda install -c conda-forge rdkit")
    sys.exit(1)

try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
    _HAS_MURCKO = True
except ImportError:
    _HAS_MURCKO = False

import numpy as np
from concurrent.futures import (ThreadPoolExecutor, ProcessPoolExecutor, as_completed)


# =============================================================================
# 工具函式
# =============================================================================


# =============================================================================
# 資料品質過濾：standard_relation 處理（三種方法）
# =============================================================================

def _normalize_relation(raw: str) -> str:
    """
    將 ChEMBL/SDF 中各種 relation 字串統一為 '=' / '>' / '<'。
    常見原始值：'=' / '>' / '<' / '>=' / '<=' /
               "'='" / ">" / "<" / "greater than" / "less than"
    """
    s = raw.strip().strip("'").strip('"').lower()
    if s in ("=", "==", "equal", "exactly"):
        return "="
    if s in (">", ">=", "greater", "greater than", "gthan", "g"):
        return ">"
    if s in ("<", "<=", "less", "less than", "lthan", "l"):
        return "<"
    # 若無法識別，視為精確值
    return "="


def _print_relation_summary(records: list) -> None:
    """印出 relation 分佈統計。"""
    from collections import Counter
    if not records:
        return
    # 支援 3-tuple（舊格式）和 4-tuple（新格式）
    if len(records[0]) < 4:
        return
    cnt = Counter(r[3] for r in records)
    total = len(records)
    print(f"  Standard Relation 分佈：", end="")
    for rel, n in sorted(cnt.items()):
        pct = n / total * 100
        symbol = {"=": "精確值(=)", ">": "截尾上界(>)", "<": "截尾下界(<)"}.get(rel, rel)
        print(f"  {symbol}={n}({pct:.1f}%)", end="")
    print()


def apply_relation_filter(
    records: list,
    method: str = "strict",
    pic50_lo_boundary: float = 7.0,   # < boundary：用於方法2的截尾下界替換值（pIC50）
    pic50_hi_boundary: float = 4.92,  # > boundary：用於方法2的截尾上界替換值（pIC50）
    exact_weight: float = 1.0,
    censored_weight: float = 0.2,
) -> list:
    """
    依指定方法處理 standard_relation，回傳清洗後的 records 列表。

    每個 record 格式：(mol, pic50, smiles, relation)
    輸出格式相同，但 method="weighting" 時在 smiles 後附加 sample_weight。

    method 選項：
    ─────────────────────────────────────────────────────────────────
    "strict"    嚴格過濾法（衝刺 R² 首選）
                 僅保留 relation='=' 的精確值，刪除所有 > 和 <。
                 建議：過濾後 ≥ 1000 筆時使用。

    "impute"    邊界修正法（資料量不足時的備用方案）
                 relation='=' → 保留原值
                 relation='<'（截尾下界，如 < 200nM = pIC50 > 7.0）
                    → 替換為 pic50_lo_boundary（預設 7.0）
                 relation='>'（截尾上界，如 > 6000nM = pIC50 < 4.92）
                    → 替換為 pic50_hi_boundary（預設 4.92）
                 風險：R² 圖在邊界點會出現垂直點陣，略微拉低 R²。

    "weighting" 加權法（最科學，需訓練框架支援 sample_weight）
                 保留全部數據，但附加 sample_weight：
                   relation='='     → weight = exact_weight（預設 1.0）
                   relation='>'/'<' → weight = censored_weight（預設 0.2）
                 輸出 CSV 會多一欄 sample_weight。
                 在 GpuQsarEngine 的加權損失函數中使用此欄位。
    ─────────────────────────────────────────────────────────────────

    Returns:
        list of (mol, pic50, smiles, relation) or
                (mol, pic50, smiles, relation, sample_weight)  [weighting 模式]
    """
    if not records:
        return records

    # 支援舊版 3-tuple（無 relation 欄位）直接回傳
    if len(records[0]) < 4:
        print("  [Relation] 資料集無 relation 欄位，跳過過濾。")
        return records

    n_exact    = sum(1 for r in records if r[3] == "=")
    n_censored = len(records) - n_exact

    print(f"\n  [Relation Filter] 方法：{method}")
    print(f"  精確值(=)={n_exact}  截尾值(>/< )={n_censored}  共{len(records)}筆")

    if method == "strict":
        filtered = [r for r in records if r[3] == "="]
        n_kept = len(filtered)
        print(f"  → 嚴格過濾：保留 {n_kept} 筆精確值，刪除 {len(records)-n_kept} 筆截尾值")
        if n_kept < 1000:
            print(f"  ⚠ 過濾後僅 {n_kept} 筆（< 1000），建議改用 impute 或 weighting 方法")
        elif n_kept >= 3000:
            print(f"  ✓ {n_kept} 筆高品質精確值，R² 優化效果最佳")
        return filtered

    elif method == "impute":
        result = []
        n_replaced = 0
        for r in records:
            mol, pic50, smi, rel = r
            if rel == "<":
                # < X nM → 真實值比 X 更有活性（pIC50 更高）→ 保守替換為下界
                new_pic50 = pic50_lo_boundary
                n_replaced += 1
                result.append((mol, new_pic50, smi, "="))
            elif rel == ">": 
                # > X nM → 真實值比 X 更無活性（pIC50 更低）→ 保守替換為上界
                new_pic50 = pic50_hi_boundary
                n_replaced += 1
                result.append((mol, new_pic50, smi, "="))
            else:
                result.append(r)
        print(f"  → 邊界修正：{n_replaced} 筆截尾值替換（<→{pic50_lo_boundary:.2f} / >→{pic50_hi_boundary:.2f}）")
        print(f"  ⚠ 注意：R² 圖在 pIC50={pic50_lo_boundary:.2f} 和 {pic50_hi_boundary:.2f} 點會出現垂直點陣")
        return result

    elif method == "weighting":
        result = []
        for r in records:
            mol, pic50, smi, rel = r
            w = exact_weight if rel == "=" else censored_weight
            result.append((mol, pic50, smi, rel, w))
        print(f"  → 加權法：精確值 weight={exact_weight}，截尾值 weight={censored_weight}")
        print(f"  ✓ 全部 {len(result)} 筆保留，sample_weight 欄位已加入輸出 CSV")
        return result

    else:
        print(f"  ⚠ 不支援的方法：{method!r}，略過過濾")
        return records


def ic50_to_pic50(value: float, unit: str = "nM") -> Optional[float]:
    """IC50 → pIC50（-log10[M]）。"""
    factors = {"nm": 1e-9, "um": 1e-6, "µm": 1e-6, "mm": 1e-3, "m": 1.0}
    f = factors.get(unit.lower(), 1e-9)
    try:
        v = float(value) * f
        if v <= 0:
            return None
        return -math.log10(v)
    except (ValueError, TypeError):
        return None


def is_organic_small_molecule(mol) -> bool:
    """
    判斷是否為有機小分子：
      - 含碳原子
      - 分子量 100–1000 Da
      - 無金屬原子（排除金屬配合物）
    """
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    if mw < 100 or mw > 1000:
        return False
    has_carbon = any(a.GetAtomicNum() == 6 for a in mol.GetAtoms())
    if not has_carbon:
        return False
    # 金屬原子列表（常見過渡金屬 + 主族金屬）
    metals = {
        3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 37, 38, 39, 40, 41, 42, 43, 44,
        45, 46, 47, 48, 49, 50, 55, 56, 57, 72, 73, 74, 75,
        76, 77, 78, 79, 80, 81, 82, 83
    }
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in metals:
            return False
    return True


def standardize_mol(mol):
    """
    標準化分子：
      1. 去除 salt（保留最大片段）
      2. 中性化電荷（去除反離子效應）
      3. 統一芳香性標記
    """
    try:
        # 保留最大片段
        remover = rdMolStandardize.LargestFragmentChooser()
        mol = remover.choose(mol)
        # 中性化
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return mol


def compute_admet(mol) -> dict:
    """計算常用 ADMET 描述符。"""
    try:
        mw   = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hba  = rdMolDescriptors.CalcNumHBA(mol)
        hbd  = rdMolDescriptors.CalcNumHBD(mol)
        tpsa = Descriptors.TPSA(mol)
        rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
        qed  = QED.qed(mol)
        # SA Score（合成可行性）
        try:
            from rdkit.Contrib.SA_Score import sascorer
            sa = sascorer.calculateScore(mol)
        except Exception:
            sa = None
        # Lipinski RO5
        ro5 = int(mw <= 500 and logp <= 5 and hba <= 10 and hbd <= 5)
        return {
            "MW": round(mw, 2), "cLogP": round(logp, 3),
            "HBA": hba, "HBD": hbd, "TPSA": round(tpsa, 2),
            "RotBonds": rotb, "QED": round(qed, 4),
            "SA_Score": round(sa, 3) if sa else None,
            "Lipinski_RO5": ro5,
        }
    except Exception:
        return {}


def get_murcko_scaffold(mol) -> str:
    """取 Murcko Scaffold（SMILES）。"""
    if not _HAS_MURCKO or mol is None:
        return ""
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return ""



# =============================================================================
# 多進程 3D 生成 Worker（頂層函式，ProcessPoolExecutor 用）
# =============================================================================

def _minimize_3d_worker(args):
    """
    ProcessPoolExecutor worker — 必須是頂層函式才可序列化。
    args: (idx, smiles, mmff_variant, n_conformers)
    returns: (idx, mol3d_or_None, pic50, smiles)
    """
    idx, smiles, pic50, mmff_variant, n_conformers = args
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return idx, None, pic50, smiles
        mol3d = minimize_3d(mol, mmff_variant, n_conformers)
        return idx, mol3d, pic50, smiles
    except Exception:
        return idx, None, pic50, smiles


def minimize_3d(mol, mmff_variant: str = "MMFF94s",
                n_conformers: int = 1) -> Optional[object]:
    """ETKDGv3 嵌入 + MMFF 最小化，回傳含 3D 座標的 mol（或 None）。"""
    try:
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.numThreads = 1

        if n_conformers > 1:
            cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n_conformers,
                                              params=params)
            if not cids:
                raise RuntimeError("EmbedMultipleConfs failed")
            # 選最低能量構象
            props_obj = AllChem.MMFFGetMoleculeProperties(mol_h,
                                                          mmffVariant=mmff_variant)
            best_e, best_cid = float("inf"), cids[0]
            for cid in cids:
                ff = AllChem.MMFFGetMoleculeForceField(mol_h, props_obj, confId=cid)
                if ff and ff.Minimize(maxIts=500) == 0:
                    e = ff.CalcEnergy()
                    if e < best_e:
                        best_e, best_cid = e, cid
            # 只保留最低能量構象
            pos = mol_h.GetConformer(best_cid).GetPositions().copy()
            mol_h.RemoveAllConformers()
            AllChem.EmbedMolecule(mol_h, params)
            for i, xyz in enumerate(pos):
                mol_h.GetConformer(0).SetAtomPosition(i, xyz.tolist())
        else:
            if AllChem.EmbedMolecule(mol_h, params) != 0:
                # fallback: ETDG
                params2 = AllChem.ETDGv2() if hasattr(AllChem, 'ETDGv2') else AllChem.ETKDG()
                if AllChem.EmbedMolecule(mol_h, params2) != 0:
                    return None
            props_obj = AllChem.MMFFGetMoleculeProperties(mol_h,
                                                          mmffVariant=mmff_variant)
            ff = AllChem.MMFFGetMoleculeForceField(mol_h, props_obj)
            if ff is None:
                # fallback: UFF
                ff = AllChem.UFFGetMoleculeForceField(mol_h)
            if ff:
                ff.Minimize(maxIts=500)

        mol_3d = Chem.RemoveHs(mol_h)
        if mol_3d.GetNumConformers() == 0:
            return None
        return mol_3d
    except Exception:
        return None


# =============================================================================
# 讀取函式
# =============================================================================

def read_sdf(path: str, label_field: str,
             convert_ic50: bool = True, unit: str = "nM",
             relation_field: str = "Standard Relation") -> List[Tuple]:
    """
    讀取 SDF 檔，回傳 [(mol, pic50, smiles, relation), ...]。

    relation_field: SDF property 中存放 standard_relation 的欄位名稱。
        典型值：'=' / '>' / '<' / '>=' / '<='
        若欄位不存在，relation 預設為 '='（視為精確值）。
    """
    supplier = Chem.SDMolSupplier(path, removeHs=True, sanitize=True)
    records  = []
    failed   = 0
    for mol in supplier:
        if mol is None:
            failed += 1; continue
        props = mol.GetPropsAsDict()
        # 取活性值
        raw = props.get(label_field)
        if raw is None:
            failed += 1; continue
        if convert_ic50:
            pic50 = ic50_to_pic50(raw, unit)
        else:
            try:
                pic50 = float(raw)
            except (ValueError, TypeError):
                pic50 = None
        if pic50 is None:
            failed += 1; continue
        smi = Chem.MolToSmiles(mol)
        # 讀取 standard_relation（='/' >'/' <'，預設 '='）
        rel_raw  = str(props.get(relation_field, "=")).strip().strip("'").strip('"')
        relation = _normalize_relation(rel_raw)
        records.append((mol, pic50, smi, relation))
    print(f"  SDF 讀取：成功 {len(records)} 筆，失敗 {failed} 筆")
    _print_relation_summary(records)
    return records


def _detect_csv_dialect(path: str) -> "csv.Dialect":
    """
    自動偵測 CSV 分隔符號（逗號 / 分號 / Tab）。
    ChEMBL 匯出檔常用分號。
    """
    with open(path, newline="", encoding="utf-8-sig", errors="ignore") as f:
        sample = f.read(8192)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect
    except csv.Error:
        # Sniffer 失敗時手動計數
        first_line = sample.split("\n")[0]
        counts = {d: first_line.count(d) for d in [",", ";", "\t"]}
        best   = max(counts, key=counts.get)
        class _D(csv.Dialect):
            delimiter      = best
            quotechar      = '"'
            doublequote    = True
            skipinitialspace = True
            lineterminator = "\r\n"
            quoting        = csv.QUOTE_MINIMAL
        return _D()


def _strip_header_quotes(headers: list) -> dict:
    """
    去除欄位名稱的前後引號（ChEMBL 常見：'"smiles"' → 'smiles'）。
    回傳 {stripped_lower: original} 映射。
    """
    result = {}
    for h in headers:
        stripped = h.strip().strip('"').strip("'").strip()
        result[stripped.lower()] = h
    return result


def read_csv(path: str, smiles_col: str, label_col: str,
             convert_ic50: bool = False, unit: str = "nM") -> List[Tuple]:
    """
    讀取 CSV 檔，回傳 [(mol, pic50, smiles), ...]。

    自動處理：
      - 分號分隔（ChEMBL 匯出格式）/ 逗號 / Tab
      - 欄位名稱帶引號（如 "smiles"）
      - 大小寫不敏感欄位比對
    """
    records = []
    failed  = 0

    # ── 自動偵測分隔符 ────────────────────────────────────────────────────
    dialect = _detect_csv_dialect(path)
    print(f"  偵測到分隔符：{repr(dialect.delimiter)}")

    with open(path, newline="", encoding="utf-8-sig", errors="ignore") as f:
        reader  = csv.DictReader(f, dialect=dialect)
        headers = reader.fieldnames or []

        # ── 去除欄位名稱引號並建立映射 ──────────────────────────────────
        col_map = _strip_header_quotes(headers)

        # 也加入原始（含引號）的版本，避免某些邊界情況
        for h in headers:
            col_map[h.lower()] = h

        def _find_col(preferred: str, keywords: list) -> Optional[str]:
            """優先精確比對，其次模糊關鍵字比對。"""
            # 1. 精確比對（去引號後）
            key = preferred.strip().strip('"').strip("'").lower()
            if key in col_map:
                return col_map[key]
            # 2. 模糊比對
            for kw in keywords:
                for cl, orig in col_map.items():
                    if kw in cl:
                        return orig
            return None

        smi_key = _find_col(smiles_col, ["smiles", "smi", "smile", "canonical"])
        lbl_key = _find_col(label_col,  ["pchembl", "pic50", "pic50",
                                          "standard_value", "standard value",
                                          "ic50", "ki", "kd", "activity",
                                          "label", "value"])

        if not smi_key:
            print(f"  [錯誤] 找不到 SMILES 欄（嘗試：{smiles_col!r}）")
            print(f"  現有欄位（前 10 個）：")
            for h in list(col_map.values())[:10]:
                print(f"    {h!r}")
            return []
        if not lbl_key:
            print(f"  [錯誤] 找不到活性欄（嘗試：{label_col!r}）")
            print(f"  現有欄位（前 10 個）：")
            for h in list(col_map.values())[:10]:
                print(f"    {h!r}")
            return []

        # 嘗試偵測 standard_relation 欄位（ChEMBL 匯出常有此欄）
        rel_key = _find_col("Standard Relation",
                            ["standard_relation", "relation", "operator",
                             "standard relation"])
        print(f"  CSV 欄位映射：SMILES={smi_key!r}  活性={lbl_key!r}"              f"  關係={rel_key!r}")

        # ── 讀取資料列 ────────────────────────────────────────────────────
        for row in reader:
            smi_raw = str(row.get(smi_key, "")).strip().strip('"').strip("'")
            lbl_raw = str(row.get(lbl_key, "")).strip().strip('"').strip("'")
            if not smi_raw or not lbl_raw or smi_raw.lower() in ("", "nan", "none"):
                failed += 1; continue
            mol = Chem.MolFromSmiles(smi_raw)
            if mol is None:
                failed += 1; continue
            if convert_ic50:
                pic50 = ic50_to_pic50(lbl_raw, unit)
            else:
                try:
                    pic50 = float(lbl_raw)
                except (ValueError, TypeError):
                    pic50 = None
            if pic50 is None:
                failed += 1; continue
            smi = Chem.MolToSmiles(mol)
            # 讀取 standard_relation
            if rel_key:
                rel_raw = str(row.get(rel_key, "=")).strip().strip('"').strip("'")
            else:
                rel_raw = "="
            relation = _normalize_relation(rel_raw)
            records.append((mol, pic50, smi, relation))

    print(f"  CSV 讀取：成功 {len(records)} 筆，失敗 {failed} 筆")
    _print_relation_summary(records)
    return records


def read_smiles_file(path: str, label_col_idx: int = 1,
                     convert_ic50: bool = False, unit: str = "nM") -> List[Tuple]:
    """
    讀取純 SMILES 檔（每行：SMILES [活性值]）。
    """
    records = []
    failed  = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smi_raw = parts[0]
            mol = Chem.MolFromSmiles(smi_raw)
            if mol is None:
                failed += 1; continue
            pic50 = None
            if len(parts) > label_col_idx:
                raw = parts[label_col_idx]
                if convert_ic50:
                    pic50 = ic50_to_pic50(raw, unit)
                else:
                    try:
                        pic50 = float(raw)
                    except ValueError:
                        pass
            if pic50 is None:
                failed += 1; continue
            smi = Chem.MolToSmiles(mol)
            records.append((mol, pic50, smi))
    print(f"  SMILES 讀取：成功 {len(records)} 筆，失敗 {failed} 筆")
    return records



def _read_smiles_col(path: str, smiles_col: str) -> list:
    """
    用指定欄位名稱從 CSV 讀取 SMILES 列表。
    自動處理 BOM、分隔符偵測、引號欄位名稱。
    """
    import csv as _csv
    dialect = _detect_csv_dialect(path)
    smiles_list = []
    with open(path, newline="", encoding="utf-8-sig", errors="ignore") as f:
        reader = _csv.DictReader(f, dialect=dialect)
        # 標準化欄位名（去引號）
        if reader.fieldnames:
            clean_map = {h.strip().strip('"').strip("'"): h
                         for h in reader.fieldnames}
            # 嘗試多種大小寫匹配
            actual_col = (clean_map.get(smiles_col)
                          or clean_map.get(smiles_col.lower())
                          or clean_map.get(smiles_col.upper())
                          or smiles_col)
        else:
            actual_col = smiles_col
        for row in reader:
            val = row.get(actual_col, "").strip().strip('"').strip("'")
            if val and Chem.MolFromSmiles(val) is not None:
                smiles_list.append(val)
    print(f"  [讀取] 欄位={smiles_col!r}  有效 SMILES={len(smiles_list):,} 筆")
    return smiles_list


def _parse_smiles_file(path: str) -> list:
    """
    智慧型 SMILES 檔案解析器（支援 ChEMBL 原始 CSV）。

    自動處理：
      - 純 SMILES 檔（每行一個 SMILES）
      - ChEMBL 匯出的分號分隔 CSV（含 BOM，欄位名稱帶引號）
      - 逗號 / Tab / 分號 / 空白等各種分隔符
      - 標題行自動跳過
      - SMILES 欄位自動定位（不限定必須是第一欄）

    Returns:
        list of str — 有效 SMILES 字串列表
    """
    smiles_list   = []
    warned_header = False

    with open(path, newline="", encoding="utf-8-sig", errors="ignore") as f:
        raw_content = f.read()

    # ── 偵測分隔符 ────────────────────────────────────────────────────────
    first_line = raw_content.split("\n")[0].strip()
    candidates = [("\t", "Tab"), (",", "逗號"), (";", "分號"), (" ", "空白")]
    best_delim, best_count = "\t", 0
    for delim, _ in candidates:
        cnt = first_line.count(delim)
        if cnt > best_count:
            best_count, best_delim = cnt, delim
    if best_count == 0:
        best_delim = None   # 單欄模式

    smi_col_idx  = None
    is_first_row = True

    for line in raw_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if best_delim is None:
            smi = line
        else:
            # 切分並去除各欄的前後引號
            parts = [p.strip().strip('"').strip("'") for p in line.split(best_delim)]

            if smi_col_idx is None:
                # 試每一欄找 SMILES
                found = False
                for idx, part in enumerate(parts):
                    if part and Chem.MolFromSmiles(part) is not None:
                        smi_col_idx = idx
                        found = True
                        break
                if not found:
                    if is_first_row:
                        # 標題行，靜默跳過
                        is_first_row = False
                        continue
                    else:
                        is_first_row = False
                        continue
                smi = parts[smi_col_idx]
            else:
                smi = parts[smi_col_idx] if smi_col_idx < len(parts) else ""

        smi = smi.strip()
        if smi and Chem.MolFromSmiles(smi) is not None:
            smiles_list.append(smi)
        is_first_row = False

    print(f"  [解析] 分隔符={repr(best_delim) if best_delim else '無（單欄）'}"
          f"  SMILES 欄索引={smi_col_idx if smi_col_idx is not None else 0}"
          f"  有效={len(smiles_list)} 筆")
    return smiles_list


def _parse_smiles_or_csv(path: str,
                          smiles_col: str = "SMILES",
                          label_col:  str = "",
                          convert_ic50: bool = False,
                          unit: str = "nM") -> list:
    """
    通用讀取入口：根據副檔名選擇最適合的讀取器。

    適用於 VS 庫（無標籤）與外部驗證集（有標籤）兩種場景。

    Returns:
        若 label_col 為空：list of str（SMILES）
        若 label_col 非空：list of (mol, pic50, smiles)
    """
    ext = os.path.splitext(path)[1].lower()

    if label_col:
        # 需要活性值
        if ext == ".sdf":
            return read_sdf(path, label_col, convert_ic50, unit)
        else:
            return read_csv(path, smiles_col, label_col, convert_ic50, unit)
    else:
        # 只需要 SMILES
        if ext == ".sdf":
            supplier = Chem.SDMolSupplier(path, removeHs=True)
            smis = []
            for mol in supplier:
                if mol:
                    smis.append(Chem.MolToSmiles(mol))
            return smis
        else:
            # CSV / SMI / TXT 全部交給智慧解析器
            return _parse_smiles_file(path)

# =============================================================================
# 主清洗流程
# =============================================================================

def clean_and_prepare(
    records:             List[Tuple],
    output_dir:          str   = "prepared_ligands",
    workers:             int   = 8,
    mmff_variant:        str   = "MMFF94s",
    n_conformers:        int   = 1,
    pic50_min:           float = 3.0,
    pic50_max:           float = 12.0,
    dedup:               bool  = True,
    dedup_method:        str   = "best",
    # "best"    = 保留最高活性（原有行為）
    # "mean"    = 取 pIC50 平均值（平滑實驗誤差）
    # "median"  = 取 pIC50 中位數（對離群值更穩健）
    # "std_filter" = 標準差過濾（σ > std_threshold 直接剔除整組）
    dedup_std_threshold: float = 1.0,
    # std_filter 模式：pIC50 標準差超過此值的 SMILES 全部剔除
    # σ > 1.0 = 活性相差 10 倍以上，視為爭議性數據
    organic_only:        bool  = True,
    # ── Standard Relation 過濾（新增）────────────────────────────
    relation_method:     str   = "strict",
    # "strict"=嚴格過濾  "impute"=邊界修正  "weighting"=加權法  "none"=不過濾
    impute_lo_boundary:  float = 7.0,    # method=impute 時，< 截尾替換成此 pIC50
    impute_hi_boundary:  float = 4.92,   # method=impute 時，> 截尾替換成此 pIC50
    exact_weight:        float = 1.0,    # method=weighting 時，精確值的 sample_weight
    censored_weight:     float = 0.2,    # method=weighting 時，截尾值的 sample_weight
) -> dict:
    """
    完整清洗 + 3D 生成流程。

    Args:
        records            : [(mol, pic50, smiles, relation), ...]
                             relation 欄位由 read_csv/read_sdf 自動填入
        output_dir         : 輸出目錄
        workers            : 多線程數（3D 生成使用）
        mmff_variant       : MMFF94 / MMFF94s
        n_conformers       : 3D 生成構象數（1 = 快速）
        pic50_min/max      : 活性值合理範圍
        dedup              : 是否處理重複 SMILES
        dedup_method       : 重複值處理方式
            "best"         → 保留最高活性（原有行為，最快）
            "mean"         → pIC50 取平均（σ≤0.5 時推薦）
            "median"       → pIC50 取中位數（對離群值穩健）
            "std_filter"   → σ > dedup_std_threshold 整組剔除
        dedup_std_threshold: std_filter 的 σ 門檻（預設 1.0）
        organic_only       : 是否過濾非有機小分子
        relation_method    : Standard Relation 處理方式（見 apply_relation_filter）
        impute_lo_boundary : impute 方法中 < 截尾的替換 pIC50
        impute_hi_boundary : impute 方法中 > 截尾的替換 pIC50
        exact_weight       : weighting 方法中精確值的 sample_weight
        censored_weight    : weighting 方法中截尾值的 sample_weight

    Returns:
        dict 含統計資訊
    """
    os.makedirs(output_dir, exist_ok=True)
    stats = {
        "total_input":    len(records),
        "after_relation": 0,
        "after_organic":  0,
        "after_range":    0,
        "after_dedup":    0,
        "final_3d":       0,
        "relation_method": relation_method,
    }

    print(f"\n{'='*60}")
    print(f"  小分子資料清洗流程")
    print(f"  輸入：{len(records)} 筆")
    print(f"{'='*60}")

    # ── 步驟 0：Standard Relation 過濾（在所有其他過濾之前）──────────────
    _use_weighting = False
    if relation_method and relation_method != "none":
        records = apply_relation_filter(
            records,
            method            = relation_method,
            pic50_lo_boundary = impute_lo_boundary,
            pic50_hi_boundary = impute_hi_boundary,
            exact_weight      = exact_weight,
            censored_weight   = censored_weight,
        )
        _use_weighting = (relation_method == "weighting")
        print(f"  [Step 0] Relation 過濾後：{len(records)} 筆")
    else:
        print(f"  [Step 0] 跳過 Relation 過濾（relation_method='none'）")
    stats["after_relation"] = len(records)

    # ── 步驟 1：有機小分子過濾 ──────────────────────────────────────────
    if organic_only:
        records = [r for r in records if is_organic_small_molecule(r[0])]
        print(f"  [Step 1] 有機小分子過濾後：{len(records)} 筆")
    stats["after_organic"] = len(records)

    # ── 步驟 2：活性值範圍過濾 ──────────────────────────────────────────
    records = [r for r in records if pic50_min <= r[1] <= pic50_max]
    print(f"  [Step 2] 活性值範圍過濾（{pic50_min}–{pic50_max}）後：{len(records)} 筆")
    stats["after_range"] = len(records)

    # ── 步驟 3：重複 SMILES 處理（四種方法）───────────────────────────
    if dedup:
        import numpy as _np3

        # 先把所有 record 依標準化 SMILES 分組
        _groups: dict = {}
        for r in records:
            mol, pic50, smi = r[0], r[1], r[2]
            can = Chem.MolToSmiles(mol)   # 標準化 SMILES
            if can not in _groups:
                _groups[can] = []
            _groups[can].append(r)

        _singles   = {k: v for k, v in _groups.items() if len(v) == 1}
        _dupes     = {k: v for k, v in _groups.items() if len(v) > 1}
        print(f"  [Step 3] 重複 SMILES：{len(_dupes)} 組  (共 {sum(len(v) for v in _dupes.values())} 筆)")
        print(f"           唯一 SMILES：{len(_singles)} 組")
        print(f"           處理方法：{dedup_method}")

        if dedup_method == "best":
            # 原有行為：保留最高活性的那筆
            _merged = {}
            for can, recs in _groups.items():
                _merged[can] = max(recs, key=lambda r: r[1])
            records = list(_merged.values())
            print(f"  → 保留最高活性：{len(records)} 筆")

        elif dedup_method in ("mean", "median"):
            # 取平均或中位數
            _merged = []
            _skipped = 0
            for can, recs in _groups.items():
                vals = _np3.array([r[1] for r in recs])
                if dedup_method == "mean":
                    new_pic50 = float(vals.mean())
                else:
                    new_pic50 = float(_np3.median(vals))
                # 使用第一筆的 mol/smi/relation，pIC50 用計算值
                rep = recs[0]
                if len(rep) >= 5:
                    _merged.append((rep[0], new_pic50, rep[2], rep[3], rep[4]))
                elif len(rep) >= 4:
                    _merged.append((rep[0], new_pic50, rep[2], rep[3]))
                else:
                    _merged.append((rep[0], new_pic50, rep[2], "="))
            records = _merged
            _func_name = "平均值" if dedup_method == "mean" else "中位數"
            print(f"  → {_func_name} 合併：{len(records)} 筆")

        elif dedup_method == "std_filter":
            # 標準差過濾：σ > threshold 的 SMILES 整組剔除
            _merged = []
            _discarded_groups = 0
            _discarded_mols   = 0
            _stats_rows = []
            for can, recs in _groups.items():
                vals = _np3.array([r[1] for r in recs])
                if len(vals) == 1:
                    _merged.append(recs[0])
                    continue
                sigma = float(vals.std())
                if sigma > dedup_std_threshold:
                    _discarded_groups += 1
                    _discarded_mols   += len(recs)
                    _stats_rows.append((can[:30], len(recs),
                                        round(vals.mean(), 3),
                                        round(sigma, 3)))
                    continue
                # σ 合格 → 取平均
                new_pic50 = float(vals.mean())
                rep = recs[0]
                if len(rep) >= 5:
                    _merged.append((rep[0], new_pic50, rep[2], rep[3], rep[4]))
                elif len(rep) >= 4:
                    _merged.append((rep[0], new_pic50, rep[2], rep[3]))
                else:
                    _merged.append((rep[0], new_pic50, rep[2], "="))
            records = _merged
            print(f"  → 標準差過濾（σ>{dedup_std_threshold}）：")
            print(f"     保留 {len(records)} 筆，剔除 {_discarded_groups} 組"
                  f"（{_discarded_mols} 筆爭議性數據）")
            if _stats_rows:
                print(f"  → 前 5 個被剔除的 SMILES：")
                for _row in _stats_rows[:5]:
                    print(f"     SMILES={_row[0]}...  n={_row[1]}"
                          f"  mean={_row[2]}  σ={_row[3]}")
            stats["discarded_high_std"] = _discarded_mols
        else:
            # 未知方法，退回 best
            print(f"  ⚠ 未知的 dedup_method={dedup_method!r}，退回 best")
            _merged = {}
            for can, recs in _groups.items():
                _merged[can] = max(recs, key=lambda r: r[1])
            records = list(_merged.values())

        print(f"  [Step 3] 處理後：{len(records)} 筆")
    stats["after_dedup"] = len(records)

    # ── 步驟 4：標準化 ─────────────────────────────────────────────────
    print(f"  [Step 4] 分子標準化（去鹽、中性化）...")
    std_records = []
    for r in records:
        mol, pic50 = r[0], r[1]
        mol_std = standardize_mol(mol)
        if mol_std:
            std_smi = Chem.MolToSmiles(mol_std)
            # 保留 relation/weight（若有）
            if len(r) >= 5:
                std_records.append((mol_std, pic50, std_smi, r[3], r[4]))
            elif len(r) >= 4:
                std_records.append((mol_std, pic50, std_smi, r[3]))
            else:
                std_records.append((mol_std, pic50, std_smi, "="))
    records = std_records
    print(f"  → 標準化後：{len(records)} 筆")

    # ── 步驟 5：多進程 3D 構象生成（ProcessPool，真正繞過 GIL）────────
    print(f"  [Step 5] 3D 構象生成（workers={workers}，{mmff_variant}）...")
    print(f"           使用 ProcessPoolExecutor（CPU 密集任務，繞過 GIL）")

    # 先建立工作列表：保留 extra 欄位（relation, weight）
    _work_records = [(r[0], r[1], r[2], r[3] if len(r)>3 else "=",
                      r[4] if len(r)>4 else None) for r in records]
    final_records = [None] * len(_work_records)

    # ProcessPool 需要用頂層函式 _minimize_3d_worker（已定義在模組頂層）
    # 傳入 SMILES（字串）而非 mol 物件（mol 不可序列化）
    _proc_args = [
        (i, r[2], r[1], mmff_variant, n_conformers)
        for i, r in enumerate(_work_records)
    ]

    done = 0
    n_total = len(_proc_args)

    # workers=1 時退回 ThreadPool（避免 Windows spawn 開銷）
    _PoolCls = ProcessPoolExecutor if workers > 1 else ThreadPoolExecutor
    _pool_label = "ProcessPool" if workers > 1 else "ThreadPool(single)"
    print(f"           Pool 類型：{_pool_label}  workers={workers}")

    with _PoolCls(max_workers=workers) as ex:
        futs = {ex.submit(_minimize_3d_worker, arg): arg[0]
                for arg in _proc_args}
        for fut in as_completed(futs):
            try:
                res_idx, mol3d, pic50, smi = fut.result()
            except Exception:
                done += 1
                continue
            if mol3d is not None:
                _extra = _work_records[res_idx][3:]
                final_records[res_idx] = (mol3d, pic50, smi) + _extra
            done += 1
            if done % max(1, n_total // 20) == 0 or done == n_total:
                pct    = done / n_total * 100
                filled = int(20 * done / n_total)
                ok     = sum(1 for r in final_records if r)
                print(f"\r    [{'█'*filled}{'░'*(20-filled)}] {done}/{n_total}"
                      f" ({pct:.0f}%)  ✓{ok}", end="", flush=True)
    print()

    final_records = [r for r in final_records if r is not None]
    print(f"  → 3D 生成成功：{len(final_records)} / {len(records)} 筆")
    stats["final_3d"] = len(final_records)

    # ── 步驟 6：計算 ADMET 描述符 ────────────────────────────────────────
    print(f"  [Step 6] 計算 ADMET 描述符...")
    enriched = []
    for _rec in final_records:
        mol3d, pic50, smi = _rec[0], _rec[1], _rec[2]
        _rec_relation = _rec[3] if len(_rec) > 3 else "="
        _rec_weight   = _rec[4] if len(_rec) > 4 else None
        admet    = compute_admet(mol3d)
        scaffold = get_murcko_scaffold(mol3d)
        # 保留 relation 和 weight 進入 enriched
        enriched.append((mol3d, pic50, smi, admet, scaffold,
                          _rec_relation, _rec_weight))

    # ── 步驟 7：設定 SDF property 並輸出 ────────────────────────────────
    print(f"  [Step 7] 輸出 SDF 和 CSV...")
    sdf_path = os.path.join(output_dir, "cleaned_ligands.sdf")
    csv_path = os.path.join(output_dir, "cleaned_ligands.csv")

    writer = Chem.SDWriter(sdf_path)
    csv_rows = []

    for _er in enriched:
        mol3d, pic50, smi, admet, scaffold = _er[0], _er[1], _er[2], _er[3], _er[4]
        _e_relation = _er[5] if len(_er) > 5 else "="
        _e_weight   = _er[6] if len(_er) > 6 else None

        # SDF：設定 property 讓 GpuQsarEngine 直接讀取
        mol3d.SetProp("pIC50",             str(round(pic50, 4)))
        mol3d.SetProp("SMILES",            smi)
        mol3d.SetProp("scaffold",          scaffold)
        mol3d.SetProp("standard_relation", _e_relation)
        if _e_weight is not None:
            mol3d.SetProp("sample_weight", str(round(_e_weight, 4)))
        for k, v in admet.items():
            if v is not None:
                mol3d.SetProp(k, str(v))
        writer.write(mol3d)

        row = {"SMILES": smi, "pIC50": round(pic50, 4), "scaffold": scaffold,
               "standard_relation": _e_relation}
        if _e_weight is not None:
            row["sample_weight"] = round(_e_weight, 4)
        row.update({k: v for k, v in admet.items() if v is not None})
        csv_rows.append(row)

    writer.close()

    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(csv_rows)

    # ── 步驟 8：多樣性報告 ────────────────────────────────────────────────
    print(f"  [Step 8] 生成資料品質報告...")
    report_path = os.path.join(output_dir, "ligand_report.txt")

    pic50_vals = np.array([r[1] for r in final_records])
    mw_vals    = np.array([r[3].get("MW", 0) for r in enriched if r[3]])
    scaffolds  = [r[4] for r in enriched]
    n_unique_scaffolds = len(set(s for s in scaffolds if s))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  小分子資料品質報告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"[輸入]\n")
        f.write(f"  原始筆數              : {stats['total_input']}\n")
        f.write(f"  Relation 過濾後       : {stats.get('after_relation', stats['total_input'])}\n")
        f.write(f"  有機小分子過濾後      : {stats['after_organic']}\n")
        f.write(f"  活性值範圍過濾後      : {stats['after_range']}\n")
        f.write(f"  去重後                : {stats['after_dedup']}\n")
        f.write(f"  3D 生成成功           : {stats['final_3d']}\n")
        f.write(f"  Relation 過濾方法     : {stats.get('relation_method', 'none')}\n\n")
        f.write(f"[pIC50 分布]\n")
        f.write(f"  平均  : {pic50_vals.mean():.3f}\n")
        f.write(f"  標準差: {pic50_vals.std():.3f}\n")
        f.write(f"  最小  : {pic50_vals.min():.3f}\n")
        f.write(f"  最大  : {pic50_vals.max():.3f}\n")
        f.write(f"  中位數: {np.median(pic50_vals):.3f}\n\n")
        f.write(f"[分子量分布]\n")
        if len(mw_vals) > 0:
            f.write(f"  平均 MW: {mw_vals.mean():.1f} Da\n")
            f.write(f"  範圍   : {mw_vals.min():.1f} – {mw_vals.max():.1f} Da\n\n")
        f.write(f"[化學多樣性]\n")
        f.write(f"  唯一 Murcko Scaffold 數: {n_unique_scaffolds}\n")
        f.write(f"  Scaffold 覆蓋率        : {n_unique_scaffolds / max(1, len(final_records)):.1%}\n\n")
        f.write(f"[輸出]\n")
        f.write(f"  SDF: {sdf_path}\n")
        f.write(f"  CSV: {csv_path}\n")
        f.write(f"  報告: {report_path}\n")

    print(f"\n{'='*60}")
    print(f"  清洗完成！最終資料集：{stats['final_3d']} 筆")
    print(f"  SDF  → {sdf_path}")
    print(f"  CSV  → {csv_path}")
    print(f"  報告 → {report_path}")
    print(f"{'='*60}")
    print(f"\n[下一步] 把輸出的 SDF 直接餵給 GpuQsarEngine：")
    print(f"  python gpu_qsar_engine.py \\")
    print(f"    --input-mode sdf \\")
    print(f"    --sdf-path {sdf_path} \\")
    print(f"    --label-field pIC50 \\")
    print(f"    --no-convert")

    return stats


# =============================================================================
# CLI 入口
# =============================================================================

# =============================================================================
# 互動式輔助函式
# =============================================================================

def _iask(prompt: str, default=None, cast=str,
          choices: list = None, is_path: bool = False):
    """
    互動式單行輸入。直接 Enter 使用預設值；非互動環境直接回傳預設值。
    """
    import sys
    is_tty = sys.stdin.isatty()
    default_str = str(default) if default is not None else ""

    if choices:
        choice_str = "/".join(choices)
        hint = f"[{choice_str}]（預設 {default_str}）"
    else:
        hint = f"（預設 {default_str}，直接 Enter 接受）" if default_str else ""

    if not is_tty:
        return default

    while True:
        raw = input(f"  {prompt} {hint}: ").strip()
        if not raw:
            return default
        # 路徑：清理引號和空白
        if is_path:
            raw = raw.strip('"').strip("'")
            return raw
        # 選項驗證
        if choices and raw not in choices:
            print(f"  ✗ 請輸入以下之一：{choices}")
            continue
        # 型別轉換
        try:
            return cast(raw)
        except (ValueError, TypeError):
            print(f"  ✗ 格式錯誤，請重新輸入（預期 {cast.__name__}）")



# =============================================================================
# 功能 B：虛擬篩選庫準備（無需活性值）
# =============================================================================

def prepare_vs_library(
    input_path:   str,
    output_dir:   str  = "prepared_vs_library",
    workers:      int  = 8,
    mmff:         str  = "MMFF94s",
    n_conf:       int  = 1,
    dedup:        bool = True,
    organic_only: bool = True,
    smiles_col:   str  = "",    # 空白 = 互動詢問
    _preview:     dict = None,  # 已取得的預覽資訊（避免重複預覽）
) -> dict:
    """
    清洗虛擬篩選用的 SMILES 庫（無需活性值）。

    與主資料集清洗的差異：
      - 不需要活性標籤
      - 不做 pIC50 範圍過濾
      - 輸出 vs_library.smi（供 run_virtual_screening 直接使用）
      - 同時輸出 vs_library.csv（含 MW/QED/ADMET 初篩）

    Returns:
        dict 含統計資訊
    """
    import csv as _csv
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  虛擬篩選庫清洗")
    print(f"  輸入：{input_path}")
    print(f"{'='*60}")

    ext = os.path.splitext(input_path)[1].lower()

    # ── 若未傳入 smiles_col，進行互動預覽與欄位選擇 ──────────────────
    if not smiles_col and ext != ".sdf":
        prev = _preview if _preview else _rich_preview(input_path)
        col_info = _interactive_column_select(
            input_path, mode="vs", preview_info=prev)
        smiles_col = col_info.get("smiles_col", "")

    # ── 讀取 SMILES ───────────────────────────────────────────────────
    try:
        if ext == ".sdf":
            from rdkit.Chem import SDMolSupplier
            supplier   = SDMolSupplier(input_path, removeHs=True)
            raw_smiles = [Chem.MolToSmiles(mol) for mol in supplier if mol]
        elif smiles_col:
            # 用指定欄位讀取
            raw_smiles = _read_smiles_col(input_path, smiles_col)
        else:
            # 回退：自動解析器
            raw_smiles = _parse_smiles_file(input_path)
    except Exception as e:
        print(f"  [錯誤] 讀取失敗：{e}")
        return {"ok": False, "n": 0}

    if not raw_smiles:
        print("  [錯誤] 讀取後無有效 SMILES，請確認欄位名稱。")
        return {"ok": False, "n": 0}

    print(f"  原始 SMILES：{len(raw_smiles):,} 筆")

    # 轉換為 mol
    records = []
    for smi in raw_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            records.append((mol, 0.0, Chem.MolToSmiles(mol)))

    # 有機分子過濾
    if organic_only:
        records = [(mol, p, s) for mol, p, s in records
                   if is_organic_small_molecule(mol)]
        print(f"  有機分子過濾後：{len(records)} 筆")

    # 去重
    if dedup:
        seen, uniq = set(), []
        for mol, p, smi in records:
            can = Chem.MolToSmiles(mol)
            if can not in seen:
                seen.add(can)
                uniq.append((mol, p, smi))
        print(f"  去重後：{len(uniq)} 筆")
        records = uniq

    # 標準化
    records = [(standardize_mol(mol), p, s) for mol, p, s in records]
    records = [(mol, p, s) for mol, p, s in records if mol]

    # 計算快速 ADMET 描述符（用於提前過濾，含進度列）
    print(f"  計算 ADMET 描述符（{len(records):,} 筆）...")
    enriched = []
    _admet_total = len(records)
    for _admet_i, (mol, _, smi) in enumerate(records):
        admet = compute_admet(mol)
        enriched.append((mol, smi, admet))
        if (_admet_i + 1) % max(1, _admet_total // 20) == 0 or _admet_i == _admet_total - 1:
            _vs_progress_bar(_admet_i + 1, _admet_total, "ADMET")
    if _admet_total > 0:
        print()

    # 輸出 .smi 和 .csv
    smi_path = os.path.join(output_dir, "vs_library.smi")
    csv_path = os.path.join(output_dir, "vs_library.csv")

    with open(smi_path, "w", encoding="utf-8") as f:
        for _, smi, _ in enriched:
            f.write(smi + "\n")

    if enriched:
        fields = ["SMILES"] + list(enriched[0][2].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for _, smi, admet in enriched:
                row = {"SMILES": smi}
                row.update({k: v for k, v in admet.items() if v is not None})
                w.writerow(row)

    print(f"\n{'='*60}")
    print(f"  虛擬篩選庫完成！共 {len(enriched)} 筆")
    print(f"  .smi → {smi_path}  （直接餵給 VS / MPO-VS）")
    print(f"  .csv → {csv_path}  （含 ADMET 描述符，可提前過濾）")
    print(f"{'='*60}")
    return {"ok": True, "n": len(enriched), "smi_path": smi_path}


# =============================================================================
# 功能 C：外部驗證集準備（獨立來源，有活性值）
# =============================================================================

def prepare_ext_val_set(
    input_path:  str,
    smiles_col:  str   = "SMILES",
    label_col:   str   = "IC50",
    convert_ic50: bool = True,
    unit:        str   = "nM",
    output_dir:  str   = "prepared_ext_val",
    workers:     int   = 8,
    pic50_min:   float = 3.0,
    pic50_max:   float = 12.0,
) -> dict:
    """
    準備外部驗證集（供 run_external_validation_from_csv 使用）。

    外部驗證集的關鍵要求：
      - 必須與訓練集來源不同（e.g., ChEMBL vs BindingDB）
      - 必須有 SMILES 和活性值
      - 建議 50-500 筆，太少統計意義不足

    輸出：
      ext_val.csv  — 含 SMILES + pIC50 的清洗後驗證集
    """
    import csv as _csv
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  外部驗證集清洗")
    print(f"  輸入：{input_path}")
    print(f"{'='*60}")

    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".sdf":
        records = read_sdf(input_path, label_col, convert_ic50, unit)
    elif ext == ".csv":
        records = read_csv(input_path, smiles_col, label_col, convert_ic50, unit)
    elif ext in (".smi", ".txt"):
        records = read_smiles_file(input_path, 1, convert_ic50, unit)
    else:
        print(f"  [錯誤] 不支援格式：{ext}")
        return {"ok": False}

    if not records:
        print("  [錯誤] 讀取後無有效資料。")
        return {"ok": False}

    # 範圍過濾 + 有機分子
    records = [(mol, pic50, smi) for mol, pic50, smi in records
               if pic50_min <= pic50 <= pic50_max
               and is_organic_small_molecule(mol)]
    print(f"  過濾後：{len(records)} 筆")

    # 標準化 + ADMET
    out_path = os.path.join(output_dir, "ext_val.csv")
    fields   = ["SMILES", "pIC50", "MW", "cLogP", "HBA", "HBD",
                "TPSA", "QED", "Lipinski_RO5"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for mol, pic50, smi in records:
            mol_std = standardize_mol(mol)
            if not mol_std:
                continue
            admet = compute_admet(mol_std)
            row   = {"SMILES": Chem.MolToSmiles(mol_std),
                     "pIC50":  round(pic50, 4)}
            for k in ["MW","cLogP","HBA","HBD","TPSA","QED","Lipinski_RO5"]:
                row[k] = admet.get(k, "")
            w.writerow(row)

    print(f"\n  外部驗證集完成！{len(records)} 筆 → {out_path}")
    print(f"  （此檔案可直接輸入「外部驗證」功能）")
    return {"ok": True, "n": len(records), "csv_path": out_path}


# =============================================================================
# 功能 D：多任務輔助標籤準備（LogP / Solubility）
# =============================================================================

def prepare_mtl_labels(
    main_csv_path: str,
    smiles_col:    str = "SMILES",
    output_dir:    str = "prepared_mtl",
) -> dict:
    """
    為多任務學習（MTL）模式補充輔助標籤。

    GpuQsarEngine 的 MTL 模式同時預測：
      1. pIC50（主任務）
      2. LogP（疏水性，用 RDKit 計算）
      3. Solubility（水溶性，用 ESOL 模型估算）

    實際上 LogP 和 Solubility 可以從 SMILES 直接計算，
    此函式讀入主訓練 CSV，補充這兩個欄位後輸出。

    輸出：mtl_labels.csv（原始 CSV + LogP + ESOL_logS 欄位）
    """
    import csv as _csv
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  多任務輔助標籤計算...")

    rows_in, rows_out = [], []
    try:
        dialect = _detect_csv_dialect(main_csv_path)
        with open(main_csv_path, newline="", encoding="utf-8-sig") as f:
            reader = _csv.DictReader(f, dialect=dialect)
            headers = reader.fieldnames or []
            col_map = _strip_header_quotes(headers)
            smi_key = col_map.get(smiles_col.lower()) or                       next((v for k, v in col_map.items() if "smiles" in k), None)
            if not smi_key:
                print(f"  [錯誤] 找不到 SMILES 欄位")
                return {"ok": False}
            for row in reader:
                rows_in.append(row)
    except Exception as e:
        print(f"  [錯誤] 讀取 CSV 失敗：{e}")
        return {"ok": False}

    ok_count = 0
    for row in rows_in:
        smi = str(row.get(smi_key, "")).strip().strip('"')
        mol = Chem.MolFromSmiles(smi)
        if mol:
            logp = round(Chem.rdMolDescriptors.CalcCrippenDescriptors(mol)[0], 3)
            # ESOL 水溶性估算（Delaney 2004 簡化版）
            mw   = Descriptors.MolWt(mol)
            rb   = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
            ap   = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
            hbd  = Chem.rdMolDescriptors.CalcNumHBD(mol)
            esol = 0.16 - 0.63*logp - 0.0062*mw + 0.066*rb - 0.74*ap
            row["LogP"]     = logp
            row["ESOL_logS"]= round(esol, 3)
            ok_count += 1
        else:
            row["LogP"]      = ""
            row["ESOL_logS"] = ""
        rows_out.append(row)

    out_path = os.path.join(output_dir, "mtl_labels.csv")
    new_fields = list(rows_in[0].keys()) + ["LogP", "ESOL_logS"]
    # 去除重複欄位
    seen, new_fields = set(), [x for x in new_fields
                                if not (x in seen or seen.add(x))]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=new_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)

    print(f"  ✓ 計算 LogP + ESOL_logS：{ok_count}/{len(rows_out)} 筆")
    print(f"  輸出 → {out_path}")
    return {"ok": True, "n": ok_count, "csv_path": out_path}


# =============================================================================
# 功能 E：MPO 雷達圖對比分子清單準備
# =============================================================================

def prepare_radar_molecules(
    source:     str,     # "file" 或 "manual"
    input_path: str = "",
    output_dir: str = "prepared_radar",
) -> dict:
    """
    準備 MPO 雷達圖的對比分子清單（2–10 個分子）。

    支援兩種輸入方式：
      1. 從檔案讀取：.smi 或 .csv（每行 SMILES + 名稱）
      2. 互動輸入：逐行輸入 SMILES 和名稱

    輸出：
      radar_mols.smi  — 格式：SMILES<tab>名稱（可直接輸入雷達圖功能）
    """
    import csv as _csv
    os.makedirs(output_dir, exist_ok=True)

    mols = []

    if source == "file" and input_path and os.path.isfile(input_path):
        print(f"  從檔案讀取對比分子：{input_path}")
        # 智慧型讀取：自動偵測分隔符 + SMILES 欄位
        try:
            raw_smiles_r = _parse_smiles_file(input_path)
        except Exception as e:
            print(f"  [錯誤] 讀取失敗：{e}")
            raw_smiles_r = []
        # 嘗試同時讀取名稱（Tab 或空白分隔）
        try:
            with open(input_path, encoding="utf-8-sig", errors="ignore") as f:
                _lines_r = [ln.strip() for ln in f if ln.strip()]
            # 如有標題行，跳過（第一行所有欄都解析失敗）
            start_i = 0
            if _lines_r and Chem.MolFromSmiles(_lines_r[0].split("\t")[0].split(",")[0]) is None:
                start_i = 1
            for _ri, _rl in enumerate(_lines_r[start_i:], 1):
                # 優先 Tab 分隔，其次空白
                _rp = _rl.split("\t") if "\t" in _rl else _rl.split(None, 1)
                if len(_rp) >= 2:
                    _rsmi, _rname = _rp[0].strip(), _rp[1].strip().strip('"')
                else:
                    _rsmi, _rname = _rp[0].strip(), f"Mol-{_ri}"
                _rmol = Chem.MolFromSmiles(_rsmi)
                if _rmol:
                    mols.append((Chem.MolToSmiles(_rmol), _rname))
        except Exception:
            # fallback：用 _parse_smiles_file 的結果，自動編號
            for _ri, _rsmi in enumerate(raw_smiles_r, 1):
                mols.append((_rsmi, f"Mol-{_ri}"))
    else:
        # 互動輸入
        print("  請逐行輸入對比分子（格式：SMILES 名稱，輸入空行結束）：")
        print("  範例：Cc1ccc(Nc2ncnc3[nH]ccc23)cc1 Erlotinib")
        while len(mols) < 10:
            line = input(f"  [{len(mols)+1}] > ").strip()
            if not line:
                break
            parts = line.split(None, 1)
            smi   = parts[0]
            name  = parts[1] if len(parts) > 1 else f"Mol-{len(mols)+1}"
            mol   = Chem.MolFromSmiles(smi)
            if mol:
                mols.append((Chem.MolToSmiles(mol), name))
                print(f"       ✓ {name}")
            else:
                print(f"       ✗ SMILES 無效，請重新輸入")

    if len(mols) < 2:
        print("  [警告] 需要至少 2 個分子才能繪製雷達圖。")
        return {"ok": False, "mols": []}

    out_path = os.path.join(output_dir, "radar_mols.smi")
    with open(out_path, "w", encoding="utf-8") as f:
        for smi, name in mols:
            f.write(f"{smi}\t{name}\n")

    print(f"  ✓ {len(mols)} 個對比分子 → {out_path}")
    for smi, name in mols:
        print(f"     {name:20s} {smi[:50]}")
    return {"ok": True, "n": len(mols), "mols": mols, "path": out_path}

def _section_lig(title: str):
    print(f"\n  ╔{'═'*(len(title)+4)}╗")
    print(f"  ║  {title}  ║")
    print(f"  ╚{'═'*(len(title)+4)}╝")


def _rich_preview(path: str, n_rows: int = 5) -> dict:
    """
    增強版檔案預覽，回傳完整的欄位資訊供後續互動使用。

    Returns:
        dict with keys:
          ext, delimiter, headers (list), clean_headers (list),
          sample_rows (list of dicts), n_total_est (int),
          ok (bool)
    """
    import csv as _csv

    result = {"ok": False, "ext": "", "delimiter": "",
              "headers": [], "clean_headers": [], "sample_rows": [],
              "n_total_est": 0}

    ext = os.path.splitext(path)[1].lower()
    result["ext"] = ext

    if not os.path.isfile(path):
        return result

    size_kb = os.path.getsize(path) / 1024
    print(f"\n┌─ 檔案預覽：{os.path.basename(path)}")
    print(f"  │  路徑：{os.path.abspath(path)}")
    print(f"  │  大小：{size_kb:.1f} KB")

    if ext == ".sdf":
        from rdkit.Chem import SDMolSupplier
        sup   = SDMolSupplier(path, removeHs=True)
        total = 0
        valid = 0
        prop_names: list = []
        rows = []
        for i, mol in enumerate(sup):
            total += 1
            if mol is not None:
                valid += 1
                if not prop_names and mol.GetPropsAsDict():
                    prop_names = list(mol.GetPropsAsDict().keys())
                if len(rows) < n_rows:
                    rows.append({"SMILES": Chem.MolToSmiles(mol),
                                 **{k: str(v) for k, v in mol.GetPropsAsDict().items()}})
        print(f"  │  格式：SDF  分子總數：{total}  有效：{valid}")
        if prop_names:
            print(f"  │  Properties（共 {len(prop_names)} 個）：")
            for i in range(0, len(prop_names), 4):
                chunk = prop_names[i:i+4]
                print("  │    " + "  |  ".join(f"{j+i+1}.{x}" for j, x in enumerate(chunk)))
        print(f"  │  前 {min(len(rows), n_rows)} 筆：")
        for i, row in enumerate(rows[:n_rows]):
            smi = row.get("SMILES", "")[:50]
            print(f"  │    [{i+1}] {smi}")
        print("  └" + "─"*56)
        result.update({"ok": valid > 0, "n_total_est": total,
                       "headers": ["SMILES"] + prop_names,
                       "clean_headers": ["SMILES"] + prop_names,
                       "sample_rows": rows})
        return result

    # ── CSV / SMI / TXT ─────────────────────────────────────────────
    try:
        dialect = _detect_csv_dialect(path)
        delim_name = {",":"逗號", ";":"分號", "	":"Tab", "|":"Pipe"}.get(
            dialect.delimiter, repr(dialect.delimiter))
        result["delimiter"] = dialect.delimiter

        with open(path, newline="", encoding="utf-8-sig", errors="ignore") as f:
            reader  = _csv.DictReader(f, dialect=dialect)
            headers = list(reader.fieldnames or [])
            clean   = [h.strip().strip('"').strip("'") for h in headers]
            result["headers"]       = headers
            result["clean_headers"] = clean

            rows = []
            for row in reader:
                if len(rows) >= n_rows:
                    break
                rows.append({k.strip().strip('"').strip("'"): str(v).strip()
                              for k, v in row.items()})
            result["sample_rows"] = rows

        # 估算總行數
        with open(path, encoding="utf-8-sig", errors="ignore") as f:
            n_lines = sum(1 for _ in f)
        result["n_total_est"] = max(0, n_lines - 1)  # 減標題行

        has_header = bool(clean)
        print(f"  │  格式：CSV  分隔符：{delim_name}")
        print(f"  │  估計資料行數：{result['n_total_est']:,}")
        if has_header:
            print(f"  │  欄位（共 {len(clean)} 個）：")
            for i in range(0, len(clean), 3):
                chunk = clean[i:i+3]
                print("  │    " + "  |  ".join(f"{j+i+1:>2}. {x}" for j, x in enumerate(chunk)))
        print(f"  │  前 {len(rows)} 筆資料：")
        for i, row in enumerate(rows):
            items = [(k, v[:28]) for k, v in row.items() if v and v != "nan"]
            line  = "  ".join(f"{k}={v}" for k, v in items[:4])
            print(f"  │    [{i+1}] {line}")
        print("  └" + "─"*56)
        result["ok"] = True

    except Exception as e:
        print(f"  │  ⚠ 預覽失敗：{e}")
        print("  └" + "─"*56)

    return result


def _interactive_column_select(
    path: str,
    mode: str = "vs",
    preview_info: dict = None,
) -> dict:
    """
    互動式欄位選擇器（比照主程式 _preview_file 設計）。

    mode：
      'vs'      — 只需要 SMILES（虛擬篩選庫）
      'ext_val' — 需要 SMILES + 活性值（外部驗證集）
      'mtl'     — 需要 SMILES + 輔助標籤（MTL 資料集）
      'radar'   — 只需要 SMILES + 名稱

    Returns:
      dict with smiles_col (str), label_col (str or None),
           name_col (str or None), confirmed (bool)
    """
    result = {"smiles_col": "", "label_col": None,
              "name_col": None, "confirmed": False}

    if preview_info is None:
        preview_info = _rich_preview(path)
    if not preview_info["ok"]:
        return result

    ext = preview_info["ext"]

    # SDF：SMILES 和欄位已知，直接確認
    if ext == ".sdf":
        result["smiles_col"] = "SMILES"
        if mode == "ext_val":
            props = [h for h in preview_info["clean_headers"] if h != "SMILES"]
            if props:
                print(f"\n可用的活性值欄位：{props}")
                lbl = input(f"  請輸入活性值欄位名稱（Enter=預設 {props[0]}）: "
                            ).strip() or props[0]
            else:
                lbl = input("  請輸入活性值欄位名稱：").strip()
            result["label_col"] = lbl
        result["confirmed"] = True
        return result

    clean_headers = preview_info["clean_headers"]
    if not clean_headers:
        print("  ⚠ 無法取得欄位清單（可能是純 SMILES 檔）")
        result["smiles_col"] = ""
        result["confirmed"]  = True
        return result

    # ── 自動偵測 SMILES 欄位 ──────────────────────────────────────
    SMILES_KEYWORDS = {"smiles","canonical_smiles","smi","structure",
                       "canonical","molecule","mol_smiles","smil"}
    auto_smi = ""
    # 先用關鍵字比對
    for h in clean_headers:
        if h.lower().replace(" ","_") in SMILES_KEYWORDS:
            auto_smi = h; break
    # 再驗證：取第一筆資料試解析
    if not auto_smi and preview_info["sample_rows"]:
        row = preview_info["sample_rows"][0]
        for h in clean_headers:
            val = row.get(h, "").strip()
            if val and Chem.MolFromSmiles(val) is not None:
                auto_smi = h; break

    # ── 自動偵測活性值欄位 ────────────────────────────────────────
    LABEL_KEYWORDS = {"pic50","standard_value","pchembl_value","activity",
                      "value","ic50","ki","kd","ec50","pki","pkd","inhibition",
                      "potency","binding"}
    auto_lbl = ""
    for h in clean_headers:
        if h.lower().replace(" ","_").replace("-","_") in LABEL_KEYWORDS:
            auto_lbl = h; break

    # ── SMILES 欄位確認 ───────────────────────────────────────────
    print()
    print(f"  ┌─ 欄位設定 ({'VS 庫' if mode=='vs' else '外部驗證集' if mode=='ext_val' else '輔助標籤'})")

    if auto_smi:
        print(f"  │  自動偵測 SMILES 欄位：「{auto_smi}」")
        confirm = input(f"  │  使用此欄位？[y/n]（Enter=y）：").strip().lower() or "y"
        if confirm == "y":
            smi_col = auto_smi
        else:
            print(f"  │  可用欄位：")
            for i, h in enumerate(clean_headers):
                print(f"  │    {i+1:>3}. {h}")
            while True:
                raw = input(f"  │  請輸入 SMILES 欄位名稱或編號：").strip()
                if raw.isdigit() and 1 <= int(raw) <= len(clean_headers):
                    smi_col = clean_headers[int(raw)-1]; break
                elif raw in clean_headers:
                    smi_col = raw; break
                else:
                    print(f"  │  ✗ 找不到欄位「{raw}」，請重新輸入。")
    else:
        print(f"  │  ⚠ 無法自動偵測 SMILES 欄位，請手動指定：")
        print(f"  │  可用欄位：")
        for i, h in enumerate(clean_headers):
            print(f"  │    {i+1:>3}. {h}")
        while True:
            raw = input(f"  │  請輸入 SMILES 欄位名稱或編號：").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(clean_headers):
                smi_col = clean_headers[int(raw)-1]; break
            elif raw in clean_headers:
                smi_col = raw; break
            else:
                print(f"  │  ✗ 找不到欄位「{raw}」，請重新輸入。")

    result["smiles_col"] = smi_col

    # 顯示 SMILES 欄前幾筆確認
    if preview_info["sample_rows"]:
        print(f"  │  SMILES 欄預覽（前 3 筆）：")
        for i, row in enumerate(preview_info["sample_rows"][:3]):
            val = row.get(smi_col, "（空）")[:55]
            mol_ok = Chem.MolFromSmiles(val) is not None if val != "（空）" else False
            print(f"  │    [{i+1}] {val}  {'✓' if mol_ok else '✗'}")

    # ── 活性值欄位（ext_val 和 mtl 模式）────────────────────────
    if mode in ("ext_val", "mtl"):
        if auto_lbl:
            print(f"  │  自動偵測活性值欄位：「{auto_lbl}」")
            confirm = input(f"  │  使用此欄位？[y/n]（Enter=y）：").strip().lower() or "y"
            if confirm == "y":
                lbl_col = auto_lbl
            else:
                remaining = [h for h in clean_headers if h != smi_col]
                print(f"  │  可用欄位：")
                for i, h in enumerate(remaining):
                    print(f"  │    {i+1:>3}. {h}")
                while True:
                    raw = input(f"  │  請輸入活性值欄位名稱或編號：").strip()
                    if raw.isdigit() and 1 <= int(raw) <= len(remaining):
                        lbl_col = remaining[int(raw)-1]; break
                    elif raw in clean_headers:
                        lbl_col = raw; break
                    else:
                        print(f"  │  ✗ 找不到欄位「{raw}」，請重新輸入。")
        else:
            remaining = [h for h in clean_headers if h != smi_col]
            print(f"  │  請指定活性值欄位（可用：{remaining[:6]!r}{'...' if len(remaining)>6 else ''}）：")
            while True:
                raw = input(f"  │  活性值欄位名稱或編號：").strip()
                if raw.isdigit() and 1 <= int(raw) <= len(remaining):
                    lbl_col = remaining[int(raw)-1]; break
                elif raw in clean_headers:
                    lbl_col = raw; break
                else:
                    print(f"  │  ✗ 找不到欄位「{raw}」，請重新輸入。")
        result["label_col"] = lbl_col

        # 活性值預覽
        if preview_info["sample_rows"]:
            vals = [row.get(lbl_col, "") for row in preview_info["sample_rows"][:3]]
            print(f"  │  活性值欄預覽：{vals}")

    # ── 名稱/ID 欄位（radar 模式）─────────────────────────────────
    if mode == "radar":
        name_candidates = [h for h in clean_headers if h != smi_col
                           if any(k in h.lower() for k in ["name","id","compound","cmpd","title"])]
        if name_candidates:
            print(f"  │  偵測到名稱欄位：「{name_candidates[0]}」")
            confirm = input(f"  │  使用此欄位作為化合物名稱？[y/n]（Enter=y）：").strip().lower() or "y"
            result["name_col"] = name_candidates[0] if confirm == "y" else None
        else:
            print(f"  │  未偵測到名稱欄位（可選）")
            raw = input(f"  │  請輸入名稱欄位（Enter=略過）：").strip()
            result["name_col"] = raw if raw in clean_headers else None

    print("  └" + "─"*56)
    result["confirmed"] = True
    return result


def _vs_progress_bar(current: int, total: int, stage: str = "") -> None:
    """統一的進度列印函式。"""
    if total <= 0:
        return
    pct    = current / total * 100
    filled = int(30 * current / total)
    bar    = "█" * filled + "░" * (30 - filled)
    print(f"  [{bar}] {current:>6,}/{total:,}  ({pct:5.1f}%)  {stage}",
          end="", flush=True)
    if current >= total:
        print()


def _preview_file(path: str, n: int = 5):
    """
    印出檔案前 N 行預覽，並自動偵測分隔符 + 列出欄位清單。
    協助使用者確認欄位名稱。
    """
    import csv as _csv
    ext = os.path.splitext(path)[1].lower()
    print(f"  ── 檔案預覽（前 {n} 筆）──")
    try:
        if ext == ".sdf":
            from rdkit.Chem import SDMolSupplier
            sup = SDMolSupplier(path, removeHs=True)
            for i, mol in enumerate(sup):
                if i >= n: break
                if mol is None:
                    print(f"    [{i+1}] (解析失敗)")
                    continue
                smi   = Chem.MolToSmiles(mol)
                props = mol.GetPropsAsDict()
                print(f"    [{i+1}] {smi[:60]}")
                for k, v in list(props.items())[:5]:
                    print(f"         {k} = {v}")
        else:
            # 自動偵測分隔符
            dialect = _detect_csv_dialect(path)
            print(f"  偵測到分隔符：{repr(dialect.delimiter)}")
            with open(path, newline="", encoding="utf-8-sig",
                      errors="ignore") as f:
                reader  = _csv.DictReader(f, dialect=dialect)
                headers = reader.fieldnames or []
                # 去除欄位名引號後顯示
                clean_headers = [h.strip().strip('"').strip("'")
                                 for h in headers]
                print(f"  欄位（共 {len(clean_headers)} 個）：")
                # 每行顯示 4 個欄位
                for i in range(0, len(clean_headers), 4):
                    chunk = clean_headers[i:i+4]
                    print("    " + "  |  ".join(f"{j+i+1}.{x}" for j, x in enumerate(chunk)))
                print()
                print(f"  前 {n} 筆資料：")
                for i, row in enumerate(reader):
                    if i >= n: break
                    # 印出前 4 個非空欄位
                    items = [(k.strip().strip('"'), str(v).strip()[:30])
                             for k, v in row.items() if str(v).strip()]
                    print(f"    [{i+1}] " + "  ".join(f"{k}={v}" for k,v in items[:4]))
    except Exception as e:
        print(f"  (預覽失敗：{e})")


# =============================================================================
# 互動式主程式
# =============================================================================

def interactive_main(args):
    """
    互動式輸入流程。
    若 args 中已有 CLI 旗標，對應步驟跳過問答直接使用。
    """
    import os, sys

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║       小分子資料預處理 — 互動式設定                       ║")
    print("║  直接 Enter 套用預設值；輸入新值後按 Enter 確認           ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ════════════════════════════════════════════════════════════════
    # 步驟 0：選擇準備資料的用途
    # ════════════════════════════════════════════════════════════════
    _section_lig("步驟 0：選擇資料用途（可多選，空格分隔）")
    print("  A. 主訓練資料集（必選，SMILES + 活性值）")
    print("  B. 虛擬篩選庫（純 SMILES，無活性值）")
    print("  C. 外部驗證集（獨立來源，SMILES + 活性值）")
    print("  D. 多任務輔助標籤（為主資料集補充 LogP/Solubility）")
    print("  E. MPO 雷達圖對比分子（2-10 個已知化合物）")
    print()
    print("  提示：若只要準備主訓練資料，直接 Enter（預設選 A）")

    _mode_raw  = _iask("用途選項", default="A").upper().split()
    _modes     = set(_mode_raw) if _mode_raw else {"A"}
    if not _modes:
        _modes = {"A"}

    print(f"  → 已選：{' '.join(sorted(_modes))}")

    # ── 若只選 B/C/D/E，詢問後直接跳到對應流程 ──────────────────────
    _non_training = _modes - {"A"}
    _do_training  = "A" in _modes

    # ═══════════════════════════════════════════════
    # 模式 B：虛擬篩選庫
    # ═══════════════════════════════════════════════
    if "B" in _modes:
        _section_lig("虛擬篩選庫準備（模式 B）")
        print("  支援格式：ChEMBL CSV（分號/逗號分隔）/ 純 .smi / SDF")

        # ── 路徑輸入 ──────────────────────────────────────────────
        while True:
            _vs_path = _iask("VS 庫輸入檔案路徑（.smi / .csv / .sdf）",
                             default="", is_path=True)
            if not _vs_path:
                print("  ✗ 路徑不可為空。")
                continue
            if not os.path.isfile(_vs_path):
                print(f"  ✗ 找不到檔案：{_vs_path!r}")
                print("    提示：Windows 路徑請用正斜線，或直接拖拉檔案到終端")
                continue
            break

        # ── 預覽檔案內容 + 互動欄位選擇 ─────────────────────────
        _vs_prev = _rich_preview(_vs_path)
        _vs_col_info = _interactive_column_select(
            _vs_path, mode="vs", preview_info=_vs_prev)
        _vs_smi_col = _vs_col_info.get("smiles_col", "")

        if not _vs_col_info["confirmed"]:
            print("  ✗ 欄位選擇失敗，略過模式 B。")
        else:
            # ── 處理設定 ─────────────────────────────────────────
            _vs_out     = _iask("VS 庫輸出目錄", default="prepared_vs_library",
                                is_path=True)
            _vs_dedup   = _iask("是否去重（相同 SMILES 只保留一筆）",
                                default="y", choices=["y","n"]) == "y"
            _vs_organic = _iask("是否過濾非有機/非藥用分子",
                                default="y", choices=["y","n"]) == "y"
            cpu_count   = os.cpu_count() or 4
            _vs_workers = _iask(f"多線程數（0=自動={min(int(cpu_count*0.9),32)}）",
                                default=0, cast=int)
            if _vs_workers == 0:
                _vs_workers = min(int(cpu_count * 0.9), 32)

            # ── 預估規模警告 ─────────────────────────────────────
            _est_n = _vs_prev.get("n_total_est", 0)
            if _est_n > 50000:
                print(f"\n⚠ 資料量較大（估計 {_est_n:,} 筆），ADMET 計算可能需要數分鐘。")
                _cont = input("  繼續執行？[y/n]（Enter=y）：").strip().lower() or "y"
                if _cont != "y":
                    print("  → 模式 B 已略過。")
                else:
                    prepare_vs_library(
                        input_path   = _vs_path,
                        output_dir   = _vs_out,
                        workers      = _vs_workers,
                        dedup        = _vs_dedup,
                        organic_only = _vs_organic,
                        smiles_col   = _vs_smi_col,
                        _preview     = _vs_prev,
                    )
            else:
                prepare_vs_library(
                    input_path   = _vs_path,
                    output_dir   = _vs_out,
                    workers      = _vs_workers,
                    dedup        = _vs_dedup,
                    organic_only = _vs_organic,
                    smiles_col   = _vs_smi_col,
                    _preview     = _vs_prev,
                )

    # ═══════════════════════════════════════════════
    # 模式 C：外部驗證集
    # ═══════════════════════════════════════════════
    if "C" in _modes:
        _section_lig("外部驗證集準備（模式 C）")
        print("  提示：外部驗證集應與主訓練集來源不同（如不同資料庫或論文）")

        # ── 路徑輸入 ──────────────────────────────────────────────
        while True:
            _ext_path = _iask("外部驗證集路徑（.csv / .sdf / .smi）",
                              default="", is_path=True)
            if not _ext_path:
                print("  ✗ 路徑不可為空。"); continue
            if not os.path.isfile(_ext_path):
                print(f"  ✗ 找不到檔案：{_ext_path!r}"); continue
            break

        # ── 預覽 + 互動欄位選擇 ───────────────────────────────────
        _ext_prev = _rich_preview(_ext_path)
        _ext_col_info = _interactive_column_select(
            _ext_path, mode="ext_val", preview_info=_ext_prev)
        _ext_smi_col = _ext_col_info.get("smiles_col", "SMILES")
        _ext_lbl_col = (_ext_col_info.get("label_col") or
                        _iask("活性標籤欄位名稱", default="Standard Value"))

        _ext_conv = _iask("活性值是否需要 IC50→pIC50 換算",
                          default="y", choices=["y","n"]) == "y"
        _ext_unit = "nM"
        if _ext_conv:
            _ext_unit = _iask("IC50 單位", default="nM",
                              choices=["nM","uM","mM","M"])
        _ext_out = _iask("輸出目錄", default="prepared_ext_val", is_path=True)
        prepare_ext_val_set(
            input_path   = _ext_path,
            smiles_col   = _ext_smi_col,
            label_col    = _ext_lbl_col,
            convert_ic50 = _ext_conv,
            unit         = _ext_unit,
            output_dir   = _ext_out,
        )

    # ═══════════════════════════════════════════════
    # 模式 E：MPO 雷達圖對比分子
    # ═══════════════════════════════════════════════
    if "E" in _modes:
        _section_lig("MPO 雷達圖對比分子準備（模式 E）")
        print("  可從檔案讀取，或互動輸入（每行：SMILES 化合物名稱）")
        _radar_src = _iask("輸入方式", default="manual",
                           choices=["file","manual"])
        _radar_file = ""
        if _radar_src == "file":
            _radar_file = _iask("對比分子檔案路徑（.smi）",
                                default="", is_path=True)
        _radar_out = _iask("輸出目錄", default="prepared_radar", is_path=True)
        prepare_radar_molecules(
            source      = _radar_src,
            input_path  = _radar_file,
            output_dir  = _radar_out,
        )

    # ── 若沒有選 A，完成退出 ─────────────────────────────────────────
    if not _do_training:
        print("\n[完成] 資料準備完畢，未選擇主訓練模式（A），結束。")
        sys.exit(0)

    # ─────────────────────────────────────────────────────────────────
    # 以下為模式 A：主訓練資料集清洗（原有流程）
    # ─────────────────────────────────────────────────────────────────

    # ── 步驟 1：輸入檔案 ─────────────────────────────────────────────────
    _section_lig("步驟 1／6：輸入檔案")
    if args.input:
        input_path = args.input
        print(f"  [CLI] 輸入檔案：{input_path}")
    else:
        while True:
            input_path = _iask(
                "請輸入資料檔案路徑（.sdf / .csv / .smi）",
                default="", is_path=True
            )
            if input_path and os.path.isfile(input_path):
                break
            print(f"  ✗ 找不到檔案：{input_path!r}，請重新輸入。")

    ext = os.path.splitext(input_path)[1].lower()
    print(f"  偵測到格式：{ext.upper()}")

    # 預覽檔案幫助使用者確認欄位（使用增強版預覽）
    _rich_preview(input_path)

    # ── 步驟 2：欄位與活性值設定 ─────────────────────────────────────────
    _section_lig("步驟 2／6：活性標籤欄位設定")

    label_field = args.label_field if args.label_field != "IC50" else None
    smiles_col  = args.smiles_col  if args.smiles_col  != "SMILES" else None

    if ext == ".csv":
        # CSV：需要確認 SMILES 欄和標籤欄
        print("  提示：ChEMBL 匯出的 CSV 常用欄位名稱：")
        print("    SMILES 欄：smiles")
        print("    活性欄  ：Standard Value（IC50原始值）或 pChEMBL Value（已換算 pIC50）")
        smiles_col  = smiles_col or _iask(
            "CSV 中 SMILES 欄位名稱", default="smiles")
        label_field = label_field or _iask(
            "CSV 中活性標籤欄位名稱", default="Standard Value")
    elif ext == ".sdf":
        label_field = label_field or _iask(
            "SDF property 中活性標籤欄位名稱（輸入 ? 列出所有 property）",
            default="IC50")
        # 若使用者輸入 ?，列出 SDF 的 property 名稱
        if label_field == "?":
            try:
                from rdkit.Chem import SDMolSupplier
                sup = SDMolSupplier(input_path)
                mol = next(m for m in sup if m is not None)
                props = list(mol.GetPropsAsDict().keys())
                print(f"  SDF property 清單：{props}")
            except Exception:
                pass
            label_field = _iask("活性標籤欄位名稱", default="IC50")

    smiles_col  = smiles_col  or "SMILES"
    label_field = label_field or "IC50"

    # IC50 換算設定
    if args.no_convert:
        convert = False
        unit    = "nM"
        print("  [CLI] 活性值直接視為 pIC50，不換算。")
    else:
        # 若欄位名稱含 pchembl / pic50，預設不換算
        _lf_lower = (label_field or "").lower().replace(" ", "")
        _auto_no_convert = any(x in _lf_lower for x in
                               ["pchembl", "pic50", "pic50"])
        if _auto_no_convert:
            print(f"  → 偵測到 pIC50 欄位（{label_field!r}），自動設定為不換算。")
            convert = False
            unit    = "nM"
        else:
            convert_raw = _iask(
                "活性欄位是否為 IC50（需換算為 pIC50）？",
                default="y", choices=["y", "n"])
            convert = convert_raw.lower() == "y"
            unit = "nM"
            if convert:
                unit = _iask(
                    "IC50 濃度單位",
                    default="nM", choices=["nM", "uM", "mM", "M"])

    # ── 步驟 3：資料品質過濾設定 ─────────────────────────────────────────
    _section_lig("步驟 3／6：資料品質過濾")

    pic50_min    = _iask("pIC50 最小值過濾（低於此值視為異常）",
                         default=3.0, cast=float)
    pic50_max    = _iask("pIC50 最大值過濾（高於此值視為異常）",
                         default=12.0, cast=float)
    dedup_raw    = _iask("是否處理重複 SMILES",
                         default="y", choices=["y", "n"])
    dedup        = dedup_raw.lower() == "y"
    dedup_method = "best"
    dedup_std_threshold = 1.0
    if dedup:
        print()
        print("  重複 SMILES 的處理方法：")
        print("  [1] best        — 保留最高活性（最快，適合乾淨資料集）")
        print("  [2] mean        — pIC50 取平均（σ≤0.5 時推薦，平滑實驗誤差）")
        print("  [3] median      — pIC50 取中位數（對少數離群值穩健）")
        print("  [4] std_filter  — 標準差過濾（σ>1.0 整組剔除，最嚴格）")
        _dm_raw = _iask("選擇方法 [1/2/3/4]", default="1",
                        choices=["1","2","3","4"])
        _dm_map = {"1":"best","2":"mean","3":"median","4":"std_filter"}
        dedup_method = _dm_map.get(_dm_raw, "best")
        if dedup_method == "std_filter":
            print()
            print("  σ > threshold：代表同一分子不同實驗 pIC50 相差超過 threshold 個單位")
            print("  例：threshold=1.0 → 活性差 10 倍以上 → 視為爭議性數據，整組剔除")
            dedup_std_threshold = _iask(
                "  σ 閾值（Enter=1.0）", default=1.0, cast=float)
        print(f"  → 已選：{dedup_method}"
              + (f"  σ 閾值={dedup_std_threshold}" if dedup_method=="std_filter" else ""))
    organic_raw  = _iask("是否過濾非有機小分子（排除無機鹽/金屬配合物）",
                         default="y", choices=["y", "n"])
    organic_only = organic_raw.lower() == "y"

    # ── Standard Relation 過濾（standard_relation 欄位處理）───────────
    _section_lig("步驟 3.5／6：Standard Relation 過濾（截尾數據處理）")
    print("  說明：ChEMBL 資料含 '=' / '>' / '<' 三種 standard_relation。")
    print("    > 例：< 200 nM（強效但未精確量測）/ > 6000 nM（無效）")
    print()
    print("  [1] strict    — 嚴格過濾（僅保留 =，刪除 > 和 <）")
    print("                  → 最高 R²；資料量足（≥1000筆）時強烈推薦")
    print("  [2] impute    — 邊界修正（< 替換為偏保守值，> 替換為偏保守值）")
    print("                  → 資料量不足時的備用方案（可能在邊界點出現垂直點陣）")
    print("  [3] weighting — 加權法（保留全部，截尾值 weight=0.2）")
    print("                  → 最科學，需 gpu_qsar_engine 的加權損失函數支援")
    print("  [4] none      — 不過濾（直接使用所有資料）")
    print()

    _rel_choice = _iask("選擇方法 [1/2/3/4]", default="1",
                        choices=["1","2","3","4"])
    _rel_method_map = {"1":"strict", "2":"impute", "3":"weighting", "4":"none"}
    relation_method = _rel_method_map.get(_rel_choice, "strict")
    print(f"  → 已選：{relation_method}")

    # 方法 2：邊界修正的參數
    impute_lo_boundary = 7.0
    impute_hi_boundary = 4.92
    if relation_method == "impute":
        print()
        print("  邊界修正設定：")
        print("    '<' 截尾（如 < 200nM = pIC50 > 7.0）→ 替換為此 pIC50 值")
        print("    '>'截尾（如 > 6000nM = pIC50 < 4.92）→ 替換為此 pIC50 值")
        impute_lo_boundary = _iask(
            "  '<' 截尾替換 pIC50（Enter=7.0）", default=7.0, cast=float)
        impute_hi_boundary = _iask(
            "  '>' 截尾替換 pIC50（Enter=4.92）", default=4.92, cast=float)
        print(f"  → < 截尾→{impute_lo_boundary}  > 截尾→{impute_hi_boundary}")

    # 方法 3：加權法的參數
    exact_weight    = 1.0
    censored_weight = 0.2
    if relation_method == "weighting":
        print()
        exact_weight    = _iask("  精確值 sample_weight（Enter=1.0）",
                                 default=1.0, cast=float)
        censored_weight = _iask("  截尾值 sample_weight（Enter=0.2）",
                                 default=0.2, cast=float)
        print(f"  → 精確值 weight={exact_weight}  截尾值 weight={censored_weight}")
        print("  ✓ 輸出 CSV 將包含 sample_weight 欄位")
        print("    在 gpu_qsar_engine.py 中請啟用 use_weighted_loss=True")

    # ── 步驟 4：3D 生成設定 ───────────────────────────────────────────────
    _section_lig("步驟 4／6：3D 構象生成")
    import os as _os
    cpu_count   = _os.cpu_count() or 4
    auto_workers= min(int(cpu_count * 0.9), 32)
    print(f"  偵測到 {cpu_count} 個邏輯 CPU，建議 workers={auto_workers}")

    workers     = _iask(f"多線程數（0=自動={auto_workers}）",
                        default=0, cast=int)
    if workers == 0:
        workers = auto_workers
    mmff        = _iask("MMFF 力場變體",
                        default="MMFF94s", choices=["MMFF94", "MMFF94s"])
    n_conf      = _iask("每個分子生成幾個構象取最低能量（1=快速, 3-5=更準確）",
                        default=1, cast=int)

    # ── 步驟 5：輸出設定 ─────────────────────────────────────────────────
    _section_lig("步驟 5／6：輸出目錄")
    output_dir  = _iask("輸出目錄路徑",
                        default="prepared_ligands", is_path=True)

    # ── 步驟 6：確認摘要 ─────────────────────────────────────────────────
    _section_lig("步驟 6／6：設定確認")
    print(f"  輸入檔案   : {input_path}")
    print(f"  格式       : {ext.upper()}")
    if ext == ".csv":
        print(f"  SMILES 欄  : {smiles_col}")
    print(f"  活性欄位   : {label_field}")
    print(f"  IC50 換算  : {'是，單位=' + unit if convert else '否（直接使用 pIC50）'}")
    print(f"  pIC50 範圍 : {pic50_min} – {pic50_max}")
    _dm_labels = {"best":"保留最高活性","mean":"平均值","median":"中位數",
                  "std_filter":f"標準差過濾（σ>{dedup_std_threshold}）"}
    print(f"  去重       : {'是，方法='+_dm_labels.get(dedup_method,dedup_method) if dedup else '否'}")
    print(f"  有機過濾   : {'是' if organic_only else '否'}")
    _rel_display = {
        "strict":    "嚴格過濾（僅保留 = 精確值）",
        "impute":    f"邊界修正（< →{impute_lo_boundary}  > →{impute_hi_boundary}）",
        "weighting": f"加權法（= weight={exact_weight}  >/< weight={censored_weight}）",
        "none":      "不過濾",
    }.get(relation_method, relation_method)
    print(f"  Relation   : {_rel_display}")
    print(f"  3D workers : {workers}")
    print(f"  MMFF 變體  : {mmff}")
    print(f"  構象數     : {n_conf}")
    print(f"  輸出目錄   : {output_dir}")

    confirm = _iask("\n以上設定確認，開始處理？",
                    default="y", choices=["y", "n"])
    if confirm.lower() != "y":
        print("  已取消。")
        sys.exit(0)

    # ── 模式 D：多任務輔助標籤（在主資料集清洗後執行）──────────────────
    _do_mtl_labels = "D" in _modes

    # ── 讀取資料 ─────────────────────────────────────────────────────────
    print(f"\n[讀取] {input_path}...")
    if ext == ".sdf":
        records = read_sdf(input_path, label_field, convert, unit)
    elif ext == ".csv":
        records = read_csv(input_path, smiles_col, label_field, convert, unit)
    elif ext in (".smi", ".txt"):
        records = read_smiles_file(input_path, 1, convert, unit)
    else:
        print(f"  ✗ 不支援的格式：{ext}")
        sys.exit(1)

    if not records:
        print("  ✗ 讀取後沒有有效資料，請確認欄位名稱。")
        sys.exit(1)

    # ── 執行清洗 ─────────────────────────────────────────────────────────
    clean_and_prepare(
        records,
        output_dir          = output_dir,
        workers             = workers,
        mmff_variant        = mmff,
        n_conformers        = n_conf,
        pic50_min           = pic50_min,
        pic50_max           = pic50_max,
        dedup               = dedup,
        dedup_method        = dedup_method,
        dedup_std_threshold = dedup_std_threshold,
        organic_only        = organic_only,
        relation_method     = relation_method,
        impute_lo_boundary  = impute_lo_boundary,
        impute_hi_boundary  = impute_hi_boundary,
        exact_weight        = exact_weight,
        censored_weight     = censored_weight,
    )

    # ── 模式 D：多任務輔助標籤計算 ────────────────────────────────────
    if _do_mtl_labels:
        _main_csv = os.path.join(output_dir, "cleaned_ligands.csv")
        if os.path.isfile(_main_csv):
            _section_lig("多任務輔助標籤計算（模式 D）")
            prepare_mtl_labels(
                main_csv_path = _main_csv,
                smiles_col    = "SMILES",
                output_dir    = output_dir,
            )
        else:
            print("  [MTL] 找不到主資料集 CSV，跳過。")

    print("\n[完成] 所有資料準備完畢。")
    print("=" * 58)
    print("  建議的下一步：")
    print("  1. 執行 gpu_qsar_engine.py 開始訓練")
    print(f"  2. 訓練資料 → {output_dir}/cleaned_ligands.sdf")
    if "B" in _modes:
        print("  3. 虛擬篩選 → prepared_vs_library/vs_library.smi")
    if "C" in _modes:
        print("  4. 外部驗證 → prepared_ext_val/ext_val.csv")
    if "E" in _modes:
        print("  5. MPO 雷達圖 → prepared_radar/radar_mols.smi")
    print("=" * 58)


def main():
    parser = argparse.ArgumentParser(
        description="小分子資料預處理腳本（GpuQsarEngine 前置處理）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="不帶任何參數直接執行進入互動模式；帶 CLI 旗標則跳過對應問答。",
    )
    parser.add_argument("--input",          default="",
                        help="輸入檔案路徑（.sdf / .csv / .smi）")
    parser.add_argument("--label-field",    default="IC50",
                        help="活性標籤欄位名稱")
    parser.add_argument("--smiles-col",     default="SMILES",
                        help="CSV 中 SMILES 欄名")
    parser.add_argument("--unit",           default="nM",
                        choices=["nM", "uM", "mM", "M"],
                        help="IC50 單位（預設 nM）")
    parser.add_argument("--no-convert",     action="store_true",
                        help="活性值直接視為 pIC50，不做換算")
    parser.add_argument("--output-dir",     default="",
                        help="輸出目錄（預設互動詢問）")
    parser.add_argument("--workers",        default=0, type=int,
                        help="多線程數（0=自動偵測）")
    parser.add_argument("--mmff",           default="MMFF94s",
                        choices=["MMFF94", "MMFF94s"],
                        help="MMFF 力場變體（預設 MMFF94s）")
    parser.add_argument("--n-conformers",   default=1, type=int,
                        help="每個分子構象數（預設 1）")
    parser.add_argument("--pic50-min",      default=3.0, type=float,
                        help="pIC50 最小值（預設 3.0）")
    parser.add_argument("--pic50-max",      default=12.0, type=float,
                        help="pIC50 最大值（預設 12.0）")
    parser.add_argument("--no-dedup",       action="store_true",
                        help="不去除重複 SMILES")
    parser.add_argument("--keep-inorganic", action="store_true",
                        help="保留無機/金屬配合物")
    args = parser.parse_args()

    interactive_main(args)


if __name__ == "__main__":
    main()
