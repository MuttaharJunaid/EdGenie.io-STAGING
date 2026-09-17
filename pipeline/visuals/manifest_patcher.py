"""
pipeline/visuals/manifest_patcher.py

Authoritative logic for linking and patching Stage 3 visual manifest images into
Stage 2 extracted JSON question records. Handles question stems, top-level parts,
compound subparts, and nested sub-subparts with question compatibility guards.
"""

import re
from typing import Any


def is_image_compatible_with_question(img_path: str, q_target: int) -> bool:
    """Sanity guard: ensure image file does not belong to a different question."""
    m_q = re.search(r"_q(\d+)_", img_path)
    if m_q and int(m_q.group(1)) != q_target:
        return False
    m_fig = re.search(r"Fig_(\d+)_", img_path)
    if m_fig and int(m_fig.group(1)) != q_target:
        return False
    m_tab = re.search(r"Table_(\d+)_", img_path)
    if m_tab and int(m_tab.group(1)) != q_target:
        return False
    return True


def patch_records_with_manifest(
    records: list[dict[str, Any]], manifest: list[dict[str, Any]]
) -> int:
    """Group manifest relative paths by (question, part) and inject image file paths
    into the matching question records and parts/subparts tree.

    Returns the number of patched image entries.
    """
    if not records or not manifest:
        return 0

    # Group manifest relative paths by (question, part)
    # part == None or "stem" -> question-level image
    q_to_stem_files: dict[int, list[str]] = {}
    q_part_to_files: dict[tuple[int, str], list[str]] = {}

    for entry in manifest:
        q = entry.get("question")
        if q is None:
            continue
        q_int = int(q)
        target_type = entry.get("target_type")
        target_label = entry.get("target_label")
        part = entry.get("part") or ""
        rp = entry.get("relative_path", "")

        if target_type == "question" or (
            target_type is None and part in ("question", "stem", "")
        ):
            q_to_stem_files.setdefault(q_int, []).append(rp)
        else:
            clean_part = str(
                target_label
                if (target_type in ("part", "sub_part") and target_label)
                else re.sub(r"^(?:sub_)?part_", "", part)
            ).strip()

            def _add_part_file(k: tuple[int, str], file_path: str):
                flist = q_part_to_files.setdefault(k, [])
                if file_path not in flist:
                    flist.append(file_path)

            m_comp = re.match(r"^([a-z])\s*\(?([ivx\d]+)\)?$", clean_part, re.IGNORECASE)
            if m_comp:
                p_l, sp_r = m_comp.group(1).lower(), m_comp.group(2).lower()
                _add_part_file((q_int, f"{p_l}({sp_r})"), rp)
                _add_part_file((q_int, f"{p_l}{sp_r}"), rp)
            else:
                norm_part = clean_part.strip("() ").lower()
                if norm_part:
                    _add_part_file((q_int, norm_part), rp)
                    _add_part_file((q_int, f"({norm_part})"), rp)

    patched = 0
    for record in records:
        q_raw = record.get("QuestionNumber") or record.get("AnswerNumber") or ""
        m = re.search(r"(?:Q|q)?(\d+)(?:\([a-z]\))?$", str(q_raw).split("/")[-1].strip())
        if not m:
            continue
        q_num = int(m.group(1))

        # Patch question-level image (stem figures)
        stem_candidates = [
            f
            for f in q_to_stem_files.get(q_num, [])
            if is_image_compatible_with_question(f, q_num)
        ]
        if stem_candidates:
            record["image"] = stem_candidates if len(stem_candidates) > 1 else stem_candidates[0]
            patched += 1

        # Patch part-level and nested sub-part images recursively
        def _patch_tree_images(
            node: dict[str, Any], is_top: bool = False, parent_lbl: str = ""
        ) -> int:
            label_key = "part_label" if is_top else "sub_part_label"
            plabel = str(node.get(label_key, "")).strip()
            cnt = 0
            if plabel:
                candidates = [
                    plabel,
                    plabel.strip("() ").lower(),
                    f"({plabel.strip('() ').lower()})",
                ]
                if parent_lbl:
                    clean_leaf = plabel.strip("() ").lower()
                    if clean_leaf.startswith(parent_lbl):
                        clean_leaf = clean_leaf[len(parent_lbl):].strip("() ")
                    candidates.extend(
                        [
                            f"{parent_lbl}({clean_leaf})",
                            f"{parent_lbl}{clean_leaf}",
                        ]
                    )

                part_files = None
                for cand in candidates:
                    if cand and (q_num, cand) in q_part_to_files:
                        matching_files = [
                            f
                            for f in q_part_to_files[(q_num, cand)]
                            if is_image_compatible_with_question(f, q_num)
                        ]
                        if matching_files:
                            part_files = matching_files
                            break
                if part_files:
                    node["image"] = part_files if len(part_files) > 1 else part_files[0]
                    cnt += 1
                else:
                    node["image"] = None
            curr_lbl = plabel.strip("() ").lower() if is_top else parent_lbl
            for sp in node.get("sub_parts", []):
                cnt += _patch_tree_images(sp, is_top=False, parent_lbl=curr_lbl)
            return cnt

        for part_rec in record.get("Parts", []):
            patched += _patch_tree_images(part_rec, is_top=True)

    return patched
