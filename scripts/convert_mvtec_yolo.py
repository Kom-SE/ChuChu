#!/usr/bin/env python3
"""
MVTec AD 数据集转换 - 统一脚本
将 MVTec AD 转换为 YOLOv8 格式，用于通用工业质检。

核心设计：
- 15 个产品级类别（0-14），所有缺陷类型归入同一个产品类
- 正常样本（good）标注为空文件
- 缺陷样本标注为 "<class_id> <cx> <cy> <w> <h>"
- 所有数据（train/good + test/good + test/defect）合并后 8:2 分层随机划分
- 标注文件命名: <category>__<image_name>.png / <category>__<defect>__<image_name>.png
"""

import argparse
import cv2
import numpy as np
import random
import shutil
from pathlib import Path
from collections import defaultdict


# 15 个产品级类别映射
PRODUCT_CATEGORY_NAMES = [
    "bottle",       # 0
    "cable",        # 1
    "capsule",      # 2
    "carpet",       # 3
    "grid",         # 4
    "hazelnut",     # 5
    "leather",      # 6
    "metal_nut",    # 7
    "pill",         # 8
    "screw",        # 9
    "toothbrush",   # 10
    "transistor",   # 11
    "wood",         # 12
    "zipper",       # 13
    "tile",         # 14
]
PRODUCT_CATEGORY_TO_ID = {name: idx for idx, name in enumerate(PRODUCT_CATEGORY_NAMES)}


def get_image_files(directory: Path) -> list:
    """获取目录中的所有图片文件"""
    if not directory.exists():
        return []
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    files = []
    for ext in image_extensions:
        files.extend(directory.glob(f"*{ext}"))
    return sorted(files)


def collect_all_samples(mvtec_root: Path) -> list:
    """
    收集所有数据样本（train/good + test/good + test/defect）

    Returns:
        list of dict:
            - src_path:     原始图片路径
            - category:    产品类别名（如 "bottle"）
            - is_defect:   是否为缺陷样本
            - defect_name: 缺陷类型名（如 "broken_large"，正常样本为 "good"）
            - new_name:    新的文件名（不含扩展名）
    """
    samples = []
    categories = sorted([d.name for d in mvtec_root.iterdir() if d.is_dir()])

    for category in categories:
        if category not in PRODUCT_CATEGORY_TO_ID:
            print(f"  [跳过] 未知类别: {category}")
            continue

        cat_dir = mvtec_root / category

        # 1. train/good（正常样本）
        train_good_dir = cat_dir / "train" / "good"
        if train_good_dir.exists():
            for img_path in get_image_files(train_good_dir):
                samples.append({
                    "src_path": img_path,
                    "category": category,
                    "is_defect": False,
                    "defect_name": "good",
                    "new_name": f"{category}__good__{img_path.stem}",
                })

        # 2. test/good（正常样本）
        test_good_dir = cat_dir / "test" / "good"
        if test_good_dir.exists():
            for img_path in get_image_files(test_good_dir):
                samples.append({
                    "src_path": img_path,
                    "category": category,
                    "is_defect": False,
                    "defect_name": "good",
                    "new_name": f"{category}__good__{img_path.stem}",
                })

        # 3. test/<defect>（缺陷样本）
        test_dir = cat_dir / "test"
        if test_dir.exists():
            for defect_dir in sorted(test_dir.iterdir()):
                if not defect_dir.is_dir() or defect_dir.name == "good":
                    continue
                defect_name = defect_dir.name
                for img_path in get_image_files(defect_dir):
                    samples.append({
                        "src_path": img_path,
                        "category": category,
                        "is_defect": True,
                        "defect_name": defect_name,
                        "new_name": f"{category}__{defect_name}__{img_path.stem}",
                    })

    return samples


def stratified_split(samples: list, train_ratio: float = 0.8, seed: int = 42) -> tuple:
    """
    分层随机划分：按产品类别划分，保证每个产品类在训练集和验证集都有样本
    """
    random.seed(seed)
    class_groups = defaultdict(list)
    for sample in samples:
        # 按 category 分层（所有缺陷类型归入同一类）
        class_groups[sample["category"]].append(sample)

    train_samples = []
    val_samples = []

    for class_key, group in class_groups.items():
        random.shuffle(group)
        split_idx = max(1, int(len(group) * train_ratio))
        train_samples.extend(group[:split_idx])
        val_samples.extend(group[split_idx:])

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    return train_samples, val_samples


def mask_to_yolo_boxes(mask: np.ndarray, min_area: int = 50) -> list:
    """从二值 mask 提取 YOLO 格式的 bounding boxes"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))
    return boxes


def generate_label_for_defect(sample: dict, mvtec_root: Path) -> tuple:
    """
    为缺陷样本生成 YOLO 标注内容

    Returns:
        (stats_dict, label_lines): stats_dict 统计跳过原因，label_lines 为标注行列表
    """
    stats = {"processed": 0, "skipped": 0, "no_mask": 0, "empty_mask": 0, "no_box": 0}

    category = sample["category"]
    defect_name = sample["defect_name"]
    img_path = sample["src_path"]

    # 查找 ground_truth mask
    gt_dir = mvtec_root / category / "ground_truth" / defect_name
    img_stem = img_path.stem
    mask_path = gt_dir / f"{img_stem}_mask.png"

    if not mask_path.exists():
        # 尝试不带 _mask 后缀
        alt_mask_path = gt_dir / f"{img_stem}.png"
        if alt_mask_path.exists():
            mask_path = alt_mask_path

    if not mask_path.exists():
        stats["no_mask"] = 1
        return stats, []

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        stats["skipped"] = 1
        return stats, []

    if np.max(mask) == 0:
        stats["empty_mask"] = 1
        return stats, []

    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    boxes = mask_to_yolo_boxes(mask_binary)

    if not boxes:
        stats["no_box"] = 1
        return stats, []

    class_id = PRODUCT_CATEGORY_TO_ID[category]
    img_height, img_width = mask.shape[:2]

    label_lines = []
    for (x, y, w, h) in boxes:
        cx = (x + w / 2) / img_width
        cy = (y + h / 2) / img_height
        nw = w / img_width
        nh = h / img_height
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))
        label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    stats["processed"] = 1
    return stats, label_lines


def run_conversion(mvtec_root: Path, output_dir: Path, force: bool = False):
    """执行完整的转换流程"""
    # 创建输出根目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建输出目录
    labels_train_dir = output_dir / "labels" / "train"
    labels_val_dir = output_dir / "labels" / "val"
    images_train_dir = output_dir / "images" / "train"
    images_val_dir = output_dir / "images" / "val"

    # 清理旧数据（如果 force）- 使用 rmtree 强制删除
    if force:
        for d in [labels_train_dir, labels_val_dir, images_train_dir, images_val_dir]:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        for d in [labels_train_dir, labels_val_dir, images_train_dir, images_val_dir]:
            d.mkdir(parents=True, exist_ok=True)
    else:
        for d in [labels_train_dir, labels_val_dir, images_train_dir, images_val_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # 步骤 1: 收集数据
    print("\n" + "=" * 60)
    print("步骤 1/4: 收集数据样本")
    print("=" * 60)

    all_samples = collect_all_samples(mvtec_root)
    if not all_samples:
        print("[错误] 未找到任何数据样本！请检查数据源路径。")
        return False

    defect_count = sum(1 for s in all_samples if s["is_defect"])
    good_count = len(all_samples) - defect_count
    categories_found = set(s["category"] for s in all_samples)

    print(f"  共收集到 {len(all_samples)} 个样本")
    print(f"  - 正常样本 (good): {good_count} 张")
    print(f"  - 缺陷样本 (defect): {defect_count} 张")
    print(f"  - 产品类别: {len(categories_found)} 种 ({sorted(categories_found)})")

    # 步骤 2: 分层划分
    print("\n" + "=" * 60)
    print("步骤 2/4: 分层随机划分 (80% 训练 / 20% 验证)")
    print("=" * 60)

    train_samples, val_samples = stratified_split(all_samples, train_ratio=0.8, seed=42)

    train_defect = sum(1 for s in train_samples if s["is_defect"])
    val_defect = sum(1 for s in val_samples if s["is_defect"])

    print(f"  训练集: {len(train_samples)} 张 (正常 {len(train_samples)-train_defect}, 缺陷 {train_defect})")
    print(f"  验证集: {len(val_samples)} 张 (正常 {len(val_samples)-val_defect}, 缺陷 {val_defect})")

    # 步骤 3: 复制图像并生成标注
    print("\n" + "=" * 60)
    print("步骤 3/4: 复制图像 & 生成标注文件")
    print("=" * 60)

    label_stats = {"processed": 0, "skipped": 0, "no_mask": 0, "empty_mask": 0, "no_box": 0, "good": 0}
    total_copied = 0

    def process_split(samples: list, split_name: str, img_dir: Path, lbl_dir: Path):
        nonlocal total_copied, label_stats

        for idx, sample in enumerate(samples):
            new_name = sample["new_name"]
            img_dest = img_dir / f"{new_name}.png"
            lbl_path = lbl_dir / f"{new_name}.txt"

            # 复制图像
            try:
                if not img_dest.exists():
                    shutil.copy2(sample["src_path"], img_dest)
                    total_copied += 1
            except PermissionError:
                pass  # 文件已存在但被锁定，跳过

            # 生成标注
            if sample["is_defect"]:
                stats, label_lines = generate_label_for_defect(sample, mvtec_root)
                for k in stats:
                    label_stats[k] = label_stats.get(k, 0) + stats[k]
                with open(lbl_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(label_lines))
            else:
                label_stats["good"] += 1
                # 正常样本: 空标注文件（内容为空）
                try:
                    with open(lbl_path, 'w', encoding='utf-8') as f:
                        f.write("")
                except PermissionError:
                    pass  # 文件已存在但被锁定，跳过

            if (idx + 1) % 500 == 0:
                print(f"\r    [{split_name}] 进度: {idx + 1}/{len(samples)}", end='', flush=True)

        print(f"\r    [{split_name}] 完成: {len(samples)} 个样本")

    print("  处理训练集...")
    process_split(train_samples, "train", images_train_dir, labels_train_dir)

    print("  处理验证集...")
    process_split(val_samples, "val", images_val_dir, labels_val_dir)

    print(f"\n  标注统计:")
    print(f"    - 缺陷样本标注成功: {label_stats['processed']} 个")
    print(f"    - 缺陷样本跳过: {label_stats['skipped']} 个 (读取失败)")
    print(f"    - 缺失 mask: {label_stats['no_mask']} 个")
    print(f"    - 空 mask: {label_stats['empty_mask']} 个")
    print(f"    - 无有效区域: {label_stats['no_box']} 个")
    print(f"    - 正常样本标注: {label_stats['good']} 个 (空标注)")
    print(f"  图像复制: {total_copied} 张新图像")

    # 步骤 4: 生成配置文件
    print("\n" + "=" * 60)
    print("步骤 4/4: 生成配置文件 (data.yaml, class_mapping.txt)")
    print("=" * 60)

    yaml_path = output_dir / "data.yaml"
    yaml_content = f"""# MVTec AD Dataset for YOLOv8 Training
# MVTec License: CC BY-NC-SA 4.0
# Auto-generated by convert_mvtec_yolo.py
# 数据划分: train/val (80%/20%) - 所有数据合并后分层随机划分
# 类别: 15 个产品级类别（所有缺陷类型归入对应产品类）

path: YOLOv8_data/mvtec_yolo
train: images/train
val: images/val

nc: 15
names:
"""
    for idx, name in enumerate(PRODUCT_CATEGORY_NAMES):
        yaml_content += f"  {idx}: {name}\n"

    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"  [OK] {yaml_path.name} (nc=15)")

    # 生成 class_mapping.txt
    mapping_path = output_dir / "class_mapping.txt"
    lines = [f"{idx}: {name}" for idx, name in enumerate(PRODUCT_CATEGORY_NAMES)]
    sep = "=" * 50
    mapping_content = (
        f"MVTec AD → YOLOv8 类别映射表 (产品级)\n{sep}\n"
        f"总共 15 个类别\n{sep}\n"
        + "\n".join(lines)
        + "\n\n说明: 所有缺陷类型（裂纹、污渍、变形等）统一归入对应产品类别\n"
        + "正常样本(good)标注为空文件"
    )
    with open(mapping_path, 'w', encoding='utf-8') as f:
        f.write(mapping_content)
    print(f"  [OK] {mapping_path.name}")

    # 验证数据集完整性
    print("\n" + "=" * 60)
    print("数据集验证")
    print("=" * 60)

    images_train_n = len(list(images_train_dir.glob("*")))
    images_val_n = len(list(images_val_dir.glob("*")))
    labels_train_n = len(list(labels_train_dir.glob("*.txt")))
    labels_val_n = len(list(labels_val_dir.glob("*.txt")))

    print(f"  images/train: {images_train_n} 个文件")
    print(f"  images/val:   {images_val_n} 个文件")
    print(f"  labels/train: {labels_train_n} 个文件")
    print(f"  labels/val:   {labels_val_n} 个文件")

    train_ok = images_train_n == labels_train_n
    val_ok = images_val_n == labels_val_n

    if train_ok and val_ok:
        print("\n  [OK] 图像与标注数量完全匹配！")
    else:
        if not train_ok:
            print(f"\n  [警告] train: 图像={images_train_n}, 标注={labels_train_n}")
        if not val_ok:
            print(f"\n  [警告] val: 图像={images_val_n}, 标注={labels_val_n}")

    # 统计各类别数量
    print("\n  各类别样本数量:")
    cat_counts = defaultdict(lambda: {"train": 0, "val": 0})
    for s in train_samples:
        cat_counts[s["category"]]["train"] += 1
    for s in val_samples:
        cat_counts[s["category"]]["val"] += 1

    for cat in PRODUCT_CATEGORY_NAMES:
        if cat in cat_counts:
            c = cat_counts[cat]
            print(f"    {cat:15s} (class {PRODUCT_CATEGORY_TO_ID[cat]:2d}): train={c['train']:4d}, val={c['val']:4d}")

    return train_ok and val_ok


def main():
    parser = argparse.ArgumentParser(description="MVTec AD → YOLOv8 格式转换（15产品级类别）")
    parser.add_argument("--source", type=str, default=None,
                        help="数据源目录（默认: WORKSPACE/One/mvtec-ad）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录（默认: WORKSPACE/YOLOv8_data/mvtec_yolo）")
    parser.add_argument("--force", action="store_true",
                        help="强制重新生成（删除旧数据）")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    workspace_dir = script_dir.parent

    if args.source:
        mvtec_root = Path(args.source) if Path(args.source).is_absolute() else workspace_dir / args.source
    else:
        mvtec_root = workspace_dir / "One" / "mvtec-ad"

    if args.output:
        output_dir = Path(args.output) if Path(args.output).is_absolute() else workspace_dir / args.output
    else:
        output_dir = workspace_dir / "YOLOv8_data" / "mvtec_yolo"

    print("=" * 60)
    print("MVTec AD → YOLOv8 数据集转换")
    print("统一脚本（15 个产品级类别 / 80/20 分层划分）")
    print("=" * 60)
    print(f"\n数据源:   {mvtec_root}")
    print(f"输出目录: {output_dir}")

    if not mvtec_root.exists():
        print(f"\n[错误] 数据集目录不存在: {mvtec_root}")
        return

    ok = run_conversion(mvtec_root, output_dir, force=args.force)

    print("\n" + "=" * 60)
    if ok:
        print("[OK] 转换完成！")
    else:
        print("[警告] 转换完成，但图像/标注数量不匹配")
    print("=" * 60)
    print(f"\n输出目录: {output_dir}")
    print("  ├── images/train/  训练图像")
    print("  ├── images/val/    验证图像")
    print("  ├── labels/train/  训练标注")
    print("  ├── labels/val/    验证标注")
    print("  ├── data.yaml       YOLOv8 配置文件")
    print("  └── class_mapping.txt 类别映射表")
    print(f"\n类别数量: 15")
    print("\n训练命令:")
    print("  yolo detect train data=YOLOv8_data/mvtec_yolo/data.yaml model=yolov8m.pt epochs=100 imgsz=512 batch=8 device=0")


if __name__ == "__main__":
    main()
