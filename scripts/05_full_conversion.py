#!/usr/bin/env python3
"""
MVTec AD 数据集转换 - 脚本 5: 一键完整转换
按顺序执行标签生成、图像复制、配置文件生成的完整流程
支持分层随机划分后的数据集结构
"""

import argparse
import cv2
import numpy as np
import random
import re
import shutil
from pathlib import Path
from collections import defaultdict


# 类别缺陷映射表 (与 02_generate_labels.py 保持一致)
CATEGORY_DEFECT_TO_ID = {
    # bottle (3)
    "bottle__broken_large": 0, "bottle__broken_small": 1, "bottle__contamination": 2,
    # cable (8)
    "cable__bent_wire": 3, "cable__cable_swap": 4, "cable__combined": 5,
    "cable__cut_inner_insulation": 6, "cable__cut_outer_insulation": 7,
    "cable__missing_cable": 8, "cable__missing_wire": 9, "cable__poke_insulation": 10,
    # capsule (5)
    "capsule__crack": 11, "capsule__faulty_imprint": 12, "capsule__poke": 13,
    "capsule__scratch": 14, "capsule__squeeze": 15,
    # carpet (5)
    "carpet__color": 16, "carpet__cut": 17, "carpet__hole": 18,
    "carpet__metal_contamination": 19, "carpet__thread": 20,
    # grid (5)
    "grid__bent": 21, "grid__broken": 22, "grid__glue": 23,
    "grid__metal_contamination": 24, "grid__thread": 25,
    # hazelnut (4)
    "hazelnut__crack": 26, "hazelnut__cut": 27, "hazelnut__hole": 28, "hazelnut__print": 29,
    # leather (6)
    "leather__color": 30, "leather__cut": 31, "leather__fold": 32,
    "leather__glue": 33, "leather__poke": 34,
    # metal_nut (4)
    "metal_nut__bent": 35, "metal_nut__color": 36, "metal_nut__flip": 37, "metal_nut__scratch": 38,
    # pill (7)
    "pill__color": 39, "pill__combined": 40, "pill__contamination": 41,
    "pill__crack": 42, "pill__faulty_imprint": 43, "pill__pill_type": 44, "pill__scratch": 45,
    # screw (5)
    "screw__manipulated_front": 46, "screw__scratch_head": 47,
    "screw__scratch_neck": 48, "screw__thread_side": 49, "screw__thread_top": 50,
    # toothbrush (1)
    "toothbrush__defective": 51,
    # transistor (4)
    "transistor__bent_lead": 52, "transistor__cut_lead": 53,
    "transistor__damaged_case": 54, "transistor__misplaced": 55,
    # wood (5)
    "wood__color": 56, "wood__combined": 57, "wood__hole": 58, "wood__liquid": 59, "wood__scratch": 60,
    # zipper (7)
    "zipper__broken_teeth": 61, "zipper__combined": 62, "zipper__fabric_border": 63,
    "zipper__fabric_interior": 64, "zipper__rough": 65, "zipper__split_teeth": 66, "zipper__squeezed_teeth": 67,
    # tile (5)
    "tile__crack": 68, "tile__glue_strip": 69, "tile__gray_stroke": 70, "tile__oil": 71, "tile__rough": 72,
}


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
    收集所有数据样本
    Returns:
        list of dict: 每条记录
            - src_path: 原始图片路径
            - class_key: 类别标识 (category__defect)
            - is_defect: 是否为缺陷样本
            - new_name: 新的文件名（不含扩展名）
    """
    samples = []
    categories = sorted([d.name for d in mvtec_root.iterdir() if d.is_dir()])

    for category in categories:
        cat_dir = mvtec_root / category

        # 1. train/good
        train_good_dir = cat_dir / "train" / "good"
        if train_good_dir.exists():
            for img_path in get_image_files(train_good_dir):
                samples.append({
                    "src_path": img_path,
                    "class_key": f"{category}__good",
                    "is_defect": False,
                    "new_name": f"{category}__good__{img_path.stem}",
                })

        # 2. test/good
        test_good_dir = cat_dir / "test" / "good"
        if test_good_dir.exists():
            for img_path in get_image_files(test_good_dir):
                samples.append({
                    "src_path": img_path,
                    "class_key": f"{category}__good",
                    "is_defect": False,
                    "new_name": f"{category}__good__{img_path.stem}",
                })

        # 3. test/<defect>
        test_dir = cat_dir / "test"
        if test_dir.exists():
            for defect_dir in sorted(test_dir.iterdir()):
                if not defect_dir.is_dir() or defect_dir.name == "good":
                    continue
                defect_name = defect_dir.name
                class_key = f"{category}__{defect_name}"
                for img_path in get_image_files(defect_dir):
                    samples.append({
                        "src_path": img_path,
                        "class_key": class_key,
                        "is_defect": True,
                        "new_name": f"{category}__{defect_name}__{img_path.stem}",
                    })

    return samples


def stratified_split(samples: list, train_ratio: float = 0.8, seed: int = 42) -> tuple:
    """
    分层随机划分（与脚本 2、3 保持一致）
    """
    random.seed(seed)
    class_groups = defaultdict(list)
    for sample in samples:
        class_groups[sample["class_key"]].append(sample)

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


def mask_to_yolo_boxes(mask: np.ndarray, min_area: int = 100) -> list:
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


def generate_labels_for_sample(sample: dict, mvtec_root: Path) -> tuple:
    """为单个样本生成标注，返回 (stats_dict, label_content_list)"""
    stats = {"defect_processed": 0, "defect_skipped": 0, "defect_no_mask": 0, "good": 0}
    class_key = sample["class_key"]

    if sample["is_defect"]:
        # 从 src_path 反推类别和缺陷类型
        parts = sample["src_path"].parts
        mvtec_idx = parts.index("mvtec-ad")
        category = parts[mvtec_idx + 1]
        defect_idx = parts.index("test") + 1
        defect_name = parts[defect_idx]

        gt_dir = mvtec_root / category / "ground_truth" / defect_name
        img_stem = sample["src_path"].stem
        mask_path = gt_dir / f"{img_stem}_mask.png"

        if not mask_path.exists():
            stats["defect_no_mask"] = 1
            return stats, []

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or np.max(mask) == 0:
            stats["defect_skipped"] = 1
            return stats, []

        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        boxes = mask_to_yolo_boxes(mask_binary)

        if not boxes:
            stats["defect_skipped"] = 1
            return stats, []

        if class_key not in CATEGORY_DEFECT_TO_ID:
            stats["defect_skipped"] = 1
            return stats, []

        class_id = CATEGORY_DEFECT_TO_ID[class_key]
        img_height, img_width = mask.shape[:2]

        label_content = []
        for (x, y, w, h) in boxes:
            cx = (x + w / 2) / img_width
            cy = (y + h / 2) / img_height
            nw = w / img_width
            nh = h / img_height
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))
            label_content.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        stats["defect_processed"] = 1
        return stats, label_content

    else:
        stats["good"] = 1
        return stats, []


def full_conversion(mvtec_root: Path, output_dir: Path):
    """执行完整的转换流程"""
    print("\n" + "=" * 60)
    print("步骤 1/3: 收集数据样本")
    print("=" * 60)

    all_samples = collect_all_samples(mvtec_root)
    print(f"共收集到 {len(all_samples)} 个样本")

    defect_count = sum(1 for s in all_samples if s["is_defect"])
    good_count = len(all_samples) - defect_count
    print(f"  - 正常样本: {good_count} 张")
    print(f"  - 缺陷样本: {defect_count} 张")
    print(f"  - 类别数量: {len(set(s['class_key'] for s in all_samples))} 种")

    print("\n" + "=" * 60)
    print("步骤 2/3: 分层随机划分 (80% 训练 / 20% 验证)")
    print("=" * 60)

    train_samples, val_samples = stratified_split(all_samples, train_ratio=0.8, seed=42)
    print(f"  - 训练集: {len(train_samples)} 张")
    print(f"  - 验证集: {len(val_samples)} 张")

    train_defect = sum(1 for s in train_samples if s["is_defect"])
    val_defect = sum(1 for s in val_samples if s["is_defect"])
    print(f"  训练集: 正常 {len(train_samples) - train_defect} 张, 缺陷 {train_defect} 张")
    print(f"  验证集: 正常 {len(val_samples) - val_defect} 张, 缺陷 {val_defect} 张")

    print("\n" + "=" * 60)
    print("步骤 3/3: 生成标注文件并复制图像")
    print("=" * 60)

    # 创建目录
    labels_train_dir = output_dir / "labels" / "train"
    labels_val_dir = output_dir / "labels" / "val"
    images_train_dir = output_dir / "images" / "train"
    images_val_dir = output_dir / "images" / "val"
    labels_train_dir.mkdir(parents=True, exist_ok=True)
    labels_val_dir.mkdir(parents=True, exist_ok=True)
    images_train_dir.mkdir(parents=True, exist_ok=True)
    images_val_dir.mkdir(parents=True, exist_ok=True)

    label_stats = {"defect_processed": 0, "defect_skipped": 0, "defect_no_mask": 0, "good": 0}

    # 处理训练集
    for idx, sample in enumerate(train_samples):
        stats, label_content = generate_labels_for_sample(sample, mvtec_root)
        for k in label_stats:
            label_stats[k] += stats[k]

        # 写标注
        label_path = labels_train_dir / f"{sample['new_name']}.txt"
        with open(label_path, 'w') as f:
            f.write("\n".join(label_content))

        # 复制图像
        dest_path = images_train_dir / f"{sample['new_name']}.png"
        if not dest_path.exists():
            shutil.copy2(sample["src_path"], dest_path)

        if (idx + 1) % 200 == 0:
            print(f"\r  训练集进度: {idx + 1}/{len(train_samples)}", end='', flush=True)

    print(f"\r  训练集: {len(train_samples)} 个完成")

    # 处理验证集
    for idx, sample in enumerate(val_samples):
        stats, label_content = generate_labels_for_sample(sample, mvtec_root)
        for k in label_stats:
            label_stats[k] += stats[k]

        # 写标注
        label_path = labels_val_dir / f"{sample['new_name']}.txt"
        with open(label_path, 'w') as f:
            f.write("\n".join(label_content))

        # 复制图像
        dest_path = images_val_dir / f"{sample['new_name']}.png"
        if not dest_path.exists():
            shutil.copy2(sample["src_path"], dest_path)

        if (idx + 1) % 200 == 0:
            print(f"\r  验证集进度: {idx + 1}/{len(val_samples)}", end='', flush=True)

    print(f"\r  验证集: {len(val_samples)} 个完成")

    print(f"\n标注统计:")
    print(f"  - 有标注（缺陷样本）: {label_stats['defect_processed']} 个")
    print(f"  - 空标注（正常样本）: {label_stats['good']} 个")
    print(f"  - 跳过(无mask/全黑): {label_stats['defect_skipped']} 个")
    print(f"  - 缺失mask: {label_stats['defect_no_mask']} 个")

    return label_stats


def generate_yaml(output_dir: Path):
    """生成 data.yaml"""
    labels_dir = output_dir / "labels"
    classes = set()
    pattern = re.compile(r'^([^_]+__[^_]+)__')

    for label_file in (labels_dir / "train").glob("*.txt"):
        match = pattern.match(label_file.stem)
        if match:
            classes.add(match.group(1))

    for label_file in (labels_dir / "val").glob("*.txt"):
        match = pattern.match(label_file.stem)
        if match:
            classes.add(match.group(1))

    class_mapping = {}
    for class_name in sorted(classes):
        if class_name in CATEGORY_DEFECT_TO_ID:
            class_mapping[CATEGORY_DEFECT_TO_ID[class_name]] = class_name

    yaml_path = output_dir / "data.yaml"
    yaml_content = f"""# MVTec AD Dataset for YOLOv8 Training
# MVTec License: CC BY-NC-SA 4.0
# 数据划分: train/val (80%/20%) - 所有数据合并后分层随机划分

path: YOLOv8_data/mvtec_yolo
train: images/train
val: images/val

nc: {len(class_mapping)}
names:
"""

    for class_id in sorted(class_mapping.keys(), key=lambda x: int(x)):
        yaml_content += f"  {class_id}: {class_mapping[class_id]}\n"

    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\ndata.yaml 已生成 ({len(class_mapping)} 个类别)")

    mapping_path = output_dir / "class_mapping.txt"
    lines = [f"{cid}: {cname}" for cid, cname in sorted(class_mapping.items(), key=lambda x: int(x[0]))]
    content = f"MVTec AD → YOLOv8 类别映射表\n{'='*50}\n总共 {len(class_mapping)} 个类别\n{'='*50}\n" + "\n".join(lines)

    with open(mapping_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"class_mapping.txt 已生成")

    return class_mapping


def validate_dataset(output_dir: Path) -> bool:
    """验证数据集完整性"""
    images_train = len(list((output_dir / "images" / "train").glob("*")))
    images_val = len(list((output_dir / "images" / "val").glob("*")))
    labels_train = len(list((output_dir / "labels" / "train").glob("*.txt")))
    labels_val = len(list((output_dir / "labels" / "val").glob("*.txt")))

    print(f"\n数据集验证:")
    print(f"  images/train: {images_train} 个文件")
    print(f"  images/val: {images_val} 个文件")
    print(f"  labels/train: {labels_train} 个文件")
    print(f"  labels/val: {labels_val} 个文件")

    train_match = images_train == labels_train
    val_match = images_val == labels_val

    if train_match and val_match:
        print("  [OK] 图像与标注数量匹配！")
        return True
    else:
        print("  [警告] 图像与标注数量不匹配:")
        if not train_match:
            print(f"    train: images={images_train}, labels={labels_train}")
        if not val_match:
            print(f"    val: images={images_val}, labels={labels_val}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MVTec AD → YOLOv8 一键转换（分层划分版）")
    parser.add_argument("--source", type=str, default=None,
                        help="数据源目录 (默认: WORKSPACE/One/mvtec-ad)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录 (默认: WORKSPACE/YOLOv8_data/mvtec_yolo)")
    args = parser.parse_args()

    SCRIPT_DIR = Path(__file__).parent
    WORKSPACE_DIR = SCRIPT_DIR.parent

    if args.source:
        MVTEC_ROOT = Path(args.source) if Path(args.source).is_absolute() else WORKSPACE_DIR / args.source
    else:
        MVTEC_ROOT = WORKSPACE_DIR / "One" / "mvtec-ad"

    if args.output:
        YOLO_OUTPUT = Path(args.output) if Path(args.output).is_absolute() else WORKSPACE_DIR / args.output
    else:
        YOLO_OUTPUT = WORKSPACE_DIR / "YOLOv8_data" / "mvtec_yolo"

    print("=" * 60)
    print("MVTec AD → YOLOv8 数据集转换")
    print("脚本 5: 一键完整转换（分层划分版）")
    print("=" * 60)
    print(f"\n数据源: {MVTEC_ROOT}")
    print(f"输出目录: {YOLO_OUTPUT}")

    if not MVTEC_ROOT.exists():
        print(f"\n[错误] 数据集目录不存在: {MVTEC_ROOT}")
        return

    full_conversion(MVTEC_ROOT, YOLO_OUTPUT)
    class_mapping = generate_yaml(YOLO_OUTPUT)
    validate_dataset(YOLO_OUTPUT)

    print("\n" + "=" * 60)
    print("[OK] 转换完成！")
    print("=" * 60)
    print(f"\n输出目录: {YOLO_OUTPUT}")
    print(f"  - images/train: 训练图像")
    print(f"  - images/val: 验证图像")
    print(f"  - labels/train: 训练标注")
    print(f"  - labels/val: 验证标注")
    print(f"  - data.yaml: YOLOv8 配置文件")
    print(f"  - class_mapping.txt: 类别映射表")
    print(f"\n类别数量: {len(class_mapping)}")
    print("\n训练命令:")
    print(f"  yolo detect train data=YOLOv8_data/mvtec_yolo/data.yaml model=yolov8m.pt epochs=100")


if __name__ == "__main__":
    main()
