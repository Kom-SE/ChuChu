#!/usr/bin/env python3
"""
MVTec AD 数据集转换 - 脚本 2: 生成 YOLO 标注文件
从 ground_truth mask 生成 YOLO 格式的 .txt 标注文件
与脚本 3 保持相同的分层随机划分 (seed=42, 80%/20%)
"""

import cv2
import numpy as np
import random
from pathlib import Path
from collections import defaultdict


# 类别缺陷映射表 (category__defect -> class_id)
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
    收集所有数据样本（与脚本 3 保持一致）
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
    分层随机划分（与脚本 3 保持一致）
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


def generate_label_for_sample(sample: dict, mvtec_root: Path, output_dir: Path) -> dict:
    """
    为单个样本生成标注文件

    Returns:
        dict: 统计信息
    """
    stats = {
        "defect_processed": 0,
        "defect_skipped": 0,
        "defect_no_mask": 0,
        "good": 0,
    }

    class_key = sample["class_key"]
    new_name = sample["new_name"]

    if sample["is_defect"]:
        # 从 src_path 反推类别和缺陷类型
        # src_path 格式: .../mvtec-ad/<category>/test/<defect>/<img>.png
        # 或者: .../mvtec-ad/<category>/train/good/<img>.png
        parts = sample["src_path"].parts
        # 找到 mvtec-ad 在路径中的位置
        mvtec_idx = parts.index("mvtec-ad")
        category = parts[mvtec_idx + 1]

        # 判断是 train/good 还是 test/<defect>
        if "train" in parts:
            # train/good，不会走到这里
            pass
        else:
            # test/<defect>
            defect_idx = parts.index("test") + 1
            defect_name = parts[defect_idx]

        # 读取 mask
        gt_dir = mvtec_root / category / "ground_truth" / defect_name
        img_stem = sample["src_path"].stem
        mask_path = gt_dir / f"{img_stem}_mask.png"

        if not mask_path.exists():
            print(f"  [警告] Mask 不存在: {mask_path}")
            stats["defect_no_mask"] = 1
            return stats

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or np.max(mask) == 0:
            print(f"  [警告] Mask 全黑或读取失败: {mask_path.name}")
            stats["defect_skipped"] = 1
            return stats

        # 提取 bounding boxes
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        boxes = mask_to_yolo_boxes(mask_binary)

        if not boxes:
            print(f"  [警告] 无有效区域: {mask_path.name}")
            stats["defect_skipped"] = 1
            return stats

        # 获取类别 ID
        if class_key not in CATEGORY_DEFECT_TO_ID:
            print(f"  [警告] 未知类别: {class_key}")
            stats["defect_skipped"] = 1
            return stats

        class_id = CATEGORY_DEFECT_TO_ID[class_key]
        img_height, img_width = mask.shape[:2]

        # 写入标注文件
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
        # 正常样本 - 生成空标注文件
        stats["good"] = 1
        return stats, []


def main():
    """主函数"""
    print("=" * 60)
    print("MVTec AD → YOLOv8 数据集转换")
    print("脚本 2: 生成 YOLO 标注文件（分层划分版）")
    print("=" * 60)

    # 设置路径
    SCRIPT_DIR = Path(__file__).parent
    WORKSPACE_DIR = SCRIPT_DIR.parent
    MVTEC_ROOT = WORKSPACE_DIR / "One" / "mvtec-ad"
    YOLO_OUTPUT = WORKSPACE_DIR / "YOLOv8_data" / "mvtec_yolo"

    if not MVTEC_ROOT.exists():
        print(f"[错误] 数据集目录不存在: {MVTEC_ROOT}")
        return

    # 创建输出目录
    labels_train_dir = YOLO_OUTPUT / "labels" / "train"
    labels_val_dir = YOLO_OUTPUT / "labels" / "val"
    labels_train_dir.mkdir(parents=True, exist_ok=True)
    labels_val_dir.mkdir(parents=True, exist_ok=True)

    # 步骤 1: 收集所有样本
    print("\n[步骤 1/3] 收集所有数据样本...")
    all_samples = collect_all_samples(MVTEC_ROOT)
    print(f"  共收集到 {len(all_samples)} 个样本")

    # 步骤 2: 分层随机划分（与脚本 3 使用相同参数）
    print("\n[步骤 2/3] 分层随机划分 (80% 训练 / 20% 验证)...")
    train_samples, val_samples = stratified_split(all_samples, train_ratio=0.8, seed=42)
    print(f"  - 训练集: {len(train_samples)} 张")
    print(f"  - 验证集: {len(val_samples)} 张")

    # 步骤 3: 生成标注文件
    print("\n[步骤 3/3] 生成标注文件...")

    stats = {
        "defect_processed": 0,
        "defect_skipped": 0,
        "defect_no_mask": 0,
        "good": 0,
    }

    # 处理训练集
    print("  处理训练集...")
    for i, sample in enumerate(train_samples):
        result = generate_label_for_sample(sample, MVTEC_ROOT, YOLO_OUTPUT)

        if isinstance(result, tuple):
            s, label_content = result
            label_path = labels_train_dir / f"{sample['new_name']}.txt"
            with open(label_path, 'w') as f:
                f.write("\n".join(label_content))
        else:
            s = result
            if s["good"]:
                label_path = labels_train_dir / f"{sample['new_name']}.txt"
                if not label_path.exists():
                    label_path.touch()

        for k in stats:
            stats[k] += s.get(k, 0)

        if (i + 1) % 500 == 0:
            print(f"\r    进度: {i + 1}/{len(train_samples)}", end='', flush=True)

    print(f"\r    训练集: {len(train_samples)} 个标注已完成")

    # 处理验证集
    print("  处理验证集...")
    for i, sample in enumerate(val_samples):
        result = generate_label_for_sample(sample, MVTEC_ROOT, YOLO_OUTPUT)

        if isinstance(result, tuple):
            s, label_content = result
            label_path = labels_val_dir / f"{sample['new_name']}.txt"
            with open(label_path, 'w') as f:
                f.write("\n".join(label_content))
        else:
            s = result
            if s["good"]:
                label_path = labels_val_dir / f"{sample['new_name']}.txt"
                if not label_path.exists():
                    label_path.touch()

        for k in stats:
            stats[k] += s.get(k, 0)

        if (i + 1) % 500 == 0:
            print(f"\r    进度: {i + 1}/{len(val_samples)}", end='', flush=True)

    print(f"\r    验证集: {len(val_samples)} 个标注已完成")

    # 输出汇总
    print("\n" + "-" * 60)
    print("【标注生成汇总】")
    print(f"  缺陷样本标注: {stats['defect_processed']} 个文件")
    print(f"  正常样本标注: {stats['good']} 个文件（空标注）")
    print(f"  跳过(无mask/全黑): {stats['defect_skipped']} 个")
    print(f"  缺失mask: {stats['defect_no_mask']} 个")
    print("-" * 60)

    # 验证文件数量
    actual_train_labels = len(list(labels_train_dir.glob("*.txt")))
    actual_val_labels = len(list(labels_val_dir.glob("*.txt")))
    print(f"\n输出目录: {YOLO_OUTPUT / 'labels'}")
    print(f"  train: {actual_train_labels} 个标注文件")
    print(f"  val: {actual_val_labels} 个标注文件")

    # 检查与图像数量是否匹配
    images_train = len(list((YOLO_OUTPUT / "images" / "train").glob("*")))
    images_val = len(list((YOLO_OUTPUT / "images" / "val").glob("*")))
    print(f"\n图像目录:")
    print(f"  train: {images_train} 个图像")
    print(f"  val: {images_val} 个图像")

    if images_train == actual_train_labels and images_val == actual_val_labels:
        print("\n  [OK] 标注数量与图像数量匹配！")
    else:
        print("\n  [警告] 标注数量与图像数量不匹配！")
        print(f"    train: 图像={images_train}, 标注={actual_train_labels}")
        print(f"    val: 图像={images_val}, 标注={actual_val_labels}")

    print("\n[OK] 标注文件生成完成！")
    print("\n下一步: 运行脚本 4 生成配置文件")
    print("  python scripts/04_generate_yaml.py")


if __name__ == "__main__":
    main()
