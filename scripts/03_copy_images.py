#!/usr/bin/env python3
"""
MVTec AD 数据集转换 - 脚本 3: 复制、整理图像并分层划分数据集
将 MVTec AD 所有数据合并后，按 8:2 比例进行分层随机划分
确保训练集和验证集都同时包含正常样本和缺陷样本
"""

import random
import shutil
from pathlib import Path
from collections import defaultdict


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
        list of dict: 每条记录包含
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
    分层随机划分：保证每个类别在训练集和验证集都有样本

    Args:
        samples: 所有样本列表
        train_ratio: 训练集比例 (默认 0.8)
        seed: 随机种子，保证可复现

    Returns:
        (train_samples, val_samples)
    """
    random.seed(seed)

    # 按 class_key 分组
    class_groups = defaultdict(list)
    for sample in samples:
        class_groups[sample["class_key"]].append(sample)

    train_samples = []
    val_samples = []

    for class_key, group in class_groups.items():
        # 打乱组内顺序
        random.shuffle(group)

        # 按比例划分
        split_idx = max(1, int(len(group) * train_ratio))
        train_samples.extend(group[:split_idx])
        val_samples.extend(group[split_idx:])

    # 整体打乱
    random.shuffle(train_samples)
    random.shuffle(val_samples)

    return train_samples, val_samples


def copy_and_split_images(samples: list, output_dir: Path) -> dict:
    """
    将样本复制到 train/val 目录

    Returns:
        dict: 统计信息
    """
    stats = {
        "train": 0,
        "val": 0,
        "skipped": 0,
        "errors": 0,
    }

    for sample in samples:
        pass  # 将在主循环中处理

    return stats


def main():
    """主函数"""
    print("=" * 60)
    print("MVTec AD → YOLOv8 数据集转换")
    print("脚本 3: 复制、整理图像并分层划分数据集")
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
    images_train_dir = YOLO_OUTPUT / "images" / "train"
    images_val_dir = YOLO_OUTPUT / "images" / "val"
    images_train_dir.mkdir(parents=True, exist_ok=True)
    images_val_dir.mkdir(parents=True, exist_ok=True)

    # 步骤 1: 收集所有样本
    print("\n[步骤 1/3] 收集所有数据样本...")
    all_samples = collect_all_samples(MVTEC_ROOT)
    print(f"  共收集到 {len(all_samples)} 个样本")

    # 统计各类别数量
    class_counts = defaultdict(int)
    defect_count = 0
    good_count = 0
    for s in all_samples:
        class_counts[s["class_key"]] += 1
        if s["is_defect"]:
            defect_count += 1
        else:
            good_count += 1

    print(f"  - 正常样本 (good): {good_count} 张")
    print(f"  - 缺陷样本 (defect): {defect_count} 张")
    print(f"  - 类别数量: {len(class_counts)} 种")

    # 步骤 2: 分层随机划分
    print("\n[步骤 2/3] 分层随机划分 (80% 训练 / 20% 验证)...")
    train_samples, val_samples = stratified_split(all_samples, train_ratio=0.8, seed=42)
    print(f"  - 训练集: {len(train_samples)} 张")
    print(f"  - 验证集: {len(val_samples)} 张")

    # 训练集统计
    train_defect = sum(1 for s in train_samples if s["is_defect"])
    train_good = len(train_samples) - train_defect
    val_defect = sum(1 for s in val_samples if s["is_defect"])
    val_good = len(val_samples) - val_defect

    print(f"\n  训练集构成:")
    print(f"    - 正常样本: {train_good} 张 ({100*train_good/len(train_samples):.1f}%)")
    print(f"    - 缺陷样本: {train_defect} 张 ({100*train_defect/len(train_samples):.1f}%)")
    print(f"  验证集构成:")
    print(f"    - 正常样本: {val_good} 张 ({100*val_good/len(val_samples):.1f}%)")
    print(f"    - 缺陷样本: {val_defect} 张 ({100*val_defect/len(val_samples):.1f}%)")

    # 步骤 3: 复制文件
    print("\n[步骤 3/3] 复制图像文件到目标目录...")

    total = len(train_samples) + len(val_samples)
    copied = 0
    skipped = 0
    errors = 0

    for sample in train_samples:
        dest_path = images_train_dir / f"{sample['new_name']}.png"
        try:
            if not dest_path.exists():
                shutil.copy2(sample["src_path"], dest_path)
                copied += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"\n  [警告] 复制失败 {sample['src_path'].name}: {e}")
            errors += 1

        # 进度显示（每 500 张显示一次）
        if (copied + skipped) % 500 == 0:
            pct = int(100 * (copied + skipped) / total)
            print(f"\r  进度: {copied + skipped}/{total} ({pct}%)", end='', flush=True)

    for sample in val_samples:
        dest_path = images_val_dir / f"{sample['new_name']}.png"
        try:
            if not dest_path.exists():
                shutil.copy2(sample["src_path"], dest_path)
                copied += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"\n  [警告] 复制失败 {sample['src_path'].name}: {e}")
            errors += 1

        if (copied + skipped) % 500 == 0:
            pct = int(100 * (copied + skipped) / total)
            print(f"\r  进度: {copied + skipped}/{total} ({pct}%)", end='', flush=True)

    print(f"\r  进度: {total}/{total} (100%)", end='', flush=True)
    print()  # 换行

    # 输出汇总
    print("\n" + "-" * 60)
    print("【图像复制与划分汇总】")
    print(f"  训练集图像: {len(train_samples)} 张")
    print(f"  验证集图像: {len(val_samples)} 张")
    print(f"  实际复制: {copied} 张")
    print(f"  跳过(已存在): {skipped} 张")
    if errors > 0:
        print(f"  错误: {errors} 张")
    print("-" * 60)

    # 验证目录
    actual_train = len(list(images_train_dir.glob("*")))
    actual_val = len(list(images_val_dir.glob("*")))
    print(f"\n输出目录: {YOLO_OUTPUT / 'images'}")
    print(f"  train: {actual_train} 个文件")
    print(f"  val: {actual_val} 个文件")

    print("\n[OK] 图像复制完成！")
    print("\n下一步: 运行脚本 2 生成标注文件")
    print("  python scripts/02_generate_labels.py")


if __name__ == "__main__":
    main()
