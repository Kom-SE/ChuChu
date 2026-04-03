#!/usr/bin/env python3
"""
MVTec AD 数据集转换 - 脚本 1: 环境检查
检查 Python 版本、依赖库，并统计数据集信息
"""

import sys
import importlib.util
from pathlib import Path


def check_python_version():
    """检查 Python 版本 >= 3.8"""
    print("=" * 60)
    print("检查 Python 版本...")
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version >= required_version:
        print(f"✓ Python {sys.version.split()[0]} (满足 >= 3.8 要求)")
        return True
    else:
        print(f"✗ Python {sys.version.split()[0]} 不满足要求 (需要 >= 3.8)")
        return False


def check_dependency(package_name):
    """检查单个依赖是否已安装"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None


def install_package(package_name):
    """安装缺失的依赖包"""
    import subprocess
    print(f"  正在安装 {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
        print(f"  ✓ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {package_name} 安装失败: {e}")
        return False


def check_dependencies():
    """检查并安装必要依赖"""
    print("\n" + "=" * 60)
    print("检查依赖库...")
    
    required_packages = {
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "yaml": "PyYAML",
        "skimage": "scikit-image",
    }
    
    all_installed = True
    for module_name, package_name in required_packages.items():
        if check_dependency(module_name):
            print(f"✓ {package_name} 已安装")
        else:
            print(f"✗ {package_name} 未安装")
            if install_package(package_name):
                pass
            else:
                all_installed = False
    
    return all_installed


def count_images_in_dir(dir_path: Path) -> int:
    """统计目录中的图片数量"""
    if not dir_path.exists():
        return 0
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    count = 0
    for ext in image_extensions:
        count += len(list(dir_path.glob(f"*{ext}")))
    return count


def count_dataset():
    """统计 MVTec AD 数据集"""
    print("\n" + "=" * 60)
    print("统计数据集...")
    
    SCRIPT_DIR = Path(__file__).parent
    WORKSPACE_DIR = SCRIPT_DIR.parent
    MVTEC_ROOT = WORKSPACE_DIR / "One" / "mvtec-ad"
    
    if not MVTEC_ROOT.exists():
        print(f"✗ 数据集目录不存在: {MVTEC_ROOT}")
        print("  请确保 MVTec AD 数据集已下载到 'One/mvtec-ad' 目录")
        return None
    
    categories = sorted([d.name for d in MVTEC_ROOT.iterdir() if d.is_dir()])
    
    if not categories:
        print("✗ 未找到任何类别目录")
        return None
    
    print(f"找到 {len(categories)} 个类别: {', '.join(categories)}\n")
    
    total_stats = {
        "train_good": 0,
        "test_good": 0,
        "test_defects": 0,
    }
    
    category_details = []
    
    for category in categories:
        cat_dir = MVTEC_ROOT / category
        
        train_good = count_images_in_dir(cat_dir / "train" / "good")
        test_good = count_images_in_dir(cat_dir / "test" / "good")
        
        defect_count = 0
        defect_types = []
        
        test_dir = cat_dir / "test"
        if test_dir.exists():
            for subdir in sorted(test_dir.iterdir()):
                if subdir.is_dir() and subdir.name != "good":
                    count = count_images_in_dir(subdir)
                    defect_count += count
                    defect_types.append(f"{subdir.name}({count})")
        
        total_stats["train_good"] += train_good
        total_stats["test_good"] += test_good
        total_stats["test_defects"] += defect_count
        
        defect_info = ", ".join(defect_types) if defect_types else "无"
        
        category_details.append({
            "name": category,
            "train_good": train_good,
            "test_good": test_good,
            "defect_count": defect_count,
            "defect_types": defect_info,
        })
        
        print(f"  [{category}]")
        print(f"    训练正常样本: {train_good}")
        print(f"    测试正常样本: {test_good}")
        print(f"    测试缺陷样本: {defect_count}")
        print(f"    缺陷类型: {defect_info}")
        print()
    
    print("-" * 60)
    print("【汇总统计】")
    print(f"  训练正常样本: {total_stats['train_good']}")
    print(f"  测试正常样本: {total_stats['test_good']}")
    print(f"  测试缺陷样本: {total_stats['test_defects']}")
    print(f"  样本总数: {sum(total_stats.values())}")
    print("-" * 60)
    
    return total_stats


def main():
    """主函数"""
    print("=" * 60)
    print("MVTec AD → YOLOv8 数据集转换")
    print("脚本 1: 环境检查与数据集统计")
    print("=" * 60)
    
    python_ok = check_python_version()
    deps_ok = check_dependencies()
    stats = count_dataset()
    
    print("\n" + "=" * 60)
    if python_ok and deps_ok and stats:
        print("✓ 环境检查通过！")
        print("\n下一步: 运行脚本 2 生成标注文件")
        print("  python scripts/02_generate_labels.py")
    else:
        print("✗ 环境检查未通过，请修复上述问题后重试")
    print("=" * 60)


if __name__ == "__main__":
    main()
