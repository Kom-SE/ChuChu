import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8m 训练脚本 - MVTec AD")
    parser.add_argument(
        "--data", type=str,
        default="YOLOv8_data/mvtec_yolo/data.yaml",
        help="数据集配置文件路径（相对于 WORKSPACE 或绝对路径）"
    )
    parser.add_argument(
        "--model", type=str,
        default="yolov8m.pt",
        help="预训练模型路径或模型名（默认: yolov8m.pt）"
    )
    parser.add_argument(
        "--project", type=str,
        default="runs/detect",
        help="训练输出项目目录（默认: runs/detect）"
    )
    parser.add_argument(
        "--name", type=str,
        default="mvtec_yolov8m",
        help="本次训练子目录名（默认: mvtec_yolov8m）"
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数（默认: 100）")
    parser.add_argument("--batch", type=int, default=8, help="批大小（默认: 8）")
    parser.add_argument("--imgsz", type=int, default=512, help="输入图像尺寸（默认: 512）")
    parser.add_argument("--device", type=str, default="0", help="训练设备，如 '0'、'cpu'（默认: '0'）")
    parser.add_argument("--workers", type=int, default=4, help="数据加载线程数（默认: 4）")

    # 优化器参数
    parser.add_argument("--optimizer", type=str, default="auto",
                        choices=["auto", "SGD", "Adam", "AdamW"],
                        help="优化器（默认: auto）")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率（默认: 0.01）")
    parser.add_argument("--lrf", type=float, default=0.01, help="最终学习率 = lr0 * lrf（默认: 0.01）")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD 动量（默认: 0.937）")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="权重衰减（默认: 0.0005）")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="预热轮数（默认: 3）")
    parser.add_argument("--warmup_momentum", type=float, default=0.8, help="预热动量（默认: 0.8）")
    parser.add_argument("--warmup_bias_lr", type=float, default=0.1, help="预热 bias 学习率（默认: 0.1）")

    # 数据增强参数
    parser.add_argument("--hsv_h", type=float, default=0.015, help="HSV 色调增强（默认: 0.015）")
    parser.add_argument("--hsv_s", type=float, default=0.7, help="HSV 饱和度增强（默认: 0.7）")
    parser.add_argument("--hsv_v", type=float, default=0.4, help="HSV 亮度增强（默认: 0.4）")
    parser.add_argument("--degrees", type=float, default=0.0, help="旋转角度范围（默认: 0.0）")
    parser.add_argument("--translate", type=float, default=0.1, help="平移比例（默认: 0.1）")
    parser.add_argument("--scale", type=float, default=0.5, help="缩放比例（默认: 0.5）")
    parser.add_argument("--shear", type=float, default=0.0, help="剪切角度（默认: 0.0）")
    parser.add_argument("--perspective", type=float, default=0.0, help="透视变换（默认: 0.0）")
    parser.add_argument("--flipud", type=float, default=0.0, help="上下翻转概率（默认: 0.0）")
    parser.add_argument("--fliplr", type=float, default=0.5, help="左右翻转概率（默认: 0.5）")
    parser.add_argument("--mosaic", type=float, default=1.0, help="mosaic 增强概率（默认: 1.0）")
    parser.add_argument("--mixup", type=float, default=0.1,
                        help="mixup 增强概率，0 表示关闭（默认: 0.1）")
    parser.add_argument("--copy_paste", type=float, default=0.1,
                        help="copy-paste 增强概率，0 表示关闭（默认: 0.1）")
    parser.add_argument("--copy_paste_mode", type=str, default="flip",
                        choices=["flip", "fregrid"],
                        help="copy-paste 模式（默认: flip）")
    parser.add_argument("--auto_augment", type=str, default="randaugment",
                        choices=["randaugment", "autoaugment", "augmix"],
                        help="自动增强策略（默认: randaugment）")
    parser.add_argument("--erasing", type=float, default=0.4,
                        help="随机擦除概率（默认: 0.4）")

    # 正则化与损失
    parser.add_argument("--box", type=float, default=7.5, help="box loss 权重（默认: 7.5）")
    parser.add_argument("--cls", type=float, default=0.5, help="cls loss 权重（默认: 0.5）")
    parser.add_argument("--dfl", type=float, default=1.5, help="dfl loss 权重（默认: 1.5）")

    # 其他
    parser.add_argument("--resume", type=str, default=None, help="从上次中断处继续训练（传入模型路径）")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="使用预训练权重（默认: True）")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="打印详细日志（默认: True）")
    parser.add_argument("--patience", type=int, default=50,
                        help="早停耐心值，-1 表示关闭（默认: 50）")
    parser.add_argument("--close_mosaic", type=int, default=10,
                        help="训练最后 N 轮关闭 mosaic 增强（默认: 10）")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="启用混合精度训练（默认: True）")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="使用数据集的比例（默认: 1.0）")
    parser.add_argument("--profile", action="store_true",
                        help="开启 ONNX 导出时的 op 耗时分析")
    parser.add_argument("--overlap_mask", action="store_true", default=True,
                        help="训练时 mask 重叠（默认: True）")
    parser.add_argument("--mask_ratio", type=int, default=4,
                        help="mask 下采样比例（默认: 4）")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout 比例，仅在 cls 任务模式有效（默认: 0.0）")

    return parser.parse_args()


def resolve_data_path(data_arg: str) -> str:
    """解析 data.yaml 路径，支持相对路径和绝对路径"""
    p = Path(data_arg)
    if p.is_absolute() and p.exists():
        return str(p)
    workspace = Path(__file__).parent.parent
    candidate = workspace / data_arg
    if candidate.exists():
        return str(candidate)
    return data_arg


def build_cli_args(args) -> list:
    """将 Namespace 参数转换为 yolo CLI 参数列表"""
    cli = []

    def add(key, value):
        if isinstance(value, bool):
            if value:
                cli.extend([f"{key}", "True"])
        else:
            cli.extend([f"{key}", str(value)])

    add("--data", args.data)
    add("--model", args.model)
    add("--project", args.project)
    add("--name", args.name)
    add("--epochs", args.epochs)
    add("--batch", args.batch)
    add("--imgsz", args.imgsz)
    add("--device", args.device)
    add("--workers", args.workers)
    add("--optimizer", args.optimizer)
    add("--lr0", args.lr0)
    add("--lrf", args.lrf)
    add("--momentum", args.momentum)
    add("--weight_decay", args.weight_decay)
    add("--warmup_epochs", args.warmup_epochs)
    add("--warmup_momentum", args.warmup_momentum)
    add("--warmup_bias_lr", args.warmup_bias_lr)
    add("--hsv_h", args.hsv_h)
    add("--hsv_s", args.hsv_s)
    add("--hsv_v", args.hsv_v)
    add("--degrees", args.degrees)
    add("--translate", args.translate)
    add("--scale", args.scale)
    add("--shear", args.shear)
    add("--perspective", args.perspective)
    add("--flipud", args.flipud)
    add("--fliplr", args.fliplr)
    add("--mosaic", args.mosaic)
    add("--mixup", args.mixup)
    add("--copy_paste", args.copy_paste)
    add("--copy_paste_mode", args.copy_paste_mode)
    add("--auto_augment", args.auto_augment)
    add("--erasing", args.erasing)
    add("--box", args.box)
    add("--cls", args.cls)
    add("--dfl", args.dfl)
    add("--patience", args.patience)
    add("--close_mosaic", args.close_mosaic)
    add("--amp", args.amp)
    add("--fraction", args.fraction)
    add("--overlap_mask", args.overlap_mask)
    add("--mask_ratio", args.mask_ratio)
    add("--dropout", args.dropout)

    if args.resume:
        add("--resume", args.resume)
    if args.verbose:
        add("--verbose", args.verbose)
    if args.pretrained:
        add("--pretrained", args.pretrained)
    if args.profile:
        add("--profile", args.profile)

    return cli


def main():
    args = parse_args()

    # 解析数据集路径
    data_path = resolve_data_path(args.data)
    args.data = data_path
    print("=" * 60)
    print("YOLOv8m 训练脚本 - MVTec AD 工业质检")
    print("=" * 60)
    print(f"数据集:     {data_path}")
    print(f"模型:       {args.model}")
    print(f"训练轮数:   {args.epochs}")
    print(f"批大小:     {args.batch}")
    print(f"图像尺寸:   {args.imgsz}")
    print(f"设备:       {args.device}")
    print("-" * 60)
    print("数据增强:")
    print(f"  mosaic:        {args.mosaic}")
    print(f"  mixup:         {args.mixup}")
    print(f"  copy_paste:    {args.copy_paste}")
    print(f"  copy_paste模式: {args.copy_paste_mode}")
    print(f"  auto_augment:  {args.auto_augment}")
    print(f"  erasing:       {args.erasing}")
    print(f"  hsv:           H={args.hsv_h} S={args.hsv_s} V={args.hsv_v}")
    print(f"  flipud:        {args.flipud}")
    print(f"  fliplr:        {args.fliplr}")
    print("-" * 60)
    print(f"输出目录:   {args.project}/{args.name}")
    print("=" * 60)

    from ultralytics import YOLO

    # 加载模型
    model = YOLO(args.model)

    # 构建训练参数
    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_momentum": args.warmup_momentum,
        "warmup_bias_lr": args.warmup_bias_lr,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "perspective": args.perspective,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "copy_paste_mode": args.copy_paste_mode,
        "auto_augment": args.auto_augment,
        "erasing": args.erasing,
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
        "patience": args.patience,
        "close_mosaic": args.close_mosaic,
        "amp": args.amp,
        "fraction": args.fraction,
        "overlap_mask": args.overlap_mask,
        "mask_ratio": args.mask_ratio,
        "dropout": args.dropout,
        "pretrained": args.pretrained,
        "verbose": args.verbose,
        "resume": args.resume if args.resume else False,
    }

    # 开始训练
    results = model.train(**train_args)

    print("\n" + "=" * 60)
    print("[OK] 训练完成！")
    print("=" * 60)
    print(f"最佳模型: {model.trainer.best}")
    print(f"输出目录: {model.trainer.save_dir}")
    print("\n推理示例:")
    print(f"  results = model.predict('path/to/image.png', conf=0.25)")
    print(f"\n导出 ONNX:")
    print(f"  model.export(format='onnx', imgsz={args.imgsz})")


if __name__ == "__main__":
    main()
