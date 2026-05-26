import os
import cv2
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from models.SD_PSFNet import SD_PSFNet as Net
from Loss import EdgeLoss, CharbonnierLoss, fftLoss
from warmup_scheduler import GradualWarmupScheduler
from Data import Rain100LDataset as data
from skimage.metrics import peak_signal_noise_ratio as PSNR
import logging
import time
from datetime import datetime

# 配置参数
lr = 1e-4
epochs = 800
ckpt = 50  # 检查点保存间隔
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logger(mod):
    """配置日志记录"""
    log_dir = os.path.join('logs', mod)
    os.makedirs(log_dir, exist_ok=True)

    # 设置日志格式
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


def save_checkpoint(model, epoch):
    """保存常规检查点"""
    model_out_path = os.path.join(save_path, f"{epoch}.pth")
    torch.save(model.state_dict(), model_out_path)
    logger.info(f"Model saved to {model_out_path}")


def save_best_checkpoint(model):
    """保存验证集最佳模型"""
    model_out_path = os.path.join(save_path, "BST.pth")
    torch.save(model.state_dict(), model_out_path)
    logger.info(f"Best Val Model saved to {model_out_path}")


def psnr(output_tensor, gt_tensor, y_channel=True):
    """计算PSNR（支持Y通道或RGB）"""

    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
        return tensor * std + mean

    def tensor_to_numpy(tensor):
        tensor = denormalize(tensor).clamp(0, 1)
        return (tensor * 255).cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)

    output_np = tensor_to_numpy(output_tensor)
    gt_np = tensor_to_numpy(gt_tensor)

    batch_psnr = []
    for o_img, g_img in zip(output_np, gt_np):
        if y_channel:
            o_yuv = cv2.cvtColor(o_img, cv2.COLOR_RGB2YCrCb)
            g_yuv = cv2.cvtColor(g_img, cv2.COLOR_RGB2YCrCb)
            o_data = o_yuv[:, :, 0]
            g_data = g_yuv[:, :, 0]
        else:
            o_data = o_img
            g_data = g_img
        batch_psnr.append(PSNR(g_data, o_data, data_range=255))
    return torch.tensor(batch_psnr).mean().item()


def train(training_data_loader, validate_data_loader, ckpt_path, finetune=False):
    logger.info(f"\n{'=' * 30} {mod} 训练开始 {'=' * 30}")
    model = Net().to(device)

    if finetune:
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"成功加载权重：{ckpt_path}")

    rainy_key, clear_key = 'Rainy', 'Clear'
    best_psnr = 0.0

    # 准备记录训练过程中的指标
    train_losses = []
    val_psnrs = []
    lr_rates = []

    # 优化器与调度器 - 修改后的部分
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs - warmup_epochs, eta_min=1e-6)
    # 增加multiplier至更高值
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1.0, total_epoch=warmup_epochs,
        after_scheduler=scheduler_cosine
    )

    # 混合精度训练
    scaler = torch.amp.GradScaler(init_scale=2. ** 14, growth_factor=2.0, backoff_factor=0.5, growth_interval=200)

    # 损失函数
    criterion_char = CharbonnierLoss()
    criterion_edge = EdgeLoss()
    criterion_fft = fftLoss()

    # 跟踪运行时间
    total_start_time = time.time()

    for epoch in range(epochs):
        start_time = time.time()

        # ==================== 训练阶段 ====================
        model.train()
        epoch_train_loss = []
        epoch_char_loss = []
        epoch_edge_loss = []
        epoch_fft_loss = []

        for batch_idx, batch in enumerate(tqdm(training_data_loader,
                                               desc=f'Train Epoch {epoch + 1}/{epochs}',
                                               ncols=100)):
            Clear = batch[clear_key].to(device)
            Drop = batch[rainy_key].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
                outputs = model(Drop)
                loss_char = torch.sum(torch.stack([criterion_char(outputs[j], Clear) for j in range(len(outputs))]))
                loss_edge = torch.sum(torch.stack([criterion_edge(outputs[j], Clear) for j in range(len(outputs))]))
                loss_fft = torch.sum(torch.stack([criterion_fft(outputs[j], Clear) for j in range(len(outputs))]))
                loss = loss_char + (0.05 * loss_edge) + (0.01 * loss_fft)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss.append(loss.item())
            epoch_char_loss.append(loss_char.item())
            epoch_edge_loss.append(loss_edge.item())
            epoch_fft_loss.append(loss_fft.item())

        # ==================== 验证阶段 ====================
        model.eval()
        psnr_val_rgb = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(validate_data_loader,
                                                   desc=f'Val Epoch {epoch + 1}',
                                                   ncols=100)):
                Clear = batch[clear_key].to(device)
                Drop = batch[rainy_key].to(device)
                outputs = model(Drop)

                # 计算PSNR
                for res, tar in zip(outputs[-1], Clear):
                    psnr_val_rgb.append(psnr(res.unsqueeze(0), tar.unsqueeze(0)))

        # 计算指标
        t_loss = np.mean(epoch_train_loss)
        char_loss = np.mean(epoch_char_loss)
        edge_loss = np.mean(epoch_edge_loss)
        fft_loss = np.mean(epoch_fft_loss)
        v_psnr = torch.tensor(psnr_val_rgb).mean().item()

        # 记录指标用于日志
        train_losses.append(t_loss)
        val_psnrs.append(v_psnr)

        # 修改后的学习率调度器步进策略
        if epoch < warmup_epochs:
            scheduler.step()  # 预热期间不需要传递指标
            logger.info(f"Warmup phase: {epoch + 1}/{warmup_epochs}")
        else:
            scheduler.step(metrics=v_psnr)  # 传递验证PSNR作为指标

        current_lr = optimizer.param_groups[0]['lr']
        lr_rates.append(current_lr)

        # 计算每个epoch的运行时间
        epoch_time = time.time() - start_time

        # 保存模型检查点
        if (epoch + 1) % ckpt == 0:
            save_checkpoint(model, epoch + 1)

        # 保存最佳模型
        if v_psnr > best_psnr and v_psnr != float('inf'):
            best_psnr = v_psnr
            save_best_checkpoint(model)

        # 写入日志
        logger.info(f"\nEpoch [{epoch + 1}/{epochs}] | "
                    f"Train Loss: {t_loss:.4f} | "
                    f"Val PSNR: {v_psnr:.2f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.1f}s")
        logger.info(f"Best Val PSNR: {best_psnr:.2f}")

        # 添加更详细的损失分解
        logger.info(f"Loss details - Char: {char_loss:.4f}, Edge: {edge_loss:.4f}, FFT: {fft_loss:.4f}")
        logger.info("=" * 80)

    # ==================== 训练结束 ====================
    total_time = time.time() - total_start_time
    logger.info(f"Training completed in {total_time / 3600:.2f} hours. Best validation PSNR: {best_psnr:.2f}")

    # 记录最终训练数据统计
    logger.info(f"Final training statistics:")
    logger.info(f"Final loss: {train_losses[-1]:.4f}")
    logger.info(f"Final validation PSNR: {val_psnrs[-1]:.2f}")
    logger.info(f"Best validation PSNR: {best_psnr:.2f}")


if __name__ == "__main__":
    mod = 'SD_PSFNet_Rain100H'  # 模型标识符
    save_path = f'weights/{mod}'
    os.makedirs(save_path, exist_ok=True)

    # 设置日志记录器
    logger = setup_logger(mod)

    # 记录训练配置
    logger.info(f"Model: {mod}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Total epochs: {epochs}")
    logger.info(f"Device: {device}")
    logger.info(f"Scheduler: ReduceLROnPlateau with warmup (multiplier=3.0, warmup_epochs=5)")

    # 数据集路径 (使用相对路径：点代表当前项目根目录)
    train_dir = r"./datasets/train/rain_data_train_Heavy"
    val_dir = r"./datasets/test/Rain100H"

    # 数据加载器
    train_dataset = data(root_dir=train_dir)
    val_dataset = data(root_dir=val_dir)

    # 权重路径 (使用相对路径)
    ckpt_path = r"./weights/Rain100H-sota.pth"


    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, num_workers=2, pin_memory=True)

    # 开始训练
    train(train_loader, val_loader, ckpt_path, finetune=False)
