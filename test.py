import os
import time
import torch
import numpy as np
import cv2
from PIL import Image
from Data import RealRainDataset as TestDataset
from models.SD_PSFNet import SD_PSFNet as Net
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 反归一化参数
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)


def denormalize(tensor):
    """将归一化后的张量反归一化到[0,1]范围"""
    return torch.clamp((tensor * std + mean), 0, 1)


def process_patches(model, input_tensor, patch_size=256, overlap=32):
    """分块处理大尺寸图像"""
    _, C, H, W = input_tensor.shape

    # 计算步长和填充量
    stride = patch_size - overlap
    pad_h = (patch_size - (H % stride)) % stride
    pad_w = (patch_size - (W % stride)) % stride

    # 镜像填充
    padded = torch.nn.functional.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, H_pad, W_pad = padded.shape

    # 初始化输出和权重矩阵
    output = torch.zeros_like(padded)
    weight = torch.zeros_like(padded)

    # 滑动窗口处理
    for y in range(0, H_pad - patch_size + 1, stride):
        for x in range(0, W_pad - patch_size + 1, stride):
            patch = padded[:, :, y:y + patch_size, x:x + patch_size]
            with torch.no_grad():
                pred = model(patch)[-1]
            output[:, :, y:y + patch_size, x:x + patch_size] += pred
            weight[:, :, y:y + patch_size, x:x + patch_size] += 1

    # 合并结果并裁剪
    output /= weight
    return output[:, :, :H, :W]


def calculate_metrics(output, target,y_channel=0):
    """计算RGB空间的PSNR/SSIM"""
    # 转换为numpy数组
    output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    target_np = target.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    # 转换到0-255范围
    output_uint8 = (output_np * 255).astype(np.uint8)
    target_uint8 = (target_np * 255).astype(np.uint8)

    if y_channel:
        o_yuv = cv2.cvtColor(output_uint8, cv2.COLOR_RGB2YCrCb)
        g_yuv = cv2.cvtColor(target_uint8, cv2.COLOR_RGB2YCrCb)
        o_data = o_yuv[:, :, 0]
        g_data = g_yuv[:, :, 0]
    else:
        o_data = output_uint8
        g_data = target_uint8

    # 计算PSNR
    psnr_val = psnr(g_data, o_data, data_range=255)

    # 计算SSIM（多通道版本）
    ssim_val = ssim(target_uint8, output_uint8,
                    data_range=255,
                    channel_axis=2,
                    multichannel=True)

    return psnr_val, ssim_val


def run_test_patch_metrics(test_dir, ckpt_path, patch_size=512, overlap=64):
    print("\n" + "=" * 30 + " 初始化模型 " + "=" * 30)
    model = Net().to(device)

    try:
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"成功加载权重：{ckpt_path}")
    except Exception as e:
        print(f"模型加载失败：{str(e)}")
        return

    model.eval()
    dataset = TestDataset(root_dir=test_dir, split='test')
    print(f"找到测试样本：{len(dataset)} 张")

    output_dir = "test_results_patch"
    os.makedirs(output_dir, exist_ok=True)

    # 初始化统计量
    total_psnr, total_ssim = 0.0, 0.0
    total_time = 0.0

    for idx in range(len(dataset)):
        data = dataset[idx]
        input_tensor = data['Rainy'].unsqueeze(0).to(device)
        gt_tensor = data['Clear'].unsqueeze(0).to(device)

        # 分块处理
        start_time = time.time()
        output_tensor = process_patches(model, input_tensor, patch_size, overlap)
        infer_time = time.time() - start_time
        total_time += infer_time

        # 反归一化
        output_denorm = denormalize(output_tensor)
        gt_denorm = denormalize(gt_tensor)

        # 计算指标
        psnr_val, ssim_val = calculate_metrics(output_denorm, gt_denorm)

        # 累加统计
        total_psnr += psnr_val
        total_ssim += ssim_val

        # 保存结果
        output_np = (output_denorm.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(output_np).save(os.path.join(output_dir, f"result_{idx}.png"))

        print(f"图像 {idx + 1}/{len(dataset)} | "
              f"时间: {infer_time:.1f}s | "
              f"PSNR: {psnr_val:.2f} | "
              f"SSIM: {ssim_val:.4f} | ")

    # 打印最终结果
    avg_psnr = total_psnr / len(dataset)
    avg_ssim = total_ssim / len(dataset)
    avg_time = total_time / len(dataset)

    print("\n" + "=" * 30 + " 最终结果 " + "=" * 30)
    print(f"平均推理时间: {avg_time:.1f}s/张")
    print(f"PSNR (RGB): {avg_psnr:.2f} dB")
    print(f"SSIM (RGB): {avg_ssim:.4f}")


if __name__ == '__main__':
    # 测试集路径 (使用相对路径：./ 代表当前文件夹)
    test_dir = r"./datasets/test/Realrain-1k-L"
    
    # 预训练权重路径 (使用相对路径)
    ckpt_path = r"./weights/Realrain-1k-H-sota.pth"

    # 建议参数配置：
    # 1080p图像：patch_size=512, overlap=64
    # 4K图像：patch_size=1024, overlap=128
    run_test_patch_metrics(test_dir, ckpt_path, patch_size=128, overlap=64)