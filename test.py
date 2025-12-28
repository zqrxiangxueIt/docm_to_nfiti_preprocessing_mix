import os
import glob
import sys
import numpy as np
import time
from tqdm import tqdm

# 尝试导入必要库
try:
    import SimpleITK as sitk
except ImportError:
    print("错误: 缺少 SimpleITK。请运行: pip install SimpleITK")
    sys.exit(1)

# ================= 配置区域 =================
# 这里填你 main.py 中 DIRS["nifti"] 的实际路径
INPUT_NIFTI_DIR = r"E:\MedicalData_Pipeline_Output\02_NIfTI"

# 采样率：为了加快计算，每隔多少个像素采一个样 (100表示只取1%的数据计算分布)
# 如果内存充足且想要极致精确，可以设为 1 或 10
SAMPLE_RATE = 20


# ===========================================

def analyze_distribution():
    print(f"正在扫描文件夹: {INPUT_NIFTI_DIR}")
    # 递归查找所有 .nii.gz
    files = sorted(glob.glob(os.path.join(INPUT_NIFTI_DIR, "**", "*.nii.gz"), recursive=True))

    if not files:
        print(f"❌ 未找到任何 .nii.gz 文件。请检查路径是否正确。\n当前路径: {INPUT_NIFTI_DIR}")
        return

    print(f"✅ 发现 {len(files)} 个文件，开始统计 HU 分布...")
    print(f"ℹ️  采样率: 1/{SAMPLE_RATE} (仅读取部分像素以加速计算)")

    # 存储统计量的列表
    stats = {
        "min": [],  # 绝对最小值
        "max": [],  # 绝对最大值
        "p01": [],  # 1% 分位数 (排除极低噪点)
        "p99": [],  # 99% 分位数 (排除金属伪影/极高噪点)
        "p99_5": [],  # 99.5% 分位数 (关键参考值)
        "tissue_mean": []  # 组织平均值 (排除空气)
    }

    start_time = time.time()

    for file_path in tqdm(files, desc="Analyzing"):
        try:
            # 1. 读取图像
            img = sitk.ReadImage(file_path)
            arr = sitk.GetArrayFromImage(img)  # Numpy 数组 (Z, Y, X)

            # 2. 展平并采样
            arr_flat = arr.flatten()

            # 3. 基础统计
            # 记录绝对极值
            stats["min"].append(np.min(arr_flat))
            stats["max"].append(np.max(arr_flat))

            # 4. 采样后计算分位数 (大幅提升速度)
            # 使用 step 切片进行降采样
            sample_arr = arr_flat[::SAMPLE_RATE]

            stats["p01"].append(np.percentile(sample_arr, 1))
            stats["p99"].append(np.percentile(sample_arr, 99))
            stats["p99_5"].append(np.percentile(sample_arr, 99.5))

            # 5. 计算前景均值 (排除 -900 以下的空气背景，通常空气是 -1000)
            tissue_voxels = sample_arr[sample_arr > -900]
            if len(tissue_voxels) > 0:
                stats["tissue_mean"].append(np.mean(tissue_voxels))

        except Exception as e:
            print(f"\n⚠️ 读取失败 {os.path.basename(file_path)}: {e}")

    duration = time.time() - start_time

    # ================= 打印报告 =================
    if len(stats["max"]) == 0:
        print("未成功处理任何文件。")
        return

    avg_min = np.mean(stats["min"])
    avg_max = np.mean(stats["max"])
    avg_p01 = np.mean(stats["p01"])
    avg_p99 = np.mean(stats["p99"])
    avg_p99_5 = np.mean(stats["p99_5"])
    avg_tissue = np.mean(stats["tissue_mean"])

    print("\n" + "=" * 50)
    print(" 📊  数据集 HU 值分布统计报告")
    print("=" * 50)
    print(f" 📂 数据源: {INPUT_NIFTI_DIR}")
    print(f" 📄 文件数: {len(files)}")
    print(f" ⏱️ 耗时  : {duration:.2f} 秒")
    print("-" * 50)
    print(f"【绝对范围】(包含噪点/伪影)")
    print(f"  Avg Min Value : {avg_min:.2f}")
    print(f"  Avg Max Value : {avg_max:.2f}")
    print("-" * 50)
    print(f"【有效范围】(推荐用于 Clipping 的参考)")
    print(f"  1% 分位数 (下界参考)    : {avg_p01:.2f}")
    print(f"  99% 分位数 (上界参考)   : {avg_p99:.2f}")
    print(f"  99.5% 分位数 (保留高亮) : {avg_p99_5:.2f}  <-- 重点关注这个")
    print("-" * 50)
    print(f"【其他指标】")
    print(f"  前景组织均值 (> -900HU) : {avg_tissue:.2f}")
    print("=" * 50)

    # ================= 给出建议 =================
    suggested_min = int(avg_p01 // 10 * 10) - 10  # 向下取整留余量
    suggested_max = int(avg_p99_5 // 10 * 10) + 50  # 向上取整并加一点余量

    print("\n💡 针对 Swin UNETR 的参数建议：")
    print(f"建议修改 main.py 中的参数为：")
    print(f"HU_MIN = {max(suggested_min, -1000)}.0  (通常设为 -100 或 -50 即可)")
    print(f"HU_MAX = {suggested_max}.0    (这能覆盖绝大多数高亮血管)")
    print("=" * 50)


if __name__ == "__main__":
    analyze_distribution()