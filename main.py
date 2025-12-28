import os
import sys
import glob
import subprocess
import logging
import time
import traceback
import shutil
import csv
import re
from pathlib import Path


# ==============================================================================
#                                 1. 库自动修复
# ==============================================================================
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except:
        print(f"自动安装 {package} 失败，请手动安装。")


try:
    import nibabel
except ImportError:
    print("正在安装必要库 nibabel ...")
    install_package("nibabel")

try:
    import SimpleITK as sitk
    import torch
    import numpy as np
    from tqdm import tqdm
    from monai.transforms import (
        LoadImaged, EnsureChannelFirstd, Orientationd,
        ScaleIntensityRanged, Compose, SaveImage
    )
    from monai.data import Dataset, DataLoader
except ImportError as e:
    print(f"缺失必要库: {e}")
    sys.exit(1)

# ==============================================================================
#                                 2. 全局配置
# ==============================================================================

DCM2NIIX_EXE = r"E:\dcm2niix_win\dcm2niix.exe"
DCMDJPEG_EXE = r"E:\dcmtk-3.6.9-win64-dynamic\bin\dcmdjpeg.exe"

INPUT_ROOT = r"E:\MedicalData_Raw"
PROJECT_OUTPUT_ROOT = r"E:\MedicalData_Pipeline_Output"

HU_MIN = -50.0
HU_MAX = 800
TARGET_SPACING = [1.0, 1.0, 1.0]

# 【重要】增量模式下，通常设为 False。
# 只有当你修改了代码逻辑想彻底重跑时，才手动删掉输出文件夹或设为 True
OVERWRITE = False

RUN_STEPS = {
    "1_unzip": True,
    "2_convert": True,
    "3_clip": True,
    "4_resample": True,  # <--- 已升级为“增量追加模式”
    "5_stats": True  # 每次都会重新计算全局均值
}

DIRS = {
    "raw": INPUT_ROOT,
    "unzipped": os.path.join(PROJECT_OUTPUT_ROOT, "01_Uncompressed"),
    "nifti": os.path.join(PROJECT_OUTPUT_ROOT, "02_NIfTI"),
    "clipped": os.path.join(PROJECT_OUTPUT_ROOT, "03_Clipped_HU"),
    "final": os.path.join(PROJECT_OUTPUT_ROOT, "04_Resampled_1mm"),
}


# ==============================================================================
#                                 3. 工具函数
# ==============================================================================

def setup_logging():
    if not os.path.exists(PROJECT_OUTPUT_ROOT): os.makedirs(PROJECT_OUTPUT_ROOT)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(PROJECT_OUTPUT_ROOT, "pipeline_log.txt"), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def run_command(cmd_list):
    try:
        return subprocess.run(cmd_list, capture_output=True, text=True, errors='replace')
    except Exception as e:
        logging.error(f"CMD Error: {e}")
        return None


def sanitize_name(name):
    clean = re.sub(r'[^\x00-\x7F]+', '', name).strip('-_ .')
    return clean if clean else "Unknown_ID"


# ==============================================================================
#                                 4. 步骤实现
# ==============================================================================

def step_1_unzip():
    logging.info("\n>>> [Step 1] Unzip (Incremental)...")
    src_root = DIRS["raw"]
    dst_root = DIRS["unzipped"]
    count = 0
    for root, dirs, files in os.walk(src_root):
        dcm_files = [f for f in files if f.lower().endswith(('.dcm', '.ima'))]
        if not dcm_files: continue

        clean_rel = os.sep.join([sanitize_name(p) for p in os.path.relpath(root, src_root).split(os.sep)])
        target_dir = os.path.join(dst_root, clean_rel)
        if not os.path.exists(target_dir): os.makedirs(target_dir)

        for f in dcm_files:
            clean_f = sanitize_name(f)
            if not clean_f.lower().endswith('.dcm'): clean_f += os.path.splitext(f)[1]
            src = os.path.abspath(os.path.join(root, f))
            dst = os.path.abspath(os.path.join(target_dir, clean_f))

            # 增量检查
            if not OVERWRITE and os.path.exists(dst): continue

            res = run_command([DCMDJPEG_EXE, src, dst])
            if res is None or res.returncode != 0:
                try:
                    shutil.copy2(src, dst)
                except:
                    pass
            count += 1
    logging.info(f"Step 1 Processed: {count}")


def step_2_convert():
    logging.info("\n>>> [Step 2] Convert (Incremental)...")
    src_root = DIRS["unzipped"]
    dst_root = DIRS["nifti"]
    base_cmd = [DCM2NIIX_EXE, "-f", "%i_%p_%t_%s", "-z", "y", "-b", "y", "-v", "n"]

    for root, dirs, files in os.walk(src_root):
        is_dcm_dir = False
        if files:
            for f in files[:3]:
                if f.lower().endswith(('.dcm', '.ima')) or open(os.path.join(root, f), 'rb').read(132)[
                                                           128:132] == b'DICM':
                    is_dcm_dir = True;
                    break

        if is_dcm_dir:
            rel = os.path.relpath(root, src_root)
            out_dir = os.path.join(dst_root, rel)
            # 增量检查：如果文件夹存在且里面有 .nii.gz，跳过
            if not OVERWRITE and os.path.exists(out_dir) and glob.glob(os.path.join(out_dir, "*.nii.gz")): continue
            if not os.path.exists(out_dir): os.makedirs(out_dir)
            run_command(base_cmd + ["-o", os.path.abspath(out_dir), os.path.abspath(root)])


def step_3_clip():
    logging.info("\n>>> [Step 3] HU Clipping (Incremental)...")
    src_root = DIRS["nifti"]
    dst_root = DIRS["clipped"]

    image_paths = sorted(glob.glob(os.path.join(src_root, "**", "*.nii.gz"), recursive=True))
    if not image_paths: return

    process_list = []
    for p in image_paths:
        filename = os.path.basename(p)
        rel_dir = os.path.dirname(os.path.relpath(p, src_root))
        target_file = os.path.join(dst_root, rel_dir, filename)

        # 增量检查：文件存在且大小正常，跳过
        if not OVERWRITE and os.path.exists(target_file) and os.path.getsize(target_file) > 1024:
            continue
        process_list.append(p)

    logging.info(f"New files to clip: {len(process_list)}")
    if not process_list: return

    ds = Dataset(data=[{"image": i} for i in process_list], transform=Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX, b_min=HU_MIN, b_max=HU_MAX, clip=True)
    ]))

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    saver = SaveImage(output_dir=dst_root, output_postfix="", output_ext=".nii.gz", separate_folder=False,
                      print_log=False)

    for batch_data in tqdm(loader, desc="Clipping"):
        try:
            img_tensor = batch_data["image"]
            filename = img_tensor.meta["filename_or_obj"]
            if isinstance(filename, list): filename = filename[0]

            rel_dir_path = os.path.dirname(os.path.relpath(filename, src_root))
            target_folder = os.path.join(dst_root, rel_dir_path)

            if not os.path.exists(target_folder): os.makedirs(target_folder)
            saver.output_dir = target_folder
            saver(img_tensor[0])
        except Exception as e:
            logging.error(f"Error: {e}")


# --- 【关键升级】 Step 4: 智能追加模式 ---
def step_4_resample():
    logging.info("\n>>> [Step 4] Resample (Smart Append)...")
    src_root = DIRS["clipped"]
    dst_root = DIRS["final"]
    if not os.path.exists(dst_root): os.makedirs(dst_root)

    csv_path = os.path.join(dst_root, "name_mapping.csv")

    # 1. 读取现有的映射表，防止序号重排
    existing_map = {}  # key=原始相对路径, value=case_xxx.nii.gz
    max_id = 0

    if os.path.exists(csv_path) and not OVERWRITE:
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_map[row["Rel Path"]] = row["New Name"]
                    # 提取数字: case_005.nii.gz -> 5
                    try:
                        curr_id = int(re.search(r'case_(\d+)', row["New Name"]).group(1))
                        if curr_id > max_id: max_id = curr_id
                    except:
                        pass
            logging.info(f"已加载现有映射表: {len(existing_map)} 条记录, 最大ID: {max_id}")
        except Exception as e:
            logging.warning(f"读取映射表失败，将重建: {e}")

    # 2. 扫描当前所有文件
    image_paths = sorted(glob.glob(os.path.join(src_root, "**", "*.nii.gz"), recursive=True))

    final_mapping = []  # 存储最终的完整映射
    files_to_process = []  # 需要执行重采样的文件

    for img_path in image_paths:
        rel_path = os.path.relpath(img_path, src_root)

        if rel_path in existing_map and not OVERWRITE:
            # 旧文件：保持原来的名字
            assigned_name = existing_map[rel_path]

            # 只有当文件在硬盘上不见了，才需要重新跑
            if not os.path.exists(os.path.join(dst_root, assigned_name)):
                files_to_process.append((img_path, assigned_name))

            final_mapping.append([assigned_name, rel_path, img_path])
        else:
            # 新文件：分配新的ID (Max + 1)
            max_id += 1
            new_name = f"case_{max_id:03d}.nii.gz"
            files_to_process.append((img_path, new_name))
            final_mapping.append([new_name, rel_path, img_path])

    print(f"检测到 {len(image_paths)} 个文件。其中 {len(files_to_process)} 个需要处理/新增。")

    # 3. 处理文件
    for img_path, new_name in tqdm(files_to_process, desc="Resampling"):
        out_path = os.path.join(dst_root, new_name)
        try:
            itk_img = sitk.ReadImage(img_path)
            res = sitk.Resample(itk_img,
                                [int(round(s * sp / 1.0)) for s, sp in zip(itk_img.GetSize(), itk_img.GetSpacing())],
                                sitk.Transform(), sitk.sitkBSpline,
                                itk_img.GetOrigin(), [1.0, 1.0, 1.0],
                                itk_img.GetDirection(), HU_MIN, itk_img.GetPixelID())
            sitk.WriteImage(res, out_path)
        except Exception as e:
            logging.error(f"Fail: {img_path} - {e}")

    # 4. 更新 CSV (全量写入)
    # 按 New Name 排序一下，好看点
    final_mapping.sort(key=lambda x: x[0])

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["New Name", "Rel Path", "Full Path"])
        writer.writerows(final_mapping)

    logging.info(f"Step 4 Done. Mapping updated.")


def step_5_stats():
    logging.info("\n>>> [Step 5] Stats (Global Recalculate)...")
    data_dir = DIRS["final"]
    images = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
    if not images: return

    # 统计必须每次基于所有文件重算
    ds = Dataset(data=[{"image": i} for i in images],
                 transform=Compose([LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"])]))
    loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)

    mean_s, var_s, voxels = 0.0, 0.0, 0
    for b in tqdm(loader, desc="Calc Stats"):
        d = b["image"].double()
        mean_s += torch.sum(d).item()
        var_s += torch.sum(d ** 2).item()
        voxels += d.numel()

    gm = mean_s / voxels
    gs = ((var_s / voxels) - (gm ** 2)) ** 0.5
    logging.info("=" * 40)
    logging.info(f"Global Mean: {gm:.4f}")
    logging.info(f"Global Std : {gs:.4f}")
    logging.info("=" * 40)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    setup_logging()
    if RUN_STEPS["1_unzip"]: step_1_unzip()
    if RUN_STEPS["2_convert"]: step_2_convert()
    if RUN_STEPS["3_clip"]: step_3_clip()
    if RUN_STEPS["4_resample"]: step_4_resample()
    if RUN_STEPS["5_stats"]: step_5_stats()