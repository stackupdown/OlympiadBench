#!/usr/env python3
import argparse
import os
from huggingface_hub import snapshot_download
import tempfile

def download_model(model_name: str, save_base_dir):
    # 创建保存目录（如果不存在）
    save_path = os.path.join(save_base_dir, model_name)
    os.makedirs(save_path, exist_ok=True)

    # 使用临时缓存目录确保不污染系统缓存
    with tempfile.TemporaryDirectory() as tmp_cache:
        snapshot_download(
            repo_id=model_name,
            local_dir=save_path,
            local_dir_use_symlinks=False,  # 直接保存文件而非符号链接
            cache_dir=tmp_cache,          # 使用临时缓存目录
            resume_download=True,         # 支持断点续传
        )

    print(f"✅ 模型已保存到: {save_path}")
    return save_path

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='下载Hugging Face模型到指定目录')
    parser.add_argument('model_name', type=str, help='模型名称（如 "bert-base-uncased"）')
    parser.add_argument('--save_dir', type=str, default="./models",
                      help='保存目录（默认为当前目录下的models文件夹）')
    
    args = parser.parse_args()
    
    # 执行下载
    download_model(
        model_name=args.model_name,
        save_base_dir=args.save_dir
    )
