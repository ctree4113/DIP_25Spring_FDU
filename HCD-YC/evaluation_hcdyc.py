#!/usr/bin/env python3
"""
HCD-YC improved method evaluation system
Evaluate HCD-YC improved method on RTTS dataset following the original paper standards
"""

import os
import sys
import argparse
import torch
import json
from pathlib import Path
import pyiqa
import scipy
import subprocess

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eval_metrics import DehazeEvaluator


def setup_environment():
    """Setup evaluation environment"""
    print("Setting up HCD-YC evaluation environment...")
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("pyiqa available")
    print("scipy available")
    
    return device


def run_inference_if_needed(method_name, config_path, input_folder, output_folder):
    """Run inference if results don't exist"""
    
    if os.path.exists(output_folder) and len(os.listdir(output_folder)) > 0:
        print(f"{method_name} results exist: {output_folder}")
        return True
        
    print(f"{method_name} results missing, running inference...")
    
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        if 'stage1' in config_path.lower():
            cmd = [sys.executable, "inference_stage1.py", "--config", config_path]
        else:
            cmd = [sys.executable, "inference_stage2.py", "--config", config_path]
            
        subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        print(f"{method_name} inference completed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"{method_name} inference failed: {e}")
        return False


def prepare_rtts_dataset(rtts_path):
    """Prepare RTTS dataset"""
    if not os.path.exists(rtts_path):
        print(f"RTTS dataset path does not exist: {rtts_path}")
        print("Please download RESIDE dataset and set correct RTTS path")
        return False
    
    # Check for PASCAL VOC format structure
    jpeg_images_path = os.path.join(rtts_path, 'JPEGImages')
    imagesets_path = os.path.join(rtts_path, 'ImageSets', 'Main')
    
    if os.path.exists(jpeg_images_path) and os.path.exists(imagesets_path):
        print("Detected PASCAL VOC format RTTS dataset")
        
        # Count images in JPEGImages directory
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend([f for f in os.listdir(jpeg_images_path) if f.lower().endswith(ext)])
        
        # Check if test.txt exists
        test_txt_path = os.path.join(imagesets_path, 'test.txt')
        if os.path.exists(test_txt_path):
            with open(test_txt_path, 'r') as f:
                test_list = [line.strip() for line in f.readlines() if line.strip()]
            print(f"Found test list with {len(test_list)} entries")
        
        print(f"RTTS dataset ready, found {len(image_files)} images in JPEGImages directory")
        return len(image_files) > 0
    else:
        # Fallback to simple directory structure
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend([f for f in os.listdir(rtts_path) if f.lower().endswith(ext)])
            
        print(f"RTTS dataset ready, found {len(image_files)} images")
        return len(image_files) > 0


def main():
    parser = argparse.ArgumentParser(description='Run HCD-YC improved method evaluation')
    parser.add_argument('--rtts_path', type=str, default='datasets/RTTS', 
                        help='RTTS dataset path')
    parser.add_argument('--eval_input', action='store_true',
                        help='Whether to evaluate input hazy images')
    parser.add_argument('--output_dir', type=str, default='evaluation_outputs',
                        help='Evaluation results output directory')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HCD-YC Improved Method Evaluation System")
    print("=" * 60)
    
    # 1. Setup environment
    device = setup_environment()
    
    # 2. Prepare dataset
    if not prepare_rtts_dataset(args.rtts_path):
        print("Dataset preparation failed, exiting evaluation")
        return
    
    # 3. Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 4. Define evaluation methods and configurations
    evaluation_configs = {
        'Input_Hazy': {
            'description': 'Input hazy images (reference)',
            'result_folder': args.rtts_path,
            'skip_inference': True
        },
        'HCD_YC_Stage2': {
            'description': 'HCD-YC improved method - Stage2 (dehazing)',
            'config': 'configs/inference/stage2_evaluation.yaml',
            'result_folder': f'{args.output_dir}/hcd_yc_stage2_results',
        }
    }
    
    # 5. Run inference (if needed)
    print("\n" + "=" * 40)
    print("Step 1: Generate inference results")
    print("=" * 40)
    
    available_methods = {}
    
    for method_name, config in evaluation_configs.items():
        if config.get('skip_inference', False):
            if args.eval_input or method_name != 'Input_Hazy':
                available_methods[method_name] = config
                print(f"{method_name}: {config['description']}")
            continue
            
        # Check if config file exists
        config_file = config.get('config')
        if config_file and os.path.exists(config_file):
            success = run_inference_if_needed(
                method_name, 
                config_file,
                args.rtts_path,
                config['result_folder']
            )
            
            if success:
                available_methods[method_name] = config
                print(f"{method_name}: {config['description']}")
            else:
                print(f"{method_name}: inference failed, skipping evaluation")
        else:
            print(f"{method_name}: config file not found {config_file}")
    
    if not available_methods:
        print("No available evaluation methods, exiting")
        return
    
    # 6. Run evaluation
    print("\n" + "=" * 40) 
    print("Step 2: Run image quality assessment")
    print("=" * 40)
    
    evaluator = DehazeEvaluator(device=device)
    
    # Prepare method folder dictionary
    method_folders = {}
    for method_name, config in available_methods.items():
        folder_path = config['result_folder']
        if os.path.exists(folder_path):
            method_folders[method_name] = folder_path
        else:
            print(f"Skip {method_name}: result folder not found {folder_path}")
    
    if not method_folders:
        print("No valid result folders found")
        return
    
    # Run comparative evaluation
    output_file = os.path.join(args.output_dir, 'hcdyc_evaluation_results.json')
    
    try:
        print(f"\nStarting evaluation of {len(method_folders)} methods...")
        comparison_results = evaluator.compare_methods(
            method_folders,
            output_file=output_file
        )
        
        # 7. Generate detailed report
        print("\n" + "=" * 40)
        print("Step 3: Generate evaluation report")
        print("=" * 40)
        
        generate_detailed_report(comparison_results, available_methods, args.output_dir)
        
        print(f"\nHCD-YC evaluation completed! Results saved in: {args.output_dir}")
        print(f"Detailed data: {output_file}")
        print(f"Comparison table: {output_file.replace('.json', '_table.txt')}")
        print(f"Detailed report: {os.path.join(args.output_dir, 'hcdyc_evaluation_report.md')}")
        
    except Exception as e:
        print(f"Evaluation process error: {e}")
        import traceback
        traceback.print_exc()


def generate_detailed_report(comparison_results, method_configs, output_dir):
    """Generate detailed evaluation report"""
    
    report_file = os.path.join(output_dir, 'hcdyc_evaluation_report.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# HCD-YC Improved Method Evaluation Report\n\n")
        f.write("## Evaluation Overview\n\n")
        f.write("This report evaluates the HCD-YC improved method with three main innovations on RTTS dataset.\n\n")
        
        f.write("## Evaluation Metrics\n\n")
        f.write("Using the following no-reference image quality assessment metrics:\n\n")
        f.write("- **FADE** (lower better): Fog density assessment\n")
        f.write("- **Q-Align** (higher better): Large model-based visual quality assessment\n")
        f.write("- **LIQE** (higher better): Vision-language correspondence blind image quality assessment\n") 
        f.write("- **CLIPIQA** (higher better): CLIP-based image quality assessment\n")
        f.write("- **ManIQA** (higher better): Multi-dimensional attention network quality assessment\n")
        f.write("- **MUSIQ** (higher better): Multi-scale image quality assessment\n")
        f.write("- **BRISQUE** (lower better): Spatial domain no-reference quality assessment\n\n")
        
        f.write("## Quantitative Results\n\n")
        f.write("| Method | FADE | Q-Align | LIQE | CLIPIQA | ManIQA | MUSIQ | BRISQUE |\n")
        f.write("|--------|------|---------|------|---------|--------|-------|----------|\n")
        
        for method_name, results in comparison_results.items():
            f.write(f"| {method_name} |")
            
            metrics = ['fade', 'qalign', 'liqe', 'clipiqa', 'maniqa', 'musiq', 'brisque']
            for metric in metrics:
                if metric in results:
                    value = results[metric]['mean']
                    f.write(f" {value:.4f} |")
                else:
                    f.write(" - |")
            f.write("\n")
        
        f.write("\n## Method Description\n\n")
        for method_name, config in method_configs.items():
            if method_name in comparison_results:
                f.write(f"### {method_name}\n")
                f.write(f"{config['description']}\n\n")

if __name__ == "__main__":
    main() 