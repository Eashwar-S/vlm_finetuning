import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

def load_jsonl(filepath: str) -> List[dict]:
    """Load JSONL file and return list of records"""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records

def load_ground_truth(filepath: str) -> Dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt_dict = {}
    for key, val in data.items():
        if "filename" in val and "ground_truth" in val:
            gt_dict[val["filename"]] = val["ground_truth"]
    return gt_dict

def compare_against_ground_truth(pro_file: str, flash_file: str, gt_file: str) -> Dict:
    """
    Compare teacher answers between Gemini 3 Pro/Preview and Gemini 3 Flash
    against the Ground Truth.
    """
    pro_data = load_jsonl(pro_file)
    flash_data = load_jsonl(flash_file)
    gt_dict = load_ground_truth(gt_file)
    
    # Create lookup by filename
    pro_dict = {record['filename']: record for record in pro_data}
    flash_dict = {record['filename']: record for record in flash_data}
    
    # Find common images
    common_filenames = set(pro_dict.keys()) & set(flash_dict.keys()) & set(gt_dict.keys())
    
    print(f"\n{'='*100}")
    print(f"COMPARISON: Gemini 3 Pro vs Gemini 3 Flash vs Ground Truth")
    print(f"{'='*100}\n")
    print(f"Total images in Pro dataset: {len(pro_data)}")
    print(f"Total images in Flash dataset: {len(flash_data)}")
    print(f"Total ground truth annotations: {len(gt_dict)}")
    print(f"Common images: {len(common_filenames)}\n")
    
    # Categories to compare
    categories = [
        "forest_fire_smoke_visible",
        "forest_fire_flames_visible",
        "confirm_uncontrolled_forest_fire",
        "fire_state",
        "fire_type",
        "fire_intensity",
        "fire_size",
        "fire_hotspots",
        "infrastructure_nearby",
        "people_nearby",
        "tree_vitality"
    ]
    
    # Track accuracy per category
    category_stats = defaultdict(lambda: {
        "total": 0,
        "pro_matches": 0,
        "flash_matches": 0,
    })
    
    for filename in common_filenames:
        pro_answer = pro_dict[filename]['teacher_answer']
        flash_answer = flash_dict[filename]['teacher_answer']
        gt_answer = gt_dict[filename]
        
        for category in categories:
            category_stats[category]["total"] += 1
            
            pro_value = pro_answer.get(category, "N/A")
            flash_value = flash_answer.get(category, "N/A")
            gt_value = gt_answer.get(category, "N/A")
            
            if pro_value == gt_value:
                category_stats[category]["pro_matches"] += 1
            if flash_value == gt_value:
                category_stats[category]["flash_matches"] += 1
    
    # Print overall summary
    print(f"\n{'='*100}")
    print(f"OVERALL SUMMARY (Accuracy wrt Ground Truth)")
    print(f"{'='*100}\n")
    
    overall_pro_matches = 0
    overall_flash_matches = 0
    overall_total = 0
    
    print(f"{'Category':<40} {'Pro Accuracy':<20} {'Flash Accuracy':<20}")
    print(f"{'-'*40} {'-'*20} {'-'*20}")
    
    for category in categories:
        stats = category_stats[category]
        pro_accuracy = (stats["pro_matches"] / stats["total"] * 100) if stats["total"] > 0 else 0
        flash_accuracy = (stats["flash_matches"] / stats["total"] * 100) if stats["total"] > 0 else 0
        
        overall_pro_matches += stats["pro_matches"]
        overall_flash_matches += stats["flash_matches"]
        overall_total += stats["total"]
        
        print(f"{category:<40} {pro_accuracy:>6.2f}% ({stats['pro_matches']}/{stats['total']})   {flash_accuracy:>6.2f}% ({stats['flash_matches']}/{stats['total']})")
    
    overall_pro_accuracy = (overall_pro_matches / overall_total * 100) if overall_total > 0 else 0
    overall_flash_accuracy = (overall_flash_matches / overall_total * 100) if overall_total > 0 else 0
    
    print(f"{'-'*40} {'-'*20} {'-'*20}")
    print(f"{'OVERALL':<40} {overall_pro_accuracy:>6.2f}% ({overall_pro_matches}/{overall_total})   {overall_flash_accuracy:>6.2f}% ({overall_flash_matches}/{overall_total})")
    
    return {
        "category_stats": dict(category_stats),
        "overall_pro_accuracy": overall_pro_accuracy,
        "overall_flash_accuracy": overall_flash_accuracy,
        "overall_pro_matches": overall_pro_matches,
        "overall_flash_matches": overall_flash_matches,
        "overall_total": overall_total,
        "common_images": len(common_filenames)
    }

if __name__ == "__main__":
    pro_file = "dataset/data/train_short_pro.jsonl"
    flash_file = "dataset/data/train_short_flash.jsonl"
    gt_file = "dataset/data/ground_truth.json"
    
    results = compare_against_ground_truth(pro_file, flash_file, gt_file)
    
    # Save results to JSON
    output_file = "dataset/data/comparison_with_gt_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed JSON results saved to: {output_file}")
    
    # Export to Excel
    try:
        import pandas as pd
        
        excel_file = "dataset/data/accuracy_wrt_ground_truth.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Sheet 1: Overall Summary
            summary_data = []
            for category, stats in results['category_stats'].items():
                pro_acc = (stats['pro_matches'] / stats['total'] * 100) if stats['total'] > 0 else 0
                flash_acc = (stats['flash_matches'] / stats['total'] * 100) if stats['total'] > 0 else 0
                
                summary_data.append({
                    'Category': category,
                    'Total Samples': stats['total'],
                    'Pro Matches': stats['pro_matches'],
                    'Flash Matches': stats['flash_matches'],
                    'Pro Accuracy (%)': round(pro_acc, 2),
                    'Flash Accuracy (%)': round(flash_acc, 2)
                })
            
            # Add overall row
            summary_data.append({
                'Category': 'OVERALL',
                'Total Samples': results['overall_total'],
                'Pro Matches': results['overall_pro_matches'],
                'Flash Matches': results['overall_flash_matches'],
                'Pro Accuracy (%)': round(results['overall_pro_accuracy'], 2),
                'Flash Accuracy (%)': round(results['overall_flash_accuracy'], 2)
            })
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Accuracy vs Ground Truth', index=False)
            
        print(f"Excel results saved to: {excel_file}")
        
    except ImportError:
        print("\nNote: pandas not installed. Install with 'pip install pandas openpyxl' to enable Excel export.")
    except Exception as e:
        print(f"\nError creating Excel file: {e}")
