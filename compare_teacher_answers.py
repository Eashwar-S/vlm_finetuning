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

def print_category_table(category_name: str, preview_dict: dict, flash_dict: dict, common_filenames: set):
    """Print detailed table for a specific category"""
    print(f"\n{'='*100}")
    print(f"CATEGORY: {category_name}")
    print(f"{'='*100}\n")
    
    # Collect all values for this category
    preview_values = []
    flash_values = []
    matches = 0
    mismatches = 0
    
    value_pair_counts = Counter()
    
    for filename in common_filenames:
        preview_val = preview_dict[filename]['teacher_answer'].get(category_name, "N/A")
        flash_val = flash_dict[filename]['teacher_answer'].get(category_name, "N/A")
        
        preview_values.append(preview_val)
        flash_values.append(flash_val)
        
        if preview_val == flash_val:
            matches += 1
        else:
            mismatches += 1
        
        value_pair_counts[(preview_val, flash_val)] += 1
    
    # Count value distributions
    preview_counter = Counter(preview_values)
    flash_counter = Counter(flash_values)
    
    # Print summary statistics
    total = len(common_filenames)
    accuracy = (matches / total * 100) if total > 0 else 0
    
    print(f"Summary Statistics:")
    print(f"  Total samples: {total}")
    print(f"  Matches (Agreement): {matches} ({matches/total*100:.1f}%)")
    print(f"  Mismatches (Disagreement): {mismatches} ({mismatches/total*100:.1f}%)")
    print(f"  Accuracy: {accuracy:.2f}%\n")
    
    # Print value distribution comparison
    all_values = sorted(set(preview_counter.keys()) | set(flash_counter.keys()))
    
    print(f"Value Distribution Comparison:")
    print(f"  {'Value':<40} {'Preview Count':<15} {'Flash Count':<15} {'Difference':<15}")
    print(f"  {'-'*40} {'-'*15} {'-'*15} {'-'*15}")
    
    for value in all_values:
        preview_count = preview_counter.get(value, 0)
        flash_count = flash_counter.get(value, 0)
        diff = flash_count - preview_count
        diff_str = f"{diff:+d}" if diff != 0 else "0"
        print(f"  {value:<40} {preview_count:<15} {flash_count:<15} {diff_str:<15}")
    
    # Print agreement/disagreement matrix
    print(f"\nAgreement/Disagreement Matrix:")
    print(f"  (Preview → Flash)")
    header = "Preview \\ Flash"
    print(f"  {header:<40} ", end="")
    
    # Get unique values for matrix headers
    unique_values = sorted(set(preview_values + flash_values))
    
    # Truncate value names for display
    def truncate(s, max_len=12):
        return s[:max_len-2] + '..' if len(s) > max_len else s
    
    for val in unique_values:
        print(f"{truncate(val):<14}", end="")
    print()
    print(f"  {'-'*40} " + '-'*14*len(unique_values))
    
    for preview_val in unique_values:
        print(f"  {truncate(preview_val):<40} ", end="")
        for flash_val in unique_values:
            count = value_pair_counts.get((preview_val, flash_val), 0)
            if count > 0:
                if preview_val == flash_val:
                    print(f"\033[92m{count:<14}\033[0m", end="")  # Green for matches
                else:
                    print(f"\033[91m{count:<14}\033[0m", end="")  # Red for mismatches
            else:
                print(f"{'-':<14}", end="")
        print()
    
    # Print top disagreements
    if mismatches > 0:
        print(f"\nTop Disagreement Patterns:")
        disagreement_pairs = [(pair, count) for pair, count in value_pair_counts.items() if pair[0] != pair[1]]
        disagreement_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, ((prev_val, flash_val), count) in enumerate(disagreement_pairs[:5], 1):
            pct = (count / total * 100)
            print(f"  {i}. Preview: '{prev_val}' → Flash: '{flash_val}' ({count} cases, {pct:.1f}%)")

def compare_teacher_answers(preview_file: str, flash_file: str) -> Dict:
    """
    Compare teacher answers between Gemini 3 Preview and Gemini 3 Flash
    
    Args:
        preview_file: Path to train_short.jsonl (Gemini 3 Preview)
        flash_file: Path to train_short_flash.jsonl (Gemini 3 Flash)
    
    Returns:
        Dictionary with comparison results
    """
    preview_data = load_jsonl(preview_file)
    flash_data = load_jsonl(flash_file)
    
    # Create lookup by filename
    preview_dict = {record['filename']: record for record in preview_data}
    flash_dict = {record['filename']: record for record in flash_data}
    
    # Find common images
    common_filenames = set(preview_dict.keys()) & set(flash_dict.keys())
    
    print(f"\n{'='*100}")
    print(f"COMPARISON: Gemini 3 Preview vs Gemini 3 Flash")
    print(f"{'='*100}\n")
    print(f"Total images in Preview dataset: {len(preview_data)}")
    print(f"Total images in Flash dataset: {len(flash_data)}")
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
        "matches": 0, 
        "mismatches": [],
        "value_distributions": {"preview": {}, "flash": {}},
        "confusion_matrix": {}
    })
    
    for filename in common_filenames:
        preview_answer = preview_dict[filename]['teacher_answer']
        flash_answer = flash_dict[filename]['teacher_answer']
        
        for category in categories:
            category_stats[category]["total"] += 1
            
            preview_value = preview_answer.get(category, "N/A")
            flash_value = flash_answer.get(category, "N/A")
            
            # Track value distributions
            category_stats[category]["value_distributions"]["preview"][preview_value] = \
                category_stats[category]["value_distributions"]["preview"].get(preview_value, 0) + 1
            category_stats[category]["value_distributions"]["flash"][flash_value] = \
                category_stats[category]["value_distributions"]["flash"].get(flash_value, 0) + 1
            
            # Track confusion matrix
            pair_key = f"{preview_value} → {flash_value}"
            category_stats[category]["confusion_matrix"][pair_key] = \
                category_stats[category]["confusion_matrix"].get(pair_key, 0) + 1
            
            if preview_value == flash_value:
                category_stats[category]["matches"] += 1
            else:
                category_stats[category]["mismatches"].append({
                    "filename": filename,
                    "preview": preview_value,
                    "flash": flash_value
                })
    
    # Print detailed tables for each category
    for category in categories:
        print_category_table(category, preview_dict, flash_dict, common_filenames)
    
    # Print overall summary
    print(f"\n{'='*100}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*100}\n")
    
    overall_matches = 0
    overall_total = 0
    
    print(f"{'Category':<45} {'Accuracy':<15} {'Matches':<15} {'Mismatches':<15}")
    print(f"{'-'*45} {'-'*15} {'-'*15} {'-'*15}")
    
    for category in categories:
        stats = category_stats[category]
        accuracy = (stats["matches"] / stats["total"] * 100) if stats["total"] > 0 else 0
        overall_matches += stats["matches"]
        overall_total += stats["total"]
        
        print(f"{category:<45} {accuracy:>6.2f}% {stats['matches']:>14} {len(stats['mismatches']):>14}")
    
    overall_accuracy = (overall_matches / overall_total * 100) if overall_total > 0 else 0
    print(f"{'-'*45} {'-'*15} {'-'*15} {'-'*15}")
    print(f"{'OVERALL':<45} {overall_accuracy:>6.2f}% {overall_matches:>14} {overall_total - overall_matches:>14}")
    
    return {
        "category_stats": dict(category_stats),
        "overall_accuracy": overall_accuracy,
        "overall_matches": overall_matches,
        "overall_total": overall_total,
        "common_images": len(common_filenames)
    }

if __name__ == "__main__":
    preview_file = "dataset/data/train_short.jsonl"
    flash_file = "dataset/data/train_short_flash.jsonl"
    
    results = compare_teacher_answers(preview_file, flash_file)
    
    # Save results to JSON
    output_file = "dataset/data/comparison_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Export to Excel
    try:
        import pandas as pd
        
        excel_file = "dataset/data/comparison_results.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Sheet 1: Overall Summary
            summary_data = []
            for category, stats in results['category_stats'].items():
                accuracy = (stats['matches'] / stats['total'] * 100) if stats['total'] > 0 else 0
                summary_data.append({
                    'Category': category,
                    'Total Samples': stats['total'],
                    'Matches': stats['matches'],
                    'Mismatches': len(stats['mismatches']),
                    'Accuracy (%)': round(accuracy, 2)
                })
            
            # Add overall row
            summary_data.append({
                'Category': 'OVERALL',
                'Total Samples': results['overall_total'],
                'Matches': results['overall_matches'],
                'Mismatches': results['overall_total'] - results['overall_matches'],
                'Accuracy (%)': round(results['overall_accuracy'], 2)
            })
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Overall Summary', index=False)
            
            # Sheet 2: Value Distributions for each category
            for category, stats in results['category_stats'].items():
                if len(category) > 31:  # Excel sheet name limit
                    sheet_name = category[:28] + '...'
                else:
                    sheet_name = category
                
                # Create value distribution comparison
                preview_dist = stats['value_distributions']['preview']
                flash_dist = stats['value_distributions']['flash']
                
                all_values = sorted(set(list(preview_dist.keys()) + list(flash_dist.keys())))
                
                dist_data = []
                for value in all_values:
                    prev_count = preview_dist.get(value, 0)
                    flash_count = flash_dist.get(value, 0)
                    diff = flash_count - prev_count
                    
                    dist_data.append({
                        'Value': value,
                        'Preview Count': prev_count,
                        'Flash Count': flash_count,
                        'Difference': diff,
                        'Preview %': round(prev_count / stats['total'] * 100, 1) if stats['total'] > 0 else 0,
                        'Flash %': round(flash_count / stats['total'] * 100, 1) if stats['total'] > 0 else 0
                    })
                
                df_dist = pd.DataFrame(dist_data)
                df_dist.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Sheet 3: Confusion Matrix Summary
            confusion_data = []
            for category, stats in results['category_stats'].items():
                for pair, count in stats['confusion_matrix'].items():
                    preview_val, flash_val = pair.split(' → ')
                    is_match = preview_val == flash_val
                    
                    confusion_data.append({
                        'Category': category,
                        'Preview Value': preview_val,
                        'Flash Value': flash_val,
                        'Count': count,
                        'Percentage': round(count / stats['total'] * 100, 1) if stats['total'] > 0 else 0,
                        'Match': 'Yes' if is_match else 'No'
                    })
            
            df_confusion = pd.DataFrame(confusion_data)
            df_confusion.to_excel(writer, sheet_name='Confusion Matrix', index=False)
            
            # Sheet 4: Detailed Mismatches
            mismatch_data = []
            for category, stats in results['category_stats'].items():
                for mismatch in stats['mismatches']:
                    mismatch_data.append({
                        'Category': category,
                        'Filename': mismatch['filename'],
                        'Preview Value': mismatch['preview'],
                        'Flash Value': mismatch['flash']
                    })
            
            if mismatch_data:
                df_mismatches = pd.DataFrame(mismatch_data)
                df_mismatches.to_excel(writer, sheet_name='Detailed Mismatches', index=False)
        
        print(f"Excel results saved to: {excel_file}")
        print(f"\nExcel file contains:")
        print(f"  - Overall Summary: Category-level accuracy statistics")
        print(f"  - Individual Category Sheets: Value distribution comparisons")
        print(f"  - Confusion Matrix: All prediction pairs with counts")
        print(f"  - Detailed Mismatches: All disagreements with filenames")
        
    except ImportError:
        print("\nNote: pandas not installed. Install with 'pip install pandas openpyxl' to enable Excel export.")
    except Exception as e:
        print(f"\nError creating Excel file: {e}")
