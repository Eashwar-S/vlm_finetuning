import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Parse evaluation results and output accuracy percentages.")
    parser.add_argument("file_path", nargs="?", default="evaluation_results.xlsx", help="Path to the evaluation results excel file")
    args = parser.parse_args()

    try:
        df = pd.read_excel(args.file_path)
    except FileNotFoundError:
        print(f"Error: Could not find file at {args.file_path}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print("📊 Evaluation Summary:")
    print("=" * 40)
    
    # Identify correct_* columns
    correct_cols = [c for c in df.columns if c.startswith("correct_")]
    
    if not correct_cols:
        print("No correctness columns starting with 'correct_' found in the Excel sheet.")
        return

    # Process each correctness column
    summary_data = []
    
    for col in correct_cols:
        question = col.replace("correct_", "")
        
        # Calculate accuracy ignoring NaN values if any
        if df[col].notna().any():
            acc = df[col].astype(bool).mean() * 100
        else:
            acc = 0.0
            
        summary_data.append({"Category": question, "VLM Model": f"{acc:.1f}%"})
        print(f"  {question:<35} : {acc:5.1f}%")
        
    print("-" * 40)
    
    # Calculate overall accuracy
    if "sample_accuracy" in df.columns:
        overall_mean = df["sample_accuracy"].mean() * 100
    else:
        # Fallback if sample_accuracy column is missing
        # Calculate mean across all questions for all samples
        overall_correct = df[correct_cols].sum().sum()
        total_questions = df[correct_cols].count().sum()
        overall_mean = (overall_correct / total_questions * 100) if total_questions > 0 else 0.0

    print(f"  {'Overall Average Accuracy':<35} : {overall_mean:5.1f}%")
    print("=" * 40)

    # Print markdown table format as requested
    print("\n\nMarkdown Table Format:")
    summary_df = pd.DataFrame(summary_data)
    summary_df.loc[len(summary_df)] = ["Overall Average Accuracy", f"{overall_mean:.1f}%"]
    print(summary_df.to_markdown(index=False))

    # Save to Excel
    output_summary_path = args.file_path.replace(".xlsx", "_summary.xlsx")
    if output_summary_path == args.file_path:
        output_summary_path = "summary_" + args.file_path
        
    summary_df.to_excel(output_summary_path, index=False)
    print(f"\n✅ Saved summary to {output_summary_path}")

if __name__ == "__main__":
    main()
