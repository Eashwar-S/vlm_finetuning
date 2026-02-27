import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt
from PIL import Image

import ipywidgets as widgets
from IPython.display import display, clear_output

# ----------------------------
# Paths (edit if needed)
# ----------------------------
DATASET_ROOT = Path("dataset")
JSONL_PATH = DATASET_ROOT / "data" / "train_short.jsonl"
IMAGES_DIR = DATASET_ROOT / "images_short"

OUT_LOG_JSONL = DATASET_ROOT / "data" / "review_log.jsonl"
OUT_STATE_JSON = DATASET_ROOT / "data" / "review_state.json"
OUT_SUMMARY_TXT = DATASET_ROOT / "data" / "review_summary.txt"
OUT_GROUND_TRUTH_JSON = DATASET_ROOT / "data" / "ground_truth.json"

# ----------------------------
# Load data
# ----------------------------
assert JSONL_PATH.exists(), f"Could not find {JSONL_PATH}"
assert IMAGES_DIR.exists(), f"Could not find {IMAGES_DIR}"

entries = []
with JSONL_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            entries.append(json.loads(line))

assert len(entries) > 0, "train_short.jsonl is empty?"
print(f"Loaded {len(entries)} entries")

# Determine fields from first entry
FIELDS = list(entries[0]["teacher_answer"].keys())

# Build field options from schema (enum values per field)
# Try to extract from the first entry's response_schema; fall back to None
def build_field_options(entries):
    """Extract enum options per field from response_schema in entries."""
    options_map = {}
    for entry in entries:
        schema = entry.get("response_schema", {})
        props = schema.get("properties", {})
        for field, meta in props.items():
            if "enum" in meta and field not in options_map:
                options_map[field] = meta["enum"]
        if len(options_map) == len(FIELDS):
            break
    return options_map

FIELD_OPTIONS = build_field_options(entries)

# ----------------------------
# Load existing state (resume)
# ----------------------------
state = {
    "idx": 0,
    "reviews": {}  # sha1 -> {"per_field": {field: "correct"/"incorrect"}, "ground_truth_overrides": {field: chosen_value}, "timestamp": "..."}
}

if OUT_STATE_JSON.exists():
    try:
        state = json.loads(OUT_STATE_JSON.read_text(encoding="utf-8"))
        print(f"Resumed from {OUT_STATE_JSON} (idx={state.get('idx',0)})")
    except Exception as e:
        print("Warning: could not read state file, starting fresh:", e)

# Ensure required keys
state.setdefault("idx", 0)
state.setdefault("reviews", {})

# Load existing ground truth
ground_truth = {}
if OUT_GROUND_TRUTH_JSON.exists():
    try:
        ground_truth = json.loads(OUT_GROUND_TRUTH_JSON.read_text(encoding="utf-8"))
        print(f"Loaded existing ground_truth.json ({len(ground_truth)} entries)")
    except Exception as e:
        print("Warning: could not read ground_truth.json, starting fresh:", e)

def write_state():
    OUT_STATE_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")

def write_ground_truth():
    OUT_GROUND_TRUTH_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_GROUND_TRUTH_JSON.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")

def compute_ground_truth_for_entry(entry, per_field, gt_overrides):
    """
    Build the ground truth dict for a single entry.
    - correct  => teacher_answer value is the ground truth
    - incorrect => use gt_overrides[field] if set, else mark as None
    """
    ta = entry.get("teacher_answer", {})
    gt = {}
    for field in FIELDS:
        verdict = per_field.get(field, "correct")
        if verdict == "correct":
            gt[field] = ta.get(field)
        else:  # incorrect
            override = gt_overrides.get(field)
            gt[field] = override  # may be None if user hasn't picked yet
    return gt

def compute_summary(reviews):
    counts = {field: {"correct": 0, "incorrect": 0} for field in FIELDS}
    total_done = 0

    for sha1, rec in reviews.items():
        per_field = rec.get("per_field", {})
        if per_field:
            total_done += 1
        for field in FIELDS:
            v = per_field.get(field, "correct")
            if v not in ("correct", "incorrect"):
                v = "correct"
            counts[field][v] += 1

    lines = []
    lines.append(f"Review summary updated: {datetime.utcnow().isoformat()}Z")
    lines.append(f"Total images reviewed: {total_done}")
    lines.append(f"Total images in dataset: {len(entries)}")
    lines.append("")

    for field in FIELDS:
        c = counts[field]["correct"]
        ic = counts[field]["incorrect"]
        denom = c + ic
        pct = (100.0 * c / denom) if denom > 0 else 0.0
        lines.append(f"{field}: correct={c}, incorrect={ic}, accuracy={pct:.1f}%")

    total_c = sum(counts[f]["correct"] for f in FIELDS)
    total_ic = sum(counts[f]["incorrect"] for f in FIELDS)
    total_denom = total_c + total_ic
    overall_pct = (100.0 * total_c / total_denom) if total_denom > 0 else 0.0
    lines.append("")
    lines.append(f"OVERALL accuracy: {overall_pct:.1f}%")
    return "\n".join(lines)

def write_summary_txt():
    OUT_SUMMARY_TXT.write_text(compute_summary(state["reviews"]), encoding="utf-8")

def append_log(entry, review_rec):
    log_row = {
        "sha1": entry.get("sha1"),
        "filename": entry.get("filename"),
        "image": entry.get("image"),
        "teacher_answer": entry.get("teacher_answer"),
        "review": review_rec,
        "logged_at": datetime.utcnow().isoformat() + "Z",
    }
    OUT_LOG_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_LOG_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_row) + "\n")

# ----------------------------
# UI Widgets
# ----------------------------
title = widgets.HTML("")
progress = widgets.HTML("")

# Left: image
img_out = widgets.Output()

# Right: teacher answer text
answer_out = widgets.Output()

# CSS injection to make buttons bigger
style_html = widgets.HTML("""
<style>
.widget-toggle-buttons .widget-toggle-button {
    font-size: 14px !important;
    min-width: 110px !important;
    min-height: 36px !important;
    padding: 6px 14px !important;
}
</style>
""")

# Per-field correctness widgets + override dropdowns
field_widgets = {}      # field -> ToggleButtons
override_widgets = {}   # field -> Dropdown (shown when incorrect)
field_rows = {}         # field -> VBox containing both

for field in FIELDS:
    opts = FIELD_OPTIONS.get(field, [])
    
    toggle = widgets.ToggleButtons(
        options=[
            ("✓ Correct", "correct"),
            ("✗ Incorrect", "incorrect"),
        ],
        value="correct",
        description=field,
        style={"description_width": "260px", "button_width": "120px"},
        layout=widgets.Layout(width="95%"),
        button_style="",
    )
    
    dd_opts = [(o, o) for o in opts] if opts else [("(no options)", None)]
    override_dd = widgets.Dropdown(
        options=dd_opts,
        description="True answer:",
        style={"description_width": "100px"},
        layout=widgets.Layout(width="60%", display="none"),
    )
    
    # Show/hide dropdown based on toggle value
    def make_toggle_handler(dd):
        def handler(change):
            dd.layout.display = "" if change["new"] == "incorrect" else "none"
        return handler
    
    toggle.observe(make_toggle_handler(override_dd), names="value")
    
    field_widgets[field] = toggle
    override_widgets[field] = override_dd
    field_rows[field] = widgets.VBox([toggle, override_dd])

# Buttons
btn_prev = widgets.Button(
    description="◀ Prev", button_style="",
    layout=widgets.Layout(width="120px", height="40px"),
)
btn_save_next = widgets.Button(
    description="Save + Next ▶", button_style="success",
    layout=widgets.Layout(width="160px", height="40px"),
)
btn_jump = widgets.IntText(value=0, description="Jump to idx:", layout=widgets.Layout(width="220px"))
btn_go = widgets.Button(
    description="Go", button_style="info",
    layout=widgets.Layout(width="80px", height="40px"),
)

status = widgets.HTML("")

# Layout
fields_box = widgets.VBox(list(field_rows.values()), layout=widgets.Layout(height="520px", overflow="auto"))
right_panel = widgets.VBox([answer_out, widgets.HTML("<hr>"), fields_box], layout=widgets.Layout(width="55%"))
left_panel = widgets.VBox([img_out], layout=widgets.Layout(width="45%"))

top_row = widgets.HBox([left_panel, right_panel])
controls = widgets.HBox([btn_prev, btn_save_next, btn_jump, btn_go])
ui = widgets.VBox([style_html, title, progress, top_row, controls, status])

def load_entry_to_ui(idx):
    idx = max(0, min(idx, len(entries)-1))
    state["idx"] = idx
    entry = entries[idx]

    sha1 = entry.get("sha1", f"idx-{idx}")
    fname = entry.get("filename", entry.get("image", ""))
    img_rel = entry.get("image", fname)
    img_path = IMAGES_DIR / img_rel

    title.value = f"<h3>Index {idx}/{len(entries)-1} — {fname}</h3>"
    done = len(state["reviews"])
    progress.value = f"<b>Reviewed entries:</b> {done}  |  <b>Current sha1:</b> {sha1}"

    # Load image
    with img_out:
        clear_output(wait=True)
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            plt.figure(figsize=(7, 4))
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        else:
            print(f"Image not found: {img_path}")

    # Show teacher answer text
    with answer_out:
        clear_output(wait=True)
        ta = entry.get("teacher_answer", {})
        rows = []
        for field in FIELDS:
            val = ta.get(field, "")
            rows.append(f"<tr><td style='padding:4px 8px;'><b>{field}</b></td>"
                        f"<td style='padding:4px 8px;'>{val}</td></tr>")
        html = f"""
        <div style="font-family: monospace; font-size: 13px;">
        <div style="margin-bottom:6px;"><b>teacher_answer</b></div>
        <table style="border-collapse:collapse; width:100%;">
            {''.join(rows)}
        </table>
        </div>
        """
        display(widgets.HTML(value=html))

    # Restore previous selections
    prev = state["reviews"].get(sha1, {})
    prev_per_field = prev.get("per_field", {})
    prev_gt_overrides = prev.get("gt_overrides", {})

    for field in FIELDS:
        verdict = prev_per_field.get(field, "correct")
        if verdict not in ("correct", "incorrect"):
            verdict = "correct"
        field_widgets[field].value = verdict
        # Restore override dropdown value & visibility
        override_dd = override_widgets[field]
        if verdict == "incorrect":
            override_dd.layout.display = ""
            saved_val = prev_gt_overrides.get(field)
            opts = FIELD_OPTIONS.get(field, [])
            if saved_val and saved_val in opts:
                override_dd.value = saved_val
            elif opts:
                override_dd.value = opts[0]
        else:
            override_dd.layout.display = "none"

    status.value = ""

def save_current():
    idx = state["idx"]
    entry = entries[idx]
    sha1 = entry.get("sha1", f"idx-{idx}")
    ta = entry.get("teacher_answer", {})

    per_field = {field: field_widgets[field].value for field in FIELDS}
    
    # Collect gt_overrides: only for "incorrect" fields
    gt_overrides = {}
    for field in FIELDS:
        if per_field[field] == "incorrect":
            gt_overrides[field] = override_widgets[field].value

    review_rec = {
        "per_field": per_field,
        "gt_overrides": gt_overrides,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # Update review state
    state["reviews"][sha1] = review_rec

    # Build ground truth for this entry
    gt_entry = {}
    for field in FIELDS:
        if per_field[field] == "correct":
            gt_entry[field] = ta.get(field)
        else:
            gt_entry[field] = gt_overrides.get(field)  # user-chosen correct value

    ground_truth[sha1] = {
        "filename": entry.get("filename", entry.get("image", "")),
        "ground_truth": gt_entry,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    # Persist
    write_state()
    write_ground_truth()
    write_summary_txt()
    append_log(entry, review_rec)

    status.value = (
        f"<span style='color: green;'><b>Saved.</b></span> "
        f"Updated {OUT_GROUND_TRUTH_JSON.name}, {OUT_SUMMARY_TXT.name}, "
        f"appended to {OUT_LOG_JSONL.name}"
    )

def on_prev(_):
    load_entry_to_ui(state["idx"] - 1)

def on_save_next(_):
    save_current()
    load_entry_to_ui(state["idx"] + 1)

def on_go(_):
    load_entry_to_ui(btn_jump.value)

btn_prev.on_click(on_prev)
btn_save_next.on_click(on_save_next)
btn_go.on_click(on_go)

# Initial render
display(ui)
load_entry_to_ui(state["idx"])
