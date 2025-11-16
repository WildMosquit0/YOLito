import customtkinter as ctk
import tkinter as tk  # only for filedialog, etc.
import yaml
import os
import sys
import subprocess
from pathlib import Path
from tkinter import filedialog

# ------------------ Global paths or constants ------------------ #
CONFIG_INFER   = "configs/infer.yaml"
CONFIG_ANALYZE = "configs/analyze.yaml"
BASE_VIDEO_DIR = "./videos"
BASE_OUTPUT_DIR = "./output"

# ------------------ Appearance Setup ------------------ #
ctk.set_appearance_mode("Dark")          # "System", "Dark", or "Light"
ctk.set_default_color_theme("dark-blue") # "blue", "green", or "dark-blue"

# ---------------------------------------------------------------------- #
#                        Utility / Directory Helpers                     #
# ---------------------------------------------------------------------- #
def get_two_latest_dirs(base_path):
    """Return up to two newest subdirectories under base_path."""
    base = Path(base_path)
    if not base.is_dir():
        return []
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)  # newest first
    return [str(d) for d in subdirs[:2]]

def browse_directory(initial_dir, option_menu: ctk.CTkOptionMenu):
    """Open a folder dialog and set the OptionMenu selection."""
    folder = filedialog.askdirectory(initialdir=initial_dir)
    if folder:
        option_menu.set(folder)

# ---------------------------------------------------------------------- #
#                       Config Loading / Saving                          #
# ---------------------------------------------------------------------- #
def load_infer_config():
    """Load infer config from YAML, populate the Infer tab widgets."""
    try:
        with open(CONFIG_INFER, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}

    model = cfg.get("model", {})
    # Weights
    entry_weights.delete(0, tk.END)
    entry_weights.insert(0, model.get("weights", ""))

    # track or predict
    om_infer_task.set(model.get("task", "track"))

    # small or big
    om_infer_object_size.set(model.get("object_size", "small"))

    # conf threshold
    entry_conf_threshold.delete(0, tk.END)
    entry_conf_threshold.insert(0, str(model.get("conf_threshold", 0.25)))

    # iou threshold
    entry_iou_threshold.delete(0, tk.END)
    entry_iou_threshold.insert(0, str(model.get("iou_threshold", 0.2)))

    # vid stride
    entry_vid_stride.delete(0, tk.END)
    entry_vid_stride.insert(0, str(model.get("vid_stride", 1000)))

    # directories
    images_dir = cfg.get("images_dir", "")
    om_infer_images.set(images_dir if images_dir else (infer_video_dirs[0] if infer_video_dirs else ""))
    output_dir = cfg.get("output_dir", "")
    om_infer_output.set(output_dir if output_dir else (infer_output_dirs[0] if infer_output_dirs else ""))

def save_infer_config():
    """Write only updated fields (partial) back to infer.yaml."""
    try:
        with open(CONFIG_INFER, "r") as f:
            existing_cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        existing_cfg = {}

    # Ensure sub-dicts exist
    model_cfg = existing_cfg.get("model", {})
    sahi_cfg = existing_cfg.get("sahi", {})

    # 1) Basic model fields from the GUI
    model_cfg["weights"]        = entry_weights.get()
    model_cfg["task"]           = om_infer_task.get()       # "track" or "predict"
    model_cfg["object_size"]    = om_infer_object_size.get() # "small" or "big"
    model_cfg["conf_threshold"] = float(entry_conf_threshold.get())
    model_cfg["iou_threshold"]  = float(entry_iou_threshold.get())
    model_cfg["vid_stride"]     = int(entry_vid_stride.get())

    # 2) Special condition: if object_size == "small" => model.task = "slice"
    if om_infer_object_size.get() == "small":
        model_cfg["task"] = "slice"

    # 3) Another special condition: if model_action == "track", then sahi.track = True, else False
    if om_infer_task.get() == "track":
        sahi_cfg["track"] = True
    else:
        sahi_cfg["track"] = False

    existing_cfg["model"] = model_cfg
    existing_cfg["sahi"]  = sahi_cfg

    existing_cfg["images_dir"] = om_infer_images.get()
    existing_cfg["output_dir"] = om_infer_output.get()

    # Write updated config back to file
    with open(CONFIG_INFER, "w") as f:
        yaml.safe_dump(existing_cfg, f)

def load_analyze_config():
    """Load analyze config from YAML, populate the Analyze tab widgets."""
    try:
        with open(CONFIG_ANALYZE, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}

    task_cfg = cfg.get("task", {})
    # Checkboxes
    if task_cfg.get("plotxy", False):
        cb_plotxy.select()
    else:
        cb_plotxy.deselect()

    if task_cfg.get("average_visits", False):
        cb_average.select()
    else:
        cb_average.deselect()

    if task_cfg.get("duration", False):
        cb_duration.select()
    else:
        cb_duration.deselect()

    if task_cfg.get("heatmap", False):
        cb_heatmap.select()
    else:
        cb_heatmap.deselect()

def save_analyze_config():
    """Write only updated fields (partial) back to analyze.yaml."""
    try:
        with open(CONFIG_ANALYZE, "r") as f:
            existing_cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        existing_cfg = {}

    task_cfg = existing_cfg.get("task", {})
    task_cfg["plotxy"]         = bool(cb_plotxy.get())
    task_cfg["average_visits"] = bool(cb_average.get())
    task_cfg["duration"]       = bool(cb_duration.get())
    task_cfg["heatmap"]        = bool(cb_heatmap.get())
    existing_cfg["task"]       = task_cfg

    with open(CONFIG_ANALYZE, "w") as f:
        yaml.safe_dump(existing_cfg, f)

# ---------------------------------------------------------------------- #
#                           Main "Run" Logic                             #
# ---------------------------------------------------------------------- #
def run_main():
    """Depending on which tab is active, save config and run main.py."""
    current_tab = tabview.get()  # "Infer" or "Analyze"
    if current_tab == "Infer":
        save_infer_config()
        task = "infer"
    else:
        save_analyze_config()
        task = "analyze"

    # Run main.py
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), "main.py")
    subprocess.run([python_exe, script_path, "--task", task])

def on_tab_switch():
    """
    Called whenever the user changes tabs. We can load the config 
    so that the correct settings display.
    """
    current_tab = tabview.get()
    if current_tab == "Infer":
        load_infer_config()
    else:
        load_analyze_config()

# ---------------------------------------------------------------------- #
#                     Build the CustomTkinter Window                     #
# ---------------------------------------------------------------------- #
app = ctk.CTk()
app.title("Simple Config GUI")
app.geometry("800x500")

# Make the main window stretch nicely
app.grid_rowconfigure(0, weight=1)
app.grid_columnconfigure(0, weight=1)

#
# Create a CTkTabview
#
tabview = ctk.CTkTabview(app, width=600, height=400)
tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

# Add two tabs
infer_tab = tabview.add("Infer")
analyze_tab = tabview.add("Analyze")

# We'll do a small trick: schedule a repeated check for tab changes
last_tab = [tabview.get()]  # store in a list for mutability
def check_tab():
    current = tabview.get()
    if current != last_tab[0]:
        last_tab[0] = current
        on_tab_switch()
    app.after(500, check_tab)

app.after(500, check_tab)

# --------------- "Infer" tab layout ---------------
infer_tab.grid_rowconfigure((0,1,2,3,4,5,6), weight=0, pad=5)
infer_tab.grid_columnconfigure((0,1), weight=1)

row_idx = 0

lbl_weights = ctk.CTkLabel(infer_tab, text="Weights:")
lbl_weights.grid(row=row_idx, column=0, sticky="e", padx=10, pady=5)
entry_weights = ctk.CTkEntry(infer_tab, width=300)
entry_weights.grid(row=row_idx, column=1, sticky="w", padx=10, pady=5)
row_idx += 1

lbl_task = ctk.CTkLabel(infer_tab, text="Model path:")
lbl_task.grid(row=row_idx, column=0, sticky="e", padx=10, pady=5)
om_infer_task = ctk.CTkOptionMenu(infer_tab, values=["Track", "Predict"])
om_infer_task.grid(row=row_idx, column=1, sticky="w", padx=10, pady=5)
row_idx += 1

lbl_object_size = ctk.CTkLabel(infer_tab, text="Object Size:")
lbl_object_size.grid(row=row_idx, column=0, sticky="e", padx=10, pady=5)
om_infer_object_size = ctk.CTkOptionMenu(infer_tab, values=["Small", "Big"])
om_infer_object_size.grid(row=row_idx, column=1, sticky="w", padx=10, pady=5)
row_idx += 1

lbl_conf = ctk.CTkLabel(infer_tab, text="Conf Threshold:")
lbl_conf.grid(row=row_idx, column=0, sticky="e", padx=10, pady=5)
entry_conf_threshold = ctk.CTkEntry(infer_tab, width=100)
entry_conf_threshold.grid(row=row_idx, column=1, sticky="w", padx=10, pady=5)
row_idx += 1

lbl_iou = ctk.CTkLabel(infer_tab, text="NMS Threshold:")
lbl_iou.grid(row=row_idx, column=0, sticky="e", padx=10, pady=5)
entry_iou_threshold = ctk.CTkEntry(infer_tab, width=100)
entry_iou_threshold.grid(row=row_idx, column=1, sticky="w", padx=10, pady=5)
row_idx += 1

lbl_stride = ctk.CTkLabel(infer_tab, text="Frame sampling:")
lbl_stride.grid(row=row_idx, column=0, sticky="e", padx=10, pady=5)
entry_vid_stride = ctk.CTkEntry(infer_tab, width=100)
entry_vid_stride.grid(row=row_idx, column=1, sticky="w", padx=10, pady=5)
row_idx += 1

lbl_images = ctk.CTkLabel(infer_tab, text="Images/Video Dir:")
lbl_images.grid(row=row_idx, column=0, sticky="e", padx=10, pady=5)

infer_video_dirs = get_two_latest_dirs(BASE_VIDEO_DIR)
om_infer_images = ctk.CTkOptionMenu(
    infer_tab,
    values=infer_video_dirs if infer_video_dirs else ["<No Subfolders>"]
)
om_infer_images.grid(row=row_idx, column=1, sticky="w", padx=10, pady=5)

browse_images_btn = ctk.CTkButton(
    infer_tab,
    text="Browse",
    command=lambda: browse_directory(BASE_VIDEO_DIR, om_infer_images)
)
browse_images_btn.grid(row=row_idx, column=2, sticky="w", padx=5)
row_idx += 1

lbl_output = ctk.CTkLabel(infer_tab, text="Output Dir:")
lbl_output.grid(row=row_idx, column=0, sticky="e", padx=10, pady=5)

infer_output_dirs = get_two_latest_dirs(BASE_OUTPUT_DIR)
om_infer_output = ctk.CTkOptionMenu(
    infer_tab,
    values=infer_output_dirs if infer_output_dirs else ["<No Subfolders>"]
)
om_infer_output.grid(row=row_idx, column=1, sticky="w", padx=10, pady=5)

browse_output_btn = ctk.CTkButton(
    infer_tab,
    text="Browse",
    command=lambda: browse_directory(BASE_OUTPUT_DIR, om_infer_output)
)
browse_output_btn.grid(row=row_idx, column=2, sticky="w", padx=5)
row_idx += 1

# --------------- "Analyze" tab layout ---------------
analyze_tab.grid_rowconfigure((0,1,2,3), weight=0, pad=5)
analyze_tab.grid_columnconfigure(0, weight=1)

cb_plotxy = ctk.CTkCheckBox(analyze_tab, text="Plot XY")
cb_plotxy.grid(row=0, column=0, sticky="w", padx=20, pady=5)

cb_average = ctk.CTkCheckBox(analyze_tab, text="Average Visits")
cb_average.grid(row=1, column=0, sticky="w", padx=20, pady=5)

cb_duration = ctk.CTkCheckBox(analyze_tab, text="Duration")
cb_duration.grid(row=2, column=0, sticky="w", padx=20, pady=5)

cb_heatmap = ctk.CTkCheckBox(analyze_tab, text="Heatmap")
cb_heatmap.grid(row=3, column=0, sticky="w", padx=20, pady=5)

# --------------- "Run" button at bottom ---------------
run_button = ctk.CTkButton(app, text="Run", command=run_main)
run_button.grid(row=1, column=0, pady=(0, 10))

#
# Load defaults for whichever tab is first ("Infer" by default):
#
tabview.set("Infer")  # Switch to "Infer" tab by default
load_infer_config()   # Load infer config once at startup

app.mainloop()
