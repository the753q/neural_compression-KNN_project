import os
import re
import datetime
import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "outputs"

class ResultVisualizer:
    def __init__(self):
        self.available_runs = []
        self.selected_runs = []  # List of dicts: {"path": str, "name": str, "model_rd": (bpp, psnr, ssim), "jpeg_rd": list of (bpp, psnr, ssim)}
        
        self.load_available_runs()
        self.setup_dpg()

    def load_available_runs(self):
        self.available_runs = []
        if not os.path.exists(OUTPUT_DIR):
            return
        
        # Sort by modification time to show latest first
        dirs = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
        dirs.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
        
        for d in dirs:
            results_txt = os.path.join(OUTPUT_DIR, d, "results.txt")
            if os.path.exists(results_txt):
                self.available_runs.append(d)

    def parse_results(self, run_dir_name):
        path = os.path.join(OUTPUT_DIR, run_dir_name, "results.txt")
        model_rd = None # (bpp, psnr, ssim)
        jpeg_rd = []    # list of (bpp, psnr, ssim)
        
        try:
            with open(path, "r") as f:
                content = f.read()
                
                # Parse model RD
                model_bpp = re.search(r"model_bpp:\s+([\d.]+)", content)
                model_psnr = re.search(r"model_psnr:\s+([\d.]+)", content)
                model_msssim = re.search(r"model_ms-ssim:\s+([\d.]+)", content)
                
                if model_bpp and model_psnr:
                    bpp = float(model_bpp.group(1))
                    psnr = float(model_psnr.group(1))
                    msssim = float(model_msssim.group(1)) if model_msssim else 0.0
                    model_rd = (bpp, psnr, msssim)

                # Parse JPEG RD curve
                jpeg_section = re.search(r"\[JPEG_RD_CURVE\](.*?)(?:\n\n|\Z)", content, re.DOTALL)
                if jpeg_section:
                    lines = jpeg_section.group(1).strip().split("\n")
                    for line in lines:
                        m_bpp = re.search(r"bpp=([\d.]+)", line)
                        m_psnr = re.search(r"psnr=([\d.]+)", line)
                        m_ssim = re.search(r"ms-ssim=([\d.]+)", line)
                        if m_bpp and m_psnr:
                            bpp = float(m_bpp.group(1))
                            psnr = float(m_psnr.group(1))
                            msssim = float(m_ssim.group(1)) if m_ssim else 0.0
                            jpeg_rd.append((bpp, psnr, msssim))
                
                jpeg_rd.sort(key=lambda x: x[0])
        except Exception as e:
            print(f"Error parsing {path}: {e}")
            
        return model_rd, jpeg_rd

    def save_matplotlib_plot(self):
        if not self.selected_runs:
            print("No runs selected!")
            return

        now = datetime.datetime.now().strftime("%m_%d_%H_%M")
        metrics = [("psnr", "PSNR (dB)", 1), ("ms-ssim", "MS-SSIM", 2)]
        
        for metric_key, y_label, metric_idx in metrics:
            # Use original single-plot size
            plt.figure(figsize=(12, 8))
            
            # Plot JPEG curve from first selected run
            first_run = self.selected_runs[0]
            if first_run["jpeg_rd"]:
                bpps = [x[0] for x in first_run["jpeg_rd"]]
                vals = [x[metric_idx] for x in first_run["jpeg_rd"]]
                # Keep the darker line improvement
                plt.plot(bpps, vals, 'o--', label="JPEG (Baseline)", color='#333333', alpha=0.9, linewidth=1.5)

            # Plot model dots for all selected runs
            colors = plt.cm.rainbow(np.linspace(0, 1, len(self.selected_runs)))
            for i, run in enumerate(self.selected_runs):
                if run["model_rd"]:
                    bpp = run["model_rd"][0]
                    val = run["model_rd"][metric_idx]
                    plt.scatter(bpp, val, color=colors[i], label=run["name"], s=100, edgecolors='black', zorder=5)

            plt.xlabel("Bitrate (bpp)")
            plt.ylabel(y_label)
            plt.title(f"Rate-Distortion Comparison ({metric_key.upper()})")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Restore legend with boundary/frame below the graph
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(len(self.selected_runs)+1, 3), frameon=True, borderaxespad=0.5)
            
            # Adjust layout to make room for legend at bottom
            plt.tight_layout()
            
            save_path = os.path.join(OUTPUT_DIR, f"rd_{metric_key}_{now}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Plot saved to {save_path}")

    def add_run(self, sender, app_data):
        selected_available = dpg.get_value("available_runs_list")
        if not selected_available:
            return
        
        model_rd, jpeg_rd = self.parse_results(selected_available)
        self.selected_runs.append({
            "path": selected_available,
            "name": selected_available,
            "model_rd": model_rd,
            "jpeg_rd": jpeg_rd
        })
        self.refresh_selected_list()

    def on_available_double_click(self):
        self.add_run(None, None)

    def on_selected_click(self, sender, app_data):
        selected_str = dpg.get_value(sender)
        for i, run in enumerate(self.selected_runs):
            if run["name"] == selected_str:
                dpg.set_value("selected_runs_list_idx", i)
                dpg.set_value("rename_input", run["name"])
                break

    def remove_run(self, sender, app_data):
        idx = dpg.get_value("selected_runs_list_idx")
        if idx is not None and 0 <= idx < len(self.selected_runs):
            self.selected_runs.pop(idx)
            self.refresh_selected_list()

    def move_up(self, sender, app_data):
        idx = dpg.get_value("selected_runs_list_idx")
        if idx is not None and idx > 0:
            self.selected_runs[idx], self.selected_runs[idx-1] = self.selected_runs[idx-1], self.selected_runs[idx]
            self.refresh_selected_list()
            dpg.set_value("selected_runs_list_idx", idx - 1)

    def move_down(self, sender, app_data):
        idx = dpg.get_value("selected_runs_list_idx")
        if idx is not None and idx < len(self.selected_runs) - 1:
            self.selected_runs[idx], self.selected_runs[idx+1] = self.selected_runs[idx+1], self.selected_runs[idx]
            self.refresh_selected_list()
            dpg.set_value("selected_runs_list_idx", idx + 1)

    def rename_run(self, sender, app_data):
        idx = dpg.get_value("selected_runs_list_idx")
        new_name = dpg.get_value("rename_input")
        if idx is not None and 0 <= idx < len(self.selected_runs) and new_name:
            self.selected_runs[idx]["name"] = new_name
            self.refresh_selected_list()

    def refresh_selected_list(self):
        items = [run["name"] for run in self.selected_runs]
        dpg.configure_item("selected_runs_list", items=items)

    def setup_dpg(self):
        dpg.create_context()
        
        with dpg.item_handler_registry(tag="available_list_handler"):
            dpg.add_item_double_clicked_handler(callback=self.on_available_double_click)

        with dpg.window(label="RD Plot Generator (PSNR & MS-SSIM)", width=1000, height=600):
            with dpg.group(horizontal=True):
                # Left Column: Available Runs
                with dpg.child_window(width=480, height=450):
                    dpg.add_text("Available Runs (Double-click to add)")
                    dpg.add_listbox(items=self.available_runs, tag="available_runs_list", num_items=20, width=-1)
                    dpg.bind_item_handler_registry("available_runs_list", "available_list_handler")
                    
                    dpg.add_button(label="Add to List", callback=self.add_run, width=-1)
                    dpg.add_button(label="Refresh Disk", callback=lambda: (self.load_available_runs(), dpg.configure_item("available_runs_list", items=self.available_runs)), width=-1)

                # Right Column: Selected Runs
                with dpg.child_window(width=480, height=450):
                    dpg.add_text("Selected Runs")
                    dpg.add_listbox(items=[], tag="selected_runs_list", num_items=15, width=-1, callback=self.on_selected_click)
                    
                    with dpg.group(horizontal=True):
                        dpg.add_text("Idx:")
                        dpg.add_input_int(tag="selected_runs_list_idx", min_value=0, width=-1, step=0)
                    
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Up", callback=self.move_up, width=150)
                        dpg.add_button(label="Down", callback=self.move_down, width=150)
                        dpg.add_button(label="Remove", callback=self.remove_run, width=150)
                    
                    dpg.add_spacer(height=5)
                    dpg.add_input_text(label="Rename", tag="rename_input", width=-1)
                    dpg.add_button(label="Update Name", callback=self.rename_run, width=-1)
            
            dpg.add_spacer(height=10)
            dpg.add_button(label="GENERATE AND SAVE PLOTS (PSNR & MS-SSIM)", callback=self.save_matplotlib_plot, width=-1, height=50)
            dpg.add_text("Note: First selected run defines the JPEG baseline curves.", color=(255, 200, 100))

        dpg.create_viewport(title='RD Plot Generator', width=1020, height=640)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

if __name__ == "__main__":
    ResultVisualizer()
