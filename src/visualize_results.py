import os
import re
import matplotlib
matplotlib.use('Agg') # Ensure headless stability
import matplotlib.pyplot as plt
import dearpygui.dearpygui as dpg
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelResult:
    run_id: str          # Folder name
    method_type: str     # 'ours', 'cae_only', 'jpeg'
    display_name: str    # Customizable
    psnr: float
    ssim: float
    mse: float
    avg_bpp: float = 0.0
    enabled: bool = True

class ResultVisualizer:
    def __init__(self, root_dir="outputs"):
        self.root_dir = root_dir
        self.all_results: List[ModelResult] = []
        self.load_data()
        self.setup_gui()

    def parse_results_file(self, filepath: str) -> List[ModelResult]:
        results = []
        run_id = os.path.basename(os.path.dirname(filepath))
        
        with open(filepath, 'r') as f:
            content = f.read()

        # 1. Parse BPP
        # New format: | ours: 3.983 bpp | jpeg: 0.450 bpp
        # Old format: | 3.983 bpp
        ours_bpp_matches = re.findall(r"ours:\s+([\d.]+) bpp", content)
        jpeg_bpp_matches = re.findall(r"jpeg:\s+([\d.]+) bpp", content)
        
        # Fallback to old format if new one isn't found
        if not ours_bpp_matches:
            ours_bpp_matches = re.findall(r"\|\s+([\d.]+) bpp", content)
            
        avg_ours_bpp = sum(float(x) for x in ours_bpp_matches) / len(ours_bpp_matches) if ours_bpp_matches else 0.0
        avg_jpeg_bpp = sum(float(x) for x in jpeg_bpp_matches) / len(jpeg_bpp_matches) if jpeg_bpp_matches else 0.0

        # 2. Parse Metric Blocks
        blocks = re.split(r"={10,}", content)
        
        for i in range(len(blocks)):
            if "Image metrics comparison" in blocks[i]:
                header = blocks[i]
                data_block = blocks[i+1] if i+1 < len(blocks) else ""
                
                method_match = re.search(r"vs (\w+)", header)
                if not method_match: continue
                method_type = method_match.group(1)

                psnr_match = re.search(r"PSNR:\s+([\d.]+)", data_block)
                ssim_match = re.search(r"SSIM:\s+([\d.]+)", data_block)
                mse_match = re.search(r"MSE:\s+([\d.]+)", data_block)
                
                if not (psnr_match and ssim_match and mse_match):
                    continue

                psnr = float(psnr_match.group(1))
                ssim = float(ssim_match.group(1))
                mse = float(mse_match.group(1))
                
                # Assign correct BPP
                if method_type == "ours" or method_type == "cae_only":
                    current_bpp = avg_ours_bpp
                elif method_type == "jpeg":
                    current_bpp = avg_jpeg_bpp
                else:
                    current_bpp = 0.0
                
                results.append(ModelResult(
                    run_id=run_id,
                    method_type=method_type,
                    display_name=f"{run_id} ({method_type})",
                    psnr=psnr,
                    ssim=ssim,
                    mse=mse,
                    avg_bpp=current_bpp,
                    enabled=(method_type == "ours")
                ))
        return results

    def load_data(self):
        self.all_results = []
        if not os.path.exists(self.root_dir):
            return
        
        # Walk through the outputs directory to find results.txt files
        for root, dirs, files in os.walk(self.root_dir):
            if "results.txt" in files:
                path = os.path.join(root, "results.txt")
                try:
                    self.all_results.extend(self.parse_results_file(path))
                except Exception as e:
                    print(f"Error parsing {path}: {e}")

    def generate_plots(self):
        # Filter enabled items and maintain order
        selected = [res for res in self.all_results if res.enabled]
        if not selected:
            print("No models selected for plotting.")
            return

        names = [res.display_name for res in selected]
        
        # Calculate PSNR/BPP ratio (Efficiency Score)
        efficiency_scores = []
        for res in selected:
            if res.avg_bpp > 0:
                efficiency_scores.append(res.psnr / res.avg_bpp)
            else:
                efficiency_scores.append(0.0)

        plt.style.use('default')
        fig, ax = plt.subplots() # Default figsize
        
        x = range(len(names))
        
        # Plot only the Efficiency Score using default colors
        ax.bar(x, efficiency_scores)
        
        ax.set_ylabel('Efficiency Score (PSNR / BPP)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha='right')
        ax.set_title("Model Compression Efficiency Comparison")
        ax.grid(True, axis='y')

        save_path = os.path.join(self.root_dir, "efficiency_comparison.png")
        plt.tight_layout()
        plt.savefig(save_path) # Default dpi
        plt.close(fig)
        
        self.show_popup("Plots Saved", f"Efficiency graph saved to:\n\n{os.path.abspath(save_path)}")

    def show_popup(self, title, message):
        if dpg.does_item_exist("modal_popup"):
            dpg.delete_item("modal_popup")
            
        with dpg.window(label=title, modal=True, tag="modal_popup", no_title_bar=False, pos=(250, 200)):
            dpg.add_text(message)
            dpg.add_spacer(height=10)
            dpg.add_button(label="OK", width=75, callback=lambda: dpg.configure_item("modal_popup", show=False))

    def move_item(self, index, direction):
        target = index + direction
        if 0 <= target < len(self.all_results):
            self.all_results[index], self.all_results[target] = self.all_results[target], self.all_results[index]
            self.refresh_table()

    def update_enabled(self, sender, app_data, user_data):
        self.all_results[user_data].enabled = app_data

    def update_name(self, sender, app_data, user_data):
        self.all_results[user_data].display_name = app_data

    def refresh_table(self):
        # Clear existing rows
        if dpg.does_item_exist("results_table"):
            rows = dpg.get_item_children("results_table", 1)
            for row in rows:
                dpg.delete_item(row)

        for i, res in enumerate(self.all_results):
            with dpg.table_row(parent="results_table"):
                dpg.add_checkbox(default_value=res.enabled, 
                                callback=self.update_enabled, 
                                user_data=i)
                
                dpg.add_input_text(default_value=res.display_name, 
                                  callback=self.update_name, 
                                  user_data=i, width=-1)
                
                dpg.add_text(res.method_type)
                dpg.add_text(f"{res.psnr:.2f}")
                dpg.add_text(f"{res.avg_bpp:.3f}" if res.avg_bpp > 0 else "N/A")
                
                with dpg.group(horizontal=True):
                    dpg.add_button(label="^", callback=lambda s, a, u: self.move_item(u, -1), user_data=i, width=25)
                    dpg.add_button(label="v", callback=lambda s, a, u: self.move_item(u, 1), user_data=i, width=25)

    def setup_gui(self):
        dpg.create_context()
        
        with dpg.window(label="Neural Compression Results Visualizer", width=900, height=700, tag="PrimaryWindow"):
            dpg.add_text("1. Scan outputs directory for results.txt")
            dpg.add_text("2. Select which models/runs to include in the plots")
            dpg.add_text("3. Reorder and rename them as desired")
            dpg.add_spacer(height=10)
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Refresh / Reload Data", callback=lambda: (self.load_data(), self.refresh_table()), width=200)
                dpg.add_button(label="GENERATE MATPLOTLIB PLOTS", callback=self.generate_plots, width=250)
            
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=10)

            with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp,
                          borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True,
                          tag="results_table"):
                dpg.add_table_column(label="Plot?", width_fixed=True)
                dpg.add_table_column(label="Display Name (Click to edit)")
                dpg.add_table_column(label="Method", width_fixed=True)
                dpg.add_table_column(label="PSNR", width_fixed=True)
                dpg.add_table_column(label="Avg BPP", width_fixed=True)
                dpg.add_table_column(label="Order", width_fixed=True)
                
            self.refresh_table()
        
        dpg.create_viewport(title='Compression Results Visualizer', width=950, height=750)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("PrimaryWindow", True)
        dpg.start_dearpygui()
        dpg.destroy_context()

if __name__ == "__main__":
    visualizer = ResultVisualizer()
