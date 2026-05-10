# This script draws RD curves based on evaluation resutls
# the curves are grouped by name, keys are "df2k", "minecraft", "combined"
# TODO marks places that can be customized based on need

import os
import re
import matplotlib.pyplot as plt

eval_dirs = [d for d in os.listdir("outputs") if os.path.isdir(f"outputs/{d}")]

df2k_dirs = [d for d in eval_dirs if "df2k" in d]
minecraft_dirs = [d for d in eval_dirs if "minecraft" in d]
combined_dirs = [d for d in eval_dirs if "combined" in d]

def get_curve(dirs):
    bpps = []
    psnrs = []
    mssims = []

    for d in dirs:
        with open(f"outputs/{d}/results.txt", "r") as f:
            content = f.read()
            print(f"outputs/{d}/results.txt")
            bpp = float(re.search(r"model_bpp:\s+([\d.]+)", content).group(0).split(": ")[1])
            psnr = float(re.search(r"model_psnr:\s+([\d.]+)", content).group(0).split(": ")[1])
            mssim = float(re.search(r"model_ms-ssim:\s+([\d.]+)", content).group(0).split(": ")[1])

            bpps.append(bpp)
            psnrs.append(psnr)
            mssims.append(mssim)

    return {"bpp": bpps, "psnr": psnrs, "mssim": mssims}

def get_jpeg_curve(file):
    bpps = []
    psnrs = []
    mssims = []

    with open(file, "r") as f:
        content = f.read()
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
                    mssim = float(m_ssim.group(1)) if m_ssim else 0.0
                    
                    bpps.append(bpp)
                    psnrs.append(psnr)
                    mssims.append(mssim)

    return {"bpp": bpps, "psnr": psnrs, "mssim": mssims}

def draw_curve(ax, x, y, color, label):
    points = sorted(zip(x, y), key=lambda p: p[0])
    x_sorted, y_sorted = zip(*points)
    ax.plot(x_sorted, y_sorted, marker='o', color=color, label=label)

jpeg_curve = get_jpeg_curve(f"outputs/{minecraft_dirs[0]}/results.txt")
minecraft_curve = get_curve(minecraft_dirs)
df2k_curve = get_curve(df2k_dirs)
combined_curve = get_curve(combined_dirs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# TODO, change here if needed
fig.suptitle("Rate-Distortion Curve, Minecraft Screenshots dataset")

for ax, metric in [(ax1, 'psnr'), (ax2, 'mssim')]:
    ax.set_xlabel("BPP")
    ax.set_ylabel(metric.upper())
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    draw_curve(ax, jpeg_curve['bpp'][:-2], jpeg_curve[metric][:-2], "gray", "jpeg")
    draw_curve(ax, minecraft_curve['bpp'], minecraft_curve[metric], "red", "minecraft")
    draw_curve(ax, df2k_curve['bpp'], df2k_curve[metric], "blue", "df2k")
    draw_curve(ax, combined_curve['bpp'], combined_curve[metric], "green", "combined")
    ax.legend()

# TODO, change here if needed
plt.savefig("rd_curve_minecraft.pdf", bbox_inches='tight')
plt.tight_layout()
plt.show()
