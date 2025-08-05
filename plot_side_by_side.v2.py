import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from matplotlib.colors import TwoSlopeNorm
from astropy.io import fits as pyfits
from astropy.stats import sigma_clipped_stats
from astropy.visualization.wcsaxes import add_beam
from astropy import units as u
from astropy.wcs import WCS
import argparse
import matplotlib as mpl

mpl.rcParams.update({
    'figure.dpi':       300,
    'font.family':      'serif',
    'font.serif':       ['Times New Roman'],
    'text.usetex':      False,
    'axes.titlesize':   18,
    'mathtext.fontset': 'cm',
    'axes.labelsize':   18,
    'xtick.labelsize':  15,
    'ytick.labelsize':  15,
    'legend.fontsize':  18,
    'axes.grid':        False,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.05,
    'font.weight': 'bold',
})

def get_rms_and_peak(fp):
    with pyfits.open(fp) as hdul:
        data = np.squeeze(hdul[0].data).astype(float)
    flat = data.flatten()
    plain_rms = np.std(flat)
    _, _, clipped_rms = sigma_clipped_stats(flat, sigma=3.0, maxiters=5)
    peak = np.max(flat)
    return clipped_rms, peak

def plot_wcs_panel(ax, fp, rms, peak, vmin, vmax, show_legend=True, title=""):
    with pyfits.open(fp) as hdul:
        data = np.squeeze(hdul[0].data).astype(float) * 1e6
        hdr = hdul[0].header

    im = ax.imshow(data, origin='lower', cmap='RdGy_r', norm=TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax))

    try:
        add_beam(ax, header=hdr, corner="bottom left", facecolor="black", alpha=0.9, edgecolor="yellow")
        bmaj = hdr.get('BMAJ', 0) * 3600.0
        bmin = hdr.get('BMIN', 0) * 3600.0
        beam_label = f"{bmaj:.4f}\"×{bmin:.4f}\""
        ax.text(0.1, 0.1, beam_label, transform=ax.transAxes, ha='left', va='bottom', fontsize=11,
                color='black', path_effects=[pe.withStroke(linewidth=1.5, foreground='white')], zorder=20)
    except Exception:
        pass

    pos_levels = np.array([2,3,4,5,6,8,10,12,14,16,20,25,30,35,40,45,50,55,60,70,80,90,100,110,120,130,140,150]) * rms * 1e6
    neg_levels = np.sort(-pos_levels)
    ax.contour(data, levels=pos_levels, colors='black', linewidths=0.5)
    ax.contour(data, levels=neg_levels, colors='white', linewidths=0.5, linestyles='dashed')

    ax.coords['ra'].set_format_unit(u.hour)
    ax.coords['ra'].set_major_formatter('hh:mm:ss.ss')
    ax.coords['dec'].set_major_formatter('dd:mm:ss.ss')
    ax.coords['ra'].set_ticks(number=1)
    ax.coords['dec'].set_ticks(number=4)
    ax.coords['ra'].set_ticklabel(fontsize=11)
    ax.coords['dec'].set_ticklabel(fontsize=11)
    ax.coords['ra'].set_axislabel("")
    ax.coords['dec'].set_axislabel("")
    ax.coords.grid(color='gray', ls=':', alpha=0.5)

    if show_legend:
        txt = f"rb = -1\nσ = {rms*1e6:.2f} µJy/bm\npk = {peak*1e6:.2f} µJy/bm"
        ax.text(0.98, 0.98, txt, transform=ax.transAxes, va='top', ha='right', fontsize=13,
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')], zorder=10)

    if title:
        ax.set_title(title, fontsize=12, pad=8, fontweight='bold')

    return im

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fits_files', nargs='+', help="One or more FITS files to plot side by side")
    parser.add_argument('--titles', nargs='+', default=['C200 (2h)', 'C300 (2h)'], help="Titles above each subplot")
    parser.add_argument('--output', default='side_by_side.png')
    parser.add_argument('--nsig', type=int, default=5)
    args = parser.parse_args()

    fits_paths = args.fits_files
    n = len(fits_paths)
    if n < 1:
        raise ValueError("At least one FITS file is required.")

    rms_list, pk_list, wcs_list, hdrs = [], [], [], []
    for path in fits_paths:
        rms, pk = get_rms_and_peak(path)
        rms_list.append(rms)
        pk_list.append(pk)
        hdrs.append(pyfits.getheader(path))
        wcs_list.append(WCS(hdrs[-1]).celestial)

    rms_max = max(rms_list)
    σ = rms_max * 1e6
    vmin, vmax = -args.nsig * σ, args.nsig * σ

    fig_width = 4.7 * n
    fig = plt.figure(figsize=(fig_width, 3.8))
    gs = GridSpec(2, n, height_ratios=[18, 1], hspace=0.2, wspace=0.05)

    axes = [fig.add_subplot(gs[0, i], projection=wcs_list[i]) for i in range(n)]
    cax = fig.add_subplot(gs[1, :])

    im = None
    labels = [f"({chr(97+i)})" for i in range(n)]
    titles = args.titles if len(args.titles) == n else ["" for _ in range(n)]

    for i, (ax, path, rms, peak, title) in enumerate(zip(axes, fits_paths, rms_list, pk_list, titles)):
        im = plot_wcs_panel(ax, path, rms, peak, vmin, vmax, title=title)
        ax.text(0.02, 1.02, labels[i], transform=ax.transAxes, fontsize=13, fontweight='bold', va='bottom')
        #if i != 0:
         #   ax.coords['dec'].set_ticklabel_visible(False)

    cbar = fig.colorbar(im, cax=cax, orientation='horizontal', extend='both')
    ticks = np.array([-5, -3, 0, +3, +5]) * σ
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(["−5σ", "−3σ", "0", "+3σ", "+5σ"])
    cbar.set_label("Flux Density (µJy beam⁻¹)", fontsize=12, fontweight='bold', labelpad=5)

    fig.text(0.5, -0.05, "RA (J2000)", ha='center', va='top', fontsize=12, fontweight='bold')
    fig.text(0.04, 0.5, "Dec (J2000)", ha='left', va='center', rotation='vertical', fontsize=12, fontweight='bold')

    fig.savefig(args.output, dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()
