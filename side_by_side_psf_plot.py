cat > ~/side_by_side_psf_plot.py << 'EOF'
import os
import matplotlib.pyplot as plt
import numpy as np
from casatools import image
from astropy.io import fits

# List your PSF FITS or .psf image files here
psf_files = [
    '~/alma_sims/C43-10.psf',
    '~/alma_sims/C73-OSF.psf',
    '~/alma_sims/C80-OSFp7.psf'
]

titles = ['C43-10', 'C73-OSF', 'C80-OSFp7']
output_file = '/mnt/c/Users/hampu/OneDrive/Skrivbord/alma2040/CASA/psf_comparison.png'

def load_and_crop(imfile, size=30):
    ia = image()
    ia.open(os.path.expanduser(imfile))
    data = ia.getchunk()
    ia.close()
    img = data[:, :, 0, 0]

    # Normalize
    peak = np.nanmax(np.abs(img))
    img = img / peak if peak else img

    # Center crop
    cy, cx = img.shape[0] // 2, img.shape[1] // 2
    return img[cy-size:cy+size, cx-size:cx+size]

def plot_side_by_side(images, titles, output_file):
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    levels = np.array([0.1, 0.2, 0.5, 0.7, 0.9, 1.0])

    for ax, img, title in zip(axes, images, titles):
        im = ax.imshow(img, origin='lower', cmap='RdGy_r', vmin=-0.2, vmax=1.0)
        ax.contour(img, levels=levels, colors='black', linewidths=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=11, fontweight='bold')

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label('Normalized Intensity', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.close()

if __name__ == '__main__':
    images = [load_and_crop(fp) for fp in psf_files]
    plot_side_by_side(images, titles, output_file)
    print(f"✅ Saved to {output_file}")
EOF
chmod +x ~/side_by_side_psf_plot.py
