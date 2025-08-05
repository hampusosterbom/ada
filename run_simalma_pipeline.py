cat > ~/run_simalma_pipeline.py << 'EOF'
import os
import json
import numpy as np
from math import pi
from pathlib import Path
from casatools import imager, image
from casatasks import simalma, tclean, exportfits, imhead
import argparse
import logging
import subprocess

# Logging setup
def setup_logging(log_file=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(handler)

# Sky model generation
def generate_sky_model(dec, sky_script, fitsname, casa_bin):
    import json
    logging.info(f"Generating sky model for declination {dec} → {fitsname} (overwriting if exists)")

    if Path(fitsname).exists():
        Path(fitsname).unlink()

    # Load the base config
    base_config_path = Path("sky_config.json")
    if not base_config_path.exists():
        raise FileNotFoundError("Missing sky_config.json in working directory!")

    with open(base_config_path) as f:
        sky_config = json.load(f)

    # Update the declination
    sky_config["dec_center"] = dec

    # Construct a unique filename for the temp config
    tag = dec.replace('-', 'm').replace('+', 'p').replace('d', '').replace('.', '')
    temp_config_path = Path(f"skyconfig_{tag}.json")
    with open(temp_config_path, "w") as f:
        json.dump(sky_config, f, indent=2)

    # Run CASA with the JSON config
    cmd = [casa_bin, '--nogui', '--nologger', '-c', str(sky_script), str(temp_config_path)]
    logging.info(f"Running sky generator with: {cmd}")
    subprocess.run(cmd, check=True)

    if not Path(fitsname).exists():
        raise RuntimeError(f"Failed to generate sky model {fitsname} after execution.")


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def search_config_path(cfg, casa_bin=None, extra_dirs=None):
    """
    Search for an ALMA config file in likely directories.
    
    Args:
        cfg (str): Name of the config file (e.g., 'alma.cycle11.10_80.cfg').
        casa_bin (str, optional): Path to CASA binary, used to infer CASA data dir.
        extra_dirs (list, optional): Additional directories to search.
    
    Returns:
        Path: Path to the config file.
    
    Raises:
        FileNotFoundError: If the config file is not found in any candidate path.
    """
    candidates = []
    simmos_dir = Path.home() / '.casa' / 'data' / 'alma' / 'simmos'
    candidates.append(simmos_dir / cfg)
    candidates.append(Path.home() / '.casa' / cfg)
    if casa_bin:
        casa_root = Path(casa_bin).parent.parent
        candidates.append(casa_root / 'data' / 'alma' / 'simmos' / cfg)
    if extra_dirs:
        for d in extra_dirs:
            candidates.append(Path(d) / cfg)
    candidates.append(Path.cwd() / cfg)
    for p in candidates:
        if p.exists():
            logging.info(f"Found config file at {p}")
            return p
    raise FileNotFoundError(f"Config {cfg!r} not found in candidate paths: {candidates}")

def find_config_path(cfg, casa_bin=None, extra_dirs=None):
    """Find the path to an ALMA config file."""
    return search_config_path(cfg, casa_bin, extra_dirs)

def get_max_baseline(cfg, casa_bin=None, extra_dirs=None):
    """Calculate the maximum baseline distance from an ALMA config file."""
    cfg_path = search_config_path(cfg, casa_bin, extra_dirs)
    ants = []
    try:
        with open(cfg_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    x, y, z = map(float, parts[:3])
                    ants.append((x, y, z))
                except ValueError:
                    logging.warning(f"Skipping invalid line in {cfg_path}: {line}")
    except IOError as e:
        raise IOError(f"Failed to read config file {cfg_path}: {e}")
    if not ants:
        raise ValueError(f"No valid antenna positions found in {cfg_path}")
    mb = 0
    for i in range(len(ants)):
        for j in range(i + 1, len(ants)):
            dx, dy, dz = (ants[i][k] - ants[j][k] for k in range(3))
            d = (dx * dx + dy * dy + dz * dz) ** 0.5
            if d > mb:
                mb = d
    logging.info(f"Maximum baseline for {cfg}: {mb:.2f} meters")
    return mb

def load_antenna_positions(cfg_path):
    cfg_path = Path(cfg_path)
    ants = []
    with open(cfg_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            x, y, z = map(float, parts[:3])
            ants.append((x, y, z))
    if not ants:
        raise ValueError("No valid antenna positions found in the file.")
    return ants

def get_beam_size(freq, max_b):
    lam = 3e8 / (freq * 1e9)
    return lam / max_b * (180 / pi) * 3600

def compute_lambda(freq_ghz):
    c = 3e8  # speed of light, m/s
    freq_hz = freq_ghz * 1e9
    return c / freq_hz

def get_rms_casa(image_name):
    """
    Compute RMS of the background in a CASA image by masking the central region.
    """
    ia = image()
    ia.open(image_name)
    shape = ia.shape()
    data = ia.getchunk()
    mask = ia.getchunk(getmask=True)
    ia.close()
    nx, ny = shape[:2]
    area_frac = 0.5
    lin_frac = np.sqrt(area_frac)
    half_x = int(lin_frac * nx / 2)
    half_y = int(lin_frac * ny / 2)
    cx, cy = nx // 2, ny // 2
    x_start = max(0, cx - half_x)
    x_end = min(nx, cx + half_x)
    y_start = max(0, cy - half_y)
    y_end = min(ny, cy + half_y)
    new_mask = mask.copy()
    slices = [slice(x_start, x_end), slice(y_start, y_end)] + [slice(None)] * (len(shape) - 2)
    new_mask[tuple(slices)] = False
    valid_data = data[new_mask]
    if valid_data.size == 0:
        return 0.0
    return np.sqrt(np.mean(valid_data ** 2))

def get_peak(im):
    ia = image()
    ia.open(im)
    stats = ia.statistics()
    ia.close()
    return stats['max'][0]

def next_smooth(target):
    """
    Find the smallest number >= target with prime factors only 2, 3, 5, and ensure it is even.
    """
    candidates = []
    max_exp = 20
    for a in range(1, max_exp):
        for b in range(max_exp):
            for c in range(max_exp):
                s = (2 ** a) * (3 ** b) * (5 ** c)
                if s >= target and s < target * 2:
                    candidates.append(s)
    if not candidates:
        raise ValueError(f"No smooth number found near {target}; increase max_exp")
    return min(candidates)

def make_short_tag(sky, cfg_short, itime, dec, wt, rb, tap):
    sky_t = sky.replace('ninePlusCenter', '9C')
    it_t = 't' + itime.rstrip('h')
    sign = 'm' if dec.startswith('-') else 'p'
    val = dec.lstrip('+-').split('d')[0]
    dec_t = f"{sign}{val}"
    wm = {'natural': 'N', 'uniform': 'U', 'briggs': 'B'}[wt]
    rb_t = f"r{str(rb).replace('.', 'p')}" if wt == 'briggs' and rb is not None else ''
    tap_t = 't' + tap.replace('arcsec', '').replace('.', 'p')
    return '_'.join([sky_t, cfg_short, it_t, dec_t, wm + rb_t, tap_t])

def main(args):
    setup_logging(args.log_file)
    try:
        root_dir = Path(__file__).parent.resolve()
        proj_base_dir = root_dir / args.project_base
        proj_base_dir.mkdir(exist_ok=True)

        config_map = args.config_map
        config_files = list(config_map.keys())

        fits_inputs = []
        for dec in args.declinations:
            tag = dec.replace('-', 'm').replace('+','p').replace('d','').replace('m','').replace('.','')
            fitsname = f"eightPlusCenter_dec{tag}.fits"
            generate_sky_model(dec, args.sky_script, fitsname, args.casa_bin)
            fits_inputs.append((fitsname, dec))

        log_data = []

        for fits, dec in fits_inputs:
            fits_path = root_dir / fits
            fits_name = fits_path.stem

            for config in config_files:
                config_tag = config_map[config]
                cfg_path = find_config_path(config, args.casa_bin, args.extra_dirs)
                ants = load_antenna_positions(cfg_path)
                lam = compute_lambda(float(args.center_freq.rstrip('GHz')))

                for itime in args.integration_times:
                    mb = get_max_baseline(config, args.casa_bin, args.extra_dirs)
                    beam = get_beam_size(float(args.center_freq.rstrip('GHz')), mb)
                    dyn_cell = f"{beam / args.sampling_factor}arcsec"
                    logging.info(f'Estimated beam in arcsec: {beam}')
                    pixsize = beam / args.sampling_factor

                    imsize_raw = int(np.ceil(args.fov_arcsec / pixsize))
                    if imsize_raw % 2 == 1:
                        imsize_raw += 1
                    imsize = next_smooth(imsize_raw)
                    logging.info(f"Adjusted imsize from {imsize_raw} to efficient {imsize}")

                    project = f"{args.project_base}_{fits_name}_{config_tag}_{itime}_{dec.replace('d', '_').replace('m', '').replace('.', '')}"
                    pf = proj_base_dir / project
                    pf.mkdir(exist_ok=True)
                    os.chdir(proj_base_dir)

                    sim_params = {
                        'project': project,
                        'skymodel': str(fits_path),
                        'incenter': args.center_freq,
                        'inwidth': args.width,
                        'integration': '10s',
                        'totaltime': itime,
                        'hourangle': 'transit',
                        'pwv': args.pwv,
                        'cell': dyn_cell,
                        'antennalist': str(cfg_path),
                        'image': True,
                        'setpointings': True,
                        'graphics': 'both',
                        'verbose': True,
                        'indirection': f"J2000 {args.phasecenter_ra} {dec}",
                        'overwrite': True,
                        'dryrun': False,
                        'mapsize': [f"{args.fov_arcsec}arcsec"]
                    }
                    logging.info(f"simalma: {sim_params}")
                    simalma(**sim_params)

                    os.chdir(pf)
                    ms_files = list(Path('.').glob('*.noisy.ms'))
                    if len(ms_files) != 1:
                        raise RuntimeError(f"Expected one .noisy.ms in {pf}, found: {ms_files}")
                    vis = str(ms_files[0])

                    for wt in args.weightings:
                        robusts = args.robust_values if wt == 'briggs' else [None]
                        for rb in robusts:
                            for tap in args.uvtaper_values:
                                tag = make_short_tag(fits_name, config_tag, itime, dec, wt, rb, tap)
                                img = f"{tag}_d{imsize}"
                                dirty = f"{img}_dirty"

                                tclean_params_dirty = {
                                    'vis': vis,
                                    'imagename': dirty,
                                    'weighting': wt,
                                    'cell': dyn_cell,
                                    'imsize': imsize,
                                    'niter': 0,
                                    'phasecenter': f"J2000 {args.phasecenter_ra} {dec}",
                                    'uvtaper': [tap],
                                    'interactive': False,
                                    'calcpsf': True,
                                    'calcres': True,
                                }
                                if wt == 'briggs' and rb is not None:
                                    tclean_params_dirty['robust'] = rb

                                logging.info(f"Running tclean (dirty) with parameters: {tclean_params_dirty}")
                                tclean(**tclean_params_dirty)

                                ia = image()
                                ia.open(f"{dirty}.psf")
                                summary = ia.summary()
                                ia.close()

                                beam_major = summary['restoringbeam']['major']['value']
                                cell_arcsec = float(dyn_cell.rstrip('arcsec'))
                                effective_beam_pix = beam_major / cell_arcsec

                                scalefactors = [1, 2, 5]
                                scales = [0] + [max(1, int(round(f * effective_beam_pix))) for f in scalefactors]
                                scales = sorted(set(scales))
                                logging.info(f"Updated multiscale sizes for taper={tap}: {scales} (beam_pix={effective_beam_pix:.1f})")

                                rms_dirty = get_rms_casa(f"{dirty}.image")
                                thr_str = args.threshold_factor * rms_dirty
                                logging.info(f"rms_dirty = {rms_dirty:.3e} Jy/beam → threshold = {thr_str} Jy/beam")

                                tclean_params = {
                                    'vis': vis,
                                    'imagename': img,
                                    'weighting': wt,
                                    'cell': dyn_cell,
                                    'imsize': imsize,
                                    'deconvolver': args.deconvolver,
                                    'scales': scales,
                                    'niter': args.niter,
                                    'threshold': thr_str,
                                    'phasecenter': f"J2000 {args.phasecenter_ra} {dec}",
                                    'uvtaper': [tap],
                                    'interactive': False,
                                    'savemodel': 'modelcolumn'
                                }
                                if wt == 'briggs' and rb is not None:
                                    tclean_params['robust'] = rb

                                logging.info(f"Running tclean with parameters: {tclean_params}")
                                tclean(**tclean_params)

                                rms_resid = get_rms_casa(f"{img}.residual")
                                logging.info(f"Post-clean residual RMS = {rms_resid:.3e}")

                                imhead(imagename=f"{img}.image", mode='put', hdkey='OBJECT', hdvalue=img)
                                exportfits(imagename=f"{img}.image", fitsimage=f"{img}.image.fits",
                                           dropdeg=True, dropstokes=True, history=False, velocity=False,
                                           optical=False, overwrite=True)

                                rms_cleaned = get_rms_casa(f"{img}.image")
                                logging.info(f'rms_cleaned = {rms_cleaned}')
                                rms_res = get_rms_casa(f"{img}.residual")
                                logging.info(f'rms_cleaned_res = {rms_res}')
                                pk = get_peak(f"{img}.image")
                                ia = image()
                                ia.open(f"{img}.image")
                                bm = ia.summary()['restoringbeam']
                                ia.close()

                                log_entry = {
                                    'fits': fits_name,
                                    'config': config,
                                    'int_time': itime,
                                    'dec': dec,
                                    'weighting': wt,
                                    'uvtaper': tap,
                                    'rms': rms_res,
                                    'peak_flux': pk,
                                    'beam_major': bm['major']['value'],
                                    'beam_minor': bm['minor']['value']
                                }
                                if wt == 'briggs':
                                    log_entry['robust'] = rb
                                log_data.append(log_entry)

                    os.chdir(proj_base_dir)

        with open(proj_base_dir / 'pipeline_log.json', 'w') as f:
            json.dump(log_data, f, indent=4)

        logging.info(f"Pipeline complete. Log at {proj_base_dir / 'pipeline_log.json'}")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        with open(sys.argv[1]) as f:
            config = json.load(f)
        from types import SimpleNamespace
        args = SimpleNamespace(**config)
        main(args)
    else:
        parser = argparse.ArgumentParser(description="ALMA Simulation Pipeline")
        parser.add_argument('--project_base', default="alma2040_pipeline", help="Base project directory name")
        parser.add_argument('--config_map', type=json.loads, default='{"alma.cycle11.10.cfg": "C43-10test"}', help="JSON dict of config files to tags")
        parser.add_argument('--integration_times', action='append', default=['0.5h'], help="Integration times (e.g., '0.5h' '2h' '8h')")
        parser.add_argument('--declinations', action='append', default=['-70d00m00.00', '-35d00m00.00', '+10d00m00.00'], help="Declinations")
        parser.add_argument('--center_freq', default='343.5GHz', help="Center frequency")
        parser.add_argument('--width', default='7.5GHz', help="Bandwidth")
        parser.add_argument('--pwv', type=float, default=1.262, help="Precipitable water vapor")
        parser.add_argument('--phasecenter_ra', default='12h00m00.00s', help="Phase center RA")
        parser.add_argument('--weightings', action='append', default=['briggs'], help="Weighting schemes")
        parser.add_argument('--robust_values', action='append', type=float, default=[-1], help="Robust values for briggs")
        parser.add_argument('--uvtaper_values', action='append', default=['0.0arcsec'], help="UV tapers")
        parser.add_argument('--niter', type=int, default=7000, help="Number of iterations for tclean")
        parser.add_argument('--deconvolver', default='multiscale', help="Deconvolver for tclean")
        parser.add_argument('--threshold_factor', type=float, default=1.5, help="Threshold factor for tclean")
        parser.add_argument('--fov_arcsec', type=float, default=0.11264, help="Field of view in arcsec")
        parser.add_argument('--sampling_factor', type=float, default=5, help="Sampling factor for cell size")
        parser.add_argument('--sky_script', default="generate_sky_model.py", help="Path to sky model script")
        parser.add_argument('--casa_bin', default=os.environ.get('CASA_PATH'), help="Path to CASA binary")
        parser.add_argument('--extra_dirs', action='append', default=[], help="Additional directories to search for config files")
        parser.add_argument('--log_file', default=None, help="Path to log file (optional)")

        args = parser.parse_args()
        main(args)

EOF
chmod +x ~/run_simalma_pipeline.py