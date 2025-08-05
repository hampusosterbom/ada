cat > ~/generate_sky_model.py << 'EOF'
import os
import sys
import argparse
import logging
from casatools import quanta, componentlist, image
from casatasks import exportfits
from math import sin, cos, pi

def setup_logging(log_file=None):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(handler)

def validate_ra_dec(value, type_str):
    logging.debug(f"Validating {type_str}: {value}")
    try:
        qa = quanta()
        result = qa.convert(value, "rad")
        logging.debug(f"Successfully converted {type_str} {value} to {result}")
        return value
    except Exception as e:
        logging.error(f"Failed to validate {type_str} {value}: {e}")
        raise ValueError(f"Invalid {type_str} format: {value}. Expected format like '12h00m00.00s' for RA or '-70d00m00.00' for Dec.")

def main(args):
    setup_logging(args.log_file)
    logging.info(f"Generating sky model with declination {args.dec_center}")

    try:
        qa_tool = quanta()
        comp_list = componentlist()
        img_tool = image()
        comp_list.done()

        comp_list.addcomponent(
            dir=f"J2000 {args.ra_center} {args.dec_center}",
            flux=args.flux,
            fluxunit="Jy",
            freq=args.freq,
            shape="Gaussian",
            majoraxis=args.major_beam,
            minoraxis=args.minor_beam,
            positionangle=args.pa_beam
        )

        dec_rad = qa_tool.convert(args.dec_center, "rad")
        cos_dec = cos(dec_rad["value"])
        for i in range(args.n_ring):
            ang = 2 * pi * i / args.n_ring
            dx_arcsec = args.radius * cos(ang) / cos_dec
            dy_arcsec = args.radius * sin(ang)
            ra_rad = qa_tool.convert(args.ra_center, "rad")
            dec_rad = qa_tool.convert(args.dec_center, "rad")
            dx_rad = qa_tool.convert(f"{dx_arcsec}arcsec", "rad")
            dy_rad = qa_tool.convert(f"{dy_arcsec}arcsec", "rad")
            new_ra_rec = qa_tool.add(ra_rad, dx_rad)
            new_dec_rec = qa_tool.add(dec_rad, dy_rad)
            offs_ra = qa_tool.tos(new_ra_rec)
            offs_dec = qa_tool.tos(new_dec_rec)
            comp_list.addcomponent(
                dir=f"J2000 {offs_ra} {offs_dec}",
                flux=args.flux,
                fluxunit="Jy",
                freq=args.freq,
                shape="Gaussian",
                majoraxis=args.major_beam,
                minoraxis=args.minor_beam,
                positionangle=args.pa_beam
            )

        dec_tag = args.dec_center.replace('-', 'm').replace('+', 'p').replace('d', '').replace('m', '').replace('.', '')
        imagename = f"{args.output_base}_dec{dec_tag}"
        im_shape = list(args.im_shape) + [1, 1]
        img_tool.fromshape(f"{imagename}.im", im_shape, overwrite=True)
        cs = img_tool.coordsys()
        cs.setunits(["rad", "rad", "", "Hz"])
        cs.setreferencevalue(
            [qa_tool.convert(args.ra_center, "rad")["value"], qa_tool.convert(args.dec_center, "rad")["value"]],
            type="direction"
        )
        cell = qa_tool.convert(f"{args.cell_size}arcsec", "rad")["value"]
        cs.setincrement([-cell, cell], "direction")
        cs.setreferencevalue(args.freq, "spectral")
        cs.setreferencepixel([args.im_shape[0] // 2, args.im_shape[1] // 2, 0, 0])
        cs.setincrement("7.5GHz", "spectral")
        img_tool.setcoordsys(cs.torecord())
        img_tool.setbrightnessunit("Jy/pixel")

        img_tool.modify(comp_list.torecord(), subtract=False)
        img_tool.done()
        exportfits(
            imagename=f"{imagename}.im",
            fitsimage=f"{imagename}.fits",
            overwrite=True
        )
        logging.info(f"✅ Done: {imagename}.fits created at declination {args.dec_center}")
    except Exception as e:
        logging.error(f"Failed to generate sky model: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.debug(f"sys.argv: {sys.argv}")
    logging.debug(f"CASA_ARGUMENTS: {os.environ.get('CASA_ARGUMENTS', 'Not set')}")

    # JSON support block
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        import json
        from argparse import Namespace
        with open(sys.argv[1]) as f:
            config = json.load(f)
        args = Namespace(**config)
        args.ra_center = validate_ra_dec(args.ra_center, "RA")
        args.dec_center = validate_ra_dec(args.dec_center, "Dec")
        main(args)
        sys.exit(0)

    # Restore this fallback for non-JSON command-line use
    if len(sys.argv) == 1:
        logging.debug("No command-line args; checking CASA_ARGUMENTS and DEC")
        raw_args = os.environ.get("CASA_ARGUMENTS", "")
        if not raw_args and 'DEC' in os.environ:
            raw_args = f"--dec_center={os.environ['DEC']} --log_file=sky_test.log"
        args_list = raw_args.split() if raw_args else []
    else:
        args_list = sys.argv[1:]


    parser = argparse.ArgumentParser(description="Generate ALMA Sky Model (8+1 Gaussians)", allow_abbrev=False)
    parser.add_argument('--ra_center', default="12h00m00.00s", type=str, help="RA center (e.g., 12h00m00.00s)")
    parser.add_argument('--dec_center', default=os.environ.get("DEC", "-23d00m00.00"), type=str, help="Dec center")
    parser.add_argument('--flux', type=float, default=27e-5, help="Flux in Jy")
    parser.add_argument('--freq', default="343.5GHz", help="Frequency")
    parser.add_argument('--n_ring', type=int, default=8, help="Number of ring sources")
    parser.add_argument('--radius', type=float, default=0.044, help="Ring radius in arcsec")
    parser.add_argument('--major_beam', default="0.022arcsec", help="Gaussian major axis")
    parser.add_argument('--minor_beam', default="0.022arcsec", help="Gaussian minor axis")
    parser.add_argument('--pa_beam', default="0deg", help="Position angle")
    parser.add_argument('--output_base', default="eightPlusCenter", help="Base name for output files")
    parser.add_argument('--im_shape', type=int, nargs=2, default=[128, 128], help="Image shape [x, y]")
    parser.add_argument('--cell_size', type=float, default=0.0044, help="Cell size in arcsec")
    parser.add_argument('--log_file', default=None, help="Path to log file")
    
    try:
        args = parser.parse_args(args_list)
    except SystemExit as e:
        logging.error(f"Argument parsing failed: {e}")
        logging.error(f"Attempted args: {args_list}")
        raise

    args.ra_center = validate_ra_dec(args.ra_center, "RA")
    args.dec_center = validate_ra_dec(args.dec_center, "Dec")
    main(args)

EOF
chmod +x ~/generate_sky_model.py