import re
import pandas as pd
from pyproj import CRS, Transformer
import simplekml
import argparse

def parse_cfg(path):
    rows = []
    with open(path) as fh:
        for L in fh:
            L = L.strip()
            if not L or L.startswith('#'):
                continue
            parts = re.split(r"\s+", L)
            if len(parts) == 5:
                x, y, z, diam, name = parts
            elif len(parts) == 4:
                name, x, y, z = parts
                diam = '12.'
            else:
                continue
            rows.append({
                'name': name,
                'x': float(x),
                'y': float(y),
                'z': float(z),
                'diam': float(diam)
            })
    return pd.DataFrame(rows)
#"C:/Users/hampu/OneDrive/Skrivbord/alma2040/CASA/DDP/finalCFGs/48kMaxBL/200ants/reduced_pads.cfg"
def main():
    parser = argparse.ArgumentParser(description="Convert CFG to KML for Google Earth")
    parser.add_argument('--cfg_path', default=r"C:\Users\hampu\OneDrive\Skrivbord\alma2040\CASA\realgg\alma.cycle80_v2.cfg", help="Path to input CFG file")
    parser.add_argument('--kml_path', default="C:/Users/hampu/OneDrive/Skrivbord/alma2040/GoogleEarth/C80.kml", help="Path to output KML file")
    args = parser.parse_args()
    #args.cfg_path = "C:/Users/hampu/OneDrive/Skrivbord/alma2040/customCfgs/alma.cycle11.10.120_OSFV3.cfg"
    # Load pads from CFG
    df = parse_cfg(args.cfg_path)

    # Setup projection transformation (from local projected to WGS84 lat/long)
    lat0, lon0 = -23.029, -67.755000
    proj4 = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    crs_proj = CRS.from_proj4(proj4)
    crs_wgs = CRS.from_epsg(4326)  # WGS84
    transformer = Transformer.from_crs(crs_proj, crs_wgs, always_xy=True)

    # Create KML file
    kml = simplekml.Kml(name="ALMA Pads")

    for _, row in df.iterrows():
        # Transform (x, y) to (lon, lat)
        lon, lat = transformer.transform(row['x'], row['y'])
        
        # Create Placemark
        pnt = kml.newpoint(name=row['name'], coords=[(lon, lat)])
        # Optional: Use absolute altitude if z is meaningful (uncomment if needed)
        # pnt.altitudemode = simplekml.AltitudeMode.absolute
        # pnt.coords = [(lon, lat, row['z'])]

    # Save KML
    kml.save(args.kml_path)
    print(f"Saved KML to {args.kml_path}. Open in Google Earth to view markers.")

if __name__ == "__main__":
    main()