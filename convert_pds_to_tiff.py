"""
Convert PDS4 (.xml + .img) or PDS3 (.lbl + .img) files to GeoTIFF format
for use with the Lunar Registration Portal.

Usage:
  python convert_pds_to_tiff.py <path_to_extracted_bundle_folder> [output.tif]
  
Example:
  python convert_pds_to_tiff.py "C:\Users\arunk\Downloads\ch1_tmc_ncn_20090529T..._Bundle\data" source.tif
"""
import sys
import glob
import os
import rasterio
import numpy as np
from pathlib import Path


def find_pds4_xml(folder: str) -> str:
    """Find the PDS4 .xml label file in a folder."""
    xmls = glob.glob(os.path.join(folder, "*.xml"))
    # Filter out browse/context XMLs, prefer the one referencing image data
    for xml in xmls:
        with open(xml, 'r', errors='ignore') as f:
            content = f.read(2000)
            if 'Array_2D' in content or 'Array_3D' in content or '.img' in content:
                return xml
    # Fallback: return any xml
    if xmls:
        return xmls[0]
    return ""


def convert_pds_to_tiff(input_path: str, output_path: str):
    """Convert a PDS4 XML label or raw .img to GeoTIFF."""
    input_path = str(Path(input_path).resolve())
    
    # If input is a directory, search for the XML label
    if os.path.isdir(input_path):
        xml_path = find_pds4_xml(input_path)
        if xml_path:
            print(f"Found PDS4 label: {xml_path}")
            input_path = xml_path
        else:
            # Try .lbl (PDS3) or .img files
            lbls = glob.glob(os.path.join(input_path, "*.lbl"))
            imgs = glob.glob(os.path.join(input_path, "*.img"))
            if lbls:
                input_path = lbls[0]
                print(f"Found PDS3 label: {input_path}")
            elif imgs:
                input_path = imgs[0]
                print(f"Found raw .img: {input_path}")
            else:
                print("ERROR: No PDS4 (.xml), PDS3 (.lbl), or .img files found in folder.")
                sys.exit(1)

    print(f"Opening: {input_path}")
    
    try:
        with rasterio.open(input_path) as src:
            print(f"  Size: {src.width} x {src.height}, Bands: {src.count}, Dtype: {src.dtypes[0]}")
            print(f"  CRS: {src.crs}")
            
            profile = src.profile.copy()
            profile.update(driver='GTiff', compress='deflate', tiled=True, blockxsize=256, blockysize=256)
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    data = src.read(i)
                    dst.write(data, i)
            
            print(f"  Written to: {output_path}")
            print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
            print("SUCCESS!")
            
    except Exception as e:
        print(f"ERROR opening with rasterio: {e}")
        print("\nTrying raw binary fallback...")
        
        # Try reading as raw binary with common TMC dimensions
        img_files = glob.glob(os.path.join(os.path.dirname(input_path), "*.img"))
        if not img_files:
            img_files = [input_path]
        
        for img_file in img_files:
            fsize = os.path.getsize(img_file)
            print(f"  Raw file: {img_file} ({fsize / (1024*1024):.1f} MB)")
            
            # TMC-2 typical dimensions: samples x lines, 16-bit
            # Try to guess dimensions from file size
            for dtype, bpp in [('uint16', 2), ('uint8', 1), ('float32', 4)]:
                npixels = fsize // bpp
                # Try common widths
                for width in [4096, 2048, 1024, 512, 256]:
                    if npixels % width == 0:
                        height = npixels // width
                        if 100 < height < 1000000:  # reasonable range
                            print(f"  Guessed: {width} x {height}, dtype={dtype}")
                            data = np.fromfile(img_file, dtype=dtype).reshape(height, width)
                            
                            with rasterio.open(output_path, 'w', driver='GTiff',
                                             height=height, width=width, count=1,
                                             dtype=dtype, compress='deflate') as dst:
                                dst.write(data, 1)
                            
                            print(f"  Written to: {output_path}")
                            print("SUCCESS (raw binary conversion)!")
                            return
            
        print("ERROR: Could not determine image dimensions. Please provide the .xml label file.")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "converted_output.tif"
    
    convert_pds_to_tiff(input_path, output_path)
