import img2pdf
from pathlib import Path
import re

folder = Path(r"/home/namdp/Downloads/cosmo_data/output_quyen3")
output_pdf = Path(r"/home/namdp/Downloads/cosmo_data_outputQuyen3_end.pdf")

def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', p.stem)]

png_files = sorted(folder.glob("*.png"), key=natural_key)

if not png_files:
    raise FileNotFoundError("Không tìm thấy ảnh PNG.")

with open(output_pdf, "wb") as f:
    f.write(img2pdf.convert([str(p) for p in png_files]))
print(f"Đã tạo PDF: {output_pdf}")
