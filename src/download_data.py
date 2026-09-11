"""
Descarga el dataset "Individual Household Electric Power Consumption"
del repositorio UCI y lo deja listo en data/raw/.

Uso:
    python src/download_data.py
"""

import zipfile
from pathlib import Path

import requests

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
ZIP_PATH = RAW_DIR / "household_power_consumption.zip"
CSV_NAME = "household_power_consumption.txt"


def download_zip(url: str = UCI_URL, dest: Path = ZIP_PATH) -> Path:
    if dest.exists():
        print(f"Ya existe {dest}, no se vuelve a descargar.")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando dataset desde:\n  {url}")

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:5.1f}%", end="", flush=True)
    print("\nDescarga completa.")
    return dest


def extract(zip_path: Path = ZIP_PATH, dest_dir: Path = RAW_DIR) -> Path:
    csv_path = dest_dir / CSV_NAME
    if csv_path.exists():
        print(f"Ya existe {csv_path}, no se vuelve a extraer.")
        return csv_path

    print(f"Extrayendo {zip_path.name}...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    print(f"Listo: {csv_path}")
    return csv_path


if __name__ == "__main__":
    zip_path = download_zip()
    extract(zip_path)
