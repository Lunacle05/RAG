import os
import sys
import traceback
import subprocess
from pathlib import Path

from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import pytesseract
except Exception as e:
    print("[FATAL] import pytesseract failed:", repr(e))
    raise

IMG_DIR = Path("/root/autodl-tmp/RAG/data/.rag_pdf_images")
VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def run_cmd(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return -1, "", repr(e)


def check_tesseract_env():
    print("=" * 100)
    print("[1] Tesseract environment check")

    print("\n[pytesseract version]")
    print(getattr(pytesseract, "__version__", "unknown"))

    print("\n[pytesseract tesseract_cmd]")
    print(pytesseract.pytesseract.tesseract_cmd)

    rc, out, err = run_cmd(["tesseract", "--version"])
    print("\n[tesseract --version]")
    print("returncode:", rc)
    print("stdout:", out[:2000] if out else "[EMPTY]")
    print("stderr:", err[:1000] if err else "[EMPTY]")

    rc, out, err = run_cmd(["tesseract", "--list-langs"])
    print("\n[tesseract --list-langs]")
    print("returncode:", rc)
    print("stdout:", out[:4000] if out else "[EMPTY]")
    print("stderr:", err[:1000] if err else "[EMPTY]")

    langs = set()
    if out:
        for line in out.splitlines():
            line = line.strip()
            if line and not line.lower().startswith("list of available languages"):
                langs.add(line)

    print("\n[language check]")
    print("has eng    :", "eng" in langs)
    print("has chi_sim:", "chi_sim" in langs)
    print("has chi_tra:", "chi_tra" in langs)


def safe_image_info(path: Path):
    try:
        with Image.open(path) as img:
            return {
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
            }
    except Exception as e:
        return {
            "error": repr(e)
        }


def try_ocr(path: Path, lang: str):
    try:
        with Image.open(path) as img:
            text = pytesseract.image_to_string(img, lang=lang)
        text = (text or "").strip()
        return {
            "ok": True,
            "len": len(text),
            "text": text,
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "len": 0,
            "text": "",
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }


def main():
    check_tesseract_env()

    print("\n" + "=" * 100)
    print("[2] OCR images check")

    if not IMG_DIR.exists():
        print(f"[FATAL] image dir not found: {IMG_DIR}")
        return

    files = sorted(
        [p for p in IMG_DIR.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    )

    print(f"image dir: {IMG_DIR}")
    print(f"total images: {len(files)}")

    if not files:
        return

    empty_count = 0
    error_count = 0
    nonempty_count = 0

    for i, path in enumerate(files, 1):
        print("\n" + "=" * 100)
        print(f"[{i}] {path.name}")
        print("path:", path)
        print("file_size_bytes:", path.stat().st_size)

        info = safe_image_info(path)
        print("image_info:", info)

        # 先测 chi_sim+eng
        result_main = try_ocr(path, "chi_sim+eng")
        print("\n[OCR lang=chi_sim+eng]")
        print("ok:", result_main["ok"])
        print("len:", result_main["len"])

        if not result_main["ok"]:
            error_count += 1
            print("error:", result_main["error"])
            print(result_main.get("traceback", "")[:3000])
        elif result_main["len"] == 0:
            empty_count += 1
            print("text:", "[EMPTY]")
        else:
            nonempty_count += 1
            print("text preview:")
            print(result_main["text"][:1000])

        # 如果主 OCR 为空，再单独测 eng 看是不是中文包问题
        if result_main["ok"] and result_main["len"] == 0:
            result_eng = try_ocr(path, "eng")
            print("\n[Fallback OCR lang=eng]")
            print("ok:", result_eng["ok"])
            print("len:", result_eng["len"])
            if result_eng["ok"] and result_eng["len"] > 0:
                print("text preview:")
                print(result_eng["text"][:1000])
            elif not result_eng["ok"]:
                print("error:", result_eng["error"])
            else:
                print("text:", "[EMPTY]")

    print("\n" + "=" * 100)
    print("[3] Summary")
    print("total     :", len(files))
    print("nonempty  :", nonempty_count)
    print("empty     :", empty_count)
    print("errors    :", error_count)


if __name__ == "__main__":
    main()
