from pathlib import Path


def load_contract(path: str | Path) -> str:
    """
    Load a contract from a .txt or .pdf file.
    Returns clean plain text ready to feed into the graph.

    Raises:
        FileNotFoundError — file does not exist
        ValueError        — unsupported file type
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Contract file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return _load_text(path)

    if suffix == ".pdf":
        return _load_pdf(path)

    raise ValueError(
        f"Unsupported file type '{suffix}'. "
        f"Only .txt and .pdf are supported."
    )


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------

def _load_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Contract file is empty: {path}")
    return text


def _load_pdf(path: Path) -> str:
    """
    Strategy:
    1. Try pdfplumber  — works well for digitally created PDFs (text layer present)
    2. Fall back to OCR — for scanned PDFs with no text layer
    """
    text = _extract_with_pdfplumber(path)

    # If we got very little text, the PDF is probably scanned — try OCR
    if len(text.strip()) < 200:
        print(f"[pdf_loader] Low text yield from pdfplumber ({len(text)} chars), trying OCR...")
        text = _extract_with_ocr(path)

    if not text.strip():
        raise ValueError(
            f"Could not extract text from PDF: {path}\n"
            f"Make sure the file is not password-protected."
        )

    return text


def _extract_with_pdfplumber(path: Path) -> str:
    """Extract text using pdfplumber (best for structured PDFs)."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required: pip install pdfplumber")

    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)

    return "\n\n".join(pages)


def _extract_with_ocr(path: Path) -> str:
    """OCR fallback for scanned PDFs using pytesseract."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
        from PIL import Image
    except ImportError:
        raise ImportError(
            "OCR dependencies not installed. Run:\n"
            "  pip install pytesseract pdf2image Pillow\n"
            "  # Also install Tesseract: https://github.com/tesseract-ocr/tesseract"
        )

    print("[pdf_loader] Running OCR — this may take a minute for long contracts...")
    images = convert_from_path(path, dpi=300)
    pages  = [pytesseract.image_to_string(img, lang="eng") for img in images]
    return "\n\n".join(pages)
