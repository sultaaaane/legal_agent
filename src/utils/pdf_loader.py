from pathlib import Path


def load_contract(path: Path) -> str:
    """Load a contract from .txt or .pdf. Returns clean plain text."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Contract not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        return _load_pdf(path)

    raise ValueError(f"Unsupported file type: {suffix}. Use .txt or .pdf")


def _load_pdf(path: Path) -> str:
    """Try pdfplumber first (handles structured PDFs), fall back to OCR."""
    import pdfplumber

    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    text = "\n\n".join(text_parts).strip()

    # If pdfplumber got nothing (scanned PDF), fall back to OCR
    if len(text) < 100:
        text = _ocr_pdf(path)

    return text


def _ocr_pdf(path: Path) -> str:
    """OCR fallback for scanned PDFs using pytesseract."""
    import pytesseract
    from pdf2image import convert_from_path

    images = convert_from_path(path, dpi=300)
    pages = [pytesseract.image_to_string(img) for img in images]
    return "\n\n".join(pages).strip()
