"""
OCR Fallback service for IntelliAssist AI.
Detects scanned/image-only PDFs and provides modular OCR extraction fallback
without disrupting native digital text parsing.
"""

import io
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class OCRService:
    """Provides modular OCR text extraction for scanned document pages and images."""

    def __init__(self):
        self._tesseract_available: Optional[bool] = None

    @staticmethod
    def is_scanned_pdf(text: str) -> bool:
        """Check if extracted text indicates a scanned or image-only PDF (<20 readable characters)."""
        if not text:
            return True
        clean = text.strip()
        return len(clean) < 20

    def process_scanned_pdf(self, file_bytes: bytes, filename: str, total_pages: int = 1) -> Dict[str, Any]:
        """Process a scanned PDF and return document dictionary with warning metadata."""
        pages, warning = self.handle_scanned_pdf_pages(file_bytes, filename, total_pages)
        full_text = "\n\n".join(p.get("text", "") for p in pages)
        return {
            "filename": filename,
            "pages": pages,
            "total_pages": total_pages,
            "full_text": full_text,
            "warning": warning or "OCR fallback active for document.",
            "is_scanned": True
        }

    def is_ocr_available(self) -> bool:
        """Check if pytesseract or native OCR toolchain is installed."""
        if self._tesseract_available is not None:
            return self._tesseract_available
        try:
            import pytesseract
            # Verify tesseract binary path
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
        except Exception:
            self._tesseract_available = False
        return self._tesseract_available

    def process_scanned_page(self, page_image_bytes: bytes, page_number: int = 1) -> str:
        """Perform OCR on an extracted page image if OCR toolchain is installed."""
        if not self.is_ocr_available():
            logger.info("OCR requested for page %d but local tesseract engine is not installed.", page_number)
            return ""

        try:
            import pytesseract
            from PIL import Image
            image = Image.open(io.BytesIO(page_image_bytes))
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.warning("OCR page %d processing failed: %s", page_number, e)
            return ""

    def handle_scanned_pdf_pages(
        self,
        file_bytes: bytes,
        filename: str,
        total_pages: int
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Modular OCR fallback handler when PDF has no selectable text.
        Attempts OCR if toolchain is available, or returns informative diagnostic status.
        """
        if self.is_ocr_available():
            try:
                # If pdf2image / pypdfium2 is installed
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(file_bytes)
                pages_data = []
                for i in range(len(pdf)):
                    page = pdf[i]
                    pil_image = page.render(scale=2.0).to_pil()
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    ocr_text = self.process_scanned_page(img_byte_arr.getvalue(), page_number=i+1)
                    if ocr_text:
                        pages_data.append({
                            "page_number": i + 1,
                            "total_pages": len(pdf),
                            "text": ocr_text,
                            "char_count": len(ocr_text),
                            "word_count": len(ocr_text.split()),
                            "filename": filename,
                            "is_ocr": True
                        })
                if pages_data:
                    return pages_data, None
            except Exception as e:
                logger.debug("PDF image conversion for OCR failed: %s", e)

        warning_msg = (
            f"ℹ️ Scanned document detected: '{filename}' ({total_pages} page(s)) contains image-only or scanned content without selectable text. "
            "Native digital PDFs or OCR-preprocessed documents are recommended for highest extraction accuracy."
        )
        diagnostics = [{
            "page_number": 1,
            "total_pages": total_pages,
            "text": f"Scanned/Image Document: '{filename}' contains {total_pages} page(s) of visual/scanned content. Native text stream was empty.",
            "char_count": 0,
            "word_count": 0,
            "filename": filename,
            "is_scanned": True
        }]
        return diagnostics, warning_msg
