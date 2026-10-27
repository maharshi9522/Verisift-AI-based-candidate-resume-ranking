import re
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import pymupdf
import docx

@dataclass
class TextExtractionResult:
    success: bool
    text: Optional[str]
    error_message: Optional[str]
    total_elements: int = 0  # pages for PDF, blocks for DOCX

class TextExtractionService:
    @staticmethod
    def normalize_text(raw_text: str) -> str:
        """Normalizes excessive whitespace while preserving paragraph and line structure."""
        if not raw_text:
            return ""

        # Normalize line endings
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

        # Strip zero-width or non-printable control characters except \n and \t
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        # Normalize spaces/tabs within lines
        lines = []
        for line in text.split("\n"):
            cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
            lines.append(cleaned_line)

        # Collapse more than 2 consecutive empty lines
        normalized = "\n".join(lines)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
        return normalized

    @classmethod
    def extract_from_pdf(cls, file_source: str | Path | bytes) -> TextExtractionResult:
        """Extracts text page by page from a PDF document using PyMuPDF."""
        doc = None
        try:
            if isinstance(file_source, bytes):
                doc = pymupdf.open(stream=file_source, filetype="pdf")
            else:
                doc = pymupdf.open(str(file_source))

            if doc.is_encrypted:
                return TextExtractionResult(
                    success=False,
                    text=None,
                    error_message="Document is password protected or encrypted."
                )

            page_count = doc.page_count
            if page_count == 0:
                return TextExtractionResult(
                    success=False,
                    text=None,
                    error_message="PDF document contains zero pages."
                )

            extracted_pages = []
            for i in range(page_count):
                page = doc.load_page(i)
                page_text = page.get_text("text")
                if page_text and page_text.strip():
                    extracted_pages.append(page_text.strip())

            raw_combined = "\n\n".join(extracted_pages)
            normalized = cls.normalize_text(raw_combined)

            # Check if text is meaningful (at least 15 alphanumeric characters)
            alpha_num_count = len(re.findall(r"[a-zA-Z0-9]", normalized))
            if alpha_num_count < 15:
                return TextExtractionResult(
                    success=False,
                    text=None,
                    error_message="No extractable text found in PDF document (image-only or scanned documents without OCR are not supported).",
                    total_elements=page_count
                )

            return TextExtractionResult(
                success=True,
                text=normalized,
                error_message=None,
                total_elements=page_count
            )

        except Exception as e:
            return TextExtractionResult(
                success=False,
                text=None,
                error_message=f"Failed to parse PDF document: {type(e).__name__}."
            )
        finally:
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

    @classmethod
    def extract_from_docx(cls, file_source: str | Path | bytes) -> TextExtractionResult:
        """Extracts paragraphs and tables from a DOCX document using python-docx."""
        try:
            import io
            if isinstance(file_source, bytes):
                source = io.BytesIO(file_source)
            else:
                source = str(file_source)

            doc = docx.Document(source)
            content_blocks = []

            # Extract body paragraphs
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    content_blocks.append(text)

            # Extract tables in document
            for table in doc.tables:
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_cells:
                        # Deduplicate repeated cell text from merged cells
                        deduped_cells = []
                        for cell in row_cells:
                            if not deduped_cells or deduped_cells[-1] != cell:
                                deduped_cells.append(cell)
                        if deduped_cells:
                            content_blocks.append(" | ".join(deduped_cells))

            raw_combined = "\n\n".join(content_blocks)
            normalized = cls.normalize_text(raw_combined)

            alpha_num_count = len(re.findall(r"[a-zA-Z0-9]", normalized))
            if alpha_num_count < 15:
                return TextExtractionResult(
                    success=False,
                    text=None,
                    error_message="No extractable text found in DOCX document.",
                    total_elements=len(content_blocks)
                )

            return TextExtractionResult(
                success=True,
                text=normalized,
                error_message=None,
                total_elements=len(content_blocks)
            )

        except Exception as e:
            return TextExtractionResult(
                success=False,
                text=None,
                error_message=f"Failed to parse DOCX document: {type(e).__name__}."
            )

    @classmethod
    def extract_document_text(cls, file_source: str | Path | bytes, file_type: str) -> TextExtractionResult:
        """Dispatches to the format-specific extractor based on file type."""
        ext = file_type.lower().strip().lstrip(".")
        if ext == "pdf":
            return cls.extract_from_pdf(file_source)
        elif ext in ("docx", "doc"):
            return cls.extract_from_docx(file_source)
        else:
            return TextExtractionResult(
                success=False,
                text=None,
                error_message=f"Unsupported file type: '.{ext}'. Only PDF and DOCX documents are accepted."
            )
