
import hashlib
import os
import re
import tempfile
import uuid
from pathlib import Path

import fitz
import pandas as pd
import pytesseract

from PIL import Image
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from openpyxl import load_workbook
from pptx import Presentation

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from backend.services.embedding_service import (
    get_embedding_model,
    embed_documents,
    EMBEDDING_BATCH_SIZE,
)

from backend.core.config import (
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from backend.services.auth_service import normalize_user_id
from backend.services.latency import LatencyTracker


# --------------------------------------------------
# Configuration
# --------------------------------------------------

COLLECTION_NAME = "earnings_calls"

MAX_FILE_SIZE = 25 * 1024 * 1024

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".csv",
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".bmp",
}


# --------------------------------------------------
# Vector Store
# --------------------------------------------------

def get_vector_store(embedding_model=None):
    """Initialize and return the ChromaDB vector store."""

    if embedding_model is None:
        embedding_model = get_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_model,
    )


# --------------------------------------------------
# Metadata Validation
# --------------------------------------------------

def validate_metadata(
    company: str,
    department: str = "",
    document_type: str = "",
    fiscal_year=None,
):
    """
    Validate manually provided document metadata.

    Company is required for backward compatibility
    with the existing ATA upload workflow.
    """

    company = (company or "").strip()
    department = (department or "").strip()
    document_type = (document_type or "").strip()

    if not company:
        raise ValueError("Company name is required.")

    if len(company) > 150:
        raise ValueError("Company name is too long.")

    if len(department) > 150:
        raise ValueError("Department name is too long.")

    if len(document_type) > 100:
        raise ValueError("Document type is too long.")

    if fiscal_year in ("", None):
        fiscal_year = None
    else:
        try:
            fiscal_year = int(fiscal_year)
        except (TypeError, ValueError):
            raise ValueError("Fiscal year must be a valid year.")

        if fiscal_year < 1900 or fiscal_year > 2100:
            raise ValueError("Fiscal year must be between 1900 and 2100.")

    return {
        "company": company,
        "department": department,
        "document_type": document_type,
        "fiscal_year": fiscal_year,
    }


# --------------------------------------------------
# File Validation
# --------------------------------------------------

def validate_filename(filename: str) -> str:

    if not filename:
        raise ValueError("Filename is required.")

    filename = Path(filename).name

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    return filename


# --------------------------------------------------
# Save Uploaded File
# --------------------------------------------------

def save_uploaded_file(upload, filename: str) -> str:

    filename = validate_filename(filename)

    extension = Path(filename).suffix.lower()

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension,
    )

    temp_path = temp_file.name
    total_size = 0

    try:
        with temp_file:

            while True:

                chunk = upload.file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise ValueError(
                        "File size exceeds the 25 MB limit."
                    )

                temp_file.write(chunk)

        if total_size == 0:
            raise ValueError("Uploaded file is empty.")

        return temp_path

    except Exception:

        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise


# --------------------------------------------------
# PDF Parser - PyMuPDF
# --------------------------------------------------

def extract_pdf_text(file_path: str) -> str:

    extracted_pages = []

    with fitz.open(file_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            page_text = page.get_text("text").strip()

            if page_text:
                extracted_pages.append(
                    f"\n[Page {page_number}]\n{page_text}"
                )

    return "\n".join(extracted_pages)


# --------------------------------------------------
# DOCX Parser
# --------------------------------------------------

def extract_docx_text(file_path: str) -> str:

    document = DocxDocument(file_path)

    content = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            content.append(text)

    for table_index, table in enumerate(
        document.tables,
        start=1,
    ):

        content.append(f"\n[Table {table_index}]")

        for row in table.rows:

            row_data = [
                cell.text.strip()
                for cell in row.cells
            ]

            content.append(" | ".join(row_data))

    return "\n".join(content)


# --------------------------------------------------
# XLSX Parser
# --------------------------------------------------

def extract_xlsx_text(file_path: str) -> str:

    workbook = load_workbook(
        file_path,
        read_only=True,
        data_only=True,
    )

    content = []

    try:

        for worksheet in workbook.worksheets:

            content.append(
                f"\n[Sheet: {worksheet.title}]"
            )

            for row in worksheet.iter_rows(values_only=True):

                values = [
                    str(value)
                    for value in row
                    if value is not None
                ]

                if values:
                    content.append(" | ".join(values))

    finally:
        workbook.close()

    return "\n".join(content)


# --------------------------------------------------
# PPTX Parser
# --------------------------------------------------

def extract_pptx_text(file_path: str) -> str:

    presentation = Presentation(file_path)

    content = []

    for slide_number, slide in enumerate(
        presentation.slides,
        start=1,
    ):

        content.append(f"\n[Slide {slide_number}]")

        for shape in slide.shapes:

            if shape.has_text_frame:

                text = shape.text.strip()

                if text:
                    content.append(text)

            if shape.has_table:

                for row in shape.table.rows:

                    row_data = [
                        cell.text.strip()
                        for cell in row.cells
                    ]

                    content.append(" | ".join(row_data))

    return "\n".join(content)


# --------------------------------------------------
# CSV Parser
# --------------------------------------------------

def extract_csv_text(file_path: str) -> str:

    dataframe = pd.read_csv(
        file_path,
        dtype=str,
        keep_default_na=False,
    )

    return dataframe.to_string(index=False)


# --------------------------------------------------
# TXT Parser
# --------------------------------------------------

def extract_txt_text(file_path: str) -> str:

    return Path(file_path).read_text(
        encoding="utf-8",
        errors="replace",
    )


# --------------------------------------------------
# Markdown Parser
# --------------------------------------------------

def extract_markdown_text(file_path: str) -> str:

    return Path(file_path).read_text(
        encoding="utf-8",
        errors="replace",
    )


# --------------------------------------------------
# HTML Parser
# --------------------------------------------------

def extract_html_text(file_path: str) -> str:

    html_content = Path(file_path).read_text(
        encoding="utf-8",
        errors="replace",
    )

    soup = BeautifulSoup(
        html_content,
        "html.parser",
    )

    for element in soup([
        "script",
        "style",
        "noscript",
    ]):
        element.decompose()

    return soup.get_text(
        separator="\n",
        strip=True,
    )


# --------------------------------------------------
# Image Parser - OCR
# --------------------------------------------------

def extract_image_text(file_path: str) -> str:

    image = Image.open(file_path)

    try:

        extracted_text = pytesseract.image_to_string(
            image
        )

    finally:
        image.close()

    return extracted_text.strip()


# --------------------------------------------------
# Document Parser Router
# --------------------------------------------------

def extract_document_text(file_path: str) -> str:

    extension = Path(file_path).suffix.lower()

    parsers = {
        ".pdf": extract_pdf_text,
        ".docx": extract_docx_text,
        ".xlsx": extract_xlsx_text,
        ".pptx": extract_pptx_text,
        ".csv": extract_csv_text,
        ".txt": extract_txt_text,
        ".md": extract_markdown_text,
        ".html": extract_html_text,
        ".htm": extract_html_text,
        ".png": extract_image_text,
        ".jpg": extract_image_text,
        ".jpeg": extract_image_text,
        ".tiff": extract_image_text,
        ".bmp": extract_image_text,
    }

    parser = parsers.get(extension)

    if parser is None:
        raise ValueError(
            f"No parser available for {extension}"
        )

    extracted_text = parser(file_path)

    if not extracted_text or not extracted_text.strip():
        raise ValueError(
            "No readable text could be extracted from the document."
        )

    return extracted_text.strip()


# --------------------------------------------------
# Extract Page Number from PDF Text
# --------------------------------------------------

def extract_page_number(text: str, start_index: int):
    """
    Find the last PDF page marker before a chunk's
    starting character position.

    Returns None when a page marker is unavailable.
    """

    if start_index is None:
        return None

    preceding_text = text[:start_index]

    matches = re.findall(
        r"\[Page\s+(\d+)\]",
        preceding_text,
        flags=re.IGNORECASE,
    )

    if not matches:
        return None

    return int(matches[-1])


# --------------------------------------------------
# Document Chunking
# --------------------------------------------------

def create_document_chunks(
    text: str,
    filename: str,
    company: str,
    owner_id: str,
    document_id: str,
    file_type: str,
    department: str = "",
    document_type: str = "",
    fiscal_year=None,
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    base_metadata = {
        "source": filename,
        "filename": filename,
        "company": company,
        "department": department,
        "document_type": document_type,
        "owner_id": owner_id,
        "document_id": document_id,
        "file_type": file_type,
    }

    # Chroma metadata must not contain None values.
    if fiscal_year is not None:
        base_metadata["fiscal_year"] = fiscal_year

    document = Document(
        page_content=text,
        metadata=base_metadata,
    )

    chunks = splitter.split_documents([document])

    # Attach page number to each chunk where available.
    for chunk in chunks:

        start_index = chunk.metadata.get("start_index")

        page_number = extract_page_number(
            text,
            start_index,
        )

        if page_number is not None:
            chunk.metadata["page_number"] = page_number

    return chunks


# --------------------------------------------------
# Document Ingestion
# --------------------------------------------------

def ingest_uploaded_document(
    upload,
    filename: str,
    company: str,
    owner_id: str,
    department: str = "",
    document_type: str = "",
    fiscal_year=None,
):

    tracker = LatencyTracker("DOCUMENT INGESTION")

    temp_path = None

    try:

        # 1. Validate input and metadata.

        with tracker.track("input_validation"):

            owner_id = normalize_user_id(owner_id)

            filename = validate_filename(filename)

            metadata = validate_metadata(
                company=company,
                department=department,
                document_type=document_type,
                fiscal_year=fiscal_year,
            )

            company = metadata["company"]
            department = metadata["department"]
            document_type = metadata["document_type"]
            fiscal_year = metadata["fiscal_year"]

            document_id = str(uuid.uuid4())

            file_type = (
                Path(filename)
                .suffix
                .lower()
                .lstrip(".")
            )

        # 2. Save uploaded file temporarily.

        with tracker.track("file_saving"):

            temp_path = save_uploaded_file(
                upload,
                filename,
            )

        # 3. Extract document text.

        with tracker.track("text_extraction"):

            extracted_text = extract_document_text(
                temp_path
            )

        # 4. Split document into chunks and attach metadata.

        with tracker.track("document_chunking"):

            chunks = create_document_chunks(
                text=extracted_text,
                filename=filename,
                company=company,
                owner_id=owner_id,
                document_id=document_id,
                file_type=file_type,
                department=department,
                document_type=document_type,
                fiscal_year=fiscal_year,
            )

            if not chunks:
                raise ValueError(
                    "No document chunks were created."
                )

        # 5. Generate deterministic chunk IDs.

        with tracker.track("chunk_id_generation"):

            chunk_ids = []

            for index, chunk in enumerate(chunks):

                content_hash = hashlib.sha256(
                    chunk.page_content.encode("utf-8")
                ).hexdigest()

                chunk_id = str(
                    uuid.uuid5(
                        uuid.UUID(document_id),
                        f"{index}-{content_hash}",
                    )
                )

                chunk_ids.append(chunk_id)

        # 6. Initialize embedding model.

        with tracker.track("embedding_model_initialization"):

            embedding_model = get_embedding_model()

        # 7. Generate embeddings.

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        embeddings = embed_documents(
            texts=texts,
            tracker=tracker,
            batch_size=EMBEDDING_BATCH_SIZE,
        )

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Embedding count does not match chunk count."
            )

        # 8. Initialize ChromaDB.

        with tracker.track("chroma_collection_initialization"):

            vector_store = get_vector_store(
                embedding_model=embedding_model
            )

        # 9. Prepare metadata.

        with tracker.track("embedding_payload_preparation"):

            metadatas = [
                chunk.metadata
                for chunk in chunks
            ]

        # 10. Store vectors and metadata.

        with tracker.track("chroma_vector_storage"):

            vector_store._collection.add(
                ids=chunk_ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        # 11. Return ingestion information.

        return {
            "success": True,
            "message": "Document ingested successfully.",
            "document_id": document_id,
            "filename": filename,
            "company": company,
            "department": department,
            "document_type": document_type,
            "fiscal_year": fiscal_year,
            "file_type": file_type,
            "chunks_count": len(chunks),
            "total_characters": len(extracted_text),
            "embedding_batch_size": EMBEDDING_BATCH_SIZE,
            "total_batches": (
                len(texts) + EMBEDDING_BATCH_SIZE - 1
            ) // EMBEDDING_BATCH_SIZE,
        }

    finally:

        # 12. Clean up temporary file.

        try:

            with tracker.track("temporary_file_cleanup"):

                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        finally:

            # 13. Print latency report.

            tracker.report()
