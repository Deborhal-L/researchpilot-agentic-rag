
import json
import re
from pathlib import Path
import os

import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.retrieve import Retriever
from src.agent import run_agent
from src.ingest import main as build_index

app = FastAPI(title="Agentic RAG Research Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

RAW = Path("data/raw")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
TESSERACT_DIR = Path(r"C:\Program Files\Tesseract-OCR")
TESSDATA_DIR = TESSERACT_DIR / "tessdata"

if TESSERACT_DIR.exists():
    os.environ["PATH"] = (
        str(TESSERACT_DIR)
        + os.pathsep
        + os.environ.get("PATH", "")
    )

if TESSDATA_DIR.exists():
    os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)

R = Retriever()
ACTIVE_DOC_FILE = Path("data/index/active_document.txt")

LAST_UPLOADED_DOC_ID = (
    ACTIVE_DOC_FILE.read_text(encoding="utf-8").strip()
    if ACTIVE_DOC_FILE.exists()
    else None
)


class Ask(BaseModel):
    question: str
    history: list[dict[str, str]] = []


@app.get("/")
def home():
    return {
        "message": "Agentic RAG Research Assistant API",
        "status": "running",
    }


@app.get("/documents")
def list_documents():
    RAW.mkdir(parents=True, exist_ok=True)

    documents = [
        {
            "name": path.name,
            "type": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(RAW.iterdir())
        if path.is_file() and path.suffix.lower() == ".txt"
    ]

    return {"documents": documents, "count": len(documents)}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global R, LAST_UPLOADED_DOC_ID

    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()

    if not filename or suffix not in {".pdf", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="The file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 10 MB limit.",
        )

    try:
        if suffix == ".pdf":
            if not content.startswith(b"%PDF-"):
                raise HTTPException(
                    status_code=400,
                    detail="The uploaded file is not a valid PDF.",
                )

            
            with pymupdf.open(stream=content, filetype="pdf") as pdf:
                extracted_pages = []

                for page in pdf:
                    page_text = page.get_text().strip()

                    # Use OCR when the page contains no selectable text.
                    if not page_text:
                        text_page = page.get_textpage_ocr(
                            language="eng",
                            dpi=200,
                            full=True,
                        )
                        page_text = page.get_text(
                            textpage=text_page
                        ).strip()

                    extracted_pages.append(page_text)

            text = "\n\n".join(extracted_pages)


        else:
            text = content.decode("utf-8-sig")

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not extract document text: {exc}",
        ) from exc
    finally:
        await file.close()

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "No text was extracted. Scanned PDFs may require OCR "
                "before they can be indexed."
            ),
        )

    RAW.mkdir(parents=True, exist_ok=True)

    # Create a safe, unique document filename.
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem).strip("_")
    stem = (stem or "document")[:70]
    destination = RAW / f"{stem}.txt"
    counter = 2

    while destination.exists():
        destination = RAW / f"{stem}_{counter}.txt"
        counter += 1

    try:
        destination.write_text(text, encoding="utf-8")

        # Rebuild indexes from all TXT files in data/raw.
        build_index()

        # Load the newly built indexes into the active retriever.
        R = Retriever()
        # Save the most recently uploaded document ID.
        ACTIVE_DOC_FILE.parent.mkdir(parents=True, exist_ok=True)
        ACTIVE_DOC_FILE.write_text(destination.stem, encoding="utf-8")
        LAST_UPLOADED_DOC_ID = destination.stem

    except Exception as exc:
        # Do not leave an unindexed document in the collection.
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=(
                "Indexing failed. The uploaded text file was removed. "
                "Check the backend terminal for details."
            ),
        ) from exc

    return {
        "status": "success",
        "filename": filename,
        "document_id": destination.stem,
        "characters_extracted": len(text),
        "message": "Document uploaded and indexed successfully.",
    }


@app.post("/ask")
def ask(body: Ask):
    question = body.question.strip()
    normalized_question = re.sub(
        r"[^a-z0-9]+", " ", question.casefold()
    ).strip()

    # Explicit requests involving multiple documents use global search.
    global_search_terms = (
        "compare",
        "difference",
        "differences",
        "both documents",
        "both reports",
        "across documents",
        "between the reports",
        "two vulnerabilities",
        "both vulnerabilities",
        "example web server",
        "example database server",
        "sample a",
        "sample b",
    )

    use_global_search = any(
        term in normalized_question
        for term in global_search_terms
    )

    selected_document_id = None

    if not use_global_search:
        # Default to the latest uploaded document.
        selected_document_id = LAST_UPLOADED_DOC_ID

        # If the question names another indexed document, use that one.
        indexed_doc_ids = sorted(
            {
                chunk["doc_id"]
                for chunk in R.chunks
                if chunk.get("doc_id")
            },
            key=len,
            reverse=True,
        )

        for doc_id in indexed_doc_ids:
            # Ignore numeric suffixes added to duplicate uploads.
            base_id = re.sub(r"_\d+$", "", doc_id)
            title = re.sub(
                r"[^a-z0-9]+", " ", base_id.casefold()
            ).strip()

            if title and title in normalized_question:
                selected_document_id = doc_id
                break

    history = body.history[-6:]

    context = ""

    if history:
        context = "\n".join(
        f"{item.get('role', 'user')}: {item.get('content', '')}"
        for item in history
        if item.get("role") in {"user", "assistant"}
        and item.get("content", "").strip()
    )

    contextual_question = question

    if context:
      contextual_question = (
        "Previous conversation:\n"
        f"{context}\n\n"
        "Current question:\n"
        f"{question}\n\n"
        "Interpret the current question using the previous "
        "conversation when needed. Answer only using retrieved "
        "document evidence."
    )

    t = run_agent(
        contextual_question,
        R,
       document_id=selected_document_id,
)
    return {
        "status": t["status"],
        "answer": t["answer"],
        "citations": t.get("citations", []),
        "evidence": t.get("evidence", []),
        "trace_id": t["id"],
        "latency": t["latency"],
        "cost": t["cost"],
    }