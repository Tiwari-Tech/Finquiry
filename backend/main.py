
from typing import Optional

from fastapi import (
    FastAPI,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel, EmailStr, Field

from backend.services.auth_service import (
    register_user,
    authenticate_user,
    create_access_token,
    get_current_user,
)

from backend.services.ingestion_service import (
    ingest_uploaded_document,
)

from backend.services.retrieval_service import (
    list_user_documents,
    delete_user_document,
)

from backend.services.chat_service import (
    answer_question,
)

from backend.services.history import (
    list_conversations,
    get_conversation,
    get_conversation_messages,
    create_conversation,
    delete_conversation,
)

from backend.services.embedding_service import (
    get_embedding_config,
)


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Finquiry - Multi-User Document Search",
    description=(
        "Multi-user document search and conversational "
        "question-answering application."
    ),
    version="2.1.0",
)


# --------------------------------------------------
# REQUEST SCHEMAS
# --------------------------------------------------

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None

    # Optional legacy/manual filters.
    # Normally metadata routing is automatic.
    company: Optional[str] = None
    document_id: Optional[str] = None


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "Finquiry",
        "version": "2.1.0",
    }


# --------------------------------------------------
# AUTHENTICATION
# --------------------------------------------------

@app.post("/auth/register")
def register(request: RegisterRequest):

    try:
        user = register_user(
            email=request.email,
            password=request.password,
        )

        return {
            "message": "User registered successfully.",
            "user": user,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@app.post("/auth/login")
def login(request: LoginRequest):

    try:
        user = authenticate_user(
            email=request.email,
            password=request.password,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password.",
        )

    access_token = create_access_token(
        user_id=user["user_id"],
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user,
    }


# --------------------------------------------------
# EMBEDDING CONFIGURATION
# --------------------------------------------------

@app.get("/system/embedding-config")
def embedding_configuration(
    current_user: str = Depends(get_current_user),
):

    try:
        return {
            "embedding": get_embedding_config(),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve embedding configuration: {str(exc)}",
        )


# --------------------------------------------------
# DOCUMENT INGESTION
# --------------------------------------------------

@app.post("/documents/ingest")
def ingest_document(
    file: UploadFile = File(...),
    company: str = Form(...),
    department: str = Form(""),
    document_type: str = Form(""),
    fiscal_year: Optional[int] = Form(None),
    current_user: str = Depends(get_current_user),
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if not company.strip():
        raise HTTPException(
            status_code=400,
            detail="Company cannot be empty.",
        )

    try:
        result = ingest_uploaded_document(
            upload=file,
            filename=file.filename,
            company=company.strip(),
            owner_id=current_user,
            department=department.strip(),
            document_type=document_type.strip(),
            fiscal_year=fiscal_year,
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {str(exc)}",
        )

    finally:
        file.file.close()


# --------------------------------------------------
# LIST USER DOCUMENTS
# --------------------------------------------------

@app.get("/documents")
def get_documents(
    current_user: str = Depends(get_current_user),
):

    try:
        documents = list_user_documents(
            email=current_user,
        )

        return {
            "documents": documents,
            "total": len(documents),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve documents: {str(exc)}",
        )


# --------------------------------------------------
# DELETE USER DOCUMENT
# --------------------------------------------------

@app.delete("/documents/{document_id}")
def remove_document(
    document_id: str,
    current_user: str = Depends(get_current_user),
):

    try:
        result = delete_user_document(
            document_id=document_id,
            email=current_user,
        )

        if not result:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        return {
            "message": "Document deleted successfully.",
            "result": result,
        }

    except HTTPException:
        raise

    except PermissionError as exc:
        raise HTTPException(
            status_code=403,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document deletion failed: {str(exc)}",
        )


# --------------------------------------------------
# CHAT / QUESTION ANSWERING
# --------------------------------------------------

@app.post("/chat")
def chat(
    request: ChatRequest,
    current_user: str = Depends(get_current_user),
):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:

        conversation_id = request.conversation_id

        # Create a conversation using the actual
        # history.py function signature.
        if not conversation_id:

            conversation_id = create_conversation(
                user_id=current_user,
                question=question,
                company=request.company or "",
            )

        result = answer_question(
            user_id=current_user,
            question=question,
            conversation_id=conversation_id,
            company=request.company,
            document_id=request.document_id,
        )

        return result

    except PermissionError as exc:
        raise HTTPException(
            status_code=403,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Question answering failed: {str(exc)}",
        )


# --------------------------------------------------
# LIST USER CONVERSATIONS
# --------------------------------------------------

@app.get("/conversations")
def get_user_conversations(
    current_user: str = Depends(get_current_user),
):

    try:
        conversations = list_conversations(
            user_id=current_user,
        )

        return {
            "conversations": conversations,
            "total": len(conversations),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve conversations: {str(exc)}",
        )


# --------------------------------------------------
# GET CONVERSATION AND MESSAGES
# --------------------------------------------------

@app.get("/conversations/{conversation_id}")
def get_user_conversation(
    conversation_id: str,
    current_user: str = Depends(get_current_user),
):

    try:
        conversation = get_conversation(
            conversation_id=conversation_id,
            user_id=current_user,
        )

        if conversation is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        messages = get_conversation_messages(
            conversation_id=conversation_id,
            user_id=current_user,
        )

        return {
            "conversation": dict(conversation),
            "messages": messages,
        }

    except HTTPException:
        raise

    except PermissionError as exc:
        raise HTTPException(
            status_code=403,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve conversation: {str(exc)}",
        )


# --------------------------------------------------
# DELETE CONVERSATION
# --------------------------------------------------

@app.delete("/conversations/{conversation_id}")
def remove_conversation(
    conversation_id: str,
    current_user: str = Depends(get_current_user),
):

    try:
        result = delete_conversation(
            conversation_id=conversation_id,
            user_id=current_user,
        )

        if not result:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        return {
            "message": "Conversation deleted successfully.",
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not delete conversation: {str(exc)}",
        )
