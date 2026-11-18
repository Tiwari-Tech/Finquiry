import requests
import streamlit as st

# ==================================================
# CONFIGURATION
# ==================================================

API_URL = "http://127.0.0.1:8000"
NEW_CONVERSATION = "__new_conversation__"
ALL_DOCUMENTS = "__all_documents__"

st.set_page_config(
    page_title="Finquiry | Document Q&A",
    page_icon="📄",
    layout="wide",
)

# ==================================================
# SESSION STATE
# ==================================================

DEFAULTS = {
    "token": None,
    "messages": [],
    "conversation_id": None,
    "loaded_conversation_id": None,
    "conversations": [],
    "documents": [],
    "selected_document_id": ALL_DOCUMENTS,
    "active_scope": None,
    "document_selector": ALL_DOCUMENTS,
    "conversation_selector": NEW_CONVERSATION,
    "pending_new_conversation_selection": False,
    "confirm_delete_document": False,
    "pending_delete_document_id": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        # Avoid sharing mutable defaults between sessions/reruns.
        st.session_state[key] = value.copy() if isinstance(value, (list, dict)) else value


# ==================================================
# API HELPERS
# ==================================================

def get_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}


def show_api_error(response, action):
    try:
        payload = response.json()
        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
    except ValueError:
        detail = response.text or "Unknown API error."

    st.error(f"{action} failed ({response.status_code}): {detail}")


def extract_list(payload, *keys):
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value

    return []


def api_get(path, action, timeout=30):
    try:
        response = requests.get(
            f"{API_URL}{path}",
            headers=get_headers(),
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"{action}: {exc}")
    except ValueError as exc:
        st.error(f"{action}: backend returned invalid JSON ({exc}).")
    return None


def load_documents(show_errors=True):
    try:
        response = requests.get(
            f"{API_URL}/documents",
            headers=get_headers(),
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        st.session_state.documents = extract_list(
            payload, "documents", "items", "data"
        )
        return True
    except requests.RequestException as exc:
        if show_errors:
            st.error(f"Could not load documents: {exc}")
    except ValueError as exc:
        if show_errors:
            st.error(f"Invalid document API response: {exc}")
    return False


def load_conversations(show_errors=True):
    try:
        response = requests.get(
            f"{API_URL}/conversations",
            headers=get_headers(),
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        st.session_state.conversations = extract_list(
            payload, "conversations", "items", "data"
        )
        return True
    except requests.RequestException as exc:
        if show_errors:
            st.error(f"Could not load conversations: {exc}")
    except ValueError as exc:
        if show_errors:
            st.error(f"Invalid conversation API response: {exc}")
    return False


def reset_chat():
    st.session_state.messages = []
    st.session_state.conversation_id = None
    st.session_state.loaded_conversation_id = None
    # Defer widget-key changes until before the selectbox is instantiated.
    st.session_state.pending_new_conversation_selection = True


def source_label(source):
    """Format source metadata returned by chat_service."""
    filename = (
        source.get("filename")
        or source.get("source")
        or source.get("file_name")
        or "Unknown file"
    )
    company = source.get("company") or "Unknown company"
    page = source.get("page_number")
    page_text = f" · Page {page}" if page is not None else ""
    return f"**{company}** — {filename}{page_text}"


# ==================================================
# DOCUMENT ACTIONS
# ==================================================

def delete_document(document_id):
    try:
        response = requests.delete(
            f"{API_URL}/documents/{document_id}",
            headers=get_headers(),
            timeout=60,
        )
        if response.status_code not in (200, 204):
            show_api_error(response, "Document deletion")
            return

        st.session_state.documents = [
            doc for doc in st.session_state.documents
            if doc.get("document_id") != document_id
        ]
        st.session_state.document_selector = ALL_DOCUMENTS
        st.session_state.selected_document_id = ALL_DOCUMENTS
        st.session_state.active_scope = None
        reset_chat()
        st.session_state.confirm_delete_document = False
        st.session_state.pending_delete_document_id = None
        load_documents(show_errors=False)
        load_conversations(show_errors=False)
        st.success("Document deleted successfully.")
        st.rerun()

    except requests.RequestException as exc:
        st.error(f"Could not delete document: {exc}")


def open_conversation(conversation_id):
    try:
        response = requests.get(
            f"{API_URL}/conversations/{conversation_id}",
            headers=get_headers(),
            timeout=30,
        )
        if response.status_code != 200:
            show_api_error(response, "Opening conversation")
            return False

        payload = response.json()
        conversation = payload.get("conversation") or {}
        messages = payload.get("messages") or []

        if not isinstance(messages, list):
            messages = []

        st.session_state.conversation_id = conversation_id
        st.session_state.loaded_conversation_id = conversation_id
        st.session_state.messages = messages
        return True

    except requests.RequestException as exc:
        st.error(f"Could not open conversation: {exc}")
    except (ValueError, TypeError, AttributeError) as exc:
        st.error(f"Invalid conversation response: {exc}")
    return False


def delete_conversation(conversation_id):
    try:
        response = requests.delete(
            f"{API_URL}/conversations/{conversation_id}",
            headers=get_headers(),
            timeout=30,
        )
        if response.status_code not in (200, 204):
            show_api_error(response, "Conversation deletion")
            return

        reset_chat()
        load_conversations(show_errors=False)
        st.success("Conversation deleted successfully.")
        st.rerun()

    except requests.RequestException as exc:
        st.error(f"Could not delete conversation: {exc}")


# ==================================================
# HEADER
# ==================================================

st.title("📄 Finquiry — Document Q&A")
st.caption(
    "Upload documents, manage conversations, and ask questions "
    "across one document or your entire document library."
)

# ==================================================
# AUTHENTICATION (LOGIN / REGISTER)
# ==================================================

if not st.session_state.token:
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if not email.strip() or not password:
                st.warning("Please enter your email and password.")
            else:
                try:
                    response = requests.post(
                        f"{API_URL}/auth/login",
                        json={"email": email.strip(), "password": password},
                        timeout=30,
                    )

                    if response.status_code == 200:
                        payload = response.json()
                        token = payload.get("access_token") or payload.get("token")

                        if not token:
                            st.error("Login response did not contain an access token.")
                            st.stop()

                        st.session_state.token = token
                        reset_chat()
                        st.session_state.documents = []
                        st.session_state.conversations = []
                        st.session_state.document_selector = ALL_DOCUMENTS
                        st.session_state.selected_document_id = ALL_DOCUMENTS
                        load_documents()
                        load_conversations()
                        st.rerun()
                    else:
                        show_api_error(response, "Login")

                except requests.RequestException as exc:
                    st.error(f"Backend connection failed: {exc}")
                except ValueError as exc:
                    st.error(f"Invalid login response: {exc}")

    with register_tab:
        with st.form("register_form"):
            reg_email = st.text_input("Email", key="register_email")
            reg_password = st.text_input(
                "Password",
                type="password",
                help="Must be at least 8 characters.",
                key="register_password",
            )
            reg_confirm = st.text_input(
                "Confirm Password",
                type="password",
                key="register_confirm_password",
            )
            reg_submitted = st.form_submit_button("Register", use_container_width=True)

        if reg_submitted:
            if not reg_email.strip() or not reg_password:
                st.warning("Please enter an email and password.")
            elif len(reg_password) < 8:
                st.warning("Password must be at least 8 characters.")
            elif reg_password != reg_confirm:
                st.warning("Passwords do not match.")
            else:
                try:
                    response = requests.post(
                        f"{API_URL}/auth/register",
                        json={"email": reg_email.strip(), "password": reg_password},
                        timeout=30,
                    )

                    if response.status_code in (200, 201):
                        st.success("Account registered successfully. You can now log in.")
                    else:
                        show_api_error(response, "Registration")

                except requests.RequestException as exc:
                    st.error(f"Backend connection failed: {exc}")

    st.stop()


# ==================================================
# REFRESH DATA
# ==================================================

load_conversations(show_errors=False)

# ==================================================
# SIDEBAR
# ==================================================

selected_document = None
selected_document_id = ALL_DOCUMENTS
selected_company = None

with st.sidebar:
    st.header("My Documents")

    # -------------------- Upload --------------------

    with st.expander("➕ Upload New Document", expanded=False):
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=[
                "pdf", "docx", "xlsx", "pptx", "csv", "txt",
                "md", "html", "htm", "png", "jpg", "jpeg",
                "tiff", "bmp",
            ],
        )

        company_name = st.text_input(
            "Company / Document Category",
            placeholder="e.g. TCS, Axis Bank",
        )
        department = st.text_input(
            "Department (optional)",
            placeholder="e.g. Finance, HR, Engineering",
        )
        document_type = st.text_input(
            "Document Type (optional)",
            placeholder="e.g. Annual Report, Earnings Call",
        )
        fiscal_year = st.number_input(
            "Fiscal Year (optional)",
            min_value=2000,
            max_value=2100,
            value=None,
            step=1,
            placeholder="e.g. 2026",
        )

        if st.button("Upload Document", use_container_width=True):
            if uploaded_file is None:
                st.warning("Please select a file.")
            elif not company_name.strip():
                st.warning("Please enter a company or document category.")
            else:
                upload_data = {"company": company_name.strip()}
                if department.strip():
                    upload_data["department"] = department.strip()
                if document_type.strip():
                    upload_data["document_type"] = document_type.strip()
                if fiscal_year is not None:
                    upload_data["fiscal_year"] = str(int(fiscal_year))

                try:
                    with st.spinner("Uploading and processing document..."):
                        response = requests.post(
                            f"{API_URL}/documents/ingest",
                            headers=get_headers(),
                            files={
                                "file": (
                                    uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    uploaded_file.type or "application/octet-stream",
                                )
                            },
                            data=upload_data,
                            timeout=300,
                        )

                    if response.status_code in (200, 201):
                        payload = response.json()
                        st.success(
                            f"Successfully uploaded: "
                            f"{payload.get('filename', uploaded_file.name)}"
                        )
                        load_documents(show_errors=False)
                        new_id = payload.get("document_id")
                        st.session_state.document_selector = (
                            new_id if new_id else ALL_DOCUMENTS
                        )
                        st.session_state.selected_document_id = (
                            new_id if new_id else ALL_DOCUMENTS
                        )
                        st.session_state.active_scope = None
                        reset_chat()
                        load_conversations(show_errors=False)
                        st.rerun()
                    else:
                        show_api_error(response, "Upload")

                except requests.RequestException as exc:
                    st.error(f"Backend connection failed: {exc}")
                except ValueError as exc:
                    st.error(f"Invalid upload response: {exc}")

    st.divider()

    col_refresh, col_all = st.columns(2)
    with col_refresh:
        if st.button("↻ Refresh", use_container_width=True):
            load_documents()
            load_conversations()
            st.rerun()
    with col_all:
        if st.button("All documents", use_container_width=True):
            st.session_state.document_selector = ALL_DOCUMENTS
            st.session_state.selected_document_id = ALL_DOCUMENTS
            st.session_state.active_scope = ALL_DOCUMENTS
            reset_chat()
            st.rerun()

    documents = st.session_state.documents
    document_options = {
        doc.get("document_id"): doc
        for doc in documents
        if isinstance(doc, dict) and doc.get("document_id")
    }
    document_ids = list(document_options.keys())

    # Normalize widget value before instantiating the selectbox.
    allowed_values = [ALL_DOCUMENTS, *document_ids]
    current_selection = st.session_state.get("document_selector")
    if current_selection not in allowed_values:
        current_selection = ALL_DOCUMENTS
        st.session_state.document_selector = current_selection

    if documents:
        def document_label(value):
            if value == ALL_DOCUMENTS:
                return "🌐 All my documents"
            doc = document_options[value]
            filename = doc.get("filename") or "Unknown file"
            company = doc.get("company") or "Unknown company"
            return f"{filename} ({company})"

        selected_document_id = st.selectbox(
            "Search scope",
            options=allowed_values,
            format_func=document_label,
            key="document_selector",
        )
        st.session_state.selected_document_id = selected_document_id

        if selected_document_id != ALL_DOCUMENTS:
            selected_document = document_options[selected_document_id]
            selected_company = selected_document.get("company")

            st.subheader("Manage Selected Document")
            st.caption(
                f"Selected: {selected_document.get('filename', 'Unknown file')}"
            )

            if (
                st.session_state.pending_delete_document_id
                != selected_document_id
            ):
                st.session_state.confirm_delete_document = False
                st.session_state.pending_delete_document_id = selected_document_id

            if st.button(
                "🗑️ Delete Document",
                use_container_width=True,
                key="delete_document_button",
            ):
                st.session_state.confirm_delete_document = True
                st.session_state.pending_delete_document_id = selected_document_id

            if st.session_state.confirm_delete_document:
                st.warning(
                    "This permanently deletes the selected document "
                    "and its stored vector chunks."
                )
                confirm_col, cancel_col = st.columns(2)
                with confirm_col:
                    if st.button(
                        "Confirm Delete",
                        type="primary",
                        use_container_width=True,
                        key="confirm_delete_document_button",
                    ):
                        delete_document(selected_document_id)
                with cancel_col:
                    if st.button(
                        "Cancel",
                        use_container_width=True,
                        key="cancel_delete_document_button",
                    ):
                        st.session_state.confirm_delete_document = False
                        st.session_state.pending_delete_document_id = None
                        st.rerun()
        else:
            selected_company = None
            st.caption("Questions will search across your document library.")
    else:
        st.info("No documents uploaded yet.")

    st.divider()

    # -------------------- Conversations --------------------

    st.header("My Conversations")

    if selected_document_id == ALL_DOCUMENTS:
        visible_conversations = st.session_state.conversations
    else:
        visible_conversations = [
            item for item in st.session_state.conversations
            if item.get("company") == selected_company
        ]

    conversation_options = {
        item.get("conversation_id"): item
        for item in visible_conversations
        if item.get("conversation_id")
    }

    current_conversation_id = st.session_state.get("conversation_selector")
    valid_conversation_ids = [NEW_CONVERSATION, *conversation_options.keys()]

    # Apply deferred resets before creating the conversation selectbox.
    if st.session_state.get("pending_new_conversation_selection", False):
        current_conversation_id = NEW_CONVERSATION
        st.session_state.pending_new_conversation_selection = False

    if current_conversation_id not in valid_conversation_ids:
        current_conversation_id = NEW_CONVERSATION
    if st.session_state.conversation_id in conversation_options:
        current_conversation_id = st.session_state.conversation_id

    # Set widget state before creating the selectbox.
    st.session_state.conversation_selector = current_conversation_id

    if st.button("＋ New Conversation", use_container_width=True):
        reset_chat()
        st.rerun()

    def conversation_label(value):
        if value == NEW_CONVERSATION:
            return "New chat"
        item = conversation_options[value]
        title = item.get("title") or "Untitled conversation"
        updated_at = item.get("updated_at") or ""
        company = item.get("company") or "All documents"
        suffix = f" · {updated_at[:16]}" if updated_at else ""
        return f"{title} [{company}]{suffix}"

    selected_conversation_id = st.selectbox(
        "Previous conversations",
        options=valid_conversation_ids,
        format_func=conversation_label,
        key="conversation_selector",
    )

    if selected_conversation_id == NEW_CONVERSATION:
        if st.session_state.conversation_id is not None:
            reset_chat()
    elif selected_conversation_id != st.session_state.loaded_conversation_id:
        if open_conversation(selected_conversation_id):
            st.rerun()

    if selected_conversation_id != NEW_CONVERSATION:
        if st.button("🗑️ Delete Conversation", use_container_width=True):
            delete_conversation(selected_conversation_id)

    st.divider()

    # -------------------- Logout --------------------

    if st.button("Logout", use_container_width=True):
        st.session_state.token = None
        st.session_state.documents = []
        st.session_state.conversations = []
        st.session_state.selected_document_id = ALL_DOCUMENTS
        st.session_state.active_scope = None
        st.session_state.document_selector = ALL_DOCUMENTS
        st.session_state.confirm_delete_document = False
        st.session_state.pending_delete_document_id = None
        reset_chat()
        st.rerun()


# ==================================================
# RESET CHAT WHEN SEARCH SCOPE CHANGES
# ==================================================

current_scope = (
    selected_document_id
    if selected_document_id == ALL_DOCUMENTS
    else selected_document_id
)

if st.session_state.active_scope != current_scope:
    st.session_state.active_scope = current_scope
    reset_chat()

# ==================================================
# MAIN CHAT INTERFACE
# ==================================================

if documents := st.session_state.documents:
    if selected_document_id == ALL_DOCUMENTS:
        st.subheader("🌐 Chat with All Documents")
        st.caption(
            "Finquiry will use metadata-aware retrieval to find relevant "
            "information across your uploaded documents."
        )
    elif selected_document:
        filename = selected_document.get("filename") or "Unknown file"
        company = selected_document.get("company") or "Not specified"
        st.subheader(f"Document: {filename}")
        st.caption(f"Company: {company}")

    st.divider()

    # -------------------- Existing messages --------------------

    for message in st.session_state.messages:
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue

        with st.chat_message(role):
            st.markdown(message.get("content", ""))
            sources = message.get("sources") or []
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(source_label(source))

    # -------------------- Chat input --------------------

    question = st.chat_input("Ask a question about your documents...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving documents and generating answer..."):
                try:
                    request_payload = {
                        "question": question,
                        "conversation_id": st.session_state.conversation_id,
                        # A selected document constrains retrieval to that
                        # document. All-documents mode enables automatic
                        # metadata-aware routing across the user's library.
                        "document_id": (
                            None if selected_document_id == ALL_DOCUMENTS
                            else selected_document_id
                        ),
                        "company": (
                            None if selected_document_id == ALL_DOCUMENTS
                            else selected_company
                        ),
                    }

                    response = requests.post(
                        f"{API_URL}/chat",
                        headers=get_headers(),
                        json=request_payload,
                        timeout=180,
                    )

                    if response.status_code == 200:
                        payload = response.json()
                        answer = payload.get(
                            "answer",
                            "The backend returned no answer.",
                        )
                        sources = payload.get("sources") or []
                        conversation_id = payload.get("conversation_id")

                        if conversation_id:
                            st.session_state.conversation_id = conversation_id
                            st.session_state.loaded_conversation_id = conversation_id
                            # The selector key is widget-owned after creation;
                            # conversation_id is synchronized before the next render.

                        st.markdown(answer)

                        if sources:
                            with st.expander("Sources"):
                                for source in sources:
                                    st.markdown(source_label(source))

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                            }
                        )
                        load_conversations(show_errors=False)

                    else:
                        show_api_error(response, "Chat")
                        st.session_state.messages.pop()

                except requests.RequestException as exc:
                    st.error(f"Backend connection failed: {exc}")
                    st.session_state.messages.pop()
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    st.error(f"Invalid chat API response: {exc}")
                    st.session_state.messages.pop()

else:
    st.info("Upload a document first, then select a search scope from the sidebar.")
