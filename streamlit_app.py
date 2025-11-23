import json
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------- CONFIG ----------

DATA_PATH = "complaints.xlsx"  # use your 100-row file
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"  # or "gpt-4o-mini" / any other chat model
TOP_K_FOR_CATEGORY = 5       # how many neighbours to vote on category/subcategory
TOP_K_CONTEXT = 5            # how many examples to show to the LLM

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ---------- DATA & INDEX ----------

@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = ["Complaint_Text", "Category", "Subcategory", "Bank_Response"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = df[col].fillna("").astype(str)
    return df


def get_embedding(text: str) -> np.ndarray:
    """Single text -> embedding vector."""
    text = text.replace("\n", " ")
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    )
    return np.array(resp.data[0].embedding, dtype="float32")


@st.cache_resource(show_spinner=False)
def build_embedding_index(complaints: list[str]) -> np.ndarray:
    """Precompute embeddings for all complaint texts."""
    embs = []
    for i, c in enumerate(complaints):
        emb = get_embedding(c)
        embs.append(emb)
    return np.vstack(embs)


def cosine_similarity_matrix_to_vec(matrix: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Cosine similarity between each row of matrix and vec."""
    vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
    mat_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return mat_norm @ vec_norm


# ---------- PRIORITY & ROUTING ----------

def assign_priority(category: str, subcategory: str, complaint_text: str) -> int:
    """
    Simple rule-based priority.
    1 = highest, 5 = lowest.
    """
    t = complaint_text.lower()
    cat = category.lower()
    sub = subcategory.lower()

    # Fraud / unauthorised / phishing
    if any(
        kw in t
        for kw in [
            "unauthorised", "unauthorized", "fraud", "scam",
            "phishing", "hacked", "card skimmed", "card cloned"
        ]
    ) or "unauthorised" in sub or "unauthorized" in sub:
        return 1

    # Account blocked / no access to money
    if any(
        kw in t
        for kw in [
            "account blocked", "account frozen", "unable to access",
            "cannot login", "login issue", "login failed", "account locked"
        ]
    ):
        return 2

    # Pension / senior citizens / differently abled
    if "pension" in t or "senior citizen" in t or "differently abled" in t:
        return 2

    # Loans / foreclosure / high ticket
    if "loan" in cat or "advances" in cat or "foreclosure" in t:
        return 3

    # Charges & fees
    if "charge" in t or "penal" in t or "fee" in t:
        return 3

    # Generic delays
    if any(kw in t for kw in ["delay", "no response", "not responded"]):
        return 4

    return 4


DEPARTMENT_MAPPING = {
    "atm/debit cards": [
        "Department of Payment and Settlement Systems",
        "Consumer Education and Protection Department",
    ],
    "credit cards": [
        "Department of Supervision",
        "Consumer Education and Protection Department",
    ],
    "internet/mobile/electronic banking": [
        "Department of Payment and Settlement Systems",
        "Department of Information Technology",
    ],
    "account opening/difficulty in operation of accounts": [
        "Consumer Education and Protection Department",
        "Department of Supervision",
    ],
    "mis-selling/para-banking": [
        "Department of Supervision",
        "Department of Regulation",
        "Enforcement Department",
    ],
    "recovery agents/direct sales agents": [
        "Department of Supervision",
        "Enforcement Department",
    ],
    "pension and facilities for senior citizens/differently abled": [
        "Consumer Education and Protection Department",
        "Financial Inclusion and Development Department",
    ],
    "loans and advances": [
        "Department of Regulation",
        "Department of Supervision",
    ],
    "charges / fees": [
        "Consumer Education and Protection Department",
    ],
    "cheques/drafts/bills": [
        "Department of Payment and Settlement Systems",
    ],
    "non-observance of fair practices code": [
        "Consumer Education and Protection Department",
        "Department of Regulation",
    ],
    "exchange of coins, issuance/acceptance of small denomination notes and coins": [
        "Department of Currency Management",
    ],
    "staff behaviour": [
        "Human Resource Management Department",
        "Consumer Education and Protection Department",
    ],
    "facilities for customers visiting the branch/adherence to prescribed working hours by the branch, etc.": [
        "Consumer Education and Protection Department",
    ],
    "others": [
        "Consumer Education and Protection Department",
    ],
}


def route_departments(category: str) -> list[str]:
    cat_l = category.lower()
    for key, depts in DEPARTMENT_MAPPING.items():
        if key in cat_l:
            return depts
    return ["Consumer Education and Protection Department"]


# ---------- PIPELINE STEPS ----------

def classify_category_and_subcategory(
    user_text: str, df: pd.DataFrame, embeddings_matrix: np.ndarray
):
    """
    Use semantic similarity against the whole dataset to decide category & subcategory
    (no training, just nearest-neighbour + majority vote).
    """
    user_emb = get_embedding(user_text)
    sims = cosine_similarity_matrix_to_vec(embeddings_matrix, user_emb)
    top_idx = np.argsort(-sims)[:TOP_K_FOR_CATEGORY]
    neigh = df.iloc[top_idx]

    # majority vote; fall back to top-1 if tie
    if not neigh["Category"].mode().empty:
        category = neigh["Category"].mode().iloc[0]
    else:
        category = neigh.iloc[0]["Category"]

    if not neigh["Subcategory"].mode().empty:
        subcategory = neigh["Subcategory"].mode().iloc[0]
    else:
        subcategory = neigh.iloc[0]["Subcategory"]

    return category, subcategory, user_emb, sims


def select_context_examples(
    user_emb: np.ndarray,
    df: pd.DataFrame,
    embeddings_matrix: np.ndarray,
    category: str,
    subcategory: str,
    top_n: int = TOP_K_CONTEXT,
) -> pd.DataFrame:
    """
    After category & subcategory are decided, filter DF to that slice,
    then retrieve top-N most similar complaints within that slice.
    """
    mask = (df["Category"] == category) & (df["Subcategory"] == subcategory)
    idxs = np.where(mask.values)[0]

    if len(idxs) == 0:
        # Fallback to global
        sims_global = cosine_similarity_matrix_to_vec(embeddings_matrix, user_emb)
        top_idx = np.argsort(-sims_global)[:top_n]
        out = df.iloc[top_idx].copy()
        out["similarity"] = sims_global[top_idx]
        return out

    sub_embs = embeddings_matrix[idxs]
    sims_sub = cosine_similarity_matrix_to_vec(sub_embs, user_emb)

    n = min(top_n, len(idxs))
    local_top = np.argsort(-sims_sub)[:n]
    chosen_idxs = [idxs[i] for i in local_top]

    out = df.iloc[chosen_idxs].copy()
    out["similarity"] = sims_sub[local_top]
    return out


def generate_fresh_reply(
    user_text: str,
    category: str,
    subcategory: str,
    examples_df: pd.DataFrame,
) -> str:
    """
    Use GPT as an 'agent' to craft a new reply,
    *informed by* but not copied from the examples.
    """
    examples_blocks = []
    for _, row in examples_df.iterrows():
        examples_blocks.append(
            f"Sample complaint:\n{row['Complaint_Text']}\n"
            f"Sample bank reply:\n{row['Bank_Response']}\n"
        )
    examples_text = "\n---\n".join(examples_blocks)

    system_msg = (
        "You are drafting responses for a bank / RBI-facing grievance redressal system. "
        "You will be given a customer complaint and some sample complaints+replies from a database. "
        "Your job is to write a *fresh*, formal, precise reply in bureaucratic Indian banking language. "
        "Do NOT copy sentences verbatim from the examples; instead, infer the appropriate structure and content. "
        "Acknowledge the grievance, briefly describe what will be examined, mention reference to relevant guidelines "
        "only at a high level, and clearly mention what the customer can expect next (e.g., reversal timelines, investigation, "
        "who will contact them). Keep everything in one cohesive text paragraph (or two short paragraphs)."
    )

    user_msg = (
        f"Customer complaint:\n{user_text}\n\n"
        f"Predicted category: {category}\n"
        f"Predicted subcategory: {subcategory}\n\n"
        "Here are sample complaints and replies for similar cases. "
        "Use them only as *examples of style and content*, not as text to copy:\n\n"
        f"{examples_text}\n\n"
        "Now write a fresh reply tailored to THIS customer's complaint."
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        response_format={"type": "text"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip()


# ---------- STREAMLIT UI ----------

st.set_page_config(
    page_title="RBI Complaint Router (Agentic)",
    page_icon="📨",
    layout="centered",
)

st.title("📨 RBI Complaint Router + Agentic Reply (No Training)")

st.markdown(
    """
Enter any **customer complaint** below.  
The app will:
1. Decide **Category & Subcategory**  
2. Assign a **Priority Level** and **Routing Department(s)**  
3. Use embeddings to pull **similar complaints** from the DB (within that category/subcategory)  
4. Ask GPT to craft a **fresh, formal reply**, using those examples as guidance (not as a template to copy).
"""
)

with st.spinner("Loading dataset and semantic index..."):
    df = load_dataset(DATA_PATH)
    emb_matrix = build_embedding_index(df["Complaint_Text"].tolist())

user_complaint = st.text_area(
    "Paste / type the customer complaint:",
    height=220,
    placeholder="Example: I tried to withdraw cash from your ATM in Mumbai. The machine did not dispense cash but my account was debited...",
)

if st.button("Analyse & Draft Reply", type="primary"):
    if not user_complaint.strip():
        st.warning("Please enter a complaint first.")
        st.stop()

    with st.spinner("Classifying complaint and fetching context..."):
        # 1. Decide category & subcategory from whole DB
        category, subcategory, user_emb, sims_all = classify_category_and_subcategory(
            user_complaint, df, emb_matrix
        )

        # 2. Priority & routing
        priority = assign_priority(category, subcategory, user_complaint)
        depts = route_departments(category)
        primary_dept = depts[0]
        extra_depts = depts[1:]

        # 3. Within that slice, pull similar complaints as examples
        examples_df = select_context_examples(
            user_emb, df, emb_matrix, category, subcategory, TOP_K_CONTEXT
        )

    with st.spinner("Letting the agent draft a fresh reply..."):
        reply_text = generate_fresh_reply(
            user_text=user_complaint,
            category=category,
            subcategory=subcategory,
            examples_df=examples_df,
        )

    # ---------- OUTPUT ----------

    st.subheader("📌 Classification & Routing")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Predicted Category**")
        st.write(category)
        st.markdown("**Predicted Subcategory**")
        st.write(subcategory)
    with col2:
        st.markdown("**Priority Level**")
        st.write(f"🔢 {priority}")
        st.markdown("**Primary Department**")
        st.write(primary_dept)
        if extra_depts:
            st.caption("Other relevant departments:")
            for d in extra_depts:
                st.caption(f"• {d}")

    st.subheader("📝 Fresh Agent-Generated Reply")
    st.write(reply_text)

    with st.expander("🔍 Context: examples used from the DB"):
        st.dataframe(
            examples_df[
                ["similarity", "Complaint_Text", "Category", "Subcategory", "Bank_Response"]
            ]
        )
