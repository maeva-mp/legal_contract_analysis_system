# AI-Powered Legal Contract Analysis System (for Mauritius)

# Imports
import os
import io
import math
import json

#reads text cases from a JSON file/writes new ones to that file
GROUND_TRUTH_PATH = "ground_truth_cases.json"
def load_ground_truth_cases():
    if os.path.exists(GROUND_TRUTH_PATH):
        with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
def save_ground_truth_cases(data):
    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import streamlit as st
from PyPDF2 import PdfReader
import docx
from sentence_transformers import SentenceTransformer
import faiss
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import re
import numpy as np
from rouge_score import rouge_scorer
import spacy
import matplotlib.pyplot as plt

# Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARIZER_MODEL_NAME = "t5-small"
CHUNK_TOKENS = 300
EMBED_BATCH = 32
TOP_K = 5
FAISS_INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.txt"
METRICS_FILE = "metrics.json"
GROUND_TRUTH_FILE = "ground_truth.json"

# Contract-specific risk keywords
RISK_KEYWORDS = {
    "critical": [
        "unlimited liability", "indemnify", "indemnification", "hold harmless",
        "waive", "waiver", "terminate without cause", "unilateral termination",
        "exclusive remedy", "penalty", "liquidated damages", "non-compete"
    ],
    "high": [
        "liability", "breach", "default", "termination", "damages",
        "dispute", "arbitration", "confidential", "proprietary",
        "intellectual property", "warranty disclaimer"
    ],
    "medium": [
        "payment terms", "delivery", "deadline", "milestone",
        "force majeure", "notice period", "renewal", "amendment"
    ]
}

# Mauritius-specific compliance patterns
MAURITIUS_COMPLIANCE = {
    "workers_rights_act": [
        "employment contract", "written particulars", "14 days",
        "workers rights", "termination notice", "severance"
    ],
    "civil_code": [
        "article 1108", "consent", "capacity", "object", "cause",
        "potestative", "three year", "limitation period"
    ],
    "data_protection": [
        "personal data", "data protection", "gdpr", "consent",
        "data subject", "processing"
    ]
}

# Text Extraction
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs if p.text and not p.text.isspace()])

def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode(errors="ignore")

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b\d+\s*\n', '', text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and len(ln.strip()) > 3]
    return "\n".join(lines)

def chunk_text(text: str, approx_tokens: int = CHUNK_TOKENS, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + approx_tokens)
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
        if start >= len(words) - overlap:
            break
    return chunks

# Load NER Model
@st.cache_resource
def load_spacy_model():
    """Load spaCy model for Named Entity Recognition"""
    try:
        return spacy.load("en_core_web_sm")
    except:
        st.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        return None

# Contract Analysis Functions
def extract_entities(text: str, nlp):
    """Extract dates, money, organizations, and people using NER"""
    if nlp is None:
        return {"dates": [], "money": [], "orgs": [], "persons": []}
    
    doc = nlp(text[:100000])  # Limit text length 
    
    entities = {
        "dates": [],
        "money": [],
        "orgs": [],
        "persons": []
    }
    
    for ent in doc.ents:
        if ent.label_ == "DATE":
            entities["dates"].append(ent.text)
        elif ent.label_ == "MONEY":
            entities["money"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["orgs"].append(ent.text)
        elif ent.label_ == "PERSON":
            entities["persons"].append(ent.text)
    
    # Remove duplicates while preserving order
    for key in entities:
        entities[key] = list(dict.fromkeys(entities[key]))
    
    return entities

def detect_risk_level(text: str) -> Tuple[str, List[str], float]:
    """Detect risk level and matched keywords in contract text"""
    text_lower = text.lower()
    matched_keywords = []
    risk_score = 0.0
    
    # Check for critical risks
    for keyword in RISK_KEYWORDS["critical"]:
        if keyword in text_lower:
            matched_keywords.append((keyword, "critical"))
            risk_score += 3.0
    
    # Check for high risks
    for keyword in RISK_KEYWORDS["high"]:
        if keyword in text_lower:
            matched_keywords.append((keyword, "high"))
            risk_score += 2.0
    
    # Check for medium risks
    for keyword in RISK_KEYWORDS["medium"]:
        if keyword in text_lower:
            matched_keywords.append((keyword, "medium"))
            risk_score += 1.0
    
    # Determine overall risk level
    if risk_score >= 10:
        risk_level = "CRITICAL"
    elif risk_score >= 5:
        risk_level = "HIGH"
    elif risk_score >= 2:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return risk_level, matched_keywords, risk_score

def check_mauritius_compliance(text: str) -> Dict[str, List[str]]:
    """Check compliance with Mauritius regulations"""
    text_lower = text.lower()
    compliance_findings = {}
    
    for law, patterns in MAURITIUS_COMPLIANCE.items():
        matches = [p for p in patterns if p in text_lower]
        if matches:
            compliance_findings[law] = matches
    
    return compliance_findings

def extract_obligations(text: str, nlp) -> List[Dict]:
    """Extract key obligations, deadlines, and payment terms"""
    obligations = []
    
    # Find sentences with obligation keywords
    obligation_keywords = ["shall", "must", "will", "agree to", "required to", "obligated"]
    deadline_keywords = ["within", "by", "before", "after", "days", "date"]
    payment_keywords = ["payment", "pay", "fee", "amount", "price", "cost"]
    
    sentences = text.split('.')
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Check for obligations
        if any(kw in sentence_lower for kw in obligation_keywords):
            obligation_type = "General"
            
            # Classify obligation type
            if any(kw in sentence_lower for kw in deadline_keywords):
                obligation_type = "Deadline"
            elif any(kw in sentence_lower for kw in payment_keywords):
                obligation_type = "Payment"
            
            obligations.append({
                "text": sentence.strip(),
                "type": obligation_type
            })
    
    return obligations[:20]  # Limit to top 20

def calculate_deadline_warnings(dates: List[str]) -> List[Dict]:
    """Calculate time-bar warnings (3-year Mauritius Civil Code limitation)"""
    warnings = []
    
    for date_str in dates:
        # Simple heuristic: look for year patterns
        year_match = re.search(r'\b(20\d{2})\b', date_str)
        if year_match:
            year = int(year_match.group(1))
            current_year = datetime.now().year
            years_diff = current_year - year
            
            if years_diff >= 2:
                warnings.append({
                    "date": date_str,
                    "warning": f"⚠️ {years_diff} years old - approaching 3-year Civil Code limitation",
                    "critical": years_diff >= 3
                })
    
    return warnings

# Embeddings and FAISS management
def build_faiss_index(embedding_dim: int) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embedding_dim)
    return index

def normalize_vectors(vectors):
    import numpy as np
    norms = (vectors**2).sum(axis=1, keepdims=True) ** 0.5
    norms[norms == 0] = 1.0
    return vectors / norms

def save_metadata(metadata: List[Tuple[str, int, str]], path: str = METADATA_FILE):
    with open(path, "w", encoding="utf-8") as f:
        for doc_id, idx, text in metadata:
            safe = text.replace("\n", "\\n")
            f.write(f"{doc_id}\t{idx}\t{safe}\n")

def load_metadata(path: str = METADATA_FILE) -> List[Tuple[str, int, str]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            doc_id, idx_s, text_safe = line.rstrip("\n").split("\t", 2)
            text = text_safe.replace("\\n", "\n")
            rows.append((doc_id, int(idx_s), text))
    return rows

#Evaluation Metrics
def calculate_precision_recall(retrieved_docs: List[str], relevant_docs: List[str]) -> Tuple[float, float]:
    """Calculate precision and recall"""            #if no doc were retrieved, 
    if not retrieved_docs:
        return 0.0, 0.0
    
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    
    true_positives = len(retrieved_set & relevant_set)
    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(relevant_set) if relevant_set else 0.0
    
    return precision, recall

def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1 score"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_mrr(results: List[Dict], relevant_doc: str) -> float:
    """Calculate Mean Reciprocal Rank"""
    for i, result in enumerate(results, 1):
        if result['doc_id'] == relevant_doc:
            return 1.0 / i
    return 0.0

def calculate_ndcg(results: List[Dict], relevance_scores: Dict[str, float], k: int = 5) -> float:
    """Calculate Normalized Discounted Cumulative Gain"""
    dcg = 0.0
    for i, result in enumerate(results[:k], 1):
        rel = relevance_scores.get(result['doc_id'], 0.0)
        dcg += rel / math.log2(i + 1)
    
    # Ideal DCG
    ideal_scores = sorted(relevance_scores.values(), reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_scores[:k]))
    
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_rouge(generated: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE scores for summarization"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def save_metrics(metrics: Dict, path: str = METRICS_FILE):
    """Save evaluation metrics to JSON"""
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_metrics(path: str = METRICS_FILE) -> Dict:
    """Load evaluation metrics from JSON"""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

# Summarization
def summarize_text(summarizer_model, tokenizer, text: str, max_length: int = 150) -> str:
    words = text.split()
    if len(words) > 400:
        chunk_size = 400
        summaries = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            input_chunk = "summarize: " + chunk
            inputs = tokenizer.encode(input_chunk, return_tensors="pt", truncation=True, max_length=512)
            summary_ids = summarizer_model.generate(inputs, max_length=max_length // 2, min_length=20,
                                                   length_penalty=2.0, num_beams=4, early_stopping=True)
            summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
        combined = " ".join(summaries)
        input_text = "summarize: " + combined
    else:
        input_text = "summarize: " + text
    
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
    summary_ids = summarizer_model.generate(inputs, max_length=max_length, min_length=30,
                                           length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Models
@st.cache_resource
def load_models():
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    summarizer_tokenizer = T5TokenizerFast.from_pretrained(SUMMARIZER_MODEL_NAME)
    summarizer = T5ForConditionalGeneration.from_pretrained(SUMMARIZER_MODEL_NAME)
    nlp = load_spacy_model()
    return embedder, summarizer, summarizer_tokenizer, nlp

def index_documents(files: List[Tuple[str, bytes]]):
    embedder, _, _, _ = load_models()
    all_chunks = []
    metadata = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for file_idx, (filename, file_bytes) in enumerate(files):
        status_text.text(f"Processing {filename}...")
        progress_bar.progress((file_idx + 1) / len(files))
        
        ext = filename.lower().split(".")[-1]
        if ext == "pdf":
            raw = extract_text_from_pdf(file_bytes)
        elif ext in ("docx", "doc"):
            raw = extract_text_from_docx(file_bytes)
        else:
            raw = extract_text_from_txt(file_bytes)
        
        raw = clean_text(raw)
        chunks = chunk_text(raw)
        
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            metadata.append((filename, i, c))
    
    if not all_chunks:
        st.warning("No text extracted from uploaded files.")
        return None, None
    
    status_text.text("Generating embeddings...")
    vectors = []
    for i in range(0, len(all_chunks), EMBED_BATCH):
        batch = all_chunks[i:i+EMBED_BATCH]
        emb = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        vectors.append(emb)
    
    vectors = np.vstack(vectors)
    vectors = normalize_vectors(vectors)
    dim = vectors.shape[1]
    index = build_faiss_index(dim)
    index.add(vectors)
    
    faiss.write_index(index, FAISS_INDEX_FILE)
    save_metadata(metadata)
    
    progress_bar.empty()
    status_text.empty()
    st.success(f"✅ Indexed {len(all_chunks)} chunks from {len(files)} contract(s).")
    return index, metadata

def load_index_and_metadata():
    if os.path.exists(FAISS_INDEX_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
        metadata = load_metadata()
        return index, metadata
    return None, []

def search(query: str, top_k: int = TOP_K):
    embedder, _, _, _ = load_models()
    index, metadata = load_index_and_metadata()
    
    if index is None or len(metadata) == 0:
        st.warning("No index found. Please upload and index documents first.")
        return []
    
    qv = embedder.encode([query], convert_to_numpy=True)
    qv = normalize_vectors(qv)
    D, I = index.search(qv, top_k)
    
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        doc_id, chunk_idx, chunk_text = metadata[idx]
        results.append({
            "score": float(score),
            "doc_id": doc_id,
            "chunk_idx": chunk_idx,
            "text": chunk_text
        })
    
    return results

# Streamlit UI
st.set_page_config(page_title="SmartContract MU", layout="wide", page_icon="⚖️")

#-------------------------------------
# Header
st.title("⚖️ SmartContract MU")
st.caption("- AI-Powered Legal Contract Analysis ")

#------------------------------
# Sidebar
with st.sidebar:
    st.markdown("## 📊 SmartContract MU Overview")
    _, metadata = load_index_and_metadata()

    if metadata:
        num_docs = len(set(m[0] for m in metadata))
        num_chunks = len(metadata)

        # 📄 Contract Stats
        st.markdown("### 📁 Indexed Contracts")
        col1, col2 = st.columns(2)
        col1.metric("Contracts", num_docs)
        col2.metric("Chunks", num_chunks)

        # 📈 Latest Evaluation Metrics
        metrics = load_metrics()
        if metrics:
            st.markdown("### 📈 Evaluation Snapshot")
            for key, value in metrics.items():
                st.markdown(f"""
                <div style='background-color:#f0f8ff;
                            border-left: 5px solid #1891C3;
                            padding: 8px 12px;
                            border-radius: 6px;
                            margin-bottom: 6px'>
                    <b>{key.upper()}</b>: {value:.3f}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📂 No contracts indexed yet")

    # 📜 Legal Compliance Section
    st.markdown("---")
    st.markdown("### 🇲🇺 Mauritius Compliance")
    st.markdown("""
    <ul style='padding-left:15px'>
        <li>✓ Workers' Rights Act 2019</li>
        <li>✓ Civil Code Articles</li>
        <li>✓ Data Protection Act 2017</li>
    </ul>
    """, unsafe_allow_html=True)

    # 📌 Footer
    st.markdown("---")
st.caption("- ✓ Data Protection Act 2017")
st.caption("- Built with ❤️ for Mauritius legal teams • Powered by AI")

# Main Tabs
tabs = st.tabs([
    "📁 Upload Contracts",
    "🔍 Contract Analysis",
    "⚖️ Risk Assessment",
    "🔎 Search & Compare",
    "📈 Evaluation"
])

# TAB 1: Upload & Index
with tabs[0]:
    st.header("📁 Upload and Index Contracts")
    st.markdown("Upload PDF, DOCX, or TXT contracts for analysis")
    
    uploaded = st.file_uploader(
        "Select contract files",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt"],
        key="upload"
    )
    
    if st.button("🚀 Index Contracts", type="primary"):
        if not uploaded:
            st.warning("Please upload at least one contract file")
        else:
            files = [(f.name, f.read()) for f in uploaded]
            with st.spinner("Indexing contracts..."):
                index_documents(files)

# TAB 2: Contract Analysis
with tabs[1]:
    st.header("🔍 Complete Contract Analysis")
    
    _, metadata = load_index_and_metadata()
    if not metadata:
        st.info("👆 Please upload and index contracts first")
    else:
        # Select contract to analyze
        contract_names = sorted(list(set(m[0] for m in metadata)))
        selected_contract = st.selectbox("Select Contract", contract_names)
        
        if st.button("🔬 Analyze Contract", type="primary"):
            # Get full text of selected contract
            contract_chunks = [m[2] for m in metadata if m[0] == selected_contract]
            full_text = "\n\n".join(contract_chunks)
            
            _, summarizer, tokenizer, nlp = load_models()
            
            with st.spinner("Analyzing contract..."):
                # Analysis sections
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk Analysis
                    st.subheader("🚨 Risk Analysis")
                    risk_level, risk_keywords, risk_score = detect_risk_level(full_text)
                    
                    # Display risk level with color
                    risk_colors = {
                        "CRITICAL": "🔴",
                        "HIGH": "🟠",
                        "MEDIUM": "🟡",
                        "LOW": "🟢"
                    }
                    
                    st.markdown(f"### {risk_colors.get(risk_level, '⚪')} {risk_level} RISK")
                    st.metric("Risk Score", f"{risk_score:.1f}")
                    
                    if risk_keywords:
                        st.markdown("**Detected Risk Factors:**")
                        for keyword, level in risk_keywords[:10]:
                            st.markdown(f"- {keyword} ({level})")
                    
                    # Compliance Check
                    st.subheader("✅ Mauritius Compliance")
                    compliance = check_mauritius_compliance(full_text)
                    
                    if compliance:
                        for law, matches in compliance.items():
                            with st.expander(f"📋 {law.replace('_', ' ').title()}"):
                                for match in matches:
                                    st.markdown(f"✓ {match}")
                    else:
                        st.info("No specific compliance patterns detected")
                
                with col2:
                    # Entity Extraction
                    st.subheader("🏢 Extracted Entities")
                    entities = extract_entities(full_text, nlp)
                    
                    if entities["dates"]:
                        with st.expander(f"📅 Dates ({len(entities['dates'])})"):
                            for date in entities["dates"][:15]:
                                st.markdown(f"- {date}")
                        
                        # Time-bar warnings
                        warnings = calculate_deadline_warnings(entities["dates"])
                        if warnings:
                            st.warning(f"⚠️ {len(warnings)} potential time-bar concerns")
                            for w in warnings[:5]:
                                st.markdown(f"- {w['date']}: {w['warning']}")
                    
                    if entities["money"]:
                        with st.expander(f"💰 Amounts ({len(entities['money'])})"):
                            for amount in entities["money"][:10]:
                                st.markdown(f"- {amount}")
                    
                    if entities["orgs"]:
                        with st.expander(f"🏢 Organizations ({len(entities['orgs'])})"):
                            for org in entities["orgs"][:10]:
                                st.markdown(f"- {org}")
                    
                    if entities["persons"]:
                        with st.expander(f"👤 Persons ({len(entities['persons'])})"):
                            for person in entities["persons"][:10]:
                                st.markdown(f"- {person}")
                
                # Obligations
                st.subheader("📋 Key Obligations")
                obligations = extract_obligations(full_text, nlp)
                
                if obligations:
                    tab_payment, tab_deadline, tab_general = st.tabs(["💰 Payment", "⏰ Deadlines", "📄 General"])
                    
                    with tab_payment:
                        payment_obs = [o for o in obligations if o["type"] == "Payment"]
                        for ob in payment_obs[:5]:
                            st.markdown(f"• {ob['text']}")
                    
                    with tab_deadline:
                        deadline_obs = [o for o in obligations if o["type"] == "Deadline"]
                        for ob in deadline_obs[:5]:
                            st.markdown(f"• {ob['text']}")
                    
                    with tab_general:
                        general_obs = [o for o in obligations if o["type"] == "General"]
                        for ob in general_obs[:5]:
                            st.markdown(f"• {ob['text']}")
                else:
                    st.info("No explicit obligations detected")
                
                # Summary
                st.subheader("📝 Executive Summary")
                with st.spinner("Generating summary..."):
                    summary = summarize_text(summarizer, tokenizer, full_text, max_length=200)
                    st.info(summary)

# TAB 3:Risk Assessment Dashboard
with tabs[2]:
    st.header("## ⚖️ Risk Assessment Dashboard")

    _, metadata = load_index_and_metadata()
    if not metadata:
        st.info("👆 Please upload contracts first")
    else:
        st.markdown("### 📋 Contract Risk Overview")

        contract_names = sorted(list(set(m[0] for m in metadata)))
        risk_data = []

        with st.spinner("Analyzing all contracts..."):
            for contract_name in contract_names:
                contract_chunks = [m[2] for m in metadata if m[0] == contract_name]
                full_text = "\n\n".join(contract_chunks)
                risk_level, _, risk_score = detect_risk_level(full_text)
                risk_data.append({
                    "Contract": contract_name,
                    "Risk Level": risk_level,
                    "Risk Score": risk_score
                })

        import pandas as pd
        df = pd.DataFrame(risk_data)
        df.columns = df.columns.str.replace(" ", "_")

        # Define elegant color palette
        def risk_color(level):
            return {
                "CRITICAL": "#003B6F",
                "HIGH": "#1891C3",
                "MEDIUM": "#3DC6C3",
                "LOW": "#3AC0DA"
            }.get(level, "#50E3C2")

        # Render styled cards
        for row in df.itertuples():
            row_dict = row._asdict()
            level = row_dict["Risk_Level"]
            score = row_dict["Risk_Score"]
            contract = row_dict["Contract"]
            background = f"{risk_color(level)}33"

            st.markdown(f"""
            <div style='background-color:{background};
                        border-left: 5px solid {risk_color(level)};
                        padding: 10px 15px;
                        border-radius: 8px;
                        margin-bottom: 10px'>
                <h4 style='margin-bottom:5px'>{contract}</h4>
                <b>Risk Level:</b> {level} <br>
                <b>Risk Score:</b> {score:.2f}
            </div>
            """, unsafe_allow_html=True)

        # View toggle: Summary or Chart
        view = st.radio("📊 Choose view:", ["Summary", "Chart"], horizontal=True)

        risk_counts = df['Risk_Level'].value_counts()
        levels = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

        if view == "Summary":
            st.markdown("### 📋 Risk Summary")
            for level in levels:
                count = risk_counts.get(level, 0)
                st.markdown(f"""
                <div style='display:inline-block; background-color:{risk_color(level)}33;
                            border-left: 5px solid {risk_color(level)};
                            padding: 10px 20px; margin: 5px;
                            border-radius: 8px; min-width: 120px; text-align:center'>
                    <h4 style='margin:0'>{level}</h4>
                    <b>{count} contract{'s' if count != 1 else ''}</b>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("### 📊 Risk Score by Contract")

            chart_data = df.copy()

            import altair as alt
            color_scale = alt.Scale(
                domain=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                range=["#003B6F", "#1891C3", "#3DC6C3", "#3AC0DA"]
            )

            bar = alt.Chart(chart_data).mark_bar(size=25, opacity=0.85).encode(
                y=alt.Y('Contract', sort='-x', title='Contract'),
                x=alt.X('Risk_Score', title='Risk Score'),
                color=alt.Color('Risk_Level', scale=color_scale, legend=None),
                tooltip=['Contract', 'Risk_Level', 'Risk_Score']
            ).properties(
                height=200,
                title="📊 Risk Score by Contract"
            ).configure_axis(
                labelFontSize=12,
                titleFontSize=13
            ).configure_title(
                fontSize=14,
                anchor='start',
                color='#444'
            )

            st.altair_chart(bar, use_container_width=True)
            st.caption("📊 Each bar shows a contract’s risk score and level. Hover to see details.")

# TAB 4: Search & Compare
with tabs[3]:
    st.header("🔍 Search & Compare Contracts")

    _, metadata = load_index_and_metadata()
    if not metadata:
        st.info("👆 Please upload contracts first")
    else:
        contract_names = sorted(list(set(m[0] for m in metadata)))

        # 🔎 Search Section
        st.markdown("### 🔎 Search Contracts")
        query = st.text_input("Search for specific clauses or terms")
        col1, col2 = st.columns([3, 1])
        with col1:
            k = st.slider("Number of results", min_value=1, max_value=10, value=5)
        with col2:
            show_summary = st.checkbox("Generate Summary", value=True)

        if st.button("🔍 Search", type="primary"):
            if not query:
                st.warning("Please enter a search query")
            else:
                results = search(query, top_k=k)

                if not results:
                    st.info("No results found")
                else:
                    st.subheader(f"📑 Top {len(results)} Results")

                    concatenated = []
                    for idx, r in enumerate(results, 1):
                        risk_level, _, _ = detect_risk_level(r["text"])
                        with st.expander(
                            f"Result {idx}: {r['doc_id']} (Score: {r['score']:.3f}) - Risk: {risk_level}"
                        ):
                            st.markdown(r["text"])
                            st.caption(f"Chunk {r['chunk_idx']}")
                        concatenated.append(r["text"])

                    if show_summary and concatenated:
                        _, summarizer, tokenizer, _ = load_models()
                        joined = "\n\n".join(concatenated)
                        if len(joined.split()) >= 20:
                            with st.spinner("Generating summary..."):
                                summary = summarize_text(summarizer, tokenizer, joined)
                                st.subheader("📝 Summary")
                                st.info(summary)

        # ⚖️ Compare Section
        st.markdown("### ⚖️ Compare Two Contracts")
        col1, col2 = st.columns(2)
        with col1:
            contract1 = st.selectbox("Select first contract", contract_names)
        with col2:
            contract2 = st.selectbox("Select second contract", [c for c in contract_names if c != contract1])

        if contract1 and contract2:
            text1 = "\n\n".join([m[2] for m in metadata if m[0] == contract1])
            text2 = "\n\n".join([m[2] for m in metadata if m[0] == contract2])

            risk1, _, score1 = detect_risk_level(text1)
            risk2, _, score2 = detect_risk_level(text2)

            st.markdown("### 📊 Comparison Result")
            st.markdown(f"""
            <div style='display:flex; gap:30px'>
                <div style='background-color:{risk_color(risk1)}33;
                            border-left: 5px solid {risk_color(risk1)};
                            padding: 10px 15px;
                            border-radius: 8px; width:45%'>
                    <h4>{contract1}</h4>
                    <b>Risk Level:</b> {risk1}<br>
                    <b>Risk Score:</b> {score1:.2f}
                </div>
                <div style='background-color:{risk_color(risk2)}33;
                            border-left: 5px solid {risk_color(risk2)};
                            padding: 10px 15px;
                            border-radius: 8px; width:45%'>
                    <h4>{contract2}</h4>
                    <b>Risk Level:</b> {risk2}<br>
                    <b>Risk Score:</b> {score2:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

# TAB 5: Evaluation
with tabs[4]:
    st.header("📈 Evaluation & Metrics")
    st.markdown("Test system accuracy with ground truth data")

    # ➕ Add Ground Truth Test Case
    with st.expander("➕ Add Ground Truth Test Case"):
        with st.form("ground_truth_form_eval"):
            test_query = st.text_input("Test Query (e.g., 'liability clauses')")
            relevant_doc = st.text_area("Expected Relevant Content", height=100)

            if st.form_submit_button("Add Test Case"):
                if test_query and relevant_doc:
                    ground_truth = load_ground_truth()
                    ground_truth[test_query] = relevant_doc
                    save_ground_truth(ground_truth)
                    st.success("✅ Test case added!")

    # 📋 Display Saved Test Cases
    ground_truth = load_ground_truth_cases()
    if ground_truth:
        st.markdown("#### 📋 Saved Test Cases")
        for query, doc in ground_truth.items():
            with st.expander(f"🔎 Query: {query}"):
                st.markdown(f"**Expected Content:**\n\n{doc[:200]}...")

    st.divider()

    # 🧪 Run Evaluation
    if st.button("🧪 Run Full Evaluation", type="primary"):
        if not ground_truth:
            st.warning("⚠️ No test cases found. Please add ground truth data first.")
        else:
            with st.spinner("Running evaluation..."):
                all_metrics = {
                    'precision': [],
                    'recall': [],
                    'f1': [],
                    'mrr': []
                }
                detailed_results = []

                for query, relevant_doc in ground_truth.items():
                    results = search(query, top_k=5)
                    retrieved_docs = [r['text'] for r in results]

                    precision, recall = calculate_precision_recall(retrieved_docs, [relevant_doc])
                    f1 = calculate_f1(precision, recall)
                    mrr = calculate_mrr(results, relevant_doc)

                    top_match = results[0]['text'] if results else ""
                    similarity = 1.0 if relevant_doc.strip().lower() in top_match.lower() else 0.0
                    match = similarity > 0.75  # adjustable threshold

                    detailed_results.append({
                        "Query": query,
                        "Expected": relevant_doc,
                        "Top Match": top_match,
                        "Similarity": similarity,
                        "Match": match,
                        "Precision": precision,
                        "Recall": recall,
                        "F1": f1,
                        "MRR": mrr
                    })

                avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in all_metrics.items()}
                save_metrics(avg_metrics)

                st.success("✅ Evaluation complete!")

                # 📊 Metric Summary
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Precision", f"{avg_metrics['precision']:.3f}")
                col2.metric("Recall", f"{avg_metrics['recall']:.3f}")
                col3.metric("F1 Score", f"{avg_metrics['f1']:.3f}")
                col4.metric("MRR", f"{avg_metrics['mrr']:.3f}")

                # 📘 Metric Interpretation
                st.info("""
                **Metric Interpretation:**
                - **Precision**: % of retrieved results that are relevant  
                - **Recall**: % of relevant documents that were retrieved  
                - **F1 Score**: Harmonic mean of precision and recall  
                - **MRR**: Mean Reciprocal Rank (position of first relevant result)  

                **Target**: All metrics should be > 0.85 for production use
                """)

                # 🔍 Detailed Evaluation Results
                st.markdown("### 🔍 Detailed Evaluation Results")
                for r in detailed_results:
                    st.markdown(f"""
                    <div style='background-color:#f0f8ff;
                                border-left: 5px solid #1891C3;
                                padding: 10px 15px;
                                border-radius: 8px;
                                margin-bottom: 10px'>
                        <h4>{r["Query"]}</h4>
                        <p><b>Expected:</b> {r["Expected"]}</p>
                        <p><b>Top Match:</b> {r["Top Match"]}</p>
                        <p><b>Similarity:</b> {r["Similarity"]:.2f}</p>
                        <p><b>Match:</b> {"✅" if r["Match"] else "❌"}</p>
                    </div>
                    """, unsafe_allow_html=True)
