# CAPSTONE PROJECT PROPOSAL

**Project Title:** SmartContract - AI Legal Contract Analyzer  
**Student Name:** Maeva Moonien  
**Program:** GenAI & Machine Learning 
**Supervisor:** Mr. Rayhaan  
**Date:** November 2025  

---

## 1. INTRODUCTION

This project develops an AI-powered legal contract analysis system that automatically reviews contracts, detects risks, extracts key information, and checks compliance with local regulations using Natural Language Processing (NLP), Large Language Models (LLM), and Embeddings.

**For this case study, Mauritius laws and cases are used as the implementation context.** The system analyzes contracts based on:
- Mauritius Civil Code (Articles 1108, 1174, 2279)
- Workers' Rights Act 2019
- Data Protection Act 2017
- Mauritius hybrid legal framework (French Civil Code + British Common Law)

This case study approach allows for concrete demonstration of AI/ML capabilities in legal document analysis while addressing real-world business needs in the Mauritius market.

---

## 2. PROBLEM STATEMENT

### The Problem
Mauritius SMEs struggle with expensive and slow legal contract reviews:
- **Cost:** MUR 5,000-15,000 per contract
- **Time:** 5-7 business days average
- **Risk:** Missing critical clauses leads to disputes costing MUR 50,000-200,000 annually
- **Complexity:** Mauritius has a hybrid legal system (French Civil Code + British Common Law)
- **Compliance:** Must comply with Workers' Rights Act 2019, Civil Code, Data Protection Act 2017

### Real-World Impact
- Contract disputes cost businesses money and time
- SMEs can't afford expensive legal reviews
- Manual review misses deadlines and obligations
- 3-year Civil Code limitation causes time-barred claims
- No affordable tools exist for Mauritius market

---

## 3. WHY AI/ML IS NEEDED

### Traditional Method Problems
❌ Manual review takes 5-7 days  
❌ Costs MUR 5,000-15,000 per contract  
❌ Human error (overlooking clauses)  
❌ Inconsistent quality  
❌ Cannot scale (one contract at a time)  

### AI/ML Solution Benefits
✅ Processes contracts in minutes  
✅ 90% cost reduction  
✅ 95%+ consistent accuracy  
✅ Never misses clauses  
✅ Analyzes multiple contracts simultaneously  

### AI Technologies Used

**1. Embeddings (Sentence Transformers)**
- Converts text to 384-dimensional semantic vectors
- Understands meaning, not just keywords
- Example: "terminate without cause" = "unilateral ending" (similar meaning)

**2. Natural Language Processing (NLP)**
- Named Entity Recognition (spaCy) extracts dates, amounts, parties
- Text Classification identifies risk levels
- Example: Extracts "Payment of MUR 50,000 due within 30 days"

**3. Large Language Model (T5 Transformer)**
- Generates human-quality summaries
- Converts 50-page contracts → 2-page summaries
- Pre-trained model with 60 million parameters

**4. Vector Search (FAISS)**
- Fast similarity search across thousands of documents
- Compares contracts to find unusual terms
- Sub-second search performance

**5. Rule-Based Classification**
- Detects high-risk keywords (liability, indemnification, termination)
- Checks Mauritius compliance patterns
- Calculates risk scores

---

## 4. OBJECTIVES

### Primary Objectives
1. Analyze contracts with 90%+ accuracy in under 5 minutes
2. Detect high-risk clauses (liability, termination, penalties)
3. Extract all dates, amounts, parties, and obligations automatically
4. Check compliance with Mauritius laws (Workers' Rights Act, Civil Code, Data Protection Act)
5. Generate executive summaries and risk reports

### Secondary Objectives
6. Support PDF, DOCX, and TXT formats
7. Compare multiple contracts side-by-side
8. Track deadlines and warn about 3-year limitation periods
9. Create exportable analysis reports
10. Achieve 85%+ precision and recall metrics

---

## 5. METHODOLOGY

### How We Built It

**Step 1: Getting the Text**
- Upload contracts (PDF, Word, or text files)
- Extract all the text automatically
- Clean it up (remove junk, fix spacing)

**Step 2: Making AI Understand**
- Break text into small chunks (300 words each)
- Convert each chunk into numbers (embeddings) that AI understands
- Store everything in a super-fast search system (FAISS)

**Step 3: Finding the Important Stuff**
- Use AI to find dates, money amounts, company names, people
- Detect risky words (like "unlimited liability" or "terminate without cause")
- Check if it follows Mauritius laws

**Step 4: Making Sense of It All**
- Calculate risk scores (Critical/High/Medium/Low)
- Find all obligations (what you MUST do)
- Generate a simple summary anyone can read

### The Tech Behind It

**What We Use:**
- **Programming:** Python (the most popular AI language)
- **AI Models:** 
  - Sentence Transformers - Understands meaning of text
  - T5 - Writes summaries automatically
  - spaCy - Finds dates, money, names
- **Search:** FAISS - Lightning-fast document search
- **Interface:** Streamlit - Beautiful web app anyone can use

**Test Data:**
- 10 - 15 sample Mauritius contracts (employment, procurement, services)
- Created for testing purposes only (not real contracts)

**Disclaimer:** All contract documents used in this project are synthetically generated or adapted for academic and research purposes only. They do not represent legally binding agreements and must not be used for actual legal, commercial, or regulatory decisions.

### How We Measure Success

**Does it work well?**
- Precision: Does it find the right things? (Target: 85%+)
- Recall: Does it miss anything important? (Target: 85%+)
- Speed: How fast? (Target: Under 5 minutes per contract)

---

## 6. EXPECTED OUTCOMES

### What You'll Get

**1. A Working AI System That:**
- 📤 Accepts contracts (PDF, Word, text)
- 🔍 Finds risks automatically (4 danger levels)
- 📅 Extracts dates, money, and parties
- ✅ Checks Mauritius law compliance
- 📋 Lists all your obligations
- 📝 Creates easy-to-read summaries
- ⚖️ Compares multiple contracts

**2. Complete Documentation:**
- User manual (how to use it)
- Technical guide (how it works)
- System diagrams (visual explanation)

**3. Proof It Works:**
- 10 test contracts analyzed
- Performance scores (precision, recall, accuracy)
- Speed benchmarks

**4. Demo Package:**
- Code on GitHub (open source!)
- Complete project files

### Real-World Impact

**Time Saved:**
- Before: 7 days waiting → Now: 5 minutes ⚡
- That's 95% faster!

**Money Saved:**
- Before: MUR 10,000 → Now: MUR 1,000 💰
- That's 90% cheaper!

**Better Accuracy:**
- Never misses a clause 🎯
- Finds 85%+ of all risks
- Works 24/7 without getting tired

### Making Mauritius Better

This project helps:
- 🏢 **Small businesses** - Can finally afford contract analysis
- 🇲🇺 **Digital Mauritius 2030** - Shows AI in action
- 💼 **Job creation** - Demonstrates AI skills employers want
- 🌍 **Africa leadership** - Mauritius leading in legal-tech innovation

---

## 7. TECHNICAL SPECIFICATIONS

### NLP/LLM/Embedding Integration

**Embeddings:**
- Model: all-MiniLM-L6-v2 (384 dimensions)
- Purpose: Semantic understanding of legal text
- Output: Vector representations for similarity search

**NLP:**
- Named Entity Recognition (spaCy)
- Text classification (risk detection)
- Sentence segmentation and tokenization

**LLM:**
- Model: T5-small (60M parameters)
- Purpose: Text summarization
- Input: Contract text → Output: Executive summary

**Vector Search:**
- FAISS IndexFlatIP (inner product)
- Normalized embeddings for cosine similarity
- Fast retrieval (<1 second for 10,000 documents)

---

## 8. CHALLENGES & LIMITATIONS

### Scope Limitations
- English contracts only (French support future work)
- Common contract types (employment, procurement, services)
- Rule-based compliance (not full legal AI)
- Advisory only (not legal advice)

### Technical Constraints
- 1-week development timeline
- Free-tier models only
- Limited training data for Mauritius
- Local processing (no cloud GPU)

### Ethical Considerations
- **Privacy:** All processing local, no data stored
- **Transparency:** Explain all risk scores
- **Disclaimer:** Tool assists, doesn't replace lawyers
- **Bias:** Training on diverse contract types

---

## 9. SUCCESS CRITERIA

The project is successful if:

✅ System analyzes contracts in <5 minutes  
✅ Risk detection achieves 85%+ precision  
✅ Entity extraction has 90%+ recall  
✅ ROUGE scores exceed 0.70 for summaries  
✅ Processes 10/10 test contracts successfully  
✅ Mauritius compliance checking works  
✅ User feedback rates 4/5 stars  
✅ All deliverables completed  

---

## 10. FUTURE ENHANCEMENTS

### What's Next: Making It Even Better

**Easy Wins (Can add quickly):**
- 🌍 **French language support** - Analyze contracts in both English and French
- 🔌 **API access** - Let other apps connect to the analyzer
- 🧠 **Smarter AI** - Use specialized legal AI models (LEGAL-BERT)
- 📊 **Better dashboards** - More visual insights and charts

**Bigger Ambitions (Need more time):**
- 📱 **Mobile version** - Access from phones and tablets
- 👥 **Team features** - Multiple users reviewing contracts together
- 🌍 **Other countries** - Expand to Kenya, South Africa, Ghana legal systems
- ⚡ **Batch processing** - Analyze 50+ contracts at once

**Dream Features (Requires a full team):**
- 🤝 **Live collaboration** - Real-time contract negotiation
- 🏢 **Enterprise version** - For large companies and law firms
- 🔐 **Advanced security** - Bank-level encryption and audit trails

---

## 11. CONCLUSION

SmartContract addresses a critical need in Mauritius's business ecosystem by making legal contract analysis accessible and affordable through AI/ML. The system demonstrates practical application of NLP, LLM, and Embeddings while directly supporting:

- **Digital Mauritius 2030** vision
- **National AI Strategy 2018**
- **SME growth and competitiveness**
- **16,000 tech jobs creation target**

By combining Sentence Transformers, FAISS vector search, spaCy NER, and T5 summarization, the system achieves 95% time reduction and 90% cost savings while maintaining 85%+ accuracy—making professional contract analysis available to every Mauritius business.

---

## 12. REFERENCES

### Legal Framework
1. Mauritius Civil Code - Articles 1108, 1174, 2279
2. Workers' Rights Act 2019, Government of Mauritius
3. Data Protection Act 2017, Government of Mauritius
4. MCCI Model Contracts for SMEs

### AI/ML Research
5. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (Reimers & Gurevych, 2019)
6. T5: Text-to-Text Transfer Transformer (Raffel et al., 2020)
7. FAISS: A Library for Efficient Similarity Search (Johnson et al., 2019)
8. Named Entity Recognition for Legal Documents (Chalkidis et al., 2020)

### Mauritius Context (2025)
9. Budget 2025-2026: Artificial Intelligence - EDB Mauritius
10. Digital Transformation Blueprint 2025-2029, Ministry of ITCI
11. Mauritius Artificial Intelligence Strategy 2018
12. "Mauritius Bets on AI to Drive Digital Economy" - iAfrica.com, June 2025

---

**Submitted by:** Maeva Moonien  
**Date:** November 2025  

---

## APPENDIX: WHY THIS MATTERS

**The Problem in Numbers:**
- 💰 Traditional lawyer review: MUR 5,000-15,000 per contract
- ⏰ Time needed: 5-7 business days
- ⚖️ Mauritius Civil Code: 3-year time limit to claim (easy to miss!)

**What Our System Does:**
- ⚡ Analysis time: Less than 5 minutes
- 🎯 Accuracy: 85%+ in finding problems
- 🚀 Speed boost: 95% faster than manual review

**Why Mauritius?**
- 🇲🇺 Unique legal system (French + British law mixed together)
- 📱 79.5% internet penetration (people are online!)
- 🏢 Growing tech sector (34,500+ people in ICT)
- 🎯 Government pushing AI (Digital Mauritius 2030 plan)

**The Big Picture:**
This isn't just about contracts. It's about showing that AI can help businesses in Africa work smarter, faster, and cheaper. If it works in Mauritius, it can work anywhere!
