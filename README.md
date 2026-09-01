# E-Commerce Experience Product Aspect Reinforcement & Benchmarking Model

An academic & practical framework for extracting valid **Aspects and Sub-aspects** from experience-based e-commerce products (product details + 1-to-5 star reviews), featuring a **Reinforcement Reward Alignment Model (RLAIF / DPO Candidate Optimization)** evaluated against **Llama 3.2 Baseline** and **Human Consensus Ground Truth**.

---

## Key Features

1. **Multi-Criteria Reward Engine ($R_{\text{total}}$)**:
   - $R_{\text{relevance}}$: Semantic alignment with review text & specs.
   - $R_{\text{groundedness}}$: Star sentiment coverage (1-star complaint triggers to 5-star praise).
   - $R_{\text{diversity}}$: Disincentivizes redundant/overlapping aspect predictions.
   - $R_{\text{validity}}$: Academic validation penalizing non-product generic fluff.

2. **Quantitative Benchmarking Suite**:
   - Compares predictions against Human Expert Annotations (Kartikey, Nakul, Anshul, Kushagra, Aditya -> Final Consensus).
   - Metrics: Aspect Precision, Recall, F1-Score, Invalid Aspect Rate, and Sentiment Grounding Ratio.

3. **Interactive Web Dashboard**:
   - Modern glassmorphism UI with side-by-side comparison tables, metric comparison cards, and pre-loaded test datasets (*MuscleBlaze Biozyme Whey Protein* & *MuscleBlaze Micronised Creatine Monohydrate*).

4. **Future Step Preview: MCQ Feedback Generator**:
   - Automatically synthesizes 5–10 MCQ feedback questions from validated sub-aspects so customers don't have to write long text reviews.

---

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Web Server**:
   ```bash
   python app.py
   ```

3. **Access Dashboard**:
   Open browser at `http://127.0.0.1:5000`
