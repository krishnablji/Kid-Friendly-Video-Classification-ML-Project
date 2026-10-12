"""
generate_guide_pdf.py
Converts Stage3_Architecture_and_Transformer_Guide.md into a professional,
multi-page PDF publication complete with headers, footers, tables, math formulas,
and the beginner breakdown callout blocks.
"""

import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print 'Page X of Y'."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#475569"))

        # Header
        self.drawString(36, 762, "KID-SAFE AI PIPELINE | TECHNICAL ARCHITECTURE GUIDE: STAGE 3")
        self.drawRightString(576, 762, "LOCAL TRANSFORMER SEQUENCE CLASSIFIER")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(36, 756, 576, 756)

        # Footer
        self.line(36, 36, 576, 36)
        self.setFont("Helvetica", 7.5)
        self.drawString(36, 24, "CONFIDENTIAL & PROPRIETARY — ADVANCED NLP ENGINEERING DOCUMENTATION")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(576, 24, page_str)
        self.restoreState()


def build_pdf(filename="Stage3_Architecture_and_Transformer_Guide.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=46,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2563EB"),
        spaceAfter=8,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=8,
        spaceAfter=4,
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.8,
        leading=11.5,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=5,
        spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.4,
        textColor=colors.HexColor("#334155"),
        spaceAfter=4,
    )
    callout_box_style = ParagraphStyle(
        "CalloutBoxText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.6,
        leading=10.2,
        textColor=colors.HexColor("#0F172A"),
    )
    callout_title_style = ParagraphStyle(
        "CalloutTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#B45309"),
        spaceAfter=3,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#334155"),
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.4,
        leading=9.4,
        textColor=colors.HexColor("#0F172A"),
    )

    def create_callout(title, text_paragraphs):
        story_content = [Paragraph(f"<b>{title}</b>", callout_title_style)]
        for p in text_paragraphs:
            story_content.append(Paragraph(p, callout_box_style))
            story_content.append(Spacer(1, 2))
        
        box_table = Table([[story_content]], colWidths=[540])
        box_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFBEB")),
            ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#FCD34D")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        return box_table

    story = []

    # Title Banner
    story.append(Paragraph("KID-SAFE AI PIPELINE: TECHNICAL ARCHITECTURE GUIDE", title_style))
    story.append(Paragraph("Stage 3 Local Transformer Classifier: Foundations, DeBERTa-v3 Mechanics, and Enterprise Alignment", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0"), spaceAfter=6))

    # =========================================================================
    # SECTION 1
    # =========================================================================
    story.append(Paragraph("1. The Architectural Paradigm Shift (What Changed in Stage 3 & Why)", h1_style))
    story.append(Paragraph(
        "In production AI safety pipelines, the content moderation classifier acts as the deterministic <b>'Judge'</b> "
        "evaluating visual scene transcripts. The previous architecture utilized a classical <b>TF-IDF (Term Frequency–Inverse "
        "Document Frequency) + Logistic Regression</b> pipeline. While computationally lightweight, classical Bag-of-Words (BoW) models "
        "suffer from severe mathematical and conceptual limitations that modern NLP completely rejects:",
        body_style
    ))

    comp_data = [
        [
            Paragraph("<b>Dimension</b>", table_header),
            Paragraph("<b>Legacy Classical Architecture (TF-IDF + Linear Model)</b>", table_header),
            Paragraph("<b>Modern Stage 3 Architecture (Transformer Backbone)</b>", table_header),
        ],
        [
            Paragraph("<b>Mathematical Representation</b>", table_cell),
            Paragraph("Sparse token count coordinates in R^25,000. Words treated as orthogonal, independent axes.", table_cell),
            Paragraph("Dense contextual continuous vectors in R^768 produced by multi-head attention.", table_cell),
        ],
        [
            Paragraph("<b>Context & Syntax Awareness</b>", table_cell),
            Paragraph("Zero syntax comprehension. Ignores token ordering, clause hierarchies, and dependencies.", table_cell),
            Paragraph("Bidirectional self-attention dynamically encodes who does what to whom in every sentence.", table_cell),
        ],
        [
            Paragraph("<b>Negation Resolution</b>", table_cell),
            Paragraph("Catastrophic failure: <i>'no weapon present'</i> yields identical high hazard weight to <i>'weapon present'</i>.", table_cell),
            Paragraph("Attention heads bind negation modifiers (<i>'no'</i>, <i>'without'</i>) directly to target nouns.", table_cell),
        ],
        [
            Paragraph("<b>Spurious Correlation & Bias</b>", table_cell),
            Paragraph("Memorizes isolated keywords (e.g., assigning danger weights to neutral demographic nouns like <i>'man'</i>).", table_cell),
            Paragraph("Eliminates keyword memorization; decision boundaries govern verb-object action triples.", table_cell),
        ],
    ]
    t_comp = Table(comp_data, colWidths=[95, 220, 225])
    t_comp.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 4))

    # Breakdown 1
    story.append(create_callout("💡 Beginner's Breakdown: Understanding Section 1 in Plain English", [
        "<b>• What is TF-IDF (Bag-of-Words)?</b> Imagine throwing a book into a blender. You get a huge bag of individual words. TF-IDF counts how often words appear in a text compared to a whole library. <i>The Problem:</i> It loses all word order, grammar, and context. To a TF-IDF model, <i>'A dog bites a man'</i> and <i>'A man bites a dog'</i> look identical because they contain the exact same bag of words.",
        "<b>• Sparse Vectors (R^25,000) vs. Dense Vectors (R^768):</b> Sparse vectors are a massive grid of 25,000 numbers where 24,990 of them are zeros. The model thinks <i>'king'</i> and <i>'queen'</i> are as unrelated as <i>'king'</i> and <i>'banana'</i>. Dense vectors are 768 continuous decimal numbers where every number encodes a property of meaning (e.g., action intensity, safety hazard).",
        "<b>• Negation Resolution:</b> TF-IDF sees <i>'weapon'</i> in <i>'There is no weapon'</i> and immediately flags it as dangerous. A Transformer reads the sentence as a whole and understands that <i>'no'</i> cancels out the hazard.",
        "<b>• Spurious Correlations (Dataset Bias):</b> If action movie scenes happen to feature men, a classical TF-IDF model blindly memorizes: <i>man = UNSAFE</i>. A Transformer looks at the entire action phrase (<i>'a man reading a storybook to kids'</i>) and correctly classifies it as <b>SAFE</b> regardless of gender words."
    ]))
    story.append(Spacer(1, 6))

    # =========================================================================
    # SECTION 2
    # =========================================================================
    story.append(Paragraph("2. Deep Dive: What is BERT and What Does It Contain?", h1_style))
    story.append(Paragraph(
        "<b>BERT (Bidirectional Encoder Representations from Transformers)</b>, introduced by Devlin et al. (Google AI, 2018), "
        "fundamentally redefined natural language processing by replacing recurrent neural networks (RNNs/LSTMs) and word n-grams "
        "with a stack of <b>Transformer Encoders</b>. Unlike autoregressive language models (such as GPT) that read text left-to-right "
        "with causal masking, BERT is an <b>Encoder-Only</b> network that conditions on both left and right context simultaneously across all layers.",
        body_style
    ))

    story.append(Paragraph("<b>A. Subword Tokenization & Multi-Component Input Embeddings:</b> Raw input sentences are tokenized using subword tokenization (WordPiece/BPE) into token indices. For every token, BERT sums three distinct embedding vectors:<br/>"
                           "• <b>Token Embeddings:</b> Maps token IDs into a continuous vector space R^d_model (d=768).<br/>"
                           "• <b>Positional Embeddings:</b> Learned or sinusoidal vectors that inject sequence order into permutation-invariant attention layers.<br/>"
                           "• <b>Segment Embeddings:</b> Distinguishes between paired sentences in multi-sequence tasks.", body_style))

    story.append(Paragraph("<b>B. Multi-Head Scaled Dot-Product Self-Attention (The Core Engine):</b> Self-attention allows every token in an input sequence to dynamically assign relevance weights to every other token. For an input matrix of vectors H, the layer projects linear transformations into Query (Q), Key (K), and Value (V) matrices:<br/>"
                           "&nbsp;&nbsp;&nbsp;&nbsp;<b>Attention(Q, K, V) = softmax( (Q · K^T) / √d_k ) · V</b><br/>"
                           "Dividing by √d_k prevents dot products from growing excessively large in high dimensions (which would cause vanishing gradients in softmax). <b>Multi-Head Attention</b> runs this operation across h independent projection subspaces (h=12 in base models), allowing the network to simultaneously attend to syntactic relationships, semantic dependencies, and safety indicators.", body_style))

    story.append(Paragraph("<b>C. Position-Wise Feed-Forward Networks & Residual Normalization:</b> Each encoder block contains two sub-layers: (1) Multi-Head Self-Attention, and (2) a Two-Layer Feed-Forward Network: <b>FFN(x) = GELU(x · W_1 + b_1) · W_2 + b_2</b>, projecting from 768 to 3072 dimensions and back. Crucially, each sub-layer is wrapped in a <b>Residual Connection</b> followed by <b>Layer Normalization</b>: <i>Output = LayerNorm(x + SubLayer(x))</i>. Residuals preserve original signal flow and prevent gradient degradation.", body_style))

    story.append(Paragraph("<b>D. Pretraining Paradigms (MLM and NSP):</b> BERT derives its linguistic prowess from unsupervised pretraining over billions of web tokens via: (1) <b>Masked Language Modeling (MLM)</b>: 15% of tokens are randomly masked ([MASK]) and the model reconstructs them using bidirectional context; and (2) <b>Next Sentence Prediction (NSP)</b>: Evaluates whether sentence pairs are logically sequential.", body_style))
    story.append(Spacer(1, 4))

    # Breakdown 2
    story.append(create_callout("💡 Beginner's Breakdown: Understanding Section 2 in Plain English", [
        "<b>• Bidirectional vs. Left-to-Right:</b> GPT reads only left-to-right to guess the next word. BERT reads both directions at the exact same time to understand the whole sentence's meaning. It is an <b>Encoder-Only</b> model, meaning it reads and understands rather than writing new text.",
        "<b>• The Attention Formula Analogy (Q, K, V):</b> Think of a Database Search: <b>Query (Q):</b> What a word is searching for (e.g., <i>'attacking'</i> asks: <i>'Who is being attacked?'</i>). <b>Key (K):</b> The label of every other word (e.g., <i>'monster'</i> says: <i>'I am the actor'</i>). <b>Value (V):</b> The actual meaning carried by each word. <b>Softmax & Scaling (√d_k):</b> Calculates a percentage score (0% to 100%) showing how much attention word A should pay to word B.",
        "<b>• Residual Highway Bypass:</b> Deep networks can forget early details across 12+ layers. A residual connection takes the original input x and adds it directly to the output (x + SubLayer(x)), ensuring the original meaning is never lost."
    ]))
    story.append(Spacer(1, 6))

    # =========================================================================
    # SECTION 3
    # =========================================================================
    story.append(Paragraph("3. The DeBERTa-v3 Upgrade: Beyond Standard BERT", h1_style))
    story.append(Paragraph(
        "In Stage 3 of our project, we upgrade standard BERT to <b>Microsoft DeBERTa-v3 (Decoding-enhanced BERT with Disentangled Attention)</b>. "
        "DeBERTa-v3 represents the state-of-the-art in encoder backbones, outperforming RoBERTa and BERT across GLUE/SuperGLUE benchmarks through three innovations:",
        body_style
    ))

    story.append(Paragraph("<b>1. Disentangled Attention Mechanism:</b> Standard BERT sums word content and absolute position into a single vector before attention. "
                           "DeBERTa represents each word using <i>two separate vectors</i>: content <b>c_i</b> and relative position <b>p_(i|j)</b>. Its cross-attention "
                           "matrix explicitly calculates 4 disentangled interaction components: (Content-to-Content) + (Content-to-Position) + (Position-to-Content) + (Position-to-Position). "
                           "This means the model understands that a verb like <i>'strikes'</i> relates differently to a subject appearing 2 tokens before versus 10 tokens after.", body_style))
    story.append(Paragraph("<b>2. Enhanced Masked Decoder (EMD):</b> Injects absolute position information directly prior to the sequence classification head, providing sequence-level grammatical orientation.", body_style))
    story.append(Paragraph("<b>3. ELECTRA-Style Replaced Token Detection (RTD):</b> Unlike BERT's MLM which trains on only 15% masked words, DeBERTa-v3 employs a discriminator network that predicts whether <i>every token</i> was replaced by a generator, boosting pretraining sample efficiency by over 300%.", body_style))
    story.append(Spacer(1, 4))

    # Breakdown 3
    story.append(create_callout("💡 Beginner's Breakdown: Understanding Section 3 in Plain English", [
        "<b>• Disentangled Attention (Content vs. Position):</b> In BERT, meaning and position are crushed together into one vector. In DeBERTa-v3, meaning (c_i) and relative distance (p_i|j) are kept in separate mathematical channels. This allows DeBERTa to natively recognize that <i>'boy hits ball'</i> and <i>'ball hits boy'</i> have the exact same words, but the distance and direction between <i>'hits'</i> and <i>'boy'</i> completely flips who is in danger.",
        "<b>• Replaced Token Detection (RTD):</b> Standard BERT plays fill-in-the-blank on only 15% of words (<i>'The [MASK] sat on the mat'</i>). DeBERTa-v3 uses a secondary AI to swap words and forces the model to inspect <b>every single word</b> in the sentence (<i>'The cat ate on the mat'</i> &rarr; flags <i>'ate'</i> as fake). This makes it learn 3x faster and significantly more accurately."
    ]))
    story.append(Spacer(1, 6))

    # =========================================================================
    # SECTION 4
    # =========================================================================
    story.append(Paragraph("4. Granular Technical Terminology Reference for Stage 3", h1_style))

    terms_data = [
        [Paragraph("<b>Technical Term</b>", table_header), Paragraph("<b>Exact Architectural Definition & Role in Stage 3</b>", table_header)],
        [
            Paragraph("<b>AutoModelForSequenceClassification</b>", table_cell),
            Paragraph("Hugging Face abstraction wrapping the pretrained transformer backbone and appending an initialized linear pooling head: LayerNorm &rarr; Dropout(p=0.1) &rarr; Linear(768, 2).", table_cell)
        ],
        [
            Paragraph("<b>[CLS] Pooling Token</b>", table_cell),
            Paragraph("Special classification token prepended to input sequence index 0. Its final-layer hidden state serves as the sequence-level semantic vector summary.", table_cell)
        ],
        [
            Paragraph("<b>Cross-Entropy Loss</b>", table_cell),
            Paragraph("Log-loss objective: L = -sum(y_i * log(p_i)). In Stage 3, balanced class weights are injected to penalize false negatives on hazardous content.", table_cell)
        ],
        [
            Paragraph("<b>AdamW Optimizer</b>", table_cell),
            Paragraph("Adaptive Moment Estimation with Decoupled Weight Decay (Loshchilov & Hutter). Applies L2 penalty directly to parameters rather than gradient moments, preventing catastrophic forgetting.", table_cell)
        ],
        [
            Paragraph("<b>Calibrated Decision Engine (tau = 0.35)</b>", table_cell),
            Paragraph("Asymmetric risk boundary: If P(UNSAFE) >= 0.35 &rarr; UNSAFE. Essential in child safety systems where the cost of false negatives (missed hazards) greatly outweighs false positives.", table_cell)
        ],
    ]
    t_terms = Table(terms_data, colWidths=[140, 400])
    t_terms.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_terms)
    story.append(Spacer(1, 4))

    # Breakdown 4
    story.append(create_callout("💡 Beginner's Breakdown: Understanding Section 4 in Plain English", [
        "<b>• AutoModelForSequenceClassification:</b> A pre-packaged PyTorch class by Hugging Face that attaches a 2-output decision layer (SAFE vs. UNSAFE) right on top of DeBERTa.",
        "<b>• [CLS] Token:</b> A special dummy token added at the very beginning of the sentence (Index 0). As attention layers calculate, the [CLS] token acts as a <b>'sponge'</b> that absorbs the summary of the entire sentence.",
        "<b>• Cross-Entropy Loss:</b> The mathematical grade book used during training. If the model predicts 90% Safe on a scene that was actually Unsafe, Cross-Entropy heavily penalizes the model's weights so it corrects itself.",
        "<b>• AdamW Optimizer:</b> The weight updater that stops the model from memorizing exact training sentences (overfitting).",
        "<b>• Calibrated Threshold (tau=0.35):</b> Normal ML models split decisions at 50% (0.50). In child safety, we lower the risk line to 35% (0.35). If the model is even 35% suspicious of fighting or injury, it flags UNSAFE. It is far better to block a safe video by mistake than to let violence reach a child."
    ]))
    story.append(Spacer(1, 6))

    # =========================================================================
    # SECTION 5
    # =========================================================================
    story.append(Paragraph("5. Enterprise Industry Standards vs. Stage 3 Alignment", h1_style))
    story.append(Paragraph(
        "Leading tech enterprises (Google Trust & Safety, Meta Content Moderation, OpenAI Safety Systems, Stripe Radar) handle "
        "high-scale content filtering using a well-defined architectural pattern. <b>Why don't production systems use large LLM prompts for classification?</b><br/>"
        "• <b>Latency:</b> Cloud LLM inference takes 800ms–3000ms per request. Specialized fine-tuned encoders execute in 5ms–25ms on local accelerators.<br/>"
        "• <b>Compute & Marginal Cost:</b> Generative autoregressive decoding scales per-token costs linearly. Discriminative sequence heads require negligible compute.<br/>"
        "• <b>Hallucination & Determinism:</b> LLMs hallucinate rules and exhibit stochastic drift. Encoder sequence classifiers output calibrated, deterministic softmax logits.<br/>"
        "• <b>Explainability & Safety Bounds:</b> Encoders allow fine-tuning exact probability thresholds (e.g., tau=0.35) aligned with enterprise compliance policies.",
        body_style
    ))

    ind_data = [
        [
            Paragraph("<b>Architecture Attribute</b>", table_header),
            Paragraph("<b>Production Enterprise Standard (Google/Meta/Stripe)</b>", table_header),
            Paragraph("<b>Our Kid-Safe Project Stage 3 Implementation</b>", table_header),
        ],
        [
            Paragraph("<b>Perception Engine (Stage 2)</b>", table_cell),
            Paragraph("Multimodal Foundation Models (Gemini Vision / Video-LLaVA) extracting objective scene narrative transcripts.", table_cell),
            Paragraph("<b>Identical:</b> Google Gemini API deployed strictly as the objective perception narrator ('Eyes').", table_cell),
        ],
        [
            Paragraph("<b>Classifier Execution (Stage 3)</b>", table_cell),
            Paragraph("100% Offline, on-premise local discriminative Transformer (DeBERTa / RoBERTa / DistilBERT) with zero external API calls.", table_cell),
            Paragraph("<b>Identical:</b> Fine-tuned local `microsoft/deberta-v3-base` via PyTorch executing 100% offline ('Judge').", table_cell),
        ],
        [
            Paragraph("<b>Classification Head</b>", table_cell),
            Paragraph("Sequence classification head with Dropout (0.1) and softmax over dense [CLS] pooled contextual embeddings.", table_cell),
            Paragraph("<b>Identical:</b> Standard Hugging Face Sequence Head with CrossEntropyLoss and AdamW optimization.", table_cell),
        ],
        [
            Paragraph("<b>Decision Boundary</b>", table_cell),
            Paragraph("Asymmetric calibrated probability thresholds tuned for maximum safety recall (Safety-Critical Risk Tuning).", table_cell),
            Paragraph("<b>Identical:</b> Calibrated decision engine triggering UNSAFE at P(UNSAFE) >= 0.35 to prevent child exposure.", table_cell),
        ],
    ]
    t_ind = Table(ind_data, colWidths=[100, 220, 220])
    t_ind.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_ind)
    story.append(Spacer(1, 4))

    # Breakdown 5
    story.append(create_callout("💡 Beginner's Breakdown: Understanding Section 5 in Plain English", [
        "<b>• Why not just ask ChatGPT/Gemini to decide if every video is safe?</b><br/>"
        "1. <b>Speed (Latency):</b> Cloud LLM APIs take 1 to 3 seconds per sentence. Our local DeBERTa model takes 15 milliseconds.<br/>"
        "2. <b>Cost:</b> Cloud LLM APIs charge money per token for every video scanned. Our local model costs $0.00 after training because it runs entirely on your local machine.<br/>"
        "3. <b>Reliability & Consistency:</b> LLMs can change their answers depending on slight rephrasings or cloud API updates. Our local encoder outputs exact, reproducible probability scores.<br/>"
        "4. <b>Privacy & Offline Security:</b> Stage 2 ('Eyes') extracts the description once. Stage 3 ('Judge') runs completely offline without sending user data across the internet."
    ]))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully built: {filename}")


if __name__ == "__main__":
    build_pdf()
