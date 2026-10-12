"""
generate_guide_pdf.py
Generates an executive, highly technical 2-page engineering whitepaper/guide:
- Explaining the Stage 3 Architecture transition from Classical NLP to Modern Transformers.
- Deep dive into BERT (internal structure, attention mechanics, layers).
- Elaborating on every new technical term (DeBERTa-v3, Disentangled Attention, BPE, [CLS] Pooling, Cross-Entropy, AdamW, Calibrated Thresholding).
- Industry enterprise text classification practices and 1:1 architectural alignment.
- Exactly 2 pages.
"""

import os
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
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Header (pages 1 and 2)
        self.drawString(36, 762, "KID-SAFE AI PIPELINE | TECHNICAL ARCHITECTURE GUIDE: STAGE 3")
        self.drawRightString(576, 762, "LOCAL TRANSFORMER SEQUENCE CLASSIFIER")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(36, 756, 576, 756)

        # Footer
        self.line(36, 38, 576, 38)
        self.drawString(36, 26, "CONFIDENTIAL & PROPRIETARY — ADVANCED NLP ENGINEERING DOCUMENTATION")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(576, 26, page_str)
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

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=3,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2563EB"),
        spaceAfter=8,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=5,
        spaceAfter=4,
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=4,
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.6,
        leading=10.0,
        textColor=colors.HexColor("#334155"),
        spaceAfter=4,
    )
    code_inline = ParagraphStyle(
        "CodeInline",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#0F172A"),
    )
    callout_style = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.4,
        leading=9.6,
        textColor=colors.HexColor("#1E293B"),
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.0,
        leading=8.8,
        textColor=colors.HexColor("#334155"),
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9.0,
        textColor=colors.HexColor("#0F172A"),
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, PARADIGM SHIFT, AND DEEP DIVE INTO BERT
    # =========================================================================
    story.append(Paragraph("Technical Guide: Stage 3 Text Classifier & Modern NLP Architecture", title_style))
    story.append(Paragraph("From Classical TF-IDF to Fine-Tuned Bidirectional Transformers: Foundations, Mechanics, and Enterprise Alignment", subtitle_style))

    # SECTION 1: THE ARCHITECTURAL PARADIGM SHIFT
    story.append(Paragraph("1. The Architectural Paradigm Shift (What Changed in Stage 3 & Why)", h1_style))
    story.append(Paragraph(
        "In production AI safety pipelines, the content moderation classifier acts as the deterministic <b>'Judge'</b> "
        "evaluating visual scene transcripts. The previous architecture utilized a classical <b>TF-IDF (Term Frequency–Inverse "
        "Document Frequency) + Logistic Regression</b> pipeline. While computationally lightweight, classical Bag-of-Words (BoW) models "
        "suffer from severe mathematical and conceptual limitations that modern NLP completely rejects:",
        body_style
    ))

    # Comparison box
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
    t_comp = Table(comp_data, colWidths=[90, 220, 230])
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
    story.append(Spacer(1, 5))

    # SECTION 2: WHAT IS BERT & WHAT IS INSIDE IT?
    story.append(Paragraph("2. Deep Dive: What is BERT and What Does It Contain?", h1_style))
    story.append(Paragraph(
        "<b>BERT (Bidirectional Encoder Representations from Transformers)</b>, introduced by Devlin et al. (Google AI, 2018), "
        "fundamentally redefined natural language processing by replacing recurrent neural networks (RNNs/LSTMs) and word n-grams "
        "with a stack of <b>Transformer Encoders</b>. Unlike autoregressive language models (such as GPT) that read text left-to-right "
        "with causal masking, BERT is an <b>Encoder-Only</b> network that conditions on both left and right context simultaneously across all layers.",
        body_style
    ))

    # BERT Subsections
    story.append(Paragraph("<b>A. Subword Tokenization & Multi-Component Input Embeddings</b>", h2_style))
    story.append(Paragraph(
        "Raw input sentences are tokenized using subword tokenization (WordPiece/BPE) into token indices. For every token, BERT sums three distinct embedding vectors:<br/>"
        "• <b>Token Embeddings:</b> Maps token IDs into a continuous vector space R^d_model (d=768).<br/>"
        "• <b>Positional Embeddings:</b> Learned or sinusoidal vectors that inject sequence order into permutation-invariant attention layers.<br/>"
        "• <b>Segment Embeddings:</b> Distinguishes between paired sentences in multi-sequence tasks.",
        body_style
    ))

    story.append(Paragraph("<b>B. Multi-Head Scaled Dot-Product Self-Attention (The Core Engine)</b>", h2_style))
    story.append(Paragraph(
        "Self-attention allows every token in an input sequence to dynamically assign relevance weights to every other token. "
        "For an input matrix of vectors H, the layer projects linear transformations into Query (Q), Key (K), and Value (V) matrices:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Attention(Q, K, V) = softmax( (Q · K^T) / √d_k ) · V</b><br/>"
        "Dividing by √d_k prevents dot products from growing excessively large in high dimensions (which would cause vanishing gradients in softmax). "
        "<b>Multi-Head Attention</b> runs this operation across h independent projection subspaces (h=12 in base models), allowing the network to simultaneously "
        "attend to syntactic relationships (subject-verb agreement), semantic dependencies (coreference), and risk indicators across disparate spans.",
        body_style
    ))

    story.append(Paragraph("<b>C. Position-Wise Feed-Forward Networks & Residual Normalization</b>", h2_style))
    story.append(Paragraph(
        "Each encoder block contains two sub-layers: (1) Multi-Head Self-Attention, and (2) a Two-Layer Feed-Forward Network: "
        "<b>FFN(x) = GELU(x · W_1 + b_1) · W_2 + b_2</b>, projecting from 768 to 3072 dimensions and back. Crucially, each sub-layer is wrapped in a "
        "<b>Residual Connection</b> followed by <b>Layer Normalization</b>: <i>Output = LayerNorm(x + SubLayer(x))</i>. Residuals preserve original signal flow "
        "and prevent gradient degradation, enabling stable optimization across deep stacks.",
        body_style
    ))

    story.append(Paragraph("<b>D. Pretraining Paradigms: MLM and NSP</b>", h2_style))
    story.append(Paragraph(
        "BERT derives its foundational linguistic prowess from unsupervised pretraining over billions of web tokens via: (1) <b>Masked Language Modeling (MLM)</b>: "
        "15% of tokens are randomly masked ([MASK]) and the model learns to reconstruct them using bidirectional context; and (2) <b>Next Sentence Prediction (NSP)</b>: "
        "The model learns discourse-level relationships between sentence pairs.",
        body_style
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: DEBERTA-V3, TECHNICAL TERMINOLOGY, AND ENTERPRISE ALIGNMENT
    # =========================================================================
    story.append(Paragraph("3. The DeBERTa-v3 Upgrade: Beyond Standard BERT", h1_style))
    story.append(Paragraph(
        "In Stage 3 of our project, we upgrade standard BERT to <b>Microsoft DeBERTa-v3 (Decoding-enhanced BERT with Disentangled Attention)</b>. "
        "DeBERTa-v3 represents the state-of-the-art in encoder backbones, outperforming RoBERTa and BERT across GLUE/SuperGLUE benchmarks through three innovations:",
        body_style
    ))

    # DeBERTa Features
    story.append(Paragraph("<b>1. Disentangled Attention Mechanism:</b> Standard BERT sums word content and absolute position into a single vector before attention. "
                           "DeBERTa represents each word using <i>two separate vectors</i>: content <b>c_i</b> and relative position <b>p_(i|j)</b>. Its cross-attention "
                           "matrix explicitly calculates 4 disentangled interaction components: (Content-to-Content) + (Content-to-Position) + (Position-to-Content) + (Position-to-Position). "
                           "This means the model understands that a verb like <i>'strikes'</i> relates differently to a subject appearing 2 tokens before versus 10 tokens after.", body_style))
    story.append(Paragraph("<b>2. Enhanced Masked Decoder (EMD):</b> Solves relative attention's lack of absolute syntax position by incorporating absolute position embeddings "
                           "directly prior to the classification head.", body_style))
    story.append(Paragraph("<b>3. ELECTRA-Style Replaced Token Detection (RTD):</b> Unlike BERT's MLM which trains on only the 15% masked words, DeBERTa-v3 employs a discriminator "
                           "network that predicts whether <i>every token</i> was replaced by a small generator, boosting pretraining sample efficiency by over 300%.", body_style))

    story.append(Spacer(1, 3))
    story.append(Paragraph("4. Granular Technical Terminology Reference for Stage 3", h1_style))

    # Terms Table
    terms_data = [
        [Paragraph("<b>Technical Term</b>", table_header), Paragraph("<b>Exact Architectural Definition & Role in Stage 3</b>", table_header)],
        [
            Paragraph("<b>AutoModelForSequenceClassification</b>", table_cell),
            Paragraph("Hugging Face abstraction wrapping the pretrained transformer backbone and appending an initialized linear pooling head: LayerNorm -> Dropout(p=0.1) -> Linear(768, 2).", table_cell)
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
            Paragraph("Asymmetric risk boundary: If P(UNSAFE) >= 0.35 -> UNSAFE. Essential in child safety systems where the cost of false negatives (missed hazards) greatly outweighs false positives.", table_cell)
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

    # SECTION 5: ENTERPRISE STANDARD PRACTICES & 1:1 ALIGNMENT
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

    # Industry comparison table
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

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully built: {filename}")


if __name__ == "__main__":
    build_pdf()
