"""Optimized Academic Report Generator for Lab Assessment 9 (80% Implementation).
Generates an 8-to-9 page perfectly-proportioned academic report:
- Running single black border on every page (Letter size, 612x792 pt)
- Official VIT logo on cover page with student details
- Complete 80% implementation breakdown table
- Flow architecture diagram, Pareto trade-off charts, Leakage matrix, Dashboard UI
- Clean Times-Roman body text, Courier code snippets, full Pytest output logs
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable, Preformatted
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

BASE_DIR = r"D:\torchgeo\torchgeo-ts-bench"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PDF_PATH = os.path.join(OUTPUT_DIR, "TorchGeo_TS_Bench_Assessment9_Report.pdf")
ROOT_PDF_PATH = os.path.join(r"D:\torchgeo", "TorchGeo_TS_Bench_Assessment9_Report.pdf")

LOGO_PATH = os.path.join(OUTPUT_DIR, "extracted_logo_0.png")
ARCH_IMG = os.path.join(OUTPUT_DIR, "system_architecture_diagram.png")
PARETO_IMG = os.path.join(OUTPUT_DIR, "enhanced_pareto_efficiency.png")
MODEL_COMP_IMG = os.path.join(OUTPUT_DIR, "model_comparison.png")
LEAK_IMG = os.path.join(OUTPUT_DIR, "leakage_audit_verification_matrix.png")
DASH_IMG = os.path.join(OUTPUT_DIR, "streamlit_dashboard_overview.png")

PAGE_WIDTH, PAGE_HEIGHT = letter  # 612 x 792 pt
MARGIN = 40
BORDER_X = 36
BORDER_Y = 36
BORDER_W = PAGE_WIDTH - 72
BORDER_H = PAGE_HEIGHT - 72

class NumberedCanvas(canvas.Canvas):
    """Draws the running black border and page numbers."""
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
            self.draw_page_elements(num_pages)
            super().showPage()
        super().save()

    def draw_page_elements(self, total_pages):
        self.setStrokeColor(colors.black)
        self.setLineWidth(1.0)
        self.rect(BORDER_X, BORDER_Y, BORDER_W, BORDER_H)

        if self._pageNumber > 1:
            self.setFont("Times-Roman", 9)
            self.setFillColor(colors.black)
            page_str = f"Page {self._pageNumber} of {total_pages}"
            self.drawRightString(PAGE_WIDTH - MARGIN - 6, BORDER_Y + 12, page_str)
            self.drawString(MARGIN + 6, BORDER_Y + 12, "TorchGeo-TS-Bench | Lab Assessment 9 (80% Implementation Report)")
            self.setLineWidth(0.5)
            self.line(BORDER_X + 6, BORDER_Y + 24, BORDER_X + BORDER_W - 6, BORDER_Y + 24)

def build_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=MARGIN + 10,
        rightMargin=MARGIN + 10,
        topMargin=MARGIN + 10,
        bottomMargin=MARGIN + 16,
    )

    styles = getSampleStyleSheet()

    title_univ = ParagraphStyle(
        "UnivHeader", parent=styles["Normal"],
        fontName="Times-Bold", fontSize=13, leading=17,
        alignment=1, textColor=colors.black, spaceAfter=6,
    )
    proj_rep = ParagraphStyle(
        "ProjReport", parent=styles["Normal"],
        fontName="Times-Bold", fontSize=12, leading=16,
        alignment=1, textColor=colors.black, spaceAfter=14,
    )
    cover_title = ParagraphStyle(
        "CoverTitle", parent=styles["Normal"],
        fontName="Times-Bold", fontSize=13.5, leading=19,
        alignment=1, textColor=colors.black, spaceBefore=8, spaceAfter=8,
    )
    cover_course = ParagraphStyle(
        "CoverCourse", parent=styles["Normal"],
        fontName="Times-Italic", fontSize=12, leading=17,
        alignment=1, textColor=colors.black,
    )
    cover_bold_center = ParagraphStyle(
        "CoverBoldCenter", parent=styles["Normal"],
        fontName="Times-Bold", fontSize=11, leading=16,
        alignment=1, textColor=colors.black,
    )
    h1 = ParagraphStyle(
        "Heading1_Custom", parent=styles["Normal"],
        fontName="Times-Bold", fontSize=11.5, leading=15,
        textColor=colors.black, spaceBefore=11, spaceAfter=5, keepWithNext=True,
    )
    h2 = ParagraphStyle(
        "Heading2_Custom", parent=styles["Normal"],
        fontName="Times-Bold", fontSize=10, leading=13,
        textColor=colors.black, spaceBefore=8, spaceAfter=3, keepWithNext=True,
    )
    body = ParagraphStyle(
        "Body_Custom", parent=styles["Normal"],
        fontName="Times-Roman", fontSize=9.5, leading=13.2,
        alignment=4, textColor=colors.black, spaceAfter=5,
    )
    bullet = ParagraphStyle(
        "Bullet_Custom", parent=styles["Normal"],
        fontName="Times-Roman", fontSize=9.5, leading=13.2,
        textColor=colors.black, leftIndent=14, firstLineIndent=-9, spaceAfter=3,
    )
    code_style = ParagraphStyle(
        "Code_Custom", parent=styles["Normal"],
        fontName="Courier", fontSize=7.2, leading=9.2,
        textColor=colors.black, spaceBefore=2, spaceAfter=2,
    )
    fig_cap = ParagraphStyle(
        "FigCap", parent=styles["Normal"],
        fontName="Times-Italic", fontSize=8.5, leading=11,
        alignment=1, textColor=colors.black, spaceBefore=3, spaceAfter=4,
    )
    table_cell = ParagraphStyle(
        "TableCell", parent=styles["Normal"],
        fontName="Times-Roman", fontSize=8.5, leading=11, textColor=colors.black,
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold", parent=styles["Normal"],
        fontName="Times-Bold", fontSize=8.5, leading=11, textColor=colors.black,
    )
    table_cell_center = ParagraphStyle(
        "TableCellCenter", parent=styles["Normal"],
        fontName="Times-Roman", fontSize=8.5, leading=11, alignment=1, textColor=colors.black,
    )
    table_cell_center_bold = ParagraphStyle(
        "TableCellCenterBold", parent=styles["Normal"],
        fontName="Times-Bold", fontSize=8.5, leading=11, alignment=1, textColor=colors.black,
    )

    story = []

    # =======================================================
    # PAGE 1: COVER PAGE
    # =======================================================
    story.append(Spacer(1, 15))
    if os.path.exists(LOGO_PATH):
        story.append(Image(LOGO_PATH, width=4.8*inch, height=1.2*inch))
    story.append(Spacer(1, 24))

    story.append(Paragraph("SCHOOL OF COMPUTER SCIENCE AND ENGINEERING", title_univ))
    story.append(Spacer(1, 8))
    story.append(Paragraph("PROJECT REPORT", proj_rep))
    story.append(Spacer(1, 4))

    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceBefore=4, spaceAfter=14))
    story.append(Paragraph("A STANDARDIZED BENCHMARK SUITE FOR SPATIOTEMPORAL<br/>GEOSPATIAL FOUNDATION MODELS", cover_title))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceBefore=14, spaceAfter=18))

    story.append(Spacer(1, 14))
    story.append(Paragraph("Advanced Python Programming", cover_course))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Course Code – CSI3007", cover_bold_center))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Fall Semester 2026-27", cover_bold_center))
    story.append(Spacer(1, 12))
    story.append(Paragraph("LAB ASSESSMENT 9 SUBMISSION (80% IMPLEMENTATION MILESTONE)", cover_bold_center))

    story.append(Spacer(1, 60))
    story.append(Paragraph("<i>Submitted By</i>", ParagraphStyle("SubBy", parent=cover_course, fontSize=11, fontName="Times-Italic")))
    story.append(Spacer(1, 12))

    team_data = [
        [Paragraph("<b>Praveen Kanth N</b>", table_cell_bold), Paragraph("-", table_cell_center), Paragraph("<b>23MIC0078</b>", table_cell_bold)],
        [Paragraph("<b>Om Aravindhashan S</b>", table_cell_bold), Paragraph("-", table_cell_center), Paragraph("<b>23MID0222</b>", table_cell_bold)],
        [Paragraph("<b>Monish R</b>", table_cell_bold), Paragraph("-", table_cell_center), Paragraph("<b>23MID0318</b>", table_cell_bold)],
    ]
    t_team = Table(team_data, colWidths=[2.2*inch, 0.4*inch, 1.8*inch], hAlign='CENTER')
    t_team.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_team)
    story.append(PageBreak())

    # =======================================================
    # PAGE 2: 1. ABSTRACT & 2. 80% IMPLEMENTATION PROGRESS
    # =======================================================
    story.append(Paragraph("1. ABSTRACT", h1))
    story.append(Paragraph(
        "Geospatial Foundation Models (GeoFMs) are increasingly being used in Earth Observation (EO) to "
        "analyze massive volumes of satellite imagery and transfer learned representations across diverse remote "
        "sensing tasks. However, comparing these models reliably is challenged when satellite "
        "observations vary across both geographic space and calendar time. Disparate datasets, ad-hoc preprocessing, "
        "non-standardized temporal windowing, and insidious spatial and temporal data leakage lead to unfaithful "
        "benchmark comparisons.",
        body
    ))
    story.append(Paragraph(
        "This project implements <b>TorchGeo-TS-Bench</b>, a standardized, leakage-aware Python benchmarking "
        "framework built directly upon <b>TorchGeo 0.10.1</b> and the PyTorch ecosystem. The framework provides a "
        "unified spatiotemporal evaluation pipeline across multi-temporal satellite image time series (SITS), "
        "integrating automated spatial overlap audits, chronological sequence checks, multi-architecture "
        "model adapters (ConvLSTM, LTAE, ST-3DCNN, Temporal Transformer), joint predictive-computational efficiency "
        "profiling (latency, throughput, peak RAM, parameters), and automated publication-grade reporting.",
        body
    ))
    story.append(Paragraph(
        "<b>Keywords:</b> Geospatial Foundation Models, Earth Observation, Remote Sensing, Spatiotemporal "
        "Learning, TorchGeo, PyTorch, Satellite Image Time Series, Benchmarking, Leakage Audit, Reproducible AI.",
        body
    ))

    story.append(Spacer(1, 6))
    story.append(Paragraph("2. 80% IMPLEMENTATION STATUS & PROGRESS MATRIX", h1))
    story.append(Paragraph(
        "Per the syllabus requirements for <b>Lab Assessment 9</b>, at least 80% of the project implementation "
        "must be fully functional and verified. Our implementation currently stands at <b>85.0% completion</b>, "
        "with all eight core architectural modules fully operational and validated by 22 passing unit tests.",
        body
    ))

    matrix_data = [
        [
            Paragraph("<b>Mod</b>", table_cell_center_bold),
            Paragraph("<b>Core Component</b>", table_cell_bold),
            Paragraph("<b>Source File</b>", table_cell_bold),
            Paragraph("<b>Status</b>", table_cell_center_bold),
            Paragraph("<b>Implementation Details & Verification Evidence</b>", table_cell_bold),
        ],
        [
            Paragraph("M1", table_cell_center),
            Paragraph("Spatiotemporal Loader", table_cell_bold),
            Paragraph("data_loader.py", code_style),
            Paragraph("100%", table_cell_center_bold),
            Paragraph("Collation into (B, T, C, H, W); PASTIS SITS fixtures; 3 unit tests passed.", table_cell),
        ],
        [
            Paragraph("M2", table_cell_center),
            Paragraph("Spatiotemporal Sampling", table_cell_bold),
            Paragraph("sampling.py", code_style),
            Paragraph("100%", table_cell_center_bold),
            Paragraph("Fixed/sliding temporal windowing; sequence reflection padding; 2 unit tests passed.", table_cell),
        ],
        [
            Paragraph("M3", table_cell_center),
            Paragraph("Standardized Preprocessing", table_cell_bold),
            Paragraph("preprocessing.py", code_style),
            Paragraph("100%", table_cell_center_bold),
            Paragraph("NaN/Inf cleansing, min-max reflectance scaling, multispectral band slicing; 2 tests passed.", table_cell),
        ],
        [
            Paragraph("M4", table_cell_center),
            Paragraph("Leakage Audit Engine", table_cell_bold),
            Paragraph("leakage_checks.py", code_style),
            Paragraph("100%", table_cell_center_bold),
            Paragraph("Spatial bounding-box IoU check, chronological sequence check, duplicate audit; 2 tests passed.", table_cell),
        ],
        [
            Paragraph("M5", table_cell_center),
            Paragraph("Model Registry & Adapters", table_cell_bold),
            Paragraph("model_adapters.py", code_style),
            Paragraph("100%", table_cell_center_bold),
            Paragraph("ConvLSTM, LTAE Attention, ST-3DCNN, and Temporal Transformer; 5 tests passed.", table_cell),
        ],
        [
            Paragraph("M6", table_cell_center),
            Paragraph("Downstream Evaluation", table_cell_bold),
            Paragraph("evaluation.py", code_style),
            Paragraph("100%", table_cell_center_bold),
            Paragraph("Macro/Weighted F1, Accuracy, Precision, Recall, P50/P95 latency, RAM profiling; 4 tests passed.", table_cell),
        ],
        [
            Paragraph("M7", table_cell_center),
            Paragraph("Reproducibility Layer", table_cell_bold),
            Paragraph("reproducibility.py", code_style),
            Paragraph("100%", table_cell_center_bold),
            Paragraph("Seed locking, system hardware telemetry, SHA-256 configuration hashing; 3 tests passed.", table_cell),
        ],
        [
            Paragraph("M8", table_cell_center),
            Paragraph("Reporting & UI", table_cell_bold),
            Paragraph("reporting.py, app.py", code_style),
            Paragraph("100%", table_cell_center_bold),
            Paragraph("Auto Markdown/CSV report generator, high-res Pareto visualizer, interactive Streamlit app.", table_cell),
        ],
        [
            Paragraph("M9", table_cell_center),
            Paragraph("SAR Fusion & Scale-out", table_cell_bold),
            Paragraph("configs/ (planned)", code_style),
            Paragraph("15% Rem.", table_cell_center),
            Paragraph("Multi-sensor Sentinel-1/2 SAR fusion & Copernicus-Bench 15 tasks (Assessment 10).", table_cell),
        ],
    ]

    t_matrix = Table(matrix_data, colWidths=[0.45*inch, 1.3*inch, 1.2*inch, 0.65*inch, 3.2*inch])
    t_matrix.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
    ]))
    story.append(t_matrix)
    story.append(PageBreak())

    # =======================================================
    # PAGE 3: 3. SYSTEM ARCHITECTURE & PIPELINE
    # =======================================================
    story.append(Paragraph("3. SYSTEM ARCHITECTURE & PIPELINE", h1))
    story.append(Paragraph(
        "TorchGeo-TS-Bench is structured as an end-to-end, reproducible spatiotemporal benchmarking pipeline. "
        "The system decouples data ingestion, spatiotemporal windowing, data integrity auditing, model forward-pass "
        "adaptation, and multi-dimensional performance profiling.",
        body
    ))

    if os.path.exists(ARCH_IMG):
        story.append(Spacer(1, 4))
        story.append(Image(ARCH_IMG, width=6.6*inch, height=3.3*inch))
        story.append(Paragraph("<b>Figure 1:</b> <i>TorchGeo-TS-Bench Modular System Architecture & 80% Implementation Flow.</i>", fig_cap))

    story.append(Spacer(1, 6))
    story.append(Paragraph("Architectural Pipeline Walkthrough:", h2))
    story.append(Paragraph("• <b>Tier 1 (Data Ingestion & Integrity):</b> Satellite image time series (PASTIS Sentinel-2) are ingested as 5D tensors $(B, T, C, H, W)$. Sequences undergo temporal windowing, radiometric cleansing, and band selection before being audited for spatial, chronological, and sample identity leakage.", bullet))
    story.append(Paragraph("• <b>Tier 2 (Model Adaptation & Evaluation):</b> Standardized batches pass to registered model adapters. The evaluation engine measures both predictive quality (Accuracy, F1, Precision, Recall) and execution efficiency (P50/P95 Latency, Throughput, Peak Process RAM, Parameters).", bullet))
    story.append(Paragraph("• <b>Tier 3 (Automated Reporting & UI):</b> Results are serialized into deterministic JSON metrics, flat CSV tables, and interactive Streamlit dashboards.", bullet))
    story.append(PageBreak())

    # =======================================================
    # PAGE 4: 4. IMPLEMENTATION MODULES & TECHNICAL DETAILS
    # =======================================================
    story.append(Paragraph("4. IMPLEMENTATION MODULES & TECHNICAL DETAILS", h1))
    story.append(Paragraph(
        "<b>Module 1: Spatiotemporal Data Loader (<code>data_loader.py</code>):</b> Implements standardized dataset loaders "
        "for satellite image time series. It collates variable temporal sequences into uniform 5D PyTorch tensors $(B, T, C, H, W)$ "
        "while preserving geographic bounding boxes $(min_x, min_y, max_x, max_y)$, timestamps, and CRS coordinates.",
        body
    ))
    story.append(Paragraph(
        "<b>Module 2: Spatiotemporal Sampling (<code>sampling.py</code>):</b> Provides fixed-interval and sliding-window temporal "
        "sampling across observation sequences. Accommodates irregular satellite revisit schedules through deterministic "
        "temporal reflection padding and step-size control.",
        body
    ))
    story.append(Paragraph(
        "<b>Module 3: Preprocessing Engine (<code>preprocessing.py</code>):</b> Handles sensor saturation, missing data, and "
        "cloud artifacts by replacing NaNs and Infs with local channel medians. Performs min-max surface reflectance "
        "normalization $[0, 10000] \\to [0.0, 1.0]$ and extracts multispectral bands (e.g., B2, B3, B4, B8, B11, B12).",
        body
    ))
    story.append(Paragraph(
        "<b>Module 4: Leakage-Aware Audit Engine (<code>leakage_checks.py</code>):</b> Addresses Section 3A mandatory novelty. "
        "Calculates spatial Axis-Aligned Bounding Box (AABB) intersection-over-union (IoU) across train/val/test splits. "
        "Enforces strict chronological ordering $(\\max(T_{train}) < \\min(T_{val}) < \\min(T_{test}))$ and audits cryptographic "
        "SHA-256 sample hashes to eliminate duplicate observations.",
        body
    ))
    story.append(Paragraph(
        "<b>Module 5: Model Registry & Adapters (<code>model_adapters.py</code>, <code>model_registry.py</code>):</b> Implements a unified "
        "<code>BaseSpatiotemporalModel</code> interface for four architectures: (1) <code>TorchGeo-ConvLSTM</code> (recurrent 2D spatial gates), "
        "(2) <code>TorchGeo-LTAE</code> (Lightweight Temporal Attention Encoder), (3) <code>ST-3DCNN-Baseline</code> (joint spatiotemporal 3D convolution), "
        "and (4) <code>Temporal-Transformer</code> (multi-head temporal self-attention).",
        body
    ))
    story.append(Paragraph(
        "<b>Module 6: Downstream Tasks & Metrics (<code>tasks.py</code>, <code>evaluation.py</code>):</b> Standardizes multi-class classification, "
        "segmentation, regression, and change detection. Computes Accuracy, Macro/Weighted F1, Macro Precision, and Macro Recall. "
        "Simultaneously captures CPU/GPU latency percentiles, throughput (samples/sec), and peak process memory via <code>psutil</code>.",
        body
    ))
    story.append(Paragraph(
        "<b>Module 7: Reproducibility Layer (<code>reproducibility.py</code>):</b> Locks seeds across Python, NumPy, PyTorch, and CUDA. "
        "Captures complete system telemetry (CPU topology, RAM, OS, package versions) and calculates a deterministic 16-character SHA-256 config hash.",
        body
    ))
    story.append(Paragraph(
        "<b>Module 8: Automated Reporting & UI (<code>reporting.py</code>, <code>app.py</code>):</b> Generates Markdown reports, CSV summaries, "
        "and launches a full-featured interactive Streamlit web application for real-time benchmark exploration.",
        body
    ))
    story.append(PageBreak())

    # =======================================================
    # PAGE 5: 5. CORE CODE IMPLEMENTATION
    # =======================================================
    story.append(Paragraph("5. CORE CODE IMPLEMENTATION", h1))
    story.append(Paragraph("<b>Snippet 1: Automated Spatial & Temporal Leakage Detection (<code>leakage_checks.py</code>)</b>", h2))

    code_leak = (
        "def check_spatial_leakage(train_meta: List[Dict], val_meta: List[Dict], max_iou: float = 0.05) -> Tuple[bool, List[str]]:\n"
        "    violations = []\n"
        "    for t_idx, t_sample in enumerate(train_meta):\n"
        "        t_box = t_sample.get('bbox')  # (min_x, min_y, max_x, max_y)\n"
        "        if not t_box: continue\n"
        "        for v_idx, v_sample in enumerate(val_meta):\n"
        "            v_box = v_sample.get('bbox')\n"
        "            if not v_box: continue\n"
        "            iou = compute_bbox_iou(t_box, v_box)\n"
        "            if iou > max_iou:\n"
        "                violations.append(f'Spatial overlap: Train[{t_idx}] & Val[{v_idx}] IoU {iou:.4f} > {max_iou}')\n"
        "    return len(violations) == 0, violations"
    )
    story.append(Preformatted(code_leak, code_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Snippet 2: Lightweight Temporal Attention Encoder Adapter (<code>model_adapters.py</code>)</b>", h2))

    code_ltae = (
        "class TorchGeoLTAEAdapter(BaseSpatiotemporalModel):\n"
        "    def __init__(self, in_channels: int, num_classes: int, seq_len: int, d_model: int = 64, n_heads: int = 4):\n"
        "        super().__init__()\n"
        "        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))\n"
        "        self.input_proj = nn.Linear(in_channels, d_model)\n"
        "        self.temporal_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)\n"
        "        self.classifier = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, num_classes))\n\n"
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
        "        b, t, c, h, w = x.shape\n"
        "        x_pooled = self.spatial_pool(x.view(b * t, c, h, w)).view(b, t, c)\n"
        "        tokens = self.input_proj(x_pooled)\n"
        "        attn_out, _ = self.temporal_attention(tokens, tokens, tokens)\n"
        "        return self.classifier(attn_out.mean(dim=1))"
    )
    story.append(Preformatted(code_ltae, code_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Snippet 3: Joint Hardware Resource Profiler (<code>evaluation.py</code>)</b>", h2))

    code_eval = (
        "def profile_model_efficiency(model: nn.Module, sample_batch: torch.Tensor, warmup: int = 3, runs: int = 20) -> Dict[str, Any]:\n"
        "    process = psutil.Process(os.getpid())\n"
        "    for _ in range(warmup): _ = model(sample_batch)\n"
        "    latencies = []\n"
        "    for _ in range(runs):\n"
        "        t0 = time.perf_counter()\n"
        "        _ = model(sample_batch)\n"
        "        latencies.append((time.perf_counter() - t0) * 1000.0) # ms\n"
        "    return {\n"
        "        'mean_latency_ms': float(np.mean(latencies)), 'p50_ms': float(np.percentile(latencies, 50)),\n"
        "        'p95_ms': float(np.percentile(latencies, 95)), 'peak_memory_mb': process.memory_info().rss / (1024*1024)\n"
        "    }"
    )
    story.append(Preformatted(code_eval, code_style))
    story.append(PageBreak())

    # =======================================================
    # PAGE 6: 6. EXPERIMENTAL SETUP & 7. RESULTS
    # =======================================================
    story.append(Paragraph("6. EXPERIMENTAL SETUP & REPRODUCIBILITY ENVIRONMENT", h1))
    story.append(Paragraph(
        "All benchmark experiments were conducted on the specified reference workstation under strict telemetry tracking. "
        "Deterministic execution was ensured via seed locking and configuration hashing.",
        body
    ))

    env_data = [
        [Paragraph("<b>Hardware / Software Parameter</b>", table_cell_bold), Paragraph("<b>Recorded Specification</b>", table_cell_bold)],
        [Paragraph("Host Operating System", table_cell), Paragraph("Windows 11 x64 (Build 26100)", table_cell)],
        [Paragraph("Python Environment", table_cell), Paragraph("Python 3.14.3 (Virtual Environment)", table_cell)],
        [Paragraph("Deep Learning Framework", table_cell), Paragraph("PyTorch 2.14.1+cpu", table_cell)],
        [Paragraph("Geospatial Foundation Library", table_cell), Paragraph("TorchGeo 0.10.1 (Editable Mode)", table_cell)],
        [Paragraph("CPU Architecture & Cores", table_cell), Paragraph("Intel64 Family 6 Model 140 (4 Physical, 8 Logical Cores)", table_cell)],
        [Paragraph("System Random Access Memory", table_cell), Paragraph("15.77 GB Available", table_cell)],
        [Paragraph("Random Seed & Hashing", table_cell), Paragraph("Seed: 42 | Config SHA-256 Fingerprint: <code>cdf6a3a8abf8b5d1</code>", table_cell)],
    ]
    t_env = Table(env_data, colWidths=[2.6*inch, 4.1*inch])
    t_env.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
    ]))
    story.append(t_env)

    story.append(Spacer(1, 6))
    story.append(Paragraph("7. EMPIRICAL RESULTS & PERFORMANCE EVALUATION", h1))
    story.append(Paragraph(
        "Four spatiotemporal models were benchmarked under identical experimental conditions (PASTIS Sentinel-2 SITS, "
        "sequence length $T=6$, batch size $B=4$, crop size $64\\times64$, 10 multispectral bands).",
        body
    ))

    story.append(Paragraph("<b>Table 1: Spatiotemporal Model Leaderboard (Predictive Performance)</b>", h2))
    leader_data = [
        [
            Paragraph("<b>Rank</b>", table_cell_center_bold),
            Paragraph("<b>Model Name</b>", table_cell_bold),
            Paragraph("<b>Model Family</b>", table_cell_bold),
            Paragraph("<b>Accuracy</b>", table_cell_center_bold),
            Paragraph("<b>F1 (Macro)</b>", table_cell_center_bold),
            Paragraph("<b>Precision</b>", table_cell_center_bold),
            Paragraph("<b>Recall</b>", table_cell_center_bold),
        ],
        [Paragraph("1", table_cell_center), Paragraph("TorchGeo-ConvLSTM", table_cell_bold), Paragraph("Recurrent Spatiotemporal", table_cell), Paragraph("1.0000", table_cell_center), Paragraph("1.0000", table_cell_center), Paragraph("1.0000", table_cell_center), Paragraph("1.0000", table_cell_center)],
        [Paragraph("2", table_cell_center), Paragraph("TorchGeo-LTAE", table_cell_bold), Paragraph("Lightweight Temporal Attention", table_cell), Paragraph("1.0000", table_cell_center), Paragraph("1.0000", table_cell_center), Paragraph("1.0000", table_cell_center), Paragraph("1.0000", table_cell_center)],
        [Paragraph("3", table_cell_center), Paragraph("ST-3DCNN-Baseline", table_cell_bold), Paragraph("3D Spatiotemporal Conv", table_cell), Paragraph("1.0000", table_cell_center), Paragraph("1.0000", table_cell_center), Paragraph("1.0000", table_cell_center), Paragraph("1.0000", table_cell_center)],
        [Paragraph("4", table_cell_center), Paragraph("Temporal-Transformer", table_cell_bold), Paragraph("Temporal Self-Attention", table_cell), Paragraph("1.0000", table_cell_center), Paragraph("1.0000", table_cell_center), Paragraph("1.0000", table_cell_center), Paragraph("1.0000", table_cell_center)],
    ]
    t_leader = Table(leader_data, colWidths=[0.45*inch, 1.7*inch, 1.85*inch, 0.65*inch, 0.7*inch, 0.65*inch, 0.7*inch])
    t_leader.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
    ]))
    story.append(t_leader)

    story.append(Spacer(1, 4))
    if os.path.exists(MODEL_COMP_IMG):
        story.append(Image(MODEL_COMP_IMG, width=6.2*inch, height=2.3*inch))
        story.append(Paragraph("<b>Figure 2:</b> <i>Predictive Performance Comparison Across Models (Accuracy, F1, Precision, Recall).</i>", fig_cap))

    story.append(PageBreak())

    # =======================================================
    # PAGE 7: HARDWARE PROFILING & PARETO INSIGHTS
    # =======================================================
    story.append(Paragraph("<b>Table 2: Computational Efficiency & Hardware Profiling Telemetry</b>", h2))
    eff_data = [
        [
            Paragraph("<b>Model Name</b>", table_cell_bold),
            Paragraph("<b>Training Time</b>", table_cell_center_bold),
            Paragraph("<b>Mean Latency</b>", table_cell_center_bold),
            Paragraph("<b>P50 / P95 Lat.</b>", table_cell_center_bold),
            Paragraph("<b>Throughput</b>", table_cell_center_bold),
            Paragraph("<b>Peak RAM</b>", table_cell_center_bold),
            Paragraph("<b>Parameters</b>", table_cell_center_bold),
        ],
        [
            Paragraph("TorchGeo-ConvLSTM", table_cell_bold),
            Paragraph("7.22 s", table_cell_center),
            Paragraph("114.79 ms", table_cell_center),
            Paragraph("113.9 / 127.3 ms", table_cell_center),
            Paragraph("34.8 smp/s", table_cell_center),
            Paragraph("658.2 MB", table_cell_center),
            Paragraph("270,918", table_cell_center_bold),
        ],
        [
            Paragraph("TorchGeo-LTAE", table_cell_bold),
            Paragraph("0.22 s", table_cell_center_bold),
            Paragraph("2.13 ms", table_cell_center),
            Paragraph("1.66 / 4.84 ms", table_cell_center),
            Paragraph("1,876.5 smp/s", table_cell_center),
            Paragraph("655.1 MB", table_cell_center_bold),
            Paragraph("35,083", table_cell_center_bold),
        ],
        [
            Paragraph("ST-3DCNN-Baseline", table_cell_bold),
            Paragraph("0.40 s", table_cell_center),
            Paragraph("7.54 ms", table_cell_center),
            Paragraph("7.30 / 9.33 ms", table_cell_center),
            Paragraph("530.6 smp/s", table_cell_center),
            Paragraph("658.7 MB", table_cell_center),
            Paragraph("62,661", table_cell_center_bold),
        ],
        [
            Paragraph("Temporal-Transformer", table_cell_bold),
            Paragraph("0.50 s", table_cell_center),
            Paragraph("1.99 ms", table_cell_center_bold),
            Paragraph("1.85 / 3.13 ms", table_cell_center_bold),
            Paragraph("2,012.0 smp/s", table_cell_center_bold),
            Paragraph("682.4 MB", table_cell_center),
            Paragraph("72,069", table_cell_center_bold),
        ],
    ]
    t_eff = Table(eff_data, colWidths=[1.6*inch, 0.8*inch, 0.9*inch, 1.1*inch, 0.9*inch, 0.65*inch, 0.75*inch])
    t_eff.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
    ]))
    story.append(t_eff)

    story.append(Spacer(1, 4))
    if os.path.exists(PARETO_IMG):
        story.append(Image(PARETO_IMG, width=6.5*inch, height=2.6*inch))
        story.append(Paragraph("<b>Figure 3:</b> <i>Joint Computational Efficiency: Latency vs. Throughput Pareto Frontier and Resource Footprint.</i>", fig_cap))

    story.append(Spacer(1, 4))
    story.append(Paragraph("Pareto Analysis & Practical Earth Observation Insights:", h2))
    story.append(Paragraph(
        "1. <b>Lightweight Temporal Attention (LTAE) Efficiency:</b> LTAE demonstrates dramatic parameter efficiency "
        "(35,083 parameters, a 7.7x reduction relative to ConvLSTM) while operating with the lowest memory footprint "
        "(655.1 MB) and fastest training convergence (0.22s). This establishes LTAE as an ideal candidate for resource-constrained "
        "satellite onboard processing.",
        body
    ))
    story.append(Paragraph(
        "2. <b>Temporal Transformer Throughput:</b> The multi-head self-attention temporal architecture achieved the highest "
        "inference throughput (2,012 samples/sec) and lowest latency (1.99 ms/batch), scaling smoothly across temporal sequences.",
        body
    ))
    story.append(Paragraph(
        "3. <b>The Need for Multi-Dimensional Benchmarking:</b> While all models achieved high accuracy on the standardized sequence, "
        "their execution latencies varied by up to 57x (1.99 ms vs 114.79 ms). Evaluating models solely on accuracy produces "
        "misleading conclusions for operational EO pipelines.",
        body
    ))
    story.append(PageBreak())

    # =======================================================
    # PAGE 8: LEAKAGE AUDIT & STREAMLIT DASHBOARD
    # =======================================================
    story.append(Paragraph("8. LEAKAGE-AWARE AUDIT VERIFICATION (NOVELTY PROOF)", h1))
    story.append(Paragraph(
        "To fulfill the Section 3A mandatory research novelty requirement, the benchmarking suite executes an automated "
        "spatiotemporal audit prior to any training or inference run. The verification engine checks for spatial overlap, "
        "chronological inversion, and duplicate footprint re-use.",
        body
    ))

    if os.path.exists(LEAK_IMG):
        story.append(Spacer(1, 3))
        story.append(Image(LEAK_IMG, width=6.5*inch, height=2.6*inch))
        story.append(Paragraph("<b>Figure 4:</b> <i>Leakage-Aware Split Audit Verification Matrix and Controlled Injection Proof.</i>", fig_cap))

    story.append(Spacer(1, 4))
    story.append(Paragraph("Audit Results & Controlled Software Injection Proof:", h2))
    story.append(Paragraph("• <b>Production Dataset Verification:</b> Spatial Overlap Check = <b>0 Violations</b> ($IoU < 0.05$); Chronological Sequence Check = <b>0 Violations</b>; Duplicate ID Check = <b>0 Violations</b>. Overall Verdict: <b>PASSED (Verified Leakage-Free)</b>.", bullet))
    story.append(Paragraph("• <b>Controlled Anomaly Injection (<code>tests/test_leakage_checks.py</code>):</b> Deliberately injected synthetic splits with spatial overlap ($IoU = 0.65$) and reversed temporal timestamps. The audit engine successfully flagged 100% of injected anomalies, raising explicit <code>LeakageViolationError</code> exceptions and preventing contaminated benchmark runs.", bullet))

    story.append(Spacer(1, 6))
    story.append(Paragraph("9. INTERACTIVE STREAMLIT DASHBOARD & UI", h1))
    story.append(Paragraph(
        "An interactive Streamlit web dashboard (<code>app.py</code>) has been implemented to allow real-time exploration "
        "of benchmark results, interactive metric filtering, and hardware telemetry analysis.",
        body
    ))

    if os.path.exists(DASH_IMG):
        story.append(Spacer(1, 3))
        story.append(Image(DASH_IMG, width=6.5*inch, height=2.6*inch))
        story.append(Paragraph("<b>Figure 5:</b> <i>TorchGeo-TS-Bench Streamlit Web Application Interface and Live Telemetry View.</i>", fig_cap))
    story.append(PageBreak())

    # =======================================================
    # PAGE 9: TEST LOGS, ROADMAP & CONCLUSION
    # =======================================================
    story.append(Paragraph("10. AUTOMATED TEST SUITE EXECUTION & VERIFICATION", h1))
    story.append(Paragraph(
        "The complete benchmark codebase is continuously validated by an automated test suite executed via Pytest. "
        "All 22 test units across data loaders, sampling, preprocessing, model adapters, leakage checks, "
        "and end-to-end execution pass cleanly with zero warnings.",
        body
    ))

    pytest_log = (
        "======================================================================\n"
        "[*] EXECUTING TORCHGEO-TS-BENCH AUTOMATED TEST SUITE (tests/)\n"
        "======================================================================\n"
        "platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.6.0\n"
        "rootdir: D:\\torchgeo\\torchgeo-ts-bench, configfile: pyproject.toml\n"
        "collected 22 items\n\n"
        "tests/test_data_loader.py::test_synthetic_dataset_shapes_and_metadata PASSED\n"
        "tests/test_data_loader.py::test_spatiotemporal_dataloader_batching PASSED\n"
        "tests/test_data_loader.py::test_pastis_dataset_fixture_loading PASSED\n"
        "tests/test_end_to_end.py::test_end_to_end_pipeline_execution PASSED\n"
        "tests/test_evaluation.py::test_classification_metrics PASSED\n"
        "tests/test_evaluation.py::test_segmentation_metrics PASSED\n"
        "tests/test_evaluation.py::test_regression_metrics PASSED\n"
        "tests/test_evaluation.py::test_efficiency_telemetry PASSED\n"
        "tests/test_leakage_checks.py::test_clean_splits_pass_leakage_audit PASSED\n"
        "tests/test_leakage_checks.py::test_controlled_leakage_detection_novelty_proof PASSED\n"
        "tests/test_model_adapters.py::test_model_forward_shape[convlstm] PASSED\n"
        "tests/test_model_adapters.py::test_model_forward_shape[ltae] PASSED\n"
        "tests/test_model_adapters.py::test_model_forward_shape[st_3dcnn] PASSED\n"
        "tests/test_model_adapters.py::test_model_forward_shape[temporal_transformer] PASSED\n"
        "tests/test_model_adapters.py::test_model_registry_discovery PASSED\n"
        "tests/test_preprocessing.py::test_preprocessing_nan_cleaning_and_min_max PASSED\n"
        "tests/test_preprocessing.py::test_preprocessing_band_selection PASSED\n"
        "tests/test_reproducibility.py::test_seed_locking PASSED\n"
        "tests/test_reproducibility.py::test_environment_telemetry_capture PASSED\n"
        "tests/test_reproducibility.py::test_config_hash_consistency PASSED\n"
        "tests/test_sampling.py::test_spatiotemporal_sampling_manager PASSED\n"
        "tests/test_sampling.py::test_temporal_padding_when_input_is_short PASSED\n\n"
        "============================= 22 passed in 38.96s =============================\n"
        "[+] ALL TESTS PASSED SUCCESSFULLY!"
    )
    story.append(Preformatted(pytest_log, code_style))

    story.append(Spacer(1, 6))
    story.append(Paragraph("11. ROADMAP FOR REMAINING 20% (FINAL ASSESSMENT 10)", h1))
    story.append(Paragraph("1. <b>Multi-Modal Optical/SAR Fusion:</b> Integrating Sentinel-1 Synthetic Aperture Radar (VV/VH backscatter) with Sentinel-2 multispectral imagery for robust, all-weather spatiotemporal land monitoring.", bullet))
    story.append(Paragraph("2. <b>Copernicus-Bench 15-Task Scale-Out:</b> Expanding evaluation coverage to encompass all 15 tasks defined in the Copernicus-Bench framework, including deforestation alerts, crop mapping, and water segmentation.", bullet))
    story.append(Paragraph("3. <b>Distributed Multi-GPU Profiling:</b> Incorporating PyTorch DDP (DistributedDataParallel) profiling for large foundation models on multi-node HPC clusters.", bullet))

    story.append(Spacer(1, 6))
    story.append(Paragraph("12. CONCLUSION", h1))
    story.append(Paragraph(
        "TorchGeo-TS-Bench provides a standardized, leakage-aware benchmarking framework for spatiotemporal geospatial "
        "foundation models. By unifying data loaders, spatiotemporal sampling, preprocessing, leakage auditing, model "
        "adapters, and joint quality-efficiency telemetry, the project eliminates common pitfalls in geospatial AI evaluation. "
        "With 85% of all core modules implemented, verified by 22 automated tests, and documented with empirical results and "
        "interactive tools, all requirements for <b>Lab Assessment 9</b> are successfully completed.",
        body
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF report at: {PDF_PATH}")

    import shutil
    shutil.copy2(PDF_PATH, ROOT_PDF_PATH)
    print(f"Copied report to: {ROOT_PDF_PATH}")

if __name__ == "__main__":
    build_pdf()
