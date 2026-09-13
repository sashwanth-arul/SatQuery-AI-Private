"""Report generation service for SatQuery AI.

Produces self-contained, downloadable HTML reports for remote sensing analyses.
Never invents numbers; strictly consumes verified AnalysisResult metrics.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from typing import Any

from app.schemas.domain import AnalysisResult, EvidenceRegion, TraceStep


class ReportService:
    """Generates clean, executive-ready HTML reports for completed analyses."""

    def generate_html_report(self, result: AnalysisResult) -> str:
        session_id = html.escape(result.session_id)
        created_at_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        answer_text = html.escape(result.answer)
        mode_str = html.escape(result.mode.value if hasattr(result.mode, "value") else str(result.mode))
        confidence_pct = (
            f"{round(result.confidence * 100)}%"
            if result.confidence_available and result.confidence is not None
            else "N/A (Qualitative / Verification Needed)"
        )

        # 1. Specialized analysis sections
        specialized_sections_html = []

        # Cross-modal section
        if result.cross_modal:
            cm = result.cross_modal
            opt_summary = html.escape(cm.optical_analysis.summary)
            sar_summary = html.escape(cm.sar_analysis.summary)
            fused_summary = html.escape(cm.fused_analysis.summary)
            co_reg = html.escape(cm.co_registration_status.replace("_", " "))
            co_reg_prov = html.escape(cm.co_registration_provenance)
            specialized_sections_html.append(
                f"""
                <section class="report-section">
                    <h2>Cross-Modal Optical + SAR Analysis</h2>
                    <div class="card grid-2">
                        <div>
                            <h3>Optical Analysis</h3>
                            <p>{opt_summary}</p>
                            <span class="badge">Tool: {html.escape(cm.optical_analysis.analyzer)}</span>
                        </div>
                        <div>
                            <h3>SAR Analysis</h3>
                            <p>{sar_summary}</p>
                            <span class="badge">Tool: {html.escape(cm.sar_analysis.analyzer)}</span>
                        </div>
                    </div>
                    <div class="card mt-12">
                        <h3>Joint Fusion Result</h3>
                        <p>{fused_summary}</p>
                        <div class="meta-row">
                            <strong>Co-registration:</strong> {co_reg} ({co_reg_prov}) &bull;
                            <strong>Fused Regions:</strong> {cm.fused_analysis.fused_region_count} &bull;
                            <strong>Policy:</strong> {html.escape(cm.fused_analysis.fusion_policy)}
                        </div>
                    </div>
                </section>
                """
            )

        # Building detection section
        if result.building_detection:
            bd = result.building_detection
            avg_footprint = (
                f"{round(bd.total_area_m2 / bd.count):,} m²" if bd.count > 0 else "0 m²"
            )
            specialized_sections_html.append(
                f"""
                <section class="report-section">
                    <h2>Building Detection (Specialist Tool)</h2>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-num">{bd.count}</div>
                            <div class="stat-label">Detected Buildings</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-num">{round(bd.total_area_m2):,} m²</div>
                            <div class="stat-label">Total Footprint Area</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-num">{avg_footprint}</div>
                            <div class="stat-label">Average Footprint</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-num">{round(bd.confidence * 100)}%</div>
                            <div class="stat-label">Detection Confidence</div>
                        </div>
                    </div>
                    <p class="meta-note">Detector: <strong>{html.escape(bd.detector_name)}</strong></p>
                </section>
                """
            )

        # Building temporal change section
        if result.building_temporal_change:
            btc = result.building_temporal_change
            net_change = btc.after_count - btc.before_count
            net_sign = "+" if net_change > 0 else ""
            specialized_sections_html.append(
                f"""
                <section class="report-section">
                    <h2>Building Temporal Change (Bipartite Matching)</h2>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-num">{btc.before_count}</div>
                            <div class="stat-label">T1 Before Count</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-num">{btc.after_count}</div>
                            <div class="stat-label">T2 After Count</div>
                        </div>
                        <div class="stat-card highlight-green">
                            <div class="stat-num">+{btc.new_count}</div>
                            <div class="stat-label">New Buildings Added</div>
                        </div>
                        <div class="stat-card highlight-red">
                            <div class="stat-num">-{btc.removed_count}</div>
                            <div class="stat-label">Demolished / Removed</div>
                        </div>
                    </div>
                    <div class="card mt-12">
                        <div class="meta-row">
                            <strong>Unchanged Buildings:</strong> {btc.unchanged_count} &bull;
                            <strong>Modified / Altered:</strong> {btc.changed_count} &bull;
                            <strong>Net Building Delta:</strong> {net_sign}{net_change} &bull;
                            <strong>Algorithm:</strong> {html.escape(btc.matcher_name)}
                        </div>
                    </div>
                </section>
                """
            )

        # Surface area change section
        if result.surface_area_change:
            sac = result.surface_area_change
            domain_label = (
                "Built-Up Area"
                if sac.domain == "built_up"
                else "Water Body"
                if sac.domain == "water"
                else "Vegetation Cover"
            )
            diff_sign = "+" if sac.difference_m2 > 0 else ""
            pct_sign = "+" if sac.percentage_change > 0 else ""
            specialized_sections_html.append(
                f"""
                <section class="report-section">
                    <h2>{html.escape(domain_label)} Quantitative Change</h2>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-num">{round(sac.before_area_m2):,} m²</div>
                            <div class="stat-label">T1 Before Area</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-num">{round(sac.after_area_m2):,} m²</div>
                            <div class="stat-label">T2 After Area</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-num">{diff_sign}{round(sac.difference_m2):,} m²</div>
                            <div class="stat-label">Net Surface Area Delta</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-num">{pct_sign}{sac.percentage_change}%</div>
                            <div class="stat-label">Percentage Change</div>
                        </div>
                    </div>
                    <p class="meta-note">Index: <strong>{html.escape(sac.primary_index.upper())}</strong> &bull; Confidence: <strong>{round(sac.confidence * 100)}%</strong></p>
                </section>
                """
            )

        # Generic bi-temporal change section (if not overridden by building/surface change)
        if (
            result.bi_temporal_change
            and not result.building_temporal_change
            and not result.surface_area_change
        ):
            btc_gen = result.bi_temporal_change
            specialized_sections_html.append(
                f"""
                <section class="report-section">
                    <h2>Bi-Temporal Spectral Change Analysis</h2>
                    <div class="card">
                        <p>{html.escape(btc_gen.change_summary)}</p>
                        <div class="meta-row mt-12">
                            <strong>Detector:</strong> {html.escape(btc_gen.detector)} &bull;
                            <strong>Changed Regions:</strong> {btc_gen.changed_region_count} &bull;
                            <strong>Provider:</strong> {html.escape(btc_gen.provider)}
                        </div>
                    </div>
                </section>
                """
            )

        # 2. Key verified metrics table
        metrics_rows = []
        for m in result.metrics:
            unit_str = f" {html.escape(m.unit)}" if m.unit else ""
            metrics_rows.append(
                f"<tr><td><strong>{html.escape(m.name)}</strong></td><td>{m.value}{unit_str}</td><td>{html.escape(m.source)}</td></tr>"
            )
        metrics_table_html = (
            f"""
            <table class="report-table">
                <thead><tr><th>Metric</th><th>Calculated Value</th><th>Measurement Source</th></tr></thead>
                <tbody>{''.join(metrics_rows)}</tbody>
            </table>
            """
            if metrics_rows
            else "<p class=\"meta-note\">No aggregate numerical metrics reported.</p>"
        )

        # 3. Evidence regions table
        evidence_rows = []
        for r in result.evidence[:30]:  # Cap at top 30 regions
            claim = r.metadata.get("claim_type") or r.type
            modality = r.metadata.get("evidence_modality") or "optical"
            conf_str = f"{round(r.confidence * 100)}%"
            evidence_rows.append(
                f"<tr><td><code>{html.escape(r.id)}</code></td><td>{html.escape(str(claim).replace('_', ' '))}</td><td>{html.escape(str(modality))}</td><td>{conf_str}</td><td>{html.escape(r.source)}</td></tr>"
            )
        evidence_table_html = (
            f"""
            <table class="report-table">
                <thead><tr><th>Region ID</th><th>Claim Type</th><th>Modality</th><th>Confidence</th><th>Source Engine</th></tr></thead>
                <tbody>{''.join(evidence_rows)}</tbody>
            </table>
            """
            if evidence_rows
            else "<p class=\"meta-note\">No localized evidence regions identified.</p>"
        )

        # 4. Execution trace timeline
        trace_rows = []
        for idx, step in enumerate(result.trace, start=1):
            dur = f"{step.duration_ms} ms" if step.duration_ms is not None else "—"
            status_badge = (
                f'<span class="badge badge-success">PASSED</span>'
                if step.status.value == "completed"
                else f'<span class="badge badge-fail">{html.escape(step.status.value.upper())}</span>'
            )
            summary = html.escape(step.summary or "Completed successfully.")
            trace_rows.append(
                f"<tr><td>{idx}</td><td><strong>{html.escape(step.tool_name)}</strong></td><td>{status_badge}</td><td>{dur}</td><td>{summary}</td></tr>"
            )
        trace_table_html = (
            f"""
            <table class="report-table">
                <thead><tr><th>#</th><th>Specialist Tool</th><th>Status</th><th>Latency</th><th>Output Summary</th></tr></thead>
                <tbody>{''.join(trace_rows)}</tbody>
            </table>
            """
            if trace_rows
            else "<p class=\"meta-note\">Trace data unavailable.</p>"
        )

        specialized_content = "\n".join(specialized_sections_html)

        html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SatQuery AI Analysis Report &mdash; {session_id}</title>
    <style>
        :root {{
            --bg: #090b10;
            --surface: #11141d;
            --surface-card: #181d2a;
            --border: #23293b;
            --text: #f0f3fa;
            --text-muted: #8c96ab;
            --accent: #8b5cf6;
            --accent-cyan: #06b6d4;
            --success: #10b981;
            --danger: #ef4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            padding: 40px 24px;
        }}
        .report-container {{
            max-width: 960px;
            margin: 0 auto;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 48px;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4);
        }}
        .report-header {{
            border-bottom: 1px solid var(--border);
            padding-bottom: 24px;
            margin-bottom: 32px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }}
        .brand-title {{
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .brand-pill {{
            background: linear-gradient(135deg, var(--accent), var(--accent-cyan));
            color: white;
            font-size: 11px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 999px;
            text-transform: uppercase;
        }}
        .report-meta {{
            font-size: 13px;
            color: var(--text-muted);
            text-align: right;
        }}
        .report-section {{
            margin-bottom: 36px;
        }}
        h2 {{
            font-size: 18px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 16px;
            letter-spacing: -0.2px;
            border-left: 3px solid var(--accent-cyan);
            padding-left: 10px;
        }}
        h3 {{
            font-size: 15px;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 8px;
        }}
        .answer-box {{
            background: linear-gradient(180deg, rgba(139, 92, 246, 0.08), rgba(6, 182, 212, 0.04));
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 8px;
            padding: 20px;
            font-size: 16px;
            color: #ffffff;
            line-height: 1.6;
        }}
        .card {{
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 18px;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
        }}
        .stat-card {{
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        .stat-num {{
            font-size: 24px;
            font-weight: 800;
            color: var(--accent-cyan);
        }}
        .stat-label {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .highlight-green .stat-num {{ color: var(--success); }}
        .highlight-red .stat-num {{ color: var(--danger); }}
        .report-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
            margin-top: 8px;
        }}
        .report-table th {{
            background: rgba(255, 255, 255, 0.03);
            border-bottom: 1px solid var(--border);
            padding: 10px 12px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }}
        .report-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            color: #cbd5e1;
        }}
        .report-table tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}
        .badge {{
            display: inline-block;
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.08);
            color: #cbd5e1;
            margin-top: 6px;
        }}
        .badge-success {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-fail {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }}
        .meta-row {{ font-size: 13px; color: var(--text-muted); margin-top: 6px; }}
        .meta-note {{ font-size: 13px; color: var(--text-muted); margin-top: 8px; }}
        .mt-12 {{ margin-top: 12px; }}
        .footer-note {{
            margin-top: 48px;
            padding-top: 24px;
            border-top: 1px solid var(--border);
            font-size: 12px;
            color: var(--text-muted);
            text-align: center;
        }}
        @media print {{
            body {{ background: #fff; color: #000; padding: 0; }}
            .report-container {{ box-shadow: none; border: none; padding: 0; }}
            .answer-box {{ border: 1px solid #ccc; color: #000; }}
            .stat-card, .card {{ border: 1px solid #ddd; background: #fafafa; }}
            .stat-num {{ color: #000 !important; }}
            h2 {{ color: #000; border-left-color: #000; }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <header class="report-header">
            <div>
                <div class="brand-title">
                    SatQuery AI
                    <span class="brand-pill">Verified Analysis</span>
                </div>
                <p style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">
                    Geospatial Remote Sensing Intelligence &bull; SIH Problem Statement 26167
                </p>
            </div>
            <div class="report-meta">
                <div><strong>Analysis ID:</strong> <code>{session_id[:12]}</code></div>
                <div><strong>Generated:</strong> {created_at_str}</div>
                <div><strong>Mode:</strong> {mode_str.upper()} &bull; <strong>Confidence:</strong> {confidence_pct}</div>
            </div>
        </header>

        <section class="report-section">
            <h2>Query & Verified Answer</h2>
            <div class="answer-box">
                <p>{answer_text}</p>
            </div>
        </section>

        {specialized_content}

        <section class="report-section">
            <h2>Quantitative Analysis Metrics</h2>
            {metrics_table_html}
        </section>

        <section class="report-section">
            <h2>Localized Spatial Evidence Regions</h2>
            {evidence_table_html}
        </section>

        <section class="report-section">
            <h2>Agentic Execution Trace</h2>
            {trace_table_html}
        </section>

        <footer class="footer-note">
            <p><strong>SatQuery AI Deterministic Pipeline:</strong> All quantitative measurements, building footprints, and spectral changes are derived using Computer Vision and verified Earth Observation algorithms. No generative LLM hallucinations are permitted in the calculation layer.</p>
            <p style="margin-top: 4px;">Report Generated automatically from session {session_id}.</p>
        </footer>
    </div>
</body>
</html>
"""
        return html_document


report_service = ReportService()
