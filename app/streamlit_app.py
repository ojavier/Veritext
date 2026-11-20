"""Streamlit interface for paragraph-level Veritext analysis."""

from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from report.alert import build_report  # noqa: E402
from src.pipeline import Pipeline, load_config  # noqa: E402

LEVEL_COLORS = {
    "high": ("#fdecea", "#c62828", "Alta"),
    "uncertain": ("#fff4e5", "#ef6c00", "Incierta"),
    "low": ("#edf7ee", "#2e7d32", "Baja"),
}


def paragraph_style(level: str) -> dict[str, str]:
    background, border, _ = LEVEL_COLORS.get(
        level,
        ("#f7f7f7", "#999999", "Desconocida"),
    )
    return {
        "background": background,
        "border": f"6px solid {border}",
    }


def evidence_text(evidence: dict[str, Any]) -> list[str]:
    """Convert distilled evidence to short, readable rows."""
    rows: list[str] = []
    source = evidence.get("closest_source")
    if source:
        rows.append(
            f"Fuente cercana: {source.get('title') or source.get('id') or '?'} "
            f"(similitud {source.get('similarity')})"
        )
        if source.get("excerpt"):
            rows.append(f"Fragmento fuente: {source['excerpt']}")

    generation = evidence.get("closest_known_generation")
    if generation:
        shared = (
            f"; texto compartido: {generation['shared_text']}"
            if generation.get("shared_text")
            else ""
        )
        rows.append(
            f"Generación conocida: {generation.get('id') or '?'} "
            f"({generation.get('generator') or '?'}, "
            f"puntuación {generation.get('score')}{shared})"
        )
        if generation.get("excerpt"):
            rows.append(f"Fragmento generado: {generation['excerpt']}")

    zeroshot = evidence.get("zeroshot")
    if zeroshot:
        rows.append(
            "Zero-shot: "
            + ", ".join(f"{key}={value}" for key, value in zeroshot.items())
        )

    fusion = evidence.get("fusion")
    if fusion:
        rows.append(f"Fusión: {fusion}")
    return rows


@st.cache_resource
def get_pipeline(device: str, use_fusion: bool) -> Pipeline:
    config = load_config()
    config["zeroshot"]["device"] = device
    if not use_fusion:
        return Pipeline(config, combiner=False)
    return Pipeline(config)


def render_results(alerts: list[Any]) -> None:
    report = build_report(alerts)
    levels = report["summary"]["levels"]
    st.caption(report["note"])
    st.write(
        f"{report['summary']['segments']} segmentos: "
        f"{levels['high']} alta, {levels['uncertain']} incierta, "
        f"{levels['low']} baja."
    )

    for item in report["alerts"]:
        level = item["level"]
        style = paragraph_style(level)
        segment = item["segment"]
        signals = item["signals"]
        _, _, label = LEVEL_COLORS.get(level, ("", "", "Desconocida"))
        text = html.escape(segment["text"])

        st.markdown(
            f"""
            <div style="
                border-left:{style['border']};
                background:{style['background']};
                padding:12px 16px;
                margin:8px 0;
                border-radius:6px;">
                <div style="font-size:0.82rem;color:#444;margin-bottom:6px;">
                    <b>#{segment['id']} · {label}</b> ·
                    puntuación {item['score']:.4f} ·
                    similitud {signals['similarity']:.4f} ·
                    coincidencia {signals['matching']:.4f} ·
                    IA {signals['ai_prob']:.4f} ·
                    idioma {html.escape(str(segment['lang']))}
                </div>
                <div>{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        evidence_rows = evidence_text(item["evidence"])
        with st.expander("Evidencia", expanded=False):
            if evidence_rows:
                for row in evidence_rows:
                    st.write(row)
            else:
                st.caption("No hay evidencia disponible para este segmento.")


def main() -> None:
    st.set_page_config(
        page_title="Veritext",
        page_icon="V",
        layout="wide",
    )
    st.title("Veritext")
    st.caption("Análisis para revisión humana; no constituye un veredicto de autoría.")

    with st.sidebar:
        st.header("Configuración")
        device = st.selectbox("Dispositivo", ("cpu", "cuda", "mps"), index=0)
        use_fusion = st.toggle("Usar modelo de fusión", value=True)
        st.divider()
        st.caption(
            "Los párrafos se colorean según el nivel: alta, incierta o baja "
            "probabilidad de generación por IA."
        )

    text = st.text_area(
        "Texto a analizar",
        height=280,
        placeholder="Pegue aquí uno o varios párrafos...",
    )

    analyze = st.button("Analizar", type="primary", use_container_width=True)
    if analyze:
        if not text.strip():
            st.warning("Ingrese texto antes de analizar.")
            return
        try:
            with st.spinner("Analizando párrafos..."):
                pipeline = get_pipeline(device, use_fusion)
                alerts = pipeline.run(text)
        except Exception as error:
            st.error(f"No fue posible analizar el texto: {error}")
            return

        if alerts:
            render_results(alerts)
        else:
            st.info("No se encontraron segmentos de texto para analizar.")


if __name__ == "__main__":
    main()