#!/usr/bin/env python3
"""
Generate a small synthetic corpus for testing.

This script creates a handful of PDF and HTML documents containing
fictional internal procedures.  It writes them into the
`data/corpus/` directory relative to the project root.  The content
is deliberately simple but covers multiple topics to support
retrieval and evaluation.

Run this script before running `scripts/ingest.py` to populate the
database with sample data.  The Makefile exposes this via
`make ingest`.
"""

from __future__ import annotations

import os
from pathlib import Path
from fpdf import FPDF  # type: ignore


def write_pdf(path: Path, title: str, paragraphs: list[str]) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=14)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_font("Helvetica", size=11)
    for para in paragraphs:
        pdf.multi_cell(0, 8, para)
        pdf.ln(2)
    pdf.output(str(path))


def write_html(path: Path, title: str, paragraphs: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>{}</title></head><body>".format(title))
        f.write(f"<h1>{title}</h1>\n")
        for para in paragraphs:
            f.write(f"<p>{para}</p>\n")
        f.write("</body></html>")


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    corpus_dir = base_dir / "data" / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    docs = [
        (
            "procurement_policy.pdf",
            "Procédure d'achat interne",
            [
                "Cette procédure décrit les étapes à suivre pour toute commande de matériel ou de services.",
                "Les employés doivent soumettre une demande d'achat via le système interne, qui sera ensuite validée par le responsable hiérarchique.",
                "Une fois la demande approuvée, le service des achats sélectionne un fournisseur et passe la commande.",
                "Toutes les factures doivent être transmises au service comptable pour paiement dans un délai de trente jours.",
            ],
        ),
        (
            "employee_handbook.pdf",
            "Manuel de l'employé",
            [
                "Bienvenue dans l'entreprise. Ce manuel résume les règles et attentes concernant le comportement professionnel.",
                "Les horaires de travail habituels sont de 9h à 17h du lundi au vendredi, avec une pause déjeuner d'une heure.",
                "Toute forme de discrimination ou de harcèlement est strictement interdite et peut entraîner des sanctions disciplinaires.",
                "En cas de question, contactez le service des ressources humaines via hr@example.com.",
            ],
        ),
        (
            "security_procedure.html",
            "Procédure de sécurité informatique",
            [
                "La sécurité des systèmes d'information repose sur des mesures techniques et organisationnelles.",
                "Les mots de passe doivent contenir au moins douze caractères et être renouvelés tous les trois mois.",
                "Ne partagez jamais vos identifiants de connexion et verrouillez votre session lorsque vous quittez votre poste.",
                "Tout incident de sécurité doit être signalé immédiatement à l'équipe IT.",
            ],
        ),
        (
            "tech_guidelines.html",
            "Guide de développement logiciel",
            [
                "Ce guide fournit des bonnes pratiques pour le développement et la révision de code.",
                "Utilisez des revues de code systématiques afin d'améliorer la qualité et de partager les connaissances.",
                "Respectez les conventions de nommage et documentez les fonctions et classes avec des docstrings.",
                "Les tests unitaires doivent couvrir les cas critiques et être automatisés dans le pipeline CI.",
            ],
        ),
    ]
    for filename, title, paragraphs in docs:
        out_path = corpus_dir / filename
        if filename.endswith(".pdf"):
            write_pdf(out_path, title, paragraphs)
        else:
            write_html(out_path, title, paragraphs)
    print(f"Corpus synthétique généré dans {corpus_dir}")


if __name__ == "__main__":
    main()