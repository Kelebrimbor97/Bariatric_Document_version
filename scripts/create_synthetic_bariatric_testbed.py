#!/usr/bin/env python3
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: reportlab\n"
        "Install it with:\n"
        "  pip install reportlab"
    ) from exc


SYNTHETIC_CASES = {
    "Synthetic 1 - SYN001": {
        "Clinical Documents/Outpatient Core/nutrition_follow_up.pdf": """
Nutrition Clinic Note
Result title: IDEAL Nutrition Follow Up

Reason for visit:
Follow-up after laparoscopic sleeve gastrectomy.

Assessment:
The patient is six months status post sleeve gastrectomy.
Weight loss is progressing as expected.
The patient reports taking a bariatric multivitamin daily.

Nutrition and PA Goals:
Continue bariatric multivitamin.
Continue calcium citrate 600 mg twice daily.
Continue vitamin D3 3000 IU daily.
Continue vitamin B12 500 mcg daily.
Start ferrous sulfate 325 mg every other day because ferritin is low.

Monitoring plan:
Repeat vitamin D, B12, iron panel, ferritin, calcium, and PTH in three months.
Thiamine level was not ordered at this visit.
""",
        "Perioperative Documents/operative_report_sleeve_gastrectomy.pdf": """
Operative Report

Preoperative diagnosis:
Class II obesity with obesity-related comorbidity.

Procedure performed:
Laparoscopic sleeve gastrectomy.

Findings:
Normal appearing stomach and liver.
No hiatal hernia was identified.

Procedure details:
A sleeve gastrectomy was performed over a bougie.
Staple line was inspected and hemostasis was achieved.

Postoperative plan:
Advance diet per bariatric protocol.
Follow up with surgery clinic and nutrition clinic.
""",
        "Laboratory Documents/lab_results_micronutrients.pdf": """
Laboratory Results

Vitamin B12: 310 pg/mL
Iron: 42 mcg/dL
Ferritin: 12 ng/mL
25-OH Vitamin D: 18 ng/mL
Calcium: 9.1 mg/dL
Parathyroid hormone PTH: 68 pg/mL

Interpretation:
Ferritin is low.
Vitamin D is low.
B12 is borderline low.
Calcium is within reference range.
PTH is mildly elevated.
Thiamine was not measured.
""",
        "Clinical Documents/Outpatient Core/medication_list.pdf": """
Medication List

Active medications:
Bariatric multivitamin one tablet daily.
Calcium citrate 600 mg by mouth twice daily.
Vitamin D3 3000 IU by mouth daily.
Vitamin B12 500 mcg by mouth daily.
Ferrous sulfate 325 mg by mouth every other day.

Allergies:
No known drug allergies.
""",
    },
    "Synthetic 2 - SYN002": {
        "Clinical Documents/Outpatient Core/clinic_visit.pdf": """
Clinic Note

Reason for visit:
Preoperative bariatric surgery evaluation.

Assessment:
Patient is being evaluated for Roux-en-Y gastric bypass.
Comorbidities include type 2 diabetes mellitus and hypertension.
No prior bariatric procedure is documented.

Plan:
Proceed with preoperative clearance pathway.
Nutrition evaluation requested.
Psychology evaluation requested.
No micronutrient laboratory monitoring is documented in this note.
""",
        "Clinical Documents/Outpatient Core/nutrition_initial_evaluation.pdf": """
Nutrition Initial Evaluation

Diet history:
Patient reports frequent sugar-sweetened beverages and late-night snacking.
Protein intake is inconsistent.

Assessment:
Nutrition knowledge deficit related to bariatric surgery preparation.

Plan:
Begin preoperative nutrition education.
Increase protein intake.
Avoid sugar-sweetened beverages.
Begin daily complete multivitamin.

There is not evidence to update the Nutrition Diagnosis based on today's visit.
Completed Action List:
Perform by synthetic clinician.
Modify by synthetic clinician.
Verify by synthetic clinician.
Printed by synthetic system.
Printed on synthetic date.
""",
        "Laboratory Documents/basic_labs.pdf": """
Laboratory Results

Hemoglobin A1c: 8.2 percent
Glucose: 168 mg/dL
Creatinine: 0.9 mg/dL
ALT: 32 U/L
AST: 28 U/L

No vitamin B12 result is present.
No ferritin result is present.
No iron panel is present.
No thiamine result is present.
No vitamin D result is present.
No calcium or PTH result is present in this document.
""",
    },
    "Synthetic 3 - SYN003": {
        "Clinical Documents/Outpatient Core/discharge_summary.pdf": """
Discharge Summary

Hospital course:
Patient was admitted for nausea and dehydration after recent Roux-en-Y gastric bypass.
Symptoms improved with intravenous fluids and antiemetics.

Discharge diagnoses:
Dehydration.
Status post Roux-en-Y gastric bypass.
Nausea.

Discharge medications:
Ondansetron as needed.
Omeprazole daily.
Bariatric multivitamin daily.
Thiamine 100 mg daily for 14 days due to poor oral intake.
Continue calcium citrate and vitamin D.

Follow-up:
Bariatric surgery follow-up in one week.
Nutrition follow-up in two weeks.
Repeat basic metabolic panel if poor intake continues.
""",
        "Laboratory Documents/lab_results_postop.pdf": """
Laboratory Results

Sodium: 134 mmol/L
Potassium: 3.4 mmol/L
Creatinine: 1.1 mg/dL
Calcium: 8.6 mg/dL
Magnesium: 1.7 mg/dL
Phosphorus: 2.9 mg/dL

No ferritin result is available.
No B12 result is available.
No PTH result is available.
No vitamin D result is available.
""",
        "Radiology/radiology_upper_gi.pdf": """
Radiology Report

Exam:
Upper GI contrast study.

Findings:
Postoperative anatomy compatible with Roux-en-Y gastric bypass.
No contrast leak is identified.
Mild delayed transit through the gastrojejunal anastomosis.

Impression:
No evidence of postoperative leak.
""",
    },
}


SMOKE_QUESTIONS = [
    {
        "patient_id": "SYN001",
        "question": "What bariatric procedure or surgical history is documented?",
    },
    {
        "patient_id": "SYN001",
        "question": "Is there evidence of B12, iron, ferritin, thiamine, vitamin D, calcium, or PTH monitoring?",
    },
    {
        "patient_id": "SYN001",
        "question": "What vitamin or micronutrient supplementation is documented?",
    },
    {
        "patient_id": "SYN002",
        "question": "What bariatric procedure is planned or documented?",
    },
    {
        "patient_id": "SYN002",
        "question": "Is there evidence of micronutrient laboratory monitoring?",
    },
    {
        "patient_id": "SYN003",
        "question": "What postoperative complication or follow-up issue is documented?",
    },
    {
        "patient_id": "SYN003",
        "question": "What thiamine or vitamin supplementation is documented?",
    },
]


def write_pdf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter

    x = 72
    y = height - 72
    line_height = 13
    max_chars = 92

    for raw_line in text.strip().splitlines():
        raw_line = raw_line.rstrip()

        if not raw_line:
            y -= line_height
            continue

        wrapped = textwrap.wrap(raw_line, width=max_chars) or [""]
        for line in wrapped:
            if y < 72:
                c.showPage()
                y = height - 72

            c.drawString(x, y, line)
            y -= line_height

    c.save()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a tiny synthetic bariatric PDF testbed with no PHI."
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("Data/public_testbed/synthetic_bariatric_pdf/Test Patients"),
        help="Output root to use as PATIENTS_ROOT.",
    )
    parser.add_argument(
        "--questions-out",
        type=Path,
        default=Path("eval/synthetic_bariatric_smoke_questions.jsonl"),
        help="Where to write JSONL smoke questions.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing PDFs/questions.",
    )
    args = parser.parse_args()

    if args.out_root.exists() and not args.force:
        raise SystemExit(
            f"Output root already exists: {args.out_root}\n"
            "Use --force if you want to overwrite/regenerate files."
        )

    for patient_folder, docs in SYNTHETIC_CASES.items():
        for relative_doc_path, text in docs.items():
            pdf_path = args.out_root / patient_folder / relative_doc_path
            write_pdf(pdf_path, text)

    args.questions_out.parent.mkdir(parents=True, exist_ok=True)
    with args.questions_out.open("w", encoding="utf-8") as f:
        for item in SMOKE_QUESTIONS:
            import json

            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote synthetic PDF corpus under: {args.out_root}")
    print(f"Wrote smoke questions: {args.questions_out}")
    print()
    print("Example build env:")
    print(f"  PATIENTS_ROOT='{args.out_root}' COLLECTION_NAME=ehr_chunks_synth_v1 ./run_build.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())