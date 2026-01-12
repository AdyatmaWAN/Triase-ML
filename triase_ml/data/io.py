from __future__ import annotations

import pandas as pd

from .cleaning import build_datasets_from_dataframe, TriaseDatasets



COLUMN_MAPPING = {
    # Identity & metadata (mostly dropped)
    "no": "id",
    "Tanggal masuk ICCU": "date_iccu",
    "nomor rekam medis ": "mrn",

    # Demographics
    "Usia (tahun)": "age",
    "Gender (L/P)": "gender",
    "Berat badan ": "weight",
    "Tinggi Badan ": "height",
    "IMT ": "bmi",

    # Pain Anamnesis
    "Onset (menit) ": "onset_min",
    "Durasi (menit)": "duration_min",
    "Perut": "loc_abdomen",
    "Dada ": "loc_chest",
    "Tajam ": "pain_sharp",
    "Menusuk": "pain_stabbing",
    "Tumpul ": "pain_dull",
    "Tertekan atau merasa berat ": "pain_pressure",
    "Terbakar ": "pain_burning",
    "berdenyut": "pain_throbbing",
    "Penjalaran nyeri ke dagu/leher/ lengan kiri/ bahu kiri": "rad_neck_arm",
    "Skor VAS ": "vas_score",
    
    # Aggravating/Relieving
    "Palpasi/ ditekan ": "agg_palpation",
    "Perubahan posisi": "agg_position",
    "Tarik nafas atau batuk": "agg_breath",
    "aktivitas berat/stress (naik tangga atau jalan jauh)": "agg_exertion",
    "Nyeri saat aktivitas ringansedang (jalan kaki) ": "pain_light_activity",
    "Setelah/sebelum makan ": "pain_meal",
    "Nyeri saat istirahat": "pain_rest",

    # Associated Symptoms
    "Akral dingin ": "cold_acral",
    "Keringat dingin ": "cold_sweat",
    "Sesak ": "dyspnea",
    "Mual atau muntah ": "nausea_vomit",
    "Lemas ": "weakness",
    "Pingsan/syncope/kehilangan kesadaran sementara ": "syncope",
    "Rasa berdebardebar": "palpitations",
    "Pucat ": "pallor",
    "Riwayat nyeri yang serupa sebelumnya": "hist_similar_pain",

    # History
    "Diabetes mellitus": "hist_dm",
    "Hipertensi ": "hist_htn",
    "Dislipidemia ": "hist_dyslipidemia",
    "Riwayat rokok ": "hist_smoking",
    "Riwayat sindrom koroner akut sebelumnya ": "hist_acs",
    "Riwayat penyakit jantung pada keluarga ": "hist_family_heart",
    "Riwayat PCI sebelumnya": "hist_pci",
    "Riwayat CABG sebelumnya": "hist_cabg",
    "Riwayat gagal jantung kronik ": "hist_chf",
    "Riwayat gagal ginjal kronik ": "hist_ckd",
    "Riwayat stroke hemorrhagik ": "hist_stroke_hem",
    "Riwayat stoke iskemik": "hist_stroke_isch",
    "Riwayat penggunaan antiplatelet dan/atau anticoagulant ": "hist_antiplatelet",

    # Complications / Status
    "Syok kardiogenik ": "shock_cardio",
    "Aritmia ": "arrhythmia",
    "Perikarditis": "pericarditis",
    "Tamponade": "tamponade",
    "gagal jantung": "heart_failure",
    "Kesadaran pasien ": "consciousness",
    "Glasgow coma scale ": "gcs",
    "Keadaan umum pasien ": "general_condition",

    # Vitals
    "Tekanan darah sistolik": "bp_sys",
    "Tekanan darah diastolik": "bp_dia",
    "Frekuensi nadi ": "hr",
    "Frekuensi pernafasan": "rr",
    "Suhu ": "temp",
    "Saturasi Oksigen ": "oxygen_sat",

    # Labs
    "Hemoglobin (g/dL)": "hb",
    "Leukosit (sel/uL)": "leukocytes",
    "Thrombosit (ribu/uL)": "platelets",
    "Ureum (mg/dL)": "urea",
    "Kreatinin (mg/dL)": "creatinine",
    "EGFR (mL/min/1.73m2)": "egfr",
    "Sodium (mEq/L)": "sodium",
    "Kalium (mEq/L)": "potassium",
    "TroponinT (ng/ml)": "trop_t",
    "Troponin I(ng/mL)": "trop_i",
    "CKMB (ng/mL)": "ckmb",

    # EKG & Diagnosis
    "EKG (STE (st-elevasi). STD (ST depresi). STN (ST normal))": "ecg_st",
    "Deskripsi EKG": "ecg_desc",
    "Pengobatan lengkap ": "medication",
    "Early Warning system (EWS)": "ews",
    "hijau/kuning/ merah/ hitam": "triage_color",
    "Diagnosa Penyakit jantung pasien  (text) ": "diag_text",
    "Diagnosa Akhir Pasien (text) ": "final_diag_text",
    
    # Targets / Outcomes
    "Ruangan biasa ": "room_ward",
    "Ruangan ICU ": "room_icu",
    "Coroangiography": "coroangiography",
    "Primary PCI (<60 menit)": "primary_pci",
    "Elective PCI": "elective_pci",
    "CABG CITO": "cabg_cito",
    "CABG Elective": "cabg_elective",
}

def load_triase_excel(excel_path: str, sheet_name: str = "iccu_cleaned_imputed") -> TriaseDatasets:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    # Rename columns using our mapping
    # We use a partial map (ignore errors) just in case some cols are missing/typoed in excel
    df.rename(columns=COLUMN_MAPPING, inplace=True)
    
    return build_datasets_from_dataframe(df)
