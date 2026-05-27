"""
PPG Signal Processing
ML + Statistics
Tiny RAG / Retrieval System

Run:
    python kos_ai_3day_practice.py

Optional install:
    pip install numpy pandas scipy matplotlib scikit-learn wfdb sentence-transformers faiss-cpu

Notes:
- This script uses synthetic data by default.
- If internet is available and wfdb is installed, it also tries to load BIDMC data from PhysioNet.
- The synthetic glucose/PPG dataset is for coding practice only, not medical use.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, welch, find_peaks, detrend
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(42)


# ============================================================
# Utility
# ============================================================

def print_section(title: str):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# DAY 1 — SIGNAL PROCESSING
# ============================================================

def generate_synthetic_ppg(fs=25, duration_sec=120, heart_rate_bpm=78, noise_level=0.08):
    """
    Generate synthetic PPG-like signal.

    fs: sampling frequency in Hz
    duration_sec: signal duration
    heart_rate_bpm: simulated heart rate
    noise_level: Gaussian noise level
    """
    t = np.arange(0, duration_sec, 1 / fs)
    hr_hz = heart_rate_bpm / 60.0

    ppg = (
        1.0 * np.sin(2 * np.pi * hr_hz * t)
        + 0.35 * np.sin(2 * np.pi * 2 * hr_hz * t)
        + 0.25 * np.sin(2 * np.pi * 0.15 * t)
        + 0.15 * np.sin(2 * np.pi * 3.2 * t)
        + noise_level * np.random.randn(len(t))
    )

    return t, ppg


def bandpass_filter(signal, fs, lowcut=0.5, highcut=4.0, order=4):
    """
    Butterworth bandpass filter with zero-phase filtfilt.
    0.5–4 Hz roughly corresponds to 30–240 BPM.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="bandpass")
    filtered = filtfilt(b, a, signal)
    return filtered


def estimate_hr_from_peaks(ppg_signal, fs, min_bpm=40, max_bpm=200):
    """
    Estimate heart rate using time-domain peak detection.
    """
    min_distance_samples = int(fs * 60 / max_bpm)

    peaks, _ = find_peaks(
        ppg_signal,
        distance=min_distance_samples,
        prominence=np.std(ppg_signal) * 0.3,
    )

    if len(peaks) < 2:
        return np.nan, peaks

    peak_intervals_sec = np.diff(peaks) / fs
    bpm_values = 60 / peak_intervals_sec
    bpm_values = bpm_values[(bpm_values >= min_bpm) & (bpm_values <= max_bpm)]

    if len(bpm_values) == 0:
        return np.nan, peaks

    return float(np.mean(bpm_values)), peaks


def estimate_hr_from_welch(ppg_signal, fs, low_hz=0.5, high_hz=4.0):
    """
    Estimate heart rate using Welch power spectral density.
    """
    freqs, psd = welch(ppg_signal, fs=fs, nperseg=min(len(ppg_signal), fs * 16))

    valid = (freqs >= low_hz) & (freqs <= high_hz)
    peak_freq = freqs[valid][np.argmax(psd[valid])]
    bpm = peak_freq * 60

    return float(bpm), freqs, psd


def sliding_window_hr(ppg_signal, fs, window_sec=10, hop_sec=2):
    """
    Simulate real-time heart-rate estimation using sliding windows.
    """
    window = int(window_sec * fs)
    hop = int(hop_sec * fs)

    results = []

    for start in range(0, len(ppg_signal) - window + 1, hop):
        end = start + window
        segment = ppg_signal[start:end]
        bpm, _ = estimate_hr_from_peaks(segment, fs)

        results.append(
            {
                "start_sec": start / fs,
                "end_sec": end / fs,
                "bpm": bpm,
            }
        )

    return pd.DataFrame(results)


def try_load_bidmc_record(record_name="bidmc01"):
    """
    Try to load one BIDMC record from PhysioNet.

    This requires:
        pip install wfdb

    and internet access.
    """
    try:
        import wfdb

        record = wfdb.rdrecord(record_name, pn_dir="bidmc/1.0.0")
        print("Loaded BIDMC record:", record_name)
        print("Signal names:", record.sig_name)
        print("Sampling frequency:", record.fs)

        data = pd.DataFrame(record.p_signal, columns=record.sig_name)
        return record, data

    except Exception as exc:
        print("Could not load BIDMC from PhysioNet.")
        print("Reason:", exc)
        return None, None


def run_day1_signal_processing():
    print_section("DAY 1 — PPG SIGNAL PROCESSING")

    fs = 25
    t, raw_ppg = generate_synthetic_ppg(fs=fs, duration_sec=120, heart_rate_bpm=78)

    filtered_ppg = bandpass_filter(detrend(raw_ppg), fs)

    bpm_peak, peaks = estimate_hr_from_peaks(filtered_ppg, fs)
    bpm_welch, freqs, psd = estimate_hr_from_welch(filtered_ppg, fs)

    print(f"Peak-based estimated HR : {bpm_peak:.2f} BPM")
    print(f"Welch-based estimated HR: {bpm_welch:.2f} BPM")

    hr_df = sliding_window_hr(filtered_ppg, fs)
    print("\nSliding-window HR estimates:")
    print(hr_df.head())

    # Plot raw signal
    plt.figure(figsize=(12, 4))
    plt.plot(t[:500], raw_ppg[:500])
    plt.title("Raw Synthetic PPG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("day1_raw_synthetic_ppg.png", dpi=200)
    plt.close()

    # Plot filtered signal with peaks
    plt.figure(figsize=(12, 4))
    plt.plot(t[:500], filtered_ppg[:500], label="Filtered PPG")

    visible_peak_mask = t[peaks] < 20
    plt.plot(
        t[peaks][visible_peak_mask],
        filtered_ppg[peaks][visible_peak_mask],
        "o",
        label="Detected Peaks",
    )

    plt.title(f"Filtered PPG with Peaks: {bpm_peak:.2f} BPM")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("day1_filtered_ppg_peaks.png", dpi=200)
    plt.close()

    # Plot Welch PSD
    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs, psd)
    plt.axvline(bpm_welch / 60, linestyle="--", label=f"Peak = {bpm_welch:.2f} BPM")
    plt.title("Welch PSD of Filtered PPG")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("day1_welch_psd.png", dpi=200)
    plt.close()

    # Plot sliding HR
    plt.figure(figsize=(10, 4))
    plt.plot(hr_df["end_sec"], hr_df["bpm"], marker="o")
    plt.title("Real-Time Style Sliding-Window HR Estimate")
    plt.xlabel("Time (s)")
    plt.ylabel("BPM")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("day1_sliding_window_hr.png", dpi=200)
    plt.close()

    # Optional real BIDMC loading
    record, bidmc_df = try_load_bidmc_record()

    if bidmc_df is not None:
        print("\nBIDMC dataframe head:")
        print(bidmc_df.head())
        print("Available columns:", bidmc_df.columns.tolist())

        possible_ppg_cols = [
            c for c in bidmc_df.columns
            if "PLETH" in c.upper() or "PPG" in c.upper()
        ]

        if possible_ppg_cols:
            real_ppg_col = possible_ppg_cols[0]
            real_fs = int(record.fs)
            real_ppg = bidmc_df[real_ppg_col].values

            real_ppg_filtered = bandpass_filter(detrend(real_ppg[: real_fs * 120]), real_fs)
            real_bpm, _ = estimate_hr_from_peaks(real_ppg_filtered, real_fs)

            print(f"Real BIDMC PPG column used: {real_ppg_col}")
            print(f"Estimated HR from real PPG: {real_bpm:.2f} BPM")

            tt = np.arange(len(real_ppg_filtered)) / real_fs
            plt.figure(figsize=(12, 4))
            plt.plot(tt[: real_fs * 20], real_ppg_filtered[: real_fs * 20])
            plt.title("Filtered Real BIDMC PPG Segment")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("day1_real_bidmc_ppg.png", dpi=200)
            plt.close()


# ============================================================
# DAY 2 — ML + STATISTICS
# ============================================================

def create_synthetic_glucose_dataset(n=1000):
    """
    Synthetic dataset for glucose prediction practice.
    This is not medical-grade data.
    """
    age = np.random.randint(18, 75, n)
    bmi = np.random.normal(27, 5, n).clip(16, 45)
    hr_mean = np.random.normal(78, 12, n).clip(45, 140)
    spo2 = np.random.normal(97, 1.5, n).clip(88, 100)
    ppg_amp = np.random.normal(1.0, 0.25, n).clip(0.2, 2.5)
    ppg_hrv_proxy = np.random.normal(55, 20, n).clip(5, 160)
    motion_level = np.random.exponential(0.5, n).clip(0, 5)
    skin_temp = np.random.normal(33, 1.2, n).clip(28, 38)

    glucose = (
        85
        + 0.75 * bmi
        + 0.25 * age
        + 0.12 * hr_mean
        - 0.9 * spo2
        - 4.0 * ppg_amp
        - 0.03 * ppg_hrv_proxy
        + 6.5 * motion_level
        + 1.5 * skin_temp
        + np.random.normal(0, 12, n)
    )

    glucose = glucose.clip(60, 300)

    df = pd.DataFrame(
        {
            "age": age,
            "bmi": bmi,
            "hr_mean": hr_mean,
            "spo2": spo2,
            "ppg_amp": ppg_amp,
            "ppg_hrv_proxy": ppg_hrv_proxy,
            "motion_level": motion_level,
            "skin_temp": skin_temp,
            "glucose_mg_dl": glucose,
        }
    )

    df["hyperglycemia_label"] = (df["glucose_mg_dl"] >= 180).astype(int)
    return df


def confidence_interval_mean(data, confidence=0.95):
    """
    95% confidence interval for mean using normal approximation.
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    z = 1.96 if confidence == 0.95 else 1.96
    margin = z * std / np.sqrt(n)

    return mean, mean - margin, mean + margin


def compute_ppv_npv(sensitivity, specificity, prevalence):
    """
    sensitivity = P(test positive | disease)
    specificity = P(test negative | no disease)
    prevalence = P(disease)
    """
    p_disease = prevalence
    p_no_disease = 1 - prevalence

    true_positive = sensitivity * p_disease
    false_positive = (1 - specificity) * p_no_disease

    true_negative = specificity * p_no_disease
    false_negative = (1 - sensitivity) * p_disease

    ppv = true_positive / (true_positive + false_positive)
    npv = true_negative / (true_negative + false_negative)

    return ppv, npv


def run_day2_ml_statistics():
    print_section("DAY 2 — ML + STATISTICS")

    glucose_df = create_synthetic_glucose_dataset()
    glucose_df.to_csv("day2_synthetic_glucose_dataset.csv", index=False)

    print("\nSynthetic glucose dataset head:")
    print(glucose_df.head())

    features = [
        "age",
        "bmi",
        "hr_mean",
        "spo2",
        "ppg_amp",
        "ppg_hrv_proxy",
        "motion_level",
        "skin_temp",
    ]

    # Regression
    X = glucose_df[features]
    y = glucose_df["glucose_mg_dl"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
    )

    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print("\nGlucose Regression Results")
    print(f"MAE  : {mae:.2f} mg/dL")
    print(f"RMSE : {rmse:.2f} mg/dL")
    print(f"R2   : {r2:.3f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred, alpha=0.6)
    plt.plot([60, 300], [60, 300], linestyle="--")
    plt.xlabel("True Glucose")
    plt.ylabel("Predicted Glucose")
    plt.title("Synthetic Glucose Prediction")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("day2_glucose_regression.png", dpi=200)
    plt.close()

    # Classification
    y_cls = glucose_df["hyperglycemia_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_cls,
        test_size=0.2,
        random_state=42,
        stratify=y_cls,
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
    )

    clf.fit(X_train, y_train)
    cls_pred = clf.predict(X_test)

    print("\nHyperglycemia Classification Results")
    print("Accuracy:", accuracy_score(y_test, cls_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, cls_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, cls_pred))

    # Confidence interval
    mean_glucose, ci_low, ci_high = confidence_interval_mean(glucose_df["glucose_mg_dl"])
    print("\n95% CI for mean glucose:")
    print(f"Mean = {mean_glucose:.2f}, CI = [{ci_low:.2f}, {ci_high:.2f}] mg/dL")

    # Bayes theorem
    sensitivity = 0.95
    specificity = 0.90
    prevalence = 0.02

    ppv, npv = compute_ppv_npv(sensitivity, specificity, prevalence)

    print("\nBayes Medical Test Example")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Prevalence : {prevalence}")
    print(f"PPV        : {ppv:.4f} = {ppv * 100:.2f}%")
    print(f"NPV        : {npv:.4f} = {npv * 100:.2f}%")


# ============================================================
# DAY 3 — TINY RAG / RETRIEVAL SYSTEM
# ============================================================

def get_reference_docs():
    docs = [
        {
            "title": "PPG Heart Rate Extraction",
            "text": """
            Photoplethysmography or PPG measures blood volume changes optically.
            For heart-rate estimation, a common signal-processing pipeline is detrending,
            bandpass filtering around 0.5 to 4 Hz, peak detection, and conversion of
            inter-peak intervals into beats per minute.
            """,
        },
        {
            "title": "SpO2 Estimation",
            "text": """
            Pulse oximetry estimates oxygen saturation using red and infrared light.
            The ratio of pulsatile absorption at two wavelengths is used to estimate
            arterial oxygen saturation. Motion artifacts and poor contact can reduce accuracy.
            """,
        },
        {
            "title": "Glucose Prediction",
            "text": """
            Non-invasive glucose prediction from wearable signals is challenging because
            PPG is affected by blood volume, skin tone, motion, temperature, hydration,
            and sensor contact. Models require calibration, validation, and careful
            monitoring for drift.
            """,
        },
        {
            "title": "Healthcare AI Safety",
            "text": """
            Healthcare AI systems should include guardrails, uncertainty estimation,
            audit logs, human escalation, privacy protection, and clear boundaries
            between educational information and medical advice.
            """,
        },
        {
            "title": "RAG System",
            "text": """
            Retrieval augmented generation uses document chunking, embeddings,
            vector search, retrieval, and grounded answer generation. RAG can reduce
            hallucination by forcing the model to answer from retrieved trusted sources.
            """,
        },
    ]

    return docs


def build_tfidf_retriever(docs):
    texts = [d["text"] for d in docs]
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(texts)
    return vectorizer, doc_vectors


def retrieve_tfidf(query, docs, vectorizer, doc_vectors, top_k=3):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, doc_vectors).flatten()
    top_idx = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_idx:
        results.append(
            {
                "title": docs[idx]["title"],
                "text": docs[idx]["text"],
                "score": float(scores[idx]),
            }
        )

    return results


def try_sentence_transformer_retrieval(query, docs, top_k=3):
    """
    Uses all-MiniLM-L6-v2 embeddings.
    Falls back to TF-IDF if model download fails.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        texts = [d["text"] for d in docs]

        doc_emb = model.encode(texts, normalize_embeddings=True)
        query_emb = model.encode([query], normalize_embeddings=True)

        scores = np.dot(doc_emb, query_emb[0])
        top_idx = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_idx:
            results.append(
                {
                    "title": docs[idx]["title"],
                    "text": docs[idx]["text"],
                    "score": float(scores[idx]),
                }
            )

        return results

    except Exception as exc:
        print("SentenceTransformer failed, using TF-IDF instead.")
        print("Reason:", exc)

        vectorizer, doc_vectors = build_tfidf_retriever(docs)
        return retrieve_tfidf(query, docs, vectorizer, doc_vectors, top_k=top_k)


def generate_grounded_answer(query, retrieved_docs):
    """
    Simple template-based grounded answer.
    In real RAG, retrieved chunks would be passed to an LLM.
    """
    context = "\n\n".join(
        [f"Source: {r['title']}\n{r['text'].strip()}" for r in retrieved_docs]
    )

    answer = f"""
Question:
{query}

Grounded Answer:
Based on the retrieved reference material, a safe healthcare AI system should:

1. Use trusted medical or physiological documents as retrieval sources.
2. Retrieve relevant chunks before answering.
3. Clearly separate educational guidance from medical advice.
4. Add safety guardrails for high-risk glucose or SpO2 cases.
5. Log user input, retrieved sources, model output, timestamp, and system confidence.
6. Escalate uncertain or high-risk cases to a clinician or emergency workflow.
7. Monitor data drift, model drift, and hallucination risk over time.

Retrieved Context Used:
{context}
"""
    return answer


def run_day3_rag():
    print_section("DAY 3 — MINI RAG / RETRIEVAL SYSTEM")

    docs = get_reference_docs()

    query_1 = "How do I extract heart rate from a noisy PPG signal?"
    vectorizer, doc_vectors = build_tfidf_retriever(docs)
    tfidf_results = retrieve_tfidf(query_1, docs, vectorizer, doc_vectors)

    print("\nTF-IDF Retrieval Results")
    for r in tfidf_results:
        print("\nTitle:", r["title"])
        print("Score:", round(r["score"], 3))
        print("Text:", r["text"].strip()[:300])

    query_2 = "Design a safe AI assistant for diabetes and glucose monitoring."
    semantic_results = try_sentence_transformer_retrieval(query_2, docs)

    print("\nSemantic Retrieval Results")
    for r in semantic_results:
        print("\nTitle:", r["title"])
        print("Score:", round(r["score"], 3))
        print("Text:", r["text"].strip()[:300])

    rag_answer = generate_grounded_answer(query_2, semantic_results)

    with open("day3_rag_answer.txt", "w", encoding="utf-8") as f:
        f.write(rag_answer)

    print("\nGenerated grounded answer saved to: day3_rag_answer.txt")
    print(rag_answer)


# ============================================================

# ============================================================
# Main
# ============================================================

def main():
    run_day1_signal_processing()
    run_day2_ml_statistics()
    run_day3_rag()
    print_mock_questions()

    print("\nGenerated output files:")
    output_files = [
        "day1_raw_synthetic_ppg.png",
        "day1_filtered_ppg_peaks.png",
        "day1_welch_psd.png",
        "day1_sliding_window_hr.png",
        "day2_synthetic_glucose_dataset.csv",
        "day2_glucose_regression.png",
        "day3_rag_answer.txt",
    ]

    for file in output_files:
        print("-", file)

    print("\nDone.")


if __name__ == "__main__":
    main()
