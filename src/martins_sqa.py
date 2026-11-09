
import numpy as np
import pandas as pd

from scipy.stats import pearsonr


def safe_pearsonr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.shape != y.shape:
        raise ValueError("Массивы должны иметь одинаковую форму")

    if x.ndim != 1 or len(x) < 2:
        return np.nan

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return np.nan

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    return float(pearsonr(x, y).statistic)


def extract_complexes(signal, centers, fs, before_ms, after_ms):
    n_before = int(round(before_ms / 1000 * fs))
    n_after = int(round(after_ms / 1000 * fs))

    complexes = []

    for center in centers:
        left = int(center) - n_before
        right = int(center) + n_after + 1

        if left < 0 or right > len(signal):
            raise ValueError("Комплекс выходит за границы сигнала")

        complexes.append(signal[left:right])

    if not complexes:
        raise ValueError("Не найдены сердечные комплексы")

    return np.stack(complexes, axis=0)


def check_rr_plausibility(beat_samples, fs):
    rr_ms = np.diff(beat_samples) / fs * 1000

    if len(rr_ms) == 0:
        raise ValueError("Недостаточно RR-интервалов")

    # Проверка диапазона 333–2000 мс
    range_mask = (rr_ms >= 333) & (rr_ms <= 2000)
    rr_in_range = rr_ms[range_mask]

    valid_mask = np.zeros(len(rr_ms), dtype=bool)

    if len(rr_in_range) > 0:
        median_rr = np.median(rr_in_range)

        median_mask = (
            np.abs(rr_in_range - median_rr)
            <= 0.33 * median_rr
        )

        valid_mask[np.flatnonzero(range_mask)] = median_mask

    rr_valid = rr_ms[valid_mask]
    outlier_count = len(rr_ms) - len(rr_valid)

    if len(rr_valid) >= 2:
        std_rr = float(np.std(rr_valid, ddof=0))
    else:
        std_rr = np.nan

    plausible = (
        len(rr_valid) >= 2
        and 1 <= std_rr <= 200
        and outlier_count <= 2
    )

    return {
        "std_rr_ms": std_rr,
        "outlier_count": int(outlier_count),
        "plausible": bool(plausible)
    }


def classify_martins(r_qrs, r_pqrst, rr_plausible):
    threshold = 0.98

    if not rr_plausible:
        return 4, "ok"

    if r_qrs >= threshold and r_pqrst >= threshold:
        return 1, "ok"

    if r_qrs >= threshold and r_pqrst < threshold:
        return 2, "ok"

    if r_qrs < threshold and r_pqrst < threshold:
        return 3, "ok"

    return None, "undefined_morphology_combination"


def analyze_martins_segment(ecg, fs, beat_samples, start_sec):
    ecg = np.asarray(ecg, dtype=np.float64)
    beat_samples = np.asarray(beat_samples, dtype=int)

    if ecg.ndim != 1 or not np.all(np.isfinite(ecg)):
        raise ValueError("ECG должна быть одномерным конечным сигналом")

    if fs <= 0:
        raise ValueError("Частота дискретизации должна быть положительной")

    if np.any(np.diff(beat_samples) <= 0):
        raise ValueError("Аннотации должны быть строго упорядочены")

    if np.any((beat_samples < 0) | (beat_samples >= len(ecg))):
        raise ValueError("Аннотации выходят за границы ECG")

    start = int(round(start_sec * fs))
    end = start + int(round(15 * fs))

    if start < 0 or end > len(ecg):
        raise ValueError("Невозможно выделить 15-секундный сегмент")

    mask = (beat_samples >= start) & (beat_samples < end)
    beats = beat_samples[mask]

    # Проверяем доступность полного PQRST-комплекса
    before = int(round(0.300 * fs))
    after = int(round(0.500 * fs))

    valid = (
        (beats - before >= 0)
        & (beats + after < len(ecg))
    )

    beats = beats[valid]

    result = {
        "start_s": float(start_sec),
        "end_s": float(start_sec + 15),
        "n_beats": len(beats),
        "r_qrs": np.nan,
        "r_pqrst": np.nan,
        "rr_std_ms": np.nan,
        "rr_outliers": np.nan,
        "rr_plausible": None,
        "quality_class": None,
        "status": "insufficient_beats"
    }

    if len(beats) < 3:
        return result

    qrs = extract_complexes(ecg, beats, fs, 50, 50)
    pqrst = extract_complexes(ecg, beats, fs, 300, 500)

    qrs_template = np.mean(qrs, axis=0)
    pqrst_template = np.mean(pqrst, axis=0)

    qrs_corr = np.array([
        safe_pearsonr(x, qrs_template)
        for x in qrs
    ])

    pqrst_corr = np.array([
        safe_pearsonr(x, pqrst_template)
        for x in pqrst
    ])

    if (
        not np.all(np.isfinite(qrs_corr))
        or not np.all(np.isfinite(pqrst_corr))
    ):
        result["status"] = "invalid_correlation"
        return result

    r_qrs = float(np.mean(qrs_corr))
    r_pqrst = float(np.mean(pqrst_corr))

    rr_result = check_rr_plausibility(beats, fs)

    quality_class, status = classify_martins(
        r_qrs,
        r_pqrst,
        rr_result["plausible"]
    )

    result.update({
        "r_qrs": r_qrs,
        "r_pqrst": r_pqrst,
        "rr_std_ms": rr_result["std_rr_ms"],
        "rr_outliers": rr_result["outlier_count"],
        "rr_plausible": rr_result["plausible"],
        "quality_class": quality_class,
        "status": status
    })

    return result


def analyze_martins_record(ecg, fs, beat_samples):
    segment_samples = int(round(15 * fs))

    results = []

    for start in range(
        0,
        len(ecg) - segment_samples + 1,
        segment_samples
    ):
        result = analyze_martins_segment(
            ecg=ecg,
            fs=fs,
            beat_samples=beat_samples,
            start_sec=start / fs
        )

        results.append(result)

    return pd.DataFrame(results)
