from .holiday import CustomHolidayFeatureSet, CustomDateFeatureSet

# Used by TimeGrad / TempFlow estimators (gluonts time features).
try:
    from gluonts.time_feature import (
        fourier_time_features_from_frequency,
        lags_for_fourier_time_features_from_frequency,
    )
except ImportError:
    # GluonTS < 0.12 (e.g. 0.9.x): Fourier helpers were named differently.
    from gluonts.time_feature import (
        get_lags_for_frequency,
        time_features_from_frequency_str,
    )

    def fourier_time_features_from_frequency(freq_str: str):
        return time_features_from_frequency_str(freq_str)

    def lags_for_fourier_time_features_from_frequency(freq_str: str):
        return get_lags_for_frequency(freq_str)
