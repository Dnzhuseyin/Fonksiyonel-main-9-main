package com.example.fonksiyonel.model

data class DiagnosisResult(
    val cancerType: CancerType,
    val confidencePercentage: Float,
    val riskLevel: RiskLevel
)

enum class CancerType {
    BENIGN,
    MELANOMA,
    BASAL_CELL_CARCINOMA,
    SQUAMOUS_CELL_CARCINOMA
}

enum class RiskLevel {
    LOW,
    MEDIUM,
    HIGH,
    VERY_HIGH
}
