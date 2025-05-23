package com.example.fonksiyonel.model

import java.util.Date

data class Report(
    val id: String = "",
    val userId: String = "",
    val imageUrl: String = "",
    val diagnosisResult: DiagnosisResult? = null,
    val createdAt: Long = Date().time,
    val sharedWithDoctors: List<String> = emptyList(),
    val doctorFeedback: String? = null
)
