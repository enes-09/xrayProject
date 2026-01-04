package com.eneskulpu.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class DtoAnalysisResult {
    private String className;   // Örn: Pneumonia
    private double confidence;  // Örn: 0.98
    private String message;     // Hata mesajı veya ek bilgi
}