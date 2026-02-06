package com.eneskulpu.controller;

import com.eneskulpu.dto.DtoAnalysisResult;
import org.springframework.http.ResponseEntity;
import org.springframework.web.multipart.MultipartFile;

public interface IXRayController {
    ResponseEntity<DtoAnalysisResult> analyzeImage(MultipartFile file, String modelName);
}