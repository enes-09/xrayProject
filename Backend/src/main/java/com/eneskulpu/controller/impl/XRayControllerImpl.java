package com.eneskulpu.controller.impl;

import com.eneskulpu.controller.IXRayController;
import com.eneskulpu.dto.DtoAnalysisResult;
import com.eneskulpu.service.IAIService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
@RequestMapping("/api/xray")
@CrossOrigin(origins = "*") // React uygulamasından gelen isteklere izin ver
public class XRayControllerImpl implements IXRayController {

    private final IAIService aiService;

    public XRayControllerImpl(IAIService aiService) {
        this.aiService = aiService;
    }

    @Override
    @PostMapping("/upload")
    public ResponseEntity<DtoAnalysisResult> analyzeImage(@RequestParam("image") MultipartFile file) {
        try {
            DtoAnalysisResult result = aiService.analyzeXRay(file);
            return ResponseEntity.ok(result);
        } catch (IOException e) {
            return ResponseEntity.internalServerError().build();
        }
    }
}