package com.eneskulpu.service;

import com.eneskulpu.dto.DtoAnalysisResult;
import org.springframework.web.multipart.MultipartFile;
import java.io.IOException;

public interface IAIService {
    DtoAnalysisResult analyzeXRay(MultipartFile file, String modelName) throws IOException;
}