package com.eneskulpu.service.impl;

import com.eneskulpu.dto.DtoAnalysisResult;
import com.eneskulpu.service.IAIService;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Service
public class AIServiceImpl implements IAIService {

    private final RestTemplate restTemplate;

    // application.properties dosyasından URL'i çekecek
    @Value("${ai.service.url}")
    private String aiApiUrl;

    public AIServiceImpl(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    @Override
    public DtoAnalysisResult analyzeXRay(MultipartFile file) throws IOException {
        // Header ayarı (Multipart Form Data)
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        // Dosyayı byte'a çevirip Resource yapıyoruz
        ByteArrayResource fileResource = new ByteArrayResource(file.getBytes()) {
            @Override
            public String getFilename() {
                return file.getOriginalFilename();
            }
        };

        // Body hazırlığı
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", fileResource);

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);

        try {
            // Python API'ye POST isteği at
            ResponseEntity<DtoAnalysisResult> response = restTemplate.postForEntity(
                    aiApiUrl,
                    requestEntity,
                    DtoAnalysisResult.class
            );
            return response.getBody();
        } catch (Exception e) {
            // Hata olursa boş dönmemek için hata mesajı oluştur
            return new DtoAnalysisResult("HATA", 0.0, "AI Servisine ulaşılamadı: " + e.getMessage());
        }
    }
}