package com.eneskulpu.starter;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;

@SpringBootApplication
// DİKKAT: Diğer paketlerin (service, controller vb.) taranması için bunu ekledik:
@ComponentScan(basePackages = "com.eneskulpu")
public class XRayProjectApplication {

    public static void main(String[] args) {
        SpringApplication.run(XRayProjectApplication.class, args);
    }

}
