package com.example.Helloworldapp;


import jdk.jfr.Registered;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class helloController {

    @GetMapping("/")
    public  String  greeting() {
         return  " Hello from  elastic beans app";

    }



}
