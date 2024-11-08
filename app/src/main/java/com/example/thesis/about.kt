package com.example.thesis

import android.os.Bundle
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class about : AppCompatActivity(){
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.about)

        val info = findViewById<TextView>(R.id.aboutTxt)

        info.text = "E-Detect is a mobile application designed to assist poultry farmers, veterinarians, and hobbyists in identifying common poultry diseases quickly and accurately. It utilizes advance image classification which enables users to identify three common diseases in chickens—Bumblefoot, Fowlpox, and Coryza—alongside detecting healthy chickens.\n" +
                "\nSimply open the app and take a photo of your chicken, or upload a pre-existing image. Tap on the displayed image, and this application will process it to assess the chicken’s health. Within seconds, you’ll receive the result, showing the identified condition and a confidence level for the diagnosis."

    }
}