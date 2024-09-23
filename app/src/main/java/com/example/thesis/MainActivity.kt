package com.example.thesis

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.starting_ui)

        val startButton = findViewById<Button>(R.id.StartButt)

        startButton.setOnClickListener {
            Log.d("MainActivity", "Start button clicked")
            try {
                val intent = Intent(this, StartingUi::class.java)
                startActivity(intent)
            } catch (e: Exception) {
                Log.e("MainActivity", "Error starting activity", e)
            }
        }
    }
}