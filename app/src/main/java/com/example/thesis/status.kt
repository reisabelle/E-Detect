package com.example.e_detect

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

class status : AppCompatActivity() {
    private lateinit var backButton: ImageView
    private lateinit var imageView: ImageView
    private lateinit var diseaseTextView: TextView
    private lateinit var confidenceTextView: TextView

    private val confidenceThreshold = 0.5f
    private val classes = listOf("Bumblefoot", "Fowlpox", "Coryza", "CRD", "Healthy")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_status)

        imageView = findViewById(R.id.imageView2)
        diseaseTextView = findViewById(R.id.disease)
        confidenceTextView = findViewById(R.id.confidence)

        backButton = findViewById(R.id.backbtn)
        backButton.setOnClickListener {
            val intent = Intent(this, StartingUi::class.java)
            startActivity(intent)
        }

        val imageUri = intent.getStringExtra("imageUri")
        if (imageUri != null) {
            try {
                val uri = Uri.parse(imageUri)
                val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)

                imageView.setImageBitmap(bitmap)
                classifyImage(bitmap)
            } catch (e: IOException) {
                Log.e("ImageClassification", "Error loading image: ${e.message}")
                diseaseTextView.text = "Error loading image"
                confidenceTextView.text = "Please try again"
            } catch (e: Exception) {
                Log.e("ImageClassification", "Unknown error: ${e.message}")
                diseaseTextView.text = "Unknown error"
                confidenceTextView.text = "Please try again"
            }
        }
    }

    // Load the TFLite model (GPU delegate is now optional)
    private fun loadModel(): Interpreter {
        val modelFile = "model(1).tflite" // Ensure this file is correctly named and placed in assets
        val assetFileDescriptor = assets.openFd(modelFile)
        val inputStream = assetFileDescriptor.createInputStream()
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        val mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

        val options = Interpreter.Options()

        // GPU delegate is optional
        try {
            // Temporarily commenting out GPU delegate for testing, as it could cause crashes
            // val gpuDelegate = GpuDelegate()
            // options.addDelegate(gpuDelegate)
        } catch (e: Exception) {
            Log.e("InterpreterOptions", "Error initializing GPU delegate: ${e.message}")
        }

        return Interpreter(mappedByteBuffer, options)
    }

    // Preprocess the image before feeding into the model
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val inputSize = 224
        val byteBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        byteBuffer.order(java.nio.ByteOrder.nativeOrder())

        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false)
        val intValues = IntArray(inputSize * inputSize)
        scaledBitmap.getPixels(
            intValues,
            0,
            scaledBitmap.width,
            0,
            0,
            scaledBitmap.width,
            scaledBitmap.height
        )

        for (pixel in intValues) {
            byteBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // Red
            byteBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // Green
            byteBuffer.putFloat((pixel and 0xFF) / 255.0f)          // Blue
        }

        return byteBuffer
    }

    // Function to classify the image and get confidence levels
    private fun classifyImage(bitmap: Bitmap) {
        val tfliteInterpreter by lazy { loadModel() }

        val input = preprocessImage(bitmap)

        // Define output shape based on your model
        val output = Array(1) { FloatArray(5) } // Assuming 5 classes: Bumblefoot, Fowlpox, etc.

        // Run the model inference
        try {
            tfliteInterpreter.run(input, output)
        } catch (e: Exception) {
            Log.e("ModelInference", "Error during model inference: ${e.message}")
            diseaseTextView.text = "Error during classification"
            confidenceTextView.text = "Please try again"
            return
        }

        // Get confidence scores and find the max score
        val confidences = output[0]
        val maxPos = confidences.indices.maxByOrNull { confidences[it] } ?: -1
        val maxConfidence = confidences[maxPos]

        // Check if the max confidence is above the threshold
        if (maxConfidence > confidenceThreshold) {
            val result = classes[maxPos]
            diseaseTextView.text = "Disease: $result"
            confidenceTextView.text = "Confidence: ${"%.1f".format(maxConfidence * 100)}%"
        } else {
            diseaseTextView.text = "Unknown"
            confidenceTextView.text = "Confidence: Not available"
        }
    }
}
