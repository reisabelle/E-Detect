package com.example.thesis

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
    private val classes = listOf("Bumblefoot", "Fowlpox", "Healthy", "Coryza")

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

                // Enhance image before classification
                val enhancedBitmap = enhanceImageResolution(bitmap)
                classifyImage(enhancedBitmap)
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
        val modelFile = "model11.tflite" // Ensure this file is correctly named and placed in assets
        val assetFileDescriptor = assets.openFd(modelFile)
        val inputStream = assetFileDescriptor.createInputStream()
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        val mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

        val options = Interpreter.Options()

        return Interpreter(mappedByteBuffer, options)
    }

    // Enhance image resolution by upscaling and sharpening
    private fun enhanceImageResolution(bitmap: Bitmap): Bitmap {
        val upscaleFactor = 2
        val width = bitmap.width * upscaleFactor
        val height = bitmap.height * upscaleFactor

        val highResBitmap = Bitmap.createScaledBitmap(bitmap, width, height, true)

        return sharpenImage(highResBitmap)
    }

    // Sharpen the upscaled image
    private fun sharpenImage(bitmap: Bitmap): Bitmap {
        val sharpenKernel = floatArrayOf(
            0f, -1f, 0f,
            -1f, 5f, -1f,
            0f, -1f, 0f
        )
        val rs = android.renderscript.RenderScript.create(this)
        val convolveFilter = android.renderscript.ScriptIntrinsicConvolve3x3.create(
            rs, android.renderscript.Element.U8_4(rs)
        )
        val inputAllocation = android.renderscript.Allocation.createFromBitmap(rs, bitmap)
        val outputBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, bitmap.config)
        val outputAllocation = android.renderscript.Allocation.createFromBitmap(rs, outputBitmap)
        convolveFilter.setInput(inputAllocation)
        convolveFilter.setCoefficients(sharpenKernel)
        convolveFilter.forEach(outputAllocation)
        outputAllocation.copyTo(outputBitmap)

        return outputBitmap
    }

    // Preprocess the image for quantized models (8-bit integer values)
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val inputSize = 224
        val byteBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3)
        byteBuffer.order(java.nio.ByteOrder.nativeOrder())

        // Resize with bicubic interpolation for better quality
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        // Normalize the pixel values and prepare buffer
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
            val red = (pixel shr 16) and 0xFF
            val green = (pixel shr 8) and 0xFF
            val blue = pixel and 0xFF

            // Put the pixel values directly as bytes (0-255 range for quantized model)
            byteBuffer.put(red.toByte())
            byteBuffer.put(green.toByte())
            byteBuffer.put(blue.toByte())
        }

        return byteBuffer
    }

    // Function to classify the image and get confidence levels
    private fun classifyImage(bitmap: Bitmap) {
        val tfliteInterpreter by lazy { loadModel() }

        val input = preprocessImage(bitmap)

        // Define output shape based on your model (quantized output)
        val output = Array(1) { ByteArray(4) } // Assuming 4 classes

        // Run the model inference
        try {
            tfliteInterpreter.run(input, output)
        } catch (e: Exception) {
            Log.e("ModelInference", "Error during model inference: ${e.message}")
            diseaseTextView.text = "Error during classification"
            confidenceTextView.text = "Please try again"
            return
        }

        // Dequantize the output based on correct model parameters (adjust scale and zero point)
        val outputScale = 1.0f / 255.0f // (0.00392157f) Example scale (1.0 / 255.0)
        val outputZeroPoint = 0 // Example zero point

        // Convert byte output to float confidence scores
        val confidences = output[0].map { (it.toInt() and 0xFF) * outputScale + outputZeroPoint }

        // Get the class with the highest confidence
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
