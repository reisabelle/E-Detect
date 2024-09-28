package com.example.e_detect

import android.content.Context
import android.content.Intent
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class status : AppCompatActivity() {
    private lateinit var backbtn: ImageView
    private lateinit var imageView: ImageView
    private lateinit var diseaseTextView: TextView
    private lateinit var confidenceTextView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_status)

        imageView = findViewById(R.id.imageView2) // ImageView in your layout
        diseaseTextView = findViewById(R.id.disease) // TextView for classification result
        confidenceTextView = findViewById(R.id.confidence) // TextView for confidence

        // Get the image URI from the intent
        val imageUri = intent.getStringExtra("imageUri")
        if (imageUri != null) {
            val uri = Uri.parse(imageUri)
            val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)

            // Display the image in the ImageView
            imageView.setImageBitmap(bitmap)

            // Preprocess and classify the image
            val (result, confidence) = classifyImage(bitmap)
            diseaseTextView.text = result // Display the classification result in the TextView
            confidenceTextView.text = confidence // Display confidence levels
        }

        backbtn = findViewById(R.id.backbtn)
        backbtn.setOnClickListener {
            val intent = Intent(this, StartingUi::class.java)
            startActivity(intent)
        }
    }

    // Load the TFLite model
    fun loadModelFile(context: Context): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = context.assets.openFd("model(1).tflite")
        val inputStream = fileDescriptor.createInputStream()
        val fileChannel: FileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // Preprocess the image before feeding into the model
    fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val inputSize = 224 // Updated input size for your model
        val byteBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4) // float32 input
        byteBuffer.order(java.nio.ByteOrder.nativeOrder())

        // Resize the image to the input size
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false)
        val intValues = IntArray(inputSize * inputSize)
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.width, 0, 0, scaledBitmap.width, scaledBitmap.height)

        // Convert pixel values to float and store in byteBuffer
        for (pixel in intValues) {
            byteBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // Red
            byteBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // Green
            byteBuffer.putFloat((pixel and 0xFF) / 255.0f)          // Blue
        }

        return byteBuffer
    }

    // Function to classify the image and get confidence levels
    fun classifyImage(bitmap: Bitmap): Pair<String, String> {
        val tfliteInterpreter: Interpreter by lazy {
            Interpreter(loadModelFile(this))
        }

        // Preprocess the image
        val input = preprocessImage(bitmap)

        // Define output shape based on your model
        val output = Array(1) { FloatArray(5) } // Assuming 5 classes: Bumblefoot, Fowlpox, etc.

        // Run the model inference
        tfliteInterpreter.run(input, output)

        // Get confidence scores and find the max score
        val confidences = output[0]
        val maxPos = confidences.indices.maxByOrNull { confidences[it] } ?: -1
        val maxConfidence = confidences[maxPos]

        // Define class labels (update these based on your model's classes)
        val classes = arrayOf("Bumblefoot", "Fowlpox", "Coryza", "CRD", "Healthy")

        // Create confidence string to display
        val confidenceText = StringBuilder()
        for (i in classes.indices) {
            confidenceText.append("${classes[i]}: %.1f%%\n".format(confidences[i] * 100))
        }

        // Return the class label with the highest confidence and the confidence string
        return if (maxPos != -1) {
            Pair(classes[maxPos], confidenceText.toString())
        } else {
            Pair("Unknown", "Confidence not available")
        }
    }
}
