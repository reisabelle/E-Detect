package com.example.e_detect

import android.content.Intent
import android.os.Bundle
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import android.net.Uri
import android.provider.MediaStore
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import android.content.res.AssetFileDescriptor
import android.content.Context
import android.graphics.Bitmap
import java.io.IOException

class status : AppCompatActivity() {
    private lateinit var backbtn: ImageView
    private lateinit var imageView: ImageView
    private lateinit var diseaseTextView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_status)

        imageView = findViewById(R.id.imageView2) // Link to your ImageView in the layout
        diseaseTextView = findViewById(R.id.disease) // Link to your TextView for displaying the result

        // Get the image URI from the intent
        val imageUri = intent.getStringExtra("imageUri")
        if (imageUri != null) {
            val uri = Uri.parse(imageUri)
            val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)

            // Display the image in the ImageView
            imageView.setImageBitmap(bitmap)

            // Preprocess and classify the image
            val result = classifyImage(bitmap)
            diseaseTextView.text = result // Display the classification result in the TextView
        }

        backbtn = findViewById(R.id.backbtn)
        backbtn.setOnClickListener {
            val intent = Intent(this, StartingUi::class.java)
            startActivity(intent)
        }
    }

    // Load the TFLite model
    fun loadModelFile(context: Context): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = context.assets.openFd("yolov5s-fp16 (1).tflite")
        val inputStream = fileDescriptor.createInputStream()
        val fileChannel: FileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // Preprocess the image before feeding into the model
    fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val inputSize = 640 // Input size for YOLOv5 model
        val byteBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4) // float32 input
        byteBuffer.rewind()

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


    // Function to classify the image


    // Function to classify the image
    fun classifyImage(bitmap: Bitmap): String {
        val tfliteInterpreter: Interpreter by lazy {
            Interpreter(loadModelFile(this))
        }

        // Preprocess the image
        val input = preprocessImage(bitmap)

        // Define output shape based on your model
        val output = Array(1) { Array(25200) { FloatArray(85) } } // Adjust to match your model output shape

        // Run the model inference
        tfliteInterpreter.run(input, output)

        // Process the output to get the index of the highest score for each bounding box
        val classScores = FloatArray(85) // Assuming you have 85 classes
        for (i in output[0].indices) {
            val currentClassScores = output[0][i]
            for (j in currentClassScores.indices) {
                classScores[j] += currentClassScores[j] // Aggregate scores for each class
            }
        }

        // Find the index of the class with the maximum score
        val maxIndex = classScores.indices.maxByOrNull { classScores[it] } ?: -1

        // Define class labels (update this based on your model's classes)
        val classes = arrayOf("Bumblefoot", "Fowlpox", "Coryza", "CRD", "Healthy")

        // Return the class label corresponding to the max index
        return if (maxIndex != -1) {
            classes[maxIndex]
        } else {
            "Unknown"
        }
    }

}
