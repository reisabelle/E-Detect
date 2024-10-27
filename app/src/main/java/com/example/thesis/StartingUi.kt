package com.example.thesis


import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.MediaScannerConnection
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class StartingUi : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var textView: TextView
    private lateinit var textView2: TextView
    private val REQUEST_CODE_CAMERA = 1
    private val REQUEST_CODE_GALLERY = 2

    private val confidenceThreshold = 0.5f
    private val classes = listOf("Chicken", "Other Objects")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        textView = findViewById(R.id.textView)
        textView2 = findViewById(R.id.textView2)
        imageView = findViewById(R.id.imageView)

        val cameraButton = findViewById<Button>(R.id.camera)
        val galleryButton = findViewById<Button>(R.id.gallery)

        cameraButton.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED &&
                ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                openCamera()
            } else {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE), REQUEST_CODE_CAMERA)
            }
        }

        galleryButton.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                openGallery()
            } else {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE), REQUEST_CODE_GALLERY)
            }
        }

        imageView.isClickable = imageView.drawable != null
        if (imageView.drawable == null){
            Toast.makeText(this, "No Image Displayed", Toast.LENGTH_SHORT).show()
        }
    }

    private fun openCamera() {
        val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        startActivityForResult(cameraIntent, REQUEST_CODE_CAMERA)
    }

    private fun openGallery() {
        val galleryIntent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(galleryIntent, REQUEST_CODE_GALLERY)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_CAMERA && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            openCamera()
        } else if (requestCode == REQUEST_CODE_GALLERY && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            openGallery()
        } else {
            Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == Activity.RESULT_OK) {
            when (requestCode) {
                REQUEST_CODE_CAMERA -> {
                    val imageBitmap = data?.extras?.get("data") as Bitmap
                    imageView.setImageBitmap(imageBitmap)
                    saveImageToStorage(imageBitmap)
                    setImage(imageBitmap)

                    val enhancedBitmap = enhanceImageResolution(imageBitmap)
                    classifyImage(enhancedBitmap)
                }
                REQUEST_CODE_GALLERY -> {
                    val selectedImage: Uri? = data?.data
                    imageView.setImageURI(selectedImage)
                    setImage(selectedImage)

                    val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, selectedImage)
                    val enhancedBitmap = enhanceImageResolution(bitmap)
                    classifyImage(enhancedBitmap)
                }
            }
        }
    }

    // Load the TFLite model (GPU delegate is now optional)
    private fun loadModel(): Interpreter {
        val modelFile = "modelc_300e-1^-6.tflite" // Ensure this file is correctly named and placed in assets
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
        val output = Array(1) { ByteArray(2) } // Assuming 4 classes, adjust based on your actual model

        // Run the model inference
        try {
            tfliteInterpreter.run(input, output)
        } catch (e: Exception) {
            Log.e("ModelInference", "Error during model inference: ${e.message}")
            return
        }

        // Dequantize the output based on the correct model parameters
        val outputScale = 0.00392157f // Example scale (1.0 / 255.0) for quantized models
        val outputZeroPoint = 0 // Zero point is usually 0 for quantized models

        // Convert byte output to float confidence scores
        val confidences = output[0].map { (it.toInt() and 0xFF) * outputScale + outputZeroPoint }

        // Get the class with the highest confidence
        val maxPos = confidences.indices.maxByOrNull { confidences[it] } ?: -1
        val maxConfidence = confidences[maxPos]

        // Check if the max confidence is above the threshold
        if (maxPos != -1 && maxConfidence > confidenceThreshold) {
            // If "Chicken" is detected
            if (maxPos == classes.indexOf("Chicken")) {
                textView.text = "Chicken Detected!"
                textView2.text = "Click image to see health status"
            } else {
                // If another object is detected
                textView.text = "No Chicken Found"
                imageView.setOnClickListener(null)  // Prevent navigation if no chicken is detected
            }
        } else {
            // If confidence is too low
            textView.text = "Confidence too low\nNo Chicken Found"
            imageView.setOnClickListener(null)
        }
    }


    private fun setImage(image: Any?) {
        when (image) {
            is Bitmap -> {
                imageView.setImageBitmap(image)
                imageView.isClickable = true
                if (image != null && imageView.drawable != null) {
                    Toast.makeText(this, "Image displayed!", Toast.LENGTH_SHORT).show()
                }
            }
            is Uri? -> {
                if (image != null) {
                    imageView.setImageURI(image)
                    imageView.isClickable = imageView.drawable != null
                    if (image != null && imageView.drawable != null) {
                        Toast.makeText(this, "Image displayed!", Toast.LENGTH_SHORT).show()
                    }
                }
            }
            else -> {
                // Handle unexpected image type
                Toast.makeText(this, "Invalid image type", Toast.LENGTH_SHORT).show()
            }
        }
        imageView.setOnClickListener{
            val intent = Intent(this, status::class.java)
            when (image) {
                is Bitmap -> {
                    val bitmapUri = saveImageToCache(image)  // Save to cache and get URI
                    intent.putExtra("imageUri", bitmapUri.toString())
                }
                is Uri? -> {
                    if (image != null) {
                        intent.putExtra("imageUri", image.toString())
                    }
                }
            }
            startActivity(intent)
        }
    }

    private fun saveImageToCache(bitmap: Bitmap): Uri {
        val cachePath = File(cacheDir, "images")
        cachePath.mkdirs() // Create the directory if it doesn't exist
        val file = File(cachePath, "image.png")
        val fileOutputStream = FileOutputStream(file)
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, fileOutputStream)
        fileOutputStream.flush()
        fileOutputStream.close()
        return Uri.fromFile(file)
    }

    private fun saveImageToStorage(bitmap: Bitmap) {
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val fileName = "IMG_$timeStamp.jpg"

        // Save to DCIM/Camera directory
        val cameraDir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM), "Camera")
        if (!cameraDir.exists()) {
            cameraDir.mkdirs()
        }
        val file = File(cameraDir, fileName)

        try {
            val fileOutputStream = FileOutputStream(file)
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fileOutputStream)
            fileOutputStream.flush()
            fileOutputStream.close()

            // Notify the media scanner to make the image available in the gallery
            MediaScannerConnection.scanFile(this, arrayOf(file.toString()), null) { path, uri ->
                runOnUiThread {
                    Toast.makeText(this, "Image saved: $path", Toast.LENGTH_LONG).show()
                }
            }
        } catch (e: IOException) {
            e.printStackTrace()
            Toast.makeText(this, "Failed to save image", Toast.LENGTH_SHORT).show()
        }
    }
}
