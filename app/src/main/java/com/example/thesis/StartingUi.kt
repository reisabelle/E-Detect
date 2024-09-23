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
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class StartingUi : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private val REQUEST_CODE_CAMERA = 1
    private val REQUEST_CODE_GALLERY = 2

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

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
                }
                REQUEST_CODE_GALLERY -> {
                    val selectedImage: Uri? = data?.data
                    imageView.setImageURI(selectedImage)
                    setImage(selectedImage)
                }
            }
        }
    }

    private fun setImage(image: Any?) {
        when (image) {
            is Bitmap -> {
                imageView.setImageBitmap(image)
                imageView.isClickable = true
                if (image != null && imageView.drawable != null) {
                    Toast.makeText(this, "Image displayed!", Toast.LENGTH_SHORT).show()
                    Toast.makeText(this, "Click image to see health status", Toast.LENGTH_SHORT).show()
                }
            }
            is Uri? -> {
                if (image != null) {
                    imageView.setImageURI(image)
                    imageView.isClickable = imageView.drawable != null
                    if (image != null && imageView.drawable != null) {
                        Toast.makeText(this, "Image displayed!", Toast.LENGTH_SHORT).show()
                        Toast.makeText(this, "Click image to see health status", Toast.LENGTH_SHORT).show()
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
