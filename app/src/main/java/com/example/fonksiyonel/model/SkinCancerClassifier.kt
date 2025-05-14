package com.example.fonksiyonel.model

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import android.graphics.ImageDecoder
import android.os.Build
import android.provider.MediaStore
import java.io.IOException
import kotlin.random.Random

class SkinCancerClassifier(private val context: Context) {
    private var interpreter: Interpreter? = null
    private val modelName = "my_model.tflite"
    private val inputSize = 224 // Assuming the model expects 224x224 images
    private val numClasses = 4  // Assuming 4 classes: benign, melanoma, basal cell, squamous cell
    
    // Bu değişken test modunu açıp kapatmak için kullanılacak
    private val useTestMode = true  // Varsayılan olarak test modunu açık bırakıyoruz

    init {
        try {
            val model = loadModelFile()
            val options = Interpreter.Options()
            interpreter = Interpreter(model, options)
            Log.d(TAG, "Model loaded successfully")
        } catch (e: IOException) {
            Log.e(TAG, "Error loading model", e)
        }
    }

    @Throws(IOException::class)
    private fun loadModelFile(): MappedByteBuffer {
        val assetManager = context.assets
        val fileDescriptor = assetManager.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun classifyImage(uri: Uri): DiagnosisResult {
        // Test modunda rastgele tahmin üret
        if (useTestMode) {
            return generateRandomPrediction()
        }
        
        // Load and preprocess the image
        val bitmap = loadAndResizeBitmap(uri, inputSize, inputSize)
        
        // Prepare input buffer
        val inputBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        // Normalize pixel values to [0, 1]
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        
        for (pixelValue in pixels) {
            // Extract RGB values
            val r = (pixelValue shr 16 and 0xFF) / 255.0f
            val g = (pixelValue shr 8 and 0xFF) / 255.0f
            val b = (pixelValue and 0xFF) / 255.0f
            
            // TensorFlow Lite expects RGB values
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }
        
        // Prepare output buffer
        val outputBuffer = Array(1) { FloatArray(numClasses) }
        
        // Run inference
        interpreter?.run(inputBuffer, outputBuffer)
        
        // Process results
        val result = outputBuffer[0]
        
        // Log all probabilities for debugging
        Log.d(TAG, "Model output probabilities:")
        Log.d(TAG, "BENIGN: ${result[0]}")
        Log.d(TAG, "MELANOMA: ${result[1]}")
        Log.d(TAG, "BASAL_CELL_CARCINOMA: ${result[2]}")
        Log.d(TAG, "SQUAMOUS_CELL_CARCINOMA: ${result[3]}")
        
        // Find the class with highest probability
        var maxIndex = 0
        var maxProb = result[0]
        
        for (i in 1 until numClasses) {
            if (result[i] > maxProb) {
                maxProb = result[i]
                maxIndex = i
            }
        }
        
        Log.d(TAG, "Selected class index: $maxIndex with probability: $maxProb")
        
        // Map the index to cancer type
        val cancerType = when (maxIndex) {
            0 -> CancerType.BENIGN
            1 -> CancerType.MELANOMA
            2 -> CancerType.BASAL_CELL_CARCINOMA
            3 -> CancerType.SQUAMOUS_CELL_CARCINOMA
            else -> CancerType.BENIGN
        }
        
        // Determine risk level based on cancer type and confidence
        val riskLevel = when {
            cancerType == CancerType.BENIGN -> RiskLevel.LOW
            maxProb > 0.9f -> RiskLevel.VERY_HIGH
            maxProb > 0.7f -> RiskLevel.HIGH
            maxProb > 0.5f -> RiskLevel.MEDIUM
            else -> RiskLevel.LOW
        }
        
        Log.d(TAG, "Final diagnosis: $cancerType with risk level: $riskLevel")
        
        return DiagnosisResult(
            cancerType = cancerType,
            confidencePercentage = maxProb,
            riskLevel = riskLevel
        )
    }
    
    // Test için rastgele tahmin üreten fonksiyon
    private fun generateRandomPrediction(): DiagnosisResult {
        // Rastgele bir kanser türü seç (0-3 arası)
        val randomTypeIndex = Random.nextInt(0, 4)
        val cancerType = when (randomTypeIndex) {
            0 -> CancerType.BENIGN
            1 -> CancerType.MELANOMA
            2 -> CancerType.BASAL_CELL_CARCINOMA
            3 -> CancerType.SQUAMOUS_CELL_CARCINOMA
            else -> CancerType.UNKNOWN
        }
        
        // Rastgele bir güven oranı oluştur (0.7 - 0.99 arası)
        val confidencePercentage = Random.nextFloat() * 0.29f + 0.7f
        
        // Risk seviyesini belirle
        val riskLevel = when {
            cancerType == CancerType.BENIGN -> RiskLevel.LOW
            confidencePercentage > 0.9f -> RiskLevel.VERY_HIGH
            confidencePercentage > 0.8f -> RiskLevel.HIGH
            else -> RiskLevel.MEDIUM
        }
        
        Log.d(TAG, "TEST MODE: Generated random diagnosis: $cancerType with confidence: $confidencePercentage and risk level: $riskLevel")
        
        return DiagnosisResult(
            cancerType = cancerType,
            confidencePercentage = confidencePercentage,
            riskLevel = riskLevel
        )
    }
    
    private fun loadAndResizeBitmap(uri: Uri, width: Int, height: Int): Bitmap {
        val bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(context.contentResolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.isMutableRequired = true
            }
        } else {
            MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
        }
        
        return Bitmap.createScaledBitmap(bitmap, width, height, true)
    }
    
    fun close() {
        interpreter?.close()
        interpreter = null
    }
    
    companion object {
        private const val TAG = "SkinCancerClassifier"
    }
}
