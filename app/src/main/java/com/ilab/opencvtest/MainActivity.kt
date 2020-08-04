package com.ilab.opencvtest

import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import android.content.DialogInterface
import android.os.Bundle
import android.annotation.TargetApi
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import android.view.SurfaceView
import android.view.WindowManager
import org.opencv.core.Mat
import java.util.Collections
import android.Manifest.permission.CAMERA
import android.Manifest.permission.WRITE_EXTERNAL_STORAGE
import android.app.Activity
import android.content.Intent
import android.content.res.Configuration
import android.graphics.Bitmap
import android.media.FaceDetector
import android.net.Uri
import android.os.Environment
import android.util.Pair
import android.view.View
import android.view.ViewTreeObserver
import android.widget.Toast
import kotlinx.android.synthetic.main.activity_main.*
import org.opencv.android.*
import org.opencv.core.Core
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.lang.Exception
import java.lang.Float

private const val REQUEST_CHOOSE_IMAGE = 1002
private const val BASE_WIDTH = 1280
private const val BASE_HEIGHT = 720

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {
    private var matInput: Mat? = null
    private var matResult: Mat? = null
    private var mOpenCvCameraView: CameraBridgeViewBase? = null
    external fun ConvertRGBtoGray(matAddrInput: Long, matAddrResult: Long)
    external fun loadCascade(cascadeFileName: String): Long
    external fun detect(cascadeClassifierFace: Long, cascadeClassifierEye: Long, matAddrInput: Long, matAddrResult: Long): Int
    private var cascadeClassifierFace: Long = 0
    private var cascadeClassifierEye: Long = 0


    private var imageInput: Mat? = null
    private var imageOutput: Mat? = null

    // Max width (portrait mode)
    private var imageMaxWidth = 0
    // Max height (portrait mode)
    private var imageMaxHeight = 0


    private val isUseCamera = false

    companion object {
        private const val TAG = "opencv"

        //여기서부턴 퍼미션 관련 메소드
        private const val CAMERA_PERMISSION_REQUEST_CODE = 200

        init {
            System.loadLibrary("opencv_java4")
            System.loadLibrary("native-lib")
        }
    }

    private val mLoaderCallback: BaseLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
//                    mOpenCvCameraView!!.enableView()
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)


        window.setFlags(
            WindowManager.LayoutParams.FLAG_FULLSCREEN,
            WindowManager.LayoutParams.FLAG_FULLSCREEN
        )
        window.setFlags(
            WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
            WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
        )


        setContentView(R.layout.activity_main)

        if (isUseCamera) {
            mOpenCvCameraView = findViewById<View>(R.id.activity_surface_view) as CameraBridgeViewBase
            mOpenCvCameraView!!.visibility = SurfaceView.VISIBLE
            mOpenCvCameraView!!.setCvCameraViewListener(this)
            mOpenCvCameraView!!.setCameraIndex(1) // front-camera(1),  back-camera(0)
        }
    }

    public override fun onPause() {
        super.onPause()
        if (mOpenCvCameraView != null) mOpenCvCameraView!!.disableView()
    }

    public override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "onResume :: Internal OpenCV library not found.")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback)
        } else {
            Log.d(TAG, "onResum :: OpenCV library found inside package. Using it!")
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }
    }

    public override fun onDestroy() {
        super.onDestroy()
        if (mOpenCvCameraView != null) mOpenCvCameraView!!.disableView()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {}
    override fun onCameraViewStopped() {}
    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat {
        matInput = inputFrame.rgba()

        matInput.let {
            if (matResult == null) {
                matResult = Mat(it!!.rows(), it!!.cols(), it!!.type())
            }

//            ConvertRGBtoGray(it!!.getNativeObjAddr(), matResult!!.nativeObjAddr)
            Core.flip(matInput, matInput, 1)

            matResult.let { result ->
                val faceCount = detect(cascadeClassifierFace, cascadeClassifierEye,  it!!.nativeObjAddr, result!!.nativeObjAddr)

                if (faceCount > 0) {
                    runOnUiThread {
                        val msg = "face count is $faceCount"
//                        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()

                        println(msg)
                    }
                }
            }

        }

        return matResult as Mat
    }

    protected val cameraViewList: List<CameraBridgeViewBase>
        protected get() = Collections.singletonList(mOpenCvCameraView) as List<CameraBridgeViewBase>

    protected fun onCameraPermissionGranted() {
        val cameraViews = cameraViewList ?: return
        for (cameraBridgeViewBase in cameraViews) {
            cameraBridgeViewBase?.setCameraPermissionGranted()

            readCascadeFile()
        }
    }

    override fun onStart() {
        super.onStart()

        val rootView = root
        rootView.viewTreeObserver.addOnGlobalLayoutListener(
            object : ViewTreeObserver.OnGlobalLayoutListener {
                override fun onGlobalLayout() {
                    rootView.viewTreeObserver.removeOnGlobalLayoutListener(this)
                    imageMaxWidth = rootView.width
                    imageMaxHeight = rootView.height
                }
            })

        var havePermission = true
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(CAMERA) != PackageManager.PERMISSION_GRANTED
                    || checkSelfPermission(WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED ) {
                requestPermissions(arrayOf(CAMERA, WRITE_EXTERNAL_STORAGE), CAMERA_PERMISSION_REQUEST_CODE)
                havePermission = false
            }
        }
        if (havePermission) {

            if (isUseCamera) onCameraPermissionGranted()
            else readCascadeFile()

            select_image_button.setOnClickListener {
                startChooseImageIntentForResult()
            }

            processing_button.setOnClickListener {
                imageProcessingFaceDetect()
            }
        }
    }

    @TargetApi(Build.VERSION_CODES.M)
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE && grantResults.size > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED
                && grantResults[1] == PackageManager.PERMISSION_GRANTED
        ) {
            onCameraPermissionGranted()
        } else {
            showDialogForPermission("앱을 실행하려면 퍼미션을 허가하셔야합니다.")
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }

    @TargetApi(Build.VERSION_CODES.M)
    private fun showDialogForPermission(msg: String) {
        val builder: android.app.AlertDialog.Builder = android.app.AlertDialog.Builder(this@MainActivity)
        builder.setTitle("알림")
        builder.setMessage(msg)
        builder.setCancelable(false)
        builder.setPositiveButton("예", object : DialogInterface.OnClickListener {
            override fun onClick(dialog: DialogInterface?, id: Int) {
                requestPermissions(arrayOf(CAMERA, WRITE_EXTERNAL_STORAGE), CAMERA_PERMISSION_REQUEST_CODE)
            }
        })
        builder.setNegativeButton("아니오", object : DialogInterface.OnClickListener {
            override fun onClick(arg0: DialogInterface?, arg1: Int) {
                finish()
            }
        })
        builder.create().show()
    }

    private fun copyFIle(filename: String) {
        val baseDir = Environment.getExternalStorageDirectory().path
        val pathDir = baseDir + File.separator + filename

        val assetManager = this.assets

        var inputStream: InputStream? = null
        var outputStream: OutputStream? = null

        try {

            inputStream = assetManager.open(filename)
            outputStream = FileOutputStream(pathDir)

            var length: Int
            val arraySize = 1024

            val buffer = ByteArray(arraySize)
            inputStream.use { input ->
                outputStream.use { fileOut ->

                    while (true) {
                        val length = input.read(buffer)
                        if (length <= 0)
                            break
                        fileOut.write(buffer, 0, length)
                    }
                    fileOut.flush()
                    fileOut.close()

                }
            }
            inputStream.close()
        } catch (e: Exception) {

        }
    }

    private fun readCascadeFile() {
        val faceFileName = "haarcascade_frontalface_alt.xml"
        val eyeFileName = "haarcascade_eye_tree_eyeglasses.xml"
        copyFIle(faceFileName)
        copyFIle(eyeFileName)

        cascadeClassifierFace = loadCascade(faceFileName)
        cascadeClassifierEye = loadCascade(eyeFileName)
    }

    private fun test() {


        matInput.let {
            if (matResult == null) {
                matResult = Mat(it!!.rows(), it!!.cols(), it!!.type())
            }

            matResult.let { result ->
                val faceCount = detect(
                    cascadeClassifierFace,
                    cascadeClassifierEye,
                    it!!.nativeObjAddr,
                    result!!.nativeObjAddr
                )
            }
        }
    }

    private fun startChooseImageIntentForResult() {
        val intent = Intent()
        intent.type = "image/*"
        intent.action = Intent.ACTION_GET_CONTENT
        startActivityForResult(
            Intent.createChooser(intent, "Select Picture"),
            REQUEST_CHOOSE_IMAGE
        )
    }

    override fun onActivityResult(
        requestCode: Int,
        resultCode: Int,
        data: Intent?
    ) {
        if (requestCode == REQUEST_CHOOSE_IMAGE && resultCode == Activity.RESULT_OK) {
            // In this case, imageUri is returned by the chooser, save it.
            faceDetectInImage(data?.data)
        } else {
            super.onActivityResult(requestCode, resultCode, data)
        }
    }

    private fun faceDetectInImage(imageUri: Uri?) {
        imageViewInput.setImageURI(imageUri)

        try {
            val imageBitmap = BitmapUtils.getBitmapFromContentUri(contentResolver, imageUri)
                ?: return

            // Get the dimensions of the image view
            val targetedSize = targetedWidthHeight
            // Determine how much to scale down the image
            val scaleFactor = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                Float.max(
                    imageBitmap.width.toFloat() / targetedSize.first.toFloat(),
                    imageBitmap.height.toFloat() / targetedSize.second.toFloat()
                )
            } else {
                TODO("VERSION.SDK_INT < N")
            }

            val resizedBitmap = Bitmap.createScaledBitmap(
                imageBitmap,
                (imageBitmap.width / scaleFactor).toInt(),
                (imageBitmap.height / scaleFactor).toInt(),
                true
            )

            imageInput = Mat()
            val bmp32 = resizedBitmap!!.copy(Bitmap.Config.ARGB_8888, true)
            Utils.bitmapToMat(bmp32, imageInput)

        } catch (e: Exception) {
            println("error ${e.message}")
        }
    }

    private fun imageProcessingFaceDetect() {
        if (imageInput != null) {
            imageInput.let {
                if (imageOutput == null) {
                    imageOutput = Mat(it!!.rows(), it!!.cols(), it!!.type())
                }

                imageOutput.let { result ->
                    val faceCount = detect(cascadeClassifierFace, cascadeClassifierEye,  it!!.nativeObjAddr, result!!.nativeObjAddr)

                    runOnUiThread {
                        val msg = "face count is $faceCount"
                        Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
                        println(msg)
                    }
                }
            }
        }
    }

    private val targetedWidthHeight: Pair<Int, Int>
        get() {
            val targetWidth: Int
            val targetHeight: Int

            targetWidth = imageMaxWidth
            targetHeight = imageMaxHeight

            return Pair(targetWidth, targetHeight)
        }
}
