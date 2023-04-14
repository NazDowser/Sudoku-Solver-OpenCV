package com.ece420.lab7;

import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.Manifest;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.SeekBar;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Size;
import org.opencv.tracking.TrackerKCF;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";

    // UI Variables
    private Button controlButton;

    // Declare OpenCV based camera view base
    private CameraBridgeViewBase mOpenCvCameraView;
    // Camera size
    private int myWidth;
    private int myHeight;

    // Mat to store RGBA and Grayscale camera preview frame
    private Mat mRgba;
    private Mat mGray;
    private Mat imgThresh;

    // KCF Tracker variables
//    private TrackerKCF myTacker;
//    private Rect2d myROI = new Rect2d(0,0,0,0);
//    private int myROIWidth = 70;
//    private int myROIHeight = 70;
//    private Scalar myROIColor = new Scalar(0,0,0);
    private int stateFlag = -1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        super.setRequestedOrientation (ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        // Request User Permission on Camera
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, 1);}

        // OpenCV Loader and Avoid using OpenCV Manager
        if (!OpenCVLoader.initDebug()) {
            Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
        } else {
            Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
        }

        // Setup control button
        controlButton = (Button)findViewById((R.id.controlButton));
        controlButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (stateFlag == -1) {
                    // Modify UI
                    controlButton.setText("STOP");
                    // Modify state flag
                    stateFlag = 0;
                }
                else if(stateFlag == 1){
                    // Modify UI
                    controlButton.setText("START");

                    // Modify state flag
                    stateFlag = -1;
                }
            }
        });

        // Setup OpenCV Camera View
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.opencv_camera_preview);
        // Use main camera with 0 or front camera with 1
        mOpenCvCameraView.setCameraIndex(0);
        // Force camera resolution, ignored since OpenCV automatically select best ones
        // mOpenCvCameraView.setMaxFrameSize(1280, 720);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    // Helper Function to map single integer to color scalar
    // https://www.particleincell.com/2014/colormap/
    public void setColor(int value) {
        double a=(1-(double)value/100)/0.2;
        int X=(int)Math.floor(a);
        int Y=(int)Math.floor(255*(a-X));
        double newColor[] = {0,0,0};
        switch(X)
        {
            case 0:
                // r=255;g=Y;b=0;
                newColor[0] = 255;
                newColor[1] = Y;
                break;
            case 1:
                // r=255-Y;g=255;b=0
                newColor[0] = 255-Y;
                newColor[1] = 255;
                break;
            case 2:
                // r=0;g=255;b=Y
                newColor[1] = 255;
                newColor[2] = Y;
                break;
            case 3:
                // r=0;g=255-Y;b=255
                newColor[1] = 255-Y;
                newColor[2] = 255;
                break;
            case 4:
                // r=Y;g=0;b=255
                newColor[0] = Y;
                newColor[2] = 255;
                break;
            case 5:
                // r=255;g=0;b=255
                newColor[0] = 255;
                newColor[2] = 255;
                break;
        }
//        myROIColor.set(newColor);
        return;
    }

    public void filtering(Mat img) {
        Mat imgMedian = new Mat(img.rows(), img.cols(), img.type());
        Mat imgGaussian = new Mat(img.rows(), img.cols(), img.type());
//        Mat imgThresh = new Mat(img.rows(), img.cols(), img.type());
        //Applying GaussianBlur on the Image
        Imgproc.medianBlur(img, imgMedian,1);
        Imgproc.GaussianBlur(imgMedian, imgGaussian, new Size(3, 3), 0);
        Imgproc.adaptiveThreshold(imgGaussian, mGray, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 301, 7);
        imgMedian.release();
        imgGaussian.release();
//        return imgThresh;
    }

    // OpenCV Camera Functionality Code
    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
        myWidth = width;
        myHeight = height;
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
        imgThresh.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // Timer
        long start = Core.getTickCount();
        // Grab camera frame in rgba and grayscale format
        mRgba = inputFrame.rgba();
        // Grab camera frame in gray format
        mGray = inputFrame.gray();

        // Action based on states
        if(stateFlag == -1){
            // stop state
            // do nothing rn
            return mRgba;
        }
        else if(stateFlag == 0){
            // transition state
            // do nothing rn
            stateFlag = 1;
            return mRgba;
        }
        else{
            // filtering state
            Mat imgMedian = new Mat(mGray.rows(), mGray.cols(), mGray.type());
            Mat imgGaussian = new Mat(mGray.rows(), mGray.cols(), mGray.type());
            imgThresh = new Mat(mGray.rows(), mGray.cols(), mGray.type());
            //Applying GaussianBlur on the Image
            Imgproc.medianBlur(mGray, imgMedian,1);
            Imgproc.GaussianBlur(imgMedian, imgGaussian, new Size(3, 3), 0);
            Imgproc.adaptiveThreshold(imgGaussian, imgThresh, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 11, 7);
            imgMedian.release();
            imgGaussian.release();
            boolean pass = true;
            // if cannot locate grid
            if(!pass){
                double fps = Core.getTickFrequency() / (Core.getTickCount()-start);
                Imgproc.putText(mRgba, "FPS@"+fps, new Point(1000, 30), 3, 1, new Scalar(0,0,255), 2);
                Imgproc.putText(mRgba, "Tracking Failure Occurred", new Point(10, 30), 3, 1, new Scalar(0,0,255), 2);
                return mRgba;
            }

            // create FPS readings
            double fps = Core.getTickFrequency() / (Core.getTickCount()-start);
            Imgproc.putText(imgThresh, "FPS@"+fps, new Point(1000, 30), 3, 1, new Scalar(255), 2);
            return imgThresh;
        }
    }
}