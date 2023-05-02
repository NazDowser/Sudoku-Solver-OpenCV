package com.ece420.lab7;

import android.app.Activity;
//import android.app.NativeActivity;
import android.content.Context;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
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

import com.ece420.lab7.ml.Classifier;
import com.ece420.lab7.ml.Mnist;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner; // file reader ipehlivan
import java.io.File;
import java.io.FileNotFoundException;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Size;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.tracking.TrackerKCF;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";
    private Context mContext;
    // UI Variables
    private Button controlButton;
    private Button resetButton;
    private TextView sudokuTextView;

    // Declare OpenCV based camera view base
    private CameraBridgeViewBase mOpenCvCameraView;
    // Camera size
    private int myWidth;
    private int myHeight;
    private int[] mySudoku;
    private int[] myAnswer;

    // Mat to store RGBA and Grayscale camera preview frame
    private Mat mRgba;
    private Mat mGray;
    private Mat imgDisplay;
    private int stateFlag = -2;

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

        // Setup control button and textview
        sudokuTextView = (TextView)findViewById(R.id.sudokuTextView);
        controlButton = (Button)findViewById((R.id.controlButton));
        resetButton = (Button)findViewById((R.id.resetButton));
        controlButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (stateFlag == -1) {
                    // Modify UI
                    controlButton.setText("READ");
                    // Modify state flag
                    stateFlag = 0;
                }
                else if(stateFlag == 1){
//                    System.out.println("clicked!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                    // Modify UI
                    controlButton.setText("SOLVE");
                    String text = "Sudoku Text: \n";
                    String displayText = Arrays.toString(mySudoku);
                    displayText = displayText.replace(",","");
                    displayText = displayText.replace("[","");
                    displayText = displayText.replace("]","");
                    text += displayText;
                    sudokuTextView.setText(text);
                    // Modify state flag
                    stateFlag = 2;
                }
                else if(stateFlag == 2) {
                    controlButton.setText("GOT IT");
                    String text = "Sudoku Text: \n";
                    String displayText = Arrays.toString(myAnswer);
                    displayText = displayText.replace(",","");
                    displayText = displayText.replace("[","");
                    displayText = displayText.replace("]","");
                    text += displayText;
                    sudokuTextView.setText(text);
                    stateFlag = 3;
                }
                else if(stateFlag == 3) {
                    // Modify UI
                    controlButton.setText("AGAIN?");

                    // Modify state flag
                    stateFlag = 4;
                }
                else if(stateFlag == 4) {
                    // Modify UI
                    controlButton.setText("START");

                    // Modify state flag
                    stateFlag = -2;
                }
            }
        });
        resetButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                controlButton.setText("START");
                stateFlag = -2;
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
        Mat imgMedian = new Mat(mGray.rows(), mGray.cols(), mGray.type());
        Mat imgGaussian = new Mat(mGray.rows(), mGray.cols(), mGray.type());
        imgDisplay = new Mat(mGray.rows(), mGray.cols(), mGray.type());
        //Applying GaussianBlur on the Image
        Imgproc.medianBlur(mGray, imgMedian,1);
        Imgproc.GaussianBlur(imgMedian, imgGaussian, new Size(3, 3), 0);
        Imgproc.adaptiveThreshold(imgGaussian, imgDisplay, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 11, 7);
        imgMedian.release();
        imgGaussian.release();
        return;
    }

    public void coordinateUnpack(Point[] rect, Point[] sorted) {
        double min = 9999.;
        double max = 0.;
        for(int i=0; i<4; i++){
            double sum = rect[i].x + rect[i].y;
            if(sum < min){
                sorted[0] = rect[i]; // top left
                min = sum;
            }
            if(sum > max) {
                sorted[3] = rect[i];// bottom right
                max = sum;
            }
        }
        min = 9999.;
        max = 0.;
        for(int i=0; i<4; i++){
            double diff = rect[i].y - rect[i].x;
            if(diff < min){
                sorted[1] = rect[i]; //  top right
                min = diff;
            }
            if(diff > max) {
                sorted[2] = rect[i];// bottom left
                max = diff;
            }
        }
        return;
    }

    public void harmongraphyMatrix(Mat img, Point[] coordinates) {
//        double w1 = (int)Math.sqrt(Math.pow(coordinates[0].x - coordinates[1].x, 2) + Math.pow(coordinates[0].y - coordinates[1].y, 2));
//        double w2 = (int)Math.sqrt(Math.pow(coordinates[2].x - coordinates[3].x, 2) + Math.pow(coordinates[2].y - coordinates[3].y, 2));
//        double width = Math.max(w1, w2);
//
//        double h1 = (int)Math.sqrt(Math.pow(coordinates[0].x - coordinates[2].x, 2) + Math.pow(coordinates[0].y - coordinates[2].y, 2));
//        double h2 = (int)Math.sqrt(Math.pow(coordinates[1].x - coordinates[3].x, 2) + Math.pow(coordinates[1].y - coordinates[3].y, 2));
//        double height = Math.max(h1, h2);

//        Point p0 = new Point(0,0);
//        Point p1 = new Point(width -1,0);
//        Point p2 = new Point(0, height -1);
//        Point p3 = new Point(width -1, height -1);

        int temp = (myWidth-myHeight)/2;
        Point p0 = new Point(temp-1,0);
        Point p1 = new Point(temp+myHeight-1,0);
        Point p2 = new Point(temp-1, myHeight -1);
        Point p3 = new Point(temp+myHeight-1, myHeight -1);
        MatOfPoint2f src = new MatOfPoint2f(coordinates[0], coordinates[1], coordinates[2], coordinates[3]);
        MatOfPoint2f dst= new MatOfPoint2f(p0, p1, p2, p3);
        Mat M = Imgproc.getPerspectiveTransform(src, dst);
        Imgproc.warpPerspective(img, img, M, img.size());
    }

    public void correction(Mat img) {
        // find outermost contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(img, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // find the contour we want (the one with largest area)
        double maxArea = 0.0;
        MatOfPoint sudokuContor = contours.get(0);
        for (int i = 0; i < contours.size(); i++) {
            double area = Imgproc.contourArea(contours.get(i));
            if(area > maxArea) {
                maxArea = area;
                sudokuContor = contours.get(i);
            }
        }

        // use mask to clean up the area outside the contour
        Scalar colorTOP = new Scalar(255);
        Scalar colorBOT = new Scalar(0);
        List<MatOfPoint> draw = new ArrayList<>();
        draw.add(sudokuContor);
        Mat mask = new Mat(mGray.rows(), mGray.cols(), mGray.type());
        Imgproc.drawContours(mask, draw, 0, colorTOP, -1);
        Imgproc.drawContours(mask, draw, 0, colorBOT, 2);
        Core.bitwise_and(img, mask, img);

        // use the outermost contour to recover the img
        MatOfPoint2f toApprox = new MatOfPoint2f(sudokuContor.toArray());
        double peri = Imgproc.arcLength(toApprox, true);
        Imgproc.approxPolyDP(toApprox, toApprox,0.01 * peri, true);

        if(toApprox.rows() != 4)
            return;
        Point[] rect = new Point[4];
        Point[] sortedPoints = new Point[4];
        double[] data;
        for(int i=0; i<4; i++){
            data = toApprox.get(i, 0);
            rect[i] = new Point(data[0], data[1]);
        }
        coordinateUnpack(rect, sortedPoints);
        harmongraphyMatrix(img, sortedPoints);
        hierarchy.release();
        mask.release();
        return;
    }

    public int classifyDigit(Mat cellResized) {
        try {
            Mnist model = Mnist.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 28, 28, 1}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 28 * 28);
            inputFeature0.loadBuffer(byteBuffer);
            byteBuffer.order(ByteOrder.nativeOrder());

            for (int i=0; i < 28; i++) {
                for (int j=0; j<28; j++) {
                    byteBuffer.putFloat((float)(cellResized.get(i,j)[0] * (1.f/256)));
                }
            }

            // Runs model inference and gets result.
            Mnist.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            int secondPos = 1;
            float maxConfidence = 0;
            float secConfidence = 0;
            for (int i=0; i<confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            if (maxPos == 0){
                for (int i=1; i<confidences.length; i++) {
                    if (confidences[i] > secConfidence) {
                        secConfidence = confidences[i];
                        secondPos = i;
                    }
                }
            }
            // Releases model resources if no longer used.
            model.close();
            if (maxPos == 0)
                return secondPos;
            else if (maxConfidence < 0.5)
                return 0;
            else
                return maxPos;
        } catch (IOException e) {
            // TODO Handle the exception

        }
        return 0;
    }

    public int[] segAndRecog(Mat img) {
        int[] sudokuArr;
        sudokuArr = new int[81];
        int temp = (myWidth-myHeight)/2;
        int cellWidth = (int) (myHeight/9);
        Rect rectCrop = new Rect(temp-1,0, cellWidth, cellWidth);

        for(int row=0; row<9; row++) {
            int y = row * cellWidth;
            for(int col=0; col<9; col++) {
                int x = (temp-1) + col * cellWidth;
                Mat image_cell = img.submat(y+10, y+cellWidth-10, x+10, x+cellWidth-10);
                int nonZ = Core.countNonZero(image_cell);
                if(nonZ > 250) {
                    Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
                    //Applying dilate on the Image
                    Imgproc.dilate(image_cell, image_cell, kernel);
                    Mat cellResized = new Mat(28,28, CvType.CV_8UC1);
                    Imgproc.resize(image_cell, cellResized, cellResized.size(), 0, 0, Imgproc.INTER_AREA);
//                    cellResized = cellResized.reshape(0,28*28);
                    cellResized.convertTo(cellResized,CvType.CV_32FC1);
                    int num_detected = classifyDigit(cellResized);
//                    int num_detected = cellRecog(cellResized);
                    sudokuArr[row*9+col] = num_detected;
                    cellResized.release();
                }
                else {
                    sudokuArr[row*9+col] = 0;
                }
            }
        }
        return sudokuArr;
    }

    public int[] findNextBlank(int[] array, int row) {
        for(int i = row; i < 9; i++) {
            for(int j = 0; j < 9; j++) {
                if(array[i*9+j] == 0) {
                    int[] ret = {i, j};
                    return ret;
                }
            }
        }
        int[] ret = {-1,-1};
        return ret;
    }

    public boolean isAvailable(int[] array, int row, int col, int num) {
        // same col
        for(int i = 0; i < 9; i++) {
            if(array[i*9+col] == num)
                return false;
        }
        // same row
        for(int j = 0; j < 9; j++) {
            if(array[row*9+j] == num)
                return false;
        }
        // same square
        int rth = row / 3;
        int cth = col / 3;
        for(int i = rth*3; i < rth*3+3; i++) {
            for(int j = cth*3; j < cth*3+3; j++) {
                if(array[i*9+j] == num)
                    return false;
            }
        }
        return true;
    }

    public boolean sudokuSovler(int[] sudokuArr, int row, int col) {
        int[] blank = findNextBlank(sudokuArr, row);
        if(blank[0] == -1){
            return true;
        }
        for(int num = 1; num < 10; num++) {
            boolean flag = isAvailable(sudokuArr, blank[0], blank[1], num);
            if(flag == true) {
                sudokuArr[blank[0]*9+blank[1]] = num;
                boolean secFlag = sudokuSovler(sudokuArr, blank[0], blank[1]);
                if(secFlag == true)
                    return true;
                else
                    sudokuArr[blank[0]*9+blank[1]] = 0;
            }
        }
        return false;
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
        imgDisplay.release();
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
        if (stateFlag==-2) {
            // do nothing
            stateFlag = -1;
            return mRgba;
        }
        else if(stateFlag == -1){
            // stop state
            // do nothing rn
            return mRgba;
        }
        else if(stateFlag == 0){
            // transition state
            // do nothing rn
            filtering(mGray);
            correction(imgDisplay);
            mySudoku = segAndRecog(imgDisplay);
            myAnswer = new int[81];
            for(int i = 0; i < 81; i++) {
                myAnswer[i] = mySudoku[i];
            }
            stateFlag = 1;
            return imgDisplay;
        }
        else if(stateFlag == 1){

            return imgDisplay;
        }
        else if(stateFlag == 2){
//            System.out.println("wht????????????????????????");
            boolean flag = true;
            flag = sudokuSovler(myAnswer, 0, 0);
//            if(flag==false)
//                myAnswer=null;
            return imgDisplay;
        }
        else {
            return imgDisplay;
        }
//        return imgDisplay;
    }
}