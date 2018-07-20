// Color_MotorTC_Detection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <fstream>

using namespace cv;
using namespace std;

//#define DEBUG

int _tmain(int argc, _TCHAR* argv[])
{
	if (argc != 3){
		cout << "Bad input" << endl;
		cout << "USE Color_motorTC_detection.exe video_filename filenameBg" << endl;
		cout << "where:" << endl;
		cout << "video_filename is a video file compatible with FFMpeg" << endl;
		cout << "filenameBg is a background image file" << endl; 
		cout << endl;
		cout << "Press Enter to exit..." << endl;
		getchar();
		return 0;
	}
	
	double fps;
	int avgOffsetIter = 20;
	float pixelMetrics = 38.44;
	Size capImgSize;
	Size imgFinalSize = Size(1280,1024);
	Rect roiRect = Rect(400,450,1000-400,740-450);
	String filepath = "";// "C:\\Users\\PKM01\\Desktop\\";
	String filename = "DC_Tp_test3cut.avi"; //"silnik_Szybki_197pfs_0.5s_TpTest3.avi"; 
	String filenameBg = "tlo.bmp";
	VideoCapture cap(filepath + filename);
	Mat metricFile = imread(filepath + filenameBg);
	
	if (metricFile.empty()){
		printf("Cannot read image background file\n Press any key to exit");
		getchar();
		return - 1;
	}	
	if (! (metricFile.cols == imgFinalSize.width && metricFile.rows > 0)){
		printf("Improper backgorund image size\n Press any key to exit");
		getchar();
		return -1;
	}
		
	fps = cap.get(CV_CAP_PROP_FPS);
	capImgSize = Size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	if (!cap.isOpened()){  // check if we succeeded
		printf("Cannot open video source\n Press any key to exit");
		getchar();
		return -1;
	}
	else{
		printf("Opened file: %s\nFPS: %3.2f\nSize: %dx%d\n",filename.c_str(), fps,capImgSize.width, capImgSize.height);
		printf("Desired size: %dx%d", imgFinalSize.width, imgFinalSize.height);
	}


	namedWindow("Display", 1);
	int waitTime = (int)(1000.0 / fps);
	vector<Point2f> avgCenter;
	float offset = 0.0;
	vector<pair<float, float>> logdata; //position, time

	// Video writer
	VideoWriter outputVideo;
	bool readyToRecord = false;

	for (int capFrame =0; ; capFrame++)
	{
		Mat frame, resized, display;
		Mat roi, roiHsl, bgroi;
		cap >> frame;
		if (frame.empty()){
			printf("End of file\n");
			break;
		}
		if (capFrame == 0){
			outputVideo.open("ColoMotorDCdetection.avi", CV_FOURCC('D', 'I', 'V', 'X'), 20, imgFinalSize, true);
			if (outputVideo.isOpened())
				readyToRecord = true;
			else
				readyToRecord = false;
		}
		resize(frame, resized, imgFinalSize, 0, 0, CV_INTER_CUBIC); // slow but better CV_INTER_CUBIC
		
		// set bgroi and roi (add metric and processing area)
		bgroi = resized(Rect(0, imgFinalSize.height - metricFile.rows, metricFile.cols, metricFile.rows));
		metricFile.copyTo(bgroi);
		resized(roiRect).copyTo(roi);
		resized.copyTo(display);
		

		// detect color in roi
		Scalar hslThresh1, hslThresh2, hslMaxRange;
		cvtColor(roi, roiHsl, COLOR_BGR2HLS);
	
		hslMaxRange = Scalar(180., 255., 255.);
		hslThresh1 = Scalar(170., 70., 100.);
		hslThresh2 = Scalar(4., 170., 255.); //180+5

		Mat mask1, mask2, mask;

		bool hslOutScope[3];
		for (int i = 0; i<3; i++)
			hslThresh1[i] > hslThresh2[i] ? hslOutScope[i] = true : hslOutScope[i] = false;

		// for hsl out of scope, must calc 2 separate scopes
		Scalar hsl1Low = Scalar(0, 0, 0);
		Scalar hsl1High = Scalar(0, 0, 0);
		Scalar hsl2Low = Scalar(0, 0, 0);
		Scalar hsl2High = Scalar(0, 0, 0);
		for (int i = 0; i < 3; i++){
			if (hslOutScope[i]){
				hsl1Low[i] = -1, hsl1High[i] = hslThresh2[i];
				hsl2Low[i] = hslThresh1[i], hsl2High[i] = hslMaxRange[i];
			}
			else{
				hsl1Low[i] = hslThresh1[i], hsl1High[i] = hslThresh2[i];
				hsl2Low[i] = hslThresh1[i], hsl2High[i] = hslThresh2[i];
			}
		}
		inRange(roiHsl, hsl1Low, hsl1High, mask1);
		inRange(roiHsl, hsl2Low, hsl2High, mask2);
		bitwise_or(mask1, mask2, mask);
		#ifdef DEBUG 
		imshow("mask", mask);
		imshow("mask1", mask1);
		imshow("mask2", mask2);
		#endif

		//dilate or closing, calculate contours take biggest, 
		Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
		morphologyEx(mask, mask,MORPH_CLOSE, kernel,Point(-1,-1), 3);
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		int largestContourIdx = 0;
		double largestContourArea = 0.0;
		findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));

		if (contours.size() == 0){
			printf("No contours in %d frame. Frame skipped. \n", capFrame);
			continue;
		}
		for (int i = 0; i < contours.size(); i++){
			double actualContourArea = contourArea(contours[i], false);
			if (largestContourArea < actualContourArea){
				largestContourArea = actualContourArea;
				largestContourIdx = i;
			}
		}
		Moments mu = moments(contours[largestContourIdx], false);
		Point2f center = Point2f(mu.m10/mu.m00, mu.m01/mu.m00);
		if (capFrame < avgOffsetIter)
			avgCenter.push_back(center);
		else{
			float sum = 0.0;
			for (int i = 0; i < avgCenter.size(); i++){
				sum = sum + avgCenter[i].x;
			}
			offset = sum / (float)avgOffsetIter;
		}
		float position = (center.x - offset) / pixelMetrics;
		pair<float, float> data;
		data.first = position;
		data.second = (double)capFrame/fps;
		logdata.push_back(data);

		#ifdef DEBUG  // test roi, contours, moments
		Mat actualContour = Mat::zeros(mask.size(), CV_8UC3);
		drawContours(actualContour, contours, largestContourIdx, Scalar(255, 0, 0), -1, 8, hierarchy);
		rectangle(actualContour, Rect(center.x - 1.0, center.y - 1.0, 3, 3), Scalar(0, 0, 255), 3);
		imshow("contours", actualContour);
		if (waitKey(0) == 27) break;
		rectangle(display, roiRect, Scalar(0, 255, 0), 1);
		#endif



		// display results
		string msg;
		if (offset == 0.0)
			msg = format("Frame:%d, marker location with offset in pixels: %3.2f", capFrame, center.x);
		else
			msg = format("Frame:%d, marker location in pixels: %3.2f", capFrame, center.x - offset);
		putText(display, msg, Point(50, 50), 1, 1, Scalar(0, 255, 0));
		putText(display, format("Marker position:%*2.2f cm", 1, position), Point(50, 90), 1, 1, Scalar(0, 255, 0));
		putText(display, format("Movement time: %2.4f s", (double)capFrame/fps), Point(50, 130), 1, 1, Scalar(0, 255, 0));
		line(display, Point(center.x + roiRect.x, center.y), Point(center.x + roiRect.x, imgFinalSize.height), Scalar(0, 255, 0));
		imshow("Display", display);

		if (readyToRecord)
			outputVideo.write(display);
		//if (waitKey(waitTime) >= 0) break;
		if (waitKey(1) == 27) break;
	}

	if (readyToRecord)
		outputVideo.release();

	// log data to file
	ofstream outputfile;
	outputfile.open("movementlog.txt");
	if (outputfile.is_open()){
		for (int i = 0; i < logdata.size(); i++)
			outputfile << logdata[i].first << " " << logdata[i].second << endl;
	}
	else{
		printf("Cannot write log to file. Press a key.\n");
		getchar();
	}
	return 0;
}

