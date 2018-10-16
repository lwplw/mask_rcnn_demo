#include "stdio.h"
#include "stdlib.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string.h>
#include "common_opencv.h"

using namespace cv;
using namespace dnn;
using namespace std;

vector<string> classes;
vector<Scalar> colors;

// Initialize the parameters
float confThreshold = 0.97; // Confidence threshold
float maskThreshold = 0.3; // Mask threshold

// Draw the predicted bounding box
void drawBox(Mat& image, Mat& image2, int classId, float conf, Rect box, Mat& objectMask);

// Postprocess the neural network's output for image
void postprocess(Mat& image, Mat& image2, const vector<Mat>& outs);

int main(void)
{
	// Load names of classes
	string classesFile = "mscoco_labels.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) 
		classes.push_back(line);

	// Load the colors
	string colorsFile = "colors.txt";
	ifstream colorFptr(colorsFile.c_str());
	while (getline(colorFptr, line)) {
		char* pEnd;
		double r, g, b;
		r = strtod(line.c_str(), &pEnd);
		g = strtod(pEnd, NULL);
		b = strtod(pEnd, NULL);
		Scalar color = Scalar(r, g, b, 255.0);
		colors.push_back(Scalar(r, g, b, 255.0));
	}

	// Give the configuration and weight files for the model
	String textGraph    = "D:\\SWC\\algorithm\\deep_learning\\Mask_R-CNN\\MaskCNN\\mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
	String modelWeights = "D:\\SWC\\algorithm\\deep_learning\\Mask_R-CNN\\MaskCNN\\mask_rcnn_inception_v2_coco_2018_01_28\\frozen_inference_graph.pb";

	// Load the network
	Net net = readNetFromTensorflow(modelWeights, textGraph);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// Open an image file
	string str, outputFile;
	Mat image, image2, blob;
	str = "test.jpg";
	image = cv::imread(str);

	// 均值滤波
	blur(image, image2, Size(7, 7));

	str.replace(str.end() - 4, str.end(), "_mask_rcnn_out.jpg");
	outputFile = str;
	
	// Create a window
	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);

	// Create a 4D blob from an image.
	blobFromImage(image, blob, 1.0, Size(image.cols, image.rows), Scalar(), true, false);

	// Sets the input to the network
	net.setInput(blob);

	// Runs the forward pass to get output from the output layers
	std::vector<String> outNames(2);
	outNames[0] = "detection_out_final";
	outNames[1] = "detection_masks";
	vector<Mat> outs;
	net.forward(outs, outNames);

	// Extract the bounding box and mask for each of the detected objects
	postprocess(image, image2, outs);

	// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
	// vector<double> layersTimes;
	// double freq  = getTickFrequency() / 1000;
	// double t     = net.getPerfProfile(layersTimes) / freq;
	// string label = format("Mask-RCNN on 2.5 GHz Intel Core i7 CPU, Inference time for a frame : %0.0f ms", t);
	// putText(image2, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));

	// Write the image with the detection boxes
	Mat detectedImage;
	image2.convertTo(detectedImage, CV_8U);
	imwrite(outputFile, detectedImage);
	// imshow(kWinName, image2);
	// waitKey();

	return 0;
}

// For the image, extract the bounding box and mask for each detected object
void postprocess(Mat& image, Mat& image2, const vector<Mat>& outs)
{
	Mat outDetections = outs[0];
	Mat outMasks = outs[1];

	// Output size of masks is NxCxHxW where
	// N   - number of detected boxes
	// C   - number of classes (excluding background)
	// HxW - segmentation shape
	const int numDetections = outDetections.size[2];
	const int numClasses = outMasks.size[1];

	outDetections = outDetections.reshape(1, outDetections.total() / 7);
	for (int i = 0; i < numDetections; ++i)
	{
		float score = outDetections.at<float>(i, 2);
		if (score > confThreshold)
		{
			// Extract the bounding box
			int classId = static_cast<int>(outDetections.at<float>(i, 1));
			int left    = static_cast<int>(image.cols * outDetections.at<float>(i, 3));
			int top     = static_cast<int>(image.rows * outDetections.at<float>(i, 4));
			int right   = static_cast<int>(image.cols * outDetections.at<float>(i, 5));
			int bottom  = static_cast<int>(image.rows * outDetections.at<float>(i, 6));
			 
			left   = max(0, min(left,   image.cols - 1));
			top    = max(0, min(top,    image.rows - 1));
			right  = max(0, min(right,  image.cols - 1));
			bottom = max(0, min(bottom, image.rows - 1));
			Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

			// Extract the mask for the object
			Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId));

			// Draw bounding box, colorize and show the mask on the image
			drawBox(image, image2, classId, score, box, objectMask);
		}
	}
}

// Draw the predicted bounding box, colorize and show the mask on the image
void drawBox(Mat& image, Mat& image2, int classId, float conf, Rect box, Mat& objectMask)
{
	//Draw a rectangle displaying the bounding box
	// rectangle(image2, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(255, 178, 50), 3);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	box.y = max(box.y, labelSize.height);
	// rectangle(image2, Point(box.x, box.y - round(1.5*labelSize.height)), Point(box.x + round(1.5*labelSize.width), box.y + baseLine), Scalar(255, 255, 255), FILLED);
	// putText(image2, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);

	Scalar color = colors[classId%colors.size()];

	// Resize the mask, threshold, color and apply it on the image
	resize(objectMask, objectMask, Size(box.width, box.height));
	Mat mask = (objectMask > maskThreshold);
	// Mat coloredRoi = (0.3 * color + 0.7 * image(box));
	Mat coloredRoi = image(box);
	coloredRoi.convertTo(coloredRoi, CV_8UC3);
	// imshow("coloredRoi.jpg", coloredRoi);
	// waitKey();

	// Draw the contours on the image
	vector<Mat> contours;
	Mat hierarchy;
	mask.convertTo(mask, CV_8U);

	findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE); // 提取轮廓
	// drawContours(coloredRoi, contours, -1, color, 5, LINE_8, hierarchy, 100); // 绘制轮廓

	// 将绘制好的轮廓图coloredRoi，覆盖到大图image的对应box的区域
	coloredRoi.copyTo(image2(box), mask);
}