//References
//http://answers.opencv.org/question/4423/how-to-create-a-binary-image-mat/
//http://docs.opencv.org/2.4/doc/tutorials/imgproc/shapedescriptors/find_contours/find_contours.html
//http://stackoverflow.com/questions/7263621/how-to-find-corners-on-a-image-using-opencv
//http://stackoverflow.com/questions/18044404/placing-two-images-side-by-side-opencv-2-3-c
//http://docs.opencv.org/3.1.0/d2/dbd/tutorial_distance_transform.html
//http://answers.opencv.org/question/74777/how-to-use-approxpolydp-to-close-contours/
//http://stackoverflow.com/questions/34377943/opencv-is-it-possible-to-detect-rectangle-from-corners
//https://github.com/opencv/opencv/blob/master/samples/cpp/squares.cpp

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

#include "FPS.h"

using namespace std;
using namespace cv;

#define WINDOW_WIDTH 320
#define WINDOW_HEIGHT 320

vector<Point> contoursConvexHull(vector<vector<Point> > contours)
{
	vector<Point> result;
	vector<Point> pts;
	for (size_t i = 0; i< contours.size(); i++)
		for (size_t j = 0; j< contours[i].size(); j++)
			pts.push_back(contours[i][j]);
	convexHull(pts, result);
	return result;
}

// finds a cosine of angle between vectors
static double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

int info(){
	cout << "--------------------------" << endl;
	cout << "OpenCV version : " << CV_VERSION << endl;
	cout << "--------------------------" << endl;
	return 0;
}

//Load camera intrinsic parameters
int load_parameters(Mat &K, Mat &D, const char * filename){
	Mat Kdata;
	Mat Ddata;
	string line;
	ifstream myfile(filename);
	if (myfile.is_open())
	{
		double kvalues[9];
		double dvalues[4];
		int i = 0;
		int k = 0;
		int d = 0;
		while (getline(myfile, line))
		{
			if (i < 9){
				kvalues[k] = stod(line);
				k++;
			}
			else{
				dvalues[d] = stod(line);
				d++;
			}
			i++;
		}
		Kdata = Mat(Size(3, 3), CV_64F, &kvalues);
		Ddata = Mat(Size(1, 4), CV_64F, &dvalues);
		Kdata.copyTo(K);
		Ddata.copyTo(D);
		myfile.close();
	}
	else{
		cout << "Unable to open file" << endl;
	}
	return 0;
}

int segment_single_image(){
	cout << "--------------------------" << endl;
	cout << "OpenCV version : " << CV_VERSION << endl;
	cout << "--------------------------" << endl;
	int window_width = 320;
	int window_height = 320;
	// Load the image
	Mat src = imread("D:\\Projects\\UALR\\Research\\Projects\\CaveInABox\\Data\\360_0020.jpg");
	Rect roi;
	roi.x = src.size().width / 2;
	roi.y = 0;
	roi.width = src.size().width / 2;
	roi.height = src.size().height;
	src = src(roi);
	// Check if everything was fine
	if (!src.data)
		return -1;
	// Show source image
	namedWindow("Source Image", cv::WINDOW_NORMAL);
	resizeWindow("Source Image", window_width, window_height);
	imshow("Source Image", src);
	// Change the background from white to black, since that will help later to extract
	// better results during the use of Distance Transform
	for (int x = 0; x < src.rows; x++) {
		for (int y = 0; y < src.cols; y++) {
			if (src.at<Vec3b>(x, y) == Vec3b(255, 255, 255)) {
				src.at<Vec3b>(x, y)[0] = 0;
				src.at<Vec3b>(x, y)[1] = 0;
				src.at<Vec3b>(x, y)[2] = 0;
			}
		}
	}
	// Show output image
	namedWindow("Black Background Image", cv::WINDOW_NORMAL);
	resizeWindow("Black Background Image", window_width, window_height);
	imshow("Black Background Image", src);
	// Create a kernel that we will use for accuting/sharpening our image
	Mat kernel = (Mat_<float>(3, 3) <<
		1, 1, 1,
		1, -8, 1,
		1, 1, 1); // an approximation of second derivative, a quite strong kernel
	// do the laplacian filtering as it is
	// well, we need to convert everything in something more deeper then CV_8U
	// because the kernel has some negative values,
	// and we can expect in general to have a Laplacian image with negative values
	// BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
	// so the possible negative number will be truncated
	Mat imgLaplacian;
	Mat sharp = src; // copy source image to another temporary one
	filter2D(sharp, imgLaplacian, CV_32F, kernel);
	src.convertTo(sharp, CV_32F);
	Mat imgResult = sharp - imgLaplacian;
	// convert back to 8bits gray scale
	imgResult.convertTo(imgResult, CV_8UC3);
	imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
	namedWindow("New Sharped Image", cv::WINDOW_NORMAL);
	resizeWindow("New Sharped Image", window_width, window_height);
	imshow("New Sharped Image", imgResult);
	src = imgResult; // copy back
	// Create binary image from source image
	Mat bw;
	cvtColor(src, bw, CV_BGR2GRAY);
	threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	namedWindow("Binary Image", cv::WINDOW_NORMAL);
	resizeWindow("Binary Image", window_width, window_height);
	imshow("Binary Image", bw);
	// Perform the distance transform algorithm
	Mat dist;
	distanceTransform(bw, dist, CV_DIST_L2, 3);
	// Normalize the distance image for range = {0.0, 1.0}
	// so we can visualize and threshold it
	normalize(dist, dist, 0, 1., NORM_MINMAX);
	namedWindow("Distance Transform Image", cv::WINDOW_NORMAL);
	resizeWindow("Distance Transform Image", window_width, window_height);
	imshow("Distance Transform Image", dist);
	// Threshold to obtain the peaks
	// This will be the markers for the foreground objects
	threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
	// Dilate a bit the dist image
	Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
	dilate(dist, dist, kernel1);
	namedWindow("Peaks", cv::WINDOW_NORMAL);
	resizeWindow("Peaks", window_width, window_height);
	imshow("Peaks", dist);
	// Create the CV_8U version of the distance image
	// It is needed for findContours()
	Mat dist_8u;
	dist.convertTo(dist_8u, CV_8U);
	// Find total markers
	vector<vector<Point> > contours;
	findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	// Create the marker image for the watershed algorithm
	Mat markers = Mat::zeros(dist.size(), CV_32SC1);
	// Draw the foreground markers
	for (size_t i = 0; i < contours.size(); i++)
		drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
	// Draw the background marker
	circle(markers, Point(5, 5), 3, CV_RGB(255, 255, 255), -1);
	namedWindow("Markers", cv::WINDOW_NORMAL);
	resizeWindow("Markers", window_width, window_height);
	imshow("Markers", markers * 10000);
	// Perform the watershed algorithm
	watershed(src, markers);
	Mat mark = Mat::zeros(markers.size(), CV_8UC1);
	markers.convertTo(mark, CV_8UC1);
	bitwise_not(mark, mark);
	// Generate random colors
	vector<Vec3b> colors;
	for (size_t i = 0; i < contours.size(); i++)
	{
		int b = theRNG().uniform(0, 255);
		int g = theRNG().uniform(0, 255);
		int r = theRNG().uniform(0, 255);
		colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}
	// Create the result image
	Mat dst = Mat::zeros(markers.size(), CV_8UC3);
	// Fill labeled objects with random colors
	for (int i = 0; i < markers.rows; i++)
	{
		for (int j = 0; j < markers.cols; j++)
		{
			int index = markers.at<int>(i, j);
			if (index > 0 && index <= static_cast<int>(contours.size()))
				dst.at<Vec3b>(i, j) = colors[index - 1];
			else
				dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
		}
	}
	cvtColor(dst, dst, CV_BGR2GRAY);
	GaussianBlur(dst, dst, Size(7, 7), 0);
	// Visualize the final image
	namedWindow("Final Result", cv::WINDOW_NORMAL);
	resizeWindow("Final Result", window_width, window_height);
	imshow("Final Result", dst);
	//Binary image
	cv::Mat binaryMat(dst.size(), dst.type());
	cv::threshold(dst, binaryMat, 20, 255, cv::THRESH_BINARY);
	namedWindow("Flatten", cv::WINDOW_NORMAL);
	resizeWindow("Flatten", window_width, window_height);
	imshow("Flatten", binaryMat);
	Mat morph_kernel = getStructuringElement(MORPH_ELLIPSE, Size(11, 11));
	morphologyEx(binaryMat, binaryMat, CV_MOP_CLOSE, morph_kernel);
	namedWindow("Morphology", cv::WINDOW_NORMAL);
	resizeWindow("Morphology", window_width, window_height);
	imshow("Morphology", binaryMat);
	Mat input = imread("fpgJx.jpg", 1);
	Point2f inputQuad[4];
	inputQuad[0] = Point2f(0.0f, 0.0f);
	inputQuad[1] = Point2f((float)input.cols, 0.0f);
	inputQuad[2] = Point2f((float)input.cols, (float)input.rows);
	inputQuad[3] = Point2f(0.0f, (float)input.rows);
	int thresh = 100;
	int max_thresh = 255;
	RNG rng(12345);
	Mat canny_output;
	vector<vector<Point> > transform_contours;
	vector<Vec4i> hierarchy;
	Canny(binaryMat, canny_output, thresh, thresh * 2, 3);
	findContours(canny_output, transform_contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	for (int i = 0; i< transform_contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, transform_contours, i, color, 2, 8, hierarchy, 0, Point());
	}
	namedWindow("Contours", cv::WINDOW_NORMAL);
	resizeWindow("Contours", window_width, window_height);
	imshow("Contours", drawing);
	vector<Point> ConvexHullPoints = contoursConvexHull(transform_contours);
	polylines(drawing, ConvexHullPoints, true, Scalar(0, 0, 255), 20);
	cvtColor(drawing, drawing, CV_BGR2GRAY);
	drawing.convertTo(drawing, CV_32FC1, 1.0 / 255.0);
	namedWindow("Hull", cv::WINDOW_NORMAL);
	resizeWindow("Hull", window_width, window_height);
	imshow("Hull", drawing);
	Mat K;
	Mat D;
	string line;
	ifstream myfile("camera.txt");
	if (myfile.is_open())
	{
		double kvalues[9];
		double dvalues[4];
		int i = 0;
		int k = 0;
		int d = 0;
		while (getline(myfile, line))
		{
			if (i < 9){
				kvalues[k] = stod(line);
				k++;
			}
			else{
				dvalues[d] = stod(line);
				d++;
			}
			i++;
		}
		K = Mat(Size(3, 3), CV_64F, &kvalues);
		D = Mat(Size(1, 4), CV_64F, &dvalues);
		myfile.close();
	}
	else{
		cout << "Unable to open file" << endl;
	}
	cout << K << endl;
	cout << D << endl;
	Point2f outputQuad[4];
	vector<Point2f> approx;
	approxPolyDP(Mat(transform_contours[0]), approx, arcLength(Mat(transform_contours[0]), true)*0.02, true);
	cout << "Evaluating Square" << endl;
	cout << approx << endl;
	outputQuad[0] = approx[0];
	outputQuad[1] = approx[1];
	outputQuad[2] = approx[2];
	outputQuad[3] = approx[3];
	Mat lambda(2, 4, CV_32FC1);
	lambda = getPerspectiveTransform(inputQuad, outputQuad);
	Mat output;
	warpPerspective(input, output, lambda, output.size());
	//Display input and output
	namedWindow("Input", cv::WINDOW_NORMAL);
	resizeWindow("Input", window_width, window_height);
	imshow("Input", input);
	namedWindow("Output", cv::WINDOW_NORMAL);
	resizeWindow("Output", window_width, window_height);
	imshow("Output", output);
	waitKey(0);
	return 0;
}

int stream_theta(Mat K, Mat D){
	Mat frame;
	VideoCapture cap(1);
	cap >> frame;
	namedWindow("Stream");
	FPS counter;
	Mat input = imread("fpgJx.jpg", 1);
	Point2f inputQuad[4];
	inputQuad[0] = Point2f(0.0f, 0.0f);
	inputQuad[1] = Point2f((float)input.cols, 0.0f);
	inputQuad[2] = Point2f((float)input.cols, (float)input.rows);
	inputQuad[3] = Point2f(0.0f, (float)input.rows);
	while (1){
		counter.start_fps_counter();
		cap >> frame;
		// Cut the fisheye frame in half
		Rect roi;
		roi.x = frame.size().width / 2;
		roi.y = 0;
		roi.width = frame.size().width / 2;
		roi.height = frame.size().height;
		frame = frame(roi);
		// Resize the fisheye frame 
		resize(frame, frame, Size(WINDOW_WIDTH, WINDOW_HEIGHT));
		// Rotate the source frame
		Point2f src_center(frame.cols / 2.0F, frame.rows / 2.0F);
		Mat rot_mat = getRotationMatrix2D(src_center, 90, 1.0);
		warpAffine(frame, frame, rot_mat, frame.size());
		namedWindow("Stream", cv::WINDOW_NORMAL);
		resizeWindow("Stream", WINDOW_WIDTH, WINDOW_HEIGHT);
		imshow("Stream", frame);
		for (int x = 0; x < frame.rows; x++) {
			for (int y = 0; y < frame.cols; y++) {
				if (frame.at<Vec3b>(x, y) == Vec3b(255, 255, 255)) {
					frame.at<Vec3b>(x, y)[0] = 0;
					frame.at<Vec3b>(x, y)[1] = 0;
					frame.at<Vec3b>(x, y)[2] = 0;
				}
			}
		}
		namedWindow("Black Background Image", cv::WINDOW_NORMAL);
		resizeWindow("Black Background Image", WINDOW_WIDTH, WINDOW_HEIGHT);
		imshow("Black Background Image", frame);
		// Create a kernel that we will use for accuting/sharpening our image
		Mat kernel = (Mat_<float>(3, 3) <<
			1, 1, 1,
			1, -8, 1,
			1, 1, 1); // an approximation of second derivative, a quite strong kernel
		// do the laplacian filtering as it is
		// well, we need to convert everything in something more deeper then CV_8U
		// because the kernel has some negative values,
		// and we can expect in general to have a Laplacian image with negative values
		// BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
		// so the possible negative number will be truncated
		Mat imgLaplacian;
		Mat sharp = frame; // copy source image to another temporary one
		filter2D(sharp, imgLaplacian, CV_32F, kernel);
		frame.convertTo(sharp, CV_32F);
		Mat imgResult = sharp - imgLaplacian;
		// convert back to 8bits gray scale
		imgResult.convertTo(imgResult, CV_8UC3);
		imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
		namedWindow("New Sharped Image", cv::WINDOW_NORMAL);
		resizeWindow("New Sharped Image", WINDOW_WIDTH, WINDOW_HEIGHT);
		imshow("New Sharped Image", imgResult);
		frame = imgResult; // copy back
		// Create binary image from source image
		Mat bw;
		cvtColor(frame, bw, CV_BGR2GRAY);
		threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		namedWindow("Binary Image", cv::WINDOW_NORMAL);
		resizeWindow("Binary Image", WINDOW_WIDTH, WINDOW_HEIGHT);
		imshow("Binary Image", bw);
		// Perform the distance transform algorithm
		Mat dist;
		distanceTransform(bw, dist, CV_DIST_L2, 3);
		// Normalize the distance image for range = {0.0, 1.0}
		// so we can visualize and threshold it
		normalize(dist, dist, 0, 1., NORM_MINMAX);
		namedWindow("Distance Transform Image", cv::WINDOW_NORMAL);
		resizeWindow("Distance Transform Image", WINDOW_WIDTH, WINDOW_HEIGHT);
		imshow("Distance Transform Image", dist);
		// Threshold to obtain the peaks
		// This will be the markers for the foreground objects
		threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
		// Dilate a bit the dist image
		Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
		dilate(dist, dist, kernel1);
		namedWindow("Peaks", cv::WINDOW_NORMAL);
		resizeWindow("Peaks", WINDOW_WIDTH, WINDOW_HEIGHT);
		imshow("Peaks", dist);
		// Create the CV_8U version of the distance image
		// It is needed for findContours()
		Mat dist_8u;
		dist.convertTo(dist_8u, CV_8U);
		// Find total markers
		vector<vector<Point> > contours;
		findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		// Create the marker image for the watershed algorithm
		Mat markers = Mat::zeros(dist.size(), CV_32SC1);
		// Draw the foreground markers
		for (size_t i = 0; i < contours.size(); i++)
			drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
		// Draw the background marker
		circle(markers, Point(5, 5), 3, CV_RGB(255, 255, 255), -1);
		namedWindow("Markers", cv::WINDOW_NORMAL);
		resizeWindow("Markers", WINDOW_WIDTH, WINDOW_HEIGHT);
		imshow("Markers", markers * 10000);
		// Perform the watershed algorithm
		watershed(frame, markers);
		Mat mark = Mat::zeros(markers.size(), CV_8UC1);
		markers.convertTo(mark, CV_8UC1);
		bitwise_not(mark, mark);
		namedWindow("Marker", cv::WINDOW_NORMAL);
		resizeWindow("Marker", WINDOW_WIDTH, WINDOW_HEIGHT);
		imshow("Marker", mark);
		// Generate random colors
		vector<Vec3b> colors;
		for (size_t i = 0; i < contours.size(); i++)
		{
			int b = theRNG().uniform(0, 255);
			int g = theRNG().uniform(0, 255);
			int r = theRNG().uniform(0, 255);
			colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
		}
		// Create the result image
		Mat dst = Mat::zeros(markers.size(), CV_8UC3);
		// Fill labeled objects with random colors
		for (int i = 0; i < markers.rows; i++)
		{
			for (int j = 0; j < markers.cols; j++)
			{
				int index = markers.at<int>(i, j);
				if (index > 0 && index <= static_cast<int>(contours.size()))
					dst.at<Vec3b>(i, j) = colors[index - 1];
				else
					dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			}
		}
		cvtColor(dst, dst, CV_BGR2GRAY);
		GaussianBlur(dst, dst, Size(7, 7), 0);
		// Visualize the final image
		namedWindow("Final Result", cv::WINDOW_NORMAL);
		resizeWindow("Final Result", WINDOW_WIDTH, WINDOW_HEIGHT);
		imshow("Final Result", dst);
		//Binary image
		cv::Mat binaryMat(dst.size(), dst.type());
		cv::threshold(dst, binaryMat, 20, 255, cv::THRESH_BINARY);
		namedWindow("Flatten", cv::WINDOW_NORMAL);
		resizeWindow("Flatten", WINDOW_WIDTH, WINDOW_HEIGHT);
		imshow("Flatten", binaryMat);
		Mat morph_kernel = getStructuringElement(MORPH_ELLIPSE, Size(11, 11));
		morphologyEx(binaryMat, binaryMat, CV_MOP_CLOSE, morph_kernel);
		namedWindow("Morphology", cv::WINDOW_NORMAL);
		resizeWindow("Morphology", WINDOW_WIDTH, WINDOW_HEIGHT);
		imshow("Morphology", binaryMat);
		int thresh = 100;
		int max_thresh = 255;
		RNG rng(12345);
		Mat canny_output;
		vector<vector<Point> > transform_contours;
		vector<Vec4i> hierarchy;
		Canny(binaryMat, canny_output, thresh, thresh * 2, 3);
		findContours(canny_output, transform_contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
		for (int i = 0; i< transform_contours.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(drawing, transform_contours, i, color, 2, 8, hierarchy, 0, Point());
		}
		namedWindow("Contours", cv::WINDOW_NORMAL);
		resizeWindow("Contours", WINDOW_WIDTH, WINDOW_HEIGHT);
		imshow("Contours", drawing);
		vector<Point> ConvexHullPoints = contoursConvexHull(transform_contours);
		polylines(drawing, ConvexHullPoints, true, Scalar(0, 0, 255), 1);
		cvtColor(drawing, drawing, CV_BGR2GRAY);
		drawing.convertTo(drawing, CV_32FC1, 1.0 / 255.0);
		namedWindow("Hull", cv::WINDOW_NORMAL);
		resizeWindow("Hull", WINDOW_WIDTH, WINDOW_HEIGHT);
		imshow("Hull", drawing);
		Point2f outputQuad[4];
		vector<Point2f> approx;
		approxPolyDP(Mat(transform_contours[0]), approx, arcLength(Mat(transform_contours[0]), true)*0.02, true);
		outputQuad[0] = approx[0];
		outputQuad[1] = approx[1];
		outputQuad[2] = approx[2];
		outputQuad[3] = approx[3];
		Mat lambda(2, 4, CV_32FC1);
		lambda = getPerspectiveTransform(inputQuad, outputQuad);
		Mat output;
		warpPerspective(input, output, lambda, output.size());
		//Display input and output
		namedWindow("Input", cv::WINDOW_NORMAL);
		resizeWindow("Input", WINDOW_WIDTH, WINDOW_HEIGHT);
		imshow("Input", input);
		namedWindow("Output", cv::WINDOW_NORMAL);
		resizeWindow("Output", WINDOW_WIDTH, WINDOW_HEIGHT);
		imshow("Output", output);
		int key = waitKey(60);
		if (key == 27){
			destroyAllWindows();
			cap.release();
			return 0;
		}
		counter.end_fps_counter();
		counter.print_fps();
	}
	destroyAllWindows();
	cap.release();
	return 0;
}

int main(int, char** argv)
{
	int result;
	result = info();
	Mat K;
	Mat D;
	result = load_parameters(K, D, "camera.txt");
	cout << K << endl;
	cout << D << endl;
	result = stream_theta(K, D);
	//result = segment_single_image();
	return result;
}