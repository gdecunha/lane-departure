#ifndef LANEDETECTOR_H
#define LANEDETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

class LaneDetector {
    private:
        Mat original;
        Mat inverted;
        vector<Point2f> srcpoints, dstpoints;

        Mat lastLeftCurve = (Mat_<float>(3, 1) << 0, 0.5, 0);
        Mat lastRightCurve = (Mat_<float>(3, 1) << 0, -0.5, 0);

    public:
        // constructor
        LaneDetector(Mat& image, vector<Point2f> src, vector<Point2f> dst);

        Mat processFrame(Mat& current);
        void regionOfInterest(Mat& in);
        void warp(Mat& in);
        void filterColor(Mat& in);
        void smooth(Mat& in);
        void thresholdBinary(Mat& in);
        vector<Point2f> getWhitePixels(Mat& img);
        vector<Point2f> slidingWindow(Mat& in, Rect window);
        Mat leastSquares(vector<Point2f> pts);
        void drawPoints(Mat& in, vector<Point2f> left, vector<Point2f> right);
};

#endif