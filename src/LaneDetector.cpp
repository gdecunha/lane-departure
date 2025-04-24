#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>



using namespace std;
using namespace cv;


#define WIDTH 1280
#define HEIGHT 720


class LaneDetector {
    private: 
        Mat original;
        Mat inverted;
        vector<Point2f> srcpoints, dstpoints;

        Mat lastLeftCurve = (Mat_<float>(3, 1) << 0, 0.5, 0);
        Mat lastRightCurve = (Mat_<float>(3, 1) << 0, -0.5, 0);
    
        
    public:
        LaneDetector(Mat& image, vector<Point2f> src, vector<Point2f> dst) :
            original{image}, 
            srcpoints{src},
            dstpoints{dst} {}

        

        Mat processFrame(Mat& current){

            current = regionOfInterest(current).clone();
            current = warp(current).clone();
            current = filterColor(current).clone();
            current = smooth(current).clone();
            current = thresholdBinary(current).clone();

            int widthW = 500;
            int heightW = 60;

            vector<Point2f> lpoints = slidingWindow(current, Rect(0, current.rows - heightW,  widthW, heightW));
            vector<Point2f> rpoints = slidingWindow(current, Rect(current.cols - widthW, current.rows - heightW, widthW, heightW));

            original = drawPoints(original, lpoints, rpoints);
            
            return original;
        }

        Mat regionOfInterest(Mat& in){
            Mat current;
            in.copyTo(current);

            Mat mask = Mat::zeros(current.size(), current.type());
            vector<Point> intPoints;
            for (const auto& point : srcpoints) {
                intPoints.push_back(Point(static_cast<int>(point.x), static_cast<int>(point.y)));
            }
            fillPoly(mask, intPoints, Scalar(255, 255, 255));
            vector<vector<Point>> fillPolyPoints = {intPoints};
            fillPoly(mask, fillPolyPoints, Scalar(255, 255, 255));
            Mat masked;
            bitwise_and(current, mask, masked);
            return masked;
        }
        
        Mat warp(Mat& in){
            
            Mat current;
            in.copyTo(current);

            Mat perspectiveMatrix = getPerspectiveTransform(srcpoints, dstpoints);
            invert(perspectiveMatrix, inverted);
            Mat warped;
            warpPerspective(current, warped, perspectiveMatrix, Size(WIDTH, HEIGHT));
            return warped;
        }

        Mat filterColor(Mat& in){
            Mat hsv;
            cvtColor(in, hsv, COLOR_BGR2HSV);  // use BGR2HSV since imread loads as BGR
        
            // Yellow range in HSV
            Mat maskYellow;
            inRange(hsv, Scalar(15, 100, 100), Scalar(35, 255, 255), maskYellow);
        
            // White range in HSV
            Mat maskWhite;
            inRange(hsv, Scalar(0, 0, 200), Scalar(180, 30, 255), maskWhite);
        
            // Combine masks
            Mat combinedMask;
            bitwise_or(maskYellow, maskWhite, combinedMask);
        
            return combinedMask;
        }

        Mat smooth(Mat& in){

            Mat current;
            in.copyTo(current);

            Mat blurred;
            GaussianBlur(current, blurred, Size(9, 9), 0);
            Mat kernel = Mat::ones(15, 15, CV_8U);
            dilate(blurred, blurred, kernel);
            erode(blurred, blurred, kernel);
            morphologyEx(blurred, blurred, MORPH_CLOSE, kernel);
            return blurred;
        }

        Mat thresholdBinary(Mat& in){

            Mat current;
            in.copyTo(current);

            Mat binary;
            threshold(current, binary, 150, 255, THRESH_BINARY | THRESH_OTSU);
            return binary;
        }

        vector<Point2f> getWhitePixels(Mat& img){
            vector<Point2f> points;

            for (int y = 0; y < img.rows; y++) {
                for (int x = 0; x < img.cols; x++) {
                    if (img.at<uchar>(y, x) == 255) {
                        points.emplace_back(Point2f(x, y));
                    }
                }
            }
            return points;
            
        }

        vector<Point2f> slidingWindow(Mat& in, Rect window){
            
            Mat current;
            in.copyTo(current);
            cvtColor(in, current, COLOR_GRAY2BGR);

            vector<Point2f> points;
            vector<Mat> debugFrames;

            
            while (true){

                // middle of current window
                float middle = window.x + window.width * 0.5;
        
                // find region specified by current window
                //Mat roi = current(window);
        
                Mat roi = current(window);
                Mat roiGray;
                cvtColor(roi, roiGray, COLOR_BGR2GRAY);

                // find all white pixels
                vector<Point2f> whitePixels = getWhitePixels(roiGray);

                float avg;

                // keep track of previous average
                if (points.empty()) {
                    // If this is the first left window, start on left side 
                    if (window.x < 550){
                        avg = 150;
                    } else {
                        avg = WIDTH - 150;
                    }
                    
                } else {
                    // Otherwise use last known avg
                    avg = points.back().x;
                }
                bool empty = true;

                // find avg x position of pixels
                if (!whitePixels.empty()) {
                    float sum = 0.0;
                    for (const auto& pt : whitePixels) {
                        sum += window.x + pt.x;
                    }
                    avg = sum / whitePixels.size();
                    empty = false;


                    // add avg x of pixels and their height to vec
                    Point point(avg, window.y + window.height*0.5);
                    points.push_back( point);
                }


                /// DEBUG
                Mat frame;
                current.copyTo(frame);
                rectangle(frame, window, Scalar(0, 255, 0), 2);        
        
                if (!empty){
                    circle(frame, Point(avg, window.y + window.height * 0.5), 5, Scalar(0, 0, 255), 3);  
                }
                     
        
                debugFrames.push_back(frame);
                ////



                // move up
                window.y -= window.height;
                
                

                // check if at top
                if (window.y < 0) { 
                    window.y = 0;
                    break;
                }

                // move window 
                if (!empty){
                    window.x += (avg - middle);
                }
                

                // if out of range move back in 
                if (window.x < 0){
                    window.x = 0;
                } else if (window.x + window.width >= current.size().width){
                    window.x = current.size().width - window.width;
                }

                
            }
           
            return points;

        }


        Mat leastSquares(vector<Point2f> pts){

            int n = static_cast<int>(pts.size());
            Mat A(n, 3, CV_32F);
            Mat b(n, 1, CV_32F);
            Mat coef;
        
        
            for (int i = 0 ; i < pts.size(); i++){
                float x = static_cast<float>(pts[i].x);
                A.at<float>(i, 0) = x*x;
                A.at<float>(i, 1) = x;
                A.at<float>(i, 2) = 1.0;
                b.at<float>(i, 0) = static_cast<float>(pts[i].y);
            }
        
            solve(A, b, coef, DECOMP_SVD);
        
            return coef;
        }


        Mat drawPoints(Mat& in, vector<Point2f> left, vector<Point2f> right){
            Mat current;
            in.copyTo(current);

            vector<Point> lCurvePoints;
            vector<Point> rCurvePoints;

            vector<Point2f> out;
           

            perspectiveTransform(left, out, inverted);
            Mat lcurve;
            try{
                lcurve = leastSquares(out);
                lastLeftCurve = lcurve;
                
            } catch (...){
                lcurve = lastLeftCurve;
            }
            float la = lcurve.at<float>(0);
            float lb = lcurve.at<float>(1);
            float lc = lcurve.at<float>(2);
                
        
            lCurvePoints.clear();
        
            for (int i = 0; i < out.size(); i++){
                float x = out[i].x;
                float y = la*x*x + lb*x + lc;
                //circle(current, Point(x, y), 3, Scalar(255, 0 , 0), 3);
                lCurvePoints.push_back(Point(cvRound(x), cvRound(y)));
            }
            polylines(current, lCurvePoints, false, Scalar(255, 0, 0), 3);


            out.clear();

            perspectiveTransform(right, out, inverted);
            Mat rcurve;
            try {
                rcurve = leastSquares(out);
                lastRightCurve = rcurve.clone();
                
            } catch (...){
                rcurve = lastRightCurve.clone();
            }
            float ra = rcurve.at<float>(0);
            float rb = rcurve.at<float>(1);
            float rc = rcurve.at<float>(2);
        
            rCurvePoints.clear();
        
            for (int i = 0; i < out.size(); i++){
                float x = out[i].x;
                float y = ra*x*x + rb*x + rc;
                rCurvePoints.push_back(Point(cvRound(x), cvRound(y)));
            
            }
        
            polylines(current, rCurvePoints, false, Scalar(255, 0, 0), 3);

            return current;
        }   
};


int main(){

    string path = "../vid/project_video.mp4";

    VideoCapture cap(path);
    if (!cap.isOpened()) {
        cerr << "Error opening video file\n";
        return -1;
    }

    Mat frame;
    cap >> frame;

    if (frame.empty()){
        cerr << "Failed to read frame\n";
        return -1;
    }

    vector<Point2f> src;
    src.push_back(Point2f(300, 470));  // top left
    src.push_back(Point2f(980, 470));  // top right 
    src.push_back(Point2f(WIDTH - 50, 720)); // bottom right 
    src.push_back(Point2f(50, 720));


    vector<Point2f> dst = {
        Point2f(0, 0), 
        Point2f(1280, 0), 
        Point2f(1280, 720), 
        Point2f(0, 720)};
    
    LaneDetector detector(frame, src, dst);


    while(true){
        cap >> frame;
        if (frame.empty()){
            cout << "End of video\n";
            break;
        }

        detector = LaneDetector(frame, src, dst);
        Mat processed = detector.processFrame(frame);

        imshow("Output", processed);
        int key = waitKey(1);
        if(key==27)break;

    }

    cap.release();
    destroyAllWindows();
    cout << "Video processing completed\n";
    return 0;
}