#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;





Mat original;
vector<Point2f> srcpoints;
Mat inverted;


Mat regionOfInterest(const Mat& img){
    Mat mask = Mat::zeros(img.size(), img.type());

    int height = img.rows;
    int width = img.cols;

    printf("Image size: %d x %d\n", width, height);

    
    // srcpoints.clear();
    // srcpoints.push_back(Point2f(560, 470));  // top left
    // srcpoints.push_back(Point2f(720, 470));  // top right 
    // srcpoints.push_back(Point2f(1100, 680)); // bottom right 
    // srcpoints.push_back(Point2f(200, 680)); // bottom left




    srcpoints.clear();
    srcpoints.push_back(Point2f(300, 470));  // top left
    srcpoints.push_back(Point2f(980, 470));  // top right 
    srcpoints.push_back(Point2f(width - 50, 720)); // bottom right 
    srcpoints.push_back(Point2f(50, 720)); 


    vector<Point> intPoints;
    for (const auto& point : srcpoints) {
        intPoints.push_back(Point(static_cast<int>(point.x), static_cast<int>(point.y)));
    }
    fillPoly(mask, intPoints, Scalar(255, 255, 255));



    vector<vector<Point>> fillPolyPoints = {intPoints};
    fillPoly(mask, fillPolyPoints, Scalar(255, 255, 255));
    Mat masked;
    bitwise_and(img, mask, masked);
    return masked;
}


Mat warp(const Mat& in){

    Mat current;
    in.copyTo(current);

    vector<Point2f> dstpoints = {
        Point2f(0, 0), 
        Point2f(1280, 0), 
        Point2f(1280, 720), 
        Point2f(0, 720)};

    Mat perspectiveMatrix = getPerspectiveTransform(srcpoints, dstpoints);


    invert(perspectiveMatrix, inverted);

    Mat warped;
    warpPerspective(current, warped, perspectiveMatrix, Size(1280, 720));
    return warped;
}

Mat filterColor(const Mat& in){


    // TODO improve thresholding


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

Mat smooth(const Mat& in){

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


Mat threshold(const Mat& in){
    Mat current;
    in.copyTo(current);

    Mat binary;
    threshold(current, binary, 150, 255, THRESH_BINARY | THRESH_OTSU);
    return binary;
}
    



vector<Point2f> getWhitePixels(const Mat& img){
    vector<Point2f> points;
    const Size imgSize = img.size();
  

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            if (img.at<uchar>(y, x) == 255) {
                points.emplace_back(Point2f(x, y));
            }
        }
    }
    return points;
}


vector<Point2f> slidingWindow(const Mat& in, Rect window){
    Mat current;
    in.copyTo(current);

    vector<Point2f> points;

    vector<Mat> debugFrames;

    cvtColor(in , current, COLOR_GRAY2BGR);

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
                avg = 1280 - 150;
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
        


        


        

        // draw the window and the avg point
        Mat frame;
        current.copyTo(frame);
        rectangle(frame, window, Scalar(0, 255, 0), 2);        

        if (!empty){
            circle(frame, Point(avg, window.y + window.height * 0.5), 5, Scalar(0, 0, 255), 3);  
        }
             

        debugFrames.push_back(frame);



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
   
    for (const auto& frame : debugFrames) {
        namedWindow("Sliding Window Playback", WINDOW_NORMAL);
        moveWindow("Sliding Window Playback", 300, 300);
        imshow("Sliding Window Playback", frame);
        waitKey(1000);  // Longer pause to visualize motion
    }
    destroyWindow("Sliding Window Playback");



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


Mat drawPoints(const Mat& in, vector<Point2f> left, vector<Point2f> right){


    // TODO possible improve least squares


    Mat current;
    in.copyTo(current);

    vector<Point2f> out;

    // draw left lines
    perspectiveTransform(left, out, inverted);
   
    for (int i = 0; i < out.size() - 1; i++ ){
       
        line(current, out[i], out[i + 1], Scalar(0, 255, 0), 3);
        
    }

    Mat lcurve = leastSquares(out);
    float la = lcurve.at<float>(0);
    float lb = lcurve.at<float>(1);
    float lc = lcurve.at<float>(2);

    
    vector<Point> fcp;
    for (int i = 0; i < out.size(); i++){
        float x = out[i].x;
        float y = la*x*x + lb*x + lc;
        //circle(current, Point(x, y), 3, Scalar(255, 0 , 0), 3);
        fcp.push_back(Point(cvRound(x), cvRound(y)));
    }
    polylines(current, fcp, false, Scalar(255, 0, 0), 3);
    out.clear();

    
    // draw right lines
    
    perspectiveTransform(right, out, inverted);

    for (int i = 0; i < out.size() - 1; i++ ){
        line(current, out[i], out[i + 1], Scalar(0, 255, 0), 3);
        
    }
   
    Mat rcurve = leastSquares(out);
    float ra = rcurve.at<float>(0);
    float rb = rcurve.at<float>(1);
    float rc = rcurve.at<float>(2);

   vector<Point> fittedCurvePoints;
   for (int i = 0; i < out.size(); i++){
        float x = out[i].x;
        float y = ra*x*x + rb*x + rc;
        fittedCurvePoints.push_back(Point(cvRound(x), cvRound(y)));
        //circle(current, Point(x, y), 3, Scalar(255, 0, 0), 3);
   }
    

   polylines(current, fittedCurvePoints, false, Scalar(255, 0, 0), 3);



    return current;

}




int main() {
    string path = "../img/straight_lines1.jpg";
    original = imread(path);

    if (original.empty()) {
        cerr << "Could not read image\n";
        return -1;
    }

    Mat current = original.clone();
    int stage = 0;


    while (true) {
        Mat display;
        string label;

        switch(stage){
            case 0:
                current = original.clone();
                display = current.clone();
                label = "1 - Original Frame";
                break;
            case 1:

                current = regionOfInterest(current);
                display = current.clone();
                label = "2 - Region of Interest";
                break;

            case 2 : {
                
                
                current = warp(current).clone();
                display = current.clone();  
                label = "3 - Perspective Transform";
                break;
            }
            case 3: {
                
                current = filterColor(current).clone();
                display = current.clone();
                label = "4 - Grayscale Frame";
                break;
            }    
            case 4: {
              
                current = smooth(current).clone();
                display = current.clone();
                label = "5 - Gaussian Blur";
                break;
            }
            case 5:{
                current = threshold(current).clone();
                display = current.clone();  
                label = "6 - Thresholding"; 
                break; 
            }
            case 6: {
                

                int width = 300;
                int height = 30;
                // get points of left and right lanes
                vector<Point2f> lpoints = slidingWindow(current, Rect(0, current.rows - height,  width, height));
                vector<Point2f> rpoints = slidingWindow(current, Rect(current.cols - width, current.rows - height, width, height));

                // draw lines over original image
                original = drawPoints(original, lpoints, rpoints);
            


                display = original.clone();
                
                label = "7 - Sliding Window";
                break;
            }
            case 7: {

                display = original.clone();
                label = "8 - Least Squares";
                break;
            }
            
            default:
                cout << "End of processing" << endl;
                return 0;
        }

        cout << label << endl;
        namedWindow("Output",WINDOW_NORMAL);
        moveWindow("Output", 100, 100);
        imshow("Output", display);
        int key = waitKey(0);
        if (key == 27) break;  // ESC key
        stage++;
    }

    return 0;
}
