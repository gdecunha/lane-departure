
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include "LaneDetector.h"


using namespace std;
using namespace cv;
using namespace std::chrono;

#define WIDTH 1280
#define HEIGHT 720





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
  
    auto startT = high_resolution_clock::now();
    double avgFPS = 0;
    int frame_count = 0;
    while(true){
        int64 start = getTickCount();


        cap >> frame;
        if (frame.empty()){
            cout << "End of video\n";
            break;
        }

        
        Mat processed = detector.processFrame(frame);
        

        int key = waitKey(1);
        if(key==27)break;
        
        int64 end = cv::getTickCount();  // <-- End timer
        double duration = (end - start) / cv::getTickFrequency();  // seconds
        double fps = 1.0 / duration;
        avgFPS += fps;
        frame_count++;

        cout << "FPS: " << fps << endl;

    }



    auto endT = high_resolution_clock::now();
    duration<double> durationT = endT - startT;
    cout<< "duration: " << durationT.count() << endl;

    
    avgFPS /= frame_count;
    cout << "Average FPS: " << avgFPS << endl;

    cap.release();
    destroyAllWindows();
    cout << "Video processing completed\n";
    return 0;
}