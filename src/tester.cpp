#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;


Mat original;
vector<Point2f> srcpoints;


Mat regionOfInterest(const Mat& img){
    Mat mask = Mat::zeros(img.size(), img.type());

    int height = img.rows;
    int width = img.cols;

    printf("Image size: %d x %d\n", width, height);

    srcpoints.clear();
    srcpoints.push_back(Point2f(560, 470));  // top left
    srcpoints.push_back(Point2f(720, 470));  // top right 
    srcpoints.push_back(Point2f(1100, 680)); // bottom right 
    srcpoints.push_back(Point2f(200, 680)); // bottom left

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


int main() {
    string path = "straight_lines1.jpg";
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
                label = "5 - Region of Interest";
                break;
            case 2:
                
                cvtColor(current, current, COLOR_BGR2GRAY);
                display = current.clone();
                label = "2 - Grayscale Frame";
                break;
                
            case 3: {
                Mat blurred;
                GaussianBlur(current, blurred, Size(5, 5), 0);
                current = blurred.clone();
                display = current.clone();
                label = "3 - Gaussian Blur";
                break;
            }
            case 4: {
                Mat edges;
                Canny(current, edges, 50, 150);
                current = edges.clone();
                display = current.clone();
                label = "4 - Canny Edge Detection";
                break;
            }
            case 5 : {
                
                vector<Point2f> dstpoints = {
                    Point2f(0, 0), 
                    Point2f(1280, 0), 
                    Point2f(1280, 720), 
                    Point2f(0, 720)};

                Mat perspectiveMatrix = getPerspectiveTransform(srcpoints, dstpoints);

                Mat warped;
                warpPerspective(current, warped, perspectiveMatrix, Size(1280, 720));
                current = warped.clone();
                display = current.clone();  
                label = "6 - Perspective Transform";
                break;
            }
            default:
                cout << "End of processing" << endl;
                return 0;
        }

        cout << label << endl;
        imshow("Output", display);
        int key = waitKey(0);
        if (key == 27) break;  // ESC key
        stage++;
    }

    return 0;
}
