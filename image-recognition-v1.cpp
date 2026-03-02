#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 1. Download image from this web link
    string url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/smarties.png";
    VideoCapture cap(url);
    Mat img;
    cap >> img;

    if (img.empty()) {
        cout << "Error: Could not download image from link." << endl;
        return -1;
    }

    // 2. Process the image
    Mat gray, canned;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Canny(gray, canned, 50, 150);

    vector<vector<Point>> contours;
    findContours(canned, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++) {
        vector<Point> approx;
        approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);
        if (approx.size() == 4) {
            rectangle(img, boundingRect(approx), Scalar(0, 255, 0), 2);
        }
    }

    // 3. Save the result for the browser
    imwrite("output.jpg", img);
    cout << "SUCCESS! Check your browser link now." << endl;
    return 0;
}