#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Load the pre-trained Haar Cascade face detector
    CascadeClassifier faceCascade;
    String faceCascadePath = "haarcascade_frontalface_default.xml";
    
    if (!faceCascade.load(faceCascadePath)) {
        cerr << "Error loading face cascade file" << endl;
        return -1;
    }

    // Initialize video capture from default camera
    VideoCapture cap(0);
    
    if (!cap.isOpened()) {
        cerr << "Error opening video stream" << endl;
        return -1;
    }

    Mat frame;
    namedWindow("Face Detection", WINDOW_AUTOSIZE);

    while (true) {
        // Capture frame-by-frame
        cap >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        // Convert to grayscale
        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        // Detect faces
        vector<Rect> faces;
        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 5, 0, Size(30, 30));

        // Draw rectangles around the faces
        for (const Rect& face : faces) {
            rectangle(frame, face, Scalar(0, 255, 0), 2);
        }

        // Display the resulting frame
        imshow("Face Detection", frame);

        // Press ESC to exit
        if (waitKey(1) == 27)
            break;
    }

    // Release the video capture object and close windows
    cap.release();
    destroyAllWindows();

    return 0;
}