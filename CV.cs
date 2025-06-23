using OpenCvSharp;
using System;
using System.Collections.Generic;

class FaceDetection
{
    static void Main()
    {
        // Load the pre-trained Haar Cascade face detector
        var faceCascade = new CascadeClassifier();
        if (!faceCascade.Load("haarcascade_frontalface_default.xml"))
        {
            Console.WriteLine("Error loading face cascade file");
            return;
        }

        // Initialize video capture from default camera
        using var capture = new VideoCapture(0);
        if (!capture.IsOpened())
        {
            Console.WriteLine("Error opening video stream");
            return;
        }

        using var window = new Window("Face Detection");
        using var frame = new Mat();

        while (true)
        {
            // Capture frame-by-frame
            capture.Read(frame);

            // If the frame is empty, break immediately
            if (frame.Empty())
                break;

            // Convert to grayscale
            using var grayFrame = new Mat();
            Cv2.CvtColor(frame, grayFrame, ColorConversionCodes.BGR2GRAY);

            // Detect faces
            Rect[] faces = faceCascade.DetectMultiScale(
                grayFrame,
                scaleFactor: 1.1,
                minNeighbors: 5,
                flags: HaarDetectionTypes.ScaleImage,
                minSize: new Size(30, 30)
            );

            // Draw rectangles around the faces
            foreach (var face in faces)
            {
                Cv2.Rectangle(frame, face, Scalar.Green, thickness: 2);
            }

            // Display the resulting frame
            window.ShowImage(frame);

            // Press ESC to exit
            if (Cv2.WaitKey(1) == 27)
                break;
        }
    }
}