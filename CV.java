import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class FaceDetection {
    public static void main(String[] args) {
        // Load OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
        // Load the Haar Cascade classifier for face detection
        String classifierPath = "haarcascade_frontalface_default.xml";
        CascadeClassifier faceDetector = new CascadeClassifier(classifierPath);
        
        // Check if classifier loaded properly
        if (faceDetector.empty()) {
            System.err.println("Error loading classifier file: " + classifierPath);
            return;
        }
        
        // Open the default camera (0)
        VideoCapture camera = new VideoCapture(0);
        
        if (!camera.isOpened()) {
            System.err.println("Error: Camera not accessible");
            return;
        }
        
        // Create a window
        String windowName = "Face Detection";
        opencv_highgui.namedWindow(windowName, opencv_highgui.WINDOW_AUTOSIZE);
        
        Mat frame = new Mat();
        
        while (true) {
            // Read a frame from the camera
            if (camera.read(frame)) {
                // Convert to grayscale
                Mat grayFrame = new Mat();
                Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
                
                // Detect faces
                MatOfRect faceDetections = new MatOfRect();
                faceDetector.detectMultiScale(grayFrame, faceDetections);
                
                // Draw rectangles around detected faces
                for (Rect rect : faceDetections.toArray()) {
                    Imgproc.rectangle(
                        frame, 
                        new Point(rect.x, rect.y),
                        new Point(rect.x + rect.width, rect.y + rect.height),
                        new Scalar(0, 255, 0), // Green color
                        3 // Thickness
                    );
                }
                
                // Show the frame
                opencv_highgui.imshow(windowName, frame);
                
                // Exit on ESC key
                if (opencv_highgui.waitKey(10) == 27) {
                    break;
                }
            }
        }
        
        // Release resources
        camera.release();
        opencv_highgui.destroyAllWindows();
    }
}