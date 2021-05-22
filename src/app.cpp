#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define PATTERN_SIZE Size(11, 8)
#define PATTERN_SCALE 30


class CalibratorApp
{
private:
    string config_path;
    bool fish_eye = false;
    Size image_size;
    Mat K, D;
    Mat R, t;
    Mat map1, map2;

    bool save()
    {
        FileStorage fs(this->config_path, FileStorage::WRITE);
        fs << "fish_eye" << fish_eye;
        fs << "image_size" << image_size;
        fs << "K" << K;
        fs << "D" << D;
        // fs << "R" << R;
        // fs << "t" << t;
        fs.release();

        cout << "========== storage intrinsics finish." << endl;
    }

    void init()
    {
        FileStorage fs(this->config_path, FileStorage::READ);
        fs["fish_eye"] >> fish_eye;
        fs["image_size"] >> image_size;
        fs["K"] >> K;
        fs["D"] >> D;
        // fs["R"] >> R;
        // fs["t"] >> t;
        fs.release();

        initUndistortRectifyMap(K, D, Mat(), K, image_size, CV_32FC1, map1, map2);

        cout << "========== init finish!" << endl;
    }
    
public:
    CalibratorApp()
    {
        config_path = "../config/camera_intrinsics_extrinsics.yaml";
        if (1)
        {
            CalibrateIntrinsics();
            // CalibrateExtrinsics();
            save();
        }
        init();
    }

    Mat UndistortImage(Mat frame)
    {
        Mat dst;
        remap(frame, dst, this->map1, this->map2, INTER_LINEAR);
        return dst;
    }

    void CalibrateIntrinsics()
    {
        String img_pattern = "../images/intrinsics/*.png";
        vector<String> v_path;
        glob(img_pattern, v_path, false);
        this->image_size = imread(v_path[0]).size();

        // 查找棋盘格图片，二次精确棋盘格角点的像素坐标
        vector<vector<Point2f>> corners;
        for(size_t i = 0, count = v_path.size(); i < count; i++)
        {
            Mat image = imread(v_path[i]);
            Mat gray;
            cvtColor(image, gray, COLOR_BGR2GRAY);
            vector<Point2f> points;
            bool found = findChessboardCorners(gray, PATTERN_SIZE, points, CALIB_CB_ADAPTIVE_THRESH|CALIB_CB_FAST_CHECK|CALIB_CB_NORMALIZE_IMAGE);
            if(found)
            {
                // string path = v_path[i];
                // path = path.replace(path.find("intrinsics"), 10, "draw");
                TermCriteria criteria(TermCriteria::Type::MAX_ITER, 30, 0.001);
                cornerSubPix(gray, points, Size(5, 5), Size(-1, -1), criteria);
                drawChessboardCorners(image, PATTERN_SIZE, points, found);
                // imwrite(path, image);
                corners.push_back(points);
            }
            
        }

        // 生成标定板坐标序列，每张图片对应一个真实坐标系的点集
        vector<vector<Point3f>> object_points;
        for(size_t t = 0, count = corners.size(); t < count; t++)
        {
            vector<Point3f> v_points;
            for(size_t i = 0; i < 8; i++)
            {
                for(size_t j = 0; j < 11; j++)
                {
                    Point3f real_point;
                    real_point.x = j * PATTERN_SCALE;
                    real_point.y = i * PATTERN_SCALE;
                    real_point.z = 0;
                    v_points.push_back(real_point);
                }
            }
            object_points.push_back(v_points);
        }
        // 迭代求解相机内参
        vector<Mat> rvecs, tvecs;
        bool success = false;
        if(fish_eye) {
            int flags = 0;
            flags |= fisheye::CALIB_RECOMPUTE_EXTRINSIC;
            flags |= fisheye::CALIB_CHECK_COND;
            flags |= fisheye::CALIB_FIX_SKEW;
            success = fisheye::calibrate(object_points, corners, image_size, this->K, this->D, rvecs, tvecs, flags, TermCriteria(3, 20, 1e-6));
        } else {
            success = calibrateCamera(object_points, corners, image_size, this->K, this->D, rvecs, tvecs, 0);
        }

        cout << "========== calibrate finish:" << success << endl;

    }

    void CalibrateExtrinsics()
    {
        vector<Point3f> object_points;
        vector<Point2f> image_points;

        Mat rvec;
        // opencv4: SOLVEPNP_ITERATIVE
        // opencv3: CV_ITERATIVE
        solvePnP(object_points, image_points, this->K, this->D, rvec, this->t, false, SOLVEPNP_ITERATIVE);
        Rodrigues(rvec, this->R);
    }

};

int main()
{
    CalibratorApp app;
    VideoCapture cap;
    cap.open(4, CAP_ANY);
    Mat frame;
    while(1)
    {
        cap >> frame;
        if(frame.empty())
        {
            cout << "read fail!" << endl;
            break;
        }
        frame = app.UndistortImage(frame);
        imshow("live", frame);
        waitKey(1);
    }
    destroyAllWindows();
    return 0;
}