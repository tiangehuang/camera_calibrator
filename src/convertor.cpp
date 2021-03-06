#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Convertor
{
private:
    Size image_size;
    Mat K, D;
    Mat R, T;
    Mat map1, map2;
public:
    Convertor()
    {
        string config_path = "../config/calibrrator.yaml";
        FileStorage fs(config_path, FileStorage::READ);
        fs["image_size"] >> image_size;
        fs["K"] >> K;
        fs["D"] >> D;
        fs["R"] >> R;
        fs["T"] >> T;
        fs.release();

        initUndistortRectifyMap(K, D, Mat(), K, image_size, CV_32FC1, map1, map2);

        cout << "========== init finish!" << endl;
    }

    Point3d i2w(const Point2d &point)
    {
        double Z = 0.;
        Mat pih = Mat::ones(3, 1, CV_64F);
        pih.at<double>(0, 0) = point.x;
        pih.at<double>(1, 0) = point.y;

        Mat LS = R.inv() * K.inv() * pih;
        Mat RS = R.inv() * T;
        double s = (Z + RS.at<double>(2, 0)) / LS.at<double>(2, 0);
        Mat pw = s * LS - RS;

        return Point3d(pw);
    }

    Point2d w2i(const Point3d &point)
    {
        Mat RT;
        hconcat(R, T, RT);

        Mat pwh = Mat::ones(4, 1, CV_64F);
        pwh.at<double>(0, 0) = point.x;
        pwh.at<double>(1, 0) = point.y;
        pwh.at<double>(2, 0) = point.z;

        Mat pih = K * RT * pwh;
        pih = pih / pih.at<double>(2, 0);

        return Point2d(pih.at<double>(0, 0), pih.at<double>(1, 0));
    }
};

int main()
{
    Mat src = imread("../images/extrinsics/2u.jpg");
    Convertor convertor;
    Point3d pw(0, 10000, 0);
    Point2d pi = convertor.w2i(pw);
    cout << "pi:" << pi << endl;

    //Point2d pi(857.90546, 988.28033);
    pw = convertor.i2w(pi);
    cout << "pw:" << pw << endl;
    circle(src, pi, 5, Scalar(255, 0, 0), -1);
    imshow("draw", src);
    waitKey(0);
    return 0;
}
