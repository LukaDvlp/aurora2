/*!
 *  @file    disparity.cpp
 *  @author  Taiki Mashimo <mashimo.taiki@ac.jaxa.jp>
 *  @date    2015-6-15
 *  @brief   Caluculate disparity from stereo images, and get 3D depth data from them.
 */

#include <iostream>

#include <boost/python.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "np2cv.h"

/*!
 *   disparity(Caluculate disparity from stereo images)
 *  @param imL        Input imageL.
 *  @param imR        Input imageR.
 *  @param imD        Output disparity image.
 */

void disparity(const cv::Mat &imL, cv::Mat &imR, cv::Mat &imD) {

    //cv::StereoSGBM sgbm(1, 16*2, 3, 200, 255, 1, 0, 0, 0, 0, true);
    int w = imL.cols;
    int h = imL.rows;
    cv::Mat imL_(h/2, w/2, CV_8UC1);
    cv::Mat imR_(h/2, w/2, CV_8UC1);
    cv::Mat imD_;
    cv::resize(imL, imL_, imL_.size());
    cv::resize(imR, imR_, imR_.size());

    int max_disp = 16 * 3;
    int sad_win = 5;
    double contrast_threshold = 0.8;
    double uniqueness_threshold = 30;
    double distance_threshold = 1;

    int P1 = 8  * sad_win * sad_win;
    int P2 = 32 * sad_win * sad_win;
    int prefiltercap = 63. * contrast_threshold;
    
    int speckle_win = 150;
    int speckle_range = 2;
    cv::StereoSGBM sgbm(0, max_disp, sad_win, P1, P2, distance_threshold, prefiltercap, uniqueness_threshold, speckle_win, speckle_range, false);
    sgbm(imL_, imR_, imD_);

    imD.create(h, w, imD_.type());
    cv::resize(imD_ * 2, imD, imD.size());
}

/*!
 *   reproject(Caluculate 3D coordinates from disparity by Q matrix)
 *  @param imD       Input disparity image.
 *  @param im3D      Output xyz 3D data.
 *  @param Q         Input Q matrix.
 */

void reproject(const cv::Mat &imD, cv::Mat &im3D, cv::Mat &Q) {

    cv::reprojectImageTo3D(imD, im3D, Q);

}


#ifdef UNITTEST

/*!
 *  Test function
 *  Usage   : ./dense_stereo.out tsukuba_l.png tsukuba_r.png > ***.ply
 *  Input   : stdin (the 1st argument: left image, the 2nd argument: right image)
 *  Output  : stdout (ply script)
 *  write ply data from stdout to ***.ply
 *  output range z: MIX_DEPTH_Z < z < MAX_DEPTH_Z
 *  output range y: MIX_DEPTH_Y < z < MAX_DEPTH_Y
 */

//define output range
#define MAX_DEPTH_Z 2000
#define MIN_DEPTH_Z 0
#define MAX_DEPTH_Y 2000
#define MIN_DEPTH_Y -2000

cv::Mat makeQMatrix(cv::Point2d image_center, double focal_length, double baseline) {
    cv::Mat Q = cv::Mat::eye(4, 4, CV_64F);
    Q.at<double>(0,3) = -image_center.x;
    Q.at<double>(1,3) = -image_center.y;
    Q.at<double>(2,3) = focal_length;
    Q.at<double>(3,3) = 0.0;
    Q.at<double>(2,2) = 0.0;
    Q.at<double>(3,2) = 1.0/baseline;

    return Q;
}

void exportPLY(cv::Mat& im3D, cv::Mat& imL) {

    //1.count total points:(x,y,z)
    int counter=0;
    for(int j=0; j<im3D.rows; j++) {
        for(int i=0; i<im3D.cols; i++) {
            float y, z;
            z = im3D.at<cv::Vec3f>(j, i)[2];
            y = im3D.at<cv::Vec3f>(j, i)[1];

            if(z<MAX_DEPTH_Z && z>MIN_DEPTH_Z && y<MAX_DEPTH_Y && y>MIN_DEPTH_Y)    counter++;
        }
    }

    //2.write ply header
    std::cout << "ply" << std::endl;
    std::cout << "format ascii 1.0" << std::endl;
    std::cout << "element vertex " << counter << std::endl;
    std::cout << "property float x" << std::endl;
    std::cout << "property float y" << std::endl;
    std::cout << "property float z" << std::endl;
    std::cout << "property uchar red" << std::endl;
    std::cout << "property uchar green" << std::endl;
    std::cout << "property uchar blue" << std::endl;
    std::cout << "end_header"  << std::endl;

    //3.write ply data
    for(int j=0; j<im3D.rows; j++) {
        for(int i=0; i<im3D.cols; i++) {

            float x, y, z;
            x = im3D.at<cv::Vec3f>(j, i)[0];
            y = im3D.at<cv::Vec3f>(j, i)[1];
            z = im3D.at<cv::Vec3f>(j, i)[2];
            float r, g, b;
            b = imL.at<cv::Vec3b>(j, i)[0];
            g = imL.at<cv::Vec3b>(j, i)[1];
            r = imL.at<cv::Vec3b>(j, i)[2];

            if(z<MAX_DEPTH_Z && z>MIN_DEPTH_Z && y<MAX_DEPTH_Y && y>MIN_DEPTH_Y) {
                std::cout << x << " " ;
                std::cout << y << " " ;
                std::cout << z << " " ;
                std::cout << r << " " ;
                std::cout << g << " " ;
                std::cout << b << " " << std::endl;
            }

        }
    }

}

int main(int argc, char **argv) {

    //1.Input stereo images.
    std::string filename1 = argv[1];
    std::string filename2 = argv[2];

    cv::Mat imL = cv::imread(filename1, 1);
    cv::Mat imR = cv::imread(filename2, 1);
    cv::Mat imD;
    cv::Mat im3D;

    //2.Erorr check.
    if (!imL.data) {
        std::cerr << "imageL not loaded" << std::endl;
        return -1;
    }

    if (!imR.data) {
        std::cerr << "imageR not loaded" << std::endl;
        return -1;
    }

    //3.Caluculate disparity.
    disparity(imL, imR, imD);

    //4.Make Q matrix.
    const double focal_length = 598.57; //temporary value
    const double baseline = 14.0; //temporary value
    cv::Mat Q = makeQMatrix(cv::Point2d((imD.cols-1.0)/2.0, (imD.rows-1.0)/2.0), focal_length, baseline*16);

    //5.Get depth image.
    reproject(imD, im3D, Q);

    //6.Output stereo images, disparity image and depth data.
    exportPLY(im3D, imL);

    return 0;
}


#else

/*!
 *  Wrapper function for Python-C++ bridging
 */

PyObject* disparity_wrapper(PyObject *imL, PyObject *imR) {
    NDArrayConverter cvt;
    cv::Mat _imL = cvt.toMat(imL);
    cv::Mat _imR = cvt.toMat(imR);
    cv::Mat _imD;
    disparity(_imL, _imR, _imD);
    return cvt.toNDArray(_imD);
}

PyObject* reproject_wrapper(PyObject *imD, PyObject *Q) {
    NDArrayConverter cvt;
    cv::Mat _imD = cvt.toMat(imD);
    cv::Mat _Q = cvt.toMat(Q);
    cv::Mat _im3D;
    reproject(_imD, _im3D, _Q);
    return cvt.toNDArray(_im3D);
}


BOOST_PYTHON_MODULE(dense_stereo)
{
    Py_Initialize();
    import_array();
    boost::python::def("disparity", disparity_wrapper);
    boost::python::def("reproject", reproject_wrapper);
}


#endif

