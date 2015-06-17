/*!
 *  @file    libviso2.cpp
 *  @author  Kyohei Otsu <kyon@ac.jaxa.jp>
 *  @date    2015-06-13
 *  @brief   Visual Odometry using LIBVISO2 library
 */

#include <iostream>

#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "np2cv.h"
#include "viso_stereo.h"


boost::shared_ptr<VisualOdometryStereo::parameters> viso_param_;
boost::shared_ptr<VisualOdometryStereo> viso_;
cv::Mat wTc_;


void setup(const cv::Mat &p) {
    viso_param_.reset(new VisualOdometryStereo::parameters);
    viso_param_->calib.f  = p.at<FLOAT>(0);  // focal length in pixels
    viso_param_->calib.cu = p.at<FLOAT>(1);  // principal point (u-coord) in pixels
    viso_param_->calib.cv = p.at<FLOAT>(2);  // principal point (v-coord) in pixels
    viso_param_->base     = p.at<FLOAT>(3);  // baseline in meters

    viso_.reset(new VisualOdometryStereo(*viso_param_));
}


/*!
 *  Update pose from monocular images
 *  TODO: not implemented
 */
void update_mono(const cv::Mat &im, cv::Mat &wTc) {
    if (wTc.cols != 4) wTc = cv::Mat::eye(4, 4, CV_64F);
    std::cerr << "WARNING: Function update_mono() is not implemented!" << std::endl;
}


/*!
 *  Update pose from stereo images
 *  @param imL Left image
 *  @param imR Right image
 *  @param wTc Output 4x4 transformation matrix 
 */
void update_stereo(const cv::Mat &imL, const cv::Mat &imR, cv::Mat &wTc) {
    static int32_t dims[] = {imL.cols, imL.rows, imL.cols};
    if (wTc.cols != 4) wTc = cv::Mat::eye(4, 4, CV_64F);

    if (viso_->process(imL.data, imR.data, dims)) {
        Matrix motion = Matrix::inv(viso_->getMotion());
        cv::Mat pTc(4, 4, CV_64F, &motion.val[0][0]);
        wTc = wTc * pTc;
    }
    else {
        // TODO continue previous motion in case of failure
    }
}



#ifdef UNITTEST

/*!
 *  Test function
 */
int main(int argc, char **argv) {

    cv::Mat imLp = cv::imread("libviso2/img/I1p.png", 0);
    cv::Mat imRp = cv::imread("libviso2/img/I2p.png", 0);
    cv::Mat imLc = cv::imread("libviso2/img/I1c.png", 0);
    cv::Mat imRc = cv::imread("libviso2/img/I2c.png", 0);

    cv::Mat p = (cv::Mat_<double>(4, 1) << 480, 0.5 * 1344, 0.5 * 391, 0.5);
    setup(p);

    cv::imshow("image", imLp);
    cv::waitKey(500);
    cv::imshow("image", imLc);
    cv::waitKey(500);

    cv::Mat wTc;
    cv::Mat Cc = (cv::Mat_<double>(4, 1) << 0, 0, 0, 1);
    update_stereo(imLp, imRp, wTc);
    std::cout << wTc * Cc << std::endl << std::endl;
    update_stereo(imLc, imRc, wTc);
    std::cout << wTc * Cc << std::endl << std::endl;
    update_stereo(imLc, imRc, wTc);
    std::cout << wTc * Cc << std::endl << std::endl;
    update_stereo(imLc, imRc, wTc);
    std::cout << wTc * Cc << std::endl << std::endl;
    update_stereo(imLp, imRp, wTc);
    std::cout << wTc * Cc << std::endl << std::endl;

    return 0;
}

#else

/*!
 *  Wrapper function for Python-C++ bridging
 */
PyObject* setup_wrapper(PyObject *param) {
    NDArrayConverter cvt;
    cv::Mat _param = cvt.toMat(param);
    setup(_param);
}

PyObject* update_stereo_wrapper(PyObject *imL, PyObject *imR) {
    NDArrayConverter cvt;
    cv::Mat _imL = cvt.toMat(imL);
    cv::Mat _imR = cvt.toMat(imR);
    update_stereo(_imL, _imR, wTc_);
    return cvt.toNDArray(wTc_);
}

PyObject* update_mono_wrapper(PyObject *im) {
    NDArrayConverter cvt;
    cv::Mat _im = cvt.toMat(im);
    update_mono(_im, wTc_);
    return cvt.toNDArray(wTc_);
}


BOOST_PYTHON_MODULE(libviso2)
{
    Py_Initialize();
    import_array();
    boost::python::def("setup", setup_wrapper);
    boost::python::def("update_mono", update_mono_wrapper);
    boost::python::def("update_stereo", update_stereo_wrapper);
}

#endif

