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
cv::Mat bTi_;
cv::Mat iTb_;


void setup(const cv::Mat &K, const double baseline, const cv::Mat bTi=cv::Mat::eye(4, 4, CV_64F)) {
    viso_param_.reset(new VisualOdometryStereo::parameters);
    viso_param_->calib.f  = K.at<double>(0, 0);
    viso_param_->calib.cu = K.at<double>(0, 2);
    viso_param_->calib.cv = K.at<double>(1, 2);
    viso_param_->base     = baseline;
    bTi_ = bTi;
    iTb_ = bTi_.inv();

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
 *  @param wTb Transformation matrix from world to current rover base frame
 *  @param bTi Transformation matrix from rover base to left image coordinates (optional)
 */
void update_stereo(const cv::Mat &imL, const cv::Mat &imR, cv::Mat &wTb) {
    static int32_t dims[] = {imL.cols, imL.rows, imL.cols};
    static cv::Mat ppbTpb = cv::Mat::eye(4, 4, CV_64F);
    if (wTb.cols != 4) wTb = cv::Mat::eye(4, 4, CV_64F);

    if (viso_->process(imL.data, imR.data, dims)) {
        // compute motion
        Matrix motion = Matrix::inv(viso_->getMotion());
        cv::Mat piTci(4, 4, CV_64F, &motion.val[0][0]);
        cv::Mat pbTcb = bTi_ * piTci * iTb_;

        // update global pose
        wTb *= pbTcb;
        pbTcb.copyTo(ppbTpb);
    }
    else {
        wTb *= ppbTpb;
    }
}



#ifdef UNITTEST

/*!
 *  For test function, see test_libviso2.py
 */
int main(int argc, char **argv) {
    std::cout << "Sorry! Test function is not provided for C++." << std::endl;
    return 0;
}

#else

/*!
 *  Wrapper function for Python-C++ bridging
 */
PyObject* setup_wrapper(PyObject *rover) {
    NDArrayConverter cvt;
    cv::Mat K = cvt.toMat(PyObject_GetAttrString(rover, "KL"));
    double baseline = boost::python::extract<double>(PyObject_GetAttrString(rover, "baseline"));
    cv::Mat bTi = cvt.toMat(PyObject_GetAttrString(rover, "bTi"));
    setup(K, baseline, bTi);
}

PyObject* setup2_wrapper(PyObject *param) {
    NDArrayConverter cvt;
    std::cout << cvt.toMat(PyObject_GetAttrString(param, "bTi")) << std::endl;
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

