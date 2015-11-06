/*!
 *  @file    dem_impl.cpp
 *  @author  Kyohei Otsu <kyon@ac.jaxa.jp>
 *  @date    2015-10-14
 *  @brief   DEM generation with Layered V-Disparity method, etc.
 */

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "interp.h"


//! Global variables
class DemParams {
  public:
    DemParams() 
    {
    }

    ~DemParams() { }

    /* config variables */
    cv::Mat resolution;  // in pixels
    double dpm;  // dots per meter
    double grid_size;  // meter in each cell
    cv::Mat ranges;  // [[minX, maxX], [minY, maxY], [minZ, maxZ]]
    int max_disparity;  // in pixels

    cv::Mat K;
    cv::Mat Q;
    cv::Mat bTi;  // camera to base_link
    cv::Mat iTb;  // base_link to camera


    /* utility variables (should be initialized in setup()) */
    int Nx, Ny;

    double min_x, max_x;
    double min_y, max_y;
    double min_z, max_z;


    cv::Mat mask;


    void setup()
    {
        //grid_size = 1.0 / dpm * 10;  // magic number
        min_x = ranges.at<double>(0, 0);
        max_x = ranges.at<double>(0, 1);
        min_y = ranges.at<double>(1, 0);
        max_y = ranges.at<double>(1, 1);
        min_z = ranges.at<double>(2, 0);
        max_z = ranges.at<double>(2, 1);
        Nx = ceil((max_x - min_x) / grid_size);
        Ny = ceil((max_y - min_y) / grid_size);

        makeMask();
    }


    void makeMask()
    {
        int rows = round(K.at<double>(1, 2) * 2) + 1;
        int cols = round(K.at<double>(0, 2) * 2) + 1;
        mask = cv::Mat::zeros(rows, cols, CV_8U);

        // compute transformation
        cv::Mat rvec, tvec;
        cv::Rodrigues(iTb(cv::Rect(0, 0, 3, 3)), rvec);
        tvec = iTb(cv::Rect(3, 0, 1, 3));

        // project points
        std::vector<cv::Point3f> gnd_pts((Ny + 1) * 2);
        std::vector<cv::Point2f> img_pts;
        std::vector<cv::Point> img_ptsi;

        for (int i = 0; i <= Ny; ++i)
        {
            gnd_pts.at(2 * i).x = min_x;
            gnd_pts.at(2 * i).y = min_y + i * grid_size;
            gnd_pts.at(2 * i).z = 0;
            gnd_pts.at(2 * i + 1).x = max_x;
            gnd_pts.at(2 * i + 1).y = min_y + i * grid_size;
            gnd_pts.at(2 * i + 1).z = 0;

            if (i % 2) std::iter_swap(gnd_pts.begin() + 2 * i, gnd_pts.begin() + 2 * i + 1);
        }

	cv::Mat d = cv::Mat::zeros(1, 4, CV_64F);
        cv::projectPoints(gnd_pts, rvec, tvec, K, d, img_pts);
        cv::Mat(img_pts).convertTo(img_ptsi, cv::Mat(img_ptsi).type());

        // draw polygons on mask image
        for (size_t i = 0; i < Ny; ++i)
        {
            std::vector<cv::Point> cpts(img_ptsi.begin() + 2 * i, img_ptsi.begin() + 2 * i + 4);
            const cv::Point *_cpts[1] = { &cpts[0] };
            int npts[1] = { cpts.size() };
            cv::fillPoly(mask, _cpts, npts, 1, i + 1, 8);
        }

    }

    double getXc(int i) { return min_x + grid_size * i + grid_size / 2; }
    double getYc(int i) { return min_y + grid_size * i + grid_size / 2; }
    double getXl(int i) { return min_x + grid_size * i; }
    double getYl(int i) { return min_y + grid_size * i; }
    double getXh(int i) { return min_x + grid_size * i + grid_size; }
    double getYh(int i) { return min_y + grid_size * i + grid_size; }

    double getU(double x) { return x * dpm + resolution.at<double>(0) / 2; }
    double getV(double y) { return y * dpm + resolution.at<double>(1) / 2; }
};

DemParams params_;

/*!
 *  Generate v-disparity image from disparity map
 */
void convertVD(const cv::Mat &imD, cv::Mat &imVD, cv::Mat mask=cv::Mat())
{
    if (!mask.data) mask = cv::Mat::ones(imD.size(), CV_8U);
    imVD = cv::Mat::zeros(imD.rows, params_.max_disparity, CV_8U);

    for (int v = 0; v < imD.rows; ++v)
    {
        const double *imD_v  = imD.ptr<double>(v);
        const uchar  *mask_v = mask.ptr<uchar>(v);
              uchar  *imVD_v = imVD.ptr<uchar>(v);
        for (int u = 0; u < imD.cols; ++u)
        {
            if (mask_v[u]) imVD_v[(int)round(imD_v[u])]++;
        }
    }
}

/*!
 *  Compute main ground curve from v-disparity image
 */

void computeMGC_DynProg(const cv::Mat &imVD, cv::Mat &mgc)
{
    // find min/max valid v (row coordinates)
    int min_v = 0, max_v = 0;
    cv::Mat vecV;
    cv::reduce(imVD, vecV, 1, CV_REDUCE_MAX);
    for (int v = 0; v < imVD.rows; ++v)
    {
        if (vecV.at<uchar>(v, 0) > 0) 
        {
            if (min_v == 0) min_v = v;
            max_v = v;
        }
    }

    int max_u = 0;
    for (int u = 0; u < imVD.cols; ++u)
    {
        if (imVD.at<uchar>(max_v, u) > 0) max_u = u;
    }

    // compute cost map
    cv::Mat cmap(max_v - min_v + 1, imVD.cols, CV_16U);
    imVD(cv::Rect(0, min_v, imVD.cols, max_v - min_v + 1)).convertTo(cmap, cmap.type());
    for (int i = 1; i < cmap.rows; ++i)
    {
        cv::Mat lines = cv::Mat::zeros(3, cmap.cols, cmap.type());
        for (int j = 0; j < cmap.cols - 1; ++j)
            lines.at<ushort>(0, j) = cmap.at<ushort>(i - 1, j + 1);
        for (int j = 0; j < cmap.cols    ; ++j)
            lines.at<ushort>(0, j) = cmap.at<ushort>(i - 1, j);
        for (int j = 0; j < cmap.cols - 1; ++j)
            lines.at<ushort>(0, j + 1) = cmap.at<ushort>(i - 1, j);
        //cmap(cv::Rect(1, i - 1, cmap.cols - 1, 1)).copyTo(lines(cv::Rect(0, 0, lines.cols - 1, 1)));
        //cmap(cv::Rect(0, i - 1, cmap.cols    , 1)).copyTo(lines(cv::Rect(0, 1, lines.cols    , 1)));
        //cmap(cv::Rect(0, i - 1, cmap.cols - 1, 1)).copyTo(lines(cv::Rect(1, 2, lines.cols - 1, 1)));

        cv::Mat max_line;
        cv::reduce(lines, max_line, 0, CV_REDUCE_MAX);
        cmap(cv::Rect(0, i, cmap.cols, 1)) += max_line;
    }
    //cv::imshow("cost map", cmap * 5);
    //cv::waitKey(1);

    // compute main ground curve
    std::vector<cv::Point> route;  // in uncropped image coordinates
    double max_val = 0;
    cv::Point max_loc(-1, -1);
    int wsize = 5;

    cv::minMaxLoc(cmap(cv::Rect(0, cmap.rows - 1, cmap.cols, 1)), NULL, &max_val, NULL, &max_loc);
    route.push_back(cv::Point(max_loc.x, max_v));

    for (int v = cmap.rows - 1; v >= 0; --v)  // backtrack
    {
        int last_u = route.back().x;

        // copy elements in window
        cv::Mat cmap_w = cv::Mat::zeros(1, wsize, cmap.type());
        for (int j = 0; j < wsize; ++j)
        {
            try { cmap_w.at<ushort>(0, j) = cmap.at<ushort>(v, last_u - int(wsize / 2) + j); }
            catch (std::exception &e) { } 
        }
        cmap_w.at<ushort>(0, int(wsize / 2)) += 1;  // magic!

        // find maxima
        cv::minMaxLoc(cmap_w, NULL, &max_val, NULL, &max_loc);

        // add new route
        int new_u = last_u + max_loc.x - int(wsize / 2);
        route.push_back(cv::Point(new_u, min_v + v));
    }

    bool debug = false;
    if (debug)
    {
        cv::Mat imMGC = cv::Mat::zeros(imVD.size(), CV_8U);
        for (int i = 1; i < route.size(); ++i)
        {
            cv::line(imMGC, route.at(i - 1), route.at(i), 255, 1);
        }
        //cv::imshow("vd", imVD * 100);
        //cv::imshow("mgc", imMGC);
        //cv::waitKey(-1);
    }

    mgc = cv::Mat(route).reshape(1).t();
}


/*!
 *  Converts a disparity image to a DEM (digital elevation map) with Layered V-disparity method
 *  @param imD        Input disparity image (single-channel, float).
 *  @param dem        Output DEM image (single-channel, float).
 */
void lvd(const cv::Mat &imD, cv::Mat &dem) {

    dem = cv::Mat::zeros(params_.resolution.at<double>(1), params_.resolution.at<double>(0), CV_64F);
    //dem = params_.mask.clone();

    for (int i = 0; i < params_.Ny; ++i)
    {
        // generate V-disparity
        cv::Mat imVD;
        convertVD(imD, imVD, params_.mask == i + 1);
        cv::threshold(imVD, imVD, 2, 0, cv::THRESH_TOZERO);  // filter small values
        imVD(cv::Rect(0, 0, 10, imVD.rows)) = cv::Scalar(0);  // filter small disparities
        imVD(cv::Rect(100, 0, imVD.cols - 100, imVD.rows)) = cv::Scalar(0);  // filter small disparities
        //cv::imshow("imVD", 8 * imVD);
        //cv::waitKey(-1);

        // compute main ground curve
        cv::Mat mgc;
        computeMGC_DynProg(imVD, mgc);

        // coordinate transform
        cv::Mat uvd1(4, mgc.cols, CV_64F);
        uvd1.row(0) = cv::Scalar(0);
        for (int j = 0; j < mgc.cols; ++j)
	{
		uvd1.at<double>(1, j) = mgc.at<uint>(1, j);
		uvd1.at<double>(2, j) = mgc.at<uint>(0, j);
	}
        //mgc.row(1).copyTo(uvd1.row(1));
        //mgc.row(0).copyTo(uvd1.row(2));
        uvd1.row(3) = cv::Scalar(1);
        //std::cout << uvd1 << std::endl;

        cv::Mat X = params_.Q * uvd1;
        X.row(0) = -cv::Scalar(params_.getYc(i));
        X.row(1) /= X.row(3);
        X.row(2) /= X.row(3);
        X.row(3) /= X.row(3);
        X = params_.bTi * X;

        // interpolation
        cv::Mat dem_x(params_.Nx, 1, CV_64F);
        cv::Mat dem_z = cv::Mat::zeros(params_.Nx, 1, CV_64F);
        for (int j = 0; j < params_.Nx; ++j)
        {
            dem_x.at<double>(j) = params_.getXc(j);
        }
        interp1<double>(X.row(0), X.row(2), dem_x, dem_z);

        // output ply
        bool ply = false;
        if (ply)
        {
            //for (int j = 0; j < dem_x.rows; ++j) printf("%f %f %f\n", dem_x.at<double>(j), params_.getYc(i), dem_z.at<double>(j));
            for (int j = 0; j < X.cols; ++j) 
                printf("%f %f %f\n", X.at<double>(0, j), X.at<double>(1, j), X.at<double>(2, j));
        }

        // make dem image
        double min_meas_x = X.at<double>(0, 0);
        double max_meas_x = X.at<double>(0, X.cols - 1);
        for (int j = 0; j < params_.Nx; ++j)
        {
            cv::Point x1(params_.getU(params_.getXl(j)), params_.getV(params_.getYl(i)));
            cv::Point x2(params_.getU(params_.getXh(j)), params_.getV(params_.getYh(i)));
            double x_ij = params_.getXc(j);
            double z_ij = dem_z.at<double>(j);
            bool x_ok = (min_meas_x < x_ij && x_ij < max_meas_x);
            bool z_ok = (params_.min_z < z_ij && z_ij < params_.max_z);
            if (x_ok && z_ok)
                cv::rectangle(dem, x1, x2, z_ij, -1);
        }
    }
}


/*!
 *  Converts a disparity image to a DEM (digital elevation map) with least squares method
 *  @param imD        Input disparity image (single-channel, float).
 *  @param dem        Output DEM image (single-channel, float).
 */
void lsp(const cv::Mat &imD, cv::Mat &dem) {
    std::cout << "not implemented" << std::endl;
}


#ifdef UNITTEST


/*!
 *  Test function
 */
int main(int argc, char **argv) {

  // not implemented
  // use dem.py instead

  return 0;
}

#else

#include <boost/python.hpp>
#include "np2cv.h"

/*!
 *  Wrapper function for Python-C++ bridging
 */

void setup_wrapper(PyObject *dem, PyObject *rover) {
    NDArrayConverter cvt;
    params_.K = cvt.toMat(PyObject_GetAttrString(rover, "KL"));
    params_.Q = cvt.toMat(PyObject_GetAttrString(rover, "Q"));
    params_.bTi = cvt.toMat(PyObject_GetAttrString(rover, "bTi"));
    params_.iTb = cvt.toMat(PyObject_GetAttrString(rover, "iTb"));
    params_.resolution = cvt.toMat(PyObject_GetAttrString(dem, "resolution"));
    params_.ranges = cvt.toMat(PyObject_GetAttrString(dem, "ranges"));
    params_.grid_size = boost::python::extract<double>(PyObject_GetAttrString(dem, "grid_size"));
    params_.dpm = boost::python::extract<double>(PyObject_GetAttrString(dem, "dpm"));
    params_.max_disparity = boost::python::extract<int>(PyObject_GetAttrString(dem, "max_disparity"));
    params_.setup();
}

PyObject* lvd_wrapper(PyObject *imD) {
    NDArrayConverter cvt;
    cv::Mat _imD = cvt.toMat(imD);
    cv::Mat _dem;
    lvd(_imD, _dem);
    return cvt.toNDArray(_dem);
}

PyObject* lsp_wrapper(PyObject *imD) {
    NDArrayConverter cvt;
    cv::Mat _imD = cvt.toMat(imD);
    cv::Mat _dem;
    lsp(_imD, _dem);
    return cvt.toNDArray(_dem);
}

BOOST_PYTHON_MODULE(dem_impl)
{
    Py_Initialize();
    import_array();
    boost::python::def("setup", setup_wrapper);
    boost::python::def("lvd", lvd_wrapper);
    boost::python::def("lsp", lsp_wrapper);
}

#endif

