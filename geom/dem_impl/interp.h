/*!
  @file interp.h
  @copyright 2014 Kubota Lab. All rights resereved.
*/


#ifndef _AURORA_INTERP1_H_
#define _AURORA_INTERP1_H_

#include <cmath>
#include <limits>

#include <opencv2/core/core.hpp>

/*!
 *  Finds nearest index of certian vlaue
 */
template <typename T>
int findNearestNeighbourIndex(const T value, const std::vector<T> &x)
{
    int idx = -1;
    T dist = std::numeric_limits<T>::max();
    for (int i = 0; i < x.size(); ++i)
    {
        double new_dist = value - x.at(i);
        if (new_dist >= 0 && new_dist < dist)
        {
            dist = new_dist;
            idx = i;
        }
    }
    return idx;
}


/*!
 *  Linear interpolation for 1D data
 */
template <typename T>
void interp1(const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &new_x, std::vector<T> &new_y)
{
    new_y.reserve(new_x.size());

    // compute util values
    std::vector<double> dx, dy, slope, intercept;
    dx.reserve(x.size());
    dy.reserve(x.size());
    slope.reserve(x.size());
    intercept.reserve(x.size());
    for (int i = 0; i < x.size() - 1; ++i)
    {
        dx.push_back(x.at(i + 1) - x.at(i));
        dy.push_back(y.at(i + 1) - y.at(i));
        slope.push_back(dy.at(i) / dx.at(i));
        intercept.push_back(y.at(i) - x.at(i) * slope.at(i));
    }
    dx.push_back(dx.back());
    dy.push_back(dy.back());
    slope.push_back(slope.back());
    intercept.push_back(intercept.back());

    // interpolate
    for (int i = 0; i < new_x.size(); ++i) 
    {
        int idx = findNearestNeighbourIndex(new_x.at(i), x);
        new_y.push_back(slope.at(idx) * new_x.at(i) + intercept.at(idx));
    }
}


/*!
 *  Linear interpolation for 1D data (cv::Mat)
 */
template <typename T>
void interp1(const cv::Mat &x, const cv::Mat &y, const cv::Mat &new_x, cv::Mat &new_y)
{
    new_y.create(new_x.size(), new_x.type());

    int xsize = std::max(x.cols, x.rows);
    int new_xsize = std::max(new_x.cols, new_x.rows);

    // compute util values (dx, dy, slope, intercept)
    cv::Mat A(4, xsize, CV_64F);
    for (int i = 0; i < xsize - 1; ++i)
    {
        A.at<double>(0, i) = x.at<T>(i + 1) - x.at<T>(i);
        A.at<double>(1, i) = y.at<T>(i + 1) - y.at<T>(i);
        A.at<double>(2, i) = A.at<double>(1, i) / A.at<double>(0, i);
        A.at<double>(3, i) = y.at<T>(i) - x.at<T>(i) * A.at<double>(2, i);
    }
    //A.col(A.cols - 2).copyTo(A.col(A.cols - 1));
    for (int i = 0; i < xsize - 1; ++i)
        A.at<double>(i, A.cols - 1) = A.at<double>(i, A.cols - 2);


    // interpolate
    std::vector<T> xvec(xsize);
    for (int i = 0; i < xsize; ++i) xvec.at(i) = x.at<double>(i);
    for (int i = 0; i < new_xsize; ++i) 
    {
        //std::cout << xvec[i] << std::endl;
        int idx = findNearestNeighbourIndex(new_x.at<T>(i), xvec);
        new_y.at<T>(i) = A.at<double>(2, idx) * new_x.at<T>(i) + A.at<double>(3, idx);
    }
}


#endif  // _AURORA_INTERP1_H_

