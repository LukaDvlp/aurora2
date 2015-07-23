/*!
 *  @file    rockdetect.cpp
 *  @author  kan mayoshi <k_mayoshi@ac.jaxa.jp>
 *  @date    2015-06-25
 *  @brief   Detect rocks from grass
 */

#include <iostream>

#include <boost/python.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp> 

#include "np2cv.h"

using namespace cv;


/*!
 *  Sample function (Detect rocks from grass.)
 *  @param im         Input image (3channel)
 *  @param hsv        Hsv param of input image 
 *  @param imO        Output image (1channel)
 */
void from_grass(const cv::Mat &im, cv::Mat &imO) {
    if (!imO.data) imO.create(im.rows, im.cols, CV_8UC1);

    int x, y;                    
    int width = im.cols;
    int height = im.rows;

    cv::Mat hsv = cv::Mat(cv::Size(width, height), CV_8UC3);
    cv::cvtColor(im, hsv, CV_BGR2HSV);

    Mat channels[3];
    split(hsv, channels);

    for (y = 0; y<height ; y++)
        for(x = 0; x<width; x++)
        {
            uchar h = hsv.at<Vec3b>(y, x)[0];
            // int s = hsv.at<Vec3b>(y, x)[1];
            // int v = hsv.at<Vec3b>(y, x)[2];

            //if(h < 170 && h > 70 ) 
            if(h < 180 && h > 25 ) 
            {
                imO.at<uchar>(y, x)=0;
            }
            else
            {
                imO.at<uchar>(y, x)=255;
            }
        }
}  

/*!
 *  Detect rocks from sand.
 *  @param im         Input image (3channel)
 *  @param im_gray    Gray scale of input image 
 *  @param imO        Output image (1channel)
 */
void from_sand(const cv::Mat &im, cv::Mat &imO) {
  int x, y;
  int width = im.cols;
  int height = im.rows;
 
  imO.create(Size(width, height), CV_8UC1);
  cvtColor(im, imO,CV_BGR2GRAY);
  // imshow("gray",imO);
  for (y = 0; y<height ; y++)
    for(x=0; x<width; x++)
    {
      uchar v = imO.at<uchar>(y, x);
      if(v < 70)
        {
        imO.at<uchar>(y, x)=255;
        }
      else
        {
        imO.at<uchar>(y, x)=0;
      
        }
      
    }
  }    

#ifdef UNITTEST

/*!
 *  Test function
 */
int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: ./rockdetect path/to/image.jpg" << std::endl;
        return -1;
    }

    cv::Mat im = cv::imread(argv[1]);
    int width = im.cols;
    int height = im.rows;
    cv::Mat imO = cv::Mat(cv::Size(width, height), CV_8UC1);

    if (!im.data) 
    {                                  
        std::cerr << "image not loaded" << std::endl;  
        return -1;
    }

    from_grass(im, imO);

    cv::imshow("Input", im);
    cv::imshow("Output", imO);

    cv::waitKey(-1);
    return 0;
}

#else

/*!
 *  Wrapper function for Python-C++ bridging
 */

PyObject* from_grass_wrapper(PyObject *im) {
    NDArrayConverter cvt;
    cv::Mat _im = cvt.toMat(im);
    cv::Mat _imO;
    from_grass(_im, _imO);
    return cvt.toNDArray(_imO);
}

PyObject* from_sand_wrapper(PyObject *im) {
    NDArrayConverter cvt;
    cv::Mat _im = cvt.toMat(im);
    cv::Mat _imO;
    from_sand(_im, _imO);
    return cvt.toNDArray(_imO);
}


BOOST_PYTHON_MODULE(rockdetect)
{
    Py_Initialize();
    import_array();
    boost::python::def("from_grass", from_grass_wrapper);
    boost::python::def("from_sand", from_sand_wrapper);
}

#endif
