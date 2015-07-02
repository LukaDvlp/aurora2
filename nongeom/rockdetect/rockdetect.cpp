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
    cv::cvtColor(im, hsv, CV_BGR2HSV);       //imをRGBからHSVに変換する

    Mat channels[3];
    split(hsv, channels);                    //hsv画像をh,s,vごとにchannelに入れる  

    for (y = 0; y<height ; y++)             //1ピクセルごとの処理
        for(x = 0; x<width; x++)
        {
            int h = hsv.at<Vec3b>(y, x)[0];      //int h,s,v にそのピクセルのそれぞれの値を代入
            // int s = hsv.at<Vec3b>(y, x)[1];
            // int v = hsv.at<Vec3b>(y, x)[2];

            if(h < 170 && h > 70 )               //hの値が70<h<170だったら岩，それ以外は芝生にする
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
    cv::Mat im = cv::imread("img/rock.jpg");             //ソース画像の読み込み
    int width = im.cols;                                      //imは，cols x rows　のピクセル数の画像
    int height = im.rows;
    cv::Mat imO = cv::Mat(cv::Size(width, height), CV_8UC1);  //ソース画像と同じピクセル数の1channel画像の用意

    if (!im.data) 
    {                                  
        std::cerr << "image not loaded" << std::endl;  
        return -1;
    }

    from_grass(im, imO);             //関数への代入

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


BOOST_PYTHON_MODULE(rockdetect)
{
    Py_Initialize();
    import_array();
    boost::python::def("from_grass", from_grass_wrapper);
}

#endif
