/*!
 *  @file    rockdetect.cpp
 *  @author  kan mayoshi <k_mayoshi@ac.jaxa.jp>
 *  @date    2015-06-25
 *  @brief   Detect rocks from grass
 */

#include <iostream>

#include <boost/python.hpp>

#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp> 

#include "np2cv.h"

using namespace cv;


/*!
 *  Sample function (Detect rocks from grass.)
 *  @param im         Input image (3channel)
 *  @param hsv        Hsv param of input image 
 *  @param imO        Output image (1channel)
 */
void from_grass_old(const cv::Mat &im, cv::Mat &imO) {
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

void from_grass(const cv::Mat &im, cv::Mat &imO) {
	if (!imO.data) imO.create(im.rows, im.cols, CV_8UC1);

	int x, y;                    
	int width = im.cols;
	int height = im.rows;

        cv::cvtColor(im, imO, CV_BGR2GRAY);
        cv::threshold(imO, imO, 220, 255, THRESH_BINARY);

}  

void from_grass_b(const cv::Mat &im, cv::Mat &imO) {
    Mat channels[3];
    split(im, channels);
    cv::threshold(channels[0], imO, 160, 255, THRESH_BINARY);
}

void from_grass_hpf(const cv::Mat &im, cv::Mat &imO) {
    Mat gray;
    cv::cvtColor(im, gray, CV_BGR2GRAY);
    cv::threshold(gray, imO, 200, 255, THRESH_BINARY);
}

void from_grass_lpf(const cv::Mat &im, cv::Mat &imO) {
    Mat gray;
    cv::cvtColor(im, gray, CV_BGR2GRAY);
    cv::threshold(gray, imO, 70, 255, THRESH_BINARY_INV);
}


void from_grass_v(const cv::Mat &im, cv::Mat &imO) {
	if (!imO.data) imO.create(im.rows, im.cols, CV_8UC1);

	int x, y;                    
	int width = im.cols;
	int height = im.rows;

	cv::Mat hsv = cv::Mat(cv::Size(width, height), CV_8UC3);
	cv::cvtColor(im, hsv, CV_BGR2HSV);       
        cv::Mat channels[3];
	cv::split(hsv, channels);                    
        cv::Mat v = channels[2];
        cv::threshold(v, v, 230, 255, THRESH_TOZERO_INV);
        cv::threshold(v, v, 50, 255, THRESH_BINARY_INV);
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


/*!
 *  Detect rocks from grass by RGB.
 *  @param im         Input image (3channel)
 *  @param imO        Output image (1channel)
 */  
void from_grass_rgb(const cv::Mat &im, cv::Mat &imO) {       

	int x, y;                    
	int width = im.cols;
	int height = im.rows;
	cv::Mat gb = cv::Mat(cv::Size(width, height), CV_32FC1);  
	cv::Mat gr = cv::Mat(cv::Size(width, height), CV_32FC1);
	cv::Mat rb = cv::Mat(cv::Size(width, height), CV_32FC1);
	cv::Mat aa = cv::Mat(cv::Size(width, height), CV_32FC1);

	cv::Mat hsv = cv::Mat(cv::Size(width, height), CV_8UC3);
	cv::cvtColor(im, hsv, CV_BGR2HSV);       
	// imshow("hsv",hsv);

	Mat channels[3];
	split(im, channels);                    
	// cv::cvtColor(im, imO, CV_BGR2GRAY);
	// imshow("r",channels[2]);
	// imshow("g",channels[1]);
	// imshow("b",channels[0]);

	// uchar gb = channels[1] / channels[2];
	// uchar gr = channels[1] / channels[0];
	// uchar rb = channels[2] / channels[0];
	// imshow("r",gb);
	// imshow("g",channels[1]);
	// imshow("b",channels[0]);
	for (y = 0; y<height ; y++)             
		for(x = 0; x<width; x++)
		{
			float b = im.at<Vec3b>(y, x)[0];     
			float g = im.at<Vec3b>(y, x)[1];      
			float r = im.at<Vec3b>(y, x)[2];

			gb.at<float>(y,x) = g/(b+1);
			gr.at<float>(y,x) = g/(r+1);
			rb.at<float>(y,x) = r/(b+1);
			aa.at<float>(y,x) = g/((b+r+1)/2.0);


			// if( 1.1 < g/(b+1) && g/(b+1) < 10.0 )             
			//   {
			//   imO.at<uchar>(y, x)=0;
			//   }
			// else
			//   {
			//   imO.at<uchar>(y, x)=255;
			//   }

			if( 1.1 < g/(b+1) && g/(b+1) < 10 )  
			{
				if( 40 < g && g < 160 )          
				{
					imO.at<uchar>(y, x)=0;
				}
				else
				{
					if (1.1 < g/(r+1) && g/(r+1) < 10 )
					{
						imO.at<uchar>(y, x)=0;
					}  
					else
					{
						imO.at<uchar>(y, x)=255;
					}
				}
			} 
			else
			{
				if ( 90 < g && g < 120 )
				{
					imO.at<uchar>(y, x)=0;
				}
				else
				{
					if (1.3 < g/(r+1) && g/(r+1) < 10 )
					{
						imO.at<uchar>(y, x)=0;
					}  
					else
					{
						imO.at<uchar>(y, x)=255;
					}
					// imO.at<uchar>(y, x)=255;
				}
			}

			// printf("%e\n",g/(b+1));
		}

}  


/*!
 *  Detect rocks from grass by resize.
 *  @param im         Input image (3channel)
 *  @param resized        Output image (1channel)
 */
void from_grass_resize(const cv::Mat &im, cv::Mat &resized) {

	Mat gau;
	Mat im_gray;
	Mat im_gray_small(im.rows / 16, im.rows / 16, CV_32FC1);
	Mat im_gray_big(im.size(), CV_32FC1);
	Mat im_gray_float(im.size(), CV_32FC1);
	cv::cvtColor(im, im_gray, CV_BGR2GRAY);
	im_gray.convertTo(im_gray_float, im_gray_float.type());
	cv::resize(im_gray_float, im_gray_small, im_gray_small.size());

	cv::resize(im_gray_small, im_gray_big, im_gray_big.size(), INTER_LINEAR);
	gau = im_gray_big;
	// imshow("after_gau",gau/255);

	Mat differ(im.size(), im_gray_float.type());
	differ = abs(1.0 * im_gray_float - 1.0 * gau);

	//Mat differ2(differ.size(), CV_8U);
	differ.convertTo(resized, resized.type());
	// imshow("differ_reverse_before_threshold",differ2);
	threshold(resized, resized, 10, 255, THRESH_BINARY);
	// imshow("after_thresh",differ2);

	//resized = differ2;
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

PyObject* from_grass_lpf_wrapper(PyObject *im) {
	NDArrayConverter cvt;
	cv::Mat _im = cvt.toMat(im);
	cv::Mat _imO;
	from_grass_lpf(_im, _imO);
	return cvt.toNDArray(_imO);
}

PyObject* from_grass_hpf_wrapper(PyObject *im) {
	NDArrayConverter cvt;
	cv::Mat _im = cvt.toMat(im);
	cv::Mat _imO;
	from_grass_hpf(_im, _imO);
	return cvt.toNDArray(_imO);
}

PyObject* from_grass_v_wrapper(PyObject *im) {
	NDArrayConverter cvt;
	cv::Mat _im = cvt.toMat(im);
	cv::Mat _imO;
	from_grass_v(_im, _imO);
	return cvt.toNDArray(_imO);
}

PyObject* from_grass_b_wrapper(PyObject *im) {
	NDArrayConverter cvt;
	cv::Mat _im = cvt.toMat(im);
	cv::Mat _imO;
	from_grass_b(_im, _imO);
	return cvt.toNDArray(_imO);
}

PyObject* from_grass_rgb_wrapper(PyObject *im) {
	NDArrayConverter cvt;
	cv::Mat _im = cvt.toMat(im);
	cv::Mat _imO(_im.size(), CV_8UC1);
	from_grass_rgb(_im, _imO);
	return cvt.toNDArray(_imO);
}

PyObject* from_grass_resize_wrapper(PyObject *im) {
	NDArrayConverter cvt;
	cv::Mat _im = cvt.toMat(im);
	cv::Mat _imO(_im.size(), CV_8UC1);
	from_grass_resize(_im, _imO);
	return cvt.toNDArray(_imO);
}

PyObject* from_sand_wrapper(PyObject *im) {
	NDArrayConverter cvt;
	cv::Mat _im = cvt.toMat(im);
	cv::Mat _imO(_im.size(), CV_8UC1);
	from_sand(_im, _imO);
	return cvt.toNDArray(_imO);
}


BOOST_PYTHON_MODULE(rockdetect)
{
	Py_Initialize();
	import_array();
	boost::python::def("from_grass", from_grass_wrapper);
	boost::python::def("from_grass_lpf", from_grass_lpf_wrapper);
	boost::python::def("from_grass_hpf", from_grass_hpf_wrapper);
	boost::python::def("from_grass_b", from_grass_b_wrapper);
	boost::python::def("from_grass_v", from_grass_v_wrapper);
	boost::python::def("from_grass_rgb", from_grass_rgb_wrapper);
	boost::python::def("from_grass_resize", from_grass_resize_wrapper);
	boost::python::def("from_sand", from_sand_wrapper);
}

#endif
