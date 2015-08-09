#include <iostream>
#include <string>
#include <vector>

#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include <boost/python.hpp>
#include "np2cv.h"


using namespace std;
using namespace cv;


#if 0
void detect_watanabe(const cv::Mat &src, cv::Point &itokawa_uv);
int main(int argc, char *argv[])
{
	Point itokawa_uv = (0, 0);

	char name[200];
	for (int t = 0; t <= 98; t++)
	{
		//sprintf(name, "new_itokawa_picture/frame00%d.jpg", t);
		//sprintf(name, "new_itokawa_picture/itokawa0%d.jpg", t);
		sprintf(name, "itokawa/L000%d.jpg", t);
		//sprintf(name, "itokawa/R000%d.jpg", t);
		//sprintf(name, "itokawa/IMG_000%d.jpg", t + 1);
		//sprintf(name, "new_itokawa_picture/L0000%d.jpg", t);

		Mat src = imread(name);
		//Mat src = imread("itokawa/IMG_0001.jpg");

		resize(src, src, Size(640, 480), 0, 0, CV_INTER_AREA);

		//Mat src = imread("new_itokawa_picture/frame0017.jpg");
		//Mat src = imread("itokawa/L00012.jpg");

		if (src.empty())
		{
			cout << "image not found" << endl;
			return -1;
		}

		detect_watanabe(src,itokawa_uv);

		//Mat gray, gray_edge;
		//cvtColor(src, gray, CV_BGR2GRAY);

		//Mat gray4 = gray.clone();

		////detect_watanabe(src, itokawa_uv);


		//Mat dst = src.clone();
		//Mat final = src.clone();

		//Mat rgbchannel[3];
		//split(src, rgbchannel); // B G R
		//uchar r, g, b;
		//Mat rb = Mat(Size(src.cols, src.rows), CV_32FC1);
		//Mat gb = Mat(Size(src.cols, src.rows), CV_32FC1);
		//Mat gr = Mat(Size(src.cols, src.rows), CV_32FC1);
		//Mat br = Mat(Size(src.cols, src.rows), CV_32FC1);

		//Mat bi = Mat::zeros(Size(640, 480), CV_8UC1);
		//for (int y = 0; y < src.rows; y++)
		//{
		//	for (int x = 0; x < src.cols; x++)
		//	{

		//		float b = src.at<Vec3b>(y, x)[0];
		//		float g = src.at<Vec3b>(y, x)[1];
		//		float r = src.at<Vec3b>(y, x)[2];

		//		rb.at<float>(y, x) = r / (b + 1);
		//		gb.at<float>(y, x) = g / (b + 1);
		//		gr.at<float>(y, x) = g / (r + 1);
		//		br.at<float>(y, x) = b / (r + 1);

		//		if ((r / (b + 1) > 1.1 && r / (b + 1) < 10) || (1.1 < b / (r + 1) && b / (r + 1) < 10))
		//		{
		//			gray4.at<uchar>(y, x) = 0;
		//		}
		//		else
		//		{
		//			gray4.at<uchar>(y, x) = 255;
		//		}
		//	}
		//}
		//Mat gray4_2;
		//erode(gray4, gray4_2, Mat(), Point(-1, -1), 3);
		//dilate(gray4_2, gray4_2, Mat(), Point(-1, -1), 3);

		//vector<Point> pt;
		//vector<Point> point;

		//Mat hsv, channel[3];
		//cvtColor(src, hsv, CV_BGR2HSV);
		//split(hsv, channel);

		//int k = 0, kk = 0, kkk = 0;
		//int skin_hue = 20, skin_sat = 40, skin_val = 70;
		//int gray_hue_min = 15, gray_hue_max = 35;
		//int rect_width = 80, rect_height = 60;

		//for (int y = 100; y < src.rows; y++)
		//{
		//	for (int x = 0; x < src.cols; x++)
		//	{
		//		int a = hsv.step*y + (x * 3);

		//		// 地面除外
		//		if (hsv.data[a] >= 30
		//			&& hsv.data[a] <= 50
		//			&& hsv.data[a + 1] >= 50)
		//		{
		//			dst.data[a] = 255;
		//		}
		//	}
		//}

		//// 地面除外した画像で探索
		//Mat hsv_dst;
		//cvtColor(dst, hsv_dst, CV_BGR2HSV);

		//int skin_row[640], skin_col[640];
		//int skin_row_count = 0, skin_col_count = 0;

		///// 人検出 列で
		//for (int u = 0; u < src.cols; u++)
		//{
		//	for (int v = 100; v < 350; v++)
		//	{
		//		int a = hsv_dst.step*v + (u * 3);

		//		if ((hsv_dst.data[a] >= 165
		//			|| hsv_dst.data[a] <= skin_hue)
		//			&& hsv_dst.data[a + 1] >= skin_sat
		//			&& hsv_dst.data[a + 2] >= skin_val)
		//		{
		//			skin_row_count++;
		//		}
		//	}
		//	skin_row[u] = skin_row_count;
		//	if (skin_row[u] > 20)
		//	{
		//		rectangle(src, Rect(u, 0, 1, 640), Scalar(0, 255, 0), 2);
		//		pt.push_back({ 0, 0 });
		//		pt[k].x = u; //その座標渡す
		//		pt[k].y = 0;

		//		if (pt[k].x<40 || pt[k].x>src.cols - 40) break;

		//		k++;
		//	}
		//	skin_row_count = 0;
		//}

		//// 1列だけの場合人でない可能性が高いから排除
		//if (k == 1)
		//{
		//	pt[0].x = 0;
		//}

		//int skin_col_count2 = 0;

		//for (int abc = 0; abc < k; abc++)
		//{
		//	if (pt[0].x == 0) break;
		//	/// 人の範囲切り取る
		//	rectangle(src, Rect(pt[abc].x - 40, 50, 80, 300), Scalar(0, 0, 255), 2);
		//	Mat rect1 = Mat(gray4, Rect(pt[abc].x - 40, 50, 80, 300));

		//	int gray_count = 0;
		//	int gray_row[640];
		//	int area = 0;

		//	/// 肌色 行で
		//	for (int v = 100; v < 300; v++)
		//	{
		//		for (int u = pt[abc].x - 40; u < pt[abc].x + 40; u++)
		//		{
		//			int aaa = hsv_dst.step*v + (u * 3);

		//			if ((hsv_dst.data[aaa] >= 165
		//				|| hsv_dst.data[aaa] <= skin_hue)
		//				&& hsv_dst.data[aaa + 1] >= skin_sat
		//				&& hsv_dst.data[aaa + 2] >= skin_val)
		//			{
		//				skin_col_count2++;
		//			}
		//		}
		//		skin_col[v] = skin_col_count2;

		//		area = countNonZero(rect1);

		//		if (skin_col[v] > 20) // 20
		//		{
		//			rectangle(src, Rect(pt[abc].x - 10, v - 5, 20, 10), Scalar(255, 0, 0), 2);
		//			Mat rect2 = Mat(gray4_2, Rect(pt[abc].x - 10, v - 5, 20, 10)); ///Rect(pt[abc].x - 40, v - 10, 80, 20));

		//			area = countNonZero(rect2);

		//			if (area > 100) rectangle(final, Rect(pt[abc].x - 10, v, 20, 10), Scalar(0, 255, 0), 2);

		//		}
		//		skin_col_count2 = 0;
		//		gray_count = 0;
		//		area = 0;
		//	}
		//}

		imshow("src", src);
		/*imshow("dst", dst);
		imshow("final result", final);*/

		waitKey(0);

	}
	return 0;
}

#endif 
void detect_watanabe(const cv::Mat &src, cv::Point &itokawa_uv)
{
	Mat gray, gray_edge;
	cvtColor(src, gray, CV_BGR2GRAY);

	Mat gray4 = gray.clone();

	//detect_watanabe(src, itokawa_uv);


	Mat dst = src.clone();
	Mat final = src.clone();

	Mat rgbchannel[3];
	split(src, rgbchannel); // B G R
	uchar r, g, b;
	Mat rb = Mat(Size(src.cols, src.rows), CV_32FC1);
	Mat gb = Mat(Size(src.cols, src.rows), CV_32FC1);
	Mat gr = Mat(Size(src.cols, src.rows), CV_32FC1);
	Mat br = Mat(Size(src.cols, src.rows), CV_32FC1);

	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{

			float b = src.at<Vec3b>(y, x)[0];
			float g = src.at<Vec3b>(y, x)[1];
			float r = src.at<Vec3b>(y, x)[2];

			rb.at<float>(y, x) = r / (b + 1);
			gb.at<float>(y, x) = g / (b + 1);
			gr.at<float>(y, x) = g / (r + 1);
			br.at<float>(y, x) = b / (r + 1);

			if ((r / (b + 1) > 1.1 && r / (b + 1) < 10) || (1.1 < b / (r + 1) && b / (r + 1) < 10))
			{
				gray4.at<uchar>(y, x) = 0;
			}
			else
			{
				gray4.at<uchar>(y, x) = 255;
			}
		}
	}
	Mat gray4_2;
	erode(gray4, gray4_2, Mat(), Point(-1, -1), 3);
	dilate(gray4_2, gray4_2, Mat(), Point(-1, -1), 3);

	vector<Point> pt;
	vector<Point> point;

	Mat hsv, channel[3];
	cvtColor(src, hsv, CV_BGR2HSV);
	split(hsv, channel);

	int k = 0, kk = 0, kkk = 0;
	int skin_hue = 20, skin_sat = 40, skin_val = 70;
	int gray_hue_min = 15, gray_hue_max = 35;
	int rect_width = 80, rect_height = 60;

	for (int y = 100; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			int a = hsv.step*y + (x * 3);

			// 地面除外
			if (hsv.data[a] >= 30
				&& hsv.data[a] <= 50
				&& hsv.data[a + 1] >= 50)
			{
				dst.data[a] = 255;
			}
		}
	}

	// 地面除外した画像で探索
	Mat hsv_dst;
	cvtColor(dst, hsv_dst, CV_BGR2HSV);

	int skin_row[640], skin_col[640];
	int skin_row_count = 0, skin_col_count = 0;

	/// 人検出 列で
	for (int u = 0; u < src.cols; u++)
	{
		for (int v = 100; v < 350; v++)
		{
			int a = hsv_dst.step*v + (u * 3);

			//if ((hsv_dst.data[a] >= 165
			if ((hsv_dst.data[a] >= 150
				|| hsv_dst.data[a] <= skin_hue)
				&& hsv_dst.data[a + 1] >= skin_sat
				&& hsv_dst.data[a + 2] >= skin_val)
			{
				skin_row_count++;
			}
		}
		skin_row[u] = skin_row_count;
		if (skin_row[u] > 20)
		{
			//rectangle(src, Rect(u, 0, 1, 640), Scalar(0, 255, 0), 2);
			pt.push_back({ 0, 0 });
			pt[k].x = u; //その座標渡す
			pt[k].y = 0;

			if (pt[k].x<40 || pt[k].x>src.cols - 40) break;

			k++;
		}
		skin_row_count = 0;
	}

	// 1列だけの場合人でない可能性が高いから排除
	if (k == 1)
	{
		pt[0].x = 0;
	}

	int skin_col_count2 = 0;

	for (int abc = 0; abc < k; abc++)
	{
		if (pt[0].x == 0) break;
		/// 人の範囲切り取る
		//rectangle(src, Rect(pt[abc].x - 40, 50, 80, 300), Scalar(0, 0, 255), 2);

		int gray_count = 0;
		int gray_row[640];
		int area = 0;

		/// 肌色 行で
		for (int v = 100; v < 300; v++)
		{
			for (int u = pt[abc].x - 40; u < pt[abc].x + 40; u++)
			{
				int aaa = hsv_dst.step*v + (u * 3);

				if ((hsv_dst.data[aaa] >= 165
					|| hsv_dst.data[aaa] <= skin_hue)
					&& hsv_dst.data[aaa + 1] >= skin_sat
					&& hsv_dst.data[aaa + 2] >= skin_val)
				{
					skin_col_count2++;
				}
			}
			skin_col[v] = skin_col_count2;

			if (skin_col[v] > 20) // 20
			{
				//rectangle(src, Rect(pt[abc].x - 10, v - 5, 20, 10), Scalar(255, 0, 0), 2);
				Mat rect2 = Mat(gray4_2, Rect(pt[abc].x - 10, v - 5, 20, 10)); ///Rect(pt[abc].x - 40, v - 10, 80, 20));

				area = countNonZero(rect2);

				if (area > 75)
				{
					point.push_back({ 0, 0 });
					point[kk].x = pt[abc].x; //その座標渡す
					point[kk].y = v;

					//if (point[kk].x<40 || pt[k].x>src.cols - 40) break;

					kk++;
					//rectangle(final, Rect(pt[abc].x - 10, v, 20, 10), Scalar(0, 255, 0), 2);
				}

			}
			skin_col_count2 = 0;
			area = 0;
		}
	}
	if (kk == 0)
	{
		itokawa_uv.x = -1;
		itokawa_uv.y = -1;
	}
	else 
	{
		itokawa_uv.x = point[0].x;
		itokawa_uv.y = point[0].y;
	}

	return;
}


/*!
 *  Wrapper function for Python-C++ bridging
 */

PyObject* detect_watanabe_wrapper(PyObject *im) {
	NDArrayConverter cvt;
	cv::Mat _im = cvt.toMat(im);
	cv::Point _itokawa;
	detect_watanabe(_im, _itokawa);
        cv::Mat res(2, 1, CV_32F);
        res.at<float>(0, 0) = _itokawa.x;
        res.at<float>(1, 0) = _itokawa.y;
	return cvt.toNDArray(res);
}

BOOST_PYTHON_MODULE(itokawa)
{
	Py_Initialize();
	import_array();
	boost::python::def("detect_watanabe", detect_watanabe_wrapper);
}
