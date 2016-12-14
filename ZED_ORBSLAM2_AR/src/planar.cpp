#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <opencv2/opencv.hpp>
#include "planar_tracking.h"
#include "utils.h"


using namespace cv;
using namespace std;

glm::mat4 Tracker::getInitModelMatrix(){

    glm::mat4 initModelMatrix;
    Mat initR;
    Mat viewMatrix = cv::Mat::zeros(4, 4, CV_64FC1);
    Rodrigues(rvec, initR);

    for(unsigned int row=0; row<3; ++row)
    {
        for(unsigned int col=0; col<3; ++col)
        {
            viewMatrix.at<double>(row, col) = initR.at<double>(row, col);
        }
        viewMatrix.at<double>(row, 3) = tvec.at<double>(row, 0);
    }
    viewMatrix.at<double>(3, 3) = 1.0f;

    //viewMatrix = cvToGl * viewMatrix;

    viewMatrix.convertTo(viewMatrix, CV_32F);


    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++) {
            initModelMatrix[i][j] = viewMatrix.at<float>(j,i);
        }
    }

    return initModelMatrix;

}

void Tracker::setFirstFrame(const char * first_frame_path)
{
    first_frame = imread(first_frame_path);
    vector<KeyPoint> kp;

    object_bb.push_back(Point2f(0,0));
    object_bb.push_back(Point2f(8.4,0));
    object_bb.push_back(Point2f(8.4,4.7));
    object_bb.push_back(Point2f(0,4.7));

    vector<Point2f> bb;
    bb.push_back(Point2f(0,0));
    bb.push_back(Point2f(672,0));
    bb.push_back(Point2f(672,376));
    bb.push_back(Point2f(0,376));

    Mat H;
    H = findHomography(bb, object_bb);
    
    detector->detectAndCompute(first_frame, noArray(), kp, first_desc);

    vector<Point2f> tmp_kp_orgn, tmp_kp_homo;

    first_kp = kp;

    for (int i = 0; i <= kp.size(); i++) {
        tmp_kp_orgn.push_back(Point2f(kp[i].pt));
    }

    perspectiveTransform(tmp_kp_orgn, tmp_kp_homo, H);

    for (int i = 0; i <= kp.size(); i++) {
        first_kp[i].pt = tmp_kp_homo[i];
    }
}

bool Tracker::process(const Mat frame_left, bool slamMode)
{

    if (slamMode)
        return 1;

    vector<KeyPoint> kp;
    vector<Point3f> ObjectPoints;
    vector<Point2f> ImagePoints;
    Mat desc;
    
    detector->detectAndCompute(frame_left, noArray(), kp, desc);
    
    if(kp.size()<10)
    {
        return 0;
    }
    
    vector< vector<DMatch> > matches;
    vector<KeyPoint> matched1, matched2;
    matcher->knnMatch(first_desc, desc, matches, 2);
    for(unsigned i = 0; i < matches.size(); i++) {
        if(matches[i][0].distance < 0.8f * matches[i][1].distance) {
            matched1.push_back(first_kp[matches[i][0].queryIdx]);
            matched2.push_back(      kp[matches[i][0].trainIdx]);
        }
    }

    Mat inlier_mask, homography;

    int thd = (int)(0.08*first_desc.rows);

    if(matched1.size() >= thd) {
        homography = findHomography(Points(matched1), Points(matched2),
                                    RANSAC, 10.0f, inlier_mask);
    }

    if(matched1.size() < thd || homography.empty()) {
        return 0;
    }


    for(unsigned i = 0; i < matched1.size(); i++) {
        if(inlier_mask.at<uchar>(i)) {
            ObjectPoints.push_back(Point3f(matched1[i].pt.x, matched1[i].pt.y, 0));
            ImagePoints.push_back(Point2f(matched2[i].pt.x, matched2[i].pt.y));
        }
    }

    solvePnP(ObjectPoints, ImagePoints, K, noArray(), rvec, tvec);

    return 1;
}
