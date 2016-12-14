#ifndef PLANAR_TRACKING_H
#define PLANAR_TRACKING_H

using namespace std;
using namespace cv;

class Tracker
{
public:
    Tracker(Ptr<Feature2D> _detector, Ptr<DescriptorMatcher> _matcher, Mat _K) :
            detector(_detector),
            matcher(_matcher),
            K(_K)
            {}
    //Tracker(){}

    void setFirstFrame(const char * first_frame_path);

    bool process(const Mat frame_left, bool slamMode);

    glm::mat4 getInitModelMatrix();

protected:
    Mat K, rvec, tvec;
    Ptr<Feature2D> detector;
    Ptr<DescriptorMatcher> matcher;
    Mat first_frame, first_desc;
    vector<KeyPoint> first_kp;
    vector<Point2f> object_bb;
};

#endif //BOB_AR_PLANAR_TRACKING_H
