#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<chrono>
#include <opencv2/imgcodecs/legacy/constants_c.h>

using namespace std;
using namespace cv;

int main(int argc ,char**argv){
    if(argc!=3){
        cout<<"usage:feature_extraction img1 img2"<<endl;
        return 1;
    }
    //读取图像
    Mat img_1 = imread(argv[1],CV_LOAD_IMAGE_COLOR);//返回彩色图
    Mat img_2 = imread(argv[2],CV_LOAD_IMAGE_COLOR);//返回彩色图
    assert(img_1.data != nullptr&&img_2.data!=nullptr);//assert()为断言函数，如果它的条件返回错误，则终止程序执行

    //初始化
    std::vector<KeyPoint> keypoints_1,keypoints_2;//特征点
    Mat descriptors_1,descriptors_2;//描述子

    Ptr<FeatureDetector> detector = ORB::create();//特征点
    Ptr<DescriptorExtractor> descriptor = ORB::create();//描述子
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");//特征匹配

    //第一部：检测 Oriented FAST 角点位置
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    detector->detect(img_1,keypoints_1);//检测照片1中的角点
    detector->detect(img_2,keypoints_2);//检测照片2中的角点

    //第二步：根据角点位置计算BRIEF描述子
    descriptor->compute(img_1,keypoints_1,descriptors_1);
    descriptor->compute(img_2,keypoints_2,descriptors_2);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout<<"extract ORB cost:"<<time_used.count()<<"seconds."<<endl;
    Mat outimg1;

    drawKeypoints(img_1,keypoints_1,outimg1,Scalar::all(1),DrawMatchesFlags::DEFAULT);
    imshow("ORB feature",outimg1);

    //第三步：对两副图像中的BRIEF描述子进行匹配，使用汉明距离
    vector<DMatch> matches;
    t1 = chrono::steady_clock::now();
    matcher->match(descriptors_1,descriptors_2,matches);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout<<"match ORB cost"<<time_used.count()<<" seconds."<<endl;

    //第四步：匹配点对筛选
    //计算最小距离和最大距离
    auto min_max = minmax_element(matches.begin(),matches.end(),
    [](const DMatch &m1,const DMatch &m2){return m1.distance<m2.distance;});//自定义寻找最大值和最小值的函数

    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("--Max dist:%f\n",max_dist);
    printf("--Min dist:%f\n",min_dist);

    //工程经验：当描述子之间的距离大于两倍的最小距离时，即认为匹配有误，但有时距离会非常的小，所以要设置一个经验值30为下限。
    std::vector<DMatch> good_matches;
    cout<<descriptors_1.cols<<endl;
    cout<<descriptors_1.rows<<endl;
    for(int i = 0;i<descriptors_1.rows;i++){
        if(matches[i].distance<=max(2*min_dist,30.3)){
            good_matches.push_back(matches[i]);
        }
    }

    //第五步绘制图像
    Mat img_match;//未进行匹配点对筛选
    Mat img_goodmatch;//已进行筛选

    drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_match);
    drawMatches(img_1,keypoints_1,img_2,keypoints_2,good_matches,img_goodmatch);
    imshow("all matches",img_match);
    imshow("good matches",img_goodmatch);
    waitKey(0);

    return 0;
}