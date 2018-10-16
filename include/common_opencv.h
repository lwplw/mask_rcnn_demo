#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <algorithm>

// Official OpenCV
#include "opencv2/core/core.hpp"  
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/dnn/dnn.hpp>

#ifdef _DEBUG
#pragma comment(lib, "opencv\\opencv_world400d.lib")
#else
#pragma comment(lib, "opencv\\opencv_world400.lib")
#endif

