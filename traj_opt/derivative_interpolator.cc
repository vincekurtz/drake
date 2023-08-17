#include "derivative_interpolator.h"

//void DerivativeInterpolator::GetApproximateDerivsOverTrajectory(derivative_interpolator interpolator){
//
//    std::vector<int> keypoints = ComputeKeypoints(interpolator);
//
//
//}

//void DerivativeInterpolator::ComputeDerivsAtSpecifiedKeypoints(std::vector<int> key_points){
//
//}

//void DerivativeInterpolator::InterpolateDerivs(std::vector<int> keypoints){
//
//}

std::vector<int> DerivativeInterpolator::ComputeKeypoints(derivative_interpolator interpolator, int horizon){
    std::vector<int> keypoints;
    if (interpolator.keyPointMethod == "set_interval"){
        keypoints = ComputeKeypoints_SetInterval(interpolator, horizon);
    }
    else if (interpolator.keyPointMethod == "adaptive_jerk"){
        keypoints = ComputeKeypoints_AdaptiveJerk(interpolator, horizon);
    }
    else if (interpolator.keyPointMethod == "mag_vel_change"){
        keypoints = ComputeKeypoints_MagVelChange(interpolator, horizon);
    }
    else if (interpolator.keyPointMethod == "iterative_error"){
        keypoints = ComputeKeypoints_IterativeError(interpolator, horizon);
    }
    else{
        std::cout << "Invalid keypoint method" << std::endl;
    }

    return keypoints;
}

std::vector<int> DerivativeInterpolator::ComputeKeypoints_SetInterval(derivative_interpolator interpolator, int horizon){
    std::vector<int> keypoints;
    int counter = 0;
    keypoints.push_back(0);

    for(int i = 0; i < horizon; i++){
        if(counter >= interpolator.minN){
            keypoints.push_back(i);
            counter = 0;
        }
        counter ++;
    }

    if(keypoints.back() != horizon){
        keypoints.push_back(horizon);
    }

    return keypoints;
}

std::vector<int> DerivativeInterpolator::ComputeKeypoints_AdaptiveJerk(derivative_interpolator interpolator, int horizon){
    std::vector<int> keypoints;
    std::cout << interpolator.keyPointMethod << horizon << std::endl;

    return keypoints;
}

std::vector<int> DerivativeInterpolator::ComputeKeypoints_MagVelChange(derivative_interpolator interpolator, int horizon){
    std::vector<int> keypoints;
    std::cout << interpolator.keyPointMethod << horizon << std::endl;

    return keypoints;
}

std::vector<int> DerivativeInterpolator::ComputeKeypoints_IterativeError(derivative_interpolator interpolator, int horizon){
    std::vector<int> keypoints;
    std::cout << interpolator.keyPointMethod << horizon << std::endl;

    return keypoints;
}
