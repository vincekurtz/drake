#include "derivative_interpolator.h"

DerivativeInterpolator::DerivativeInterpolator(){

}

void DerivativeInterpolator::GetApproximateDerivsOverTrajectory(derivative_interpolator interpolator){

    std::vector<int> keypoints = ComputeKeypoints(interpolator);


}

void DerivativeInterpolator::ComputeDerivsAtSpecifiedKeypoints(std::vector<int> key_points){

}

void DerivativeInterpolator::InterpolateDerivs(std::vector<int> keypoints){

}

std::vector<int> DerivativeInterpolator::ComputeKeypoints(){
    if (keyPointMethod == "set_interval"){
        ComputeKeypoints_SetInterval();
    }
    else if (keyPointMethod == "adaptive_jerk"){
        ComputeKeypoints_AdaptiveJerk();
    }
    else if (keyPointMethod == "mag_vel_change"){
        ComputeKeypoints_MagVelChange();
    }
    else if (keyPointMethod == "iterative_error"){
        ComputeKeypoints_IterativeError();
    }
    else{
        std::cout << "Invalid keypoint method" << std::endl;
    }
}

std::vector<int> DerivativeInterpolator::ComputeKeypoints_SetInterval(){

}

std::vector<int> DerivativeInterpolator::ComputeKeypoints_AdaptiveJerk(){

}

std::vector<int> DerivativeInterpolator::ComputeKeypoints_MagVelChange(){

}

std::vector<int> DerivativeInterpolator::ComputeKeypoints_IterativeError(){

}
