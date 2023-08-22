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

//namespace drake {
//namespace traj_opt {

template <typename T>
std::vector<int> DerivativeInterpolator<T>::ComputeKeypoints(
        derivative_interpolator interpolator,
        int horizon) const {
    std::vector<int> keypoints;
    if (interpolator.keyPointMethod == "set_interval") {
        keypoints = ComputeKeypoints_SetInterval(interpolator, horizon);
    } else if (interpolator.keyPointMethod == "adaptive_jerk") {
        keypoints = ComputeKeypoints_AdaptiveJerk(interpolator, horizon);
    } else if (interpolator.keyPointMethod == "mag_vel_change") {
        keypoints = ComputeKeypoints_MagVelChange(interpolator, horizon);
    } else if (interpolator.keyPointMethod == "iterative_error") {
        keypoints = ComputeKeypoints_IterativeError(interpolator, horizon);
    } else {
        std::cout << "Invalid keypoint method" << std::endl;
    }

    return keypoints;
}

template <typename T>
std::vector<int> DerivativeInterpolator<T>::ComputeKeypoints_SetInterval(
        derivative_interpolator interpolator,
        int horizon) {
    std::vector<int> keypoints;
    int counter = 0;
    // Push index 0 and 1 by default
    keypoints.push_back(0);
    keypoints.push_back(1);

    for (int i = 1; i < horizon - 1; i++) {
        if (counter >= interpolator.minN) {
            keypoints.push_back(i);
            counter = 0;
        }
        counter++;
    }

    if (keypoints[keypoints.size() - 3] != horizon - 2) {
        keypoints.push_back(horizon - 2);
    }

    // If second to last index is not horizon - 1
    if (keypoints[keypoints.size() - 2] != horizon - 1) {
        keypoints.push_back(horizon - 1);
    }

    // If last index is not horizon
    if (keypoints.back() != horizon) {
        keypoints.push_back(horizon);
    }

    return keypoints;
}

template <typename T>
std::vector<int> DerivativeInterpolator<T>::ComputeKeypoints_AdaptiveJerk(
        derivative_interpolator interpolator,
        int horizon) {
    std::vector<int> keypoints;
    std::cout << interpolator.keyPointMethod << horizon << std::endl;

    return keypoints;
}

template <typename T>
std::vector<int> DerivativeInterpolator<T>::ComputeKeypoints_MagVelChange(
        derivative_interpolator interpolator,
        int horizon) {
    std::vector<int> keypoints;
    std::cout << interpolator.keyPointMethod << horizon << std::endl;

    return keypoints;
}

template <typename T>
std::vector<int> DerivativeInterpolator<T>::ComputeKeypoints_IterativeError(
        derivative_interpolator interpolator,
        int horizon) {
    std::vector<int> keypoints;
    std::cout << interpolator.keyPointMethod << horizon << std::endl;

    return keypoints;
}

//template <typename T>
//void DerivativeInterpolator<T>::SavePartials(
//      std::string file_prefix,
//      InverseDynamicsPartials<T> *id_partials) const {
//    std::ofstream fileOutput;
//
//    std::vector <drake::MatrixX<T>> &dtau_dqm = id_partials->dtau_dqm;
//    std::vector <drake::MatrixX<T>> &dtau_dqt = id_partials->dtau_dqt;
//    std::vector <drake::MatrixX<T>> &dtau_dqp = id_partials->dtau_dqp;
//
//    // Save dtau_dqm
//    std::string file_name = file_prefix + "_dtau_dqm.csv";
//    fileOutput.open(file_name);
//
//    int size = dtau_dqm.size();
//
//    for (int i = 0; i < size; i++) {
//        // Row
//        for (int j = 0; j < dtau_dqm[i].rows(); j++) {
//            // Column
//            for (int k = 0; k < dtau_dqm[i].cols(); k++) {
//                fileOutput << dtau_dqm[i](j, k) << ",";
//            }
//        }
//        fileOutput << std::endl;
//    }
//
//    fileOutput.close();
//
//    // Save dtau_dqt
//    file_name = file_prefix + "_dtau_dqt.csv";
//    fileOutput.open(file_name);
//
//    size = dtau_dqt.size();
//
//    for (int i = 0; i < size; i++) {
//        // Row
//        for (int j = 0; j < dtau_dqt[i].rows(); j++) {
//            // Column
//            for (int k = 0; k < dtau_dqt[i].cols(); k++) {
//                fileOutput << dtau_dqt[i](j, k) << ",";
//            }
//        }
//        fileOutput << std::endl;
//    }
//
//    fileOutput.close();
//
//    // Save dtau_dqp
//    file_name = file_prefix + "_dtau_dqp.csv";
//    fileOutput.open(file_name);
//
//    size = dtau_dqp.size();
//
//    for (int i = 0; i < size; i++) {
//        // Row
//        for (int j = 0; j < dtau_dqp[i].rows(); j++) {
//            // Column
//            for (int k = 0; k < dtau_dqp[i].cols(); k++) {
//                fileOutput << dtau_dqp[i](j, k) << ",";
//            }
//        }
//        fileOutput << std::endl;
//    }
//
//    fileOutput.close();
//}
