#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "drake/traj_opt/inverse_dynamics_partials.h"



struct derivative_interpolator {
    std::string keyPointMethod;
    int minN;
    int maxN;
    double jerkThreshold;
    double acellThreshold;
    double velChangeSensitivity;
    double iterativeErrorThreshold;
};

//namespace drake {
//namespace traj_opt {

template <typename T>
class DerivativeInterpolator {

public:
    DerivativeInterpolator(){

    }

    std::vector<int> ComputeKeypoints(derivative_interpolator interpolator, int horizon) const;

//    void SavePartials(std::string filename, InverseDynamicsPartials<T> *id_partials) const;
//    void GetApproximateDerivsOverTrajectory(derivative_interpolator interpolator);

private:

//    void ComputeDerivsAtSpecifiedKeypoints(std::vector<int> key_points);
//    void InterpolateDerivs(std::vector<int> key_points);

    std::vector<int> ComputeKeypoints_SetInterval(derivative_interpolator interpolator, int horizon);

    std::vector<int> ComputeKeypoints_AdaptiveJerk(derivative_interpolator interpolator, int horizon);

    std::vector<int> ComputeKeypoints_MagVelChange(derivative_interpolator interpolator, int horizon);

    std::vector<int> ComputeKeypoints_IterativeError(derivative_interpolator interpolator, int horizon);

};
//
//} // namespace traj_opt
//} // namespace drake

