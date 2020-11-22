#include <cloud_visualiser/cloud_merger.hpp>
#include <pcl/registration/warp_point_rigid_3d.h>
#include <ros/ros.h>

namespace cloud_viz {
  CloudMerger::CloudMerger()
  {
    result_ = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());

    planeExtractor_.setModelType(pcl::SACMODEL_PLANE);
    planeExtractor_.setMethodType(pcl::SAC_RANSAC);
    planeExtractor_.setDistanceThreshold(0.1);

    normalEstimator_.setRadiusSearch(params_.normalR);

    kptDetector_.setBorderRadius(params_.kptBorderR);
    kptDetector_.setNonMaxRadius(params_.kptNonMaxR);
    kptDetector_.setSalientRadius(params_.kptSalientR);
    kptDetector_.setRadiusSearch(params_.kptSearchR);

    descriptorEstimator_.setLRFRadius(params_.descLFR);
    descriptorEstimator_.setRadiusSearch(params_.descSearchR);

    correspondenceRejector_.setRefineModel(false);
    correspondenceRejector_.setInlierThreshold(params_.rejInlierTh); // 0.25
    correspondenceRejector_.setSaveInliers(true);

    icp_.setTransformationEpsilon(params_.icpTFEps);
    icp_.setMaximumIterations(params_.icpMaxIterH);
    icp_.setRANSACOutlierRejectionThreshold(params_.icpRANSACTh);
    icp_.setMaxCorrespondenceDistance(params_.icpMaxCorrDistH);
    icp_.setEuclideanFitnessEpsilon(params_.icpEuclidFitEps);
  }

  void CloudMerger::addData(const std::vector<pcl::PointCloud<PointT>::Ptr>& clouds)
  {
    if (clouds.size() != sensorNodes_.size()) {
      ROS_ERROR("Wrong number of input clouds.");
      return;
    }

    analiseClouds_(clouds);

    // Compute the TFs between clouds. Consider clouds[0] as the reference cloud.
    for (size_t sourceID = 1; sourceID < sensorNodes_.size(); ++sourceID) {
      Eigen::Matrix4f estimatedPose;

      sensorNodes_[sourceID].featureMatching =
        sensorNodes_[sourceID].numPoints == 0 || sensorNodes_[sourceID].error > base_error;

      // If the error of the current pose is too high, perform feature matching.
      if (sensorNodes_[sourceID].featureMatching) {
        ROS_INFO("S%lu: feature matching.", sourceID);
        if (currentIteration_ == 0) {
          // Find the best matching cloud for the source cloud.
          for (size_t targetID = 0; targetID < sensorNodes_.size(); ++targetID) {
            if (sourceID == targetID
                || (sensorNodes_[targetID].parent != nullptr && sensorNodes_[targetID].parent->id != sourceID))
              continue;

            pcl::CorrespondencesPtr currentCorrespondences(new pcl::Correspondences());
            correspondenceEstimator_.setInputSource(sensorNodes_[sourceID].descriptors);
            correspondenceEstimator_.setInputTarget(sensorNodes_[targetID].descriptors);
            correspondenceEstimator_.determineCorrespondences(*currentCorrespondences);

            if (sensorNodes_[sourceID].correspondences->size() < currentCorrespondences->size()) {
              sensorNodes_[sourceID].correspondences = currentCorrespondences;
              sensorNodes_[sourceID].parent = &(sensorNodes_[targetID]);
            }
          }

          ROS_INFO("S%lu: parent = S%lu.", sourceID, sensorNodes_[sourceID].parent->id);
        }
        else {
          correspondenceEstimator_.setInputSource(sensorNodes_[sourceID].descriptors);
          correspondenceEstimator_.setInputTarget(sensorNodes_[sourceID].parent->descriptors);
          correspondenceEstimator_.determineCorrespondences(*(sensorNodes_[sourceID].correspondences));
        }

        // Reject outliers.
        correspondenceRejector_.setInputSource(sensorNodes_[sourceID].keypoints);
        correspondenceRejector_.setInputTarget(sensorNodes_[sourceID].parent->keypoints);
        correspondenceRejector_.setInputCorrespondences(sensorNodes_[sourceID].correspondences);
        correspondenceRejector_.getCorrespondences(*(sensorNodes_[sourceID].refinedCorrespondences));

        std::vector<int> inliers;
        correspondenceRejector_.getInliersIndices(inliers);
        ROS_INFO("S%lu: %lu inliers remaining.", sourceID, inliers.size());

        // Compute the TF from source to target cloud.
        tfEstimator_.estimateRigidTransformation(*(sensorNodes_[sourceID].keypoints),
                                                 *(sensorNodes_[sourceID].parent->keypoints),
                                                 *(sensorNodes_[sourceID].refinedCorrespondences),
                                                 estimatedPose);

        icp_.setMaxCorrespondenceDistance(params_.icpMaxCorrDistH);
        icp_.setMaximumIterations(params_.icpMaxIterH);
      }
      // If the current error is low enough, just refine the current pose.
      else {
        ROS_INFO("S%lu: pose refinement.", sourceID);
        estimatedPose = sensorNodes_[sourceID].relativePose;
        icp_.setMaxCorrespondenceDistance(params_.icpMaxCorrDistL);
        icp_.setMaximumIterations(params_.icpMaxIterL);
      }

      // Improve the estimated TFs through ICP.
      poseRefinement_(
        sensorNodes_[sourceID].cloudWithNormals, sensorNodes_[sourceID].parent->cloudWithNormals, estimatedPose);

      double error = 2
                     * (icp_.getFitnessScore()
                        / (sensorNodes_[sourceID].cloudWithNormals->points.size()
                           + sensorNodes_[sourceID].parent->cloudWithNormals->points.size()));
      ROS_INFO("S%lu: pose error %e.", sourceID, error);
      // Update the pose only if the current result is better or slightly worse than the previous one.
      if (error < sensorNodes_[sourceID].error) {
        ROS_INFO("S%lu: new best error, pose updated.", sourceID);
        sensorNodes_[sourceID].error = error;
        sensorNodes_[sourceID].numPoints = sensorNodes_[sourceID].cloudWithNormals->points.size();
        sensorNodes_[sourceID].relativePose = icp_.getFinalTransformation() * estimatedPose;
      }
      // If the feature matching attempt failed, try with just a pose refinement.
      else if (sensorNodes_[sourceID].featureMatching) {
        ROS_INFO("S%lu: Feature matching unsuccessful, attempting pose refinement.", sourceID);
        estimatedPose = sensorNodes_[sourceID].relativePose;
        poseRefinement_(
          sensorNodes_[sourceID].cloudWithNormals, sensorNodes_[sourceID].parent->cloudWithNormals, estimatedPose);

        error = 2
                * (icp_.getFitnessScore()
                   / (sensorNodes_[sourceID].cloudWithNormals->points.size()
                      + sensorNodes_[sourceID].parent->cloudWithNormals->points.size()));
        ROS_INFO("S%lu: pose error %e.", sourceID, error);
        if (error < sensorNodes_[sourceID].error) {
          ROS_INFO("S%lu: new best error, pose updated.", sourceID);
          sensorNodes_[sourceID].error = error;
          sensorNodes_[sourceID].numPoints = sensorNodes_[sourceID].cloudWithNormals->points.size();
          sensorNodes_[sourceID].relativePose = icp_.getFinalTransformation() * estimatedPose;
        }
      }

      // Compute the absolute poses of all sensors.
      sensorNodes_[sourceID].absolutePose = sensorNodes_[sourceID].relativePose;
      Node* currentParent = sensorNodes_[sourceID].parent;
      while (currentParent != nullptr) {
        sensorNodes_[sourceID].absolutePose *= currentParent->relativePose;
        currentParent = currentParent->parent;
      }
    }

    // Fuse the input clouds.
    result_->clear();
    *result_ += *(sensorNodes_[0].cloud);
    //*result_ += *(clouds[0]);
    pcl::PointCloud<PointT>::Ptr tempCloud(new pcl::PointCloud<PointT>());

    for (size_t i = 1; i < sensorNodes_.size(); ++i) {
      pcl::transformPointCloud(*(sensorNodes_[i].cloud), *tempCloud, sensorNodes_[i].absolutePose);
      // pcl::transformPointCloud(*(clouds[i]), *tempCloud, sensorNodes_[i].absolutePose);
      *result_ += *tempCloud;
    }

    ++currentIteration_;
  }

  const pcl::PointCloud<CloudMerger::PointT>::Ptr& CloudMerger::result() const { return result_; }

  void CloudMerger::setNodeSize(size_t size)
  {
    sensorNodes_.clear();
    sensorNodes_.resize(size);
    for (size_t i = 0; i < size; ++i) {
      sensorNodes_[i].id = i;
      sensorNodes_[i].cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
      sensorNodes_[i].cloudWithNormals = pcl::PointCloud<PointTNormal>::Ptr(new pcl::PointCloud<PointTNormal>());
      sensorNodes_[i].normals = pcl::PointCloud<NormalPt>::Ptr(new pcl::PointCloud<NormalPt>());
      sensorNodes_[i].keypoints = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
      sensorNodes_[i].keypointsWithNormals = pcl::PointCloud<PointTNormal>::Ptr(new pcl::PointCloud<PointTNormal>());
      sensorNodes_[i].descriptors = pcl::PointCloud<Descriptor>::Ptr(new pcl::PointCloud<Descriptor>());
      sensorNodes_[i].correspondences = pcl::CorrespondencesPtr(new pcl::Correspondences());
      sensorNodes_[i].refinedCorrespondences = pcl::CorrespondencesPtr(new pcl::Correspondences());
      sensorNodes_[i].floor = pcl::ModelCoefficients::Ptr(new pcl::ModelCoefficients());
    }
  }

  void CloudMerger::setParams(const Params& params)
  {
    params_ = params;

    downsampler_.setLeafSize(params_.leafSize, params_.leafSize, params_.leafSize);

    normalEstimator_.setRadiusSearch(params_.normalR);

    kptDetector_.setBorderRadius(params_.kptBorderR);
    kptDetector_.setNonMaxRadius(params_.kptNonMaxR);
    kptDetector_.setSalientRadius(params_.kptSalientR);
    kptDetector_.setRadiusSearch(params_.kptSearchR);

    descriptorEstimator_.setLRFRadius(params_.descLFR);
    descriptorEstimator_.setRadiusSearch(params_.descSearchR);

    correspondenceRejector_.setRefineModel(false);
    correspondenceRejector_.setInlierThreshold(params_.rejInlierTh); // 0.25
    correspondenceRejector_.setSaveInliers(true);

    icp_.setTransformationEpsilon(params_.icpTFEps);
    icp_.setMaximumIterations(params_.icpMaxIterH);
    icp_.setRANSACOutlierRejectionThreshold(params_.icpRANSACTh);
    icp_.setMaxCorrespondenceDistance(params_.icpMaxCorrDistH);
    icp_.setEuclideanFitnessEpsilon(params_.icpEuclidFitEps);
  }

  void CloudMerger::analiseClouds_(const std::vector<pcl::PointCloud<PointT>::Ptr>& clouds)
  {
    // Loop through all the clouds.
    for (size_t i = 0; i < sensorNodes_.size(); ++i) {
      sensorNodes_[i].cloud->clear();
      sensorNodes_[i].cloudWithNormals->clear();
      sensorNodes_[i].normals->clear();
      sensorNodes_[i].keypoints->clear();
      sensorNodes_[i].keypointsWithNormals->clear();
      sensorNodes_[i].descriptors->clear();
      sensorNodes_[i].correspondences->clear();
      sensorNodes_[i].refinedCorrespondences->clear();

      // Compute normals.
      normalEstimator_.setInputCloud(clouds[i]);
      normalEstimator_.compute(*(sensorNodes_[i].normals));

      // Extract the floor and remove it from the cloud.
      pcl::PointIndices::Ptr planeInliers(new pcl::PointIndices());
      planeExtractor_.setInputCloud(clouds[i]);
      planeExtractor_.setInputNormals(sensorNodes_[i].normals);
      planeExtractor_.segment(*planeInliers, *(sensorNodes_[i].floor));
      sensorNodes_[i].cloud = removePoints_<PointT>(clouds[i], planeInliers);
      sensorNodes_[i].normals = removePoints_<NormalPt>(sensorNodes_[i].normals, planeInliers);

      pcl::PointCloud<PointT>::Ptr floorCloud(new pcl::PointCloud<PointT>());
      pcl::PointCloud<NormalPt>::Ptr floorNormals(new pcl::PointCloud<NormalPt>());
      downsampler_.setInputCloud(extractPoints_<PointT>(clouds[i], planeInliers));
      downsampler_.filter(*floorCloud);
      normalEstimator_.setInputCloud(floorCloud);
      normalEstimator_.compute(*floorNormals);
      *(sensorNodes_[i].cloud) += *(floorCloud);
      *(sensorNodes_[i].normals) += *(floorNormals);
      pcl::copyPointCloud(*(sensorNodes_[i].cloud), *(sensorNodes_[i].cloudWithNormals));
      pcl::copyPointCloud(*(sensorNodes_[i].normals), *(sensorNodes_[i].cloudWithNormals));

      // Compute keypoints.
      kptDetector_.setNormals(sensorNodes_[i].normals);
      kptDetector_.setInputCloud(sensorNodes_[i].cloud);
      kptDetector_.compute(*(sensorNodes_[i].keypoints));
      pcl::PointIndicesConstPtr kptIndices = kptDetector_.getKeypointsIndices();
      pcl::copyPointCloud(*(sensorNodes_[i].keypoints), *(sensorNodes_[i].keypointsWithNormals));
      pcl::copyPointCloud(*(sensorNodes_[i].normals), kptIndices->indices, *(sensorNodes_[i].keypointsWithNormals));

      // Compute descriptors.
      descriptorEstimator_.setInputNormals(sensorNodes_[i].normals);
      descriptorEstimator_.setInputCloud(sensorNodes_[i].keypoints);
      descriptorEstimator_.setSearchSurface(sensorNodes_[i].cloud);
      descriptorEstimator_.compute(*(sensorNodes_[i].descriptors));
      ROS_INFO("Computed %lu descriptors from %lu points.",
               sensorNodes_[i].descriptors->points.size(),
               sensorNodes_[i].cloud->points.size());
    }
  }

  void CloudMerger::poseRefinement_(const pcl::PointCloud<PointTNormal>::Ptr& sourceCloud,
                                    const pcl::PointCloud<PointTNormal>::Ptr& targetCloud,
                                    const Eigen::Matrix4f& estimatedPose)
  {
    pcl::PointCloud<PointTNormal>::Ptr alignedPointCloud(new pcl::PointCloud<PointTNormal>());
    pcl::PointCloud<PointTNormal>::Ptr icpOutputCloud(new pcl::PointCloud<PointTNormal>());
    pcl::transformPointCloud(*sourceCloud, *alignedPointCloud, estimatedPose);

    pcl::registration::WarpPointRigid3D<PointTNormal, PointTNormal> rigidTF;
    rigidTF.setParam(Eigen::Vector3f());

    icp_.setInputSource(alignedPointCloud);
    icp_.setInputTarget(targetCloud);
    icp_.align(*icpOutputCloud);
  }
} // namespace cloud_viz
