#ifndef CLOUD_MERGER_HPP
#define CLOUD_MERGER_HPP

#include <numeric>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/mls.h>

namespace cloud_viz {
  // Merge multiple point clouds together, improving the result at each iteration.
  class CloudMerger
  {
  public:
    using PointT = pcl::PointXYZRGB;
    using PointTNormal = pcl::PointXYZRGBNormal;
    using NormalPt = pcl::Normal;
    using Descriptor = pcl::SHOT1344;

    // Sampling parameters.
    inline static constexpr double leaf_size{0.01};

    // Normal estimator parameters.
    inline static constexpr double normal_radius{0.1};

    // Keypoint detector parameters.
    inline static constexpr double kpt_border_radius{0.05};
    inline static constexpr double kpt_non_max_radius{0.1};
    inline static constexpr double kpt_salient_radius{0.05};
    inline static constexpr double kpt_search_radius{0.05};

    // Descriptor estimator parameters.
    inline static constexpr float desc_lfr_radius{0.1F};
    inline static constexpr double desc_search_radius{0.05};

    // Correspondence rejector parameters.
    inline static constexpr double rej_inlier_th{0.4};

    // ICP parameters.
    inline static constexpr double icp_tf_epsilon{1e-12};
    inline static constexpr int icp_max_iter_high{25};
    inline static constexpr int icp_max_iter_low{40};
    inline static constexpr double icp_RANSAC_th{0.001};
    inline static constexpr double icp_max_corr_dist_high{0.2};
    inline static constexpr double icp_max_corr_dist_low{0.1};
    inline static constexpr double icp_euclid_fit_epsilon{2.0};

    // ICP error per point pair. Below this error, only ICP will be performed.
    static constexpr double base_error{1.0e-7};

    // Calibration parameters.
    struct Params
    {
      // Sampling.
      double leafSize;
      // Normal estimator.
      double normalR;
      // Keypoint detector.
      double kptBorderR;
      double kptNonMaxR;
      double kptSalientR;
      double kptSearchR;
      // Descriptor estimator.
      float descLFR;
      double descSearchR;
      // Correspondence rejector.
      double rejInlierTh;
      // ICP.
      double icpTFEps;
      int icpMaxIterH;
      int icpMaxIterL;
      double icpRANSACTh;
      double icpMaxCorrDistH;
      double icpMaxCorrDistL;
      double icpEuclidFitEps;
    };

    // TF node.
    struct Node
    {
      size_t id{0}; // ID of the sensor.
      Node* parent{nullptr}; // Parent in the TF tree.
      Eigen::Matrix4f relativePose{Eigen::Matrix4f::Identity()}; // Pose relative to parent.
      Eigen::Matrix4f absolutePose{Eigen::Matrix4f::Identity()}; // Pose relative to world.
      double error{std::numeric_limits<double>::infinity()}; // Error of the current pose.
      size_t numPoints{0}; // Number of cloud points used to compute the last pose.
      bool featureMatching{true}; // True if feature matching needs to be performed.

      pcl::PointCloud<PointT>::Ptr cloud;
      pcl::PointCloud<PointTNormal>::Ptr cloudWithNormals;
      pcl::PointCloud<NormalPt>::Ptr normals;
      pcl::PointCloud<PointT>::Ptr keypoints;
      pcl::PointCloud<PointTNormal>::Ptr keypointsWithNormals;
      pcl::PointCloud<Descriptor>::Ptr descriptors;
      pcl::CorrespondencesPtr correspondences;
      pcl::CorrespondencesPtr refinedCorrespondences;

      pcl::ModelCoefficients::Ptr floor;
    };

    CloudMerger();

    // Merge a new set of clouds.
    void addData(const std::vector<pcl::PointCloud<PointT>::Ptr>& clouds);

    const pcl::PointCloud<PointT>::Ptr& result() const;

    // Reset the number of nodes.
    void setNodeSize(size_t size);
    void setParams(const Params& params);

  private:
    // Perform preliminary computations on the clouds.
    void analiseClouds_(const std::vector<pcl::PointCloud<PointT>::Ptr>& clouds);

    // Remove points from a point cloud.
    template <typename T>
    typename pcl::PointCloud<T>::Ptr extractPoints_(const typename pcl::PointCloud<T>::Ptr& cloud,
                                                    const pcl::PointIndices::Ptr& indices)
    {
      pcl::ExtractIndices<T> extractor;
      typename pcl::PointCloud<T>::Ptr newCloud(new typename pcl::PointCloud<T>());
      extractor.setInputCloud(cloud);
      extractor.setIndices(indices);
      extractor.setNegative(false);
      extractor.filter(*newCloud);
      return newCloud;
    }

    // Refine a pose estimate.
    void poseRefinement_(const pcl::PointCloud<PointTNormal>::Ptr& sourceCloud,
                         const pcl::PointCloud<PointTNormal>::Ptr& targetCloud,
                         const Eigen::Matrix4f& estimatedPose);

    // Remove points from a point cloud.
    template <typename T>
    typename pcl::PointCloud<T>::Ptr removePoints_(const typename pcl::PointCloud<T>::Ptr& cloud,
                                                   const pcl::PointIndices::Ptr& indices)
    {
      pcl::ExtractIndices<T> extractor;
      typename pcl::PointCloud<T>::Ptr newCloud(new typename pcl::PointCloud<T>());
      extractor.setInputCloud(cloud);
      extractor.setIndices(indices);
      extractor.setNegative(true);
      extractor.filter(*newCloud);
      return newCloud;
    }

    std::vector<Node> sensorNodes_; // Vector containing all sensors.

    size_t currentIteration_{0}; // Index of the current iteration.
    pcl::PointCloud<PointT>::Ptr result_; // Result from the current iteration.
    bool featureMatching_{true}; // False if all nodes completed the feature matching phase.

    // PCL classes.
    pcl::VoxelGrid<pcl::PointXYZRGB> downsampler_;
    pcl::SACSegmentationFromNormals<PointT, NormalPt> planeExtractor_;
    pcl::NormalEstimation<PointT, NormalPt> normalEstimator_;
    pcl::ISSKeypoint3D<PointT, PointT, NormalPt> kptDetector_;
    pcl::SHOTColorEstimation<PointT, pcl::Normal, Descriptor> descriptorEstimator_;
    pcl::registration::CorrespondenceEstimation<Descriptor, Descriptor> correspondenceEstimator_;
    pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> correspondenceRejector_;
    pcl::registration::TransformationEstimationSVD<PointT, PointT> tfEstimator_;
    pcl::IterativeClosestPointWithNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> icp_;

    // Calibration parameters.
    Params params_{leaf_size,
                   normal_radius,
                   kpt_border_radius,
                   kpt_non_max_radius,
                   kpt_salient_radius,
                   kpt_search_radius,
                   desc_lfr_radius,
                   desc_search_radius,
                   rej_inlier_th,
                   icp_tf_epsilon,
                   icp_max_iter_high,
                   icp_max_iter_low,
                   icp_RANSAC_th,
                   icp_max_corr_dist_high,
                   icp_max_corr_dist_low,
                   icp_euclid_fit_epsilon};
  };
} // namespace cloud_viz

#endif // CLOUD_MERGER_HPP
