/************************************************************************
 *
 *
 *  Copyright 2015  Arun Das (University of Waterloo)
 *                      [adas@uwaterloo.ca]
 *                  James Servos (University of Waterloo)
 *                      [jdservos@uwaterloo.ca]
 *
 *
 *************************************************************************/

#include <cmath>
#include <iostream>
#include <map>
#include <ros/ros.h>
#include <vector>
// ros/pcl headers #include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/tf.h>

#include <eigen_conversions/eigen_msg.h>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/NavSatFix.h>

#include <laser_slam/csv_tools.h>
#include <laser_slam/laser_slam.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>

template <typename TYPE>
void LoadParameter(std::string param_name, TYPE &param_var, TYPE default_value,
                   ros::NodeHandle &nh) {
  if (nh.getParam(param_name, param_var)) {
    ROS_INFO_STREAM("Laser SLAM:: " << param_name << ": " << param_var);
  } else {
    ROS_WARN_STREAM("Laser SLAM:: could not load parameter "
                    << param_name << ". Using default of " << default_value
                    << ".");
    param_var = default_value;
  }
}

double get_2d_dist(geometry_msgs::Pose &pt1, geometry_msgs::Pose &pt2) {
  double dx, dy;
  dx = pt1.position.x - pt2.position.x;
  dy = pt1.position.y - pt2.position.y;
  return sqrt(dx * dx + dy * dy);
}

int main(int argc, char **argv) {
  // Initialize ROS
  ros::init(argc, argv, "laser_slam_direct_from_bag");
  ros::NodeHandle nh("~");

  //---------------------------------------BEGIN COPYPASTA--------------------//
  // params to be set for the algorithm
  int map_x_size_meters;
  int map_y_size_meters;
  double map_resolution;
  double mls_vehicle_height;
  int scanreg_num_vertex_knn;
  int scanreg_num_ref_iter;
  double scanreg_init_leaf_size;
  double scanreg_init_icp_distance_max;
  double scanreg_icp_trans_thresh;
  double scanreg_icp_rot_thresh;
  double mls_thickness_threshold;
  double mls_height_threshold;
  double mls_cluster_sigma_factor;
  double mls_cluster_dist_threshold;
  double mls_cluster_combine_dist;
  bool scanreg_recompute_all_edges;

  ROS_INFO_STREAM("Loading parameters from server...");
  // try to get all the params
  LoadParameter<int>("map_x_size_meter", map_x_size_meters, 2000, nh);
  LoadParameter<int>("map_y_size_meter", map_y_size_meters, 2000, nh);
  LoadParameter<double>("map_resolution", map_resolution, 0.5, nh);
  LoadParameter<double>("mls_vehicle_height", mls_vehicle_height, 1.7, nh);
  LoadParameter<int>("scanreg_num_vertex_knn", scanreg_num_vertex_knn, 10, nh);
  LoadParameter<int>("scanreg_num_refinement_iterations", scanreg_num_ref_iter,
                     10, nh);
  LoadParameter<double>("scanreg_init_leaf_size", scanreg_init_leaf_size, 5,
                        nh);
  LoadParameter<double>("scanreg_init_icp_distance_max",
                        scanreg_init_icp_distance_max, 30, nh);
  LoadParameter<double>("scanreg_icp_trans_thresh", scanreg_icp_trans_thresh,
                        10, nh);
  LoadParameter<double>("scanreg_icp_rot_thresh", scanreg_icp_rot_thresh, 0.5,
                        nh);
  LoadParameter<double>("mls_thickness_threshold", mls_thickness_threshold, 0.3,
                        nh);
  LoadParameter<double>("mls_height_threshold", mls_height_threshold, 0.3, nh);
  LoadParameter<double>("mls_cluster_sigma_factor", mls_cluster_sigma_factor, 2,
                        nh);
  LoadParameter<double>("mls_cluster_dist_threshold",
                        mls_cluster_dist_threshold, 0.5, nh);
  LoadParameter<double>("mls_cluster_combine_dist", mls_cluster_combine_dist,
                        0.2, nh);
  LoadParameter<bool>("scanreg_recompute_all_edges",
                      scanreg_recompute_all_edges, false, nh);
  ROS_INFO_STREAM("Finished loading parameters");

  int map_x_cells = (int)(map_x_size_meters / map_resolution);
  int map_y_cells = (int)(map_y_size_meters / map_resolution);

  LaserSlam laser_slam(map_x_cells, map_y_cells, map_resolution,
                       mls_vehicle_height, scanreg_init_leaf_size,
                       scanreg_init_icp_distance_max, scanreg_icp_trans_thresh,
                       scanreg_icp_rot_thresh);

  // set up all the tuning parameters
  laser_slam.setNumVertexKNN(scanreg_num_vertex_knn);
  laser_slam.setNumRefinementIterations(scanreg_num_ref_iter);
  laser_slam.global_map->setThicknessTheshold(mls_thickness_threshold);
  laser_slam.global_map->setHeightTheshold(mls_height_threshold);
  laser_slam.global_map->setClusterSigmaFactor(mls_cluster_sigma_factor);
  laser_slam.global_map->setClusterDistTheshold(mls_cluster_dist_threshold);
  laser_slam.global_map->setClusterCombineDist(mls_cluster_combine_dist);
  laser_slam.setRecomputeAllEdges(scanreg_recompute_all_edges);
  //---------------------------------------END COPYPASTA--------------------//

  // ROS PUBLISHERS
  ros::Publisher map_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>(
      "/laserslam/full_pointcloud", 1, true);
  ros::Publisher drivablility_pub = nh.advertise<nav_msgs::OccupancyGrid>(
      "/laserslam/drivability_map", 1, true);
  ros::Publisher elevation_pub =
      nh.advertise<grid_map_msgs::GridMap>("/laserslam/elevation_map", 1, true);

  std::string bag_file_path;
  std::string velo_topic;
  std::string ekf_topic;
  std::string gps_topic;
  std::vector<std::string> topics;
  double dis_inc, alt_ref;
  int skip;
  int counter = 0;

  geometry_msgs::Pose last_pose, pose_msg;
  geometry_msgs::PoseStamped velo_pose;

  bool register_next = true;
  bool initialized = false;
  if (nh.getParam("/skip_scans", skip)) {
    ROS_INFO_STREAM("skipping scans to start " << skip);
  } else {
    skip = 0;
    ROS_ERROR("not skipping scans");
    return -1;
  }

  // get the bag file path
  if (nh.getParam("/bag_file_path", bag_file_path)) {
    ROS_INFO_STREAM("using bag_file_path: " << bag_file_path);
  } else {
    ROS_ERROR("bag_file_path parameter not set");
    return -1;
  }

  if (nh.getParam("/velodyne_topic_name", velo_topic)) {
    ROS_INFO_STREAM("using pointcloud2 topic: " << velo_topic);
    topics.push_back(velo_topic);
  } else {
    ROS_ERROR("velodyne_topic_name parameter not set");
    return -1;
  }

  bool use_ekf_prior = true;
  if (nh.getParam("/ekf_topic_name", ekf_topic)) {
    ROS_INFO_STREAM("using eft topic name: " << ekf_topic);
    topics.push_back(ekf_topic);
  } else {
    use_ekf_prior = false;
    ROS_WARN("ekf_topic_name parameter not set");
  }

  if (nh.getParam("/distance_increment", dis_inc)) {
    ROS_INFO_STREAM("using distance increment: " << dis_inc);
  } else {
    ROS_ERROR("distance_increment parameter not set");
    return -1;
  }

  if (nh.getParam("/gps_topic_name", gps_topic)) {
    ROS_INFO_STREAM("using gps_topic_name: " << gps_topic);
    topics.push_back(gps_topic);
  } else {
    ROS_ERROR("gps_topic_name parameter not set");
    return -1;
  }

  rosbag::Bag bag;
  try {
    bag.open(bag_file_path, rosbag::bagmode::Read); // throws exception if fails
    ROS_INFO_STREAM("Bag is open");
  } catch (rosbag::BagException &ex) {

    ROS_ERROR("Bag exception : %s", ex.what());
  }

  sensor_msgs::PointCloud2 full_map;

  rosbag::View view(bag, rosbag::TopicQuery(topics), ros::TIME_MIN,
                    ros::TIME_MAX, true);

  ros::Time start = ros::Time::now();
  BOOST_FOREACH (rosbag::MessageInstance const m, view) {
    if (!ros::ok())
      break;

    if (use_ekf_prior && m.getTopic() == ekf_topic) {
      // ROS_INFO_STREAM("Getting ekf message");
      nav_msgs::Odometry::ConstPtr vehicle_msg =
          m.instantiate<nav_msgs::Odometry>();

      if (vehicle_msg != NULL) {
        pose_msg = vehicle_msg->pose.pose;
        if (!register_next && (get_2d_dist(last_pose, pose_msg) > dis_inc)) {
          register_next = true;
          counter = 0;
          last_pose = pose_msg;
        }
      } else {
        ROS_ERROR_STREAM("Could not interpret message from topic "
                         << m.getTopic() << " as nav_msg::Odometry.");
      }
    } else if ((m.getTopic() == velo_topic)) {
      register_next = false;
      sensor_msgs::PointCloud2::ConstPtr cloud_msg =
          m.instantiate<sensor_msgs::PointCloud2>();
      if (cloud_msg != nullptr) {
        pcl::PointCloud<velodyne_pointcloud::PointXYZIR>::Ptr input_cloud(
            new pcl::PointCloud<velodyne_pointcloud::PointXYZIR>);
        pcl::fromROSMsg(*cloud_msg, *input_cloud);

        Eigen::Affine3d input_pose;
        tf::poseMsgToEigen(pose_msg, input_pose);

        if (laser_slam.insertAndProcess(input_cloud, input_pose)) {
          pcl::PointCloud<velodyne_pointcloud::PointXYZIR> translated_cloud;
          pcl::transformPointCloud(*input_cloud, translated_cloud, input_pose);
          sensor_msgs::PointCloud2 translated_cloud_msg;
          pcl::toROSMsg(translated_cloud, translated_cloud_msg);
          translated_cloud_msg.header.frame_id = "/global";
          translated_cloud_msg.header.stamp = ros::Time::now();
          map_cloud_pub.publish(translated_cloud_msg);
        } else {
          ROS_ERROR("Failed to insert and process points.");
          return 1;
        }
      } else {
        ROS_ERROR("Could not get PointCloud2 from message.");
        ros::Duration(3).sleep();
      }
    } else if (m.getTopic() == gps_topic) {
      sensor_msgs::NavSatFix::ConstPtr gps_msg =
          m.instantiate<sensor_msgs::NavSatFix>();
      if (!initialized) {
        initialized = true;
        alt_ref = gps_msg->altitude;
      }
      // pose_msg.position.z = gps_msg->altitude - alt_ref;
    }
  }
  bag.close();

  rosbag::Bag newbag;

  try {
    bag.open("/home/cglwn/fullmap.bag",
             rosbag::bagmode::Write); // throws exception if fails
    ROS_INFO_STREAM("Bag is open");
  } catch (rosbag::BagException &ex) {
    ROS_ERROR("Bag exception : %s", ex.what());
  }
  pcl::toROSMsg(*(laser_slam.global_map->getGlobalCloud()), full_map);
  full_map.header.frame_id = "/global";
  full_map.header.stamp = ros::Time::now();
  bag.write("/laserslam/full_pointcloud", ros::Time::now(), full_map);

  ROS_INFO_STREAM("All Scans Complete");

  ROS_WARN_STREAM("Execution time: " << (ros::Time::now() - start));
  return 0;
}
