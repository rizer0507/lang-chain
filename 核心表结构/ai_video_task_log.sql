/*
 Navicat MySQL Dump SQL

 Source Server         : 253
 Source Server Type    : MySQL
 Source Server Version : 80027 (8.0.27)
 Source Host           : 192.168.1.253:3306
 Source Schema         : hetu_inference

 Target Server Type    : MySQL
 Target Server Version : 80027 (8.0.27)
 File Encoding         : 65001

 Date: 16/01/2026 17:58:34
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for ai_video_task_log
-- ----------------------------
DROP TABLE IF EXISTS `ai_video_task_log`;
CREATE TABLE `ai_video_task_log`  (
  `log_id` bigint NOT NULL AUTO_INCREMENT COMMENT '日志ID(主键)',
  `v_task_id` bigint NOT NULL COMMENT '视频任务ID',
  `camera_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '摄像头ID',
  `original_image_url` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '原始图片URL',
  `traceability_image_url` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '溯源图片URL',
  `processed_image_url` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '处理后图片URL',
  `frame_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '帧ID',
  `original_result` json NULL COMMENT '原始分析结果数据',
  `analysis_result` json NULL COMMENT '分析结果数据',
  `log_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '记录时间',
  `callback_result` json NULL COMMENT '回调结果',
  `response_result` json NULL COMMENT '响应结果',
  `processed_image_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '处理后图片名称',
  PRIMARY KEY (`log_id`) USING BTREE,
  INDEX `idx_v_task_id`(`v_task_id`) USING BTREE,
  INDEX `idx_camera_id`(`camera_id`) USING BTREE,
  INDEX `idx_log_time`(`log_time`) USING BTREE
) ENGINE = MyISAM AUTO_INCREMENT = 13876 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '视频任务分析日志表' ROW_FORMAT = DYNAMIC;

SET FOREIGN_KEY_CHECKS = 1;
