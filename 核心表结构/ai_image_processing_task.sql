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

 Date: 16/01/2026 17:57:44
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for ai_image_processing_task
-- ----------------------------
DROP TABLE IF EXISTS `ai_image_processing_task`;
CREATE TABLE `ai_image_processing_task`  (
  `task_id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `camera_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '摄像头ID',
  `original_image_url` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '原始图片URL',
  `comparison_image_url` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '对比图URL',
  `local_image_url` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '本地图片URL',
  `comparison_local_image_url` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '本地对比图片URL',
  `traceability_image_url` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '溯源图片URL',
  `processed_image_url` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '处理后的图片URL',
  `task_status` tinyint NOT NULL DEFAULT 0 COMMENT '任务状态: PENDING,PROCESSING,COMPLETED,FAILED',
  `error_message` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '错误信息',
  `analysis_result` json NULL COMMENT '分析结果',
  `retry_count` int NULL DEFAULT 0 COMMENT '重试次数',
  `next_retry_time` datetime NULL DEFAULT NULL COMMENT '下次重试时间',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `callback_result` json NULL COMMENT '回调结果',
  `response_result` json NULL COMMENT '响应结果',
  `image_ext` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '图片回调自定义字段',
  `processed_image_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '处理后图片名称',
  PRIMARY KEY (`task_id`) USING BTREE,
  INDEX `idx_camera_model`(`camera_id` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1997953828742848515 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '图片处理任务表' ROW_FORMAT = DYNAMIC;

SET FOREIGN_KEY_CHECKS = 1;
