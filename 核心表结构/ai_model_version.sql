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

 Date: 16/01/2026 17:58:03
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for ai_model_version
-- ----------------------------
DROP TABLE IF EXISTS `ai_model_version`;
CREATE TABLE `ai_model_version`  (
  `mv_id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `model_id` bigint NOT NULL COMMENT '模型ID',
  `version` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '模型版本号',
  `path` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '模型文件路径',
  `condition` json NULL COMMENT '条件参数',
  `conf` float NOT NULL COMMENT '置信度阈值',
  `enable` tinyint NOT NULL DEFAULT 0 COMMENT '状态（0未部署 1已部署）',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `model_config` json NULL COMMENT '模型完整配置参数',
  `stream_config` json NULL COMMENT '流式处理专用配置',
  `device` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '设备',
  `description` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '模型版本描述',
  PRIMARY KEY (`mv_id`) USING BTREE,
  INDEX `idx_model_id`(`model_id` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2010607572132032515 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '模型版本管理表' ROW_FORMAT = DYNAMIC;

SET FOREIGN_KEY_CHECKS = 1;
