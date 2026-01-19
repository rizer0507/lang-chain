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

 Date: 16/01/2026 17:58:25
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for ai_server_label
-- ----------------------------
DROP TABLE IF EXISTS `ai_server_label`;
CREATE TABLE `ai_server_label`  (
  `sl_id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `server_id` bigint NOT NULL COMMENT '服务ID',
  `label_id` bigint NOT NULL COMMENT '标签ID',
  `remark` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '备注',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`sl_id`) USING BTREE,
  UNIQUE INDEX `uk_server_label`(`server_id` ASC, `label_id` ASC) USING BTREE,
  INDEX `idx_server_id`(`server_id` ASC) USING BTREE,
  INDEX `idx_label_id`(`label_id` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2010607447041110019 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '服务-标签关联表' ROW_FORMAT = DYNAMIC;

SET FOREIGN_KEY_CHECKS = 1;
