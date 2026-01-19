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

 Date: 16/01/2026 17:55:48
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for ai_ability_server
-- ----------------------------
DROP TABLE IF EXISTS `ai_ability_server`;
CREATE TABLE `ai_ability_server`  (
  `as_id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `ability_id` bigint NOT NULL COMMENT '能力ID',
  `server_id` bigint NOT NULL COMMENT '服务ID',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`as_id`) USING BTREE,
  UNIQUE INDEX `uk_ability_server`(`ability_id` ASC, `server_id` ASC) USING BTREE,
  INDEX `idx_ability_id`(`ability_id` ASC) USING BTREE,
  INDEX `idx_server_id`(`server_id` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2010617761874833411 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '能力-服务关联表' ROW_FORMAT = DYNAMIC;

SET FOREIGN_KEY_CHECKS = 1;
