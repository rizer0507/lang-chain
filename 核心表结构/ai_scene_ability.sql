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

 Date: 16/01/2026 17:58:17
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for ai_scene_ability
-- ----------------------------
DROP TABLE IF EXISTS `ai_scene_ability`;
CREATE TABLE `ai_scene_ability`  (
  `sa_id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `scene_id` bigint NOT NULL COMMENT '场景ID',
  `ability_id` bigint NOT NULL COMMENT '能力ID',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`sa_id`) USING BTREE,
  UNIQUE INDEX `uk_scene_ability`(`scene_id` ASC, `ability_id` ASC) USING BTREE,
  INDEX `idx_scene_id`(`scene_id` ASC) USING BTREE,
  INDEX `idx_ability_id`(`ability_id` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2010634807019409410 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '场景-能力关联表' ROW_FORMAT = DYNAMIC;

SET FOREIGN_KEY_CHECKS = 1;
