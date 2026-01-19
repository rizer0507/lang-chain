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

 Date: 16/01/2026 17:58:12
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for ai_scene
-- ----------------------------
DROP TABLE IF EXISTS `ai_scene`;
CREATE TABLE `ai_scene`  (
  `scene_id` bigint NOT NULL AUTO_INCREMENT COMMENT '场景ID',
  `code` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '场景编码',
  `name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '场景名称',
  `description` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '场景描述',
  `bind_camera_count` int NULL DEFAULT 0 COMMENT '绑定设备数',
  `status` tinyint NOT NULL DEFAULT 1 COMMENT '状态（0停用 1启用）',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `scene_type` tinyint NULL DEFAULT NULL COMMENT '场景类型（0服务创建 1插件创建）',
  `plugin_code` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '插件代码',
  PRIMARY KEY (`scene_id`) USING BTREE,
  UNIQUE INDEX `uk_code`(`code` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2010634764841488386 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '场景管理表' ROW_FORMAT = DYNAMIC;

SET FOREIGN_KEY_CHECKS = 1;
