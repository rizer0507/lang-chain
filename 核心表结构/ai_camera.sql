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

 Date: 16/01/2026 17:56:02
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for ai_camera
-- ----------------------------
DROP TABLE IF EXISTS `ai_camera`;
CREATE TABLE `ai_camera`  (
  `id` bigint NOT NULL COMMENT '主键ID',
  `camera_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '摄像头业务ID',
  `camera_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '摄像头名称',
  `access_type` tinyint NOT NULL COMMENT '接入类型编码',
  `image_retention_days` int NULL DEFAULT 30 COMMENT '图片保留天数',
  `callback_url` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '回调URL',
  `custom` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '自定义属性',
  `is_poly` tinyint NULL DEFAULT 0,
  `is_mask` tinyint NULL DEFAULT 0 COMMENT '是否生成掩码信息',
  `condition` json NULL COMMENT '条件参数',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `camera_status` tinyint NOT NULL DEFAULT 0 COMMENT '摄像头状态: 0-待分析,1-分析中,2-停止,3-完成,4-错误',
  `stream_url` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '视频流地址',
  `config_mode` tinyint NOT NULL DEFAULT 0 COMMENT '配置模式:0-轮巡,1-实时',
  `patrol_interval` int NULL DEFAULT 5 COMMENT '轮巡间隔(分钟)',
  `analyze_time_list` json NULL COMMENT '分析时间段，格式：[{\"startTime\":\"00:00\",\"endTime\":\"23:59\"}]',
  `retry_count` int NULL DEFAULT 0 COMMENT '重试次数',
  `max_retry_count` int NULL DEFAULT 3 COMMENT '最大重试次数',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_camera_id`(`camera_id` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '摄像头配置表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for ai_camera_scene
-- ----------------------------
DROP TABLE IF EXISTS `ai_camera_scene`;
CREATE TABLE `ai_camera_scene`  (
  `cs_id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `camera_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '摄像头ID',
  `scene_id` bigint NOT NULL COMMENT '场景ID',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`cs_id`) USING BTREE,
  UNIQUE INDEX `uk_camera_scene`(`camera_id` ASC, `scene_id` ASC) USING BTREE,
  INDEX `idx_camera_id`(`camera_id` ASC) USING BTREE,
  INDEX `idx_scene_id`(`scene_id` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2010657046285254658 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '摄像头-场景关联表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for ai_channel
-- ----------------------------
DROP TABLE IF EXISTS `ai_channel`;
CREATE TABLE `ai_channel`  (
  `channel_id` bigint NOT NULL AUTO_INCREMENT COMMENT '通道ID',
  `channel_code` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '通道编码',
  `channel_name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '通道名称',
  `camera_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '绑定的摄像头ID',
  `live_status` tinyint NOT NULL DEFAULT 0 COMMENT '直播状态: 0-未直播,1-直播中,2-直播异常',
  `play_url` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '播放地址',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `remark` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '备注',
  PRIMARY KEY (`channel_id`) USING BTREE,
  UNIQUE INDEX `uk_channel_code`(`channel_code` ASC) USING BTREE,
  UNIQUE INDEX `uk_camera_id`(`camera_id` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1980942182336999433 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '直播通道表' ROW_FORMAT = DYNAMIC;

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

-- ----------------------------
-- Table structure for ai_label
-- ----------------------------
DROP TABLE IF EXISTS `ai_label`;
CREATE TABLE `ai_label`  (
  `label_id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `model_id` bigint NOT NULL COMMENT '关联模型ID',
  `class_id` int NOT NULL COMMENT '训练标签ID',
  `class_name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '训练标签名称',
  `conf` float NULL DEFAULT NULL COMMENT '置信度阈值',
  `description` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '标签描述',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`label_id`) USING BTREE,
  UNIQUE INDEX `uk_model_class`(`model_id` ASC, `class_id` ASC) USING BTREE,
  INDEX `idx_model_id`(`model_id` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2010610209543942146 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '标签管理表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for ai_minio_storage
-- ----------------------------
DROP TABLE IF EXISTS `ai_minio_storage`;
CREATE TABLE `ai_minio_storage`  (
  `storage_id` bigint NOT NULL AUTO_INCREMENT COMMENT '存储记录ID(主键)',
  `camera_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '摄像头ID',
  `bucket_name` varchar(63) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT 'MinIO桶名称',
  `object_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '对象名称',
  `content_type` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT 'image/jpeg' COMMENT '内容类型',
  `file_size` bigint NULL DEFAULT NULL COMMENT '文件大小(字节)',
  `etag` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '文件ETag',
  `storage_class` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT 'STANDARD' COMMENT '存储类别',
  `metadata` json NULL COMMENT '元数据',
  `processed` tinyint(1) NULL DEFAULT 0 COMMENT '是否已处理',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`storage_id`) USING BTREE
) ENGINE = MyISAM AUTO_INCREMENT = 2010642947695362050 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = 'MinIO存储记录表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Table structure for ai_model
-- ----------------------------
DROP TABLE IF EXISTS `ai_model`;
CREATE TABLE `ai_model`  (
  `model_id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `code` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '模型代码',
  `name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '模型名称',
  `description` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '模型描述',
  `status` tinyint NOT NULL DEFAULT 1 COMMENT '状态（0停用 1启用）',
  `type` tinyint NOT NULL DEFAULT 0 COMMENT '类型（0目标检测 1语义分割 2目标分类）',
  `enable_mv_ids` json NULL COMMENT '启用的模型版本列表',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`model_id`) USING BTREE,
  UNIQUE INDEX `uk_code`(`code` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2010610209531359234 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '模型管理表' ROW_FORMAT = DYNAMIC;

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

-- ----------------------------
-- Table structure for ai_plugin_idp
-- ----------------------------
DROP TABLE IF EXISTS `ai_plugin_idp`;
CREATE TABLE `ai_plugin_idp`  (
  `idp_id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `camera_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '摄像头ID',
  `fence_info` json NULL COMMENT '多边形信息',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`idp_id`) USING BTREE,
  INDEX `idx_camera`(`camera_id` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '人员闯入信息表' ROW_FORMAT = Dynamic;

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

-- ----------------------------
-- Table structure for ai_server
-- ----------------------------
DROP TABLE IF EXISTS `ai_server`;
CREATE TABLE `ai_server`  (
  `server_id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `code` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '服务代码',
  `name` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '服务名称',
  `rgb` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '绘制颜色',
  `type` tinyint NOT NULL DEFAULT 0 COMMENT '服务类型（0单服务 1复合服务）',
  `child_list` json NULL COMMENT '子服务列表（复合类型时，该值必填）',
  `description` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT NULL COMMENT '服务描述',
  `status` tinyint NOT NULL DEFAULT 1 COMMENT '状态（0停用 1启用）',
  `create_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '' COMMENT '创建者',
  `create_time` datetime NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_by` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NULL DEFAULT '' COMMENT '更新者',
  `update_time` datetime NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`server_id`) USING BTREE,
  UNIQUE INDEX `uk_code`(`code` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2010607447028527106 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci COMMENT = '服务管理表' ROW_FORMAT = DYNAMIC;

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

-- ----------------------------
-- Table structure for ai_video_processing_task
-- ----------------------------
DROP TABLE IF EXISTS `ai_video_processing_task`;
CREATE TABLE `ai_video_processing_task`  (
  `v_task_id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `camera_id` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '关联摄像头ID',
  `error_message` text CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL COMMENT '错误信息',
  `retry_count` int NULL DEFAULT 0 COMMENT '重试次数',
  `next_retry_time` datetime NULL DEFAULT NULL COMMENT '下次重试时间',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `del_flag` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT '0' COMMENT '逻辑删除',
  PRIMARY KEY (`v_task_id`) USING BTREE,
  INDEX `idx_camera`(`camera_id` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2010656951833722883 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '视频流任务表' ROW_FORMAT = DYNAMIC;

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
