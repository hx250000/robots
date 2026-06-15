import cv2
import numpy as np
from controller import Robot

# 1. 初始化机器人
robot = Robot()
time_step = int(robot.getBasicTimeStep())

# 2. 初始化底盘电机 (麦克纳姆轮)
# 对应官方 Youbot 的 4 个轮子节点
wheels = []
wheel_names = ['wheel1', 'wheel2', 'wheel3', 'wheel4']
for name in wheel_names:
    motor = robot.getDevice(name)
    motor.setPosition(float('inf')) # 设置为速度控制模式
    motor.setVelocity(0.0)
    wheels.append(motor)

# 3. 初始化机械臂与夹爪电机
arm_motors = []
for i in range(1, 6):
    arm_motors.append(robot.getDevice(f"arm_motor_{i}"))

# 夹爪电机 (控制 maxForce 防止穿模)
left_gripper = robot.getDevice("gripper_finger_joint_1")
right_gripper = robot.getDevice("gripper_finger_joint_2")

# 4. 初始化传感器
camera = robot.getDevice("camera") # 确保名字和你加的一致
camera.enable(time_step)

lidar = robot.getDevice("lidar") # 如果加了雷达
if lidar:
    lidar.enable(time_step)

# 5. 有限状态机状态定义
STATE_SEARCH_BOX   = 0  # 旋转寻找木块
STATE_MOVE_TO_BOX  = 1  # 驶向木块
STATE_PICK_BOX     = 2  # 机械臂抓取木块
STATE_NAV_TO_TABLE = 3  # 导航至中央桌子
STATE_PLACE_BOX    = 4  # 机械臂放置木块
STATE_RESET_ARM    = 5  # 复位机械臂
STATE_DONE         = 6  # 任务完成

current_state = STATE_SEARCH_BOX

# 6. 辅助控制函数
def set_chassis_velocity(vx, vy, omega):
    """
    麦克纳姆轮底盘逆运动学解算
    vx: 前进速度, vy: 横移速度, omega: 旋转速度
    """
    # 针对标准 Youbot 麦轮排布的速度映射
    wheels[0].setVelocity(vx - vy - omega) # 左前
    wheels[1].setVelocity(vx + vy + omega) # 右前
    wheels[2].setVelocity(vx + vy - omega) # 左后
    wheels[3].setVelocity(vx - vy + omega) # 右后

def get_box_position_from_camera():
    """
    通过 Camera 图像识别黄色木块的中心水平偏移
    返回: 目标在图像中的相对 X 轴偏移（-1.0 到 1.0），若没找到返回 None
    """
    width = camera.getWidth()
    height = camera.getHeight()
    img_rgba = camera.getImage()
    
    # 转换为 OpenCV 的 BGR 格式进行颜色分割
    img_bgr = np.frombuffer(img_rgba, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 根据木块颜色调整 HSV 阈值 (以下为一般木质黄色/棕色区间)
    lower_yellow = np.array([10, 50, 50])
    upper_yellow = np.array([30, 255, 255])
    
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    M = cv2.moments(mask)
    
    if M["m00"] > 500: # 面积大于阈值，说明看到了木块
        cx = int(M["m10"] / M["m00"])
        # 计算相对中心点的偏移量
        offset_x = (cx - (width / 2)) / (width / 2)
        return offset_x
    return None

def move_arm_to_pose(angles):
    """控制 5 个机械臂关节的角度 (弧度)"""
    for motor, angle in zip(arm_motors, angles):
        motor.setPosition(angle)

def control_gripper(close=True):
    """夹爪控制，加入力控防止穿模"""
    if close:
        left_gripper.setAvailableForce(20.0)  # 限制最大抓取力 [cite: 21]
        right_gripper.setAvailableForce(20.0)
        left_gripper.setPosition(0.0)         # 闭合
        right_gripper.setPosition(0.0)
    else:
        left_gripper.setPosition(0.025)       # 张开
        right_gripper.setPosition(0.025)

# ==================== 主循环 ====================
while robot.step(time_step) != -1:
    
    # 状态机调度
    if current_state == STATE_SEARCH_BOX:
        # 状态0：原地自转寻找木块
        offset = get_box_position_from_camera()
        if offset is not None:
            set_chassis_velocity(0, 0, 0) # 找到目标，停下
            current_state = STATE_MOVE_TO_BOX
        else:
            set_chassis_velocity(0, 0, 0.5) # 找不到就继续转
            
    elif current_state == STATE_MOVE_TO_BOX:
        # 状态1：利用 PID 思想，根据图像偏移逼近木块
        offset = get_box_position_from_camera()
        if offset is None:
            current_state = STATE_SEARCH_BOX # 目标丢了，重新搜索
        else:
            # 简单的 P 控制
            omega = -offset * 1.5  # 修正对齐
            vx = 0.4               # 前进速度
            
            # TODO: 结合 Lidar 距离判断是否足够接近木块
            # 如果没有雷达，可以判断木块在图像中的面积大小
            is_close_enough = False 
            
            if is_close_enough:
                set_chassis_velocity(0, 0, 0)
                current_state = STATE_PICK_BOX
            else:
                set_chassis_velocity(vx, 0, omega)
                
    elif current_state == STATE_PICK_BOX:
        # 状态2：执行抓取
        # 1. 打开夹爪
        control_gripper(close=False)
        # 2. 机械臂探出（这里输入预先计算好的逆解角度或硬编码对齐角度） [cite: 20]
        # 示例角度: 仅作参考，需根据你的木块高度微调
        move_arm_to_pose([0.0, 0.5, -1.0, -0.5, 0.0])
        
        # TODO: 延时或等待关节到位后闭合夹爪
        # control_gripper(close=True)
        
        # 3. 抬起机械臂，切换状态
        current_state = STATE_NAV_TO_TABLE
        
    elif current_state == STATE_NAV_TO_TABLE:
        # 状态3：搬运木块至中央桌子 (坐标 0,0) [cite: 9, 12]
        # 可以通过 Odometry(里程计) 计算自身位置，或者简单粗暴地用 Supervisor API 获取坐标导航
        # 接近桌子时，记得利用底盘传感器防止撞到桌腿 [cite: 17, 21]
        
        # 伪代码：如果到了桌子边缘
        arrived_at_table = False 
        if arrived_at_table:
            set_chassis_velocity(0, 0, 0)
            current_state = STATE_PLACE_BOX
            
    elif current_state == STATE_PLACE_BOX:
        # 状态4：机械臂伸到桌面上方平稳放下 [cite: 12]
        # 桌面高度 0.75m，Youbot底盘高约 0.1m，所以机械臂末端需要抬高到相对底盘 >0.65m 处 [cite: 9, 10]
        move_arm_to_pose([0.0, 0.2, 0.2, 0.0, 0.0]) 
        
        # 松开夹爪
        control_gripper(close=False)
        current_state = STATE_RESET_ARM
        
    elif current_state == STATE_RESET_ARM:
        # 状态5：收回机械臂，准备收下一个木块
        move_arm_to_pose([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # TODO: 计数器+1。如果4个木块都运完了 -> STATE_DONE；没运完 -> STATE_SEARCH_BOX
        current_state = STATE_SEARCH_BOX 
        
    elif current_state == STATE_DONE:
        set_chassis_velocity(0, 0, 0)
        print("所有木块收集完毕！")
        break