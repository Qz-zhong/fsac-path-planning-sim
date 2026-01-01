"""
无人车路径规划仿真（最终稳定版：优秀路径+智能起点）
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import Delaunay
from scipy.interpolate import splprep, splev
from collections import deque

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 系统参数 --------------------------
MAX_SPEED = 4.0
MAX_STEERING = 0.5
WHEELBASE = 1.57
LOOKAHEAD_DISTANCE = 6.0
ROBOT_RADIUS = 3.0
SIM_DURATION = 300
DT = 0.05
SAFETY_DISTANCE = 5.0

# -------------------------- 核心算法（邻接排序，您认可的版本） --------------------------
class AngleBasedPlanner:
    def __init__(self, left_cones, right_cones):
        self.left_cones = left_cones
        self.right_cones = right_cones
        self.num_cones = min(len(left_cones), len(right_cones))
        self.inner_indices = set(range(self.num_cones))
        self.outer_indices = set(range(self.num_cones, 2*self.num_cones))
        self.planned_path = None
        self.P = None
        self.tri = None
        self.boundary_edges = None
        self.cross_side_midpoints = None
        self.interior_triangles = None

    def plan_closed_path(self):
        self.P = np.vstack([self.left_cones[:self.num_cones], self.right_cones[:self.num_cones]])
        self.boundary_edges = []
        for i in range(self.num_cones):
            self.boundary_edges.append([i, self.num_cones + i])
        self.boundary_edges.append([self.num_cones - 1, 2*self.num_cones - 1])
        self.boundary_edges.append([0, self.num_cones])

        self.tri = Delaunay(self.P)
        boundary_set = set(frozenset(edge) for edge in self.boundary_edges)

        # 识别内部三角形
        is_interior = np.zeros(len(self.tri.simplices), dtype=bool)
        initial_tria = 0
        is_interior[initial_tria] = True
        queue = deque([initial_tria])
        while queue:
            current = queue.popleft()
            for neighbor in self.tri.neighbors[current]:
                if neighbor == -1 or is_interior[neighbor]:
                    continue
                shared_edge = frozenset(set(self.tri.simplices[current]) & set(self.tri.simplices[neighbor]))
                if shared_edge not in boundary_set:
                    is_interior[neighbor] = True
                    queue.append(neighbor)
        self.interior_triangles = self.tri.simplices[is_interior]

        # 提取跨侧中点
        cross_side_midpoints_list = []
        for i, tria in enumerate(self.tri.simplices):
            if is_interior[i]:
                for j in range(3):
                    edge = frozenset([tria[j], tria[(j+1)%3]])
                    if edge in boundary_set:
                        continue
                    p1, p2 = list(edge)
                    if (p1 in self.inner_indices and p2 in self.outer_indices) or (p2 in self.inner_indices and p1 in self.outer_indices):
                        midpoint = (self.P[p1] + self.P[p2]) / 2.0
                        cross_side_midpoints_list.append(midpoint)
        if len(cross_side_midpoints_list) < 4:
            print(f"错误：只找到了 {len(cross_side_midpoints_list)} 个跨侧中点，不足以进行3阶样条插值。")
            return None
        self.cross_side_midpoints = np.array(cross_side_midpoints_list)

        # 【核心优化】邻接排序（确保路径按顺序经过最近点）
        sorted_midpoints = [self.cross_side_midpoints[0]]
        remaining = set(range(1, len(self.cross_side_midpoints)))
        while remaining:
            last = sorted_midpoints[-1]
            dists = np.linalg.norm(self.cross_side_midpoints[list(remaining)] - last, axis=1)
            nearest_idx = list(remaining)[np.argmin(dists)]
            sorted_midpoints.append(self.cross_side_midpoints[nearest_idx])
            remaining.remove(nearest_idx)
        sorted_midpoints = np.array(sorted_midpoints)

        # 插值与平滑
        closed_midpoints = np.vstack([sorted_midpoints, sorted_midpoints[0]])
        distances = np.linalg.norm(np.diff(closed_midpoints, axis=0), axis=1)
        u_closed = np.insert(np.cumsum(distances), 0, 0)
        try:
            unique_indices = [0]
            for i in range(1, len(closed_midpoints)):
                if distances[i-1] > 1e-6:
                    unique_indices.append(i)
            if len(unique_indices) < 4:
                print("错误：移除重复点后，控制点数量不足。")
                return None
            unique_midpoints = closed_midpoints[unique_indices]
            unique_u = u_closed[unique_indices]
            tck, _ = splprep([unique_midpoints[:, 0], unique_midpoints[:, 1]], u=unique_u, k=3, s=8.0, per=True)
            u_fine = np.linspace(unique_u[0], unique_u[-1], 300, endpoint=False)
            smooth_x, smooth_y = splev(u_fine, tck)
            self.planned_path = np.column_stack([smooth_x, smooth_y])
        except Exception as e:
            print(f"B样条插值失败: {e}")
            return None

        return self.planned_path

# -------------------------- 纯追踪控制器 (优化版，防止掉头) --------------------------
class PurePursuitController:
    def __init__(self, wheelbase, max_steering, lookahead_distance):
        self.wheelbase = wheelbase
        self.max_steering = max_steering
        self.lookahead_distance = lookahead_distance
        self.last_lookahead_idx = 0  # 记录上一次预瞄点索引，避免跳变

    def run_step(self, vehicle_pose, path):
        x, y, yaw = vehicle_pose
        path_len = len(path)

        # 1. 从上次预瞄点附近搜索最近点，避免全局搜索导致的跳变
        nearest_idx = self.last_lookahead_idx
        min_dist = np.linalg.norm(path[nearest_idx] - np.array([x, y]))
        for i in range(path_len):
            current_idx = (self.last_lookahead_idx - 5 + i) % path_len # 在附近10个点内搜索
            current_dist = np.linalg.norm(path[current_idx] - np.array([x, y]))
            if current_dist < min_dist:
                min_dist = current_dist
                nearest_idx = current_idx

        # 2. 搜索前进方向的预瞄点
        forward_vec = np.array([math.cos(yaw), math.sin(yaw)])
        lookahead_idx = nearest_idx
        found = False
        # 从最近点向后搜索一圈
        for i in range(path_len):
            current_idx = (nearest_idx + i) % path_len
            current_point = path[current_idx]
            point_vec = current_point - np.array([x, y])
            # 确保预瞄点在前进方向前方
            if np.dot(point_vec, forward_vec) > 0 and np.linalg.norm(point_vec) > self.lookahead_distance:
                lookahead_idx = current_idx
                found = True
                break
        if not found:
            # 如果找不到，就取最近点的下一个点，保证有输出
            lookahead_idx = (nearest_idx + 1) % path_len

        self.last_lookahead_idx = lookahead_idx  # 更新上次预瞄点

        # 3. 计算转向角
        lookahead_point = path[lookahead_idx]
        alpha = math.atan2(lookahead_point[1] - y, lookahead_point[0] - x) - yaw
        # 将alpha归一化到[-pi, pi]
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        L = np.linalg.norm(lookahead_point - np.array([x, y]))
        steering_angle = math.atan2(2 * self.wheelbase * math.sin(alpha), L)
        return np.clip(steering_angle, -self.max_steering, self.max_steering)

# -------------------------- 主仿真流程 (选择最优起点) --------------------------
def main():
    # 使用您提供的锥桶数据
    left_cones = np.array([
        (50, 20), (114, 19), (164, 16), (226, -2), (269, -19), (310, -20), (344, -16), (377, -1),
        (378, 27), (381, 69), (376, 105), (360, 132), (334, 134), (310, 115), (278, 10), (240, 100), (204, 96), (140, 107),
        (108, 122), (89, 138), (56, 147), (16, 143), (-15, 128), (-57, 107), (-120, 101), (-179, 129), (-212, 143),
        (-234, 129), (-204, 100), (-230, 64), (-210, 37), (-184, 21), (-132, 27), (-88, 21), (63, 17), (-10, 16)
    ])
    right_cones = np.array([
        (64, -13), (114, -11), (164, -14), (209, -23), (254, -43), (304, -52), (360, -42), (399, -23), (419, 25), (421, 74),
        (414, 125), (390, 165), (337, 175), (299, 154), (256, 135), (233, 129), (208, 125), (159, 136), (134, 147), (111, 163),
        (62, 176), (10, 168), (-34, 154), (-76, 132), (-128, 129), (-171, 158), (-219, 172), (-263, 147), (-272, 101),
        (-262, 52), (-232, 12), (-186, -8), (-138, -9), (-89, -13), (-43, -17), (10, -14)
    ])

    planner = AngleBasedPlanner(left_cones, right_cones)
    global_path = planner.plan_closed_path()
    if global_path is None:
        return

    controller = PurePursuitController(WHEELBASE, MAX_STEERING, LOOKAHEAD_DISTANCE)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title('无人车路径规划仿真（最终稳定版）')
    ax.set_xlabel('X 位置 (m)')
    ax.set_ylabel('Y 位置 (m)')
    ax.grid(True)
    ax.axis('equal')

    # 绘制内部三角形
    if planner.interior_triangles is not None and planner.tri is not None:
        x_tri = planner.tri.points[:, 0]
        y_tri = planner.tri.points[:, 1]
        ax.triplot(x_tri, y_tri, planner.interior_triangles, color='gray', linestyle='--', linewidth=0.5, label='内部三角形')

    # 绘制跨侧中点
    if planner.cross_side_midpoints is not None:
        ax.scatter(planner.cross_side_midpoints[:, 0], planner.cross_side_midpoints[:, 1], c='blue', s=50, marker='o', label='跨侧中点')

    # 绘制锥桶与路径
    ax.scatter(left_cones[:, 0], left_cones[:, 1], c='cyan', s=50, label='内侧锥桶', marker='^', edgecolors='black')
    ax.scatter(right_cones[:, 0], right_cones[:, 1], c='magenta', s=50, label='外侧锥桶', marker='^', edgecolors='black')
    ax.plot(global_path[:, 0], global_path[:, 1], 'g-', linewidth=3, label='最终平滑路径')

    # 动态元素
    car_body, = ax.plot([], [], 'r-', linewidth=5)
    trajectory_line, = ax.plot([], [], 'orange', linestyle=':', linewidth=2, label='车辆轨迹')
    ax.legend(loc='upper right')
    ax.set_xlim(-300, 450)
    ax.set_ylim(-60, 200)

    # 【关键修复】选择路径上Y坐标最小的点作为起点，通常在直道上，更稳定
    start_idx = np.argmin(global_path[:, 1])
    current_pose = [global_path[start_idx, 0], global_path[start_idx, 1], 0.0]

    current_speed = 0.0
    current_steering = 0.0
    steering_smoothing = 0.3
    laps_completed = 0
    lap_check_radius = 15.0
    trajectory = []

    def update(frame):
        nonlocal current_pose, current_speed, current_steering, laps_completed, trajectory
        trajectory.append(current_pose[:2])
        if len(trajectory) > 1000:
            trajectory = trajectory[-1000:]
        trajectory_line.set_data(np.array(trajectory)[:, 0], np.array(trajectory)[:, 1])

        # 检查是否完成一圈
        if np.linalg.norm(current_pose[:2] - global_path[start_idx]) < lap_check_radius and current_speed > 1.0:
            laps_completed += 1
            ax.set_title(f'无人车路径规划仿真（已完成 {laps_completed} 圈）')

        desired_steering = controller.run_step(current_pose, global_path)
        current_steering += (desired_steering - current_steering) * steering_smoothing

        throttle = 6.0 * max(0.2, 1 - current_speed / MAX_SPEED)
        current_speed += throttle * DT
        current_speed = np.clip(current_speed, 0, MAX_SPEED)

        yaw_rate = (current_speed / WHEELBASE) * math.tan(current_steering)
        current_pose[2] = (current_pose[2] + yaw_rate * DT) % (2 * math.pi)
        current_pose[0] += current_speed * math.cos(current_pose[2]) * DT
        current_pose[1] += current_speed * math.sin(current_pose[2]) * DT

        car_len, car_wid = 8, 3
        x, y, yaw = current_pose
        corners = np.array([[-car_len/2, -car_wid/2], [car_len/2, -car_wid/2], [car_len/2, car_wid/2], [-car_len/2, car_wid/2], [-car_len/2, -car_wid/2]])
        rot_matrix = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        car_corners = (corners @ rot_matrix.T) + [x, y]
        car_body.set_data(car_corners[:, 0], car_corners[:, 1])

        return car_body, trajectory_line

    ani = animation.FuncAnimation(fig, update, frames=int(SIM_DURATION/DT), interval=DT*1000, blit=True, repeat=False)
    plt.show()

if __name__ == '__main__':
    main()