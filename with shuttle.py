import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import random
import math
from collections import deque
import pandas as pd
from sklearn.neural_network import MLPRegressor
import threading
import json
import dashscope
from dashscope import Generation
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# API KEY (请在此处填入您的阿里云 DashScope API Key)
# ==========================================
dashscope.api_key = "您的API KEY"

# 设置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# --- 1. 地图常量定义 ---
MACRO_ROWS = 5
MACRO_COLS = 5
BUILDING_SIZE = 20
ROAD_WIDTH = 5
CELL_SIZE = BUILDING_SIZE + ROAD_WIDTH

MAP_HEIGHT = MACRO_ROWS * CELL_SIZE + ROAD_WIDTH
MAP_WIDTH = MACRO_COLS * CELL_SIZE + ROAD_WIDTH

ROAD = 0
OBSTACLE = 1
CLASSROOM = 2
DORM = 3
CANTEEN = 4
PLAYGROUND = 5

# --- 2. 建筑与实体类 ---
class Building:
    def __init__(self, name, b_type, x, y, w, h):
        self.name = name
        self.b_type = b_type
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.inside_students = []
        self.door_positions = []

class Student:
    def __init__(self, student_id):
        self.id = student_id
        self.x = -1
        self.y = -1
        self.mood = 100.0
        self.has_bike = False
        self.willing_to_ride = False
        self.home_dorm = None
        self.target_building = None
        self.primary_target = None
        self.is_active = False
        self.dwell_timer = 0
        self.stuck_ticks = 0

    def update_dynamic_ride_willingness(self, current_g):
        if current_g <= 0 or math.isinf(current_g):
            self.willing_to_ride = False
        else:
            self.willing_to_ride = (random.random() < min(current_g / 100.0, 1.0))

    def try_grab_bike(self, bike_grid):
        if self.has_bike or not self.willing_to_ride:
            return False
        for oy in range(-2, 2):
            for ox in range(-2, 2):
                nx = self.x + ox
                ny = self.y + oy
                if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT and bike_grid[ny, nx] > 0:
                    self.has_bike = True
                    bike_grid[ny, nx] -= 1
                    self.mood = min(100.0, self.mood + 3.0)
                    return True
        return False

    def move(self, map_grid, gravity_field, occupancy_grid, bike_grid):
        if not self.is_active:
            return

        straight_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        diag_dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        valid_moves = []
        current_g = gravity_field[self.y, self.x]

        ideal_dx, ideal_dy = 0, 0
        if 0 < self.x < map_grid.shape[1] - 1:
            gw = gravity_field[self.y, self.x - 1]
            ge = gravity_field[self.y, self.x + 1]
            if not math.isinf(gw) and not math.isinf(ge):
                ideal_dx = gw - ge
        if 0 < self.y < map_grid.shape[0] - 1:
            gn = gravity_field[self.y - 1, self.x]
            gs = gravity_field[self.y + 1, self.x]
            if not math.isinf(gn) and not math.isinf(gs):
                ideal_dy = gn - gs

        bike_pull_x, bike_pull_y = 0, 0
        if not self.has_bike and self.willing_to_ride:
            for oy in range(-2, 3):
                for ox in range(-2, 3):
                    by = self.y + oy
                    bx = self.x + ox
                    if 0 <= bx < map_grid.shape[1] and 0 <= by < map_grid.shape[0] and bike_grid[by, bx] > 0:
                        bike_pull_x += ox
                        bike_pull_y += oy

        tolerance = 2 if self.stuck_ticks > 2 else 0
        if not self.has_bike and self.willing_to_ride and (bike_pull_x != 0 or bike_pull_y != 0):
            tolerance = max(tolerance, 1)

        max_occupancy = 3 if self.has_bike else 2

        for is_diag, dirs in [(False, straight_dirs), (True, diag_dirs)]:
            for dx, dy in dirs:
                nx = self.x + dx
                ny = self.y + dy

                if not (0 <= nx < map_grid.shape[1] and 0 <= ny < map_grid.shape[0]):
                    continue
                if map_grid[ny, nx] != ROAD or occupancy_grid[ny, nx] >= max_occupancy:
                    continue

                g_val = gravity_field[ny, nx]
                if g_val <= current_g + tolerance and not math.isinf(g_val):
                    perceived_g = g_val + (0.1 if is_diag else 0.0)
                    if not self.has_bike and self.willing_to_ride:
                        if bike_grid[ny, nx] > 0:
                            perceived_g -= 5.0
                        else:
                            bp = bike_pull_x * dx + bike_pull_y * dy
                            if bp > 0:
                                perceived_g -= 1.5
                    cp = ideal_dx * dy - ideal_dy * dx
                    if cp > 0:
                        perceived_g -= 0.6
                    elif cp < 0:
                        perceived_g += 0.6
                    if g_val > current_g:
                        perceived_g += 2.0
                    valid_moves.append((nx, ny, perceived_g))

        if not valid_moves:
            self.stuck_ticks += 1
            return

        self.stuck_ticks = 0
        min_g = min(v[2] for v in valid_moves)
        weights = [math.exp(-2.0 * (v[2] - min_g)) for v in valid_moves]
        chosen_move = random.choices(valid_moves, weights=weights, k=1)[0]

        occupancy_grid[self.y, self.x] -= 1
        self.x = chosen_move[0]
        self.y = chosen_move[1]
        occupancy_grid[self.y, self.x] += 1

        self.mood = max(0.0, self.mood - (0.05 if self.has_bike else 0.15))

class Shuttle:
    def __init__(self):
        self.x = -1
        self.y = -1
        self.is_active = False
        self.bikes_loaded = 0
        self.capacity = 30
        self.target_building = None

    def activate(self, source_b, target_b, bike_grid, b_zones):
        self.target_building = target_b
        self.is_active = True
        sx, sy = random.choice(source_b.door_positions)
        self.x = sx
        self.y = sy

        spots = list(b_zones[source_b])
        spots.sort(key=lambda p: bike_grid[p[1], p[0]], reverse=True)

        self.bikes_loaded = 0
        for px, py in spots:
            while bike_grid[py, px] > 0 and self.bikes_loaded < self.capacity:
                bike_grid[py, px] -= 1
                self.bikes_loaded += 1

        if self.bikes_loaded == 0:
            self.is_active = False

    def move(self, map_grid, gravity_field):
        if not self.is_active:
            return
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        current_g = gravity_field[self.y, self.x]
        valid_moves = []
        for dx, dy in directions:
            nx = self.x + dx
            ny = self.y + dy
            if 0 <= nx < map_grid.shape[1] and 0 <= ny < map_grid.shape[0]:
                if map_grid[ny, nx] == ROAD:
                    g_val = gravity_field[ny, nx]
                    if g_val < current_g:
                        valid_moves.append((nx, ny, g_val))
        if valid_moves:
            best_move = min(valid_moves, key=lambda v: v[2])
            self.x = best_move[0]
            self.y = best_move[1]

# --- 3. 地图与主控系统 ---
class Map:
    def __init__(self):
        self.grid = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=int)
        self.buildings = []
        self.build_map()

    def build_map(self):
        macro_data = [
            ["障碍", "教学楼一", "障碍", "障碍", "操场"],
            ["障碍", "障碍", "教学楼二", "障碍", "障碍"],
            ["宿舍3", "食堂3", "障碍", "食堂1", "障碍"],
            ["障碍", "障碍", "障碍", "障碍", "障碍"],
            ["宿舍2", "障碍", "食堂2", "障碍", "宿舍1"]
        ]
        cx = MAP_WIDTH // 2
        cy = MAP_HEIGHT // 2

        for r in range(MACRO_ROWS):
            for c in range(MACRO_COLS):
                cell_value = macro_data[r][c]
                if cell_value == "":
                    continue
                display_name = "某楼" if cell_value == "障碍" else cell_value

                if "障碍" in cell_value:
                    b_type = OBSTACLE
                elif "教学楼" in cell_value:
                    b_type = CLASSROOM
                elif "宿舍" in cell_value:
                    b_type = DORM
                elif "食堂" in cell_value:
                    b_type = CANTEEN
                elif "操场" in cell_value:
                    b_type = PLAYGROUND
                else:
                    b_type = ROAD

                start_row = r * CELL_SIZE + ROAD_WIDTH
                start_col = c * CELL_SIZE + ROAD_WIDTH
                self.grid[start_row: start_row + BUILDING_SIZE, start_col: start_col + BUILDING_SIZE] = b_type
                b = Building(display_name, b_type, start_col, start_row, BUILDING_SIZE, BUILDING_SIZE)

                if display_name == "教学楼二":
                    roads_bottom, roads_right = [], []
                    for x in range(start_col, start_col + BUILDING_SIZE):
                        if 0 <= start_row + BUILDING_SIZE < MAP_HEIGHT and self.grid[
                            start_row + BUILDING_SIZE, x] == ROAD:
                            roads_bottom.append((x, start_row + BUILDING_SIZE))
                    for y in range(start_row, start_row + BUILDING_SIZE):
                        if 0 <= start_col + BUILDING_SIZE < MAP_WIDTH and self.grid[
                            y, start_col + BUILDING_SIZE] == ROAD:
                            roads_right.append((start_col + BUILDING_SIZE, y))

                    def add_mid_4(road_list):
                        if len(road_list) >= 4:
                            b.door_positions.extend(road_list[len(road_list) // 2 - 2: len(road_list) // 2 + 2])
                        elif road_list:
                            b.door_positions.extend(road_list)

                    add_mid_4(roads_bottom)
                    add_mid_4(roads_right)

                elif display_name == "教学楼一":
                    roads_bottom = []
                    for x in range(start_col, start_col + BUILDING_SIZE):
                        if 0 <= start_row + BUILDING_SIZE < MAP_HEIGHT and self.grid[
                            start_row + BUILDING_SIZE, x] == ROAD:
                            roads_bottom.append((x, start_row + BUILDING_SIZE))
                    if len(roads_bottom) >= 6:
                        b.door_positions.extend(roads_bottom[len(roads_bottom) // 2 - 3: len(roads_bottom) // 2 + 3])
                    elif roads_bottom:
                        b.door_positions.extend(roads_bottom)

                elif display_name == "宿舍2":
                    roads_right = []
                    road_x = start_col + BUILDING_SIZE
                    for y in range(start_row, min(start_row + 5, start_row + BUILDING_SIZE)):
                        if 0 <= road_x < MAP_WIDTH and 0 <= y < MAP_HEIGHT and self.grid[y, road_x] == ROAD:
                            roads_right.append((road_x, y))
                    b.door_positions.extend(roads_right)

                elif display_name == "宿舍1":
                    roads_left = []
                    road_x = start_col - 1
                    for y in range(start_row, min(start_row + 5, start_row + BUILDING_SIZE)):
                        if 0 <= road_x < MAP_WIDTH and 0 <= y < MAP_HEIGHT and self.grid[y, road_x] == ROAD:
                            roads_left.append((road_x, y))
                    b.door_positions.extend(roads_left)

                else:
                    roads = []
                    for x in range(start_col - 1, start_col + BUILDING_SIZE + 1):
                        if 0 <= start_row - 1 < MAP_HEIGHT and 0 <= x < MAP_WIDTH and self.grid[
                            start_row - 1, x] == ROAD:
                            roads.append((x, start_row - 1))
                        if 0 <= start_row + BUILDING_SIZE < MAP_HEIGHT and 0 <= x < MAP_WIDTH and self.grid[
                            start_row + BUILDING_SIZE, x] == ROAD:
                            roads.append((x, start_row + BUILDING_SIZE))
                    for y in range(start_row, start_row + BUILDING_SIZE):
                        if 0 <= start_col - 1 < MAP_WIDTH and 0 <= y < MAP_HEIGHT and self.grid[
                            y, start_col - 1] == ROAD:
                            roads.append((start_col - 1, y))
                        if 0 <= start_col + BUILDING_SIZE < MAP_WIDTH and 0 <= y < MAP_HEIGHT and self.grid[
                            y, start_col + BUILDING_SIZE] == ROAD:
                            roads.append((start_col + BUILDING_SIZE, y))
                    roads.sort(key=lambda pos: (pos[0] - cx) ** 2 + (pos[1] - cy) ** 2)
                    if roads:
                        b.door_positions = roads[:5]
                self.buildings.append(b)

    def generate_gravity_field(self, target_building):
        dist = np.full((MAP_HEIGHT, MAP_WIDTH), np.inf)
        queue = deque()
        if target_building.door_positions:
            for dx, dy in target_building.door_positions:
                dist[dy, dx] = 0
                queue.append((dy, dx))
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while queue:
            r, c = queue.popleft()
            current_d = dist[r, c]
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < MAP_HEIGHT and 0 <= nc < MAP_WIDTH and self.grid[nr, nc] == ROAD and dist[
                    nr, nc] == np.inf:
                    dist[nr, nc] = current_d + 1
                    queue.append((nr, nc))
        return dist

class Simulation:
    def __init__(self, num_per_dorm=200, num_bikes=300):
        print("1. 初始化地图与实体系统...")
        self.campus = Map()
        self.occupancy_grid = np.zeros_like(self.campus.grid)
        self.students = []
        self.dynamic_texts = []
        self.bike_grid = np.zeros_like(self.campus.grid)

        self.valid_buildings = [b for b in self.campus.buildings if
                                b.b_type not in [ROAD, OBSTACLE] and b.name != "某楼"]

        self.nn_ready = False
        self.init_neural_network()

        self.b_zones = {}
        for b in self.campus.buildings:
            zone = []
            for y in range(max(0, b.y - 8), min(MAP_HEIGHT, b.y + b.h + 8)):
                for x in range(max(0, b.x - 8), min(MAP_WIDTH, b.x + b.w + 8)):
                    if b.door_positions:
                        min_dist = min(abs(x - dx) + abs(y - dy) for dx, dy in b.door_positions)
                        if min_dist <= 5:
                            zone.append((x, y))
            self.b_zones[b] = zone

        self.shuttles = [Shuttle() for _ in range(3)]

        self.dispatched_0600 = False
        self.dispatched_1030 = False
        self.dispatched_1600 = False
        self.dispatched_2000 = False

        self.day = 1

        self.pending_dispatch_plan = None
        self.llm_thinking = False
        self.last_dispatch_log = "等待大模型预测调度..."

        self.history_days = []
        self.history_all = []
        self.history_d1 = []
        self.history_d2 = []
        self.history_d3 = []

        print("2. 正在预计算引力场...")
        self.building_gravities = {b: self.campus.generate_gravity_field(b) for b in self.campus.buildings if
                                   b.b_type not in [ROAD, OBSTACLE]}

        print("3. 分配学生与投放单车...")
        self.init_students(num_per_dorm)
        self.init_bikes(num_bikes)

        self.time_seconds = 6 * 3600
        self.current_schedule_idx = 0
        self.current_event_name = "宿舍休息"

        self.generate_daily_schedule()

    def get_features(self, t_seconds, current_day):
        slot = int((t_seconds / 1800.0) % 48)
        time_onehot = [0] * 48
        time_onehot[slot] = 1

        day_of_week = (current_day - 1) % 7
        day_onehot = [0] * 7
        day_onehot[day_of_week] = 1

        is_weekend = 1 if day_of_week >= 5 else 0

        hour_float = (t_seconds / 3600.0) % 24.0
        has_event = 1 if (day_of_week in [1, 3] and 14.0 <= hour_float < 16.0) else 0
        in_class = 1 if (not is_weekend) and (
                (8.0 <= hour_float < 11.5) or (13.5 <= hour_float < 17.0) or (18.0 <= hour_float < 20.0)) else 0

        return time_onehot + day_onehot + [has_event, in_class]

    def init_neural_network(self):
        print("[NN 模型] 正在加载多层感知机 (MLP) 神经网络...")
        self.nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        try:
            df = pd.read_csv('historical_demand.csv', encoding='utf-8-sig')
            if len(df.columns) < 57:
                raise ValueError("Old feature format detected.")

            X = df.iloc[:, :57].values
            y = df.iloc[:, 57:].values

            # ==========================================
            # 📊 新增：模型准确度评估模块
            # ==========================================
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error, r2_score

            # 抽出 20% 的历史数据作为“期末考试”来测试准确度
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 先用 80% 的数据训练模型
            self.nn_model.fit(X_train, y_train)

            # 让模型做“期末考试”
            y_pred = self.nn_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print("-" * 50)
            print(f"📊 【神经网络预测性能评估报告】")
            print(f"   👉 数据总量: {len(X)} 条时空切片")
            print(f"   👉 R² (拟合优度) : {r2:.4f}  (注: 越接近1.0说明模型越能准确捕捉潮汐规律)")
            print(f"   👉 MAE (平均绝对误差) : {mae:.2f} 辆  (注: 预测缺口和真实缺口平均相差 {mae:.2f} 辆车)")
            print("-" * 50)

            # 考试评估完后，为了让沙盘在实际调度中达到最强性能，我们把 100% 全量数据喂给它重训一次
            self.nn_model.fit(X, y)
            print("[NN 模型] 全量数据终极拟合完毕，大模型调度大脑已接入！")

            self.nn_ready = True
        except Exception as e:
            print("⚠️ 未检测到有效的高精度 (48半小时槽) 历史数据。")
            print("💡 [强制拦截] 彻底废除造假冷启动数据！系统将以无调度采集模式运行。")
            self.nn_ready = False
    def generate_daily_schedule(self):
        day_of_week = (self.day - 1) % 7
        is_weekend = day_of_week >= 5

        daily_schedule = []
        if not is_weekend:
            daily_schedule.extend([
                (7 * 3600, "前往食堂吃早餐", CANTEEN, 0.8, False),
                (8 * 3600, "上午教学楼上课", CLASSROOM, 0.66, False),
                (11 * 3600 + 1800, "前往食堂吃午餐", CANTEEN, 0.8, False),
                (13 * 3600 + 1800, "下午教学楼上课", CLASSROOM, 0.66, False),
                (17 * 3600 + 300, "前往食堂吃晚餐", CANTEEN, 0.8, False),
                (18 * 3600 + 0, "晚课", CLASSROOM, 0.66, False),
                (20 * 3600 + 300, "晚课下课(前往操场)", "MIXED", 1.0, False),
                (21 * 3600 + 0, "操场人群陆续回寝", DORM, 1.0, False),
                (22 * 3600 + 1800, "熄灯就寝", DORM, 1.0, False)
            ])
        else:
            daily_schedule.extend([
                (8 * 3600 + 1800, "周末早餐", CANTEEN, 0.4, False),
                (12 * 3600, "周末午餐", CANTEEN, 0.8, False),
                (15 * 3600, "下午操场/自由活动", "MIXED", 1.0, False),
                (18 * 3600, "周末晚餐", CANTEEN, 0.8, False),
                (20 * 3600 + 300, "晚上操场/自由活动", "MIXED", 1.0, False),
                (21 * 3600 + 0, "操场人群陆续回寝", DORM, 1.0, False),
                (22 * 3600 + 1800, "熄灯就寝", DORM, 1.0, False)
            ])

        if day_of_week in [1, 3]:
            daily_schedule.append((14 * 3600, "卓越星活动开始", CLASSROOM, 0.5, True))
            daily_schedule.append((16 * 3600, "卓越星活动结束", "RETURN", 1.0, True))
            self.daily_event_text = "今日 14:00-16:00 教学楼举办【卓越星特别活动】"
        else:
            self.daily_event_text = "今日无特殊活动"

        daily_schedule.sort(key=lambda x: x[0])
        self.schedule = daily_schedule

    def init_students(self, num_per_dorm):
        dorms = [b for b in self.campus.buildings if b.b_type == DORM]
        sid = 1
        for d in dorms:
            for _ in range(num_per_dorm):
                s = Student(sid)
                s.home_dorm = d
                s.target_building = d
                s.primary_target = d
                self.students.append(s)
                d.inside_students.append(s)
                sid += 1

    def init_bikes(self, num_bikes):
        dorms = [b for b in self.campus.buildings if b.b_type == DORM]
        other_buildings = [b for b in self.campus.buildings if b.b_type in (CLASSROOM, CANTEEN)]
        bikes_to_place = num_bikes

        for d in dorms:
            nearby_roads = []
            for dx, dy in d.door_positions:
                for oy in range(-1, 2):
                    for ox in range(-1, 2):
                        if 0 <= dx + ox < MAP_WIDTH and 0 <= dy + oy < MAP_HEIGHT and self.campus.grid[
                            dy + oy, dx + ox] == ROAD:
                            nearby_roads.append((dx + ox, dy + oy))
            nearby_roads = list(set(nearby_roads))

            for _ in range(75):
                if bikes_to_place <= 0 or not nearby_roads:
                    break
                rx, ry = random.choice(nearby_roads)
                self.bike_grid[ry, rx] += 1
                bikes_to_place -= 1

        for b in other_buildings:
            nearby_roads = []
            for dx, dy in b.door_positions:
                for oy in range(-1, 2):
                    for ox in range(-1, 2):
                        if 0 <= dx + ox < MAP_WIDTH and 0 <= dy + oy < MAP_HEIGHT and self.campus.grid[
                            dy + oy, dx + ox] == ROAD:
                            nearby_roads.append((dx + ox, dy + oy))
            nearby_roads = list(set(nearby_roads))

            for _ in range(min(15, bikes_to_place)):
                if not nearby_roads:
                    break
                rx, ry = random.choice(nearby_roads)
                self.bike_grid[ry, rx] += 1
                bikes_to_place -= 1

        road_coords = np.argwhere(self.campus.grid == ROAD)
        for _ in range(bikes_to_place):
            y, x = road_coords[random.randint(0, len(road_coords) - 1)]
            self.bike_grid[y, x] += 1

    def get_mood_color(self, mood):
        if mood >= 90:
            return 'blue'
        elif mood >= 70:
            return 'green'
        elif mood >= 50:
            return 'yellow'
        elif mood >= 30:
            return 'red'
        else:
            return 'black'

    def format_time(self, seconds):
        h = (seconds // 3600) % 24
        m = (seconds % 3600) // 60
        return f"{int(h):02d}:{int(m):02d}"

    def is_canteen_locked(self):
        t = self.time_seconds
        return (7 * 3600 <= t < 7 * 3600 + 1800) or (11 * 3600 + 1800 <= t < 12 * 3600 + 900) or (
                17 * 3600 + 300 <= t < 17 * 3600 + 1200)

    def on_key_press(self, event):
        pass

    def calculate_imbalance(self):
        scores = {}
        for b in self.campus.buildings:
            if b.b_type in [ROAD, OBSTACLE] or b.name == "某楼":
                continue

            b_bikes = 0
            for x, y in self.b_zones[b]:
                b_bikes += self.bike_grid[y, x]

            if b.inside_students:
                avg_mood = sum(s.mood for s in b.inside_students) / len(b.inside_students)
            else:
                avg_mood = 100.0

            mood_mult = 1.0 + max(0, (100.0 - avg_mood) / 50.0)
            need_bikes = len(b.inside_students) * mood_mult

            if b.b_type == DORM and self.time_seconds >= 20.5 * 3600:
                need_bikes = 0

            scores[b] = need_bikes - b_bikes
        return scores

    def trigger_llm_dispatch(self):
        idle_shuttles = [sh for sh in self.shuttles if not sh.is_active]
        if not idle_shuttles:
            return
        if not hasattr(self, 'nn_ready') or not self.nn_ready:
            return
        if self.llm_thinking:
            return

        future_features = self.get_features(self.time_seconds + 3600.0, self.day)
        try:
            future_scores = self.nn_model.predict([future_features])[0]
        except Exception:
            return

        current_scores_dict = self.calculate_imbalance()
        snapshot_time = self.time_seconds

        nn_data = ""
        for b, f_score in zip(self.valid_buildings, future_scores):
            cur_score = current_scores_dict[b]
            nn_data += f"[{b.name}]:当前缺口 {cur_score:.1f} 辆，预测1小时后缺口{f_score:.1f} 辆\n"

        # 【修改点 2：仅修改 Prompt，引导模型让 3 辆卡车去不同地方】
        prompt = f"""
你是一个同济大学的高级 AI 调度员。时间是第{self.day}天{self.format_time(snapshot_time)}。

各区域【真实缺口】与【1小时后预测缺口】（正数=缺车需运入，负数=车满为患可抽走）：

{nn_data}

【🎯 调度核心逻辑】：

1. 寻找缺口最大的正数区域作为目标(tgt)。

2. 寻找负数绝对值最大（车最多）的区域作为来源(src)。

3. 遇到以下极端时间必须强制无视数据执行：
   - 06:00 -> 运往【缺口最大的宿舍】
   - 10:30 & 16:00 -> 运往【缺口最大的教学楼】
   - 20:00 -> 运往【操场】

任务：分配不超过3辆卡车，你可以让不同的卡车去不同的建筑运车，以最大化全校调度效率。请只输出合法的 JSON，绝对不要输出 Markdown，格式如下（不要原样照抄示例的建筑名！）：

{{"shuttle_1": {{"src": "车最多的建筑A", "tgt": "最缺车的建筑B"}},
"shuttle_2": {{"src": "车第二多的建筑C", "tgt": "缺车第二多的建筑D"}},
"shuttle_3": {{"src": "车第三多的建筑E", "tgt": "缺车第三多的建筑F"}},
"reasoning": "简要理由"}}
"""
        self.llm_thinking = True
        self.last_dispatch_log = "大模型正在进行沙盘兵棋推演...\n彻底杜绝羊群效应，动态平衡各区车辆！"

        def fetch_api_worker():
            try:
                response = Generation.call(
                    model='qwen-plus',
                    prompt=prompt,
                    result_format='message'
                )
                content = response.output.choices[0].message.content.strip()

                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                plan = json.loads(content)

                current_hour = (snapshot_time / 3600.0) % 24.0

                simulated_scores = current_scores_dict.copy()

                for key in list(plan.keys()):
                    if key.startswith("shuttle_"):
                        tgt = plan[key].get("tgt", "")
                        src = plan[key].get("src", "")
                        plan.setdefault("reasoning", "")

                        src_b = next((b for b in self.valid_buildings if b.name == src), None)
                        if not src_b or simulated_scores.get(src_b, 0) > -5:
                            best_src_b = min(self.valid_buildings, key=lambda b: simulated_scores[b])
                            if simulated_scores[best_src_b] < 0:
                                src = best_src_b.name
                                plan[key]["src"] = src
                                plan["reasoning"] += f" [防榨干：改从淤积最严重的{src}抽车]"
                            else:
                                plan[key]["tgt"] = plan[key]["src"]
                                plan["reasoning"] += " [取消：全校车辆均吃紧，停止抽车]"
                                continue

                        if 19.5 <= current_hour <= 20.5:
                            tgt = "操场"
                            plan[key]["tgt"] = tgt
                        elif 5.5 <= current_hour <= 6.5:
                            dorm_buildings = [b for b in self.valid_buildings if b.b_type == DORM]
                            worst_dorm_b = max(dorm_buildings, key=lambda b: simulated_scores[b])
                            tgt = worst_dorm_b.name
                            plan[key]["tgt"] = tgt
                        elif (10.0 <= current_hour <= 11.0) or (15.5 <= current_hour <= 16.5):
                            class_buildings = [b for b in self.valid_buildings if b.b_type == CLASSROOM]
                            worst_class_b = max(class_buildings, key=lambda b: simulated_scores[b])
                            tgt = worst_class_b.name
                            plan[key]["tgt"] = tgt
                        else:
                            tgt_b = next((b for b in self.valid_buildings if b.name == tgt), None)
                            if not tgt_b or simulated_scores.get(tgt_b, 0) <= 0:
                                best_tgt_b = max(self.valid_buildings, key=lambda b: simulated_scores[b])
                                if simulated_scores[best_tgt_b] > 0:
                                    tgt = best_tgt_b.name
                                    plan[key]["tgt"] = tgt
                                    plan["reasoning"] += f" [防旱死：重定向至真缺车的{tgt}]"
                                else:
                                    plan[key]["tgt"] = plan[key]["src"]
                                    plan["reasoning"] += " [取消：全图均不缺车]"
                                    continue

                        final_src_b = next(b for b in self.valid_buildings if b.name == src)
                        final_tgt_b = next(b for b in self.valid_buildings if b.name == tgt)

                        simulated_scores[final_src_b] += 30
                        simulated_scores[final_tgt_b] -= 30

                self.pending_dispatch_plan = plan
            except Exception as e:
                self.pending_dispatch_plan = {"error": str(e)}

        threading.Thread(target=fetch_api_worker, daemon=True).start()

    def run_visual(self):
        plt.ion()
        fig = plt.figure(figsize=(16, 9.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

        ax_map = fig.add_axes([0.02, 0.05, 0.68, 0.90])
        ax_text = fig.add_axes([0.65, 0.65, 0.22, 0.30])
        ax_leg = fig.add_axes([0.75, 0.43, 0.22, 0.18])
        ax_line = fig.add_axes([0.75, 0.10, 0.22, 0.28])

        ax_text.axis('off')
        ax_leg.axis('off')
        fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.cmap = ListedColormap(['white', '#EAEAEA', '#FFF066', '#66B2FF', '#FF99CC', '#4D4D4D'])
        vis_grid = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=int)
        vis_grid[self.campus.grid > 0] = 1

        self.map_img = ax_map.imshow(vis_grid, cmap=self.cmap, vmin=0, vmax=5, extent=[0, MAP_WIDTH, MAP_HEIGHT, 0])

        for b in self.campus.buildings:
            ax_map.add_patch(
                patches.Rectangle((b.x, b.y), b.w, b.h, linewidth=1.5, edgecolor='black', facecolor='none'))
            for dx, dy in b.door_positions:
                ax_map.add_patch(patches.Rectangle((dx, dy), 1, 1, color='saddlebrown', zorder=2))
            is_important = (b.name != "某楼")
            count_str = f"\n({len(b.inside_students)}人)" if is_important else ""
            txt = ax_map.text(b.x + b.w / 2, b.y + b.h / 2, f"{b.name}{count_str}", color='black',
                              fontsize=12 if is_important else 9, fontweight='bold' if is_important else 'normal',
                              ha='center', va='center')
            self.dynamic_texts.append((txt, b, is_important))

        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_map.set_xticks(np.arange(0, MAP_WIDTH + 1, 1), minor=True)
        ax_map.set_yticks(np.arange(0, MAP_HEIGHT + 1, 1), minor=True)
        ax_map.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.5)

        ax_map.set_title("同济大学仿真系统 (1小时超前量 + 沙盒推演)", fontsize=18, fontweight='bold', pad=15)

        sc = ax_map.scatter([], [], c=[], s=12, zorder=5, edgecolors='black', linewidths=0.3)
        shuttle_sc = ax_map.scatter([], [], c='gold', marker='*', s=450, zorder=8, edgecolors='black', linewidths=1.5)

        # 【加入 wrap=True 自动换行，并调小一点字号保证能装下】
        self.dashboard_text = ax_text.text(0.0, 1.0, "", fontsize=11, va='top', ha='left', fontfamily='SimHei',
                                           bbox=dict(facecolor='#F8F9FA', edgecolor='black', boxstyle='round,pad=1.0',
                                                     alpha=0.9), wrap=True)

        legend_elements = [
            patches.Patch(facecolor='white', edgecolor='black', label='0 辆 (无车)'),
            patches.Patch(facecolor='#FFF066', edgecolor='black', label='1~4 辆 (黄)'),
            patches.Patch(facecolor='#66B2FF', edgecolor='black', label='5~8 辆 (蓝)'),
            patches.Patch(facecolor='#FF99CC', edgecolor='black', label='9~12 辆 (粉)'),
            patches.Patch(facecolor='#4D4D4D', edgecolor='black', label='>=13 辆 (黑灰淤积)'),
            plt.Line2D([0], [0], marker='*', color='w', label='AI摆渡车(3辆满载30)', markerfacecolor='gold',
                       markersize=16, markeredgecolor='black')
        ]
        ax_leg.legend(handles=legend_elements, loc='center left', title="热力图图例", fontsize=12, title_fontsize=13,
                      frameon=True, facecolor='white', edgecolor='gray')

        ax_line.set_title("每日心情值折线图", fontsize=12)
        ax_line.set_xlabel("天数 (Day)", fontsize=11)
        ax_line.set_ylabel("心情均值", fontsize=11)

        ax_line.set_ylim(40, 90)
        ax_line.set_xlim(0.8, 5.2)
        ax_line.set_xticks([1, 2, 3, 4, 5])
        ax_line.grid(True, linestyle='--', alpha=0.6)

        line_all, = ax_line.plot(self.history_days, self.history_all, color='green', marker='o', lw=2.5,
                                 label='全校平均')
        line_d1, = ax_line.plot(self.history_days, self.history_d1, color='red', marker='.', linestyle='--', lw=1,
                                label='宿舍 1')
        line_d2, = ax_line.plot(self.history_days, self.history_d2, color='goldenrod', marker='.', linestyle='--', lw=1,
                                label='宿舍 2')
        line_d3, = ax_line.plot(self.history_days, self.history_d3, color='blue', marker='.', linestyle='--', lw=1,
                                label='宿舍 3')
        ax_line.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=10, frameon=False)

        while True:
            steps_to_run = 3

            for _ in range(steps_to_run):
                if self.pending_dispatch_plan is not None:
                    plan = self.pending_dispatch_plan
                    self.pending_dispatch_plan = None
                    self.llm_thinking = False

                    if "error" in plan:
                        self.last_dispatch_log = f"API请求拥堵或失败，稍后重试"
                    else:
                        self.last_dispatch_log = f"大模型靶向调度:\n{plan.get('reasoning', '常规调度')}"
                        idle_shuttles = [sh for sh in self.shuttles if not sh.is_active]
                        b_dict = {b.name: b for b in self.valid_buildings}

                        dispatched = 0
                        for i, sh in enumerate(idle_shuttles):
                            shuttle_key = f"shuttle_{i + 1}"
                            if shuttle_key in plan:
                                src_name = plan[shuttle_key].get("src")
                                tgt_name = plan[shuttle_key].get("tgt")
                                if src_name in b_dict and tgt_name in b_dict and src_name != tgt_name:
                                    sh.activate(b_dict[src_name], b_dict[tgt_name], self.bike_grid, self.b_zones)
                                    self.last_dispatch_log += f"\n卡车{i + 1}: {src_name} -> {tgt_name}"
                                    dispatched += 1
                        if dispatched == 0:
                            self.last_dispatch_log += "\n当前已无调度必要，卡车待命。"

                if self.llm_thinking:
                    break

                if self.time_seconds >= 24 * 3600:
                    all_m = [s.mood for s in self.students]
                    d1_m = [s.mood for s in self.students if s.home_dorm.name == "宿舍1"]
                    d2_m = [s.mood for s in self.students if s.home_dorm.name == "宿舍2"]
                    d3_m = [s.mood for s in self.students if s.home_dorm.name == "宿舍3"]

                    self.history_days.append(self.day)
                    self.history_all.append(np.mean(all_m))
                    self.history_d1.append(np.mean(d1_m) if d1_m else 100)
                    self.history_d2.append(np.mean(d2_m) if d2_m else 100)
                    self.history_d3.append(np.mean(d3_m) if d3_m else 100)

                    line_all.set_data(self.history_days, self.history_all)
                    line_d1.set_data(self.history_days, self.history_d1)
                    line_d2.set_data(self.history_days, self.history_d2)
                    line_d3.set_data(self.history_days, self.history_d3)

                    if self.day > 5:
                        ax_line.set_xlim(0.8, self.day + 0.5)
                        ax_line.set_xticks(range(1, self.day + 1))

                    fig.canvas.draw_idle()

                    prompt_txt = ax_map.text(MAP_WIDTH / 2, MAP_HEIGHT / 2,
                                             f"第 {self.day} 天结束！\n自动进入下一天...",
                                             ha='center', va='center', fontsize=20, color='red', fontweight='bold',
                                             bbox=dict(facecolor='white', alpha=0.95, edgecolor='red',
                                                       boxstyle='round,pad=1', linewidth=3), zorder=10)

                    fig.canvas.flush_events()
                    plt.pause(1.0)
                    prompt_txt.remove()

                    self.time_seconds = 6 * 3600
                    self.day += 1
                    self.current_schedule_idx = 0
                    self.current_event_name = "宿舍休息"

                    self.dispatched_0600 = False
                    self.dispatched_1030 = False
                    self.dispatched_1600 = False
                    self.dispatched_2000 = False
                    self.last_dispatch_log = "等待大模型调度指令..."

                    self.generate_daily_schedule()

                    for s in self.students:
                        s.mood = 100.0
                        s.stuck_ticks = 0
                    break

                t = self.time_seconds

                if t >= 6.0 * 3600 and not self.dispatched_0600:
                    self.trigger_llm_dispatch()
                    self.dispatched_0600 = True

                elif t >= 10.5 * 3600 and not self.dispatched_1030:
                    self.trigger_llm_dispatch()
                    self.dispatched_1030 = True

                elif t >= 16.0 * 3600 and not self.dispatched_1600:
                    self.trigger_llm_dispatch()
                    self.dispatched_1600 = True

                elif t >= 20.0 * 3600 and not self.dispatched_2000:
                    self.trigger_llm_dispatch()
                    self.dispatched_2000 = True

                if self.current_schedule_idx < len(self.schedule):
                    trigger_time, event_name, target_type, prob, is_special = self.schedule[self.current_schedule_idx]
                    if self.time_seconds >= trigger_time:
                        self.current_event_name = event_name

                        if not is_special:
                            for s in self.students:
                                if target_type == "MIXED":
                                    if random.random() > 0.5:
                                        s.target_building = random.choice(
                                            [b for b in self.campus.buildings if b.b_type == PLAYGROUND])
                                    else:
                                        s.target_building = s.home_dorm
                                elif target_type == DORM:
                                    s.target_building = s.home_dorm
                                else:
                                    if random.random() < prob:
                                        s.target_building = random.choice(
                                            [b for b in self.campus.buildings if b.b_type == target_type])
                                    else:
                                        s.target_building = s.home_dorm

                                s.primary_target = s.target_building
                                if s.y != -1:
                                    s.update_dynamic_ride_willingness(
                                        self.building_gravities[s.target_building][s.y, s.x])
                                else:
                                    s.update_dynamic_ride_willingness(100)
                        else:
                            if target_type == "RETURN":
                                for s in self.students:
                                    s.target_building = s.primary_target
                                    if s.y != -1:
                                        s.update_dynamic_ride_willingness(
                                            self.building_gravities[s.target_building][s.y, s.x])
                                    else:
                                        s.update_dynamic_ride_willingness(100)
                            else:
                                free_students = [s for s in self.students if s.primary_target == s.home_dorm]
                                participants = random.sample(free_students, int(len(free_students) * prob))
                                target_b = random.choice([b for b in self.campus.buildings if b.b_type == target_type])
                                for s in participants:
                                    s.target_building = target_b
                                    if s.y != -1:
                                        s.update_dynamic_ride_willingness(
                                            self.building_gravities[s.target_building][s.y, s.x])
                                    else:
                                        s.update_dynamic_ride_willingness(100)

                        self.current_schedule_idx += 1

                active_students = [s for s in self.students if s.is_active]

                if len(active_students) > 0 or any(sh.is_active for sh in self.shuttles):
                    delta_time = 10
                else:
                    if self.time_seconds >= 21 * 3600 or self.time_seconds < 6 * 3600:
                        delta_time = 1200
                    else:
                        delta_time = 100

                self.time_seconds += delta_time

                if int(self.time_seconds) % 1800 < delta_time:
                    scores = self.calculate_imbalance()
                    features = self.get_features(self.time_seconds, self.day)
                    row_data = features[:]
                    for b in self.valid_buildings:
                        row_data.append(scores[b])

                    file_exists = os.path.isfile('historical_demand.csv')

                    try:
                        with open('historical_demand.csv', 'a', newline='', encoding='utf-8-sig') as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                header = [f'T{i}' for i in range(48)] + ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat',
                                                                         'Sun',
                                                                         'Has_Event', 'In_Class'] + [b.name for b in
                                                                                                     self.valid_buildings]
                                writer.writerow(header)
                            writer.writerow(row_data)
                    except PermissionError:
                        pass

                canteen_locked = self.is_canteen_locked()

                for b in self.campus.buildings:
                    if not b.inside_students:
                        continue

                    for s in b.inside_students:
                        if s.dwell_timer > 0:
                            s.dwell_timer -= delta_time
                            if s.dwell_timer <= 0:
                                s.dwell_timer = 0
                                classrooms = [bx for bx in self.campus.buildings if bx.b_type == CLASSROOM]
                                s.target_building = random.choice(classrooms)
                                s.primary_target = s.target_building
                                s.update_dynamic_ride_willingness(self.building_gravities[s.target_building][b.y, b.x])

                    if b.b_type == CANTEEN and canteen_locked:
                        continue

                    to_leave = [s for s in b.inside_students if s.target_building != b]
                    for s in list(to_leave):
                        target_g = self.building_gravities[s.target_building]
                        best_doors = sorted(b.door_positions, key=lambda pos: target_g[pos[1], pos[0]])
                        for dx, dy in best_doors:
                            if self.occupancy_grid[dy, dx] < 2:
                                b.inside_students.remove(s)
                                s.x = dx
                                s.y = dy
                                s.is_active = True
                                self.occupancy_grid[s.y, s.x] += 1
                                s.try_grab_bike(self.bike_grid)
                                break

                for sh in self.shuttles:
                    if not sh.is_active:
                        continue
                    target_g = self.building_gravities[sh.target_building]
                    for _ in range(10):
                        if target_g[sh.y, sh.x] <= 2:
                            doors = sh.target_building.door_positions
                            if doors:
                                parking_spots = []
                                for dx, dy in doors:
                                    for oy in range(-1, 2):
                                        for ox in range(-1, 2):
                                            px = dx + ox
                                            py = dy + oy
                                            if 0 <= px < MAP_WIDTH and 0 <= py < MAP_HEIGHT:
                                                if self.campus.grid[py, px] == ROAD and (px, py) not in doors:
                                                    parking_spots.append((px, py))

                                parking_spots = list(set(parking_spots))

                                if parking_spots:
                                    bikes_left = sh.bikes_loaded
                                    while bikes_left > 0:
                                        best_spot = min(parking_spots, key=lambda p: self.bike_grid[p[1], p[0]])
                                        self.bike_grid[best_spot[1], best_spot[0]] += 1
                                        bikes_left -= 1

                            sh.is_active = False
                            sh.bikes_loaded = 0
                            break
                        sh.move(self.campus.grid, target_g)

                random.shuffle(active_students)
                for s in active_students:
                    s_gravity = self.building_gravities[s.target_building]

                    s.try_grab_bike(self.bike_grid)

                    for _ in range(4 if s.has_bike else 2):
                        if s_gravity[s.y, s.x] == 0:
                            s.is_active = False
                            self.occupancy_grid[s.y, s.x] -= 1
                            s.target_building.inside_students.append(s)

                            if s.target_building.b_type == CANTEEN and s.primary_target == s.target_building:
                                s.dwell_timer = 900
                            else:
                                s.dwell_timer = 0

                            if s.has_bike:
                                s.has_bike = False
                                parked = False
                                for _ in range(10):
                                    py = s.y + random.randint(-2, 2)
                                    px = s.x + random.randint(-2, 2)
                                    if 0 <= px < MAP_WIDTH and 0 <= py < MAP_HEIGHT and self.campus.grid[
                                        py, px] == ROAD:
                                        self.bike_grid[py, px] += 1
                                        parked = True
                                        break
                                if not parked:
                                    self.bike_grid[s.y, s.x] += 1
                            break

                        s.move(self.campus.grid, s_gravity, self.occupancy_grid, self.bike_grid)
                        s.update_dynamic_ride_willingness(s_gravity[s.y, s.x])
                        s.try_grab_bike(self.bike_grid)

            vis_grid = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=int)
            vis_grid[self.campus.grid > 0] = 1
            mask_road = (self.campus.grid == ROAD)
            b_cnt = self.bike_grid

            vis_grid[mask_road & (b_cnt >= 1) & (b_cnt <= 4)] = 2
            vis_grid[mask_road & (b_cnt >= 5) & (b_cnt <= 8)] = 3
            vis_grid[mask_road & (b_cnt >= 9) & (b_cnt <= 12)] = 4
            vis_grid[mask_road & (b_cnt >= 13)] = 5

            self.map_img.set_data(vis_grid)

            cur_act = [s for s in self.students if s.is_active]
            if cur_act:
                sc.set_offsets(np.c_[[s.x + 0.5 for s in cur_act], [s.y + 0.5 for s in cur_act]])
                sc.set_color([self.get_mood_color(s.mood) for s in cur_act])
            else:
                sc.set_offsets(np.empty((0, 2)))

            cur_shuttles = [sh for sh in self.shuttles if sh.is_active]
            if cur_shuttles:
                shuttle_sc.set_offsets(np.c_[[sh.x + 0.5 for sh in cur_shuttles], [sh.y + 0.5 for sh in cur_shuttles]])
            else:
                shuttle_sc.set_offsets(np.empty((0, 2)))

            for txt, b, is_imp in self.dynamic_texts:
                txt.set_text(f"{b.name}\n({len(b.inside_students)}人)" if is_imp else "")

            d1_m = [s.mood for s in self.students if s.home_dorm.name == "宿舍1"]
            d2_m = [s.mood for s in self.students if s.home_dorm.name == "宿舍2"]
            d3_m = [s.mood for s in self.students if s.home_dorm.name == "宿舍3"]

            day_type_str = "周末" if ((self.day - 1) % 7 + 1) >= 6 else "工作日"

            dash_str = (f"【 调 度 沙 盘 第 {self.day} 天 ({day_type_str}) 】\n\n"
                        f"系统时间: {self.format_time(self.time_seconds)}\n"
                        f"当前指令: {self.current_event_name}\n"
                        f"------------------\n"
                        f"今日活动预告: {self.daily_event_text}\n"
                        f"------------------\n"
                        f"心情均值 (满分100):\n"
                        f"全校总均 : {np.mean([s.mood for s in self.students]):.1f} 分\n"
                        f"宿舍一均 : {np.mean(d1_m) if d1_m else 100.0:.1f} 分\n"
                        f"宿舍二均 : {np.mean(d2_m) if d2_m else 100.0:.1f} 分\n"
                        f"宿舍三均 : {np.mean(d3_m) if d3_m else 100.0:.1f} 分\n"
                        f"------------------\n"
                        f"{self.last_dispatch_log}")
            self.dashboard_text.set_text(dash_str)

            fig.canvas.draw_idle()
            plt.pause(0.01)


if __name__ == "__main__":
    sim = Simulation(num_per_dorm=200, num_bikes=300)
    sim.run_visual()
