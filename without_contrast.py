import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import random
import math
from collections import deque

import warnings
warnings.filterwarnings("ignore")

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
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.inside_students = []
        self.door_positions = []


class Student:
    def __init__(self, student_id):
        self.id = student_id
        self.x, self.y = -1, -1
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
        if self.has_bike or not self.willing_to_ride: return False
        # 严格 4x4 探测范围
        for oy in range(-2, 2):
            for ox in range(-2, 2):
                nx, ny = self.x + ox, self.y + oy
                if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT and bike_grid[ny, nx] > 0:
                    self.has_bike = True
                    bike_grid[ny, nx] -= 1
                    self.mood = min(100.0, self.mood + 3.0)
                    return True
        return False

    def move(self, map_grid, gravity_field, occupancy_grid, bike_grid):
        if not self.is_active: return
        straight_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        diag_dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        valid_moves = []
        current_g = gravity_field[self.y, self.x]

        ideal_dx, ideal_dy = 0, 0
        if 0 < self.x < map_grid.shape[1] - 1:
            gw, ge = gravity_field[self.y, self.x - 1], gravity_field[self.y, self.x + 1]
            if not math.isinf(gw) and not math.isinf(ge): ideal_dx = gw - ge
        if 0 < self.y < map_grid.shape[0] - 1:
            gn, gs = gravity_field[self.y - 1, self.x], gravity_field[self.y + 1, self.x]
            if not math.isinf(gn) and not math.isinf(gs): ideal_dy = gn - gs

        bike_pull_x, bike_pull_y = 0, 0
        if not self.has_bike and self.willing_to_ride:
            for oy in range(-2, 3):
                for ox in range(-2, 3):
                    by, bx = self.y + oy, self.x + ox
                    if 0 <= bx < map_grid.shape[1] and 0 <= by < map_grid.shape[0] and bike_grid[by, bx] > 0:
                        bike_pull_x += ox; bike_pull_y += oy

        tolerance = 2 if self.stuck_ticks > 2 else 0
        if not self.has_bike and self.willing_to_ride and (bike_pull_x != 0 or bike_pull_y != 0):
            tolerance = max(tolerance, 1)

        max_occupancy = 3 if self.has_bike else 2

        for is_diag, dirs in [(False, straight_dirs), (True, diag_dirs)]:
            for dx, dy in dirs:
                nx, ny = self.x + dx, self.y + dy
                if not (0 <= nx < map_grid.shape[1] and 0 <= ny < map_grid.shape[0]): continue

                if map_grid[ny, nx] != ROAD or occupancy_grid[ny, nx] >= max_occupancy: continue

                g_val = gravity_field[ny, nx]
                if g_val <= current_g + tolerance and not math.isinf(g_val):
                    perceived_g = g_val + (0.1 if is_diag else 0.0)
                    if not self.has_bike and self.willing_to_ride:
                        if bike_grid[ny, nx] > 0:
                            perceived_g -= 5.0
                        else:
                            bp = bike_pull_x * dx + bike_pull_y * dy
                            if bp > 0: perceived_g -= 1.5
                    cp = ideal_dx * dy - ideal_dy * dx
                    if cp > 0:
                        perceived_g -= 0.6
                    elif cp < 0:
                        perceived_g += 0.6
                    if g_val > current_g: perceived_g += 2.0
                    valid_moves.append((nx, ny, perceived_g))

        if not valid_moves:
            self.stuck_ticks += 1
            return

        self.stuck_ticks = 0
        min_g = min(v[2] for v in valid_moves)
        weights = [math.exp(-2.0 * (v[2] - min_g)) for v in valid_moves]
        chosen_move = random.choices(valid_moves, weights=weights, k=1)[0]

        occupancy_grid[self.y, self.x] -= 1
        self.x, self.y = chosen_move[0], chosen_move[1]
        occupancy_grid[self.y, self.x] += 1

        self.mood = max(0.0, self.mood - (0.05 if self.has_bike else 0.15))


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
        cx, cy = MAP_WIDTH // 2, MAP_HEIGHT // 2
        for r in range(MACRO_ROWS):
            for c in range(MACRO_COLS):
                cell_value = macro_data[r][c]
                if cell_value == "": continue
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

                # ==========================================
                # 严格按照要求：宿舍二只在右边保留宽度为5的门
                # ==========================================
                elif display_name == "宿舍2":
                    roads_right = []
                    road_x = start_col + BUILDING_SIZE
                    for y in range(start_row, min(start_row + 5, start_row + BUILDING_SIZE)):
                        if 0 <= road_x < MAP_WIDTH and 0 <= y < MAP_HEIGHT and self.grid[y, road_x] == ROAD:
                            roads_right.append((road_x, y))
                    b.door_positions.extend(roads_right)

                # ==========================================
                # 严格按照要求：宿舍一只在左边保留宽度为5的门
                # ==========================================
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
                    if roads: b.door_positions = roads[:5]
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
                nr, nc = r + dr, c + dc
                if 0 <= nr < MAP_HEIGHT and 0 <= nc < MAP_WIDTH and self.grid[nr, nc] == ROAD and dist[
                    nr, nc] == np.inf:
                    dist[nr, nc] = current_d + 1
                    queue.append((nr, nc))
        return dist


class Simulation:
    def __init__(self, num_per_dorm=200, num_bikes=300):
        print("1. 初始化地图与实体系统（纯净可视化对比组）...")
        self.campus = Map()
        self.occupancy_grid = np.zeros_like(self.campus.grid)
        self.students = []
        self.dynamic_texts = []
        self.bike_grid = np.zeros_like(self.campus.grid)

        self.day = 1
        self.daily_event_text = ""

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

    def generate_daily_schedule(self):
        day_of_week = (self.day - 1) % 7
        is_weekend = day_of_week >= 5

        daily_schedule = []
        if not is_weekend:
            # ==========================================
            # 🛑 同步物理规律：早中晚餐概率调整为 1.0 (100%干饭)
            # 保证与 AI 调度版、采集器完全平行的基线环境
            # ==========================================
            daily_schedule.extend([
                (7 * 3600, "前往食堂吃早餐", CANTEEN, 1.0, False),
                (8 * 3600, "上午教学楼上课", CLASSROOM, 0.66, False),
                (11 * 3600 + 1800, "前往食堂吃午餐", CANTEEN, 1.0, False),
                (13 * 3600 + 1800, "下午教学楼上课", CLASSROOM, 0.66, False),
                (17 * 3600 + 300, "前往食堂吃晚餐", CANTEEN, 1.0, False),
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

        # 卓越星固定在周二(1)、周四(3)触发，用于录屏展示突发情况
        if day_of_week in [1, 3]:
            daily_schedule.append((14 * 3600, "卓越星活动开始", CLASSROOM, 0.5, True))
            daily_schedule.append((16 * 3600, "卓越星活动结束", "RETURN", 1.0, True))
            self.daily_event_text = "⚠️ 今日 14:00 教学楼举办【卓越星固定活动】"
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
                s.home_dorm = s.target_building = s.primary_target = d
                self.students.append(s)
                d.inside_students.append(s)
                sid += 1

    def init_bikes(self, num_bikes):
        dorms = [b for b in self.campus.buildings if b.b_type == DORM]
        other_buildings = [b for b in self.campus.buildings if b.b_type in (CLASSROOM, CANTEEN)]
        bikes_to_place = num_bikes

        for d in dorms:
            # 紧贴门边3x3 (-1到+1) 密集投放
            nearby_roads = [(dx + ox, dy + oy) for dx, dy in d.door_positions for oy in range(-1, 2) for ox in
                            range(-1, 2) if 0 <= dx + ox < MAP_WIDTH and 0 <= dy + oy < MAP_HEIGHT and self.campus.grid[
                                dy + oy, dx + ox] == ROAD]
            nearby_roads = list(set(nearby_roads))
            for _ in range(75):
                if bikes_to_place <= 0 or not nearby_roads: break
                rx, ry = random.choice(nearby_roads)
                self.bike_grid[ry, rx] += 1
                bikes_to_place -= 1

        for b in other_buildings:
            nearby_roads = [(dx + ox, dy + oy) for dx, dy in b.door_positions for oy in range(-1, 2) for ox in
                            range(-1, 2) if 0 <= dx + ox < MAP_WIDTH and 0 <= dy + oy < MAP_HEIGHT and self.campus.grid[
                                dy + oy, dx + ox] == ROAD]
            nearby_roads = list(set(nearby_roads))
            for _ in range(min(15, bikes_to_place)):
                if not nearby_roads: break
                rx, ry = random.choice(nearby_roads)
                self.bike_grid[ry, rx] += 1
                bikes_to_place -= 1

        road_coords = np.argwhere(self.campus.grid == ROAD)
        for _ in range(bikes_to_place):
            y, x = road_coords[random.randint(0, len(road_coords) - 1)]
            self.bike_grid[y, x] += 1

    def get_mood_color(self, mood):
        if mood >= 90: return 'blue'
        elif mood >= 70: return 'green'
        elif mood >= 50: return 'yellow'
        elif mood >= 30: return 'red'
        else: return 'black'

    def format_time(self, seconds):
        h, m = (seconds // 3600) % 24, (seconds % 3600) // 60
        return f"{int(h):02d}:{int(m):02d}"

    def is_canteen_locked(self):
        t = self.time_seconds
        return (7 * 3600 <= t < 7 * 3600 + 1800) or (11 * 3600 + 1800 <= t < 12 * 3600 + 900) or (
                17 * 3600 + 300 <= t < 17 * 3600 + 1200)

    def on_key_press(self, event):
        pass

    def run_visual(self):
        plt.ion()
        fig = plt.figure(figsize=(16, 9.5))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

        ax_map = fig.add_axes([0.02, 0.05, 0.68, 0.90])
        ax_text = fig.add_axes([0.72, 0.68, 0.25, 0.27])
        ax_leg = fig.add_axes([0.72, 0.45, 0.25, 0.18])
        ax_line = fig.add_axes([0.75, 0.13, 0.21, 0.27])

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

        ax_map.set_title("同济大学校园单车调度沙盘 (无调度对照组 Baseline)", fontsize=18, fontweight='bold', pad=15)

        sc = ax_map.scatter([], [], c=[], s=12, zorder=5, edgecolors='black', linewidths=0.3)

        self.dashboard_text = ax_text.text(0.0, 1.0, "", fontsize=14, va='top', ha='left', fontfamily='SimHei',
                                           bbox=dict(facecolor='#F8F9FA', edgecolor='black', boxstyle='round,pad=1.0',
                                                     alpha=0.9))

        legend_elements = [
            patches.Patch(facecolor='white', edgecolor='black', label='0 辆 (无车)'),
            patches.Patch(facecolor='#FFF066', edgecolor='black', label='1~4 辆 (黄)'),
            patches.Patch(facecolor='#66B2FF', edgecolor='black', label='5~8 辆 (蓝)'),
            patches.Patch(facecolor='#FF99CC', edgecolor='black', label='9~12 辆 (粉)'),
            patches.Patch(facecolor='#4D4D4D', edgecolor='black', label='>=13 辆 (黑灰淤积)')
        ]

        ax_leg.legend(handles=legend_elements, loc='center left', title="热力图图例",
                      fontsize=12, title_fontsize=13, frameon=True, facecolor='white', edgecolor='gray')

        ax_line.set_title("每日心情值折线图 (观察淤积惩罚)", fontsize=12)
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
            # 强制开启 3 倍速
            steps_to_run = 3

            for _ in range(steps_to_run):
                if self.time_seconds >= 24 * 3600:
                    all_m = [s.mood for s in self.students]
                    d1_m = [s.mood for s in self.students if s.home_dorm.name == "宿舍1"]
                    d2_m = [s.mood for s in self.students if s.home_dorm.name == "宿舍2"]
                    d3_m = [s.mood for s in self.students if s.home_dorm.name == "宿舍3"]

                    self.history_days.append(self.day)
                    self.history_all.append(np.mean(all_m))
                    self.history_d1.append(np.mean(d1_m))
                    self.history_d2.append(np.mean(d2_m))
                    self.history_d3.append(np.mean(d3_m))

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

                    self.generate_daily_schedule()

                    for s in self.students:
                        s.mood = 100.0
                        s.stuck_ticks = 0
                    break

                t = self.time_seconds

                if self.current_schedule_idx < len(self.schedule):
                    trigger_time, event_name, target_type, prob, is_special = self.schedule[self.current_schedule_idx]
                    if self.time_seconds >= trigger_time:
                        self.current_event_name = event_name

                        if not is_special:
                            for s in self.students:
                                if target_type == "MIXED":
                                    is_playing = random.random() > 0.5
                                    s.target_building = random.choice([b for b in self.campus.buildings if
                                                                       b.b_type == PLAYGROUND]) if is_playing else s.home_dorm
                                elif target_type == DORM:
                                    s.target_building = s.home_dorm
                                else:
                                    if random.random() < prob:
                                        s.target_building = random.choice(
                                            [b for b in self.campus.buildings if b.b_type == target_type])
                                    else:
                                        s.target_building = s.home_dorm

                                s.primary_target = s.target_building
                                s.update_dynamic_ride_willingness(
                                    self.building_gravities[s.target_building][s.y, s.x] if s.y != -1 else 100)
                        else:
                            if target_type == "RETURN":
                                for s in self.students:
                                    s.target_building = s.primary_target
                                    s.update_dynamic_ride_willingness(
                                        self.building_gravities[s.target_building][s.y, s.x] if s.y != -1 else 100)
                            else:
                                free_students = [s for s in self.students if s.primary_target == s.home_dorm]
                                num_to_pick = int(len(free_students) * prob)
                                participants = random.sample(free_students, num_to_pick)

                                target_b = random.choice([b for b in self.campus.buildings if b.b_type == target_type])
                                for s in participants:
                                    s.target_building = target_b
                                    s.update_dynamic_ride_willingness(
                                        self.building_gravities[s.target_building][s.y, s.x] if s.y != -1 else 100)

                        self.current_schedule_idx += 1

                active_students = [s for s in self.students if s.is_active]

                if len(active_students) > 0:
                    delta_time = 10
                else:
                    delta_time = 1200 if self.time_seconds >= 21 * 3600 or self.time_seconds < 6 * 3600 else 100

                self.time_seconds += delta_time

                canteen_locked = self.is_canteen_locked()

                for b in self.campus.buildings:
                    if not b.inside_students: continue

                    for s in b.inside_students:
                        if s.dwell_timer > 0:
                            s.dwell_timer -= delta_time
                            if s.dwell_timer <= 0:
                                s.dwell_timer = 0
                                classrooms = [bx for bx in self.campus.buildings if bx.b_type == CLASSROOM]
                                s.target_building = random.choice(classrooms)
                                s.primary_target = s.target_building
                                s.update_dynamic_ride_willingness(self.building_gravities[s.target_building][b.y, b.x])

                    if b.b_type == CANTEEN and canteen_locked: continue

                    to_leave = [s for s in b.inside_students if s.target_building != b]
                    for s in list(to_leave):
                        target_g = self.building_gravities[s.target_building]
                        best_doors = sorted(b.door_positions, key=lambda pos: target_g[pos[1], pos[0]])
                        for dx, dy in best_doors:
                            if self.occupancy_grid[dy, dx] < 2:
                                b.inside_students.remove(s)
                                s.x, s.y = dx, dy
                                s.is_active = True
                                self.occupancy_grid[s.y, s.x] += 1
                                s.try_grab_bike(self.bike_grid)
                                break

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
                                    py, px = s.y + random.randint(-2, 2), s.x + random.randint(-2, 2)
                                    if 0 <= px < MAP_WIDTH and 0 <= py < MAP_HEIGHT and self.campus.grid[
                                        py, px] == ROAD:
                                        self.bike_grid[py, px] += 1
                                        parked = True
                                        break
                                if not parked: self.bike_grid[s.y, s.x] += 1
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

            for txt, b, is_imp in self.dynamic_texts: txt.set_text(
                f"{b.name}\n({len(b.inside_students)}人)" if is_imp else "")

            d1_m = [s.mood for s in self.students if s.home_dorm.name == "宿舍1"]
            d2_m = [s.mood for s in self.students if s.home_dorm.name == "宿舍2"]
            d3_m = [s.mood for s in self.students if s.home_dorm.name == "宿舍3"]

            day_type_str = "周末" if ((self.day - 1) % 7 + 1) >= 6 else "工作日"

            dash_str = (f"【 数据采集沙盘 第 {self.day} 天 ({day_type_str}) 】\n\n"
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
                        f"[对照组纯享版 - 无干预]")
            self.dashboard_text.set_text(dash_str)

            fig.canvas.draw_idle()
            plt.pause(0.01)


if __name__ == "__main__":
    sim = Simulation(num_per_dorm=200, num_bikes=300)
    sim.run_visual()