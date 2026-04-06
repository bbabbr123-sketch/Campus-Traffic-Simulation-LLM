import numpy as np
import csv
import os
import random
import math
from collections import deque
import warnings

warnings.filterwarnings("ignore")

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
                        if bike_grid[ny, nx] > 0: perceived_g -= 5.0
                        else:
                            bp = bike_pull_x * dx + bike_pull_y * dy
                            if bp > 0: perceived_g -= 1.5
                    cp = ideal_dx * dy - ideal_dy * dx
                    if cp > 0: perceived_g -= 0.6
                    elif cp < 0: perceived_g += 0.6
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

# --- 3. 地图类 ---
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

                if "障碍" in cell_value: b_type = OBSTACLE
                elif "教学楼" in cell_value: b_type = CLASSROOM
                elif "宿舍" in cell_value: b_type = DORM
                elif "食堂" in cell_value: b_type = CANTEEN
                elif "操场" in cell_value: b_type = PLAYGROUND
                else: b_type = ROAD

                start_row = r * CELL_SIZE + ROAD_WIDTH
                start_col = c * CELL_SIZE + ROAD_WIDTH
                self.grid[start_row: start_row + BUILDING_SIZE, start_col: start_col + BUILDING_SIZE] = b_type
                b = Building(display_name, b_type, start_col, start_row, BUILDING_SIZE, BUILDING_SIZE)

                # 特殊大楼门位置定义 (同步基线)
                if display_name == "教学楼二":
                    roads_bottom, roads_right = [], []
                    for x in range(start_col, start_col + BUILDING_SIZE):
                        if 0 <= start_row + BUILDING_SIZE < MAP_HEIGHT and self.grid[start_row + BUILDING_SIZE, x] == ROAD:
                            roads_bottom.append((x, start_row + BUILDING_SIZE))
                    for y in range(start_row, start_row + BUILDING_SIZE):
                        if 0 <= start_col + BUILDING_SIZE < MAP_WIDTH and self.grid[y, start_col + BUILDING_SIZE] == ROAD:
                            roads_right.append((start_col + BUILDING_SIZE, y))
                    def add_mid_4(road_list):
                        if len(road_list) >= 4:
                            b.door_positions.extend(road_list[len(road_list) // 2 - 2: len(road_list) // 2 + 2])
                        elif road_list: b.door_positions.extend(road_list)
                    add_mid_4(roads_bottom); add_mid_4(roads_right)
                elif display_name == "教学楼一":
                    roads_bottom = []
                    for x in range(start_col, start_col + BUILDING_SIZE):
                        if 0 <= start_row + BUILDING_SIZE < MAP_HEIGHT and self.grid[start_row + BUILDING_SIZE, x] == ROAD:
                            roads_bottom.append((x, start_row + BUILDING_SIZE))
                    if len(roads_bottom) >= 6:
                        b.door_positions.extend(roads_bottom[len(roads_bottom) // 2 - 3: len(roads_bottom) // 2 + 3])
                    elif roads_bottom: b.door_positions.extend(roads_bottom)
                elif display_name == "宿舍2":
                    road_x = start_col + BUILDING_SIZE
                    roads_right = [(road_x, y) for y in range(start_row, start_row + 5) if 0 <= road_x < MAP_WIDTH and self.grid[y, road_x] == ROAD]
                    b.door_positions.extend(roads_right)
                elif display_name == "宿舍1":
                    road_x = start_col - 1
                    roads_left = [(road_x, y) for y in range(start_row, start_row + 5) if 0 <= road_x < MAP_WIDTH and self.grid[y, road_x] == ROAD]
                    b.door_positions.extend(roads_left)
                else:
                    roads = []
                    for x in range(start_col - 1, start_col + BUILDING_SIZE + 1):
                        if 0 <= start_row - 1 < MAP_HEIGHT and self.grid[start_row - 1, x] == ROAD: roads.append((x, start_row - 1))
                        if 0 <= start_row + BUILDING_SIZE < MAP_HEIGHT and self.grid[start_row + BUILDING_SIZE, x] == ROAD: roads.append((x, start_row + BUILDING_SIZE))
                    for y in range(start_row, start_row + BUILDING_SIZE):
                        if 0 <= start_col - 1 < MAP_WIDTH and self.grid[y, start_col - 1] == ROAD: roads.append((start_col - 1, y))
                        if 0 <= start_col + BUILDING_SIZE < MAP_WIDTH and self.grid[y, start_col + BUILDING_SIZE] == ROAD: roads.append((start_col + BUILDING_SIZE, y))
                    roads.sort(key=lambda pos: (pos[0] - cx) ** 2 + (pos[1] - cy) ** 2)
                    if roads: b.door_positions = roads[:5]
                self.buildings.append(b)

    def generate_gravity_field(self, target_building):
        dist = np.full((MAP_HEIGHT, MAP_WIDTH), np.inf)
        queue = deque()
        if target_building.door_positions:
            for dx, dy in target_building.door_positions: dist[dy, dx] = 0; queue.append((dy, dx))
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while queue:
            r, c = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < MAP_HEIGHT and 0 <= nc < MAP_WIDTH and self.grid[nr, nc] == ROAD and dist[nr, nc] == np.inf:
                    dist[nr, nc] = dist[r, c] + 1; queue.append((nr, nc))
        return dist

class DataCollectorSimulation:
    def __init__(self, num_per_dorm=200, num_bikes=300):
        print("🚀 数据采集器：同步 57 维特征规律...")
        if os.path.isfile('historical_demand.csv'): os.remove('historical_demand.csv')

        self.campus = Map()
        self.occupancy_grid = np.zeros_like(self.campus.grid)
        self.students = []
        self.bike_grid = np.zeros_like(self.campus.grid)
        self.valid_buildings = [b for b in self.campus.buildings if b.b_type not in [ROAD, OBSTACLE] and b.name != "某楼"]

        self.b_zones = {}
        for b in self.campus.buildings:
            zone = []
            for y in range(max(0, b.y - 8), min(MAP_HEIGHT, b.y + b.h + 8)):
                for x in range(max(0, b.x - 8), min(MAP_WIDTH, b.x + b.w + 8)):
                    if b.door_positions:
                        min_dist = min(abs(x - dx) + abs(y - dy) for dx, dy in b.door_positions)
                        if min_dist <= 5: zone.append((x, y))
            self.b_zones[b] = zone

        self.day = 1
        self.building_gravities = {b: self.campus.generate_gravity_field(b) for b in self.campus.buildings if b.b_type not in [ROAD, OBSTACLE]}

        self.init_students(num_per_dorm)
        self.init_bikes(num_bikes)

        self.time_seconds = 6 * 3600
        self.current_schedule_idx = 0
        self.generate_daily_schedule()

    def get_features(self, t_seconds, current_day):
        slot = int((t_seconds / 1800.0) % 48)
        time_onehot = [0] * 48; time_onehot[slot] = 1
        day_of_week = (current_day - 1) % 7
        day_onehot = [0] * 7; day_onehot[day_of_week] = 1
        is_weekend = 1 if day_of_week >= 5 else 0
        hour_float = (t_seconds / 3600.0) % 24.0
        has_event = 1 if (day_of_week in [1, 3] and 14.0 <= hour_float < 16.0) else 0
        in_class = 1 if (not is_weekend) and ((8.0 <= hour_float < 11.5) or (13.5 <= hour_float < 17.0) or (18.0 <= hour_float < 20.0)) else 0
        return time_onehot + day_onehot + [has_event, in_class]

    def generate_daily_schedule(self):
        day_of_week = (self.day - 1) % 7
        is_weekend = day_of_week >= 5
        self.schedule = []
        if not is_weekend:
            # 基线概率：早餐 0.8，上课 0.66
            self.schedule.extend([
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
            self.schedule.extend([
                (8 * 3600 + 1800, "周末早餐", CANTEEN, 0.4, False),
                (12 * 3600, "周末午餐", CANTEEN, 0.8, False),
                (15 * 3600, "下午操场/自由活动", "MIXED", 1.0, False),
                (18 * 3600, "周末晚餐", CANTEEN, 0.8, False),
                (20 * 3600 + 300, "晚上操场/自由活动", "MIXED", 1.0, False),
                (21 * 3600 + 0, "操场人群陆续回寝", DORM, 1.0, False),
                (22 * 3600 + 1800, "熄灯就寝", DORM, 1.0, False)
            ])
        if day_of_week in [1, 3]:
            self.schedule.append((14 * 3600, "卓越星活动开始", CLASSROOM, 0.5, True))
            self.schedule.append((16 * 3600, "卓越星活动结束", "RETURN", 1.0, True))
        self.schedule.sort(key=lambda x: x[0])

    def init_students(self, num_per_dorm):
        dorms = [b for b in self.campus.buildings if b.b_type == DORM]
        sid = 1
        for d in dorms:
            for _ in range(num_per_dorm):
                s = Student(sid); s.home_dorm, s.target_building, s.primary_target = d, d, d
                self.students.append(s); d.inside_students.append(s); sid += 1

    def init_bikes(self, num_bikes):
        dorms = [b for b in self.campus.buildings if b.b_type == DORM]
        for d in dorms:
            nearby = list(set([(dx+ox, dy+oy) for dx,dy in d.door_positions for ox in range(-1,2) for oy in range(-1,2) if 0<=dx+ox<MAP_WIDTH and 0<=dy+oy<MAP_HEIGHT and self.campus.grid[dy+oy,dx+ox]==ROAD]))
            for _ in range(75):
                if not nearby: break
                rx, ry = random.choice(nearby); self.bike_grid[ry, rx] += 1
        road_coords = np.argwhere(self.campus.grid == ROAD)
        for _ in range(num_bikes - 75*len(dorms)):
            y, x = road_coords[random.randint(0, len(road_coords) - 1)]
            self.bike_grid[y, x] += 1

    def calculate_imbalance(self):
        scores = {}
        for b in self.campus.buildings:
            if b.b_type in [ROAD, OBSTACLE] or b.name == "某楼": continue
            b_bikes = sum(self.bike_grid[y, x] for x, y in self.b_zones[b])
            avg_mood = sum(s.mood for s in b.inside_students)/len(b.inside_students) if b.inside_students else 100.0
            need_bikes = len(b.inside_students) * (1.0 + max(0, (100.0 - avg_mood) / 50.0))
            # 同步 20:30 静默规则
            if b.b_type == DORM and self.time_seconds >= 20.5 * 3600: need_bikes = 0
            scores[b] = need_bikes - b_bikes
        return scores

    def run_data_collection(self):
        while self.day <= 30:
            if self.time_seconds >= 24 * 3600:
                print(f"🌙 第 {self.day} 天采集完毕！")
                self.time_seconds, self.day = 6 * 3600, self.day + 1
                if self.day > 30: break
                self.current_schedule_idx = 0
                self.generate_daily_schedule()
                for s in self.students: s.mood, s.stuck_ticks = 100.0, 0

            if self.current_schedule_idx < len(self.schedule):
                trigger, _, target_type, prob, is_special = self.schedule[self.current_schedule_idx]
                if self.time_seconds >= trigger:
                    for s in self.students:
                        if not is_special:
                            if target_type == "MIXED": s.target_building = random.choice([b for b in self.campus.buildings if b.b_type == PLAYGROUND]) if random.random() > 0.5 else s.home_dorm
                            elif target_type == DORM: s.target_building = s.home_dorm
                            else: s.target_building = random.choice([b for b in self.campus.buildings if b.b_type == target_type]) if random.random() < prob else s.home_dorm
                        else:
                            if target_type == "RETURN": s.target_building = s.primary_target
                            elif s.primary_target == s.home_dorm and random.random() < prob:
                                s.target_building = random.choice([b for b in self.campus.buildings if b.b_type == CLASSROOM])
                        s.primary_target = s.target_building
                    self.current_schedule_idx += 1

            delta = 60; self.time_seconds += delta
            if int(self.time_seconds) % 1800 < delta:
                row = self.get_features(self.time_seconds, self.day)
                imb = self.calculate_imbalance()
                for b in self.valid_buildings: row.append(imb[b])
                with open('historical_demand.csv', 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    if os.path.getsize('historical_demand.csv') == 0:
                        header = [f'T{i}' for i in range(48)] + ['Mon','Tue','Wed','Thu','Fri','Sat','Sun','Event','Class'] + [b.name for b in self.valid_buildings]
                        writer.writerow(header)
                    writer.writerow(row)

            for b in self.campus.buildings:
                for s in list(b.inside_students):
                    if s.dwell_timer > 0:
                        s.dwell_timer -= delta
                        if s.dwell_timer <= 0:
                            s.target_building = random.choice([bx for bx in self.campus.buildings if bx.b_type == CLASSROOM])
                            s.primary_target = s.target_building
                    if s.target_building != b:
                        best_doors = sorted(b.door_positions, key=lambda p: self.building_gravities[s.target_building][p[1], p[0]])
                        for dx, dy in best_doors:
                            if self.occupancy_grid[dy, dx] < 2:
                                b.inside_students.remove(s); s.x, s.y, s.is_active = dx, dy, True
                                self.occupancy_grid[dy, dx] += 1; s.try_grab_bike(self.bike_grid); break

            active = [s for s in self.students if s.is_active]
            random.shuffle(active)
            for s in active:
                g = self.building_gravities[s.target_building]
                s.try_grab_bike(self.bike_grid)
                for _ in range(4 if s.has_bike else 2):
                    if g[s.y, s.x] == 0:
                        s.is_active = False; self.occupancy_grid[s.y, s.x] -= 1
                        s.target_building.inside_students.append(s)
                        s.dwell_timer = 900 if s.target_building.b_type == CANTEEN and s.primary_target == s.target_building else 0
                        if s.has_bike:
                            s.has_bike = False
                            for _ in range(10):
                                py, px = s.y+random.randint(-2,2), s.x+random.randint(-2,2)
                                if 0<=px<MAP_WIDTH and 0<=py<MAP_HEIGHT and self.campus.grid[py,px]==ROAD:
                                    self.bike_grid[py,px] += 1; break
                        break
                    s.move(self.campus.grid, g, self.occupancy_grid, self.bike_grid)
                    s.update_dynamic_ride_willingness(g[s.y, s.x]); s.try_grab_bike(self.bike_grid)

        print(" 数据采集同步完成！请重新训练并运行 with shuttle.py")

if __name__ == "__main__":
    sim = DataCollectorSimulation(num_per_dorm=200, num_bikes=300)
    sim.run_data_collection()
