import numpy as np
import numba
import math
import time
import taichi as ti
import matplotlib.pyplot as plt

# Настройка библиотек
plt.style.use('ggplot')
ti.init(arch=ti.cpu)


AMOUNT_OF_CITIES: int = 100
SIZE_OF_MAP: int = 10
ALPH = 0.9999
TN = 100.0
AMOUNT_OF_ITERATIONS = 300000


class Map(object):
    def __init__(self):
        self.cities_coords: np.ndarray = np.random.rand(AMOUNT_OF_CITIES, 2) * SIZE_OF_MAP
        self.distance_between_cities: np.ndarray = None
        self.start_city_index: int = 1
        self.end_city_index: int = 4


class Path(object):
    def __init__(self, map_of_cities: Map):
        self.amount_of_cities_from_start = np.random.randint(1, AMOUNT_OF_CITIES-2, size=1)
        self.all_cities_id = np.arange(0, AMOUNT_OF_CITIES, 1)
        self.path = np.random.choice(self.all_cities_id,
                                     size=self.amount_of_cities_from_start,
                                     replace=False)
        # Удаляем случайные вхождения в точки старта и финала и добавляем их в начало и конец массива пути
        self.path = np.delete(self.path,
                              np.where(self.path == map_of_cities.start_city_index)[0])
        self.path = np.delete(self.path,
                              np.where(self.path == map_of_cities.end_city_index)[0])
        self.path = np.append(self.path, map_of_cities.end_city_index)
        self.path = np.append(map_of_cities.start_city_index, self.path)
        self.amount_of_cities_from_start = self.path.shape[0]
        # После чего у нас неизвестным образом изменилась длинна массива, по этому считываем её
        self.distance_to_travel: float = float(1e10)

    def set_other_path(self, other_path):
        self.amount_of_cities_from_start = other_path.amount_of_cities_from_start
        self.all_cities_id = other_path.all_cities_id
        self.path = other_path.path
        self.amount_of_cities_from_start = other_path.amount_of_cities_from_start
        self.distance_to_travel: float = other_path.distance_to_travel



def visualize_map_and_path(map_of_cities: Map, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.set_xticks(np.arange(0, SIZE_OF_MAP, 0.5))
    ax.set_yticks(np.arange(0, SIZE_OF_MAP, 0.5))

    ax.scatter(map_of_cities.cities_coords[:, 0], map_of_cities.cities_coords[:, 1],
               c="purple", marker='h', s=60, label='Города')

    ax.scatter(map_of_cities.cities_coords[map_of_cities.start_city_index, 0],
               map_of_cities.cities_coords[map_of_cities.start_city_index, 1],
               c="orange", marker='h', s=120, label='Начало поездки')
    ax.scatter(map_of_cities.cities_coords[map_of_cities.end_city_index, 0],
               map_of_cities.cities_coords[map_of_cities.end_city_index, 1],
               c="r", marker='h', s=120, label='Конец поездки')

    coords_of_cities_from_path = map_of_cities.cities_coords[path.path, :]

    ax.plot(coords_of_cities_from_path[:, 0], coords_of_cities_from_path[:, 1], c="g", marker='h', label='Путь авто')

    ax.set_title("Карта городов")
    ax.legend(fontsize="x-small", edgecolor="inherit", facecolor="inherit", labelspacing=1, handleheight=1.5)
    plt.show()


def otjig(map_of_cities: Map, path: Path, t) -> Path:
    distance_potencial: float = 0
    new_path = Path(map_of_cities)
    new_path.set_other_path(path)

    new_city = np.random.randint(AMOUNT_OF_CITIES, size=1)

    if new_city == new_path.path[0] or new_city == new_path.path[-1]:
        return path

    if new_city in new_path.path:
        new_path.path = np.delete(new_path.path, np.where(new_path.path == new_city))
        new_path.amount_of_cities_from_start = new_path.path.shape[0]
    else:
        index_where_to_add_new_city = None

        min_dist_left = 1e10
        min_dist_right = 1e10
        for i in range(1, new_path.amount_of_cities_from_start):
            new_dist_left = map_of_cities.distance_between_cities[new_path.path[i-1], new_city]
            new_dist_right = map_of_cities.distance_between_cities[new_path.path[i], new_city]
            delta_prew = min_dist_left + min_dist_right
            delta_new = new_dist_left + new_dist_right

            if delta_new < delta_prew:
                index_where_to_add_new_city = i
                min_dist_left = new_dist_left
                min_dist_right = new_dist_right

        new_path.path = np.insert(new_path.path, index_where_to_add_new_city, new_city)
        new_path.amount_of_cities_from_start = new_path.path.shape[0]

    for i in range(1, new_path.amount_of_cities_from_start):
        distance_potencial += map_of_cities.distance_between_cities[new_path.path[i-1], new_path.path[i]]

    new_path.distance_to_travel = distance_potencial

    if new_path.distance_to_travel < path.distance_to_travel or \
        math.exp((path.distance_to_travel - new_path.distance_to_travel) / t) > np.random.random(1):

        return new_path
    else:
        return path


def main() -> None:
    @ti.kernel
    def generate_distance_matrix():
        # Расчёт матрицы расстояний между городами
        for i, j in ti.ndrange(AMOUNT_OF_CITIES, AMOUNT_OF_CITIES):
            distance_calculation_result: ti.f64 = ti.sqrt(ti.pow(cities[i, 0] - cities[j, 0], 2) +
                                                          ti.pow(cities[i, 1] - cities[j, 1], 2))
            # Если точки слишком далеки друг от друга, то расстояние будет увеличено
            # Это создано, чтобы лучшым путём не была прямая от старта до финиша
            if (distance_calculation_result >= 0.5):
                distance_calculation_result = ti.pow(distance_calculation_result, 3)
            distance[i, j] = distance_calculation_result

    map_of_cities = Map()
    # Создаём taichi массивы для быстрого расчёта матрицы расстояний
    cities = ti.field(ti.f64, (AMOUNT_OF_CITIES, 2))
    cities.from_numpy(map_of_cities.cities_coords)
    distance = ti.field(ti.f64, shape=(AMOUNT_OF_CITIES, AMOUNT_OF_CITIES))
    # Производим расчёт матрицы расстояний между городами
    generate_distance_matrix()
    # Преобразуем результат в np.ndarray
    distance = distance.to_numpy()
    map_of_cities.distance_between_cities = distance
    # Создаём путь
    path = Path(map_of_cities)

    print(path.path)
    print(path.all_cities_id)
    print(path.distance_to_travel)
    print(path.amount_of_cities_from_start, end="\n\n")
    visualize_map_and_path(map_of_cities, path)

    t = TN
    counter: int = 0
    best_path = Path(map_of_cities)
    best_path.set_other_path(path)
    for i in range(AMOUNT_OF_ITERATIONS):
        path = otjig(map_of_cities, path, t)
        counter += 1
        t *= ALPH

        if best_path.distance_to_travel > path.distance_to_travel:
            best_path.set_other_path(path)

        if counter % 1000 == 0:
            print(path.path)
            print(path.distance_to_travel)
            print(path.amount_of_cities_from_start)

    print(best_path.path)
    print(best_path.all_cities_id)
    print(best_path.distance_to_travel)
    print(best_path.amount_of_cities_from_start, end="\n\n")
    visualize_map_and_path(map_of_cities, best_path)


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print("\n TIME: ", time.perf_counter() - start, ".sec")













