import json
import time


def get_matrix(name, rows, cols):
    print(f"\nEnter {name} ({rows}x{cols}, separate values with spaces):")
    return [list(map(float, input().split())) for _ in range(rows)]


def load_filters(path="data.json"):
    """data.json의 filters를 {size: {filter_name: matrix}} 형태로 반환"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filters_by_size = {}
    for key, filters in data["filters"].items():
        size = int(key.split("_")[1])  # "size_5" -> 5
        for name, matrix in filters.items():
            if len(matrix) != size or any(len(row) != size for row in matrix):
                raise ValueError(f"{key}/{name}: expected {size}x{size} matrix")
        filters_by_size[size] = filters
    return filters_by_size


def calc_mac(filter_matrix, pattern):
    mac_value = 0
    for i in range(len(filter_matrix)):
        for j in range(len(filter_matrix[0])):
            mac_value += filter_matrix[i][j] * pattern[i][j]
    return mac_value


EPSILON = 1e-9
LABEL_MAP = {
    "+": "Cross", "cross": "Cross", "plus": "Cross",
    "x": "X", "×": "X", "cross_x": "X", "diagonal": "X",
}


def normalize_label(label):
    """'+', 'cross', 'x' 등 다양한 표기를 'Cross' / 'X'로 정규화"""
    key = str(label).strip().lower()
    if key not in LABEL_MAP:
        raise ValueError(f"unknown label: {label!r}")
    return LABEL_MAP[key]


def judge(cross_score, x_score, eps=EPSILON):
    """두 점수 차가 eps 이하이면 동점(UNDECIDED)"""
    if abs(cross_score - x_score) <= eps:
        return "UNDECIDED"
    return "Cross" if cross_score > x_score else "X"

def make_cross(size):
    """size×size 십자 행렬 (데이터에 없는 크기의 시간 측정용)"""
    c = size // 2
    return [[1.0 if i == c or j == c else 0.0 for j in range(size)] for i in range(size)]


def analyze_time(path="data.json", repeat=10, extra_sizes=(3,)):
    """크기별 MAC 1회 평균 시간(ms)과 연산 횟수(size²)를 표로 출력"""
    filters_by_size = load_filters(path)
    sizes = sorted(set(extra_sizes) | set(filters_by_size))

    print("#" + "-" * 40)
    print(f"# [3] 성능 분석 (평균/{repeat}회)")
    print("#" + "-" * 40)
    print(f"{'크기':<8}{'평균 시간(ms)':>14}{'연산 횟수':>10}")
    print("-" * 36)

    for size in sizes:
        if size in filters_by_size:
            filter_matrix = next(iter(filters_by_size[size].values()))
        else:
            filter_matrix = make_cross(size)
        pattern = make_cross(size)

        calc_mac(filter_matrix, pattern)  # 워밍업 (첫 호출 오버헤드 제외)
        elapsed = 0.0
        for _ in range(repeat):
            start = time.perf_counter()
            calc_mac(filter_matrix, pattern)
            elapsed += time.perf_counter() - start
        avg_ms = elapsed / repeat * 1000

        print(f"{f'{size}×{size}':<10}{avg_ms:>12.3f}{size * size:>12}")
    print()


def print_summary(results):
    """analyze_patterns()의 결과로 통과/실패 요약과 실패 사유를 출력"""
    failed = [r for r in results if r["result"] != "PASS"]

    print("#" + "-" * 40)
    print("# [4] 결과 요약")
    print("#" + "-" * 40)
    print(f"총 테스트: {len(results)}개")
    print(f"통과: {len(results) - len(failed)}개")
    print(f"실패: {len(failed)}개")

    if failed:
        print("\n실패 케이스:")
        for r in failed:
            if r["decision"] == "UNDECIDED":
                reason = "동점(UNDECIDED) 처리 규칙에 따라 FAIL"
            else:
                reason = f"판정 {r['decision']} ≠ expected {r['expected']}"
            print(f"- {r['name']}: {reason}")


def analyze_patterns(path="data.json"):
    """패턴마다 크기에 맞는 필터로 MAC 점수를 계산해 판정·PASS/FAIL을 출력"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    filters_by_size = load_filters(path)

    print("#" + "-" * 40)
    print("# [2] 패턴 분석 (라벨 정규화 적용)")
    print("#" + "-" * 40)

    results = []
    for name, pattern in data["patterns"].items():
        size = int(name.split("_")[1])  # "size_13_1" -> 13
        filters = {normalize_label(k): v for k, v in filters_by_size[size].items()}
        cross_score = calc_mac(filters["Cross"], pattern["input"])
        x_score = calc_mac(filters["X"], pattern["input"])
        decision = judge(cross_score, x_score)
        raw_expected = pattern["expected"]
        expected = normalize_label(raw_expected)

        if decision == expected:
            result = "PASS"
        elif decision == "UNDECIDED":
            result = "FAIL (동점 규칙)"
        else:
            result = "FAIL"
        results.append({"name": name, "decision": decision,
                        "expected": expected, "result": result})

        print(f"- -- {name} ---")
        print(f"Cross 점수: {cross_score}")
        print(f"X 점수: {x_score}")
        print(f"라벨 정규화: expected '{raw_expected}' → {expected}")
        print(f"판정: {decision} | expected: {expected} | {result}")

    print()
    return results


if __name__ == "__main__":
    separator = "-" * 44

    print("=== Mini NPU Simulator ===")
    print("\n[Model Selection]")
    print("1. User Input (3x3)")
    print("2. data.json Analysis")

    input_mode = input("\nSelect the mode (1 or 2): ")

    if input_mode == "1":
        rows, cols = 3, 3

        print(f"\n{separator}")
        print("[1] Filter Input")
        filter_a = get_matrix("Filter A", rows, cols)
        filter_b = get_matrix("Filter B", rows, cols)

        print(f"\nFilter A: {filter_a}")
        print(f"Filter B: {filter_b}")

        print(f"\n{separator}")
        print("[2] Pattern Input")
        pattern = get_matrix("Pattern", rows, cols)

        print(f"\nPattern: {pattern}")

        start_time = time.perf_counter()
        result_a = calc_mac(filter_a, pattern)
        result_b = calc_mac(filter_b, pattern)
        elapsed = time.perf_counter() - start_time

        print(f"\n{separator}")
        print("[3] MAC Results")
        print(f"Filter A: {result_a}")
        print(f"Filter B: {result_b}")

        print(f"\n{separator}")
        print("[4] Execution Time")
        print(f"Total elapsed time:   {elapsed * 1000:.3f} ms")
        print(f"Average time per MAC calculation: {elapsed / 2 * 1000:.3f} ms")

        print(f"\n{separator}")
        print("[5] Final Result")
        if result_a > result_b:
            print("A: Filter A has a higher MAC value.")
        elif result_a < result_b:
            print("B: Filter B has a higher MAC value.")
        elif result_a == result_b:
            print("Undecided: Both filters have the same MAC value.")

    elif input_mode == "2":
        print(f"\n{separator}")
        print("[Filter Load]")

        filters_by_size = load_filters("data.json")
        for size in sorted(filters_by_size):
            names = ", ".join(filters_by_size[size])
            print(f"size {size}x{size}: {names}")

        print()
        results = analyze_patterns("data.json")
        analyze_time("data.json")
        print_summary(results)
