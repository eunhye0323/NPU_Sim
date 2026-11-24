"""표준 라이브러리로 구현한 Mini NPU 콘솔 시뮬레이터."""

import json
import math
from pathlib import Path
import re
import statistics
import time


DEFAULT_DATA_PATH = Path(__file__).resolve().with_name("data.json")
EPSILON = 1e-9
LABEL_MAP = {
    "+": "Cross", "cross": "Cross", "plus": "Cross",
    "x": "X", "×": "X", "cross_x": "X", "diagonal": "X",
}


def validate_matrix(matrix, rows, cols, name="행렬"):
    """행/열 수와 유한한 숫자 여부를 MAC 실행 전에 확인한다."""
    if not isinstance(matrix, (list, tuple)) or len(matrix) != rows:
        raise ValueError(f"{name}: {rows}개 행이 필요합니다")
    for row_index, row in enumerate(matrix, start=1):
        if not isinstance(row, (list, tuple)) or len(row) != cols:
            raise ValueError(f"{name}: {row_index}행에 {cols}개 열이 필요합니다")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name}: {row_index}행의 모든 값은 숫자여야 합니다")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{name}: {row_index}행에 처리할 수 없는 숫자가 있습니다")


def get_matrix(name, rows, cols):
    print(f"\n{name} ({rows}줄, 각 줄에 숫자 {cols}개를 공백으로 구분)")
    if rows == cols:
        print("첫 행에 cross 또는 x를 입력하면 패턴을 자동 생성합니다.")
    matrix = []
    while len(matrix) < rows:
        try:
            entered = input(f"{len(matrix) + 1}행> ").strip()
            if not matrix and rows == cols and entered.lower() in ("cross", "x"):
                return make_pattern(rows, entered)
            row = [float(value) for value in entered.split()]
            validate_matrix([row], 1, cols, name)
        except ValueError:
            print(f"입력 형식 오류: 각 줄에 {cols}개의 유한한 숫자를 공백으로 구분해 입력하세요.")
            continue
        matrix.append(row)
    return matrix


def normalize_label(label):
    """'+', 'cross', 'x' 등을 표준 라벨 Cross / X로 정규화한다."""
    key = str(label).strip().lower()
    if key not in LABEL_MAP:
        raise ValueError(f"알 수 없는 라벨 {label!r}")
    return LABEL_MAP[key]


def read_data(path=DEFAULT_DATA_PATH):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("JSON 최상위 값은 객체여야 합니다")
    return data


def load_filters(path=DEFAULT_DATA_PATH, *, data=None, errors=None):
    """필터를 {size: {원래 라벨: 행렬}}로 읽는다.

    errors 딕셔너리를 주면 잘못된 필터를 기록하고 나머지 크기를 계속 읽는다.
    """
    if data is None:
        data = read_data(path)
    raw_filters = data.get("filters")
    if not isinstance(raw_filters, dict):
        message = "filters는 크기별 필터를 담은 객체여야 합니다"
        if errors is None:
            raise ValueError(message)
        errors["filters"] = message
        return {}

    filters_by_size = {}
    for key, filters in raw_filters.items():
        try:
            match = re.fullmatch(r"size_([1-9][0-9]*)", key)
            if match is None:
                raise ValueError("필터 키는 size_N 형식이어야 합니다 (N은 양의 정수)")
            size = int(match.group(1))
            if not isinstance(filters, dict):
                raise ValueError("필터 묶음은 객체여야 합니다")
            labels = set()
            for label, matrix in filters.items():
                normalized = normalize_label(label)
                if normalized in labels:
                    raise ValueError(f"정규화 후 {normalized} 필터가 중복됩니다")
                labels.add(normalized)
                validate_matrix(matrix, size, size, f"{key}/{label}")
            if labels != {"Cross", "X"}:
                raise ValueError("Cross와 X 필터가 모두 필요합니다")
            filters_by_size[size] = filters
        except ValueError as error:
            if errors is None:
                raise ValueError(f"{key}: {error}") from error
            errors[key] = str(error)
    return filters_by_size


def calc_mac(filter_matrix, pattern):
    """검증된 같은 크기의 행렬에서 위치별 곱을 반복문으로 누적한다."""
    mac_value = 0.0
    for i in range(len(filter_matrix)):
        for j in range(len(filter_matrix[0])):
            mac_value += filter_matrix[i][j] * pattern[i][j]
    if not math.isfinite(mac_value):
        raise ValueError("MAC 점수가 유한한 숫자 범위를 벗어났습니다")
    return mac_value


def judge(cross_score, x_score, eps=EPSILON):
    """두 점수 차가 eps 이하이면 동점(UNDECIDED)으로 판정한다."""
    if not math.isfinite(cross_score) or not math.isfinite(x_score):
        raise ValueError("비교할 점수는 유한한 숫자여야 합니다")
    if abs(cross_score - x_score) <= eps:
        return "UNDECIDED"
    return "Cross" if cross_score > x_score else "X"


def make_pattern(size, label):
    """양의 정수 크기의 Cross/X를 생성한다. 짝수 Cross의 중심은 N//2다."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("패턴 크기는 1 이상의 정수여야 합니다")
    label = normalize_label(label)
    center = size // 2
    if label == "X":
        return [
            [1.0 if i == j or i + j == size - 1 else 0.0 for j in range(size)]
            for i in range(size)
        ]
    return [
        [1.0 if i == center or j == center else 0.0 for j in range(size)]
        for i in range(size)
    ]


def make_cross(size):
    """size×size 십자가 행렬을 생성한다."""
    return make_pattern(size, "Cross")


def make_x(size):
    """size×size 두 대각선의 X 행렬을 생성한다."""
    return make_pattern(size, "X")


def matrix_shape(matrix, name="행렬"):
    """비어 있지 않은 직사각형 숫자 행렬을 검증하고 크기를 반환한다."""
    if not isinstance(matrix, (list, tuple)) or not matrix:
        raise ValueError(f"{name}: 비어 있지 않은 2차원 배열이 필요합니다")
    if not isinstance(matrix[0], (list, tuple)) or not matrix[0]:
        raise ValueError(f"{name}: 각 행에는 최소 한 개의 숫자가 필요합니다")
    rows, cols = len(matrix), len(matrix[0])
    validate_matrix(matrix, rows, cols, name)
    return rows, cols


def flatten_matrix(matrix):
    """검증한 2차원 행렬을 행 우선 순서의 1차원 리스트로 변환한다."""
    matrix_shape(matrix)
    return [value for row in matrix for value in row]


def calc_mac_flat(filter_values, pattern_values):
    """검증된 1차원 리스트의 같은 위치 값을 곱하고 순서대로 누적한다."""
    if len(filter_values) != len(pattern_values):
        raise ValueError("1차원 필터와 패턴의 길이가 일치해야 합니다")
    mac_value = 0.0
    for index in range(len(filter_values)):
        mac_value += filter_values[index] * pattern_values[index]
    if not math.isfinite(mac_value):
        raise ValueError("MAC 점수가 유한한 숫자 범위를 벗어났습니다")
    return mac_value


def validate_repeat(repeat):
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 10:
        raise ValueError("성능 측정은 정수 횟수로 최소 10회 반복해야 합니다")


def measure_mac(filter_matrix, pattern, repeat=10):
    """I/O와 입력 검증을 제외한 MAC 함수 호출의 평균 시간을 ms로 반환한다."""
    validate_repeat(repeat)
    calc_mac(filter_matrix, pattern)  # 측정 전 워밍업
    elapsed = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        calc_mac(filter_matrix, pattern)
        elapsed += time.perf_counter() - start
    return elapsed / repeat * 1000


def benchmark_layouts(filter_matrix, pattern, repeat=10):
    """동일 값/순서/반복 횟수로 2D와 1D를 비교하고 변환 비용도 기록한다."""
    validate_repeat(repeat)
    if matrix_shape(filter_matrix, "필터") != matrix_shape(pattern, "패턴"):
        raise ValueError("필터와 패턴의 행/열 크기가 일치해야 합니다")

    # 변환 API가 수행하는 검증을 포함한 일회성 준비 비용을 기록한다.
    start = time.perf_counter()
    flat_filter = flatten_matrix(filter_matrix)
    flat_pattern = flatten_matrix(pattern)
    conversion_ms = (time.perf_counter() - start) * 1000

    score_2d = calc_mac(filter_matrix, pattern)
    score_1d = calc_mac_flat(flat_filter, flat_pattern)
    if score_2d != score_1d:
        raise ValueError("2D와 1D의 MAC 점수가 일치하지 않습니다")

    samples = {"two_d": [], "one_d": []}
    calculations = (
        ("two_d", calc_mac, filter_matrix, pattern),
        ("one_d", calc_mac_flat, flat_filter, flat_pattern),
    )
    for iteration in range(repeat):
        order = calculations if iteration % 2 == 0 else calculations[::-1]
        for layout, calculate, filter_input, pattern_input in order:
            start = time.perf_counter()
            calculate(filter_input, pattern_input)
            elapsed_ms = (time.perf_counter() - start) * 1000
            samples[layout].append(elapsed_ms)

    result = {"repeat": repeat, "score_2d": score_2d, "score_1d": score_1d,
              "conversion_ms": conversion_ms}
    for layout, times in samples.items():
        result[layout] = {
            "mean_ms": statistics.mean(times),
            "stdev_ms": statistics.stdev(times),
            "min_ms": min(times),
            "max_ms": max(times),
        }
    average_1d = result["one_d"]["mean_ms"]
    result["speedup"] = result["two_d"]["mean_ms"] / average_1d if average_1d else None
    result["two_d_total_ms"] = sum(samples["two_d"])
    result["one_d_total_ms"] = conversion_ms + sum(samples["one_d"])
    return result


def print_layout_comparison(result):
    """MAC 함수 호출 시간과 일회성 변환을 포함한 반복 실행 비용을 출력한다."""
    print("구조    평균(ms)   표준편차(ms)      최소(ms)      최대(ms)")
    for label, key in (("2D", "two_d"), ("1D", "one_d")):
        timing = result[key]
        print(f"{label:<4} {timing['mean_ms']:>11.6f} {timing['stdev_ms']:>13.6f}"
              f" {timing['min_ms']:>13.6f} {timing['max_ms']:>13.6f}")
    speedup = result["speedup"]
    ratio = f"{speedup:.3f}배" if speedup is not None else "측정 해상도 미만"
    print(f"2D 평균 / 1D 평균: {ratio} (1 초과이면 1D의 평균 시간이 짧음)")
    print(f"필터·패턴 1회 변환 시간: {result['conversion_ms']:.6f} ms (변환 API의 검증 포함)")
    print(f"{result['repeat']}회 호출 합계: 2D {result['two_d_total_ms']:.6f} ms, "
          f"1D {result['one_d_total_ms']:.6f} ms (1D는 변환 1회 포함)")


def analyze_time(path=DEFAULT_DATA_PATH, repeat=10, extra_sizes=(3, 5, 13, 25),
                 *, filters_by_size=None):
    """크기별 MAC 연산 횟수와 동일 입력의 2D/1D 성능 비교를 출력한다."""
    validate_repeat(repeat)
    if filters_by_size is None:
        filters_by_size = load_filters(path, errors={})
    sizes = sorted(set(extra_sizes) | set(filters_by_size))
    print("\n[3] 성능 분석 (필터 하나의 MAC 함수 호출)")
    print(f"각 크기별 평균/{repeat}회, 워밍업 1회 제외")
    print("측정 순서는 매 반복마다 교대하며 변환 시간은 별도로 측정합니다.")
    measurements = []
    for size in sizes:
        pattern = make_cross(size)
        filters = filters_by_size.get(size, {})
        filter_matrix = next(iter(filters.values()), pattern)
        print(f"\n크기 {size}×{size} / 연산 횟수(N²) {size * size}")
        try:
            comparison = benchmark_layouts(filter_matrix, pattern, repeat)
        except (ValueError, OverflowError) as error:
            measurements.append({"size": size, "average_ms": None,
                                 "operations": size * size, "reason": str(error)})
            print(f"{size}×{size:<8} {'측정 불가':>12} {size * size:>16}")
            print(f"  사유: {error}")
            continue
        measurements.append({"size": size, "average_ms": comparison["two_d"]["mean_ms"],
                             "operations": size * size, **comparison})
        print_layout_comparison(comparison)
    print("\n크기(N×N)     2D 평균 시간(ms)     연산 횟수(N²)")
    for result in measurements:
        size = result["size"]
        average = result["average_ms"]
        average_text = f"{average:.6f}" if average is not None else "측정 불가"
        print(f"{size}×{size:<8} {average_text:>16} {result['operations']:>16}")
    return measurements


def analyze_patterns(path=DEFAULT_DATA_PATH, *, data=None, filters_by_size=None,
                     filter_errors=None):
    """각 패턴을 독립적으로 검증하고 판정 및 실패 사유를 반환한다."""
    if data is None:
        data = read_data(path)
    if filter_errors is None:
        filter_errors = {}
    if filters_by_size is None:
        filters_by_size = load_filters(data=data, errors=filter_errors)
    patterns = data.get("patterns")
    if not isinstance(patterns, dict):
        raise ValueError("patterns는 케이스별 객체여야 합니다")

    print("\n[2] 패턴 분석 (라벨 정규화 적용)")
    results = []
    for name, pattern in patterns.items():
        result = {"name": name, "cross_score": None, "x_score": None,
                  "decision": "UNDECIDED", "expected": None,
                  "result": "FAIL", "reason": "", "error": False}
        try:
            match = re.fullmatch(r"size_([1-9][0-9]*)_([0-9]+)", name)
            if match is None:
                raise ValueError("패턴 키는 size_N_idx 형식이어야 합니다")
            size = int(match.group(1))
            if not isinstance(pattern, dict):
                raise ValueError("패턴 항목은 input과 expected를 담은 객체여야 합니다")
            if "input" not in pattern or "expected" not in pattern:
                raise ValueError("input과 expected가 모두 필요합니다")
            result["expected"] = normalize_label(pattern["expected"])
            validate_matrix(pattern["input"], size, size, name)
            filter_key = f"size_{size}"
            if "filters" in filter_errors:
                raise ValueError(filter_errors["filters"])
            if filter_key in filter_errors:
                raise ValueError(f"{filter_key} 필터 오류: {filter_errors[filter_key]}")
            if size not in filters_by_size:
                raise ValueError(f"{filter_key} 필터가 없습니다")
            filters = {normalize_label(key): value for key, value in filters_by_size[size].items()}
            result["cross_score"] = calc_mac(filters["Cross"], pattern["input"])
            result["x_score"] = calc_mac(filters["X"], pattern["input"])
            result["decision"] = judge(result["cross_score"], result["x_score"])
            if result["decision"] == result["expected"]:
                result["result"] = "PASS"
            elif result["decision"] == "UNDECIDED":
                difference = abs(result["cross_score"] - result["x_score"])
                result["reason"] = f"동점(UNDECIDED), 점수 차 {difference:.3g} <= epsilon {EPSILON:g}"
            else:
                difference = abs(result["cross_score"] - result["x_score"])
                result["reason"] = (f"판정 {result['decision']} ≠ expected {result['expected']}, "
                                    f"점수 차 {difference:.6g}")
        except (ValueError, OverflowError) as error:
            result["error"] = True
            result["reason"] = f"데이터/연산 오류: {error}"
        results.append(result)

        print(f"\n--- {name} ---")
        for label, key in (("Cross", "cross_score"), ("X", "x_score")):
            score = result[key]
            print(f"{label} 점수: {score if score is not None else '계산 불가'}")
        expected = result["expected"] or "정규화 불가"
        print(f"판정: {result['decision']} | expected: {expected} | {result['result']}")
        if result["reason"]:
            print(f"사유: {result['reason']}")
    return results


def print_summary(results):
    """케이스별 결과와 같은 기준으로 전체 통과/실패 수를 집계한다."""
    failed = [result for result in results if result["result"] != "PASS"]
    print("\n[4] 결과 요약")
    print(f"총 테스트: {len(results)}개")
    print(f"통과: {len(results) - len(failed)}개")
    print(f"실패: {len(failed)}개")
    errors = sum(result.get("error", False) for result in results)
    ties = sum(result["decision"] == "UNDECIDED" and not result.get("error", False)
               and result.get("cross_score") is not None and result.get("x_score") is not None
               for result in results)
    print(f"동점(UNDECIDED): {ties}개")
    print(f"데이터/연산 오류: {errors}개")
    if failed:
        print("\n실패 케이스:")
        for result in failed:
            reason = result.get("reason")
            if not reason:
                reason = f"판정 {result['decision']} ≠ expected {result['expected']}"
            print(f"- {result['name']}: {reason}")


def run_user_mode():
    print("\n[1] 필터 입력")
    filter_a = get_matrix("필터 A", 3, 3)
    filter_b = get_matrix("필터 B", 3, 3)
    print(f"\n저장된 필터 A: {filter_a}")
    print(f"저장된 필터 B: {filter_b}")
    print("\n[2] 패턴 입력")
    pattern = get_matrix("패턴", 3, 3)
    print(f"\n저장된 패턴: {pattern}")

    result_a = calc_mac(filter_a, pattern)
    result_b = calc_mac(filter_b, pattern)
    decision = {"Cross": "A", "X": "B", "UNDECIDED": "UNDECIDED (판정 불가)"}
    print("\n[MAC 결과]")
    print(f"A 점수: {result_a}")
    print(f"B 점수: {result_b}")
    print(f"판정: {decision[judge(result_a, result_b)]}")

    repeat = 10
    print(f"\n[3] 성능 분석 (필터별 평균/{repeat}회, 워밍업 1회 제외)")
    print("측정 순서는 매 반복마다 교대하며 변환 시간은 별도로 측정합니다.")
    measurements = []
    for label, filter_matrix in (("A", filter_a), ("B", filter_b)):
        print(f"\n크기 3×3 / 필터 {label} / 연산 횟수(N²) 9")
        comparison = benchmark_layouts(filter_matrix, pattern, repeat)
        measurements.append({"size": 3, "filter": label, "operations": 9,
                             "average_ms": comparison["two_d"]["mean_ms"], **comparison})
        print_layout_comparison(comparison)
    print("\n크기(N×N)  필터     2D 평균 시간(ms)     연산 횟수(N²)")
    for result in measurements:
        print(f"3×3        {result['filter']}       {result['average_ms']:>16.6f} {9:>16}")
    return measurements


def print_pattern(label, matrix, preview_size=9):
    size = len(matrix)
    print(f"\n{label} ({size}×{size})")
    if size > preview_size:
        print(f"좌상단 {preview_size}×{preview_size} 미리보기")
    for row in matrix[:preview_size]:
        print(" ".join(str(int(value)) for value in row[:preview_size]))


def run_generated_mode():
    while True:
        try:
            size = int(input("패턴 크기 N (1 이상의 정수): "))
            if size < 1:
                raise ValueError
            break
        except ValueError:
            print("입력 형식 오류: 1 이상의 정수를 입력하세요.")
    patterns = {"Cross": make_cross(size), "X": make_x(size)}
    for label, pattern in patterns.items():
        print_pattern(label, pattern)
    if size % 2 == 0:
        print(f"\n짝수 크기 Cross의 중심 행/열은 0부터 세어 {size // 2}입니다.")
    for label, pattern in patterns.items():
        cross_score = calc_mac(patterns["Cross"], pattern)
        x_score = calc_mac(patterns["X"], pattern)
        print(f"\n입력 패턴 {label}")
        print(f"Cross 점수: {cross_score}")
        print(f"X 점수: {x_score}")
        print(f"판정: {judge(cross_score, x_score)}")
        print("성능 비교 (평균/10회, 각 구조 워밍업 1회 제외, 매 반복 측정 순서 교대)")
        for filter_label, filter_matrix in patterns.items():
            print(f"\n크기 {size}×{size} / 필터 {filter_label} / 연산 횟수(N²) {size * size}")
            print_layout_comparison(benchmark_layouts(filter_matrix, pattern))


def run_json_mode(path=DEFAULT_DATA_PATH):
    data = read_data(path)
    errors = {}
    filters = load_filters(data=data, errors=errors)
    print(f"\n[1] 필터 로드 ({path})")
    for size in sorted(filters):
        print(f"size_{size} 필터 로드 완료 (Cross, X)")
    for key, reason in errors.items():
        print(f"{key} 필터 로드 실패: {reason}")
    results = analyze_patterns(data=data, filters_by_size=filters, filter_errors=errors)
    print("\n측정 입력은 생성한 십자가입니다. 유효한 필터가 없는 크기는 십자가 필터도 생성합니다.")
    analyze_time(filters_by_size=filters)
    print_summary(results)
    return results


def main():
    print("=== Mini NPU Simulator ===")
    print("\n[모드 선택]")
    print("1. 사용자 입력 (3×3)")
    print("2. data.json 분석")
    print("3. 패턴 생성 및 2D/1D 성능 비교")
    try:
        mode = input("선택 (1, 2 또는 3): ").strip()
        while mode not in ("1", "2", "3"):
            print("입력 형식 오류: 1, 2 또는 3을 입력하세요.")
            mode = input("선택 (1, 2 또는 3): ").strip()
        if mode == "1":
            run_user_mode()
        elif mode == "2":
            run_json_mode()
        else:
            run_generated_mode()
    except EOFError:
        print("\n입력이 종료되었습니다. 필요한 행을 모두 입력한 뒤 다시 실행하세요.")
        return 1
    except KeyboardInterrupt:
        print("\n사용자가 실행을 중단했습니다.")
        return 130
    except MemoryError:
        print("\n메모리가 부족합니다. 더 작은 패턴 크기로 다시 실행하세요.")
        return 1
    except (OSError, ValueError, OverflowError) as error:
        print(f"\n실행 오류: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
