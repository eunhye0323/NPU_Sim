"""표준 라이브러리로 구현한 Mini NPU 콘솔 시뮬레이터."""

import json
import math
from pathlib import Path
import re
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
    matrix = []
    while len(matrix) < rows:
        try:
            row = [float(value) for value in input(f"{len(matrix) + 1}행> ").split()]
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


def make_cross(size):
    """size×size 십자가 행렬을 생성한다."""
    center = size // 2
    return [
        [1.0 if i == center or j == center else 0.0 for j in range(size)]
        for i in range(size)
    ]


def measure_mac(filter_matrix, pattern, repeat=10):
    """I/O와 입력 검증을 제외한 MAC 함수 호출의 평균 시간을 ms로 반환한다."""
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 10:
        raise ValueError("성능 측정은 정수 횟수로 최소 10회 반복해야 합니다")
    calc_mac(filter_matrix, pattern)  # 측정 전 워밍업
    elapsed = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        calc_mac(filter_matrix, pattern)
        elapsed += time.perf_counter() - start
    return elapsed / repeat * 1000


def analyze_time(path=DEFAULT_DATA_PATH, repeat=10, extra_sizes=(3, 5, 13, 25),
                 *, filters_by_size=None):
    """크기별 필터 하나와 생성한 십자가 입력으로 MAC 평균 시간을 출력한다."""
    if filters_by_size is None:
        filters_by_size = load_filters(path, errors={})
    sizes = sorted(set(extra_sizes) | set(filters_by_size))
    print("\n[3] 성능 분석 (필터 하나의 MAC 함수 호출)")
    print(f"각 크기별 평균/{repeat}회, 워밍업 1회 제외")
    print("크기(N×N)     평균 시간(ms)     연산 횟수(N²)")
    measurements = []
    for size in sizes:
        pattern = make_cross(size)
        filters = filters_by_size.get(size, {})
        filter_matrix = next(iter(filters.values()), pattern)
        try:
            average_ms = measure_mac(filter_matrix, pattern, repeat)
        except (ValueError, OverflowError) as error:
            measurements.append({"size": size, "average_ms": None,
                                 "operations": size * size, "reason": str(error)})
            print(f"{size}×{size:<8} {'측정 불가':>12} {size * size:>16}")
            print(f"  사유: {error}")
            continue
        measurements.append({"size": size, "average_ms": average_ms, "operations": size * size})
        print(f"{size}×{size:<8} {average_ms:>12.6f} {size * size:>16}")
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
                  "result": "FAIL", "reason": ""}
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
                result["reason"] = f"판정 {result['decision']} ≠ expected {result['expected']}"
        except (ValueError, OverflowError) as error:
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
    average_a = measure_mac(filter_a, pattern, repeat)
    average_b = measure_mac(filter_b, pattern, repeat)
    print(f"\n[3] 성능 분석 (필터별 평균/{repeat}회, 워밍업 1회 제외)")
    print("크기(N×N)  필터     평균 시간(ms)     연산 횟수(N²)")
    print(f"3×3        A       {average_a:>12.6f} {9:>16}")
    print(f"3×3        B       {average_b:>12.6f} {9:>16}")


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
    try:
        mode = input("선택 (1 또는 2): ").strip()
        while mode not in ("1", "2"):
            print("입력 형식 오류: 1 또는 2를 입력하세요.")
            mode = input("선택 (1 또는 2): ").strip()
        if mode == "1":
            run_user_mode()
        else:
            run_json_mode()
    except EOFError:
        print("\n입력이 종료되었습니다. 필요한 행을 모두 입력한 뒤 다시 실행하세요.")
        return 1
    except KeyboardInterrupt:
        print("\n사용자가 실행을 중단했습니다.")
        return 130
    except (OSError, ValueError, OverflowError) as error:
        print(f"\n실행 오류: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
