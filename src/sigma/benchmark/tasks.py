"""Standardized benchmark tasks for evaluating multi-agent frameworks.

Covers LITE / STANDARD / RIGOROUS tiers with ground-truth parameters
for accuracy scoring.
"""

from dataclasses import dataclass, field


@dataclass
class BenchmarkTask:
    """A single benchmark task with expected outcomes."""

    id: str
    instruction: str
    category: str                        # "lookup" | "calculation" | "analysis" | "design"
    expected_tier: str                   # "lite" | "standard" | "rigorous"
    expected_params: dict[str, float]    # ground-truth values for accuracy scoring
    tolerance: float = 0.15              # acceptable relative error (15%)
    min_agents: int = 2
    description: str = ""

    # Domain keywords that should appear in agent analyses
    key_concepts: list[str] = field(default_factory=list)


TASKS: list[BenchmarkTask] = [
    BenchmarkTask(
        id="b001_lookup_density",
        instruction="查找6061-T6铝合金的密度和屈服强度",
        category="lookup",
        expected_tier="lite",
        expected_params={"density_kg_m3": 2700, "yield_strength_mpa": 276},
        tolerance=0.05,
        min_agents=2,
        description="Simple material property lookup — LITE tier",
        key_concepts=["6061-T6", "铝合金", "密度", "屈服强度"],
    ),
    BenchmarkTask(
        id="b002_calc_tube_mass",
        instruction="计算外径150mm、壁厚3mm、长度2m的6061-T6铝管质量（kg）",
        category="calculation",
        expected_tier="lite",
        expected_params={"mass_kg": 7.5},
        tolerance=0.10,
        min_agents=2,
        description="Straightforward engineering calculation — LITE tier",
        key_concepts=["铝管", "质量", "150mm", "壁厚"],
    ),
    BenchmarkTask(
        id="b003_isentropic_flow",
        instruction="空气在γ=1.4、P0=5bar、Pe=1bar条件下，计算等熵膨胀出口马赫数和出口温度（T0=300K）",
        category="calculation",
        expected_tier="standard",
        expected_params={"exit_mach": 1.71, "exit_temperature_k": 189.2},
        tolerance=0.12,
        min_agents=3,
        description="Isentropic flow calculation — STANDARD tier",
        key_concepts=["等熵", "马赫数", "γ", "膨胀"],
    ),
    BenchmarkTask(
        id="b004_thrust_chamber_basic",
        instruction="估算KNSB推进剂在燃烧室压力30bar、扩张比5的条件下的理论比冲和特征速度",
        category="analysis",
        expected_tier="standard",
        expected_params={"isp_s": 155, "c_star_m_s": 920},
        tolerance=0.15,
        min_agents=4,
        description="Propellant performance analysis — STANDARD tier",
        key_concepts=["KNSB", "比冲", "特征速度", "燃烧室"],
    ),
    BenchmarkTask(
        id="b005_thrust_curve",
        instruction="设计一个KNSB固体火箭发动机，推力500N，燃烧时间3秒，估算推进剂质量、燃烧室压力和喷管喉径",
        category="design",
        expected_tier="rigorous",
        expected_params={"propellant_mass_kg": 1.1, "chamber_pressure_bar": 40, "nozzle_throat_mm": 16},
        tolerance=0.20,
        min_agents=5,
        description="Solid motor design — RIGOROUS tier",
        key_concepts=["固体火箭", "推力", "燃烧时间", "喷管"],
    ),
    BenchmarkTask(
        id="b006_material_selection",
        instruction="为小型火箭箭体选择结构材料：比较6061-T6铝管、304不锈钢管和碳纤维管的强度重量比和成本，给出推荐",
        category="analysis",
        expected_tier="standard",
        expected_params={},
        tolerance=0.15,
        min_agents=4,
        description="Multi-criteria material selection — STANDARD tier",
        key_concepts=["强度重量比", "成本", "铝", "不锈钢", "碳纤维"],
    ),
    BenchmarkTask(
        id="b007_stability_margin",
        instruction="一枚长度2m、直径150mm的火箭，质心距头部1.2m，压心距头部1.5m。计算静稳定裕度，判断是否稳定",
        category="calculation",
        expected_tier="lite",
        expected_params={"stability_margin_cal": 2.0},
        tolerance=0.10,
        min_agents=2,
        description="Static stability check — LITE tier",
        key_concepts=["静稳定裕度", "质心", "压心", "稳定性"],
    ),
    BenchmarkTask(
        id="b008_full_motor_design",
        instruction="设计一枚1km级验证火箭的完整动力系统：选择推进剂、确定发动机尺寸、估算总冲和推力曲线、计算结构质量并给出质量预算",
        category="design",
        expected_tier="rigorous",
        expected_params={},
        tolerance=0.25,
        min_agents=6,
        description="Full propulsion system design — RIGOROUS tier",
        key_concepts=["验证火箭", "动力系统", "推进剂", "总冲", "推力曲线", "质量预算"],
    ),
    BenchmarkTask(
        id="b009_thermal_analysis",
        instruction="计算不锈钢喷管在3000K燃烧温度下的稳态温度分布，假设喷管外壁强制对流冷却，冷却液为水（20°C，流速5m/s）",
        category="analysis",
        expected_tier="standard",
        expected_params={},
        tolerance=0.20,
        min_agents=3,
        description="Thermal analysis with cooling — STANDARD tier",
        key_concepts=["温度分布", "对流冷却", "喷管", "热分析"],
    ),
    BenchmarkTask(
        id="b010_parachute_sizing",
        instruction="为质量10kg的回收舱设计降落伞系统：计算所需伞面积、下降速度、选择伞型和材料，确保着陆速度<6m/s",
        category="design",
        expected_tier="rigorous",
        expected_params={"parachute_area_m2": 8.5, "descent_speed_m_s": 5.5},
        tolerance=0.20,
        min_agents=4,
        description="Parachute recovery system design — RIGOROUS tier",
        key_concepts=["降落伞", "回收", "下降速度", "伞面积"],
    ),
]
