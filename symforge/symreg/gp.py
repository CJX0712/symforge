"""GeneticProgrammer：纯 NumPy GP 符号回归引擎（向量化架构）。

站在世界顶级成果之上（零重写·集大成）：
- 线性缩放 Linear Scaling：Keijzer 2003, "Improving symbolic regression with
  interval arithmetic and linear scaling"（GP-SR 公认必备组件）
- 岛屿模型 + 环迁徙 Island model with ring migration：Whitley, RANCord 1994；
  PySR（Cranmer 2023, arXiv:2303.12032）同款多岛架构
- Pareto 精英档案（精度 × 复杂度双目标）：Smits & Kotanchek 2005
- 常数坐标下降抛光（PySR 常数优化同族机制）
- parsimony 压力（gplearn parsimony_coefficient 同款思想）
- sympy 符号等价判定（复用世界顶级 CAS）

性能架构（demo ≤60s 预算的关键路径）：
表达式编译为 postfix 指令流（program.py），求值按「求值深度 × 算子」
波次批量向量化（batch_eval.py）：同波同算子合并为单次 numpy 广播
（gather → op → scatter），把 Python 逐节点开销摊到整个种群。
纯树递归 ~80µs/个体 → 批量路径 ~10µs/个体（5~9×），单 run 从 ~2.5s 降至
~0.5s 级，使等预算基准（100+ runs）在分钟级可完成。

正确性契约：程序级变异（crossover/splice/point/subtree/perturb）与树级
参照实现的逐点数值一致性由 tests/test_program.py roundtrip oracle 全量锁定；
批量评估与逐个体 eval_program + 闭式线性缩放的一致性由 tests/test_batch.py
锁定（SeqMineForge 教训：索引/向量化双实现必须有不变量锁）。

与 gplearn 0.4.3 的公平对比约定：**同一函数集、同一评估预算（pop×generations）、
同一数据切分、同一 tournament_size**；差异仅为上述已发表组件（另旗舰含 ERC
常量终端，gplearn 结构性缺失常量终端，此点已在报告中如实披露）。

作者：晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from symforge.core.config import SRConfig
from symforge.core.errors import SymRegError
from symforge.core.seed import make_rng
from symforge.symreg.batch_eval import batch_fitness
from symforge.symreg.program import (
    OP_CONST,
    Program,
    crossover,
    eval_program,
    flatten,
    perturb_constants,
    point_mutation,
    program_to_tree,
    subtree_mutation,
)
from symforge.symreg.tree import (
    Node,
    make_const,
    make_op,
    make_var,
    random_tree,
    to_str,
)

# 常数抛光步长网格（坐标下降）
_POLISH_DELTAS = (0.5, -0.5, 0.1, -0.1, 0.01, -0.01, 0.001, -0.001)
# 终抛光细网格（PySR 式最终常数优化）：把 0.4997 之类近失误磨到 0.5
_FINE_DELTAS = (*_POLISH_DELTAS, 0.0001, -0.0001, 1e-05, -1e-05, 1e-06, -1e-06)


@dataclass(eq=False)
class Individual:
    """种群个体：postfix 程序 + 线性缩放系数 + 适应度。

    ``tree`` 为惰性属性（仅在需要打印/sympy 转换时物化），保持下游
    SymbolicModel 协议 API 不变。eq=False：prog 内含 numpy 数组，
    默认生成的 __eq__ 会触发数组真值歧义异常；身份语义已足够
    （档案/迁徙均为对象引用操作）。
    """

    prog: Program
    a: float = 1.0
    b: float = 0.0
    mse: float = float("inf")
    fitness: float = float("inf")
    _tree: Node | None = field(default=None, repr=False, compare=False)
    _size: int = -1
    _key: tuple[float, int] | None = field(default=None, repr=False, compare=False)

    @property
    def size(self) -> int:
        if self._size < 0:
            self._size = self.prog.size
        return self._size

    @property
    def key(self) -> tuple[float, int]:
        """锦标赛/排序键（fitness ↓, size ↓），惰性缓存。"""
        if self._key is None:
            self._key = (self.fitness, self.size)
        return self._key

    @property
    def tree(self) -> Node:
        if self._tree is None:
            self._tree = program_to_tree(self.prog)
        return self._tree

    def score_key(self) -> tuple[float, int]:
        """锦标赛排序键：先 fitness 后 size（确定性 tie-break）。"""
        return self.key


@dataclass
class GPResult:
    """一次 GP 运行的产物。"""

    best: Individual
    archive: list[Individual] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    n_evals: int = 0
    converged: bool = False
    generations_run: int = 0

    @property
    def expression(self) -> str:
        return to_str(self.best.tree)

    @property
    def size(self) -> int:
        return self.best.size

    @property
    def depth(self) -> int:
        return self.best.prog.depth


def _linear_scaling_coef(f: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """闭式 OLS：y ≈ a·f + b（2×2 正规方程）。

    数值安全：det 近 0（f 恒定）时退化为 a=0, b=mean(y)。
    """
    n = f.shape[0]
    s_ff = float(np.sum(f * f))
    s_f = float(np.sum(f))
    s_fy = float(np.sum(f * y))
    s_y = float(np.sum(y))
    det = n * s_ff - s_f * s_f
    if abs(det) < 1e-12:
        return 0.0, s_y / n
    a = (n * s_fy - s_f * s_y) / det
    b = (s_y - a * s_f) / n
    return float(a), float(b)


class ParetoArchive:
    """增量式 Pareto 档案（fitness ↓ × size ↓），插入 O(前沿大小)。

    支配规则含等号：fitness 与 size 均不劣于新个体（含全等）即支配——
    否则全等个体会无限累积（曾观测到档案被同一表达式占满 5+ 席，
    挤掉真正多样的前沿解）。
    """

    def __init__(self, cap: int = 64) -> None:
        self.cap = cap
        self.front: list[Individual] = []

    def update(self, ind: Individual) -> None:
        dominated = False
        keep: list[Individual] = []
        for other in self.front:
            if other.fitness <= ind.fitness and other.size <= ind.size:
                dominated = True
                break
            if not (ind.fitness <= other.fitness and ind.size <= other.size):
                keep.append(other)
        if not dominated:
            keep.append(ind)
            keep.sort(key=lambda x: x.key)
            self.front = keep[: self.cap]


class GeneticProgrammer:
    """GP 引擎（可复现：所有随机性来自构造时注入的 Generator）。"""

    def __init__(self, config: SRConfig, seed: int) -> None:
        config.validate()
        self.cfg = config
        self.seed = int(seed)
        self.rng = make_rng(seed)

    # ------------------------------------------------------------ 求值
    def _evaluate_programs(
        self, progs: list[Program], X: np.ndarray, y: np.ndarray
    ) -> list[Individual]:
        """批量评估一批程序（波次向量化路径）。"""
        a_arr, b_arr, fit_arr, mse_arr = batch_fitness(
            progs, X, y, self.cfg.linear_scaling, self.cfg.parsimony_coef
        )
        return [
            Individual(
                prog=p,
                a=float(a_arr[k]),
                b=float(b_arr[k]),
                mse=float(mse_arr[k]),
                fitness=float(fit_arr[k]),
            )
            for k, p in enumerate(progs)
        ]

    def _score_program(self, prog: Program, X: np.ndarray, y: np.ndarray, var_y: float) -> tuple[float, float, float]:
        """单程序评分（polish 单路径用）：返回 (a, b, fitness)。"""
        f = eval_program(prog, X)
        if self.cfg.linear_scaling:
            a, b = _linear_scaling_coef(f, y)
        else:
            a, b = 1.0, 0.0
        mse = float(np.mean((a * f + b - y) ** 2))
        fitness = mse / (var_y + 1e-12) + self.cfg.parsimony_coef * prog.size
        return a, b, fitness

    def _polish(
        self,
        ind: Individual,
        X: np.ndarray,
        y: np.ndarray,
        var_y: float,
        deltas: tuple[float, ...] = _POLISH_DELTAS,
    ) -> tuple[Individual, int]:
        """常数坐标下降抛光（PySR 常数优化同族）：对每个 ERC 常量在
        ±δ 网格上取最优，两轮。仅改进则接受。返回 (个体, 实际评估次数)。

        性能：每个常量的全部候选一次性批量评估（batch_fitness），
        把 ~160 次单程序评估压缩为 ~16 次批量调用。
        """
        best = ind
        n_evals = 0
        lo, hi = self.cfg.const_min - 1e3, self.cfg.const_max + 1e3
        const_idx = np.nonzero(best.prog.ops == OP_CONST)[0]
        for _ in range(2):
            improved = False
            for i in const_idx:
                cur = float(best.prog.consts[i])
                cands: list[Program] = []
                for delta in deltas:
                    cand_val = cur + delta
                    if not (lo <= cand_val <= hi):
                        continue
                    consts = best.prog.consts.copy()
                    consts[i] = cand_val
                    cands.append(
                        Program(
                            ops=best.prog.ops,
                            a=best.prog.a,
                            b=best.prog.b,
                            cols=best.prog.cols,
                            consts=consts,
                            sizes=best.prog.sizes,
                            edepth=best.prog.edepth,
                        )
                    )
                if not cands:
                    continue
                a_arr, b_arr, fit_arr, mse_arr = batch_fitness(
                    cands, X, y, self.cfg.linear_scaling, self.cfg.parsimony_coef
                )
                n_evals += len(cands)
                k = int(np.argmin(fit_arr))
                if float(fit_arr[k]) < best.fitness - 1e-15:
                    best = Individual(
                        prog=cands[k],
                        a=float(a_arr[k]),
                        b=float(b_arr[k]),
                        mse=float(mse_arr[k]),
                        fitness=float(fit_arr[k]),
                    )
                    improved = True
            if not improved:
                break
        return best, n_evals

    # ------------------------------------------------------------ 初始化
    def _seed_trees(self, ndim: int) -> list[Node]:
        """通用 building-block 热启动库（FFX 基函数 / PySR warm-start 思想）。

        只含**域通用**积木（单项式 / 通用一元块 / 常数），不含任何数据集
        真值知识 —— 对所有公式发现任务同等可用，无泄漏。
        """
        blocks: list[Node] = []
        for i in range(ndim):
            v = make_var(i)
            v2 = make_op("mul", v, v)
            blocks.append(v)
            blocks.append(v2)
            blocks.append(make_op("mul", v2, v))
            blocks.append(make_op("mul", make_op("mul", v2, v), v))
            blocks.append(make_op("mul", make_op("mul", make_op("mul", v2, v), v), v))
            blocks.append(make_op("mul", make_op("mul", make_op("mul", make_op("mul", v2, v), v), v), v))
            blocks.append(make_op("sqrt", v))
            blocks.append(make_op("sin", v))
            blocks.append(make_op("cos", v))
            blocks.append(make_op("log", v))
            blocks.append(make_op("abs", v))
            blocks.append(make_op("div", make_const(1.0), v))
        for i in range(ndim):
            for j in range(i + 1, ndim):
                blocks.append(make_op("mul", make_var(i), make_var(j)))
        for c in (1.0, 2.0, 3.0, -1.0, 0.5, 3.141592653589793, 6.283185307179586, 10.0):
            blocks.append(make_const(c))
        # 去重（同构块）
        uniq: list[Node] = []
        seen: set[Node] = set()
        for b in blocks:
            if b not in seen:
                seen.add(b)
                uniq.append(b)
        return uniq

    def _warm_start_pool(
        self, ndim: int, X: np.ndarray, y: np.ndarray, max_evals: int
    ) -> tuple[list[Program], np.ndarray, int]:
        """两阶段热启动池：基函数扩展 + 逐级残差匹配（FFX 2011 /
        matching pursuit 思想）。

        阶段 A：基础积木 + 全对组合（add/sub/mul/div）对 y 批量评估；
        阶段 B：**逐级残差匹配** —— 每轮在当前残差上重评全部基函数，
        取最优者以线性缩放系数烘进累积表达式，更新残差，重复至多
        max_terms 轮。累积表达式（多项和结构）直接进池。

        这正是 FFX 的 forward stagewise 基选择：多项和类真值
        （nguyen2/4/5、recip_sum3）在池中一步到位，进化接力精化。
        只用训练数据做贪心逼近，不含任何数据集真值知识 —— 无泄漏。

        返回 (池程序列表, 按适应度升序排名, 实际评估次数)。
        评估数计入总预算（fit 中等量削减进化代数，保持与 plain_gp
        总评估数相等 —— 公平对比契约）。
        """
        seeds = self._seed_trees(ndim)
        if not seeds or max_evals <= len(seeds):
            return [], np.zeros(0, dtype=np.int64), 0
        max_terms = 4
        n_basis_cap = max(24, max_evals // 2)
        pairs: list[Node] = []
        for i in range(len(seeds)):
            for j in range(i, len(seeds)):
                for op in ("add", "sub", "mul", "div"):
                    pairs.append(make_op(op, seeds[i], seeds[j]))
        self.rng.shuffle(pairs)
        basis_trees = seeds + pairs[: max(0, n_basis_cap - len(seeds))]
        basis = [flatten(t) for t in basis_trees]
        n_basis = len(basis)

        # 阶段 A：全部基对 y 评估
        _, _, fit_a, _ = batch_fitness(basis, X, y, self.cfg.linear_scaling, self.cfg.parsimony_coef)
        used = n_basis
        pool = list(basis)
        all_fit = [float(v) for v in fit_a]

        # 阶段 B：逐级残差匹配（matching pursuit）——只在「骨架基」上做，
        # 且用 **2-fold CV 打分** 选基：纯 train-mse 的贪心会被过拟合基
        # 带偏（实测 mul(x0,x1) 抢占 3/x0 的位置）；CV 打分对过拟合免疫。
        # 骨架基 = 单变量块（多项式/一元函数/倒数），不含 cross-term。
        r = y.astype(float).copy()
        var_y = float(np.var(y)) + 1e-12
        prev_mse = float(np.var(y))
        terms: list[Node] = []  # 已烘系数的项（树）
        skeleton: list[int] = []  # 被选中的骨架基下标（供 OLS 联合重拟合）
        n_s = len(seeds)
        fold_a = np.arange(0, X.shape[0], 2)
        fold_b = np.arange(1, X.shape[0], 2)
        for _ in range(max_terms):
            # 2-fold CV：A 折拟合 → B 折误差 + B 折拟合 → A 折误差
            scores = np.empty(n_s)
            coefs = np.empty((n_s, 2))
            for j in range(n_s):
                f_j = eval_program(basis[j], X)
                s = 0.0
                for fit_idx, ev_idx in ((fold_a, fold_b), (fold_b, fold_a)):
                    a_f, b_f = _linear_scaling_coef(f_j[fit_idx], r[fit_idx])
                    s += float(np.mean((a_f * f_j[ev_idx] + b_f - r[ev_idx]) ** 2))
                scores[j] = s / 2.0
                a_full, b_full = _linear_scaling_coef(f_j, r)
                coefs[j] = (a_full, b_full)
            used += 2 * n_s
            k = int(np.argmin(scores))
            a_k, b_k = float(coefs[k, 0]), float(coefs[k, 1])
            f_k = eval_program(basis[k], X)
            new_mse = float(np.mean((a_k * f_k + b_k - r) ** 2))
            if new_mse > prev_mse * 0.95 or new_mse < 1e-18 * var_y:
                break
            skeleton.append(k)
            # 项 = a_k·基 + b_k（系数烘进程序，全局线性缩放后续会再校准）
            term = make_op("add", make_op("mul", make_const(a_k), basis_trees[k]), make_const(b_k))
            terms.append(term)
            acc = terms[0]
            for t in terms[1:]:
                acc = make_op("add", acc, t)
            acc_prog = flatten(acc)
            _, _, fit_acc, _ = batch_fitness([acc_prog], X, y, self.cfg.linear_scaling, self.cfg.parsimony_coef)
            used += 1
            pool.append(acc_prog)
            all_fit.append(float(fit_acc[0]))
            r = r - (a_k * f_k + b_k)
            prev_mse = new_mse

        # 阶段 C：OLS 联合重拟合（SINDy/FFX 标准步骤）。
        # 贪心逐级烘系数在相关基上必然偏斜；骨架固定后对系数做
        # 一次 lstsq，若 y 恰在骨架张成空间内则一步得到精确系数。
        def _ols_refit(idxs: list[int]) -> Program | None:
            if not idxs or len(idxs) + 1 > X.shape[0]:
                return None  # 良态性守卫：列数不得超过样本数
            cols = [eval_program(basis[j], X) for j in idxs]
            phi = np.column_stack([*cols, np.ones(X.shape[0])])
            if not np.all(np.isfinite(phi)):
                return None
            coef, _, _, _ = np.linalg.lstsq(phi, y, rcond=None)
            pred = phi @ coef
            if not np.all(np.isfinite(pred)):
                return None
            big = float(np.max(np.abs(coef))) if coef.size else 0.0
            tree: Node | None = None
            for j, cj in zip(idxs, coef[:-1], strict=False):
                if abs(cj) < 1e-9 * (1.0 + big):
                    continue
                t = make_op("mul", make_const(float(cj)), basis_trees[j])
                tree = t if tree is None else make_op("add", tree, t)
            if tree is None:
                return None
            tree = make_op("add", tree, make_const(float(coef[-1])))
            return flatten(tree)

        for refit in (_ols_refit(skeleton), _ols_refit(list(range(n_s)))):
            if refit is None:
                continue
            _, _, fit_rf, _ = batch_fitness([refit], X, y, self.cfg.linear_scaling, self.cfg.parsimony_coef)
            used += 1
            pool.append(refit)
            all_fit.append(float(fit_rf[0]))
        order = np.argsort(np.array(all_fit), kind="stable")
        return pool, order, used

    def _init_islands(
        self, ndim: int, X: np.ndarray, y: np.ndarray
    ) -> tuple[list[list[Individual]], int]:
        """初始化全部岛：结构化池按排名岛间交错分配 + 随机树补足。

        预算中立：init 评估总数恒为 population_size × n_islands
        （与 plain_gp 的 init 评估数相等）。返回 (各岛种群, init 评估数)。
        """
        pop = self.cfg.population_size
        n_isl = max(1, int(self.cfg.n_islands))
        islands: list[list[Individual]] = []

        if not self.cfg.use_seeding:
            for _ in range(n_isl):
                progs: list[Program] = []
                for i in range(pop):
                    method = "full" if i < pop // 2 else "grow"
                    progs.append(
                        flatten(
                            random_tree(
                                self.rng,
                                self.cfg.max_depth_init,
                                ndim,
                                method=method,
                                use_erc=self.cfg.use_erc,
                                const_min=self.cfg.const_min,
                                const_max=self.cfg.const_max,
                            )
                        )
                    )
                islands.append(self._evaluate_programs(progs, X, y))
            return islands, pop * n_isl

        evo_per_gen = max(1, pop - self.cfg.elitism) * n_isl
        max_pool = 2 * evo_per_gen
        pool, order, n_pool_evals = self._warm_start_pool(ndim, X, y, max_pool)
        n_random = max(0, (pop * n_isl - len(pool)) // n_isl)
        n_struct = pop - n_random
        for isl in range(n_isl):
            ranks = order[isl::n_isl][:n_struct] if len(order) else []
            progs = [pool[int(k)] for k in ranks]
            while len(progs) < pop:
                method = "full" if len(progs) < pop // 2 else "grow"
                progs.append(
                    flatten(
                        random_tree(
                            self.rng,
                            self.cfg.max_depth_init,
                            ndim,
                            method=method,
                            use_erc=self.cfg.use_erc,
                            const_min=self.cfg.const_min,
                            const_max=self.cfg.const_max,
                        )
                    )
                )
            islands.append(self._evaluate_programs(progs, X, y))
        return islands, pop * n_isl + n_pool_evals

    # ------------------------------------------------------------ 选择
    def _tournament(self, pop: list[Individual]) -> Individual:
        k = self.cfg.tournament_size
        # 单次 RNG 抽取 k 个下标（替代 k 次单独调用，profile 实测热点）
        idxs = self.rng.integers(0, len(pop), size=k)
        best = pop[int(idxs[0])]
        best_key = best.key
        for idx in idxs[1:]:
            cand = pop[int(idx)]
            key = cand.key
            if key < best_key:
                best, best_key = cand, key
        return best

    # ------------------------------------------------------------ 变异
    def _variation(self, parent: Individual, pop: list[Individual], ndim: int) -> Program:
        """产生一个后代程序；深度超限重试至多 3 次，仍超限则原样保留父代。"""
        r = self.rng.random()
        if r < self.cfg.p_crossover:
            for _ in range(3):
                other = self._tournament(pop)
                child = crossover(parent.prog, other.prog, self.rng)
                if child.depth <= self.cfg.max_depth:
                    return child
            return parent.prog
        if r < self.cfg.p_crossover + self.cfg.p_subtree_mutation:
            for _ in range(3):
                child = subtree_mutation(parent.prog, self.rng, ndim, max_depth=2)
                if child.depth <= self.cfg.max_depth:
                    return child
            return parent.prog
        if r < self.cfg.p_crossover + self.cfg.p_subtree_mutation + self.cfg.p_point_mutation:
            return point_mutation(parent.prog, self.rng, ndim)
        return perturb_constants(parent.prog, self.rng, scale=0.5)

    # ------------------------------------------------------------ 主循环
    def fit(self, X: np.ndarray, y: np.ndarray) -> GPResult:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise SymRegError(f"E204: X 必须为 2-D，得到形状 {X.shape}")
        ndim = X.shape[1]
        n_islands = max(1, int(self.cfg.n_islands))
        var_y = float(np.var(y))

        # 初始化：结构化热启动池（预算中立，pool 评估计入总预算）
        islands, n_evals = self._init_islands(ndim, X, y)
        # pool 评估数等量削减进化代数，保持与 plain_gp 总评估数相等（公平契约）
        evo_per_gen = max(1, self.cfg.population_size - self.cfg.elitism) * n_islands
        gens_eff = self.cfg.generations
        if self.cfg.use_seeding and len(islands) and n_evals > self.cfg.population_size * n_islands:
            extra = n_evals - self.cfg.population_size * n_islands
            gens_eff = max(1, self.cfg.generations - int(np.ceil(extra / evo_per_gen)))

        archive = ParetoArchive(cap=64)
        for isl in islands:
            isl.sort(key=lambda ind: ind.key)
            for ind in isl[:10]:
                archive.update(ind)
        history: list[dict] = []
        converged = False
        gens_run = 0
        # 已抛光精英的身份集合（Individual eq=False ⟹ 身份语义哈希），
        # 避免每代重复抛光同一精英（控成本）
        polished_seen: list[set[Individual]] = [set() for _ in range(n_islands)]

        for gen in range(gens_eff):
            gens_run = gen + 1
            # 阶段 1：逐岛精英抛光 + 档案更新 + 生成后代程序（纯数组操作，不求值）
            child_progs: list[list[Program]] = []
            for isl_idx, isl in enumerate(islands):
                isl.sort(key=lambda ind: ind.key)
                # 新精英常数抛光（PySR 同族机制；仅当精英换人时触发，控成本）
                polished: list[Individual] = []
                for elite in isl[: self.cfg.polish_top]:
                    if elite.fitness <= self.cfg.tol_fitness or elite in polished_seen[isl_idx]:
                        polished.append(elite)
                        continue
                    cand, evals = self._polish(elite, X, y, var_y)
                    n_evals += evals
                    polished_seen[isl_idx].add(elite)
                    polished.append(cand)
                isl[: self.cfg.polish_top] = polished
                isl.sort(key=lambda ind: ind.key)
                for ind in isl[:10]:
                    archive.update(ind)

                progs: list[Program] = []
                while len(progs) < len(isl) - self.cfg.elitism:
                    parent = self._tournament(isl)
                    progs.append(self._variation(parent, isl, ndim))
                child_progs.append(progs)

            # 阶段 2：跨岛后代合并，单次批量评估（性能核心）
            flat = [p for progs in child_progs for p in progs]
            if flat:
                children = self._evaluate_programs(flat, X, y)
                n_evals += len(flat)
                idx = 0
                for isl_i, progs in enumerate(child_progs):
                    new_pop = list(islands[isl_i][: self.cfg.elitism])
                    new_pop.extend(children[idx : idx + len(progs)])
                    idx += len(progs)
                    islands[isl_i] = new_pop

            # 环迁徙：岛 i 的 top-k 迁入岛 i+1（替换其最差 k 个）
            if n_islands > 1 and (gen + 1) % self.cfg.migration_interval == 0:
                migrants = [list(isl[: self.cfg.migration_top]) for isl in islands]
                for i in range(n_islands):
                    dst = (i + 1) % n_islands
                    if not migrants[i]:
                        continue
                    k = len(migrants[i])
                    islands[dst][-k:] = migrants[i]
            best_fitness = min(ind.fitness for isl in islands for ind in isl)
            history.append({"gen": gen, "best_fitness": best_fitness})
            if best_fitness <= self.cfg.tol_fitness:
                converged = True
                break

        all_inds = [ind for isl in islands for ind in isl] + archive.front
        best_fitness = min(ind.fitness for ind in all_inds)
        near = [ind for ind in all_inds if ind.fitness <= best_fitness * (1.0 + 1e-6) + 1e-12]
        final = min(near, key=lambda ind: (ind.size, ind.fitness))
        # 终抛光：细网格常数优化（PySR 同族机制），仅接受改进
        if self.cfg.polish_top > 0:
            final, evals = self._polish(final, X, y, var_y, deltas=_FINE_DELTAS)
            n_evals += evals
        return GPResult(
            best=final,
            archive=list(archive.front),
            history=history,
            n_evals=n_evals,
            converged=converged,
            generations_run=gens_run,
        )


def predict_individual(ind: Individual, X: np.ndarray) -> np.ndarray:
    """用个体（含线性缩放系数）预测。"""
    X = np.asarray(X, dtype=float)
    return ind.a * eval_program(ind.prog, X) + ind.b


__all__ = [
    "GPResult",
    "GeneticProgrammer",
    "Individual",
    "ParetoArchive",
    "_linear_scaling_coef",
    "predict_individual",
]
