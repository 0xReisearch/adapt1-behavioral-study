from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable


_TOKEN = re.compile(r"[a-z0-9]+")

_QUANTITIES = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def text_tokens(value: str, *, keep_numbers: bool = False) -> tuple[str, ...]:
    values = _TOKEN.findall(str(value).lower())
    if not keep_numbers:
        values = [value for value in values if not value.isdigit()]
    return tuple(values)


@dataclass(frozen=True)
class PlanStep:
    observation: str
    action: str


@dataclass(frozen=True)
class EpisodicPlan:
    plan_id: str
    goal: str
    steps: tuple[PlanStep, ...]
    utility: float = 1.0


@dataclass(frozen=True)
class EntityRef:
    entity_type: str
    entity_id: str

    @property
    def key(self) -> str:
        return f"{self.entity_type} {self.entity_id}"


@dataclass(frozen=True)
class ActionFrame:
    head: str
    entities: tuple[EntityRef, ...]
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class _PlanSchema:
    object_type: str | None
    destination_type: str | None
    required_count: int
    source_types: tuple[str, ...]
    search_heads: tuple[str, ...]
    acquisition_heads: tuple[str, ...]
    completion_heads: tuple[str, ...]
    destination_heads: tuple[str, ...]
    relation_heads: tuple[str, ...]
    relation_target_types: tuple[str, ...]
    stage_navigation_heads: tuple[tuple[str, ...], ...]


class EpisodicPlanMemory:
    """Retrieve trajectories and reconstruct entity-bound, staged obligations."""

    def __init__(self, plans: Iterable[EpisodicPlan] = ()) -> None:
        self.plans = list(plans)
        self._idf: dict[str, float] = {}
        self._head_frequency: dict[str, float] = {}
        self._head_count: dict[str, int] = {}
        self._head_goal_binding: dict[str, float] = {}
        self._head_precondition_strength: dict[str, float] = {}
        self._head_terminal_probability: dict[str, float] = {}
        self._plan_schemas: dict[str, _PlanSchema] = {}
        self._source_type_counts: dict[str, Counter[str]] = {}
        self._search_head_counts: dict[tuple[str, str], Counter[str]] = {}
        self._rebuild_statistics()

    def add(self, plan: EpisodicPlan) -> None:
        if not plan.steps:
            raise ValueError("episodic plans require at least one step")
        self.plans.append(plan)
        self._rebuild_statistics()

    def session(self, goal: str, *, top_k: int = 24) -> "PlanMemorySession":
        if not self.plans:
            return PlanMemorySession(
                [],
                self._idf,
                text_tokens(goal),
                self._head_goal_binding,
                self._head_terminal_probability,
            )
        goal_tokens = text_tokens(goal)
        ranked = sorted(
            (
                (self._weighted_jaccard(goal_tokens, text_tokens(plan.goal)), plan)
                for plan in self.plans
            ),
            key=lambda item: (item[0], item[1].utility, item[1].plan_id),
            reverse=True,
        )
        selected = []
        for similarity, plan in ranked[: max(1, int(top_k))]:
            if similarity <= 0.0:
                continue
            mapping = _token_mapping(text_tokens(plan.goal), goal_tokens)
            selected.append(
                _ActivePlan(
                    plan=plan,
                    goal_similarity=similarity,
                    milestones=self._milestones(plan),
                    token_mapping=mapping,
                    schema=self._mapped_schema(
                        self._plan_schemas.get(plan.plan_id), mapping, goal
                    ),
                )
            )
        return PlanMemorySession(
            selected,
            self._idf,
            goal_tokens,
            self._head_goal_binding,
            self._head_terminal_probability,
            self._source_type_counts,
            self._search_head_counts,
        )

    def _rebuild_statistics(self) -> None:
        documents = [set(text_tokens(plan.goal)) for plan in self.plans]
        counts = Counter(token for document in documents for token in document)
        total = max(1, len(documents))
        self._idf = {
            token: math.log((1.0 + total) / (1.0 + count)) + 1.0
            for token, count in counts.items()
        }
        heads = Counter(
            head
            for plan in self.plans
            for step in plan.steps
            for head in text_tokens(step.action)[:1]
        )
        head_total = max(1, sum(heads.values()))
        self._head_frequency = {
            head: count / head_total for head, count in heads.items()
        }
        self._head_count = dict(heads)
        bound = Counter()
        eligible = Counter()
        for plan in self.plans:
            goal = text_tokens(plan.goal)
            for step in plan.steps:
                action = text_tokens(step.action)
                if not action:
                    continue
                eligible[action[0]] += 1
                if _weighted_precision(action[1:], goal, self._idf) > 0.08:
                    bound[action[0]] += 1
        self._head_goal_binding = {
            head: bound[head] / count
            for head, count in eligible.items()
        }
        preceding = Counter()
        terminal = Counter()
        for plan in self.plans:
            actions = [text_tokens(step.action) for step in plan.steps]
            if actions and actions[-1]:
                terminal[actions[-1][0]] += 1
            goal = text_tokens(plan.goal)
            for index in range(len(actions) - 1):
                current, following = actions[index], actions[index + 1]
                if not current or not following:
                    continue
                if (
                    self._head_goal_binding.get(following[0], 0.0) >= 0.55
                    and _weighted_precision(following[1:], goal, self._idf) > 0.08
                ):
                    preceding[current[0]] += 1
        self._head_precondition_strength = {
            head: preceding[head] / count
            for head, count in eligible.items()
        }
        self._head_terminal_probability = {
            head: terminal[head] / count
            for head, count in eligible.items()
        }
        self._rebuild_relational_statistics()

    def _rebuild_relational_statistics(self) -> None:
        self._plan_schemas = {}
        source_counts: dict[str, Counter[str]] = defaultdict(Counter)
        search_heads: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for plan in self.plans:
            schema, sources = _induce_plan_schema(plan)
            self._plan_schemas[plan.plan_id] = schema
            if not schema.object_type:
                continue
            for source_type, preceding_heads in sources:
                source_counts[schema.object_type][source_type] += 1
                search_heads[(schema.object_type, source_type)].update(preceding_heads)
        self._source_type_counts = {
            object_type: Counter(values)
            for object_type, values in source_counts.items()
        }
        self._search_head_counts = {
            key: Counter(values)
            for key, values in search_heads.items()
        }

    @staticmethod
    def _mapped_schema(
        schema: _PlanSchema | None,
        mapping: dict[str, str],
        goal: str,
    ) -> _PlanSchema:
        if schema is None:
            return _PlanSchema(None, None, _goal_quantity(goal), (), (), (), (), (), (), (), ())
        return _PlanSchema(
            object_type=(
                mapping.get(schema.object_type, schema.object_type)
                if schema.object_type
                else None
            ),
            destination_type=(
                mapping.get(schema.destination_type, schema.destination_type)
                if schema.destination_type
                else None
            ),
            required_count=max(schema.required_count, _goal_quantity(goal)),
            source_types=schema.source_types,
            search_heads=schema.search_heads,
            acquisition_heads=schema.acquisition_heads,
            completion_heads=schema.completion_heads,
            destination_heads=schema.destination_heads,
            relation_heads=schema.relation_heads,
            relation_target_types=tuple(
                mapping.get(value, value) for value in schema.relation_target_types
            ),
            stage_navigation_heads=schema.stage_navigation_heads,
        )

    def _milestones(self, plan: EpisodicPlan) -> tuple[int, ...]:
        goal_tokens = text_tokens(plan.goal)
        bound_steps: list[int] = []
        for index, step in enumerate(plan.steps):
            action_tokens = text_tokens(step.action)
            head = action_tokens[0] if action_tokens else ""
            goal_precision = _weighted_precision(action_tokens[1:], goal_tokens, self._idf)
            if (
                index == len(plan.steps) - 1
                or (
                    self._head_goal_binding.get(head, 0.0) >= 0.55
                    and goal_precision > 0.08
                )
            ):
                bound_steps.append(index)
        output = {
            prerequisite
            for index in bound_steps
            for prerequisite in (max(0, index - 1), index)
            if prerequisite == index
            or self._head_precondition_strength.get(
                (text_tokens(plan.steps[prerequisite].action) or ("",))[0], 0.0
            ) >= 0.1
        }
        return tuple(sorted(output))

    def _weighted_jaccard(self, left: tuple[str, ...], right: tuple[str, ...]) -> float:
        left_set, right_set = set(left), set(right)
        union = left_set | right_set
        if not union:
            return 0.0
        intersection = left_set & right_set
        numerator = sum(self._idf.get(token, 1.0) for token in intersection)
        denominator = sum(self._idf.get(token, 1.0) for token in union)
        return numerator / denominator if denominator else 0.0


@dataclass
class _ActivePlan:
    plan: EpisodicPlan
    goal_similarity: float
    milestones: tuple[int, ...] = ()
    token_mapping: dict[str, str] | None = None
    schema: _PlanSchema | None = None
    cursor: int = 0
    milestone_cursor: int = 0


class PlanMemorySession:
    def __init__(
        self,
        active: list[_ActivePlan],
        idf: dict[str, float],
        goal_tokens: tuple[str, ...],
        head_goal_binding: dict[str, float] | None = None,
        head_terminal_probability: dict[str, float] | None = None,
        source_type_counts: dict[str, Counter[str]] | None = None,
        search_head_counts: dict[tuple[str, str], Counter[str]] | None = None,
    ) -> None:
        self.active = active
        self.idf = idf
        self.goal_tokens = goal_tokens
        self.head_goal_binding = dict(head_goal_binding or {})
        self.head_terminal_probability = dict(head_terminal_probability or {})
        self.source_type_counts = {
            key: Counter(value) for key, value in (source_type_counts or {}).items()
        }
        self.search_head_counts = {
            key: Counter(value) for key, value in (search_head_counts or {}).items()
        }
        self.completed_goal_entities: set[str] = set()
        self.acquired_goal_entities: set[str] = set()
        self.entity_stages: dict[str, int] = {}
        self.executed_actions: Counter[str] = Counter()
        self.observed_entities: set[str] = set()
        self.current_entities: set[str] = set()

    def rank(
        self,
        *,
        observation: str,
        candidates: Iterable[str],
        lookahead: int = 8,
    ) -> list[dict[str, Any]]:
        current = _entity_refs(observation)
        self.current_entities = {entity.key for entity in current}
        self.observed_entities.update(self.current_entities)
        rows = [
            {
                "candidate": candidate,
                "score": self.score(
                    observation=observation,
                    candidate=candidate,
                    lookahead=lookahead,
                ),
            }
            for candidate in candidates
        ]
        rows.sort(key=lambda item: (item["score"], item["candidate"]), reverse=True)
        return rows

    def score(self, *, observation: str, candidate: str, lookahead: int = 8) -> float:
        if not self.active:
            return 0.0
        candidate_tokens = text_tokens(candidate)
        candidate_head = candidate_tokens[0] if candidate_tokens else ""
        observation_tokens = text_tokens(observation)
        supports: list[tuple[float, float]] = []
        for active in self.active:
            best = 0.0
            stop = min(len(active.plan.steps), active.cursor + max(1, int(lookahead)))
            for index in range(active.cursor, stop):
                reference = active.plan.steps[index]
                reference_tokens = _substitute_tokens(
                    text_tokens(reference.action), active.token_mapping or {}
                )
                reference_head = reference_tokens[0] if reference_tokens else ""
                head_match = float(bool(candidate_head and candidate_head == reference_head))
                action_similarity = _weighted_jaccard(candidate_tokens, reference_tokens, self.idf)
                observation_similarity = _weighted_jaccard(
                    observation_tokens,
                    _substitute_tokens(
                        text_tokens(reference.observation), active.token_mapping or {}
                    ),
                    self.idf,
                )
                distance = index - active.cursor
                alignment = (
                    0.55 * head_match
                    + 0.35 * action_similarity
                    + 0.10 * observation_similarity
                ) * (0.88**distance)
                best = max(best, alignment)
            supports.append((best, active.goal_similarity * max(0.0, active.plan.utility)))
        supports.sort(key=lambda item: item[0] * item[1], reverse=True)
        strongest = supports[:8]
        denominator = sum(weight for _, weight in strongest)
        return (
            sum(score * weight for score, weight in strongest) / denominator
            if denominator > 0.0
            else 0.0
        )

    def milestone_rank(self, *, candidates: Iterable[str]) -> list[dict[str, Any]]:
        rows = [
            {"candidate": candidate, "score": self.milestone_score(candidate)}
            for candidate in candidates
        ]
        rows.sort(key=lambda item: (item["score"], item["candidate"]), reverse=True)
        return rows

    def milestone_score(self, candidate: str) -> float:
        candidate_tokens = text_tokens(candidate)
        candidate_head = candidate_tokens[0] if candidate_tokens else ""
        if self._obligation_veto(candidate):
            return 0.0
        goal_precision = _primary_argument_precision(
            candidate_tokens, self.goal_tokens, self.idf
        )
        supports: list[tuple[float, float]] = []
        for active in self.active:
            if active.milestone_cursor >= len(active.milestones):
                continue
            index = active.milestones[active.milestone_cursor]
            reference_tokens = _substitute_tokens(
                text_tokens(active.plan.steps[index].action), active.token_mapping or {}
            )
            reference_head = reference_tokens[0] if reference_tokens else ""
            if not candidate_head or candidate_head != reference_head:
                continue
            alignment = (
                0.40
                + 0.60 * _weighted_jaccard(candidate_tokens, reference_tokens, self.idf)
            )
            binding_probability = self.head_goal_binding.get(candidate_head, 0.0)
            goal_factor = goal_precision if binding_probability >= 0.55 else 1.0
            score = alignment * goal_factor
            supports.append((score, active.goal_similarity * max(0.0, active.plan.utility)))
        supports.sort(key=lambda item: item[0] * item[1], reverse=True)
        strongest = supports[:8]
        denominator = sum(weight for _, weight in strongest)
        aligned = (
            sum(score * weight for score, weight in strongest) / denominator
            if denominator > 0.0
            else 0.0
        )
        return max(aligned, self._obligation_score(candidate))

    def _obligation_veto(self, candidate: str) -> bool:
        frame = _action_frame(candidate)
        schemas = [
            active.schema
            for active in self.active
            if active.schema and active.schema.object_type
        ]
        object_types = {schema.object_type for schema in schemas}
        goal_entities = [
            entity for entity in frame.entities if entity.entity_type in object_types
        ]
        if any(entity.key in self.completed_goal_entities for entity in goal_entities):
            return True
        known_relation_heads = {
            head for schema in schemas for head in schema.relation_heads
        }
        for entity in goal_entities:
            stage = self.entity_stages.get(entity.key, 0)
            expectations = [
                (schema.relation_heads[stage], schema.relation_target_types[stage])
                for schema in schemas
                if schema.object_type == entity.entity_type
                and stage < len(schema.relation_heads)
            ]
            if frame.head not in known_relation_heads or not expectations:
                continue
            if any(
                frame.head == expected_head
                and (
                    stage == 0
                    or not expected_target
                    or len(frame.entities) < 2
                    or frame.entities[1].entity_type == expected_target
                )
                for expected_head, expected_target in expectations
            ):
                continue
            return True
        return False

    def _obligation_score(self, candidate: str) -> float:
        frame = _action_frame(candidate)
        if not frame.head:
            return 0.0
        schemas = [
            active.schema
            for active in self.active
            if active.schema and active.schema.object_type
        ]
        if not schemas:
            return 0.0
        object_support = Counter(
            schema.object_type
            for schema in schemas
            if schema.object_type
        )
        object_type, _ = object_support.most_common(1)[0]
        required = max(
            schema.required_count
            for schema in schemas
            if schema.object_type == object_type
        )
        if len(self.completed_goal_entities) >= required:
            return 0.0

        referenced = {entity.key: entity for entity in frame.entities}
        visible_goal_entities = {
            key for key in self.current_entities if key.startswith(f"{object_type} ")
        }
        candidate_goal_entities = [
            entity for entity in frame.entities if entity.entity_type == object_type
        ]
        if candidate_goal_entities:
            if any(
                entity.key in self.completed_goal_entities
                for entity in candidate_goal_entities
            ):
                return 0.0
            best = 0.0
            for entity in candidate_goal_entities:
                stage = self.entity_stages.get(entity.key, 0)
                for schema in schemas:
                    if schema.object_type != object_type or stage >= len(schema.relation_heads):
                        continue
                    if frame.head != schema.relation_heads[stage]:
                        continue
                    expected_target = schema.relation_target_types[stage]
                    if (
                        stage > 0
                        and expected_target
                        and len(frame.entities) >= 2
                        and frame.entities[1].entity_type != expected_target
                    ):
                        continue
                    best = max(
                        best,
                        1.0 if entity.key in visible_goal_entities or stage > 0 else 0.82,
                    )
            if best:
                return best
            return 0.0
        pending = sorted(
            (
                (key, stage)
                for key, stage in self.entity_stages.items()
                if key.startswith(f"{object_type} ") and key not in self.completed_goal_entities
            ),
            key=lambda item: item[0],
        )
        for _, stage in pending:
            for schema in schemas:
                if schema.object_type != object_type or stage >= len(schema.relation_target_types):
                    continue
                target_type = schema.relation_target_types[stage]
                navigation_heads = (
                    set(schema.stage_navigation_heads[stage])
                    if stage < len(schema.stage_navigation_heads)
                    else set()
                )
                if (
                    frame.head in navigation_heads
                    and any(entity.entity_type == target_type for entity in frame.entities)
                ):
                    return 0.95
        if visible_goal_entities:
            return 0.0

        source_counts = self.source_type_counts.get(object_type, Counter())
        if not source_counts:
            source_counts = Counter(
                source
                for schema in schemas
                for source in schema.source_types
                if schema.object_type == object_type
            )
        total_sources = sum(source_counts.values())
        if not total_sources:
            return 0.0
        best = 0.0
        for entity in referenced.values():
            source_probability = source_counts[entity.entity_type] / total_sources
            if source_probability <= 0.0:
                continue
            head_counts = self.search_head_counts.get(
                (object_type, entity.entity_type), Counter()
            )
            if not head_counts:
                head_counts = Counter(
                    head
                    for schema in schemas
                    if schema.object_type == object_type
                    and entity.entity_type in schema.source_types
                    for head in schema.search_heads
                )
            head_probability = (
                head_counts[frame.head] / sum(head_counts.values())
                if head_counts and sum(head_counts.values())
                else 0.25
            )
            novelty = math.exp(-1.5 * self.executed_actions[candidate])
            best = max(
                best,
                novelty * (0.35 + 0.40 * source_probability + 0.25 * head_probability),
            )
        return min(1.0, best)

    def goal_binding_score(self, candidate: str) -> float:
        """Return learned compatibility between an action's arguments and the goal.

        Action families whose demonstrated arguments almost always bind to a goal
        variable are suppressed when a candidate contains no goal-bound argument.
        Other action families remain unconstrained for exploration and navigation.
        """
        candidate_tokens = text_tokens(candidate)
        if not candidate_tokens:
            return 0.0
        binding_probability = self.head_goal_binding.get(candidate_tokens[0], 0.0)
        if binding_probability < 0.55:
            return 1.0
        precision = _primary_argument_precision(
            candidate_tokens, self.goal_tokens, self.idf
        )
        return (1.0 - binding_probability) + binding_probability * precision

    def goal_argument_score(self, candidate: str) -> float:
        return _primary_argument_precision(
            text_tokens(candidate), self.goal_tokens, self.idf
        )

    def goal_binding_probability(self, candidate: str) -> float:
        candidate_tokens = text_tokens(candidate)
        return (
            self.head_goal_binding.get(candidate_tokens[0], 0.0)
            if candidate_tokens
            else 0.0
        )

    def completion_probability(self, candidate: str) -> float:
        candidate_tokens = text_tokens(candidate)
        return (
            self.head_terminal_probability.get(candidate_tokens[0], 0.0)
            if candidate_tokens
            else 0.0
        )

    def advance(self, action: str, *, lookahead: int = 12) -> None:
        self.executed_actions[action] += 1
        frame = _action_frame(action)
        action_tokens = text_tokens(action)
        action_head = action_tokens[0] if action_tokens else ""
        for active in self.active:
            schema = active.schema
            if not schema or not schema.object_type:
                continue
            goal_entities = [
                entity for entity in frame.entities if entity.entity_type == schema.object_type
            ]
            if not goal_entities:
                continue
            entity = goal_entities[0]
            stage = self.entity_stages.get(entity.key, 0)
            if stage >= len(schema.relation_heads) or frame.head != schema.relation_heads[stage]:
                continue
            expected_target = schema.relation_target_types[stage]
            if (
                stage > 0
                and expected_target
                and len(frame.entities) >= 2
                and frame.entities[1].entity_type != expected_target
            ):
                continue
            stage += 1
            self.entity_stages[entity.key] = stage
            if stage == 1:
                self.acquired_goal_entities.add(entity.key)
            if stage >= len(schema.relation_heads):
                self.completed_goal_entities.add(entity.key)
                self.acquired_goal_entities.discard(entity.key)
        for active in self.active:
            best_index = None
            best_score = 0.0
            stop = min(len(active.plan.steps), active.cursor + max(1, int(lookahead)))
            for index in range(active.cursor, stop):
                reference_tokens = _substitute_tokens(
                    text_tokens(active.plan.steps[index].action), active.token_mapping or {}
                )
                reference_head = reference_tokens[0] if reference_tokens else ""
                score = (
                    0.65 * float(bool(action_head and action_head == reference_head))
                    + 0.35 * _weighted_jaccard(action_tokens, reference_tokens, self.idf)
                ) * (0.94 ** (index - active.cursor))
                if score > best_score:
                    best_index, best_score = index, score
            if best_index is not None and best_score >= 0.45:
                active.cursor = min(len(active.plan.steps), best_index + 1)
            if active.milestone_cursor < len(active.milestones):
                milestone_index = active.milestones[active.milestone_cursor]
                milestone_tokens = _substitute_tokens(
                    text_tokens(active.plan.steps[milestone_index].action),
                    active.token_mapping or {},
                )
                milestone_head = milestone_tokens[0] if milestone_tokens else ""
                milestone_score = (
                    0.75
                    + 0.25 * _weighted_jaccard(action_tokens, milestone_tokens, self.idf)
                    if action_head and action_head == milestone_head
                    else 0.0
                )
                goal_precision = _primary_argument_precision(
                    action_tokens, self.goal_tokens, self.idf
                )
                binding_required = self.head_goal_binding.get(action_head, 0.0) >= 0.55
                if (
                    milestone_score >= 0.62
                    and (not binding_required or goal_precision > 0.08)
                ):
                    active.milestone_cursor += 1

    def diagnostics(self) -> dict[str, Any]:
        return {
            "retrieved_plans": len(self.active),
            "plans": [
                {
                    "plan_id": active.plan.plan_id,
                    "goal_similarity": round(active.goal_similarity, 6),
                    "cursor": active.cursor,
                    "length": len(active.plan.steps),
                    "milestone_cursor": active.milestone_cursor,
                    "milestone_count": len(active.milestones),
                    "required_count": active.schema.required_count if active.schema else 1,
                    "object_type": active.schema.object_type if active.schema else None,
                    "destination_type": active.schema.destination_type if active.schema else None,
                }
                for active in self.active[:8]
            ],
            "completed_goal_entities": sorted(self.completed_goal_entities),
            "acquired_goal_entities": sorted(self.acquired_goal_entities),
            "entity_stages": dict(sorted(self.entity_stages.items())),
        }


def _weighted_jaccard(
    left: tuple[str, ...],
    right: tuple[str, ...],
    weights: dict[str, float],
) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    numerator = sum(weights.get(token, 1.0) for token in left_set & right_set)
    denominator = sum(weights.get(token, 1.0) for token in union)
    return numerator / denominator if denominator else 0.0


def _weighted_precision(
    values: tuple[str, ...],
    reference: tuple[str, ...],
    weights: dict[str, float],
) -> float:
    value_set = set(values)
    if not value_set:
        return 0.0
    reference_set = set(reference)
    numerator = sum(weights.get(token, 1.0) for token in value_set & reference_set)
    denominator = sum(weights.get(token, 1.0) for token in value_set)
    return numerator / denominator if denominator else 0.0


def _primary_argument_precision(
    action: tuple[str, ...],
    goal: tuple[str, ...],
    weights: dict[str, float],
) -> float:
    argument = next((token for token in action[1:] if not token.isdigit()), "")
    if not argument:
        return 0.0
    return float(argument in set(goal))


def _token_mapping(
    source: tuple[str, ...],
    target: tuple[str, ...],
) -> dict[str, str]:
    """Infer variable bindings between two analogous token sequences."""
    mapping: dict[str, str] = {}
    matcher = SequenceMatcher(a=source, b=target, autojunk=False)
    for tag, left_start, left_stop, right_start, right_stop in matcher.get_opcodes():
        if tag != "replace":
            continue
        left = source[left_start:left_stop]
        right = target[right_start:right_stop]
        if len(left) == len(right):
            mapping.update(zip(left, right))
        elif len(left) == 1 and right:
            mapping[left[0]] = right[-1]
        elif len(right) == 1 and left:
            for token in left:
                mapping[token] = right[0]
    return mapping


def _substitute_tokens(
    values: tuple[str, ...],
    mapping: dict[str, str],
) -> tuple[str, ...]:
    return tuple(mapping.get(value, value) for value in values)


def _goal_quantity(goal: str) -> int:
    values = text_tokens(goal, keep_numbers=True)
    for value in values:
        if value.isdigit():
            return max(1, int(value))
        if value in _QUANTITIES:
            return _QUANTITIES[value]
    return 1


def _entity_refs(value: str) -> tuple[EntityRef, ...]:
    values = text_tokens(value, keep_numbers=True)
    return tuple(
        EntityRef(entity_type=values[index - 1], entity_id=token)
        for index, token in enumerate(values)
        if index > 0 and token.isdigit()
    )


def _action_frame(action: str) -> ActionFrame:
    values = text_tokens(action, keep_numbers=True)
    return ActionFrame(
        head=values[0] if values else "",
        entities=_entity_refs(action),
        tokens=values,
    )


def _induce_plan_schema(
    plan: EpisodicPlan,
) -> tuple[_PlanSchema, list[tuple[str, tuple[str, ...]]]]:
    goal_tokens = set(text_tokens(plan.goal))
    frames = [_action_frame(step.action) for step in plan.steps]
    relation_rows: dict[str, list[tuple[int, ActionFrame]]] = defaultdict(list)
    object_types: Counter[str] = Counter()
    for index, frame in enumerate(frames):
        if len(frame.entities) < 2:
            continue
        primary = frame.entities[0]
        if primary.entity_type not in goal_tokens:
            continue
        relation_rows[primary.key].append((index, frame))
        object_types[primary.entity_type] += 1
    if not object_types:
        return (
            _PlanSchema(None, None, _goal_quantity(plan.goal), (), (), (), (), (), (), (), ()),
            [],
        )

    object_type = object_types.most_common(1)[0][0]
    destination_types: Counter[str] = Counter()
    sources: list[tuple[str, tuple[str, ...]]] = []
    search_heads: Counter[str] = Counter()
    acquisition_heads: Counter[str] = Counter()
    completion_heads: Counter[str] = Counter()
    destination_heads: Counter[str] = Counter()
    relation_sequences: Counter[tuple[tuple[str, str], ...]] = Counter()
    navigation_by_stage: dict[int, Counter[str]] = defaultdict(Counter)
    distinct_objects = 0
    for object_key, relations in relation_rows.items():
        if not object_key.startswith(f"{object_type} "):
            continue
        distinct_objects += 1
        relations.sort(key=lambda item: item[0])
        relation_sequences.update([
            tuple((frame.head, frame.entities[1].entity_type) for _, frame in relations)
        ])
        first_index, first_frame = relations[0]
        last_index, last_frame = relations[-1]
        acquisition_heads[first_frame.head] += 1
        completion_heads[last_frame.head] += 1
        source = first_frame.entities[1]
        destination = last_frame.entities[1]
        destination_types[destination.entity_type] += 1
        preceding: list[str] = []
        for previous in frames[max(0, first_index - 3) : first_index]:
            if any(
                entity.key == source.key or entity.entity_type == source.entity_type
                for entity in previous.entities
            ):
                preceding.append(previous.head)
                search_heads[previous.head] += 1
        if not preceding:
            preceding.append(first_frame.head)
            search_heads[first_frame.head] += 1
        sources.append((source.entity_type, tuple(preceding)))
        previous_relation_index = -1
        for stage, (relation_index, relation_frame) in enumerate(relations):
            target = relation_frame.entities[1]
            start = max(previous_relation_index + 1, relation_index - 3)
            for previous in frames[start:relation_index]:
                if any(
                    entity.key == target.key or entity.entity_type == target.entity_type
                    for entity in previous.entities
                ):
                    navigation_by_stage[stage][previous.head] += 1
                    if stage == len(relations) - 1:
                        destination_heads[previous.head] += 1
            previous_relation_index = relation_index

    destination_type = None
    if destination_types:
        goal_destinations = Counter(
            {
                key: count
                for key, count in destination_types.items()
                if key in goal_tokens
            }
        )
        destination_type = (
            goal_destinations.most_common(1)[0][0]
            if goal_destinations
            else destination_types.most_common(1)[0][0]
        )
    required_count = max(_goal_quantity(plan.goal), distinct_objects, 1)
    relation_sequence = relation_sequences.most_common(1)[0][0] if relation_sequences else ()
    return (
        _PlanSchema(
            object_type=object_type,
            destination_type=destination_type,
            required_count=required_count,
            source_types=tuple(source for source, _ in sources),
            search_heads=tuple(search_heads.elements()),
            acquisition_heads=tuple(acquisition_heads.elements()),
            completion_heads=tuple(completion_heads.elements()),
            destination_heads=tuple(destination_heads.elements()),
            relation_heads=tuple(head for head, _ in relation_sequence),
            relation_target_types=tuple(target for _, target in relation_sequence),
            stage_navigation_heads=tuple(
                tuple(navigation_by_stage[index].elements())
                for index in range(len(relation_sequence))
            ),
        ),
        sources,
    )
