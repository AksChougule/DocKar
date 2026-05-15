from dockar.prompt_engine import PromptBuilder, PromptGenerator, PromptMutator


def schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "invoice_id": {"type": "string"},
            "total": {"type": "number"},
        },
        "required": ["invoice_id"],
    }


def examples() -> list[dict[str, object]]:
    return [
        {"invoice_id": "INV-001", "total": 10.5},
        {"invoice_id": "INV-002", "total": 20.0},
        {"invoice_id": "INV-003", "total": 30.0},
        {"invoice_id": "INV-004", "total": 40.0},
    ]


def test_prompt_builder_creates_json_only_base_prompt() -> None:
    prompt = PromptBuilder().build(
        task_description="Extract invoice fields.",
        schema=schema(),
        examples=examples(),
    )

    assert "Extract invoice fields." in prompt
    assert "## Output Schema" in prompt
    assert "Return only valid JSON." in prompt
    assert "Do not include markdown fences" in prompt
    assert "{{document_text}}" in prompt
    assert "Example 3" in prompt
    assert "Example 4" not in prompt


def test_prompt_generator_is_deterministic() -> None:
    generator = PromptGenerator(deterministic=True)

    first = generator.generate(
        task_description="Extract invoice fields.",
        schema=schema(),
        examples=examples(),
        candidate_count=5,
    )
    second = generator.generate(
        task_description="Extract invoice fields.",
        schema=schema(),
        examples=examples(),
        candidate_count=5,
    )

    assert [candidate.text for candidate in first] == [candidate.text for candidate in second]
    assert [candidate.id for candidate in first] == [
        "prompt-0001",
        "prompt-0002",
        "prompt-0003",
        "prompt-0004",
        "prompt-0005",
    ]


def test_prompt_generator_creates_multiple_mutation_strategies() -> None:
    candidates = PromptGenerator(deterministic=True).generate(
        task_description="Extract invoice fields.",
        schema=schema(),
        examples=examples(),
        candidate_count=6,
    )

    strategies = [candidate.metadata["strategy"] for candidate in candidates]

    assert strategies == [
        "base",
        "rephrase_instructions",
        "reorder_sections",
        "add_constraints",
        "remove_constraints",
        "vary_examples",
    ]
    assert "careful document extraction auditor" in candidates[1].text
    assert candidates[2].text.index("## Output Schema") < candidates[2].text.index("## Task")
    assert "strongest textual support" in candidates[3].text
    assert "Use null when" not in candidates[4].text


def test_prompt_mutator_varies_examples() -> None:
    mutator = PromptMutator(deterministic=True)
    mutation = mutator.mutations(candidate_count=6, example_count=3)[5]

    selected = mutator.select_examples(examples()[:3], mutation)

    assert [example["invoice_id"] for example in selected] == ["INV-003", "INV-002", "INV-001"]


def test_prompt_generator_rejects_invalid_candidate_count() -> None:
    generator = PromptGenerator()

    try:
        generator.generate("task", schema(), candidate_count=0)
    except ValueError as exc:
        assert "candidate_count" in str(exc)
    else:
        raise AssertionError("expected candidate_count validation error")
