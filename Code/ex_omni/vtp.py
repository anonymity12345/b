"""Parse the paper-defined Visual Thought Plan (VTP) for video conditioning."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re


VTP_FIELDS = (
    "first_frame_scene",
    "scene",
    "emotion",
    "movement_style",
    "motion_description",
)

EMOTION_LABELS = frozenset(
    {
        "neutral",
        "happy",
        "sad",
        "angry",
        "surprised",
        "fearful",
        "disgusted",
        "calm",
        "concerned",
        "confident",
        "excited",
        "nervous",
        "thoughtful",
        "engaged",
        "amused",
    }
)


@dataclass(frozen=True)
class VisualThoughtPlan:
    first_frame_scene: str
    scene: str
    emotion: str
    movement_style: str
    motion_description: str

    @classmethod
    def parse(cls, value: str, *, strict: bool = True) -> "VisualThoughtPlan":
        text = (value or "").strip()
        outer = re.fullmatch(
            r"\s*<avatar_plan\s*>(.*?)</avatar_plan\s*>\s*",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if outer:
            text = outer.group(1)
        elif strict and re.search(r"</?avatar_plan\b", text, flags=re.IGNORECASE):
            raise ValueError("malformed <avatar_plan> wrapper")

        parsed: dict[str, str] = {}
        for field_name in VTP_FIELDS:
            matches = re.findall(
                rf"<{field_name}\s*>(.*?)</{field_name}\s*>",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if len(matches) != 1:
                if strict:
                    raise ValueError(
                        f"VTP requires exactly one <{field_name}> tag; found {len(matches)}"
                    )
                parsed[field_name] = matches[0].strip() if matches else ""
            else:
                parsed[field_name] = matches[0].strip()
            if strict and not parsed[field_name]:
                raise ValueError(f"<{field_name}> cannot be empty")

        if strict:
            scrubbed = text
            for field_name in VTP_FIELDS:
                scrubbed = re.sub(
                    rf"<{field_name}\s*>.*?</{field_name}\s*>",
                    "",
                    scrubbed,
                    count=1,
                    flags=re.IGNORECASE | re.DOTALL,
                )
            if scrubbed.strip():
                raise ValueError(f"unexpected VTP content: {scrubbed.strip()!r}")
        return cls(**parsed)

    def to_xml(self, *, wrapped: bool = False) -> str:
        body = "\n".join(
            f"<{name}>{getattr(self, name)}</{name}>" for name in VTP_FIELDS
        )
        return f"<avatar_plan>\n{body}\n</avatar_plan>" if wrapped else body

    def to_video_prompt(self) -> str:
        return f"<caption>{self.to_xml()}</caption>"


def vtp_to_video_prompt(value: str | VisualThoughtPlan, *, strict: bool = True) -> str:
    vtp = value if isinstance(value, VisualThoughtPlan) else VisualThoughtPlan.parse(value, strict=strict)
    return vtp.to_video_prompt()


def apply_vtp_overrides(
    value: str,
    *,
    emotion: str | None = None,
    movement_style: str | None = None,
    strict: bool = True,
) -> str:
    """Replace user-controlled VTP fields after model generation."""
    selected_emotion = str(emotion or "").strip().lower()
    selected_movement = str(movement_style or "").strip()
    if selected_emotion == "auto":
        selected_emotion = ""
    if selected_movement.lower() == "auto":
        selected_movement = ""
    if not selected_emotion and not selected_movement:
        return value

    if selected_emotion:
        states = [
            label.strip()
            for transition in selected_emotion.split("->")
            for label in transition.split(",")
        ]
        invalid = [
            label for label in states if label not in EMOTION_LABELS
        ]
        if invalid:
            raise ValueError(
                "unsupported emotion label(s): " + ", ".join(invalid)
            )
    if selected_movement and (
        "<" in selected_movement or ">" in selected_movement
    ):
        raise ValueError("movement_style cannot contain XML tags")

    plan = VisualThoughtPlan.parse(value, strict=strict)
    effective = replace(
        plan,
        emotion=selected_emotion or plan.emotion,
        movement_style=selected_movement or plan.movement_style,
    )
    return effective.to_xml(wrapped=True)
