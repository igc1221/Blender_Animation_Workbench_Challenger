from __future__ import annotations

from dataclasses import dataclass

from .character_metadata import (
    CharacterMetadataError,
    UnsupportedCharacterSchema,
    character_ids,
    character_lifecycle_issues,
    resolve_character,
)
from .rigped_contract import RIGPED_SETUP_PROPERTY
from .semantic_adapter import control_context_for_context, runtime_control_key
from .semantic_model import AWBControlKind


@dataclass(frozen=True, slots=True)
class ActiveCharacterAssignment:
    character_id: str
    character_label: str
    binding_id: str
    semantic_key: str
    side: str
    mode: str
    usage: str


def active_character_assignment(context) -> ActiveCharacterAssignment | None:
    """Return the unique Character binding for Blender's native active selected control."""

    active = control_context_for_context(context).active
    if active is None:
        return None
    active_key = runtime_control_key(active)
    matches: list[ActiveCharacterAssignment] = []
    for character_id in character_ids(context.scene):
        view = resolve_character(context.scene, character_id)
        binding_by_id = {binding.binding_id: binding for binding in view.definition.bindings}
        for binding_id, resolved in view.resolved_bindings:
            if runtime_control_key(resolved) != active_key:
                continue
            binding = binding_by_id.get(binding_id)
            if binding is None:
                continue
            matches.append(
                ActiveCharacterAssignment(
                    character_id=view.definition.character_id,
                    character_label=view.definition.label,
                    binding_id=binding.binding_id,
                    semantic_key=binding.semantic_key,
                    side=binding.side.value,
                    mode=binding.mode.value,
                    usage=binding.usage.value,
                )
            )
    return matches[0] if len(matches) == 1 else None


def _binding_label(view, binding_id: str) -> str:
    resolved = next(
        (target for candidate_id, target in view.resolved_bindings if candidate_id == binding_id),
        None,
    )
    if resolved is None:
        return f"Missing: {binding_id[:8]}"
    if resolved.control.kind == AWBControlKind.OBJECT:
        return resolved.control.object_name
    return resolved.control.bone_name


def _character_id_for_active_owner(context, character_ids_in_scene: tuple[str, ...]) -> str | None:
    active_object = getattr(context, "active_object", None)
    store = getattr(getattr(context, "scene", None), "awb_characters", None)
    if active_object is None or store is None:
        return None
    allowed = set(character_ids_in_scene)
    matches = tuple(
        str(row.character_id)
        for row in store.characters
        if str(row.character_id) in allowed
        and any(owner.object_ref is active_object for owner in row.owners)
    )
    return matches[0] if len(matches) == 1 else None


def _draw_group_chain_editor(layout, context, character_id: str) -> None:
    try:
        view = resolve_character(context.scene, character_id)
    except CharacterMetadataError as exc:
        layout.label(text=str(exc), icon="ERROR")
        return

    group_header, group_body = layout.panel("BAW_UI_character_groups", default_closed=True)
    group_header.label(text=f"Groups  {len(view.definition.groups)}")
    create_group = group_header.operator(
        "baw.create_character_group_from_selection",
        text="",
        icon="ADD",
    )
    create_group.character_id = character_id
    if group_body is not None:
        for index, group in enumerate(view.definition.groups):
            row = group_body.row(align=True)
            title = group.label or group.semantic_key or "Group"
            row.label(text=f"{title} · {group.side.value}")
            select_op = row.operator("baw.select_character_group", text="Select")
            select_op.character_id = character_id
            select_op.group_id = group.group_id
            select_op.operation = "REPLACE"
            edit_op = row.operator("baw.edit_character_group", text="Edit")
            edit_op.character_id = character_id
            edit_op.group_id = group.group_id
            members_op = row.operator(
                "baw.set_character_group_members_from_selection",
                text="Members",
            )
            members_op.character_id = character_id
            members_op.group_id = group.group_id
            remove_op = row.operator("baw.remove_character_group", text="X")
            remove_op.character_id = character_id
            remove_op.group_id = group.group_id
            members = ", ".join(_binding_label(view, item) for item in group.members)
            group_body.label(text=members or "—")
            if index < len(view.definition.groups) - 1:
                group_body.separator()

    chain_header, chain_body = layout.panel("BAW_UI_character_chains", default_closed=True)
    chain_header.label(text=f"Chains  {len(view.definition.chains)}")
    create_chain = chain_header.operator(
        "baw.create_character_chain_from_selection",
        text="",
        icon="ADD",
    )
    create_chain.character_id = character_id
    if chain_body is not None:
        for chain_index, chain in enumerate(view.definition.chains):
            row = chain_body.row(align=True)
            title = chain.label or chain.semantic_key or "Chain"
            row.label(text=f"{title} · {chain.side.value}/{chain.mode.value}")
            select_op = row.operator("baw.select_character_chain", text="Select")
            select_op.character_id = character_id
            select_op.chain_id = chain.chain_id
            select_op.operation = "REPLACE"
            edit_op = row.operator("baw.edit_character_chain", text="Edit")
            edit_op.character_id = character_id
            edit_op.chain_id = chain.chain_id
            reset_op = row.operator(
                "baw.set_character_chain_members_from_selection",
                text="Members",
            )
            reset_op.character_id = character_id
            reset_op.chain_id = chain.chain_id
            map_op = row.operator("baw.create_kinematic_mapping", text="FK Map")
            map_op.character_id = character_id
            map_op.fk_chain_id = chain.chain_id
            remove_op = row.operator("baw.remove_character_chain", text="X")
            remove_op.character_id = character_id
            remove_op.chain_id = chain.chain_id

            for member_index, binding_id in enumerate(chain.members):
                member_row = chain_body.row(align=True)
                member_row.label(text=f"{member_index + 1}  {_binding_label(view, binding_id)}")
                up = member_row.row(align=True)
                up.enabled = member_index > 0
                up_op = up.operator("baw.move_character_chain_member", text="", icon="TRIA_UP")
                up_op.character_id = character_id
                up_op.chain_id = chain.chain_id
                up_op.member_index = member_index
                up_op.direction = "UP"
                down = member_row.row(align=True)
                down.enabled = member_index < len(chain.members) - 1
                down_op = down.operator("baw.move_character_chain_member", text="", icon="TRIA_DOWN")
                down_op.character_id = character_id
                down_op.chain_id = chain.chain_id
                down_op.member_index = member_index
                down_op.direction = "DOWN"
            if chain_index < len(view.definition.chains) - 1:
                chain_body.separator()


def _draw_opposite_kinematic_editor(layout, context, character_id: str, active_assignment) -> None:
    try:
        view = resolve_character(context.scene, character_id)
    except CharacterMetadataError as exc:
        layout.label(text=str(exc), icon="ERROR")
        return

    opposite_header, opposite_body = layout.panel(
        "BAW_UI_character_opposites",
        default_closed=True,
    )
    opposite_header.label(text=f"Opposites  {len(view.definition.opposites)}")
    create_pair = opposite_header.operator(
        "baw.create_opposite_from_selection",
        text="",
        icon="ARROW_LEFTRIGHT",
    )
    create_pair.character_id = character_id
    if opposite_body is not None:
        for pair in view.definition.opposites:
            row = opposite_body.row(align=True)
            row.label(
                text=(
                    f"{_binding_label(view, pair.left_binding_id)}  ↔  "
                    f"{_binding_label(view, pair.right_binding_id)}"
                )
            )
            remove = row.operator("baw.remove_opposite_pair", text="X")
            remove.character_id = character_id
            remove.left_binding_id = pair.left_binding_id
            remove.right_binding_id = pair.right_binding_id

    kine_header, kine_body = layout.panel(
        "BAW_UI_character_kinematics",
        default_closed=True,
    )
    kine_header.label(text=f"Kinematics  {len(view.definition.kinematics)}")
    if active_assignment is not None and active_assignment.character_id == character_id:
        create_ik = kine_header.operator("baw.create_kinematic_mapping", text="IK +")
        create_ik.character_id = character_id
        create_ik.ik_target_binding_id = active_assignment.binding_id
    if kine_body is None:
        return

    chain_by_id = {chain.chain_id: chain for chain in view.definition.chains}
    for mapping_index, mapping in enumerate(view.definition.kinematics):
        mapping_box = kine_body.box()
        row = mapping_box.row(align=True)
        row.label(text=f"{mapping.semantic_key} · {mapping.side.value}")
        select_op = row.operator("baw.select_character_kinematic", text="Select")
        select_op.character_id = character_id
        select_op.mapping_id = mapping.mapping_id
        select_op.operation = "REPLACE"
        edit_op = row.operator("baw.edit_kinematic_mapping", text="Edit")
        edit_op.character_id = character_id
        edit_op.mapping_id = mapping.mapping_id
        remove_op = row.operator("baw.remove_kinematic_mapping", text="X")
        remove_op.character_id = character_id
        remove_op.mapping_id = mapping.mapping_id

        fk_chain = chain_by_id.get(mapping.fk_chain_id or "")
        fk_row = mapping_box.row(align=True)
        fk_row.label(text="FK")
        fk_row.label(text=fk_chain.label if fk_chain else "—")
        if mapping.fk_chain_id:
            clear = fk_row.operator("baw.clear_kinematic_role", text="Clear")
            clear.character_id = character_id
            clear.mapping_id = mapping.mapping_id
            clear.role = "FK_CHAIN"

        ik_row = mapping_box.row(align=True)
        ik_row.label(text="IK")
        ik_row.label(
            text=(
                _binding_label(view, mapping.ik_target_binding_id)
                if mapping.ik_target_binding_id
                else "—"
            )
        )
        if active_assignment is not None and active_assignment.character_id == character_id:
            use_active = ik_row.operator("baw.set_kinematic_active_role", text="Active")
            use_active.character_id = character_id
            use_active.mapping_id = mapping.mapping_id
            use_active.role = "IK_TARGET"
        if mapping.ik_target_binding_id:
            clear = ik_row.operator("baw.clear_kinematic_role", text="Clear")
            clear.character_id = character_id
            clear.mapping_id = mapping.mapping_id
            clear.role = "IK_TARGET"

        pole_row = mapping_box.row(align=True)
        pole_row.label(text="Pole")
        pole_row.label(
            text=(
                _binding_label(view, mapping.pole_binding_id)
                if mapping.pole_binding_id
                else "—"
            )
        )
        if active_assignment is not None and active_assignment.character_id == character_id:
            use_active = pole_row.operator("baw.set_kinematic_active_role", text="Active")
            use_active.character_id = character_id
            use_active.mapping_id = mapping.mapping_id
            use_active.role = "POLE"
        if mapping.pole_binding_id:
            clear = pole_row.operator("baw.clear_kinematic_role", text="Clear")
            clear.character_id = character_id
            clear.mapping_id = mapping.mapping_id
            clear.role = "POLE"

        reference_chain = chain_by_id.get(mapping.reference_chain_id or "")
        ref_row = mapping_box.row(align=True)
        ref_row.label(text="Ref")
        ref_row.label(text=reference_chain.label if reference_chain else "—")
        if mapping.reference_chain_id:
            clear = ref_row.operator("baw.clear_kinematic_role", text="Clear")
            clear.character_id = character_id
            clear.mapping_id = mapping.mapping_id
            clear.role = "REFERENCE_CHAIN"

        extras_row = mapping_box.row(align=True)
        extras_text = ", ".join(_binding_label(view, item) for item in mapping.extras)
        extras_row.label(text="Extras")
        extras_row.label(text=extras_text or "—")
        set_extras = extras_row.operator("baw.set_kinematic_extras_from_selection", text="Selected")
        set_extras.character_id = character_id
        set_extras.mapping_id = mapping.mapping_id
        if mapping.extras:
            clear = extras_row.operator("baw.clear_kinematic_role", text="Clear")
            clear.character_id = character_id
            clear.mapping_id = mapping.mapping_id
            clear.role = "EXTRAS"

        for chain in view.definition.chains:
            chain_row = mapping_box.row(align=True)
            chain_row.label(text=chain.label)
            fk_op = chain_row.operator("baw.set_kinematic_chain_role", text="FK")
            fk_op.character_id = character_id
            fk_op.mapping_id = mapping.mapping_id
            fk_op.chain_id = chain.chain_id
            fk_op.role = "FK_CHAIN"
            ref_op = chain_row.operator("baw.set_kinematic_chain_role", text="Ref")
            ref_op.character_id = character_id
            ref_op.mapping_id = mapping.mapping_id
            ref_op.chain_id = chain.chain_id
            ref_op.role = "REFERENCE_CHAIN"
        if mapping_index < len(view.definition.kinematics) - 1:
            kine_body.separator()


def _draw_bone_repair_editor(layout, context, character_id: str) -> None:
    try:
        view = resolve_character(context.scene, character_id)
    except CharacterMetadataError as exc:
        layout.label(text=str(exc), icon="ERROR")
        return

    resolved_ids = {binding_id for binding_id, _target in view.resolved_bindings}
    unresolved = tuple(
        binding
        for binding in view.definition.bindings
        if binding.kind == AWBControlKind.BONE and binding.binding_id not in resolved_ids
    )
    if not unresolved:
        return

    repair_header, repair_body = layout.panel(
        "BAW_UI_character_repair",
        default_closed=False,
    )
    repair_header.label(text=f"Repair  {len(unresolved)}", icon="ERROR")
    if repair_body is None:
        return

    issue_names = {
        "MISSING_BONE_TOKEN_TARGET": "Missing",
        "AMBIGUOUS_BONE_TOKEN_TARGET": "Ambiguous",
        "MISSING_POSE_BONE_TARGET": "Missing",
        "UNRESOLVED_BONE_TARGET": "Unresolved",
    }
    issues_by_record: dict[str, list[str]] = {}
    for issue in view.issues:
        if issue.record_id:
            label = issue_names.get(issue.code, "Unresolved")
            issues_by_record.setdefault(issue.record_id, []).append(label)
    for binding in unresolved:
        row = repair_body.row(align=True)
        label = binding.bone_name_hint or binding.semantic_key or binding.binding_id[:8]
        state = "/".join(dict.fromkeys(issues_by_record.get(binding.binding_id, ())))
        row.label(text=f"{label} · {state or 'Unresolved'}", icon="ERROR")
        repair = row.operator("baw.repair_character_bone_binding", text="Repair")
        repair.character_id = character_id
        repair.binding_id = binding.binding_id


_LIFECYCLE_ISSUE_LABELS = {
    "UNSUPPORTED_CHARACTER_SCHEMA": "Unsupported Schema",
    "LEGACY_CHARACTER_SCHEMA": "Legacy Schema",
    "READ_ONLY_CHARACTER_SCENE": "Read-only Scene",
    "MISSING_LIBRARY_OWNER": "Missing Owner",
    "LINKED_OWNER_READ_ONLY": "Linked Owner",
    "NON_EDITABLE_OVERRIDE_OWNER": "Read-only Override",
    "READ_ONLY_OWNER": "Read-only Owner",
    "MISSING_LIBRARY_ARMATURE_DATA": "Missing Rig Data",
    "LINKED_ARMATURE_DATA_READ_ONLY": "Linked Rig Data",
    "NON_EDITABLE_OVERRIDE_ARMATURE_DATA": "Read-only Rig Override",
    "SHARED_ARMATURE_DATA_TOKEN_WRITE_BLOCKED": "Shared Rig Data",
    "MISSING_OBJECT_TARGET": "Missing Target",
    "CHARACTER_DECODE_ERROR": "Invalid Metadata",
    "CHARACTER_RESOLUTION_ERROR": "Resolve Error",
}


def _draw_lifecycle_issues(layout, issues) -> None:
    if not issues:
        return
    header, body = layout.panel("BAW_UI_character_lifecycle_issues", default_closed=True)
    header.label(text=f"Issues  {len(issues)}", icon="ERROR")
    if any(issue.code == "LEGACY_CHARACTER_SCHEMA" for issue in issues):
        header.operator("baw.migrate_character_schema_v2", text="Migrate")
    if body is None:
        return
    for issue in issues:
        label = _LIFECYCLE_ISSUE_LABELS.get(issue.code, issue.code.replace("_", " ").title())
        icon = "ERROR" if "MISSING" in issue.code or "UNSUPPORTED" in issue.code else "INFO"
        row = body.row(align=True)
        row.label(text=label, icon=icon)


def draw_character_assignment(layout, context) -> None:
    active = getattr(context, "active_object", None)
    if active is not None and hasattr(active, "get"):
        raw = active.get(RIGPED_SETUP_PROPERTY)
        getter = getattr(raw, "get", None)
        if callable(getter) and str(getter("character_id", "") or ""):
            row = layout.row(align=True)
            row.label(text="Rigped Character", icon="OUTLINER_OB_ARMATURE")
            row.label(text="Semantic graph managed by AWB")
            return

    try:
        lifecycle_issues = character_lifecycle_issues(context.scene)
    except CharacterMetadataError as exc:
        layout.label(text=str(exc), icon="ERROR")
        return
    _draw_lifecycle_issues(layout, lifecycle_issues)

    try:
        ids = character_ids(context.scene)
        active_assignment = active_character_assignment(context)
    except UnsupportedCharacterSchema:
        return
    except CharacterMetadataError as exc:
        layout.label(text=str(exc), icon="ERROR")
        return

    layout.label(text="Character")
    active = control_context_for_context(context).active
    if active_assignment is not None:
        status = layout.row(align=True)
        status.label(text=active_assignment.character_label, icon="OUTLINER_OB_ARMATURE")
        semantic = active_assignment.semantic_key or "Unassigned"
        status.label(
            text=(
                f"{semantic} · {active_assignment.side} · "
                f"{active_assignment.mode} · {active_assignment.usage}"
            )
        )
        edit_op = status.operator("baw.assign_character_semantics", text="Semantics")
        edit_op.character_id = active_assignment.character_id
        edit_op.binding_id = active_assignment.binding_id
        remove_op = status.operator("baw.remove_character_binding", text="Unbind")
        remove_op.character_id = active_assignment.character_id
        remove_op.binding_id = active_assignment.binding_id
    elif active is not None:
        layout.label(text="Unassigned", icon="QUESTION")

    chars_header, chars_body = layout.panel("BAW_UI_characters", default_closed=False)
    chars_header.label(text=f"Characters  {len(ids)}")
    chars_header.operator("baw.create_character", text="", icon="ADD")
    if chars_body is not None:
        for character_id in ids:
            try:
                view = resolve_character(context.scene, character_id)
            except CharacterMetadataError:
                continue
            row = chars_body.row(align=True)
            row.label(text=view.definition.label or "Character", icon="OUTLINER_OB_ARMATURE")
            select_op = row.operator("baw.select_character", text="Select")
            select_op.character_id = character_id
            select_op.operation = "REPLACE"
            clone_op = row.operator("baw.clone_character", text="Clone")
            clone_op.source_character_id = character_id
            if active_assignment is None and active is not None:
                bind_op = row.operator("baw.bind_character_active", text="Bind")
                bind_op.character_id = character_id

    if not ids:
        return

    owner_character_id = _character_id_for_active_owner(context, ids)
    editor_character_id = (
        active_assignment.character_id
        if active_assignment is not None
        else owner_character_id or (ids[0] if len(ids) == 1 else None)
    )
    if editor_character_id is None:
        return

    _draw_group_chain_editor(layout, context, editor_character_id)
    _draw_opposite_kinematic_editor(
        layout,
        context,
        editor_character_id,
        active_assignment,
    )
    _draw_bone_repair_editor(layout, context, editor_character_id)
