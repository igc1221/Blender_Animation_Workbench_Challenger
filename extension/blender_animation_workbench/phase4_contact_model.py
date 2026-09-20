from __future__ import annotations

from enum import IntEnum, StrEnum

AWB_CONTACT_STATE_PROPERTY = "awb_contact_state"
AWB_CONTACT_AUTHORING_STATE_PROPERTY = "awb_contact_authoring_state"
AWB_CONTACT_EXTERNAL_CONSTRAINT = "AWB_ContactExternalSpace"
AWB_CONTACT_PIVOT_CONSTRAINT = "AWB_ContactPointPivot"
AWB_CONTACT_EXTERNAL_PLACEHOLDER_PROPERTY = "awb_contact_external_placeholder"


class ContactPlantSpace(StrEnum):
    WORLD = "WORLD"
    OBJECT = "OBJECT"


class ContactKeyType(StrEnum):
    FREE = "FREE"
    SLIDING = "SLIDING"
    PLANTED = "PLANTED"


class ContactStateValue(IntEnum):
    UNINITIALIZED = 0
    FREE = 1
    SLIDING = 2
    PLANTED = 3


def state_value_for_type(contact_type: ContactKeyType) -> ContactStateValue:
    if contact_type is ContactKeyType.FREE:
        return ContactStateValue.FREE
    if contact_type is ContactKeyType.SLIDING:
        return ContactStateValue.SLIDING
    if contact_type is ContactKeyType.PLANTED:
        return ContactStateValue.PLANTED
    raise ValueError(f"Unsupported Contact type: {contact_type!r}")


def type_for_state_value(value: float) -> ContactKeyType | None:
    rounded = round(float(value))
    if abs(float(value) - float(rounded)) > 1e-6:
        return None
    if rounded == int(ContactStateValue.FREE):
        return ContactKeyType.FREE
    if rounded == int(ContactStateValue.SLIDING):
        return ContactKeyType.SLIDING
    if rounded == int(ContactStateValue.PLANTED):
        return ContactKeyType.PLANTED
    return None
