"""Doctor endpoints. Doctors are linked per visit, so a patient can appear under several doctors."""
from fastapi import APIRouter, Depends, Path, Query

from core.doctors import fetch_doctor_patients, fetch_doctors
from storage.cache_db import get_doctor_settings, set_doctor_default_view
import summary_tiers

from api.deps import require_api_key
from api.schemas import DoctorSettings, DoctorSettingsUpdate

router = APIRouter(prefix="/doctors", tags=["doctors"], dependencies=[Depends(require_api_key)])


@router.get("")
def list_doctors():
    return {"doctors": fetch_doctors()}


@router.get("/{doctor_id}/patients")
def list_doctor_patients(doctor_id: str,
                         q: str = Query("", max_length=36, pattern=r"^[0-9a-fA-F-]*$",
                                        description="Start of the patient ID"),
                         limit: int = Query(200, ge=1, le=200)):
    patients = fetch_doctor_patients(doctor_id, q.lower(), limit)
    return {"doctor_id": doctor_id, "total": len(patients), "patients": patients}


_DOCTOR_ID = r"^[0-9A-Za-z-]{1,64}$"


@router.get("/{doctor_id}/settings", response_model=DoctorSettings)
def doctor_settings(doctor_id: str = Path(..., pattern=_DOCTOR_ID)):
    """The doctor's saved preferences; default_view falls back to the configured default (quick) when none saved."""
    saved = get_doctor_settings(doctor_id)
    return DoctorSettings(doctor_id=doctor_id, saved=bool(saved),
                          default_view=summary_tiers.resolve_view(None, (saved or {}).get("default_view")))


@router.put("/{doctor_id}/settings", response_model=DoctorSettings)
def update_doctor_settings(body: DoctorSettingsUpdate, doctor_id: str = Path(..., pattern=_DOCTOR_ID)):
    row = set_doctor_default_view(doctor_id, body.default_view)
    return DoctorSettings(doctor_id=doctor_id, default_view=row["default_view"], saved=True)
