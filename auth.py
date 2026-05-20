from db import SessionLocal
from models import Vehicle, GateLog

def check_and_log(plate_number: str, confidence_score: float = 0.0, snapshot_path: str = None):

    db = SessionLocal()

    try:
        vehicle = db.query(Vehicle).filter(
            Vehicle.plate_number == plate_number
        ).first()

        if vehicle:
            status = "AUTHORIZED"
        else:
            status = "UNAUTHORIZED"

        log_entry = GateLog(
            plate_number=plate_number,
            status=status,
            confidence_score=confidence_score,
            snapshot_path=snapshot_path
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)

        return {
            "log_id": log_entry.id,
            "plate_number": plate_number,
            "status": status,
            "confidence_score": confidence_score,
            "timestamp": str(log_entry.timestamp)
        }

    except Exception as e:
        db.rollback()
        print(f"DB Error: {e}")
        return None

    finally:
        db.close()