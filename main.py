from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from fastapi import UploadFile, File, Form
import os
from datetime import datetime
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from db import SessionLocal, engine
from models import Vehicle, GateLog
import models
from fastapi.staticfiles import StaticFiles

# CREATE APP FIRST
app = FastAPI(title="LPR Gate System API")

app.mount(
    "/frames",
    StaticFiles(directory="frames"),
    name="frames"
)

# THEN CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models.Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------------------------
# REQUEST BODY MODELS
# -----------------------------------------------

class AuthorizeRequest(BaseModel):
    plate_number: str
    confidence_score: float = 0.0
    snapshot_path: str = None

class AddVehicleRequest(BaseModel):
    plate_number: str


# -----------------------------------------------
# ENDPOINTS
# -----------------------------------------------

@app.post("/api/authorize")
async def authorize(
    plate_number: str = Form(...),
    confidence_score: float = Form(0),
    snapshot: UploadFile = File(None),
    db: Session = Depends(get_db)
):

    vehicle = db.query(Vehicle).filter(
        Vehicle.plate_number == plate_number
    ).first()

    status = (
        "AUTHORIZED"
        if vehicle
        else "UNAUTHORIZED"
    )

    snapshot_path = None

    if snapshot:

        os.makedirs(
            "frames",
            exist_ok=True
        )

        ext = snapshot.filename.split('.')[-1]

        filename = (
            f"{plate_number}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            f".{ext}"
        )

        filepath = os.path.join(
            "frames",
            filename
        )

        with open(
            filepath,
            "wb"
        ) as buffer:

            buffer.write(
                await snapshot.read()
            )

        snapshot_path = filepath

    log = GateLog(
        plate_number=plate_number,
        status=status,
        confidence_score=confidence_score,
        snapshot_path=snapshot_path
    )

    db.add(log)
    db.commit()
    db.refresh(log)

    return {
        "log_id": log.id,
        "status": status,
        "snapshot_path": snapshot_path
    }


@app.get("/api/logs")
def get_logs(db: Session = Depends(get_db)):
    logs = db.query(GateLog).order_by(GateLog.timestamp.desc()).all()
    return logs


@app.get("/api/logs/unauthorized")
def get_unauthorized(db: Session = Depends(get_db)):
    logs = db.query(GateLog).filter(
        GateLog.status == "UNAUTHORIZED"
    ).order_by(GateLog.timestamp.desc()).all()
    return logs


@app.get("/api/vehicles")
def get_vehicles(db: Session = Depends(get_db)):
    return db.query(Vehicle).all()


@app.post("/api/vehicles")
def add_vehicle(request: AddVehicleRequest, db: Session = Depends(get_db)):
    existing = db.query(Vehicle).filter(
        Vehicle.plate_number == request.plate_number
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="Plate already registered")

    vehicle = Vehicle(plate_number=request.plate_number)
    db.add(vehicle)
    db.commit()
    db.refresh(vehicle)
    return {"message": "Vehicle added", "vehicle": vehicle}


@app.delete("/api/vehicles/{plate_number}")
def remove_vehicle(plate_number: str, db: Session = Depends(get_db)):
    vehicle = db.query(Vehicle).filter(
        Vehicle.plate_number == plate_number
    ).first()

    if not vehicle:
        raise HTTPException(status_code=404, detail="Plate not found")

    db.delete(vehicle)
    db.commit()
    return {"message": f"{plate_number} removed from whitelist"}