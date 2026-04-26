from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel


class Meta(BaseModel):
    server_timestamp_ist: str


class ScheduleItem(BaseModel):
    stationCode: str
    stationName: str
    arrivalTime: str
    departureTime: str
    routeNumber: str
    haltTime: str
    distance: str
    dayCount: str
    stnSerialNumber: str
    boardingDisabled: str


class Train(BaseModel):
    trainNumber: str
    origin: str
    destination: str
    trainName: str
    stationFrom: str
    stationTo: str
    runningOn: str
    journeyClasses: List[str]
    schedule: List[ScheduleItem]
    train_type: List[str]


class BodyItem(BaseModel):
    title: str
    trains: List[Train]
    queryText: str


class Message(BaseModel):
    title: str
    message: str


class Status(BaseModel):
    result: str
    message: Message


class Model(BaseModel):
    meta: Meta
    body: List[BodyItem]
    error: Any
    status: Status
    code: int
