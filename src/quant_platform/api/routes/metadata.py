from fastapi import APIRouter, Depends

from quant_platform.api.dependencies import metadata
from quant_platform.api.schemas.metadata import (
    ConfigResponse,
    FeatureListResponse,
    MetadataResponse,
)
from quant_platform.api.services.metadata import MetadataService

router = APIRouter(tags=["metadata"])


@router.get("/meta", response_model=MetadataResponse)
def get_metadata(service: MetadataService = Depends(metadata)) -> MetadataResponse:
    return service.metadata()


@router.get("/config", response_model=ConfigResponse)
def get_config(service: MetadataService = Depends(metadata)) -> ConfigResponse:
    return service.config()


@router.get("/features", response_model=FeatureListResponse)
def get_features(service: MetadataService = Depends(metadata)) -> FeatureListResponse:
    return FeatureListResponse(items=service.metadata().features)
