from quant_platform.api.schemas.common import ApiModel


class ReportRecord(ApiModel):
    id: str
    name: str
    type: str
    size: int
    url: str
