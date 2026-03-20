from comfy_api.latest import ComfyExtension
from .nodes import LoGeRModelLoader, LoGeRInference, LoGeRDepthOutput, LoGeRToPointcloud, LoGeRToHoudiniScript


class LoGeRExtension(ComfyExtension):
    async def get_node_list(self):
        return [LoGeRModelLoader, LoGeRInference, LoGeRDepthOutput, LoGeRToPointcloud, LoGeRToHoudiniScript]


async def comfy_entrypoint():
    return LoGeRExtension()
