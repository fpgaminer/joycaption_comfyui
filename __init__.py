from . import nodes

NODE_CLASS_MAPPINGS = {
	"JJC_DownloadAndLoadJoyCaptionModel": nodes.DownloadAndLoadJoyCaptionModel,
	"JJC_JoyCaption": nodes.JoyCaption,
	"JJC_JoyCaption_Custom": nodes.JoyCaptionCustom,
}
NODE_DISPLAY_NAME_MAPPINGS = {
	"JJC_DownloadAndLoadJoyCaptionModel": "Download and Load JoyCaption Model",
	"JJC_JoyCaption": "JoyCaption",
	"JJC_JoyCaption_Custom": "JoyCaption (Custom)",
}
