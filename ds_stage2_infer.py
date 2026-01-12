import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject

Gst.init(None)

pipeline = Gst.Pipeline.new("ds-infer-pipeline")

source = Gst.ElementFactory.make("filesrc", "source")
source.set_property("location", "/workspace/basketball_video.mp4")

demux = Gst.ElementFactory.make("qtdemux", "demux")
parser = Gst.ElementFactory.make("h264parse", "parser")
decoder = Gst.ElementFactory.make("nvv4l2decoder", "decoder")

streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
streammux.set_property("batch-size", 1)
streammux.set_property("width", 1920)
streammux.set_property("height", 1080)
streammux.set_property("batched-push-timeout", 40000)

pgie = Gst.ElementFactory.make("nvinfer", "primary-infer")
pgie.set_property("config-file-path", "/workspace/config_peoplenet.txt")

nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv")
nvosd = Gst.ElementFactory.make("nvdsosd", "nvosd")

sink = Gst.ElementFactory.make("nveglglessink", "sink")
sink.set_property("sync", False)

for e in [
    source, demux, parser, decoder,
    streammux, pgie, nvvidconv, nvosd, sink
]:
    pipeline.add(e)

source.link(demux)

def on_pad_added(demux, pad, parser):
    sink_pad = parser.get_static_pad("sink")
    if not sink_pad.is_linked():
        pad.link(sink_pad)

demux.connect("pad-added", on_pad_added, parser)

parser.link(decoder)

srcpad = decoder.get_static_pad("src")
sinkpad = streammux.request_pad_simple("sink_0")
srcpad.link(sinkpad)

streammux.link(pgie)
pgie.link(nvvidconv)
nvvidconv.link(nvosd)
nvosd.link(sink)

pipeline.set_state(Gst.State.PLAYING)

loop = GObject.MainLoop()
try:
    loop.run()
except:
    pass

pipeline.set_state(Gst.State.NULL)
