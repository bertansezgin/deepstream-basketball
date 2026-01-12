import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

Gst.init(None)

pipeline = Gst.Pipeline.new("video-pipeline")

source = Gst.ElementFactory.make("filesrc", "source")
source.set_property("location", "/workspace/basketball_video.mp4")

demux = Gst.ElementFactory.make("qtdemux", "demux")
parser = Gst.ElementFactory.make("h264parse", "parser")
decoder = Gst.ElementFactory.make("nvv4l2decoder", "decoder")
sink = Gst.ElementFactory.make("nveglglessink", "sink")
sink.set_property("sync", False)

for e in [source, demux, parser, decoder, sink]:
    pipeline.add(e)

source.link(demux)

def on_pad_added(demux, pad, parser):
    sink_pad = parser.get_static_pad("sink")
    if not sink_pad.is_linked():
        pad.link(sink_pad)

demux.connect("pad-added", on_pad_added, parser)

parser.link(decoder)
decoder.link(sink)

pipeline.set_state(Gst.State.PLAYING)

loop = GObject.MainLoop()
try:
    loop.run()
except:
    pass

pipeline.set_state(Gst.State.NULL)
