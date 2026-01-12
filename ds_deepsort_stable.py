import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject

Gst.init(None)

pipeline = Gst.Pipeline.new("ds-stable-tracking")

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

tracker = Gst.ElementFactory.make("nvtracker", "tracker")
tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream-8.0/lib/libnvds_nvmultiobjecttracker.so")
tracker.set_property("ll-config-file", "/workspace/config_tracker_deepsort_optimized.yml")
tracker.set_property("tracker-width", 640)
tracker.set_property("tracker-height", 384)
tracker.set_property("display-tracking-id", 1)

nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv")
nvosd = Gst.ElementFactory.make("nvdsosd", "nvosd")

sink = Gst.ElementFactory.make("nveglglessink", "sink")
sink.set_property("sync", False)

for e in [source, demux, parser, decoder, streammux, pgie, tracker, nvvidconv, nvosd, sink]:
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
pgie.link(tracker)
tracker.link(nvvidconv)
nvvidconv.link(nvosd)
nvosd.link(sink)

print("="*70)
print("🏀 STABLE TRACKING - Basketball Optimized")
print("="*70)
print("✅ Hızlı hareket optimizasyonu: AÇIK")
print("✅ Shadow tracking: 100 frame")
print("✅ ReID weight: 0.8 (yüksek)")
print("✅ Motion tolerance: Artırıldı")
print("="*70)
print("🔍 ID kayıplarını izleyin...")
print("="*70)

pipeline.set_state(Gst.State.PLAYING)

loop = GObject.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    print("\n⏹️  Durduruluyor...")

pipeline.set_state(Gst.State.NULL)
print("✅ Done!")
