import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject
import sys
import time
sys.path.insert(0, '/workspace')
from id_manager_ultimate import UltimateIDManager
import pyds

Gst.init(None)

manager = UltimateIDManager(
    learning_frames=90,
    voting_window=7,
    position_threshold=250
)

def tracker_src_pad_buffer_probe(pad, info, u_data):
    """Probe callback to process tracker output and apply ID normalization"""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Extract detections from tracker output
        detections = []
        l_obj = frame_meta.obj_meta_list

        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # Only process person class (class_id=0 for PeopleNet)
            if obj_meta.class_id == 0:
                detections.append({
                    'id': obj_meta.object_id,  # Tracker ID
                    'x': obj_meta.rect_params.left,
                    'y': obj_meta.rect_params.top,
                    'w': obj_meta.rect_params.width,
                    'h': obj_meta.rect_params.height,
                    'confidence': obj_meta.confidence,
                    'obj_meta': obj_meta  # Keep reference for updating
                })

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Process through manager
        if detections:
            normalized = manager.update(detections)

            # Update OSD with normalized IDs
            for norm_det in normalized:
                obj_meta = norm_det['obj_meta']
                normalized_id = norm_det['normalized_id']

                # ← DEĞIŞIKLIK: -1 ID'leri gösterme (baseline'a map edilememiş)
                if normalized_id == -1:
                    # Map edilememiş - gösterme
                    obj_meta.text_params.display_text = ""
                    obj_meta.text_params.set_bg_clr = 0  # Arka plan kaldır
                else:
                    # Baseline ID - göster
                    obj_meta.object_id = normalized_id
                    obj_meta.text_params.display_text = f"ID: {normalized_id}"

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def print_stats():
    stats = manager.get_stats()

    if not stats['learning_complete']:
        print(f"🎓 ÖĞRENME: {stats['frame_count']}/90 | "
              f"Baseline: {stats['baseline_players']}")
    else:
        # Build stats string
        stats_str = f"📊 Frame: {stats['frame_count']} | "
        stats_str += f"Voting: {stats['voting_corrections']} | "

        # Add ReID stats if available
        if 'reid_matches' in stats:
            stats_str += f"ReID: {stats['reid_matches']} | "
            stats_str += f"Confirmed: {stats['candidate_confirmations']} | "
            stats_str += f"👤 Active: {stats.get('active_profiles', 0)}/{stats.get('frozen_profiles', 0)} | "
        else:
            stats_str += f"Optimal: {stats['optimal_mappings']} | "

        stats_str += f"Multi: {stats['multi_mapping_events']} | "
        stats_str += f"🏃 Speed: {stats['avg_player_speed']:.1f}px/f | "
        stats_str += f"👻 Occluded: {stats['occlusion_events']}"

        print(stats_str)

    return True

pipeline = Gst.Pipeline.new("ds-ultimate")

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
sink.set_property("sync", True)  # Enable sync for rate control to work

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

# Add probe to tracker output to normalize IDs
tracker_src_pad = tracker.get_static_pad("src")
if not tracker_src_pad:
    sys.stderr.write("Unable to get tracker src pad\n")
else:
    tracker_src_pad.add_probe(Gst.PadProbeType.BUFFER, tracker_src_pad_buffer_probe, 0)
    print("✅ Probe added to tracker output")

print("="*70)
print("🏆 ULTIMATE ID MANAGER")
print("="*70)
print("✅ Stage 1: Voting (7-frame window)")
print("✅ Stage 2: Baseline Learning (90 frames)")
print("✅ Stage 3: Optimal Matching (Hungarian)")
print("="*70)

GObject.timeout_add_seconds(3, print_stats)

pipeline.set_state(Gst.State.PLAYING)

# Wait for pipeline to fully start
ret, state, pending = pipeline.get_state(Gst.CLOCK_TIME_NONE)
print(f"📡 Pipeline state: {state}")

# Wait a bit more to ensure all elements are ready
time.sleep(2)

# Set playback rate to 0.25x (slow motion)
playback_rate = 0.25
print(f"🎬 Attempting to set playback rate to {playback_rate}x...")

# Send seek event with rate
seek_event = Gst.Event.new_seek(
    playback_rate,                      # rate (0.25x for slow motion)
    Gst.Format.TIME,                    # format
    Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE,  # flags
    Gst.SeekType.SET,                   # start_type
    0,                                  # start position
    Gst.SeekType.NONE,                  # stop_type
    -1                                  # stop position (end of stream)
)

# Try sending to pipeline
if pipeline.send_event(seek_event):
    print(f"✅ Playback rate set to {playback_rate}x (slow motion)")
else:
    print("⚠️  Warning: Pipeline seek event failed, trying alternative method...")
    # Alternative: send to source element
    if source.send_event(seek_event):
        print(f"✅ Playback rate set to {playback_rate}x via source element")
    else:
        print("❌ Could not set playback rate - video will play at normal speed")
        print("   This can happen with some container formats or when using hardware decoder")

loop = GObject.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    print("\n⏹️  Durduruluyor...")

pipeline.set_state(Gst.State.NULL)
print("✅ Tamamlandı!")
