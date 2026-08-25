#@ String input
#@ String outdir
#@ String stem
#@ Float quality
#@ Integer target_channel
#@ Float diameter
#@ Float threshold
#@ Boolean median_filtering
#@ Boolean subpixel
#@ String tracker
#@ Float linking_max_distance
#@ Float gap_closing_max_distance
#@ Integer max_frame_gap
#@ Boolean allow_gap_closing
#@ Boolean allow_splitting
#@ Float splitting_max_distance
#@ Boolean allow_merging
#@ Float merging_max_distance

# Headless TrackMate runner.
#
# LoG-detector and LAP-tracker settings are all passed in as script parameters
# (above) so the Makefile can expose them as overridable variables. Defaults
# live in the Makefile, not here; the Makefile always supplies every parameter,
# so none is ever left to prompt a (headless-unavailable) dialog.
#
#   detector (LoG):  target_channel, diameter (GUI "Diameter" = 2*RADIUS),
#                    threshold, median_filtering, subpixel; quality is the
#                    separate initial-spot QUALITY filter.
#   tracker:         'simple' -> Simple LAP (linking / gap-closing / frame-gap
#                    only); 'lap' -> full Sparse LAP, which additionally honors
#                    allow_gap_closing and the splitting / merging toggles.
#                    The two are mutually exclusive (pick one via `tracker`).
# Reproduces the manual workflow captured in "Screenshots for Workflow/".
# Outputs (written into <outdir>, named after <stem>):
#   <stem>_spots.csv         spot statistics table
#   <stem>_tracks.avi        overlay video with tracks burned in
#   <stem>_tracks.tif        overlay stack (lossless)
#   <stem>_trackmate.xml     TrackMate session, reopenable in the GUI

import os
import sys

from ij import IJ, ImagePlus, ImageStack
from ij.io import FileSaver
from ij.plugin import AVI_Reader

from fiji.plugin.trackmate import Settings, TrackMate, SelectionModel, Logger
from fiji.plugin.trackmate.detection import LogDetectorFactory
from fiji.plugin.trackmate.tracking.jaqaman import SimpleSparseLAPTrackerFactory
from fiji.plugin.trackmate.tracking.jaqaman import SparseLAPTrackerFactory
from fiji.plugin.trackmate.io import TmXmlWriter

from java.io import File
from java.awt import Color


def open_avi(path):
    # Mirrors screenshot 1: virtual stack + grayscale. The macro form
    # ("IJ.run('AVI...')") opens a dialog in headless mode, so use the
    # AVI_Reader API directly.
    reader = AVI_Reader()
    stack = reader.makeStack(path, 1, 0, True, True, False)  # virtual, gray, !flip
    imp = ImagePlus(os.path.basename(path), stack)

    # Screenshot 3: AVIs come in as Z-stacks; swap so frames are time.
    n = imp.getStackSize()
    imp.setDimensions(1, 1, n)
    imp.setOpenAsHyperStack(True)
    return imp


def build_settings(imp):
    s = Settings(imp)

    # ---- LoG detector (screenshot: "LoG detector") -------------------------
    # The GUI exposes "Diameter"; TrackMate's setting is RADIUS = diameter / 2.
    s.detectorFactory = LogDetectorFactory()
    s.detectorSettings = s.detectorFactory.getDefaultSettings()
    s.detectorSettings['TARGET_CHANNEL'] = int(target_channel)
    s.detectorSettings['RADIUS'] = float(diameter) / 2.0
    s.detectorSettings['THRESHOLD'] = float(threshold)
    s.detectorSettings['DO_SUBPIXEL_LOCALIZATION'] = bool(subpixel)
    s.detectorSettings['DO_MEDIAN_FILTERING'] = bool(median_filtering)

    # Initial QUALITY filter (separate from the detector threshold above).
    s.initialSpotFilterValue = float(quality)

    # ---- tracker (mutually exclusive: Simple LAP vs full Sparse LAP) --------
    tr = str(tracker).strip().lower()
    if tr in ('simple', 'simple_lap', 'simplelap'):
        # Simple LAP tracker: only linking / gap-closing / frame-gap.
        s.trackerFactory = SimpleSparseLAPTrackerFactory()
        s.trackerSettings = s.trackerFactory.getDefaultSettings()
        s.trackerSettings['LINKING_MAX_DISTANCE'] = float(linking_max_distance)
        s.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = float(gap_closing_max_distance)
        s.trackerSettings['MAX_FRAME_GAP'] = int(max_frame_gap)
    elif tr in ('lap', 'sparse', 'sparse_lap', 'sparselap'):
        # Full Sparse LAP tracker: adds gap-closing toggle + splitting/merging.
        s.trackerFactory = SparseLAPTrackerFactory()
        s.trackerSettings = s.trackerFactory.getDefaultSettings()
        s.trackerSettings['LINKING_MAX_DISTANCE'] = float(linking_max_distance)
        s.trackerSettings['ALLOW_GAP_CLOSING'] = bool(allow_gap_closing)
        s.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = float(gap_closing_max_distance)
        s.trackerSettings['MAX_FRAME_GAP'] = int(max_frame_gap)
        s.trackerSettings['ALLOW_TRACK_SPLITTING'] = bool(allow_splitting)
        s.trackerSettings['SPLITTING_MAX_DISTANCE'] = float(splitting_max_distance)
        s.trackerSettings['ALLOW_TRACK_MERGING'] = bool(allow_merging)
        s.trackerSettings['MERGING_MAX_DISTANCE'] = float(merging_max_distance)
    else:
        raise ValueError(
            "Unknown tracker %r: use 'simple' (Simple LAP) or 'lap' (Sparse LAP)"
            % tracker)

    # Match the analyzer set the GUI workflow produced in the example XMLs.
    s.addAllAnalyzers()

    return s


# Columns TrackMate's GUI "Export spots to CSV" writes, in order, as
# (feature key, long name, short name, unit). The four fields become the four
# header rows the GUI emits. LABEL/ID/TRACK_ID are special columns (not spot
# features); everything after them is pulled via spot.getFeature(key).
_SPOT_COLUMNS = [
    ('LABEL',                'Label',                  'Label',      ''),
    ('ID',                   'Spot ID',                'Spot ID',    ''),
    ('TRACK_ID',             'Track ID',               'Track ID',   ''),
    ('QUALITY',              'Quality',                'Quality',    '(quality)'),
    ('POSITION_X',           'X',                      'X',          '(pixel)'),
    ('POSITION_Y',           'Y',                      'Y',          '(pixel)'),
    ('POSITION_Z',           'Z',                      'Z',          '(pixel)'),
    ('POSITION_T',           'T',                      'T',          '(frame)'),
    ('FRAME',                'Frame',                  'Frame',      ''),
    ('RADIUS',               'Radius',                 'R',          '(pixel)'),
    ('VISIBILITY',           'Visibility',             'Visibility', ''),
    ('MANUAL_SPOT_COLOR',    'Manual spot color',      'Spot color', ''),
    ('MEAN_INTENSITY_CH1',   'Mean intensity ch1',     'Mean ch1',   '(counts)'),
    ('MEDIAN_INTENSITY_CH1', 'Median intensity ch1',   'Median ch1', '(counts)'),
    ('MIN_INTENSITY_CH1',    'Min intensity ch1',      'Min ch1',    '(counts)'),
    ('MAX_INTENSITY_CH1',    'Max intensity ch1',      'Max ch1',    '(counts)'),
    ('TOTAL_INTENSITY_CH1',  'Sum intensity ch1',      'Sum ch1',    '(counts)'),
    ('STD_INTENSITY_CH1',    'Std intensity ch1',      'Std ch1',    '(counts)'),
    ('CONTRAST_CH1',         'Contrast ch1',           'Ctrst ch1',  ''),
    ('SNR_CH1',              'Signal/Noise ratio ch1', 'SNR ch1',    ''),
]


def _fmt(v):
    # Match TrackMate's number rendering: integer-valued numbers print with no
    # decimal (5.0 -> "5"), everything else at full double precision. Note
    # repr(), not str(): Jython's str() truncates floats to ~12 sig figs.
    if v is None:
        return ''
    fv = float(v)
    iv = int(fv)
    if fv == iv:
        return str(iv)
    return repr(fv)


def export_spots_csv(model, path):
    # Reproduce the file TrackMate's GUI "Export spots to CSV" writes:
    #   * comma-separated (the ".csv" name was previously a tab-separated lie),
    #   * a 4-row header (keys / long names / short names / units),
    #   * only spots that belong to a track, ordered by (track, frame).
    # The manual export drops untracked spots, so we do too — otherwise the
    # automated file carries extra rows the manual one never had.
    tm_model = model.getTrackModel()

    keys   = [c[0] for c in _SPOT_COLUMNS]
    names  = [c[1] for c in _SPOT_COLUMNS]
    shorts = [c[2] for c in _SPOT_COLUMNS]
    units  = [c[3] for c in _SPOT_COLUMNS]
    feats  = keys[3:]  # after LABEL, ID, TRACK_ID -> genuine spot features

    rows = []
    for spot in model.getSpots().iterable(True):
        tid = tm_model.trackIDOf(spot)
        if tid is None:
            continue  # untracked spot: excluded, matching the manual export
        frame = spot.getFeature('FRAME')
        rows.append((int(tid), int(frame) if frame is not None else 0, spot, tid))
    rows.sort(key=lambda r: (r[0], r[1]))

    f = open(path, 'w')
    try:
        for header_row in (keys, names, shorts, units):
            f.write(','.join(header_row) + '\n')
        for _, _, spot, tid in rows:
            row = [spot.getName(), str(spot.ID()), str(tid)]
            for feat in feats:
                row.append(_fmt(spot.getFeature(feat)))
            f.write(','.join(row) + '\n')
    finally:
        f.close()


_TRACK_PALETTE = [
    Color(228, 26, 28), Color(55, 126, 184), Color(77, 175, 74),
    Color(152, 78, 163), Color(255, 127, 0), Color(255, 255, 51),
    Color(166, 86, 40), Color(247, 129, 191), Color(153, 153, 153),
    Color(0, 206, 209), Color(218, 112, 214), Color(255, 215, 0),
]


def render_overlay(imp, model):
    # Build an RGB stack with spot circles + track edges burned per frame.
    # Headless-safe: uses ImageProcessor drawing only - no Swing displayer.
    n_frames = imp.getStackSize()
    width = imp.getWidth()
    height = imp.getHeight()

    spots_by_frame = {}
    for spot in model.getSpots().iterable(True):
        frame = int(spot.getFeature('FRAME'))
        spots_by_frame.setdefault(frame, []).append((
            spot.getFeature('POSITION_X'),
            spot.getFeature('POSITION_Y'),
            spot.getFeature('RADIUS'),
        ))

    edges_by_frame = {}
    track_model = model.getTrackModel()
    track_ids = list(track_model.trackIDs(True))
    for idx, tid in enumerate(track_ids):
        color = _TRACK_PALETTE[idx % len(_TRACK_PALETTE)]
        for edge in track_model.trackEdges(tid):
            src = track_model.getEdgeSource(edge)
            dst = track_model.getEdgeTarget(edge)
            sf = int(src.getFeature('FRAME'))
            tf = int(dst.getFeature('FRAME'))
            seg = (
                src.getFeature('POSITION_X'), src.getFeature('POSITION_Y'),
                dst.getFeature('POSITION_X'), dst.getFeature('POSITION_Y'),
                color,
            )
            # Draw on every frame from source to target, inclusive.
            for f in range(min(sf, tf), max(sf, tf) + 1):
                edges_by_frame.setdefault(f, []).append(seg)

    out_stack = ImageStack(width, height)
    src_stack = imp.getStack()
    for f in range(1, n_frames + 1):
        ip = src_stack.getProcessor(f).convertToRGB()
        ip.setLineWidth(1)

        for sx, sy, tx, ty, color in edges_by_frame.get(f - 1, []):
            ip.setColor(color)
            ip.drawLine(int(round(sx)), int(round(sy)),
                        int(round(tx)), int(round(ty)))

        ip.setColor(Color.MAGENTA)
        for x, y, r in spots_by_frame.get(f - 1, []):
            d = int(round(2 * r))
            ip.drawOval(int(round(x - r)), int(round(y - r)), d, d)

        out_stack.addSlice(ip)

    out = ImagePlus(imp.getTitle() + " tracks", out_stack)
    return out


def save_overlay(imp, model, outdir, stem):
    capture = render_overlay(imp, model)

    tif_path = os.path.join(outdir, stem + '_tracks.tif')
    FileSaver(capture).saveAsTiff(tif_path)

    avi_path = os.path.join(outdir, stem + '_tracks.avi')
    IJ.run(capture, "AVI... ",
           "compression=JPEG frame=7 save=[%s]" % avi_path)


def save_trackmate_xml(trackmate, path):
    writer = TmXmlWriter(File(path))
    writer.appendModel(trackmate.getModel())
    writer.appendSettings(trackmate.getSettings())
    writer.writeToFile()


def run():
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    print('TrackMate batch: input=%s' % input)
    print('  detector(LoG): channel=%s diameter=%s threshold=%s '
          'median=%s subpixel=%s quality>=%s'
          % (target_channel, diameter, threshold,
             median_filtering, subpixel, quality))
    print('  tracker=%s: linking=%s gap_dist=%s frame_gap=%s'
          % (tracker, linking_max_distance,
             gap_closing_max_distance, max_frame_gap))
    if str(tracker).strip().lower() not in ('simple', 'simple_lap', 'simplelap'):
        print('    gap_closing=%s splitting=%s(%s) merging=%s(%s)'
              % (allow_gap_closing, allow_splitting, splitting_max_distance,
                 allow_merging, merging_max_distance))

    imp = open_avi(input)
    settings = build_settings(imp)

    tm = TrackMate(settings)
    tm.getModel().setLogger(Logger.IJ_LOGGER)

    if not tm.checkInput() or not tm.process():
        print('TrackMate FAILED: %s' % tm.getErrorMessage())
        sys.exit(1)

    model = tm.getModel()

    csv_path = os.path.join(outdir, stem + '_spots.csv')
    export_spots_csv(model, csv_path)

    save_overlay(imp, model, outdir, stem)

    xml_path = os.path.join(outdir, stem + '_trackmate.xml')
    save_trackmate_xml(tm, xml_path)

    print('done: %s' % stem)
    imp.close()


run()
