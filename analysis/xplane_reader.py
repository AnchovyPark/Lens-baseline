"""
Raw-wire-format reader for XPlane protobuf files produced by torch_xla 2.5's
on-demand profiler (`xp.trace()` output). No protobuf schema dependency — just
parses the well-known field numbers of XSpace/XPlane/XLine/XEvent/XEventMetadata/XStat/XStatMetadata.

XPlane proto layout (relevant fields only):

  XSpace {
    repeated XPlane planes = 1;
  }

  XPlane {
    int64  id                    = 1;
    string name                  = 2;
    repeated XLine lines         = 3;
    map<int64, XEventMetadata> event_metadata = 4;
    map<int64, XStatMetadata>  stat_metadata  = 5;
    repeated XStat stats         = 6;
  }

  XLine {
    int64  id                    = 1;
    string name                  = 2;
    int64  timestamp_ns          = 3;
    repeated XEvent events       = 4;
    int64  duration_ps           = 9;
    int64  display_id            = 10;
    string display_name          = 11;
  }

  XEvent {
    int64  metadata_id           = 1;
    int64  offset_ps             = 2;   (oneof)
    int64  duration_ps           = 3;
    repeated XStat stats         = 4;
  }

  XEventMetadata {
    int64  id                    = 1;
    string name                  = 2;
    string display_name          = 3;
  }

  XStat {
    int64  metadata_id           = 1;
    int64  int64_value           = 2;   (oneof)
    double double_value          = 3;
    string str_value             = 4;
    bytes  bytes_value           = 5;
    uint64 uint64_value          = 6;
    int64  ref_value             = 7;
  }

  XStatMetadata {
    int64  id                    = 1;
    string name                  = 2;
    string description           = 3;
  }
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import struct


# ------------- raw wire parser -------------


def _decode_varint(data: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _iter_fields(data: bytes, start: int = 0, end: Optional[int] = None):
    if end is None:
        end = len(data)
    pos = start
    while pos < end:
        tag, pos = _decode_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 7
        if wire_type == 0:  # varint
            v, pos = _decode_varint(data, pos)
            yield field_num, 0, v
        elif wire_type == 1:  # 64-bit fixed
            yield field_num, 1, data[pos : pos + 8]
            pos += 8
        elif wire_type == 2:  # length-delimited
            length, pos = _decode_varint(data, pos)
            yield field_num, 2, data[pos : pos + length]
            pos += length
        elif wire_type == 5:  # 32-bit fixed
            yield field_num, 5, data[pos : pos + 4]
            pos += 4
        else:
            return  # unknown wire type — bail


# ------------- schema-aware decoders -------------


@dataclass
class XStat:
    metadata_id: int = 0
    int64_value: Optional[int] = None
    double_value: Optional[float] = None
    str_value: Optional[str] = None
    bytes_value: Optional[bytes] = None
    uint64_value: Optional[int] = None
    ref_value: Optional[int] = None


def _parse_xstat(data: bytes) -> XStat:
    s = XStat()
    for fn, wt, val in _iter_fields(data):
        if fn == 1 and wt == 0:
            s.metadata_id = val
        elif fn == 2 and wt == 0:
            s.int64_value = val
        elif fn == 3 and wt == 1:
            s.double_value = struct.unpack("<d", val)[0]
        elif fn == 4 and wt == 2:
            s.str_value = val.decode("utf-8", "ignore")
        elif fn == 5 and wt == 2:
            s.bytes_value = bytes(val)
        elif fn == 6 and wt == 0:
            s.uint64_value = val
        elif fn == 7 and wt == 0:
            s.ref_value = val
    return s


@dataclass
class XEvent:
    metadata_id: int = 0
    offset_ps: int = 0
    duration_ps: int = 0
    stats: List[XStat] = field(default_factory=list)


def _parse_xevent(data: bytes) -> XEvent:
    ev = XEvent()
    for fn, wt, val in _iter_fields(data):
        if fn == 1 and wt == 0:
            ev.metadata_id = val
        elif fn == 2 and wt == 0:
            ev.offset_ps = val
        elif fn == 3 and wt == 0:
            ev.duration_ps = val
        elif fn == 4 and wt == 2:
            ev.stats.append(_parse_xstat(val))
    return ev


@dataclass
class XLine:
    id: int = 0
    name: str = ""
    display_name: str = ""
    timestamp_ns: int = 0
    duration_ps: int = 0
    events: List[XEvent] = field(default_factory=list)


def _parse_xline(data: bytes) -> XLine:
    line = XLine()
    for fn, wt, val in _iter_fields(data):
        if fn == 1 and wt == 0:
            line.id = val
        elif fn == 2 and wt == 2:
            line.name = val.decode("utf-8", "ignore")
        elif fn == 3 and wt == 0:
            line.timestamp_ns = val
        elif fn == 4 and wt == 2:
            line.events.append(_parse_xevent(val))
        elif fn == 9 and wt == 0:
            line.duration_ps = val
        elif fn == 11 and wt == 2:
            line.display_name = val.decode("utf-8", "ignore")
    return line


@dataclass
class XMetadata:
    """Covers both XEventMetadata and XStatMetadata (same layout for our purposes)."""

    id: int = 0
    name: str = ""
    display_name: str = ""
    description: str = ""


def _parse_xmetadata(data: bytes) -> XMetadata:
    md = XMetadata()
    for fn, wt, val in _iter_fields(data):
        if fn == 1 and wt == 0:
            md.id = val
        elif fn == 2 and wt == 2:
            md.name = val.decode("utf-8", "ignore")
        elif fn == 3 and wt == 2:
            md.display_name = val.decode("utf-8", "ignore")
    return md


@dataclass
class XPlane:
    id: int = 0
    name: str = ""
    lines: List[XLine] = field(default_factory=list)
    event_metadata: Dict[int, XMetadata] = field(default_factory=dict)
    stat_metadata: Dict[int, XMetadata] = field(default_factory=dict)


def _parse_map_entry(data: bytes) -> Tuple[Optional[int], bytes]:
    """Parse a protobuf map<int64, Message> entry → (key, value_bytes)."""
    key = None
    val_bytes = b""
    for fn, wt, val in _iter_fields(data):
        if fn == 1 and wt == 0:
            key = val
        elif fn == 2 and wt == 2:
            val_bytes = val
    return key, val_bytes


def _parse_xplane(data: bytes) -> XPlane:
    plane = XPlane()
    for fn, wt, val in _iter_fields(data):
        if fn == 1 and wt == 0:
            plane.id = val
        elif fn == 2 and wt == 2:
            plane.name = val.decode("utf-8", "ignore")
        elif fn == 3 and wt == 2:
            plane.lines.append(_parse_xline(val))
        elif fn == 4 and wt == 2:
            k, v = _parse_map_entry(val)
            if k is not None:
                plane.event_metadata[k] = _parse_xmetadata(v)
        elif fn == 5 and wt == 2:
            k, v = _parse_map_entry(val)
            if k is not None:
                plane.stat_metadata[k] = _parse_xmetadata(v)
    return plane


def parse_xspace(path: str) -> List[XPlane]:
    with open(path, "rb") as f:
        data = f.read()
    planes: List[XPlane] = []
    for fn, wt, val in _iter_fields(data):
        if fn == 1 and wt == 2:
            planes.append(_parse_xplane(val))
    return planes


# ------------- convenience -------------


def iter_events(plane: XPlane):
    """Yield (line_name, event_name, offset_ps, duration_ps, resolved_stats) tuples.

    resolved_stats is a list of (stat_name, value) where value is whatever field was set.
    """
    for line in plane.lines:
        for ev in line.events:
            md = plane.event_metadata.get(ev.metadata_id)
            ev_name = md.name if md else f"<unknown:{ev.metadata_id}>"
            stats = []
            for s in ev.stats:
                smd = plane.stat_metadata.get(s.metadata_id)
                sname = smd.name if smd else f"<stat:{s.metadata_id}>"
                val = (
                    s.str_value
                    if s.str_value is not None
                    else s.int64_value
                    if s.int64_value is not None
                    else s.double_value
                    if s.double_value is not None
                    else s.uint64_value
                    if s.uint64_value is not None
                    else s.ref_value
                    if s.ref_value is not None
                    else s.bytes_value
                )
                stats.append((sname, val))
            yield line.name, ev_name, ev.offset_ps, ev.duration_ps, stats
