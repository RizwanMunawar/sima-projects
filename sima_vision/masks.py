"""Recovering per-instance masks from a segment head, and compositing with them.

A detect head emits one BBOX tensor and nothing else. A segment head emits mask
data as well, and there are three shapes it comes in -- see
:class:`~sima_vision.tasks.segment.SegmentConfig` for what each one is and when
you get it.

Boxes need none of this: Preproc records the resize and letterbox transform on
the tensor and BoxDecode undoes it, which is why detections arrive in
source-image pixels. Masks are produced in the network's own coordinate space,
so this module rebuilds that transform in order to invert it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import runtime
from .samples import BBOX_RECORD_SIZE, iter_tensors


@dataclass
class MaskBundle:
    """Mask data for one frame, already reduced to numpy.

    Attributes:
        kind: ``proto``, ``planes`` or ``none``.
        protos: Prototype bank, ``(C, mh, mw)`` float32. Only for ``proto``.
        coeffs: Per-instance coefficients, ``(N, C)`` float32. Only for ``proto``.
        planes: Per-instance masks, ``(N, mh, mw)``. Only for ``planes``.
        probabilities: Whether the values are already in 0..1 (or 0..255). When
            False they are treated as logits and compared against
            ``logit(threshold)``.
        origin: Where the data came from, ``packed`` or ``tensors``. Reporting
            only; the decode path is the same once it is in numpy.
        peak: Largest value seen across the used planes, which separates a
            0/1 binary mask from a 0..255 quantised one.
    """

    kind: str = "none"
    protos: object = None
    coeffs: object = None
    planes: object = None
    probabilities: bool = False
    origin: str = ""
    peak: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        """Mask-space ``(height, width)``, or ``(0, 0)`` when there is no mask."""
        if self.kind == "proto":
            return int(self.protos.shape[1]), int(self.protos.shape[2])
        if self.kind == "planes":
            return int(self.planes.shape[1]), int(self.planes.shape[2])
        return 0, 0

    @property
    def count(self) -> int:
        if self.kind == "planes":
            return int(self.planes.shape[0])
        if self.kind == "proto":
            return int(self.coeffs.shape[0])
        return 0


def squeeze_leading(array):
    """Drop leading length-1 axes, so ``(1, 32, 160, 160)`` becomes ``(32, 160, 160)``."""
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    return array


def plane_values_are_probabilities(planes) -> bool:
    """Whether a plane bank is already 0..1 (or 0..255) rather than raw logits.

    Sampling the first plane is enough: a bank is homogeneous, and reducing over
    all of them every frame would cost more than the masks themselves.
    """
    np = runtime.np
    if planes.dtype == np.uint8:
        return True
    sample = planes[0] if planes.shape[0] else planes
    return bool(sample.min() >= 0.0 and sample.max() <= 1.0)


def classify_mask_tensors(arrays, count: int, seg) -> MaskBundle:
    """Work out which of the leftover tensors are protos, coefficients or planes.

    Args:
        arrays: Every non-BBOX tensor from the instances field, as numpy arrays.
        count: How many instances the BBOX tensor reported.
        seg: Segmentation settings, for ``source`` and ``coeff_counts``.

    Returns:
        A populated :class:`MaskBundle`, or one with ``kind="none"`` when
        nothing recognisable was found.
    """
    np = runtime.np
    planes = protos = coeffs = None
    for array in arrays:
        array = squeeze_leading(array)
        if array.ndim == 3:
            # A plane bank has one plane per instance; a prototype bank has one
            # per prototype, and those counts only collide by coincidence. When
            # they do, the configured coeff_counts breaks the tie.
            if count > 0 and array.shape[0] == count and array.shape[0] not in seg.coeff_counts:
                planes = array
            elif array.shape[0] in seg.coeff_counts:
                protos = array
            elif count > 0 and array.shape[0] == count:
                planes = array
        elif array.ndim == 2:
            if count > 0 and array.shape == (count, count):
                continue  # ambiguous, and no real model emits this
            if count > 0 and array.shape[0] == count and array.shape[1] in seg.coeff_counts:
                coeffs = array
            elif count > 0 and array.shape[1] == count and array.shape[0] in seg.coeff_counts:
                coeffs = array.T

    if seg.source in {"auto", "planes"} and planes is not None:
        planes = np.ascontiguousarray(planes)
        return MaskBundle(
            kind="planes",
            planes=np.array(planes, copy=True),
            probabilities=plane_values_are_probabilities(planes),
            origin="tensors",
            # Same 0/1-versus-0..255 question as the packed path, so answer it
            # the same way rather than defaulting and thresholding to nothing.
            peak=int(planes[:count].max()) if count and planes.shape[0] >= count else 0,
        )
    if seg.source in {"auto", "proto"} and protos is not None and coeffs is not None:
        return MaskBundle(
            kind="proto",
            protos=np.array(protos, dtype=np.float32, copy=True),
            coeffs=np.array(coeffs, dtype=np.float32, copy=True),
            probabilities=False,
            origin="tensors",
        )
    return MaskBundle()


MIN_MASK_SIDE = 16
MAX_MASK_SIDE = 1024


def packed_mask_layout(payload_len: int, top_k: int, seg):
    """Solve a BBOX buffer that carries its masks in its own tail.

    ``neatobjectdecode`` emits one UInt8 tensor per frame for a segment head::

        [uint32 count][RawBox 24B * slots][mask side*side uint8 * slots]

    ``slots`` is the top-K capacity rather than the number of detections, so the
    buffer is the same size on every frame and the arithmetic below is exact
    rather than a guess. With ``top_k=50`` and a 640-input model:
    ``4 + 50*24 + 50*160*160 = 1281204``.

    Args:
        payload_len: Total bytes in the BBOX tensor.
        top_k: The configured ``decode.max_detections``, or 0 when the archive's
            own value is in force and we do not know it.
        seg: Segmentation settings, for ``mask_sides``.

    Returns:
        A ``(slots, side)`` pair, or None when the length does not decompose.
    """
    # The top-K we asked for is the answer whenever it was honoured, so try it
    # before searching. One exact hit beats a table of plausible ones.
    if top_k > 0:
        tail = payload_len - 4 - top_k * BBOX_RECORD_SIZE
        if tail > 0 and tail % top_k == 0:
            per = tail // top_k
            side = math.isqrt(per)
            if side * side == per and MIN_MASK_SIDE <= side <= MAX_MASK_SIDE:
                return top_k, side

    # top_k unknown, or the decoder used its own. Solve for the slot count from
    # each plausible mask edge instead; a wrong side almost never divides.
    for side in seg.mask_sides:
        per = side * side + BBOX_RECORD_SIZE
        if (payload_len - 4) % per == 0:
            slots = (payload_len - 4) // per
            if slots > 0:
                return slots, side
    return None


def masks_from_packed_payload(payload: bytes, count: int, top_k: int, seg) -> MaskBundle:
    """Read per-instance masks out of the tail of the BBOX buffer."""
    np = runtime.np
    layout = packed_mask_layout(len(payload), top_k, seg)
    if layout is None:
        return MaskBundle()
    slots, side = layout
    if slots < count:
        return MaskBundle()
    offset = 4 + slots * BBOX_RECORD_SIZE
    planes = np.frombuffer(
        payload, dtype=np.uint8, count=slots * side * side, offset=offset
    ).reshape(slots, side, side)
    # Only the first `count` slots hold a detection; the rest are unwritten
    # capacity, so peak is measured over the used ones alone.
    peak = int(planes[:count].max()) if count else 0
    return MaskBundle(
        kind="planes", planes=planes, probabilities=True, origin="packed", peak=peak
    )


def extract_masks(sample, bbox_tensor, payload: bytes, count: int, seg, top_k: int) -> MaskBundle:
    """Pull mask data out of the instances field of one joined sample.

    Packed first, because that is what the shipped SiMa model packs emit and it
    needs no tensor sniffing at all: the byte length either decomposes or it
    does not. Separate tensors are the fallback.
    """
    np = runtime.np
    if seg.masks != "on" or count <= 0:
        return MaskBundle()

    if seg.source in {"auto", "packed"}:
        bundle = masks_from_packed_payload(payload, count, top_k, seg)
        if bundle.kind != "none":
            return bundle
        if seg.source == "packed":
            return MaskBundle()

    arrays = []
    for _owner, tensor in iter_tensors(sample):
        if tensor is bbox_tensor:
            continue
        try:
            arrays.append(np.asarray(tensor.to_numpy(copy=False)))
        except Exception:
            # A tessellated or otherwise opaque tensor cannot become an array.
            # It is not mask data we can use, so skip it rather than fail.
            continue
    return classify_mask_tensors(arrays, count, seg)


# ─────────────────────────────────────────────────────────────────────────────
# Mask space -> frame space
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Letterbox:
    """Affine map from frame pixels to network pixels.

    ``nx = fx * sx + pad_x`` and ``ny = fy * sy + pad_y``. Letterbox and crop
    share one scale; stretch has a different one per axis and no padding.
    """

    sx: float
    sy: float
    pad_x: float
    pad_y: float


def letterbox_transform(frame_w: int, frame_h: int, net_w: int, net_h: int, mode: str) -> Letterbox:
    """Recreate what Preproc did to the frame before the model saw it.

    Args:
        frame_w: Source frame width.
        frame_h: Source frame height.
        net_w: Model input width.
        net_h: Model input height.
        mode: ``letterbox``, ``stretch`` or ``crop``, from ``preprocess.resize``.

    Returns:
        A :class:`Letterbox` mapping frame pixels to network pixels.
    """
    if mode == "stretch":
        return Letterbox(net_w / frame_w, net_h / frame_h, 0.0, 0.0)
    # letterbox fits the whole frame inside the net and pads the shortfall;
    # crop fills the net and spills over the edges. Same formula, min vs max,
    # and crop simply produces negative padding.
    pick = max if mode == "crop" else min
    scale = pick(net_w / frame_w, net_h / frame_h)
    return Letterbox(
        scale,
        scale,
        (net_w - frame_w * scale) / 2.0,
        (net_h - frame_h * scale) / 2.0,
    )


def instance_plane(bundle: MaskBundle, index: int):
    """The mask-space plane for one instance, as float32.

    For ``proto`` this is the coefficient-weighted sum of the prototype bank,
    left as logits. Thresholding a logit at ``logit(t)`` is the same decision as
    thresholding its sigmoid at ``t``, and sigmoid over every prototype-sized
    plane, every frame, is real work for no change in the result.
    """
    if bundle.kind == "planes":
        return bundle.planes[index]
    protos = bundle.protos
    channels, mask_h, mask_w = protos.shape
    flat = protos.reshape(channels, mask_h * mask_w)
    return (bundle.coeffs[index] @ flat).reshape(mask_h, mask_w)


def plane_cutoff(bundle: MaskBundle, threshold: float) -> float:
    """The value to compare a plane against, in that plane's own units.

    A uint8 plane is either a 0/1 binary mask or a 0..255 quantised
    probability, and the two need cut-offs 255x apart. The observed peak tells
    them apart, so a binary mask is not silently thresholded away to nothing.
    """
    np = runtime.np
    if not bundle.probabilities:
        return math.log(threshold / (1.0 - threshold))     # logit
    if bundle.kind == "planes" and bundle.planes.dtype == np.uint8:
        return threshold * (255.0 if bundle.peak > 1 else 1.0)
    return threshold


def warp_plane_to_box(plane, lb: Letterbox, net_w: int, net_h: int,
                      box: tuple[int, int, int, int]):
    """Scale one mask-space plane into a box-sized crop in frame pixels.

    Resampling only the box region rather than the whole frame is what keeps
    this affordable: a 1080p frame with 20 instances would otherwise mean 20
    full-frame resizes per frame, and every one of them would be almost entirely
    empty. The composed map is a pure scale plus translation, so a single
    inverse-mapped affine warp is exact.

    Args:
        plane: ``(mh, mw)`` float32 mask-space plane.
        lb: Frame-to-network transform.
        net_w: Model input width.
        net_h: Model input height.
        box: ``(x1, y1, x2, y2)`` in frame pixels.

    Returns:
        A ``(y2 - y1, x2 - x1)`` float32 array in the plane's own units.
    """
    cv2, np = runtime.cv2, runtime.np
    x1, y1, x2, y2 = box
    mask_h, mask_w = plane.shape
    # frame px -> network px -> mask px, then shift by half a pixel because
    # OpenCV samples at pixel centres.
    ax = lb.sx * mask_w / net_w
    ay = lb.sy * mask_h / net_h
    bx = ((x1 + 0.5) * lb.sx + lb.pad_x) * mask_w / net_w - 0.5
    by = ((y1 + 0.5) * lb.sy + lb.pad_y) * mask_h / net_h - 0.5
    matrix = np.array([[ax, 0.0, bx], [0.0, ay, by]], dtype=np.float32)
    return cv2.warpAffine(
        np.ascontiguousarray(plane, dtype=np.float32),
        matrix,
        (x2 - x1, y2 - y1),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def resize_plane_to_box(plane, box: tuple[int, int, int, int]):
    """Scale a whole plane into the box, for masks already cropped to it."""
    cv2, np = runtime.cv2, runtime.np
    x1, y1, x2, y2 = box
    return cv2.resize(
        np.ascontiguousarray(plane, dtype=np.float32),
        (x2 - x1, y2 - y1),
        interpolation=cv2.INTER_LINEAR,
    )


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detect_mask_space(plane, cutoff: float, lb: Letterbox, net_w: int, net_h: int,
                      box: tuple[int, int, int, int]) -> str:
    """Work out whether a plane covers the whole network input or just its box.

    Both conventions exist and they are indistinguishable from shape alone, but
    not from content: map the detection's own box into the plane and compare it
    with where the ink actually is. A network-space mask lines up; a
    box-cropped one fills the plane no matter where its box sits.

    Args:
        plane: One mask-space plane.
        cutoff: Threshold in the plane's own units.
        lb: Frame-to-network transform.
        net_w: Model input width.
        net_h: Model input height.
        box: ``(x1, y1, x2, y2)`` in frame pixels.

    Returns:
        ``net``, ``box``, or ``""`` when the plane is empty and tells us nothing.
    """
    np = runtime.np
    rows = np.nonzero((plane > cutoff).any(axis=1))[0]
    cols = np.nonzero((plane > cutoff).any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return ""
    mask_h, mask_w = plane.shape
    seen = (
        cols[0] / mask_w, rows[0] / mask_h,
        (cols[-1] + 1) / mask_w, (rows[-1] + 1) / mask_h,
    )
    x1, y1, x2, y2 = box
    expected = (
        (x1 * lb.sx + lb.pad_x) / net_w, (y1 * lb.sy + lb.pad_y) / net_h,
        (x2 * lb.sx + lb.pad_x) / net_w, (y2 * lb.sy + lb.pad_y) / net_h,
    )
    return "net" if _iou(seen, expected) >= 0.30 else "box"


@dataclass
class Instance:
    """One detected object, with its mask already in frame coordinates.

    Attributes:
        box: The raw detection dict, for captions and metadata.
        x1: Left edge in frame pixels, clipped to the frame.
        y1: Top edge.
        x2: Right edge, exclusive.
        y2: Bottom edge, exclusive.
        mask: Boolean ``(y2 - y1, x2 - x1)`` array, or None when the run is
            falling back to box-shaped regions.
        keep: Whether this instance counts as foreground for the blur.
    """

    box: dict
    x1: int
    y1: int
    x2: int
    y2: int
    mask: object = None
    keep: bool = True

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def mask_area(self) -> int:
        return int(self.mask.sum()) if self.mask is not None else self.area


def as_cv_mask(mask):
    """View a boolean mask as the uint8 mask OpenCV wants, without copying.

    ``np.bool_`` is one byte holding 0 or 1, and every OpenCV mask argument
    treats non-zero as set, so the view is free and the values already mean the
    right thing.
    """
    np = runtime.np
    return mask.view(np.uint8) if mask.flags["C_CONTIGUOUS"] else mask.astype(np.uint8)


def refine_mask(mask, seg, scale: float):
    """Smooth and grow a boolean mask according to the ``segmentation`` settings."""
    cv2, np = runtime.cv2, runtime.np
    if seg.blur_mask > 0:
        k = max(3, int(round(seg.blur_mask * scale))) | 1
        smoothed = cv2.GaussianBlur(mask.astype(np.uint8) * 255, (k, k), 0)
        mask = smoothed > 127
    if seg.dilate > 0:
        radius = max(1, int(round(seg.dilate * scale)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    return mask


def foreground_mask(instances: list[Instance], frame_shape):
    """Union of every kept instance, as a full-frame uint8 mask of 0 and 255."""
    cv2, np = runtime.cv2, runtime.np
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for inst in instances:
        if not inst.keep:
            continue
        region = mask[inst.y1 : inst.y2, inst.x1 : inst.x2]
        if inst.mask is None:
            region[:] = 255
        else:
            # Union rather than assignment: overlapping instances must not
            # punch each other's pixels back out of the foreground.
            cv2.bitwise_or(region, 255, dst=region, mask=as_cv_mask(inst.mask))
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Compositing
# ─────────────────────────────────────────────────────────────────────────────


def blurred_background(frame, blur, scale: float):
    """Build the treated copy of the frame that shows through outside the mask.

    Always returns a new array, never a view of ``frame``, because the caller
    writes the sharp pixels back into it.

    Args:
        frame: BGR source frame.
        blur: Background settings.
        scale: Overlay scale factor, so sizes tuned at 1080p hold at any height.

    Returns:
        A BGR image the same size as ``frame``.
    """
    cv2 = runtime.cv2
    height, width = frame.shape[:2]

    if blur.method == "pixelate":
        block = max(2, int(round(blur.pixel_size * scale)))
        small = cv2.resize(
            frame,
            (max(1, width // block), max(1, height // block)),
            interpolation=cv2.INTER_AREA,
        )
        out = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    elif blur.method == "gaussian":
        # Blurring at 1/N resolution is the whole reason this runs at frame
        # rate on 1080p. The downscale is itself a low-pass, so what comes back
        # is a blur either way; the kernel just has less area to cover.
        down = max(1, blur.downscale)
        small = (
            cv2.resize(
                frame,
                (max(1, width // down), max(1, height // down)),
                interpolation=cv2.INTER_AREA,
            )
            if down > 1
            else frame
        )
        kernel = max(1, int(round(blur.kernel * scale / down))) | 1
        small = cv2.GaussianBlur(small, (kernel, kernel), blur.sigma * scale / down)
        out = (
            cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
            if down > 1
            else small
        )
    else:
        out = frame.copy()

    if blur.grayscale:
        out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if blur.dim > 0:
        out = cv2.convertScaleAbs(out, alpha=1.0 - blur.dim, beta=0.0)

    # Opacity last, so one dial covers the blur, the desaturation and the
    # dimming together instead of needing three of them turned down in step.
    if blur.opacity < 1.0:
        out = cv2.addWeighted(out, blur.opacity, frame, 1.0 - blur.opacity, 0.0)
    return out


def composite(frame, mask, blur, scale: float):
    """Keep the frame where the mask is set and the treated copy everywhere else.

    Args:
        frame: BGR source frame, left unmodified.
        mask: Full-frame uint8 foreground mask, 0 or 255.
        blur: Background settings.
        scale: Overlay scale factor.

    Returns:
        A new BGR image.
    """
    cv2 = runtime.cv2
    background = blurred_background(frame, blur, scale)
    if blur.invert:
        # Blur the instances instead: same composite, opposite mask.
        mask = cv2.bitwise_not(mask)

    feather = int(round(blur.feather * scale))
    if feather > 0:
        # A hard cut shows every jag a 160x160 mask has after scaling to 1080p.
        # Blurring the mask turns that edge into a short cross-fade, which is
        # both cheaper and better looking than trying to refine the mask itself.
        kernel = max(3, feather) | 1
        alpha = cv2.cvtColor(cv2.GaussianBlur(mask, (kernel, kernel), 0), cv2.COLOR_GRAY2BGR)
        # Blend in uint8 through OpenCV rather than promoting two 1080p frames
        # to float32 and back. The arithmetic is identical to within a rounding
        # step, and the float path costs more than the blur, the mask decode and
        # every other sink put together.
        return cv2.add(
            cv2.multiply(frame, alpha, scale=1.0 / 255.0),
            cv2.multiply(background, cv2.bitwise_not(alpha), scale=1.0 / 255.0),
        )

    # copyTo writes only where the mask is set and leaves the rest of the
    # background untouched, which boolean fancy indexing does far more slowly.
    return cv2.copyTo(frame, mask, background)
