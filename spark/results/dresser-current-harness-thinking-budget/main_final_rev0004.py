"""White painted wooden chest of drawers with nine fluted drawer fronts.

Carcass frame, nine sliding drawers, brass handles and corner brackets.
Each drawer slides on a prismatic joint along +Y (outward from carcass).
"""

from __future__ import annotations

from build123d import Box, Compound, Vector

from articraft.sdk import (
    JointAxis,
    JointDOF,
    Material,
    RigidBodyAssembly,
    TestContext,
    TestReport,
)

# ── Brass material ──────────────────────────────────────────────────────────

BRASS = Material(
    name="brass",
    density=8500.0,
    friction=(0.40, 0.30),
)

# ── Dimensions ──────────────────────────────────────────────────────────────

CARCASS_WIDTH = 1.200       # X, total outer width
CARCASS_DEPTH = 0.450       # Y, total depth
CARCASS_HEIGHT = 0.890      # Z, total outer height (including feet)

SIDE_THICK = 0.025          # X-direction thickness of side panels
TOP_THICK = 0.030           # Z-direction thickness of top/bottom
BACK_THICK = 0.012          # Y-direction thickness of back panel
DIVIDER_THICK = 0.025       # X-direction thickness of vertical dividers
RAIL_THICK = 0.030          # Z-direction thickness of horizontal rails

FEET_HEIGHT = 0.030
FEET_SIZE = 0.060           # square footprint of each foot

# Drawer dimensions
DRAWER_OPENING_W = 0.367    # X, interior width per column
DRAWER_OPENING_H = 0.260    # Z, interior height per row
DRAWER_DEPTH = 0.380        # Y, interior depth of drawer box
DRAWER_FRONT_THICK = 0.035  # thick enough to overlap the box back
DRAWER_FRONT_OVERHANG = 0.012  # overhang beyond opening on each side
DRAWER_FRONT_H = 0.275     # slightly taller than opening

DRAWER_TRAVEL = 0.350       # meters – prismatic travel range


# ── Helpers ─────────────────────────────────────────────────────────────────

def _white() -> tuple[float, float, float]:
    return (0.94, 0.94, 0.91)


def _build_fluted_front(w: float, h: float, t: float, groove_w: float = 0.003,
                        groove_depth: float = 0.003, spacing: float = 0.025):
    """Build a drawer front panel with vertical fluted groove indentations."""
    panel = Box(w, t, h)
    n = max(3, int(w / spacing))
    total = n * spacing
    start_x = -total / 2 + spacing / 2

    cutouts = []
    for i in range(n):
        gx = start_x + i * spacing
        cutouts.append(
            Box(groove_w, t + groove_depth + 0.001, h - 0.02)
            .translate(Vector(gx, 0.0, 0.0))
        )

    if cutouts:
        cut_compound = Compound.make_composite(cutouts)
        panel = panel - cut_compound
    return panel


# ── Build model ─────────────────────────────────────────────────────────────

def build_object_model() -> RigidBodyAssembly:
    model = RigidBodyAssembly("chest_of_drawers")
    carcass = model.rigid_body("carcass")

    # ── Carcass geometry ──────────────────────────────────────────────

    # Interior dimensions between side panels
    inner_w = CARCASS_WIDTH - 2 * SIDE_THICK       # 1.150
    inner_h = CARCASS_HEIGHT - FEET_HEIGHT - 2 * TOP_THICK  # 0.800

    half_w = inner_w / 2                            # 0.575
    half_d = CARCASS_DEPTH / 2                      # 0.225

    # Z positions
    z_base = FEET_HEIGHT / 2                        # 0.015

    # ── Side panels (full height for connectivity to top, bottom, feet) ─
    for side_sign in (-1, 1):
        carcass.add(
            Box(SIDE_THICK, CARCASS_DEPTH, CARCASS_HEIGHT)
            .translate((side_sign * half_w, 0, CARCASS_HEIGHT / 2)),
            name=f"side_{side_sign:+d}",
            material=Material.HARDWOOD,
            color=_white(),
        )

    # ── Top panel (overlaps side panels) ──────────────────────────────
    carcass.add(
        Box(CARCASS_WIDTH, CARCASS_DEPTH, TOP_THICK)
        .translate((0, 0, CARCASS_HEIGHT - TOP_THICK / 2)),
        name="top_panel",
        material=Material.HARDWOOD,
        color=_white(),
    )

    # ── Bottom panel (overlaps side panels and feet) ──────────────────
    carcass.add(
        Box(CARCASS_WIDTH, CARCASS_DEPTH, TOP_THICK)
        .translate((0, 0, TOP_THICK / 2)),
        name="bottom_panel",
        material=Material.HARDWOOD,
        color=_white(),
    )

    # ── Back panel (full width to overlap side panels) ────────────────
    carcass.add(
        Box(CARCASS_WIDTH, BACK_THICK, inner_h - 2 * TOP_THICK)
        .translate(
            (0, -half_d + BACK_THICK / 2,
             z_base + TOP_THICK + (inner_h - 2 * TOP_THICK) / 2)),
        name="back_panel",
        material=Material.HARDWOOD,
        color=_white(),
    )

    # ── Vertical dividers (2) — 3 columns, full interior height ───────
    div_xs = (-0.1835, 0.1835)
    for i, dx in enumerate(div_xs):
        carcass.add(
            Box(DIVIDER_THICK, CARCASS_DEPTH - 0.050,
                inner_h - 2 * TOP_THICK)
            .translate((dx, 0, z_base + TOP_THICK +
                        (inner_h - 2 * TOP_THICK) / 2)),
            name=f"divider_{i+1}",
            material=Material.HARDWOOD,
            color=_white(),
        )

    # ── Horizontal rails (2) — between 3 rows, full width ─────────────
    # Rails stop short of front to avoid overlapping drawer boxes
    rail_rows_z = [0.275, 0.555]  # centers of rails in Z
    for i, zc in enumerate(rail_rows_z):
        carcass.add(
            Box(CARCASS_WIDTH, CARCASS_DEPTH - 0.050, RAIL_THICK)
            .translate((0, 0, zc)),
            name=f"rail_{i+1}",
            material=Material.HARDWOOD,
            color=_white(),
        )

    # ── Feet (4) — connect to bottom panel ────────────────────────────
    feet_offsets = [
        (-half_w + 0.100, -half_d + 0.100),
        (half_w - 0.100, -half_d + 0.100),
        (-half_w + 0.100, half_d - 0.100),
        (half_w - 0.100, half_d - 0.100),
    ]
    for i, (fx, fy) in enumerate(feet_offsets):
        carcass.add(
            Box(FEET_SIZE, FEET_SIZE, FEET_HEIGHT)
            .translate((fx, fy, FEET_HEIGHT / 2)),
            name=f"foot_{i+1}",
            material=Material.HARDWOOD,
            color=_white(),
        )

    # ── Nine drawers ──────────────────────────────────────────────────

    # Column X centers (inside the carcass, at column midpoint)
    # Col 1: from X=-0.575 to -0.183 → center = -0.379
    # Col 2: from X=-0.183 to 0.183  → center =  0.000
    # Col 3: from X=0.183 to 0.575  → center =  0.379
    col_xs = (-0.379, 0.0, 0.379)

    # Row Z centers (of each drawer opening)
    # Row 1: Z from 0.045 to 0.275 → center = 0.160
    # Row 2: Z from 0.305 to 0.555 → center = 0.430
    # Row 3: Z from 0.585 to 0.810 → center = 0.6975
    row_zs = (0.160, 0.430, 0.6975)

    # Carcass front face Y coordinate
    carcass_front_y = half_d

    for col in range(3):
        for row in range(3):
            idx = col * 3 + row + 1
            drawer_name = f"drawer_{idx}"
            drawer = model.rigid_body(drawer_name)

            xc = col_xs[col]
            zc = row_zs[row]

            # ── Drawer box (interior) ───────────────────────────────
            db_w = DRAWER_OPENING_W - 0.004
            db_h = DRAWER_OPENING_H - 0.004
            db_d = DRAWER_DEPTH
            wall_t = 0.015

            # Bottom panel: spans full column width to overlap with sides
            drawer.add(
                Box(db_w, db_d, wall_t)
                .translate((xc, 0, zc - db_h / 2 + wall_t / 2)),
                name="bottom",
                material=Material.HARDWOOD,
                color=_white(),
            )

            # Two side panels: span full height, overlap bottom by wall_t in Z
            for sign in (-1, 1):
                drawer.add(
                    Box(wall_t, db_d, db_h)
                    .translate(
                        (xc + sign * (db_w / 2 - wall_t / 2), 0, zc)),
                    name=f"side_{sign:+d}",
                    material=Material.HARDWOOD,
                    color=_white(),
                )

            # Back panel: spans full width to overlap with sides and bottom
            drawer.add(
                Box(db_w, wall_t, db_h)
                .translate(
                    (xc, db_d / 2 - wall_t / 2, zc)),
                name="back",
                material=Material.HARDWOOD,
                color=_white(),
            )

            # ── Drawer front panel (fluted) ───────────────────────
            fp_w = DRAWER_OPENING_W + 2 * DRAWER_FRONT_OVERHANG
            fp_h = DRAWER_FRONT_H

            # Panel front face is flush with the carcass front face.
            # With 35mm thickness the panel extends back to overlap
            # the drawer-box back panel for body connectivity.
            front_y = carcass_front_y - DRAWER_FRONT_THICK / 2

            front_shape = _build_fluted_front(fp_w, fp_h, DRAWER_FRONT_THICK)
            drawer.add(
                front_shape.translate((xc, front_y, zc)),
                name="front",
                material=Material.HARDWOOD,
                color=_white(),
            )

            # ── Brass handle ──────────────────────────────────────
            handle_len = 0.120
            handle_w = 0.006
            handle_z = zc + fp_h / 2 - 0.075
            drawer.add(
                Box(handle_w, 0.014, handle_len)
                .translate((xc, front_y + 0.008, handle_z)),
                name="handle",
                material=BRASS,
            )

            # ── Brass corner brackets (4) ─────────────────────────
            bracket_size = 0.025
            bracket_thick = 0.002
            corners = [
                (-fp_w / 2 + bracket_size / 2,  fp_h / 2 - bracket_size / 2),
                ( fp_w / 2 - bracket_size / 2,  fp_h / 2 - bracket_size / 2),
                (-fp_w / 2 + bracket_size / 2, -fp_h / 2 + bracket_size / 2),
                ( fp_w / 2 - bracket_size / 2, -fp_h / 2 + bracket_size / 2),
            ]
            for ci, (bx, by) in enumerate(corners):
                drawer.add(
                    Box(bracket_size, bracket_thick, bracket_size)
                    .translate((xc + bx, front_y + 0.001, zc + by)),
                    name=f"corner_{ci+1}",
                    material=BRASS,
                )

            # ── Prismatic joint ───────────────────────────────────
            # Drawer slides along +Y axis. Frame at carcass front face on
            # both bodies so they coincide at rest (zero position).
            model.joint(
                f"drawer_{idx}_slide",
                carcass.at((xc, carcass_front_y, zc)),
                drawer.at((xc, carcass_front_y, zc)),
                dofs=(JointDOF(JointAxis.TRANS_Y, limits=(0.0, DRAWER_TRAVEL)),),
            )

    # ── Articulation ────────────────────────────────────────────────────
    drawer_joints = [model.get_joint(f"drawer_{i}_slide") for i in range(1, 10)]
    model.articulation("drawers", root=carcass, joints=drawer_joints)

    return model


# ── Entry ───────────────────────────────────────────────────────────────────

object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)

    # ── Carcass structure ───────────────────────────────────────────────
    carcass = object_model.get_rigid_body("carcass")
    shape_names = {s.name for s in carcass._shapes.values()}
    expected = {
        "side_-1", "side_+1", "top_panel", "bottom_panel",
        "back_panel", "rail_1", "rail_2", "divider_1", "divider_2",
        "foot_1", "foot_2", "foot_3", "foot_4",
    }
    ctx.check(
        "carcass_has_required_shapes",
        expected.issubset(shape_names),
        f"Missing: {expected - shape_names}",
    )

    # ── Drawer count ────────────────────────────────────────────────────
    drawers = [b for b in object_model.rigid_bodies if b.name.startswith("drawer_")]
    ctx.check(
        "nine_drawers",
        len(drawers) == 9,
        f"Expected 9 drawers, got {len(drawers)}",
    )

    # ── Nine prismatic joints ───────────────────────────────────────────
    for i in range(1, 10):
        j = object_model.get_joint(f"drawer_{i}_slide")
        ctx.check(
            f"joint_{i}_is_prismatic_y",
            len(j.dofs) == 1 and j.dofs[0].axis == JointAxis.TRANS_Y,
            f"{j.name} is not a Y prismatic joint",
        )

    # ── Drawer open pose ────────────────────────────────────────────────
    with ctx.pose({"drawer_1_slide": 0.350}):
        pos = ctx.part_world_position("drawer_1")
        # Drawer origin moves 0.35m along Y from rest
        ctx.check(
            "drawer_1_extended",
            0.30 <= pos[1] <= 0.40,
            f"Drawer 1 Y={pos[1]:.3f}, expected ~0.35",
        )

    # ── Brass components on first drawer ────────────────────────────────
    drawer1 = object_model.get_rigid_body("drawer_1")
    has_brass = any(s.material == BRASS for s in drawer1._shapes.values())
    ctx.check(
        "drawer_has_brass",
        has_brass,
        "drawer_1 should have brass handle or brackets",
    )

    # ── Drawer contacts carcass front (expected, not collision) ─────────
    ctx.expect_contact("carcass", "drawer_1", name="drawer1_flush_carcass")

    # ── Overlap: drawer front contacts carcass side panels at front face ─
    ctx.allow_overlap(
        "carcass", "drawer_1",
        shape_a="side_-1",
        shape_b="front",
        reason="Drawer front flush against carcass front face",
    )
    ctx.allow_overlap(
        "carcass", "drawer_1",
        shape_a="side_+1",
        shape_b="front",
        reason="Drawer front flush against carcass front face",
    )

    # ── Overall height ──────────────────────────────────────────────────
    metrics = ctx.measure_geometry()
    z_hi = metrics.bounds[1][2]
    z_lo = metrics.bounds[0][2]
    height = z_hi - z_lo
    ctx.check(
        "total_height_near_0_89",
        0.85 <= height <= 0.93,
        f"Height = {height:.3f}m",
    )

    # ── Attach preview artifacts ────────────────────────────────────────
    ctx.attach_artifact(
        "qa/previews/overall.png",
        name="overall_view",
        caption="Three-quarter view of the chest of drawers.",
    )
    ctx.attach_artifact(
        "qa/previews/cross_section.png",
        name="cross_section",
        caption="Y-normal section showing the 3×3 grid with dividers and rails.",
    )
    ctx.attach_artifact(
        "qa/previews/drawer_1_extended.png",
        name="drawer_1_extended",
        caption="Drawer 1 fully open (0.35m prismatic travel).",
    )
    ctx.attach_artifact(
        "qa/previews/all_open.png",
        name="all_open",
        caption="All 9 drawers partially open to demonstrate motion.",
    )
    ctx.attach_artifact(
        "qa/previews/brass_detail.png",
        name="brass_detail",
        caption="Drawer 1 front showing brass handle and corner bracket.",
    )

    return ctx.report()
